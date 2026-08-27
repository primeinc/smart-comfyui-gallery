"""The log never says where somebody's pictures live, unless asked to.

`sg_web/redaction.py` rewrites every log record at the factory, so the
dozens of emitters written as `%s: reason` stay as they are and the
default output still carries no user path and no media filename. The
launcher's `--log-user-paths` is the one way to the old behaviour.

What must survive redaction is asserted as hard as what must not: the
suffix (the kind of file is diagnostic), a stable per-name token (two
lines about one file must still correlate), and code paths (a traceback
that cannot name its frames is not a traceback).
"""

from __future__ import annotations

import logging
import pathlib
import sys

from sg_web import redaction

A_USER_FILE = "C:/Users/somebody/pictures/nan and grandad.jpg"


def test_a_user_path_is_redacted_to_its_kind_and_a_stable_token():
    one = redaction.said(f"{A_USER_FILE}: unreadable: boom")
    assert "somebody" not in one
    assert "grandad" not in one
    assert ".jpg>" in one, "the suffix survives; the kind is diagnostic"
    assert one.endswith(": unreadable: boom"), "the reason is untouched"
    # the same file in the other spelling is the same token, so two log
    # lines about one file still correlate
    other = redaction.said("C:\\Users\\somebody\\pictures\\nan and grandad.jpg: gone")
    assert one.split(":", 1)[0] == other.split(":", 1)[0]


def test_a_posix_path_and_a_bare_media_filename_are_redacted_too():
    assert "holiday" not in redaction.said("/home/somebody/pics/holiday.png: refused")
    said = redaction.said("item 41 (holiday 12.jpg) failed: truncated")
    assert "holiday" not in said
    assert said.startswith("item 41 (")
    assert said.endswith(") failed: truncated")


def test_a_code_path_is_relativized_never_passed_through():
    """A frame stays navigable and carries no username: a checkout
    lives under a home directory, so "kept" must mean relativized."""
    ours = str(pathlib.Path(redaction.__file__))
    said = redaction.said(f'  File "{ours}", line 1')
    assert "Users" not in said
    assert said == '  File "<app>\\sg_web\\redaction.py", line 1'

    repo = str(pathlib.Path(redaction.__file__).resolve().parent.parent)
    told = redaction.said(f"vendored GPU faiss loaded from {repo}\\vendor\\faiss-gpu-win64")
    assert told == "vendored GPU faiss loaded from <app>\\vendor\\faiss-gpu-win64"


def test_a_format_string_with_a_url_shape_survives_untouched():
    """uvicorn's startup line is `%s://%s:%d` -- `s://` must not read as
    the drive `s:`, or the mangled format string raises inside logging
    and the last-resort handler prints raw frames past this module."""
    banner = "Uvicorn running on %s://%s:%d (Press CTRL+C to quit)"
    assert redaction.said(banner) == banner
    was = logging.getLogRecordFactory()
    try:
        redaction.install()
        spoke = logging.getLogger("tests.redaction.banner")
        spoke.addHandler(logging.NullHandler())
        record = spoke.makeRecord(
            "tests.redaction.banner", logging.INFO, __file__, 1, banner, ("http", "127.0.0.1", 8799), None
        )
        assert record.getMessage() == "Uvicorn running on http://127.0.0.1:8799 (Press CTRL+C to quit)"
    finally:
        logging.setLogRecordFactory(was)


def test_a_token_the_application_does_not_claim_is_not_a_filename():
    assert redaction.said("checkpoint dreamshaper_v8.safetensors loaded") == (
        "checkpoint dreamshaper_v8.safetensors loaded"
    )


def test_a_crash_and_the_last_resort_printer_are_covered_too(capsys):
    """The two channels the record factory cannot reach: an uncaught
    exception prints through sys.excepthook, and a raising FORMATTER
    prints through logging's last-resort handler -- the first is
    wrapped, the second silenced (`logging.raiseExceptions`), because
    it is the one printer nothing can launder."""
    was_hook, was_raise = sys.excepthook, logging.raiseExceptions
    was_factory = logging.getLogRecordFactory()
    try:
        redaction.install()
        assert logging.raiseExceptions is False

        def _takes_the_process_down() -> None:
            raise ValueError(f"{A_USER_FILE} took the process down")

        try:
            _takes_the_process_down()
        except ValueError:
            sys.excepthook(*sys.exc_info())
        told = capsys.readouterr().err
        assert "grandad" not in told
        assert "somebody" not in told
        assert "ValueError" in told
    finally:
        sys.excepthook, logging.raiseExceptions = was_hook, was_raise
        logging.setLogRecordFactory(was_factory)


def test_the_factory_rewrites_records_and_logged_tracebacks(caplog):
    was = logging.getLogRecordFactory()
    try:
        redaction.install()
        redaction.install()  # idempotent: the second is a no-op
        assert getattr(logging.getLogRecordFactory(), "sg_redacts", False)

        def _dies() -> None:
            raise ValueError(f"{A_USER_FILE} broke mid-read")

        spoke = logging.getLogger("tests.redaction")
        with caplog.at_level(logging.WARNING, logger="tests.redaction"):
            spoke.warning("%s: unreadable: boom", A_USER_FILE)
            try:
                _dies()
            except ValueError:
                spoke.exception("one file died")
        told = "\n".join(record.getMessage() + (record.exc_text or "") for record in caplog.records)
        assert "grandad" not in told
        assert "somebody" not in told
        assert ".jpg>" in told
        assert "ValueError" in told, "the traceback itself still arrives"
    finally:
        logging.setLogRecordFactory(was)
