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


def test_a_code_path_is_kept_because_a_traceback_needs_its_frames():
    ours = str(pathlib.Path(redaction.__file__))
    assert redaction.said(f'  File "{ours}", line 1') == f'  File "{ours}", line 1'


def test_a_token_the_application_does_not_claim_is_not_a_filename():
    assert redaction.said("checkpoint dreamshaper_v8.safetensors loaded") == (
        "checkpoint dreamshaper_v8.safetensors loaded"
    )


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
