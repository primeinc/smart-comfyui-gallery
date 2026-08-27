"""The log never says where somebody's pictures live, unless asked to.

`sg_web/redaction.py` rewrites every log record at the factory, so the
dozens of emitters written as `%s: reason` stay as they are and the
default output still carries no user path and no media filename. The
launcher's `--log-user-paths` is the one way to the old behaviour.

One behaviour per test, so a red test names the grammar or channel
that broke. What must survive is asserted as hard as what must not:
the suffix (the kind is diagnostic), a stable per-name token (lines
about one file must correlate), and relativized code frames (a
traceback that cannot name its frames is not a traceback).
"""

from __future__ import annotations

import logging
import pathlib
import sys

import pytest

from sg_web import redaction

A_USER_FILE = "C:/Users/somebody/pictures/nan and grandad.jpg"
_REPO = pathlib.Path(redaction.__file__).resolve().parent.parent


@pytest.fixture
def redacting():
    """Install the redactor with every global it touches restored after."""
    was_factory = logging.getLogRecordFactory()
    was_hook, was_raise = sys.excepthook, logging.raiseExceptions
    redaction.install()
    yield logging.getLogger("tests.redaction")
    logging.setLogRecordFactory(was_factory)
    sys.excepthook, logging.raiseExceptions = was_hook, was_raise


# --- the grammars `said` must catch, one each -------------------------------


def test_a_drive_path_is_redacted_to_its_kind():
    told = redaction.said(f"{A_USER_FILE}: unreadable: boom")
    assert "somebody" not in told
    assert "grandad" not in told
    assert ".jpg>" in told, "the suffix survives; the kind is diagnostic"
    assert told.endswith(": unreadable: boom"), "the reason is untouched"


def test_two_spellings_of_one_file_share_a_token():
    one = redaction.said(f"{A_USER_FILE}: gone")
    other = redaction.said("C:\\Users\\somebody\\pictures\\nan and grandad.jpg: gone")
    assert one == other


def test_a_posix_path_is_redacted():
    assert "holiday" not in redaction.said("/home/somebody/pics/holiday.png: refused")


def test_a_network_share_is_redacted():
    told = redaction.said("\\\\nas\\photos\\summer 2019\\holiday 12.jpg: unreadable")
    assert "nas" not in told
    assert "holiday" not in told
    assert told.endswith(": unreadable")


def test_a_parenthesized_spaced_name_is_redacted():
    """db/runner.py's `item N (named)` is the one emitter of bare
    spaced names; unbounded spaces would swallow the prose to a
    name's left."""
    told = redaction.said("item 41 (holiday 12.jpg) failed: truncated")
    assert "holiday" not in told
    assert told.startswith("item 41 (")
    assert told.endswith(") failed: truncated")


def test_an_unclaimed_suffix_is_not_a_filename():
    line = "checkpoint dreamshaper_v8.safetensors loaded"
    assert redaction.said(line) == line


def test_a_url_shaped_format_string_is_not_a_drive():
    """`%s://%s:%d` must not read as the drive `s:` -- a mangled format
    string raises inside logging, past this module's reach."""
    banner = "Uvicorn running on %s://%s:%d (Press CTRL+C to quit)"
    assert redaction.said(banner) == banner


# --- what is deliberately kept ----------------------------------------------


def test_a_code_frame_is_relativized_never_passed_through():
    ours = str(pathlib.Path(redaction.__file__))
    told = redaction.said(f'  File "{ours}", line 1')
    assert told == '  File "<app>\\sg_web\\redaction.py", line 1'


def test_a_vendored_load_line_is_relativized():
    told = redaction.said(f"vendored GPU faiss loaded from {_REPO}\\vendor\\faiss-gpu-win64")
    assert told == "vendored GPU faiss loaded from <app>\\vendor\\faiss-gpu-win64"


# --- the factory, one channel per test --------------------------------------


def test_install_wraps_the_factory_exactly_once(redacting):
    wrapped = logging.getLogRecordFactory()
    redaction.install()
    assert logging.getLogRecordFactory() is wrapped, "a second install must be a no-op, not a second wrap"


def test_a_record_and_its_tuple_args_are_redacted_in_place(redacting):
    record = redacting.makeRecord(
        redacting.name, logging.INFO, __file__, 1, "Uvicorn running on %s://%s:%d", ("http", "127.0.0.1", 8799), None
    )
    assert record.getMessage() == "Uvicorn running on http://127.0.0.1:8799"
    assert record.args == ("http", "127.0.0.1", 8799), "formatters unpack args; they must survive as a tuple"


def test_mapping_args_are_redacted_too(redacting):
    # a mapping rides as a one-tuple, the shape Logger._log builds;
    # LogRecord.__init__ unpacks it back to the dict
    record = redacting.makeRecord(
        redacting.name, logging.INFO, __file__, 1, "%(what)s failed", ({"what": A_USER_FILE},), None
    )
    assert "grandad" not in record.getMessage()
    assert ".jpg>" in record.getMessage()


def test_a_logged_traceback_is_redacted_cause_included(redacting, caplog):
    def _dies() -> None:
        why = OSError(f"{A_USER_FILE} vanished")
        raise ValueError(f"{A_USER_FILE} broke mid-read") from why

    with caplog.at_level(logging.WARNING, logger=redacting.name):
        try:
            _dies()
        except ValueError:
            redacting.exception("one file died")
    told = "\n".join((record.exc_text or "") for record in caplog.records)
    assert "grandad" not in told
    assert "somebody" not in told
    assert "ValueError" in told
    assert "OSError" in told, "the chained cause still arrives"


def test_the_emitting_files_own_path_is_relativized(redacting, caplog):
    with caplog.at_level(logging.WARNING, logger=redacting.name):
        redacting.warning("anything")
    assert all("Users" not in record.pathname for record in caplog.records), (
        "%(pathname)s in a handler format must not name anybody"
    )


def test_a_crash_prints_through_the_redactor(redacting, capsys):
    def _takes_the_process_down() -> None:
        raise ValueError(f"{A_USER_FILE} took the process down")

    try:
        _takes_the_process_down()
    except ValueError as why:
        sys.excepthook(type(why), why, why.__traceback__)
    told = capsys.readouterr().err
    assert "grandad" not in told
    assert "somebody" not in told
    assert "ValueError" in told


def test_the_last_resort_printer_is_silenced(redacting):
    """When a FORMATTER raises, logging's fallback prints raw frames --
    the one printer nothing can launder, so it must be off."""
    assert logging.raiseExceptions is False
