"""Logs and logged exceptions do not name a person's files by default.

A media library's log is full of other people's filenames -- `%s:
unreadable`, `item 8123 (grandad in the garden.jpg) failed` -- and a log
line is the output that leaves the application most casually: pasted
into an issue, left in terminal scrollback, kept by a service manager.
So the default is that a user path or media filename never reaches a
handler, and `python -m sg_web --log-user-paths` turns redaction off
for a session somebody is actively debugging. One mechanism, because
the alternative is per-site discipline across dozens of emitters that
were all written to name the file.

The mechanism is the log-record FACTORY (python/cpython
Doc/library/logging.rst:942-963, :1492-1513): every record from every
logger -- the application's, uvicorn's, any library's -- is rewritten
at creation, before a handler or formatter exists to see it. Exception
text is formatted here and cached into `exc_text`, which the stock
Formatter honours instead of re-deriving (Doc/library/logging.rst:
705-721), so a logged traceback is redacted exactly once.

Redacted, kept, and the stated boundary:

  redacted  absolute paths outside the application's own tree -- the
            library lives wherever the person keeps their pictures --
            and bare tokens ending in a suffix the application claims
            (db/scan.py KIND_BY_SUFFIX: the app's own vocabulary of
            what a media filename is).
  kept      code paths (this tree and the interpreter's), because a
            traceback that cannot name its frames is not a traceback;
            each redacted name's suffix, so the kind survives; and a
            short stable hash per name, so two lines about one file
            still correlate.
  boundary  uvicorn's access log spells request URLs, whose entity
            slugs are derived from names. That line is uvicorn's
            format, not a record this module half-covers -- stated
            here rather than silently partial.
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
import sys
import traceback

from db import naming, scan

_REPO = pathlib.Path(__file__).resolve().parent.parent

#: Prefixes whose paths are CODE, not somebody's library. Normcased once;
#: the interpreter's prefixes cover a venv outside the tree.
_OURS = tuple(
    os.path.normcase(str(one)) for one in (_REPO, sys.prefix, sys.base_prefix, sys.exec_prefix, sys.executable)
)

# Regex over log text is the narrow exception rule 18 leaves open: a log
# line is genuinely unstructured prose with no grammar and no parser,
# and these patterns FIND path-shaped tokens in it rather than parse it.
#
# A Windows path may contain spaces ("nan and grandad.jpg") and may not
# contain a colon after its drive, so the match runs to the next colon
# or end of line -- which is also this codebase's own `%s: reason`
# separator. A POSIX path stops at whitespace: spaced POSIX paths in a
# log line are not distinguishable from prose, and the filename rule
# below still catches their last component.
_WINDOWS_PATH = re.compile(r"(?<![\w])[A-Za-z]:[\\/][^:<>\"|?*\n]+")
_POSIX_PATH = re.compile(r"(?<![\w:])/(?:[^/\s]+/)+[^/\s:]+")
#: A bare token that ends in a suffix the application claims. Two
#: shapes: spaceless anywhere, and spaced only inside parentheses --
#: the one emitter that logs a bare name (db/runner.py's `(named)`)
#: parenthesizes it, and letting spaces in everywhere would swallow the
#: prose to a name's left.
_SUFFIXES = "|".join(re.escape(one) for one in sorted(scan.KIND_BY_SUFFIX))
_MEDIA_NAME = re.compile(rf"[^\s\\/:*?\"<>|()]+(?:{_SUFFIXES})(?![\w.])", re.IGNORECASE)
_NAMED_IN_PARENS = re.compile(rf"\(([^()\n]+(?:{_SUFFIXES}))\)(?![\w.])", re.IGNORECASE)


def _hidden(token: str) -> str:
    """The stable spelling of a name this module will not say."""
    suffix = pathlib.PurePath(token).suffix.lower()
    return f"<{naming.short_hash(os.path.normcase(token))}{suffix}>"


def _path(match: re.Match) -> str:
    token = match.group(0).rstrip(". ")
    if os.path.normcase(token).startswith(_OURS):
        return match.group(0)
    return _hidden(token)


def said(text: str) -> str:
    """`text` with every user path and media filename replaced."""
    text = _WINDOWS_PATH.sub(_path, text)
    text = _POSIX_PATH.sub(_path, text)
    text = _NAMED_IN_PARENS.sub(lambda m: f"({_hidden(m.group(1))})", text)
    return _MEDIA_NAME.sub(lambda m: _hidden(m.group(0)), text)


def install() -> None:
    """Wrap the current record factory, once (idempotent)."""
    current = logging.getLogRecordFactory()
    if getattr(current, "sg_redacts", False):
        return

    def make(*args, **kwargs):
        record = current(*args, **kwargs)
        record.msg = said(record.getMessage())
        record.args = ()
        if record.exc_info:
            record.exc_text = said("".join(traceback.format_exception(*record.exc_info)))
            record.exc_info = None
        return record

    make.sg_redacts = True
    logging.setLogRecordFactory(make)
