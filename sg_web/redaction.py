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
  gone      uvicorn's per-request access log, whose URLs spell entity
            slugs derived from media names -- an access log IS a list
            of what the library holds, so the launcher disables it
            unless `--log-user-paths` (sg_web/__main__.py).
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

#: Prefixes whose paths are CODE, not somebody's library -- each with the
#: label its remainder is spelled under. RELATIVIZED, never passed
#: through: the first version kept them verbatim, and a checkout under
#: `C:\Users\<name>\dev\...` then shipped the username on every kept
#: frame and every "loaded from" line -- the exact leak this module
#: exists to stop. Longest prefix first, because the venv lives inside
#: the repo and must claim its own paths before the repo does.
_OURS = tuple(
    sorted(
        {
            os.path.normcase(str(prefix)): label
            for prefix, label in (
                (sys.prefix, "<venv>"),
                (sys.base_prefix, "<python>"),
                (sys.exec_prefix, "<venv>"),
                (pathlib.Path(sys.executable).parent, "<python>"),
                (_REPO, "<app>"),
            )
        }.items(),
        key=lambda held: -len(held[0]),
    )
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
#: `(?![\\/])`: a drive spec is never `X://`, but a URL's `s://` is --
#: uvicorn's own startup format string `%s://%s:%d` read as the drive
#: `s:` until this, and a mangled format string raises inside logging,
#: whose last-resort handler prints raw frames past this module.
_WINDOWS_PATH = re.compile(r"(?<![\w])[A-Za-z]:[\\/](?![\\/])[^:<>\"|?*\n]*")
#: A network share: `\\nas\photos\...`. No drive colon, so the same
#: run-to-colon-or-EOL policy; a NAS-hosted library is an ordinary
#: place for a media collection to live.
_UNC_PATH = re.compile(r"(?<![\w\\])\\\\[^:<>\"|?*\n]+")
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
    normed = os.path.normcase(token)
    for prefix, label in _OURS:
        if normed.startswith(prefix):
            # normcase preserves length, so the slice is the original
            # spelling's remainder -- a frame stays navigable while the
            # prefix that names a person is gone.
            return label + token[len(prefix) :] + match.group(0)[len(token) :]
    return _hidden(token) + match.group(0)[len(token) :]


def said(text: str) -> str:
    """`text` with every user path and media filename replaced."""
    text = _WINDOWS_PATH.sub(_path, text)
    text = _UNC_PATH.sub(_path, text)
    text = _POSIX_PATH.sub(_path, text)
    text = _NAMED_IN_PARENS.sub(lambda m: f"({_hidden(m.group(1))})", text)
    return _MEDIA_NAME.sub(lambda m: _hidden(m.group(0)), text)


def install() -> None:
    """Everything a served process emits, redacted. Idempotent.

    Three channels, because the record factory alone cannot reach them
    all: log records (the factory below), uncaught exceptions
    (sys.excepthook -- a crashed boot prints its traceback outside
    logging, python/cpython Doc/library/sys.rst:443-447), and logging's
    own last-resort handler, which prints raw frames whenever a
    FORMATTER raises -- the one printer no factory can launder, so
    `logging.raiseExceptions = False` silences it (the record is lost;
    the leak is not)."""
    logging.raiseExceptions = False
    if not getattr(sys.excepthook, "sg_redacts", False):

        def crashed(kind, value, trace) -> None:
            sys.stderr.write(said("".join(traceback.format_exception(kind, value, trace))))

        crashed.sg_redacts = True
        sys.excepthook = crashed

    current = logging.getLogRecordFactory()
    if getattr(current, "sg_redacts", False):
        return

    def make(*args, **kwargs):
        record = current(*args, **kwargs)
        # Args are redacted IN PLACE, never flattened into the message:
        # uvicorn's access formatter unpacks record.args as a five-tuple
        # (uvicorn/logging.py AccessFormatter.formatMessage), and a first
        # version that set `args = ()` made every access line raise
        # inside logging -- whose last-resort handler prints raw frames
        # to stderr, OUTSIDE this factory. Non-strings pass through so
        # `%d` directives keep formatting.
        record.msg = said(str(record.msg)) if record.msg else record.msg
        # `%(pathname)s` in any handler's format would print the
        # emitting file's absolute path, username included; relativize
        # it the same as a frame. (`filename` and `module`, derived at
        # init, are already bare names.)
        record.pathname = said(record.pathname)
        if isinstance(record.args, dict):
            record.args = {key: said(one) if isinstance(one, str) else one for key, one in record.args.items()}
        elif record.args:
            record.args = tuple(said(one) if isinstance(one, str) else one for one in record.args)
        if record.exc_info:
            record.exc_text = said("".join(traceback.format_exception(*record.exc_info)))
            record.exc_info = None
        return record

    make.sg_redacts = True
    logging.setLogRecordFactory(make)
