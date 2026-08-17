"""Every setting in CONFIGURATION.md has to be one the program reads.

A documented setting that nothing reads is worse than an undocumented one:
someone sets it, sees no error, and believes it took effect. That is how
`ENABLE_AI_SEARCH` came to promise a working search box, and it is a whole
class of bug that only a check like this one catches -- nothing fails at
runtime, so no other test can see it.

Both directions are covered:

  * documented -> read: every variable named in a table in the docs is
    referenced by the Python, by the Docker entrypoint, or by a compose
    file. WANTED_UID and WANTED_GID are the reason the entrypoint counts;
    they are implemented in bash.
  * read -> documented: every variable the configuration helpers read is
    named somewhere in the docs, in a table or in prose. The low-level
    hardware switches are documented in a sentence rather than a row, which
    is why prose counts.
"""

from __future__ import annotations

import io
import pathlib
import re


_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Standard environment, not settings of ours.
_NOT_OURS = {"PATH", "DISPLAY", "HOME", "USERPROFILE", "TEMP", "TMP",
             "FAISS_DISABLE_CPU_FEATURES"}

_READERS = re.compile(
    r"""(?:os\.environ\.get\(|os\.getenv\(|env_or\(|env_num\(|env_flag\(
        |_env_str\(|_env_num\(|_env_bool\(|ENV_MODEL_PATH\s*=\s*)
        ["']([A-Z][A-Z0-9_]{2,})["']""",
    re.VERBOSE)


def _source_text():
    parts = []
    for path in _ROOT.rglob("*.py"):
        if any(p in path.parts for p in (".venv", "tests", "benchmarks", "probes", "vendor")):
            continue
        parts.append((path, io.open(path, encoding="utf-8").read()))
    return parts


def _doc_text():
    parts = []
    for path in list((_ROOT / "docs").rglob("*.md")) + [_ROOT / "README.md"]:
        if path.exists():
            parts.append((path, io.open(path, encoding="utf-8").read()))
    return parts


def _documented_in_tables():
    names = set()
    for _path, text in _doc_text():
        for line in text.splitlines():
            if line.startswith("|") and line.count("|") > 1:
                for match in re.finditer(r"`([A-Z][A-Z0-9_]{2,})`", line.split("|")[1]):
                    names.add(match.group(1))
    return names


def _variables_the_code_reads():
    found = {}
    for path, text in _source_text():
        for match in _READERS.finditer(text):
            found.setdefault(match.group(1), set()).add(path.name)
    return found


def test_the_audit_sees_something():
    """Control: if the parsers stop matching, both tests below pass on
    empty sets and prove nothing."""
    documented = _documented_in_tables()
    read = _variables_the_code_reads()

    assert len(documented) > 30, f"only {len(documented)} documented variables found"
    assert len(read) > 30, f"only {len(read)} variables read by the code"
    assert "BASE_OUTPUT_PATH" in documented and "BASE_OUTPUT_PATH" in read


def test_every_documented_setting_is_read_somewhere():
    """The regression this file exists for."""
    non_python = ""
    for name in ("docker_init.bash", "compose.yaml", "compose-exhibit.yaml",
                 "Makefile", "Dockerfile", "run_smartgallery.bat"):
        path = _ROOT / name
        if path.exists():
            non_python += io.open(path, encoding="utf-8", errors="replace").read()

    all_python = "\n".join(text for _p, text in _source_text())

    orphans = sorted(name for name in _documented_in_tables()
                     if name not in _NOT_OURS
                     and name not in all_python
                     and name not in non_python)

    assert orphans == [], (
        f"documented but read by nothing: {orphans}. Either the setting was "
        f"removed and the row should go, or it never worked -- say so in the "
        f"row rather than leaving it looking functional.")


def test_every_setting_the_code_reads_is_documented():
    docs = "\n".join(text for _p, text in _doc_text())

    undocumented = sorted(name for name in _variables_the_code_reads()
                          if name not in _NOT_OURS and name not in docs)

    assert undocumented == [], (
        f"read by the code but in no doc: {undocumented}. A setting nobody "
        f"can find is a setting nobody can use.")
