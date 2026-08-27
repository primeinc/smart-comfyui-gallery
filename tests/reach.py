"""Which lines of the readers a corpus actually reaches.

The corpus exists to make unreachable reader branches reachable. That
sentence is unfalsifiable until something measures it, and an
unfalsifiable goal cannot be finished -- "the unreachable branches" names
a set you re-derive every time you think about it, and you think better
each time, so the set grows and the work never closes.

So this measures it, once, and the answer is written down and frozen. A
frozen list can be exhausted. An intuition cannot.

Stdlib `sys.settrace` rather than `coverage`, deliberately: the
instrument is a means to a target list, not a reason to grow the
dependency list. It records executed LINES; the executable lines come
from the module's own AST -- so "unreached" is `executable - executed`,
both derived from the file rather than from a number anybody typed.

Not a test. `tests/test_a_corpus_reaches_what_it_claims.py` is the gate;
this is what it measures with.
"""

from __future__ import annotations

import contextlib
import pathlib
import sys
import types
from collections.abc import Callable, Iterable

REPO = pathlib.Path(__file__).resolve().parent.parent

#: The readers a media corpus is FOR. Each turns bytes on disk into
#: something the library believes, and each is full of branches only a
#: particular shape of file reaches. Everything else in the tree is out
#: of scope for this measurement and stays out.
READERS: tuple[str, ...] = (
    # `db/scan.py` is deliberately ABSENT. Its file-shaped surface is
    # three functions and a suffix table; the rest takes a CONNECTION,
    # not a path, and no corpus can or should reach it. Counting it put
    # the module at 1% and added 300 lines to a denominator a corpus
    # cannot close -- which would make the frozen target unreachable and
    # so make this creation unfinishable, which is the one thing the
    # freeze exists to prevent.
    "db/capture.py",
    "db/when.py",
    "db/graph.py",
    "db/probe.py",
    "vision/decode.py",
    "vision/sniff.py",
    "metaparse/adapters.py",
    "metaparse/containers.py",
    "metaparse/model.py",
    "metaparse/typed.py",
)


def executable(path: pathlib.Path) -> set[int]:
    """Every line of `path` that CAN execute.

    From the compiled code objects, not from the source tree: `co_lines`
    is the interpreter's own answer to "which lines can run", so the
    denominator is exactly what a tracer could ever mark and needs no
    heuristic about which statements count. Docstrings, comments,
    annotations and `if TYPE_CHECKING` bodies are absent because the
    compiler never emits a line for them -- which an AST walk had to
    special-case and get right, and now nobody has to.

    Nested functions, comprehensions and classes are reached through
    `co_consts`, so a line inside a closure counts like any other.
    """
    held: set[int] = set()
    stack = [compile(path.read_text(encoding="utf-8"), str(path), "exec")]
    while stack:
        code = stack.pop()
        for _start, _end, line in code.co_lines():
            if line is not None:
                held.add(line)
        stack.extend(one for one in code.co_consts if isinstance(one, types.CodeType))
    return held


class Reached:
    """Lines executed while something ran, per reader module."""

    __slots__ = ("_watching", "lines")

    def __init__(self, readers: Iterable[str] = READERS):
        self._watching = {str(REPO / one): one for one in readers}
        self.lines: dict[str, set[int]] = {one: set() for one in readers}

    def _trace(self, frame, event, _arg):
        name = self._watching.get(frame.f_code.co_filename)
        if name is None:
            return None
        if event == "line":
            self.lines[name].add(frame.f_lineno)
        return self._trace

    def watch(self, run: Callable[[], object]) -> object:
        """Run `run` with the tracer on.

        A reader raising on a malformed file is a branch REACHED, which
        is the whole point of feeding it malformed files -- so the raise
        is suppressed and the lines it executed on the way still count.
        What was raised is not recorded because it is not what is being
        measured; the gate asks which lines ran.
        """
        was = sys.gettrace()
        sys.settrace(self._trace)
        try:
            with contextlib.suppress(Exception):
                return run()
            return None
        finally:
            sys.settrace(was)

    def unreached(self) -> dict[str, list[int]]:
        """`{module: [line, ...]}` -- the target set, once frozen."""
        out: dict[str, list[int]] = {}
        for name, got in self.lines.items():
            missing = sorted(executable(REPO / name) - got)
            if missing:
                out[name] = missing
        return out

    def tally(self) -> dict[str, tuple[int, int]]:
        """`{module: (reached, executable)}` -- the numbers a person reads."""
        return {
            name: (len(got & executable(REPO / name)), len(executable(REPO / name))) for name, got in self.lines.items()
        }


def _judged(when, held):
    """`judge_capture` over what `capture.read` returned, or None.

    Keyword-only, like `judge_file`, and it was called positionally: the
    call raised TypeError, `watch()` swallowed it, and the richest reader
    in the set contributed nothing to the measurement while looking as
    though it had been driven.
    """
    if held is None:
        return None
    return when.judge_capture(
        captured_at=held.captured_at,
        subsec_ms=held.subsec_ms,
        tz_offset_min=held.tz_offset_min,
        maker_tz_offset_min=held.maker_tz_offset_min,
        mtime=None,
        btime=None,
    )


def _closed(image):
    """Open, then close. `decode.open_still` hands back a live handle, and
    leaking one per file makes a ResourceWarning at collection -- which
    pytest.ini turns into a failure on purpose, because a handle nobody
    closed is a defect however quiet it looks."""
    image.close()


def over(files: Iterable[pathlib.Path]) -> Reached:
    """Drive every reader over every file, and record what ran.

    One place, so the freeze and the gate measure the SAME thing. A
    baseline taken one way and checked another way compares nothing.

    Imports are local because this walks the readers under a tracer and
    a module-level import would run half of them before it started.
    """
    from db import capture, graph, probe, scan, when
    from metaparse import adapters, containers
    from vision import decode, sniff

    held = Reached()
    for one in files:
        spelled, kind = str(one), scan.KIND_BY_SUFFIX.get(one.suffix.lower())
        for run in (
            lambda p=one: capture.read(p),
            lambda p=spelled: sniff.sniff_path(p),
            lambda p=spelled: probe.read(p),
            lambda p=one, k=kind: decode.dimensions(p, k) if k else None,
            lambda p=one: _closed(decode.open_still(p)),
            lambda p=spelled: adapters.parse_file(p, allow_stealth=True),
            # KEYWORDS. `judge_file` and `judge_capture` are keyword-only
            # (db/when.py:425-431, :490-498) and were called positionally
            # here, so both raised TypeError -- which `watch()` suppresses
            # on purpose, so they reported no coverage and no complaint.
            # Two of the ten readers were measured as zero for as long as
            # this file has existed.
            lambda p=one: when.judge_file(
                name=p.name,
                folders=[q.name for q in p.parents[:3]],
                mtime=p.stat().st_mtime,
                btime=None,
            ),
            lambda p=one: when.judge_filesystem(p.stat().st_mtime, None),
            lambda n=one.name: when.name_stamp(n),
            lambda n=one.name: when.swarm_stamp(n),
        ):
            held.watch(run)
        raw = containers.load_raw(spelled)
        if raw is not None:
            held.watch(lambda r=raw: adapters.parse_raw(r))
            for chunk in ("prompt", "workflow", "parameters"):
                if (raw.text or {}).get(chunk):
                    held.watch(lambda text=raw.text[chunk]: graph.read(text))
        # INSIDE watch(), like everything else. Called bare, a reader
        # that raises takes the whole measurement down with it -- and
        # `JXL.jxl` does exactly that: pillow-jxl raises RuntimeError out
        # of `capture.read`. A measurement that dies on the files it
        # exists to measure is not a measurement.
        held.watch(lambda p=one: _judged(when, capture.read(p)))
    return held


def refreeze(note: str) -> dict:
    """Re-measure and rewrite `tests/reach_baseline.json`.

    The baseline is self-describing: its `corpus` block names the
    generator, seed and scale it was measured over, and the refreeze
    reads them back rather than letting the procedure drift from the
    record. `note` becomes the `refrozen` field -- a refreeze without a
    stated reason is a number that agrees with itself.

    Run through `just corpus refreeze-reach`, never re-typed inline.
    """
    import json
    import tempfile

    from tests import corpus

    baseline = REPO / "tests" / "reach_baseline.json"
    old = json.loads(baseline.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as scratch:
        root = pathlib.Path(scratch) / "spread"
        root.mkdir()
        corpus.spread(root, seed=old["corpus"]["seed"], scale=old["corpus"]["scale"])
        files = sorted(p for p in root.rglob("*") if p.is_file())
        held = over(files)
    tally = held.tally()
    new = dict(old)
    new["refrozen"] = note
    new["corpus"] = dict(old["corpus"], files=len(files))
    new["totals"] = {
        "reached": sum(r for r, _ in tally.values()),
        "executable": sum(e for _, e in tally.values()),
    }
    new["per_module"] = {name: {"reached": r, "executable": e} for name, (r, e) in sorted(tally.items())}
    new["unreached"] = held.unreached()
    with open(baseline, "w", encoding="utf-8", newline="") as f:
        json.dump(new, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return {"old": old["totals"], "new": new["totals"], "files": len(files)}


if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        raise SystemExit("a refreeze states its reason: python -m tests.reach '<why the readers changed>'")
    told = refreeze(sys.argv[1].strip())
    print(f"refrozen: {told['old']} -> {told['new']} over {told['files']} files")
