"""How this program starts other programs.

Three properties matter, and none of them is "the argv is a literal":

  1. Never through a shell. `shell=True` hands the whole command line to
     cmd.exe or /bin/sh, where a quote in a filename ends the argument and
     starts a new command. Everything here passes a list, which the OS
     hands to the program one argument at a time.
  2. Always with a timeout. A media tool pointed at a file on unreachable
     network storage sits there, and the caller is a request waiting on
     the answer.
  3. Always with an explicit `check=`. Left out, a failing program raises
     nothing and the caller reads an empty stdout as an empty answer.

flake8-bandit's S603 asks for something else: that every element of the
argv be a string literal or `sys.executable`. A program that runs ffprobe
over a file somebody dropped in a folder cannot satisfy that and never
will -- the path is the point. Its own documentation calls it an audit
check, and there is no edit that clears it.

So these are the checks instead, and they cover what S603 cannot see:
S603 is satisfied by `subprocess.run(["ffprobe", "-i", "fixed.mp4"])` with
no timeout and no check=, which is worse code than anything here.
"""

from __future__ import annotations

import ast

from source_tree import every_source, parsed

# Everything that ships or runs from this repo, tests included: a test that
# spawns without a timeout hangs CI exactly as a route would hang a request.
# Discovered, never listed -- the listed version missed the db/ package for
# as long as the package had existed.

_SPAWNERS = {"run", "call", "check_call", "check_output", "Popen"}


def _spawn_calls(tree):
    """Every subprocess.<spawner>(...) call node in `tree`."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _SPAWNERS:
            root = func.value
            if isinstance(root, ast.Name) and root.id == "subprocess":
                calls.append(node)
    return calls


def _keyword(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def test_the_sweep_finds_the_spawns_that_are_there():
    """Control. Each check below is an absence, and a sweep that matched
    nothing would report the same absence."""
    total = sum(len(_spawn_calls(parsed(s))) for s in every_source())

    assert total >= 2, f"only {total} subprocess calls found; the sweep is not reaching them"


def _sites(objects_to):
    """{"file:line": call} for every spawn the predicate objects to."""
    found = {}
    for source in every_source():
        for call in _spawn_calls(parsed(source)):
            if objects_to(call):
                found[f"{source.name}:{call.lineno}"] = ast.unparse(call.func)
    return found


def _through_a_shell(call):
    shell = _keyword(call, "shell")
    return shell is not None and getattr(shell, "value", True) is not False


def test_nothing_is_started_through_a_shell():
    """A quote in a filename ends the argument and starts a command."""
    offenders = _sites(_through_a_shell)

    assert not offenders, f"{offenders} start a program through a shell. Pass a list of arguments instead."


def test_every_spawn_gets_an_argument_list():
    """A single string is the shell form even without shell=True on
    Windows, where the platform re-splits it."""
    offenders = _sites(
        lambda call: bool(call.args) and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)
    )

    assert not offenders, f"{offenders} pass a bare command string instead of a list"


def test_every_spawn_says_how_long_it_will_wait():
    """Without one, an unreachable path is a request that never answers."""
    # Popen does not take a timeout: it returns immediately and the caller
    # waits with its own.
    offenders = _sites(lambda call: call.func.attr != "Popen" and _keyword(call, "timeout") is None)

    assert not offenders, f"{offenders} start a program with no timeout"


def _pipes_output(call):
    """Popen handed a PIPE this repo has nobody reading."""
    if call.func.attr != "Popen":
        return False
    for stream in ("stdout", "stderr"):
        given = _keyword(call, stream)
        if (
            isinstance(given, ast.Attribute)
            and given.attr == "PIPE"
            and isinstance(given.value, ast.Name)
            and given.value.id == "subprocess"
        ):
            return True
    return False


def test_no_long_lived_program_writes_into_an_undrained_pipe():
    """stdout=PIPE on Popen is a promise that somebody will read the pipe.
    Nothing in this repo does: the OS buffer holds ~4KB, a chatty child
    (uvicorn logs one access line per request) blocks mid-write there --
    measured at request 64 -- and every request in flight then hangs
    forever with no error anywhere. Sink to a file instead; it has no
    ceiling and holds the log for a post-mortem."""
    offenders = _sites(_pipes_output)

    assert not offenders, f"{offenders} hand a child a pipe nobody drains; the child freezes at the OS buffer"


def test_every_run_says_whether_a_failure_matters():
    """`check=` left out means a failing program raises nothing and its
    empty output reads as an empty answer."""
    offenders = _sites(lambda call: call.func.attr == "run" and _keyword(call, "check") is None)

    assert not offenders, f"{offenders} run a program without saying check="


def test_the_checks_would_catch_what_they_are_for():
    """Control for all five: each has to fail for the shape it exists to
    catch, or it passes because it understands nothing."""
    bad = ast.parse(
        "import subprocess\nsubprocess.run('ffprobe ' + path, shell=True)\nsubprocess.run([tool, path])\n"
        "subprocess.Popen([tool], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n"
    )
    calls = _spawn_calls(bad)

    assert len(calls) == 3, calls
    assert getattr(_keyword(calls[0], "shell"), "value", False) is True
    assert isinstance(calls[0].args[0], ast.BinOp)  # a built command string
    assert _keyword(calls[1], "timeout") is None
    assert _keyword(calls[1], "check") is None
    assert _pipes_output(calls[2]), "the undrained-pipe check does not see the shape it exists for"
    assert not _pipes_output(calls[1])
