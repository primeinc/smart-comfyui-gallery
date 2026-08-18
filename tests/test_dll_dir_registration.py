"""Registering CUDA directories twice must not grow PATH twice.

`import_faiss` prepares the DLL search path before importing the vendored
GPU build. It returns early when faiss is already imported, so normally
that preparation happens once -- but when neither the vendored build nor
faiss-cpu can be imported, nothing lands in sys.modules and the next
similarity query runs it all again.

Each run prepended the same nvidia directories to PATH. PATH is the
environment every child process inherits, and Windows caps that block near
32k characters, so a few dozen repeats stop ffmpeg being spawnable at all.
The symptom is video and thumbnails failing on a machine whose actual
problem is a missing faiss -- two subsystems apart from the cause.

Idempotence is the property under test.
"""

from __future__ import annotations

import os

import pytest

from smartgallery_ai import faiss_runtime


@pytest.fixture
def fake_nvidia_dirs(tmp_path, monkeypatch):
    """A purelib layout with two nvidia wheel bin directories."""
    purelib = tmp_path / "site-packages"
    made = []
    for package in ("cuda_runtime", "cublas"):
        d = purelib / "nvidia" / package / "bin"
        d.mkdir(parents=True)
        made.append(str(d))

    monkeypatch.setattr(faiss_runtime.sysconfig, "get_paths", lambda: {"purelib": str(purelib)})
    monkeypatch.setattr(faiss_runtime, "_REGISTERED_DLL_DIRS", set())
    monkeypatch.setenv("PATH", "C:\\original\\path")
    return sorted(made)


def test_the_first_call_registers_the_directories(fake_nvidia_dirs):
    """Control: without this the test below passes on a function that does
    nothing at all."""
    faiss_runtime._register_cuda_dll_dirs()

    entries = os.environ["PATH"].split(os.pathsep)
    for directory in fake_nvidia_dirs:
        assert directory in entries, f"{directory} was never added to PATH"


def test_a_second_call_changes_nothing(fake_nvidia_dirs):
    """The regression: PATH grew by the same directories every time."""
    faiss_runtime._register_cuda_dll_dirs()
    after_first = os.environ["PATH"]

    for _ in range(5):
        faiss_runtime._register_cuda_dll_dirs()

    assert os.environ["PATH"] == after_first, (
        f"PATH grew on repeat registration; it went from {len(after_first)} to {len(os.environ['PATH'])} characters"
    )


def test_path_stays_a_sane_length_under_many_calls(fake_nvidia_dirs):
    """The failure was not the duplication itself but where it ends: an
    environment block too large to start a child process."""
    for _ in range(200):
        faiss_runtime._register_cuda_dll_dirs()

    assert len(os.environ["PATH"]) < 4000, (
        f"PATH reached {len(os.environ['PATH'])} characters; Windows stops being able to spawn ffmpeg around 32000"
    )


def test_an_entry_already_present_is_not_added_again(fake_nvidia_dirs, monkeypatch):
    """Belt and braces: even with the memo cleared -- a fresh process that
    inherited a PATH already containing these -- the entry is not doubled."""
    monkeypatch.setenv("PATH", os.pathsep.join([*fake_nvidia_dirs, "C:\\original"]))
    monkeypatch.setattr(faiss_runtime, "_REGISTERED_DLL_DIRS", set())

    faiss_runtime._register_cuda_dll_dirs()

    entries = os.environ["PATH"].split(os.pathsep)
    for directory in fake_nvidia_dirs:
        assert entries.count(directory) == 1, f"{directory} appears twice"
