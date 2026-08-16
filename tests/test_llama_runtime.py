"""Tests for smartgallery_ai.llama_runtime: official-binaries override
resolution and the backend-activation contract. All hermetic -- no DLLs
are loaded (activation exits before any ctypes call unless a provisioned
llama.dll layout exists inside tmp_path)."""

from __future__ import annotations

import importlib
import os
import sys

import pytest

from smartgallery_ai import llama_runtime


@pytest.fixture()
def fresh_runtime(monkeypatch):
    """Reload the module so its _prepared/_activated latches reset, and
    keep PATH/env mutations scoped to the test."""
    importlib.reload(llama_runtime)
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.delenv("LLAMA_CPP_LIB_PATH", raising=False)
    yield llama_runtime
    importlib.reload(llama_runtime)


def _make_provisioned(tmp_path):
    lib_dir = tmp_path / "models" / llama_runtime.OFFICIAL_LIB_DIRNAME
    lib_dir.mkdir(parents=True)
    (lib_dir / "llama.dll").write_bytes(b"not a dll")
    return lib_dir


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL bootstrap")
def test_provisioned_dir_sets_override(fresh_runtime, tmp_path, monkeypatch):
    lib_dir = _make_provisioned(tmp_path)
    monkeypatch.setenv("AI_DAM_MODELS_DIR", str(tmp_path / "models"))
    fresh_runtime.prepare_llama_runtime()
    assert os.environ["LLAMA_CPP_LIB_PATH"] == str(lib_dir)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL bootstrap")
def test_user_override_wins_over_provisioned_dir(fresh_runtime, tmp_path, monkeypatch):
    _make_provisioned(tmp_path)
    monkeypatch.setenv("AI_DAM_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("LLAMA_CPP_LIB_PATH", str(tmp_path / "custom"))
    fresh_runtime.prepare_llama_runtime()
    assert os.environ["LLAMA_CPP_LIB_PATH"] == str(tmp_path / "custom")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL bootstrap")
def test_no_provisioned_dir_leaves_env_unset(fresh_runtime, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_DAM_MODELS_DIR", str(tmp_path / "empty"))
    fresh_runtime.prepare_llama_runtime()
    assert "LLAMA_CPP_LIB_PATH" not in os.environ


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL bootstrap")
def test_cudart_sibling_dir_joins_search_path(fresh_runtime, tmp_path, monkeypatch):
    lib_dir = _make_provisioned(tmp_path)
    cudart = tmp_path / "models" / (llama_runtime.OFFICIAL_LIB_DIRNAME + "-cudart")
    cudart.mkdir()
    monkeypatch.setenv("AI_DAM_MODELS_DIR", str(tmp_path / "models"))
    fresh_runtime.prepare_llama_runtime()
    assert str(cudart) in os.environ["PATH"].split(os.pathsep)
    del lib_dir


def test_activation_without_override_is_a_noop(fresh_runtime, monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_LIB_PATH", raising=False)
    fresh_runtime.activate_llama_backends()  # must not raise, must not load DLLs


def test_activation_with_missing_layout_is_a_noop(fresh_runtime, tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_CPP_LIB_PATH", str(tmp_path))  # no ggml dlls here
    fresh_runtime.activate_llama_backends()  # must not raise


def test_prepare_and_activation_are_idempotent(fresh_runtime, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_DAM_MODELS_DIR", str(tmp_path / "empty"))
    fresh_runtime.prepare_llama_runtime()
    fresh_runtime.prepare_llama_runtime()
    fresh_runtime.activate_llama_backends()
    fresh_runtime.activate_llama_backends()
