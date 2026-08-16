"""faiss_runtime.import_faiss(): vendored-GPU-first selection with a
clean fallback to the installed faiss-cpu package. No real faiss import
ever happens here -- every path is exercised with throwaway packages."""

import os
import sys
import textwrap

import pytest

from smartgallery_ai import faiss_runtime


@pytest.fixture(autouse=True)
def _isolate_faiss_modules(monkeypatch):
    """Never let a test leak a fake faiss into the process, and never let
    a previously imported real faiss short-circuit a test."""
    saved = {name: mod for name, mod in sys.modules.items()
             if name == "faiss" or name.startswith("faiss.")}
    for name in saved:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield
    for name in [n for n in list(sys.modules)
                 if n == "faiss" or n.startswith("faiss.")]:
        del sys.modules[name]
    sys.modules.update(saved)


def test_already_imported_module_is_returned_verbatim(monkeypatch):
    sentinel = object()
    monkeypatch.setitem(sys.modules, "faiss", sentinel)
    assert faiss_runtime.import_faiss() is sentinel


def _make_pkg(root, body):
    pkg = root / "faiss"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return root


def test_broken_vendor_falls_back_to_installed_package(tmp_path, monkeypatch):
    """A vendored package that fails to load (missing CUDA DLLs, wrong
    arch) must not poison the import: the installed package is returned
    and sys.path is left clean."""
    vendor_root = _make_pkg(tmp_path / "vendor", "raise RuntimeError('broken vendor')")
    fallback_root = _make_pkg(tmp_path / "fallback", "FLAVOR = 'cpu'")
    monkeypatch.setattr(faiss_runtime, "_VENDOR_ROOT", str(vendor_root))
    monkeypatch.setattr(faiss_runtime, "_register_cuda_dll_dirs", lambda: 0)
    monkeypatch.syspath_prepend(str(fallback_root))
    if sys.platform != "win32":
        pytest.skip("vendor selection is Windows-only")

    mod = faiss_runtime.import_faiss()
    assert getattr(mod, "FLAVOR", None) == "cpu"
    assert str(vendor_root) not in sys.path


def test_vendor_preferred_when_loadable(tmp_path, monkeypatch):
    vendor_root = _make_pkg(tmp_path / "vendor", "FLAVOR = 'gpu'")
    fallback_root = _make_pkg(tmp_path / "fallback", "FLAVOR = 'cpu'")
    monkeypatch.setattr(faiss_runtime, "_VENDOR_ROOT", str(vendor_root))
    monkeypatch.setattr(faiss_runtime, "_register_cuda_dll_dirs", lambda: 0)
    monkeypatch.syspath_prepend(str(fallback_root))
    if sys.platform != "win32":
        pytest.skip("vendor selection is Windows-only")

    mod = faiss_runtime.import_faiss()
    assert getattr(mod, "FLAVOR", None) == "gpu"
    assert str(vendor_root) not in sys.path


def test_optout_env_skips_vendor(tmp_path, monkeypatch):
    vendor_root = _make_pkg(tmp_path / "vendor", "FLAVOR = 'gpu'")
    fallback_root = _make_pkg(tmp_path / "fallback", "FLAVOR = 'cpu'")
    monkeypatch.setattr(faiss_runtime, "_VENDOR_ROOT", str(vendor_root))
    monkeypatch.syspath_prepend(str(fallback_root))
    monkeypatch.setenv("AI_DAM_FAISS_GPU", "0")

    mod = faiss_runtime.import_faiss()
    assert getattr(mod, "FLAVOR", None) == "cpu"


def test_vendored_package_is_in_repo():
    """The vendored build ships with the repo: package files present,
    oversized CUDA runtime DLLs (nvidia-wheel-provided) absent."""
    d = faiss_runtime.vendored_faiss_dir()
    assert os.path.isfile(os.path.join(d, "_swigfaiss.pyd"))
    assert os.path.isfile(os.path.join(d, "faiss.dll"))
    assert not os.path.exists(os.path.join(d, "cublasLt64_13.dll"))
