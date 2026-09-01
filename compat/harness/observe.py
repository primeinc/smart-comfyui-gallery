from __future__ import annotations

import contextlib
import ctypes
import inspect
import json
import os
import sys
from collections.abc import Callable, Generator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, override

ROOT: Final[Path] = Path(__file__).resolve().parent.parent


MODEL_SUFFIX: Final[frozenset[str]] = frozenset(
    {".onnx", ".pth", ".pt", ".bin", ".safetensors", ".task", ".ckpt", ".npy", ".npz", ".zip"}
)


@dataclass
class Observation:
    loader: str
    identity: str
    path: str = ""
    revision: str = ""


@dataclass
class Recorder:
    active: bool = False
    seen: dict[tuple[str, str], Observation] = field(default_factory=dict)

    def note(self, loader: str, identity: str, path: str = "", revision: str = "") -> None:
        if not self.active or not identity:
            return
        key = (loader, identity)
        if key not in self.seen:
            self.seen[key] = Observation(loader, identity, path, revision)

    def rows(self) -> list[dict[str, Any]]:
        return [asdict(one) for one in sorted(self.seen.values(), key=lambda one: (one.loader, one.identity))]


_RECORDER: Final[Recorder] = Recorder()
_INSTALLED: list[bool] = []


def model_roots() -> tuple[Path, ...]:
    from compat.harness import provenance

    found: set[Path] = set()
    with contextlib.suppress(OSError, KeyError, TypeError, ValueError):
        manifest = provenance.load_manifest()
        for row in manifest.get("weights", []):
            with contextlib.suppress(OSError, KeyError, TypeError, ValueError):
                found.add(provenance._weight_root(row).resolve())
    for variable in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "IDV2V_CHECKPOINTS"):
        value = os.environ.get(variable)
        if value:
            with contextlib.suppress(OSError, ValueError):
                found.add(Path(value).resolve())
    with contextlib.suppress(OSError, RuntimeError):
        found.add((Path.home() / ".cache" / "huggingface").resolve())
    return tuple(sorted(found))


_ROOTS: list[Path] = []


def _under_a_model_root(name: str) -> bool:
    if not _ROOTS:
        return False
    with contextlib.suppress(OSError, ValueError):
        here = Path(name).resolve()
        return any(here.is_relative_to(root) for root in _ROOTS)
    return False


NATIVE_OPENERS: Final[frozenset[str]] = frozenset(
    {"fopen", "fopen_s", "_wfopen", "_wfopen_s", "_open", "_wopen", "open", "CreateFileA", "CreateFileW"}
)


NATIVE_UNSEEN: Final[str] = "native_open_unseen"


class _Watched:
    def __init__(self, func: Any, name: str) -> None:
        object.__setattr__(self, "_func", func)
        object.__setattr__(self, "_name", name)

    def __getattr__(self, item: str) -> Any:
        return getattr(object.__getattribute__(self, "_func"), item)

    @override
    def __setattr__(self, item: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_func"), item, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        func = object.__getattribute__(self, "_func")
        name = object.__getattribute__(self, "_name")
        with contextlib.suppress(OSError, ValueError, TypeError, IndexError):
            if args and isinstance(args[0], (str, os.PathLike)):
                where = str(args[0])
                _RECORDER.note(f"ctypes:{name}", Path(where).name, where)
        return func(*args, **kwargs)


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if event == "ctypes.dlsym" and len(args) >= 2:
        symbol = str(args[1])
        if symbol in NATIVE_OPENERS:
            _RECORDER.note(NATIVE_UNSEEN, symbol, str(args[0])[:120])
        return
    if event != "open" or not args:
        return
    name = str(args[0])
    if Path(name).suffix.lower() in MODEL_SUFFIX and _under_a_model_root(name):
        _RECORDER.note("open", Path(name).name, name)


def install() -> None:
    if _INSTALLED:
        return
    sys.addaudithook(_audit)
    _INSTALLED.append(True)


def _argument(args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str) -> str:
    if name in kwargs and isinstance(kwargs[name], (str, os.PathLike)):
        return str(kwargs[name])
    if len(args) > index and isinstance(args[index], (str, os.PathLike)):
        return str(args[index])
    return ""


Identify = Callable[[tuple[Any, ...], dict[str, Any]], None]


def wrap_recording(stack: contextlib.ExitStack, owner: Any, name: str, identify: Identify) -> None:
    original = getattr(owner, name, None)
    if original is None:
        return

    declared = inspect.getattr_static(owner, name, None)
    if isinstance(declared, classmethod):
        underlying = declared.__func__

        def observed_classmethod(cls: Any, *args: Any, **kwargs: Any) -> Any:

            with contextlib.suppress(TypeError, ValueError, AttributeError, IndexError, KeyError):
                identify(args, kwargs)
            return underlying(cls, *args, **kwargs)

        setattr(owner, name, classmethod(observed_classmethod))
        stack.callback(setattr, owner, name, declared)
        return

    def observed(*args: Any, **kwargs: Any) -> Any:
        with contextlib.suppress(TypeError, ValueError, AttributeError, IndexError, KeyError):
            identify(args, kwargs)
        return original(*args, **kwargs)

    setattr(owner, name, observed)
    stack.callback(setattr, owner, name, original)


def _wrap_hub_apis(stack: contextlib.ExitStack) -> None:

    def wrap(owner: Any, name: str, identify: Identify) -> None:
        wrap_recording(stack, owner, name, identify)

    original_getattr = ctypes.CDLL.__getattr__

    def observed_getattr(self: Any, name: str) -> Any:
        found = original_getattr(self, name)
        if name in NATIVE_OPENERS:
            found = _Watched(found, name)

            object.__setattr__(self, name, found)
        return found

    ctypes.CDLL.__getattr__ = observed_getattr
    stack.callback(setattr, ctypes.CDLL, "__getattr__", original_getattr)

    with contextlib.suppress(ImportError):
        import onnxruntime

        original_session_init = onnxruntime.InferenceSession.__init__

        def observed_session_init(self: Any, *args: Any, **kwargs: Any) -> Any:
            with contextlib.suppress(TypeError, ValueError, AttributeError, IndexError):
                where = _argument(args, kwargs, 0, "path_or_bytes")
                if where:
                    _RECORDER.note("onnxruntime.InferenceSession", Path(where).name, where)
            return original_session_init(self, *args, **kwargs)

        onnxruntime.InferenceSession.__init__ = observed_session_init
        stack.callback(setattr, onnxruntime.InferenceSession, "__init__", original_session_init)

    with contextlib.suppress(ImportError):
        import huggingface_hub

        def one_file(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            repo = _argument(args, kwargs, 0, "repo_id")
            _RECORDER.note(
                "hf_hub_download",
                f"{repo}:{_argument(args, kwargs, 1, 'filename')}",
                "",
                str(kwargs.get("revision") or ""),
            )

        def whole_snapshot(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            _RECORDER.note(
                "snapshot_download", _argument(args, kwargs, 0, "repo_id"), "", str(kwargs.get("revision") or "")
            )

        wrap(huggingface_hub, "hf_hub_download", one_file)
        wrap(huggingface_hub, "snapshot_download", whole_snapshot)

    with contextlib.suppress(ImportError):
        from insightface.app import FaceAnalysis

        original_init = FaceAnalysis.__init__

        def observed_init(self: Any, *args: Any, **kwargs: Any) -> Any:
            with contextlib.suppress(TypeError, ValueError, AttributeError, IndexError):
                _RECORDER.note("FaceAnalysis", str(kwargs.get("name") or (args[0] if args else "buffalo_l(default)")))
            return original_init(self, *args, **kwargs)

        FaceAnalysis.__init__ = observed_init
        stack.callback(setattr, FaceAnalysis, "__init__", original_init)

    def pretrained(label: str) -> Identify:
        def identify(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            _RECORDER.note(
                label,
                _argument(args, kwargs, 0, "pretrained_model_name_or_path"),
                "",
                str(kwargs.get("revision") or ""),
            )

        return identify

    for module_name, class_name in (
        ("transformers", "PreTrainedModel"),
        ("transformers", "AutoModel"),
        ("transformers", "AutoProcessor"),
        ("diffusers", "DiffusionPipeline"),
        ("diffusers", "ModelMixin"),
    ):
        with contextlib.suppress(ImportError, AttributeError):
            owner = getattr(__import__(module_name, fromlist=[class_name]), class_name)
            wrap(owner, "from_pretrained", pretrained(f"{class_name}.from_pretrained"))


@contextlib.contextmanager
def recording(extra_roots: tuple[Path, ...] = ()) -> Generator[Recorder]:
    install()
    _ROOTS.clear()
    _ROOTS.extend(model_roots())
    _ROOTS.extend(one.resolve() for one in extra_roots)
    _RECORDER.seen.clear()
    _RECORDER.active = True
    try:
        with contextlib.ExitStack() as stack:
            _wrap_hub_apis(stack)
            yield _RECORDER
    finally:
        _RECORDER.active = False


CHILD_DIR: Final[Path] = Path(__file__).resolve().parent / "childobserve"


def child_env(out: Path, extra_roots: tuple[Path, ...] = ()) -> dict[str, str]:
    roots = [*model_roots(), *(one.resolve() for one in extra_roots)]
    held = dict(os.environ)
    existing = held.get("PYTHONPATH", "")
    held["PYTHONPATH"] = f"{CHILD_DIR}{os.pathsep}{existing}" if existing else str(CHILD_DIR)
    held["COMPAT_OBSERVE_CHILD_OUT"] = str(out)
    held["COMPAT_OBSERVE_CHILD_ROOTS"] = os.pathsep.join(str(one) for one in roots)
    return held


def absorb(out: Path, recorder: Recorder | None = None) -> int:
    into = recorder if recorder is not None else _RECORDER
    if not out.is_file():
        return 0
    try:
        rows = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    for row in rows:
        into.note(str(row.get("loader", "open")), str(row.get("identity", "")), str(row.get("path", "")))
    return len(rows)


def write(recorder: Recorder, where: Path | None = None) -> Path:
    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    target = where or (generated / "observed_artifacts.json")
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps({"observations": recorder.rows()}, indent=2, sort_keys=True, default=str))
        handle.write("\n")
    return target
