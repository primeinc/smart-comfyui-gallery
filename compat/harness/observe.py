"""Record every external artifact a run actually opens.

Static discovery reads the pinned source and finds what a loader COULD resolve.
It cannot see a filename assembled at runtime, a parameter default bound at
call time, or a branch a config chooses -- the regenerated population carries
168 UNRESOLVED variants and 211 unresolved call sites for exactly that reason.

This closes the other half: what is recorded is what the process resolved.

WHY AN AUDIT HOOK AND NOT A MONKEYPATCH
---------------------------------------
`from module import load` binds a reference at import time, so a patch applied
afterwards is never consulted and the observer misses the load while reporting
success. That is the instrumentation equivalent of a green over zero files.

`sys.addaudithook` is CPython's own mechanism and fires inside the
implementation, so an aliased reference is still seen. Measured here rather
than assumed: with a hook installed, both `open(path)` and
`from builtins import open as aliased; aliased(path)` were captured.

The trade is that a hook cannot be REMOVED once added -- documented CPython
behaviour -- so recording is gated on a flag rather than on installing and
uninstalling the hook.

WHAT IS STILL PATCHED, AND WHY
------------------------------
An audit hook sees files. It does not see a load that resolves entirely inside
a cache, or one whose identity is a repository rather than a path:
`from_pretrained("org/model")` names bytes without naming a file. Those APIs
are wrapped so the REQUESTED identity is recorded beside the resolved path.

RECONCILIATION IS THE POINT, not the recording. Three outcomes, three different
pieces of work:

    static and dynamic agree     the edge exists and was taken
    static only, not observed    UNEXERCISED -- nothing ran that variant
    observed, not static         POPULATION DEFECT -- discovery missed it

`reconcile.py` does that comparison and decides what UNEXERCISED may mean,
from the coverage `observe_attack` measured on the same run.
"""

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

#: Suffixes that make an opened file model bytes rather than source or data.
MODEL_SUFFIX: Final[frozenset[str]] = frozenset(
    {".onnx", ".pth", ".pt", ".bin", ".safetensors", ".task", ".ckpt", ".npy", ".npz", ".zip"}
)


@dataclass
class Observation:
    """One artifact a run actually resolved.

    No load COUNT. How many times a process opened a file is a property of
    this process's caching -- a warm corpus cache, a memoised pack, a shard
    boundary -- exactly as wall-clock is a property of the machine, and
    `attack` compares two runs of this evidence byte for byte. The claim the
    reconciler needs is THAT the artifact was resolved and by which loader.
    """

    loader: str
    identity: str
    path: str = ""
    revision: str = ""


@dataclass
class Recorder:
    """Everything one run resolved, keyed on (loader, identity)."""

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


#: One recorder for the process. An audit hook cannot be removed, so the hook
#: is installed once and reads this; `recording()` turns it on and off.
_RECORDER: Final[Recorder] = Recorder()
_INSTALLED: list[bool] = []


def model_roots() -> tuple[Path, ...]:
    """Directories under which an opened file is a MODEL rather than noise.

    Derived from the manifest's own weight roots plus the hub cache, never a
    typed list: a root this misses makes its artifacts invisible, which is the
    silent pass the observer exists to prevent.

    Suffix alone is not enough. `.zip` is here because ReActor ships
    `buffalo_l.zip`, and the interpreter's own `python313.zip` was recorded
    seventy times in the control run -- real opens, of something that is not a
    model.
    """
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


#: Symbols that open a file without CPython's `open` event ever firing. A C
#: extension resolving one of these can read a model the audit hook cannot see.
NATIVE_OPENERS: Final[frozenset[str]] = frozenset(
    {"fopen", "fopen_s", "_wfopen", "_wfopen_s", "_open", "_wopen", "open", "CreateFileA", "CreateFileW"}
)

#: A native opener resolved but not yet called. `ctypes.dlsym` is audited, so
#: arming one is recorded even where the call itself is not reachable.
NATIVE_UNSEEN: Final[str] = "native_open_unseen"


class _Watched:
    """One native function, recording the path it is handed.

    A proxy rather than a plain wrapper because a caller configures the symbol
    it gets back -- `restype`, `argtypes` -- and those must reach the real
    function or the call returns the wrong type.
    """

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
    """CPython's own events: a file open, and a native opener being armed."""
    if event == "ctypes.dlsym" and len(args) >= 2:
        # A symbol armed and never called: the call is intercepted below, and
        # this records the arming, so a resolve with no call still shows up.
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
    """Install the audit hook once, for the life of the process."""
    if _INSTALLED:
        return
    sys.addaudithook(_audit)
    _INSTALLED.append(True)


def _argument(args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str) -> str:
    """One argument, whether it arrived positionally or by keyword."""
    if name in kwargs and isinstance(kwargs[name], (str, os.PathLike)):
        return str(kwargs[name])
    if len(args) > index and isinstance(args[index], (str, os.PathLike)):
        return str(args[index])
    return ""


Identify = Callable[[tuple[Any, ...], dict[str, Any]], None]


def wrap_recording(stack: contextlib.ExitStack, owner: Any, name: str, identify: Identify) -> None:
    """Record what one loader was asked for, without changing what it does.

    A CLASSMETHOD replaced by a plain function loses its `cls` binding, and
    every subclass then runs the base class's implementation:
    `Sam3VideoModel.from_pretrained` became `PreTrainedModel`'s and raised
    "PreTrainedModel does not define `config_class`" -- four cases failing on
    the observer rather than on anything they measure.

    Module level so `observe_attack` can hold this exact mechanism to a
    control rather than a copy of it.
    """
    original = getattr(owner, name, None)
    if original is None:
        return

    declared = inspect.getattr_static(owner, name, None)
    if isinstance(declared, classmethod):
        underlying = declared.__func__

        def observed_classmethod(cls: Any, *args: Any, **kwargs: Any) -> Any:
            # Recording may never change what the run does.
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
    """Wrap the loaders whose identity is a repository rather than a file.

    Signatures read at their pins:
        hf_hub_download(repo_id, filename, *, revision=...)
            huggingface/huggingface_hub src/huggingface_hub/file_download.py:757
        snapshot_download(repo_id, *, revision=...)
            huggingface/huggingface_hub src/huggingface_hub/_snapshot_download.py
    """

    def wrap(owner: Any, name: str, identify: Identify) -> None:
        wrap_recording(stack, owner, name, identify)

    # No `open` event fires for a native open, but ctypes resolves symbols
    # through `CDLL.__getattr__` in PYTHON, so the call is reachable and its
    # first argument is the path.
    original_getattr = ctypes.CDLL.__getattr__

    def observed_getattr(self: Any, name: str) -> Any:
        found = original_getattr(self, name)
        if name in NATIVE_OPENERS:
            found = _Watched(found, name)
            # ctypes caches the resolved symbol on the instance, so the cache
            # has to hold the wrapper or the second access returns the raw
            # function and every later call is invisible again.
            object.__setattr__(self, name, found)
        return found

    ctypes.CDLL.__getattr__ = observed_getattr
    stack.callback(setattr, ctypes.CDLL, "__getattr__", original_getattr)

    with contextlib.suppress(ImportError):
        import onnxruntime

        # NATIVE: ORT opens the graph in C++, so no Python `open` fires and
        # the audit hook cannot see it. `InferenceSession(path_or_bytes, ...)`
        # -- microsoft/onnxruntime onnxruntime_inference_collection.py:476.
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
                # The PACK is the variant. `FaceAnalysis()` is buffalo_l and
                # `FaceAnalysis(name="antelopev2")` is a different embedding
                # space, so the default is recorded as itself, not as absence.
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
    """Record every artifact resolved inside the block.

    `extra_roots` widens the scope for a caller that owns a directory the
    manifest does not name. It exists for the observer's own controls: a probe
    written into a REAL model pack is loaded by the next consumer that globs
    that directory, and one did -- insightface tried to parse a dummy
    `_observer_probe.onnx` as a graph and the run died. A control may never
    write where production reads.
    """
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


#: Prepended to a child's PYTHONPATH so CPython imports it at startup.
CHILD_DIR: Final[Path] = Path(__file__).resolve().parent / "childobserve"


def child_env(out: Path, extra_roots: tuple[Path, ...] = ()) -> dict[str, str]:
    """Environment that makes a PYTHON child record its own opens.

    An audit hook is per-process, so the parent's cannot see a child's. The
    child gets its own, installed by `sitecustomize` before any user code --
    the only moment early enough for a load that happens at import time.

    Only a python child records. `git` and `taskkill` are the other things
    this repository spawns and neither opens a model, but that is a claim
    about those two programs and not a property of this mechanism, so
    `absorb` reports what came back rather than assuming it was everything.
    """
    roots = [*model_roots(), *(one.resolve() for one in extra_roots)]
    held = dict(os.environ)
    existing = held.get("PYTHONPATH", "")
    held["PYTHONPATH"] = f"{CHILD_DIR}{os.pathsep}{existing}" if existing else str(CHILD_DIR)
    held["COMPAT_OBSERVE_CHILD_OUT"] = str(out)
    held["COMPAT_OBSERVE_CHILD_ROOTS"] = os.pathsep.join(str(one) for one in roots)
    return held


def absorb(out: Path, recorder: Recorder | None = None) -> int:
    """Fold a child's recording into this process's, returning how many."""
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
