"""The canonical face record: a producer's complete output, frozen whole.

A face pass is the expensive thing in this application, and what it emits is
kilobytes. What a producer emits is therefore captured by ITERATION -- every
key, whatever its name -- and serialized in a typed envelope that records
dtype, shape, byte order and container structure, so the exact values come
back without the producer, the original result object, or the source pixels.

This module knows no field names. `freeze` walks whatever value it is
handed; a producer growing a new output changes nothing here. That is the
boundary rule the promoted columns in `derived_face_instance` are projections
OUT of -- they decide what a facet can index, never what survives.

A value this envelope cannot carry faithfully raises `Unpreservable`, naming
the exact path and runtime type. Never a silent omission: a dropped value is
recoverable only by re-reading the whole library, and only while the
originals are still on disk.

JSON alone cannot do this job. A float32 array through a JSON list comes back
as Python floats with its dtype and shape gone -- the reader has to already
know what the writer held, which is the allowlist again, one layer down. The
envelope is a JSON header (types, dtypes, shapes, offsets) over a raw byte
payload (array and scalar contents, bit-exact; `ndarray.tobytes` and
`frombuffer` are documented inverses, and byte order rides inside the
recorded dtype string).

What a record preserves beyond values, and what it does not:

  ALIASING is structure, not a detail of how the record was written down.
  One array reached through two keys is stored once and thawed once, so a
  consumer that writes through one name still reads it through the other.
  The node table is flat and every child is an index into it, which is what
  makes one object one node.

  A CYCLE is refused, by path, at capture. Nothing here follows a value into
  itself, so the report names the producer key that closed the loop rather
  than the recursion limit of this module.

  CONTAINERS are recorded per node from the value's own type. A subclass of
  list, tuple or dict is rebuilt through a registered adapter or refused by
  name -- never widened into the builtin it derives from, which would hand a
  replay a plainer object than the producer returned.

  LAYOUT is the one thing normalized rather than kept. Values survive
  bit-exact; F-contiguity is restored because a consumer can observe it, and
  every other stride pattern comes back C-contiguous. A record of a strided
  view is a record of its values, not of the slicing that made it.

Wire layout, versioned by the magic:

    b"sgface3\\n"  u32-LE header length  header JSON (utf-8)  payload bytes
    sha256 over everything before it (32 raw bytes, the trailer)

Header: {"producer", "producer_version", "container", "root", "nodes"}.
`nodes` is the flat table and `root` indexes it. `container` records the
producer's native result type (for insightface, `Face` is a dict subclass),
and thaw REBUILDS it: the capture sites hand over a mapping and name the
type beside it, so a replay subscripts and reads attributes exactly as it
would the producer's own object. A container name no adapter can rebuild is
refused at capture, so the field is a promise the envelope keeps rather than
a label nothing acts on. The trailing digest makes silent payload corruption
a loud refusal instead of a wrong array.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

import numpy as np

__all__ = ["InsightFaceRecord", "Native", "Unpreservable", "freeze", "register_container", "thaw"]

_MAGIC = b"sgface3\n"

#: Magics this build refuses by name rather than reading. An envelope whose
#: node table has another shape cannot be reinterpreted under this codec,
#: which is the failure that made the corpus cache hash its own codec.
_SUPERSEDED = {b"sgface2\n": "2"}

#: dtype kinds whose bytes identify their values: bool, ints, uints, floats,
#: complex, bytes/unicode strings, datetimes, timedeltas. 'O' holds pointers
#: and 'V' keeps its field layout outside `dtype.str`; both refuse by name.
_FIXED_KINDS = frozenset("biufcSUmM")


class Unpreservable(TypeError):
    """A producer emitted a value this envelope cannot carry faithfully.

    Raised at capture, naming the path and the runtime type, so the failure
    lands at the pass that could still re-run -- not at a replay years later
    discovering a hole."""


def _refuse(where: str, value: object, why: str) -> Unpreservable:
    return Unpreservable(f"{where}: {type(value).__module__}.{type(value).__qualname__} {why}")


def _dotted(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


#: Container name -> the callable rebuilding it from the builtin a record
#: decodes into. A name absent here is refused at capture rather than widened
#: at replay, so every stored container has a way back.
_ADAPTERS: dict[str, Callable[[Any], Any]] = {}


def register_container(name: str, rebuild: Callable[[Any], Any]) -> None:
    """Declare how a container recorded under `name` is rebuilt on thaw.

    `name` is the dotted `module.QualName` of the producer's own class, and
    `rebuild` takes the plain list, tuple or mapping the record decoded into
    and returns the container a consumer expects.

    A second, different rebuild for a name already registered is refused:
    two adapters for one recorded name is how entries written under the
    first get reinterpreted under the second.
    """
    held = _ADAPTERS.get(name)
    if held is not None and held is not rebuild:
        raise ValueError(f"{name} is already rebuilt by {held!r}; a second adapter would reinterpret stored records")
    _ADAPTERS[name] = rebuild


class InsightFaceRecord(dict):
    """insightface's `Face` protocol, rebuilt without importing insightface.

    deepinsight/insightface@7fadd420c2351d0ffa8cac403421c1a3ed733365
    python-package/insightface/app/common.py:4-48: a dict subclass whose
    writes land in both the mapping and the instance dictionary, whose
    absent attributes read as None rather than raising, and which derives
    `sex`, `embedding_norm` and `normed_embedding` from what it holds.

    Rebuilt here rather than imported because a cold replay must not load
    the producer stack: the class a consumer subscripts has to exist on a
    machine where insightface does not. The envelope still records the name
    upstream gave it, so what the record claims stays upstream's own.
    """

    def __init__(self, d: dict | None = None, **kwargs: Any) -> None:
        super().__init__()
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for key, value in d.items():
            setattr(self, key, value)

    @override
    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, (list, tuple)):
            value = [self.__class__(one) if isinstance(one, dict) else one for one in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name: str) -> Any:
        return None

    @classmethod
    def rebuilt(cls, items: dict) -> InsightFaceRecord:
        """The record as stored, with the promoting constructor bypassed.

        Upstream's `__init__` turns a nested mapping into another `Face` on
        the way in. The record already says what each nested container was,
        so re-running that promotion would overwrite a stored plain mapping
        with a class it never had.
        """
        face = cls.__new__(cls)
        dict.update(face, items)
        face.__dict__.update({key: value for key, value in items.items() if isinstance(key, str)})
        return face

    @property
    def embedding_norm(self) -> Any:
        if self.embedding is None:
            return None
        return np.linalg.norm(self.embedding)

    @property
    def normed_embedding(self) -> Any:
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property
    def sex(self) -> str | None:
        if self.gender is None:
            return None
        return "M" if self.gender == 1 else "F"


register_container("insightface.app.common.Face", InsightFaceRecord.rebuilt)


class _Written:
    """One freeze in progress: the node table, the payload, and the two
    identity maps that make aliasing survive and a cycle refuse."""

    __slots__ = ("active", "memo", "nodes", "payload")

    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.payload = bytearray()
        # id() -> (node index, the object itself). Holding the object stops a
        # value being collected mid-walk and its address handed to the next
        # one, which would read as an alias.
        self.memo: dict[int, tuple[int, Any]] = {}
        # id() -> the path a container is being written at, entered before its
        # children and left after them. A hit is a cycle.
        self.active: dict[int, str] = {}


def _dtype_str(where: str, value: object, dtype: np.dtype) -> str:
    if dtype.hasobject or dtype.kind not in _FIXED_KINDS:
        raise _refuse(where, value, f"has dtype {dtype!r}, whose bytes do not identify its values")
    return dtype.str


def _container(spec: dict[str, Any], value: object, where: str, builtin: type) -> None:
    """Record this node's own class when it is not the plain builtin.

    Every node kind whose Python type can be subclassed passes through here,
    not only the three collections: a `np.ma.MaskedArray` decoded as an
    ndarray has lost its mask, and an `IntEnum` decoded as an int has lost
    which member it was. Both are the same silent widening.
    """
    cls = type(value)
    if cls is builtin:
        return
    name = _dotted(cls)
    if name not in _ADAPTERS:
        raise _refuse(where, value, f"subclasses {builtin.__name__} and no registered adapter rebuilds it")
    spec["c"] = name


def _encode(written: _Written, value: object, where: str) -> int:
    """One value as a node index, appending nodes and raw bytes as it goes.

    An object already written comes back as its existing index, which is what
    carries aliasing through the wire. An object still being written is a
    cycle, and refuses here, where the producer path is still known.
    """
    known = written.memo.get(id(value))
    if known is not None:
        return known[0]
    opened = written.active.get(id(value))
    if opened is not None:
        raise _refuse(where, value, f"is the container already being written at {opened}, so the record holds itself")
    node = _node(written, value, where)
    written.nodes.append(node)
    written.memo[id(value)] = (len(written.nodes) - 1, value)
    return len(written.nodes) - 1


def _node(written: _Written, value: object, where: str) -> dict[str, Any]:
    """One value as a typed node, appending raw bytes to the payload.

    The isinstance ladder's order is load-bearing: `bool` subclasses `int`
    and `np.generic` scalars would also answer to the plain-number tests, so
    the narrower questions come first.
    """
    payload = written.payload
    if value is None:
        return {"t": "z"}
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        # A tensor can only exist in a process that already imported torch,
        # so the module reference costs no import. Values serialize through
        # numpy; a dtype numpy cannot spell (bfloat16, complex32) refuses.
        try:
            flat = value.detach().cpu().contiguous().numpy()
        except TypeError as why:
            raise _refuse(where, value, f"holds torch dtype {value.dtype}, which numpy cannot spell: {why}") from why
        spec: dict[str, Any] = {
            "t": "tt",
            "d": _dtype_str(where, value, flat.dtype),
            "s": [int(one) for one in value.shape],
            "at": len(payload),
            "n": flat.nbytes,
            "dev": str(value.device),
        }
        _container(spec, value, where, torch.Tensor)
        payload.extend(flat.tobytes())
        return spec
    if isinstance(value, np.ndarray):
        spec = {
            "t": "nd",
            "d": _dtype_str(where, value, value.dtype),
            "s": [int(one) for one in value.shape],
            "at": len(payload),
            "n": value.nbytes,
        }
        # tobytes emits C-order whatever the strides are, so the values survive
        # and the layout does not. F-contiguity is restored because a consumer
        # can observe it; every other stride pattern comes back C-contiguous.
        if value.ndim > 1 and value.flags["F_CONTIGUOUS"] and not value.flags["C_CONTIGUOUS"]:
            spec["order"] = "F"
        _container(spec, value, where, np.ndarray)
        payload.extend(value.tobytes())
        return spec
    if isinstance(value, np.generic):
        spec = {
            "t": "ns",
            "d": _dtype_str(where, value, value.dtype),
            "at": len(payload),
            "n": value.nbytes,
        }
        payload.extend(value.tobytes())
        return spec
    if isinstance(value, bool):
        # bool takes no subclass, so this node needs no container of its own
        return {"t": "bool", "v": value}
    if isinstance(value, int):
        # json round-trips arbitrary-precision ints exactly, and formats an int
        # subclass through `int.__repr__`, so a member's own class travels in
        # the container beside the number rather than in it.
        spec = {"t": "i", "v": value}
        _container(spec, value, where, int)
        return spec
    if isinstance(value, float):
        # hex, not repr: exact for every finite value, and 'nan'/'inf' parse back
        spec = {"t": "f", "v": value.hex()}
        _container(spec, value, where, float)
        return spec
    if isinstance(value, complex):
        # Packed IEEE doubles, which is what a complex IS; hex text would do
        # too, but the payload already exists and struct carries NaN and the
        # infinities without a parser in the loop.
        spec = {"t": "c", "at": len(payload), "n": 16}
        _container(spec, value, where, complex)
        payload.extend(struct.pack("<dd", value.real, value.imag))
        return spec
    if isinstance(value, str):
        spec = {"t": "s", "v": value}
        _container(spec, value, where, str)
        return spec
    if isinstance(value, (bytes, bytearray)):
        held = isinstance(value, bytearray)
        spec = {"t": "ba" if held else "by", "at": len(payload), "n": len(value)}
        _container(spec, value, where, bytearray if held else bytes)
        payload.extend(value)
        return spec
    if isinstance(value, (list, tuple)):
        held = isinstance(value, tuple)
        spec = {"t": "tu" if held else "l"}
        _container(spec, value, where, tuple if held else list)
        written.active[id(value)] = where
        try:
            spec["v"] = [_encode(written, one, f"{where}[{i}]") for i, one in enumerate(value)]
        finally:
            del written.active[id(value)]
        return spec
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, (str, int, float, bool, bytes)) and key is not None:
                raise _refuse(f"{where} key {key!r}", key, "cannot key a restored mapping")
        spec = {"t": "d"}
        _container(spec, value, where, dict)
        written.active[id(value)] = where
        try:
            spec["k"] = [_encode(written, key, f"{where} key {key!r}") for key in value]
            spec["v"] = [_encode(written, value[key], f"{where}[{key!r}]") for key in value]
        finally:
            del written.active[id(value)]
        return spec
    raise _refuse(where, value, "is not a type this envelope preserves")


def _restore(name: str | None, built: Any) -> Any:
    if name is None:
        return built
    rebuild = _ADAPTERS.get(name)
    if rebuild is None:
        raise ValueError(f"the record names container {name!r}, which this build has no adapter to rebuild")
    return rebuild(built)


def _decode(nodes: list[dict[str, Any]], index: int, payload: memoryview, done: dict[int, Any]) -> Any:
    """One node by index, memoized so a shared node thaws to a shared object."""
    if index in done:
        return done[index]
    value = _value(nodes, nodes[index], payload, done)
    done[index] = value
    return value


def _value(nodes: list[dict[str, Any]], node: dict[str, Any], payload: memoryview, done: dict[int, Any]) -> Any:
    kind = node["t"]
    held = node.get("c")
    if kind == "z":
        return None
    if kind == "nd":
        raw = payload[node["at"] : node["at"] + node["n"]]
        # copy(): frombuffer views the stored bytes read-only, and a
        # producer's own arrays are writable -- a replay must hand back what
        # a consumer could have mutated.
        values = np.frombuffer(raw, dtype=np.dtype(node["d"])).reshape(node["s"]).copy()
        return _restore(held, np.asfortranarray(values) if node.get("order") == "F" else values)
    if kind == "tt":
        # The reader of a tensor record needs torch; on CPU, whatever device
        # produced it -- the recorded device is provenance, not a demand a
        # replay machine must satisfy.
        import torch

        raw = payload[node["at"] : node["at"] + node["n"]]
        values = np.frombuffer(raw, dtype=np.dtype(node["d"])).reshape(node["s"]).copy()
        return _restore(held, torch.from_numpy(values))
    if kind == "ns":
        # A numpy scalar's own type is what its dtype spells, so the node
        # needs no container beside it.
        raw = payload[node["at"] : node["at"] + node["n"]]
        return np.frombuffer(raw, dtype=np.dtype(node["d"]))[0]
    if kind == "bool":
        return bool(node["v"])
    if kind == "i":
        return _restore(held, int(node["v"]))
    if kind == "f":
        return _restore(held, float.fromhex(node["v"]))
    if kind == "c":
        real, imag = struct.unpack("<dd", payload[node["at"] : node["at"] + node["n"]])
        return _restore(held, complex(real, imag))
    if kind == "s":
        return _restore(held, str(node["v"]))
    if kind in ("by", "ba"):
        raw = bytes(payload[node["at"] : node["at"] + node["n"]])
        return _restore(held, bytearray(raw) if kind == "ba" else raw)
    if kind == "l":
        return _restore(held, [_decode(nodes, one, payload, done) for one in node["v"]])
    if kind == "tu":
        return _restore(held, tuple(_decode(nodes, one, payload, done) for one in node["v"]))
    if kind == "d":
        built = {
            _decode(nodes, key, payload, done): _decode(nodes, one, payload, done)
            for key, one in zip(node["k"], node["v"], strict=True)
        }
        return _restore(held, built)
    raise ValueError(f"envelope names a node type {kind!r} this build does not know")


@dataclass(frozen=True)
class Native:
    """One thawed result: the producer's complete output plus its provenance.

    `value` is whatever the producer returned -- a mapping, an array, a
    tuple, a scalar, bytes -- in the container `container` names, rebuilt. A
    mapping is one structural shape among many, never the required root:
    demanding one would force a wrapper around every non-mapping return,
    which is semantic narrowing at the door.
    """

    producer: str
    producer_version: str
    container: str
    value: Any

    @property
    def record(self) -> dict[str, Any]:
        """The value as the mapping older callers expect; refuses by type
        when the root is not one, instead of wrapping or guessing."""
        if not isinstance(self.value, dict):
            raise TypeError(f"this result's root is {type(self.value).__name__}, not a mapping")
        return self.value


def _declared(value: object, container: str) -> None:
    """The declared container must be one thaw can honour.

    Either it names the root's own type, or an adapter rebuilds it. A name
    that is neither is a promise this envelope would break at replay, which
    is worse than refusing a capture that can still be re-run.
    """
    cls = type(value)
    if container in (_dotted(cls), cls.__qualname__):
        return
    rebuild = _ADAPTERS.get(container)
    if rebuild is None:
        raise Unpreservable(
            f"result declares container {container!r}, which is neither this value's own"
            f" {cls.__qualname__} nor a registered container adapter"
        )
    # The adapter runs on the root now, so a declaration it cannot take fails
    # at the pass that can re-run rather than at a replay. A decoded root has
    # the same builtin kind as this one, so what passes here passes there.
    try:
        rebuild(value)
    except (AttributeError, KeyError, TypeError, ValueError) as why:
        raise Unpreservable(
            f"result declares container {container!r}, which cannot be rebuilt from a {cls.__qualname__}: {why}"
        ) from why


def freeze(value: Any, *, producer: str, producer_version: str, container: str) -> bytes:
    """The complete result as one durable blob. Walks by iteration; no field
    is named, and any encodable root is accepted as returned. Raises
    `Unpreservable` -- never drops -- on a value the envelope cannot carry."""
    _declared(value, container)
    written = _Written()
    root = _encode(written, value, "result")
    header = json.dumps(
        {
            "producer": producer,
            "producer_version": producer_version,
            "container": container,
            "root": root,
            "nodes": written.nodes,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    body = b"".join((_MAGIC, len(header).to_bytes(4, "little"), header, bytes(written.payload)))
    return body + hashlib.sha256(body).digest()


def thaw(blob: bytes) -> Native:
    """The record back, values bit-exact in their original dtypes and shapes,
    in the container the record names.

    Raises ValueError on bytes that are not this envelope -- a truncated blob,
    a foreign format, or a superseded version -- because a record must never
    read as an empty one, and never as a plausible wrong one."""
    if blob[: len(_MAGIC)] != _MAGIC:
        stale = _SUPERSEDED.get(blob[: len(_MAGIC)])
        if stale is not None:
            raise ValueError(
                f"this record was written by face-native envelope version {stale}, whose node table has"
                f" another shape; re-run the producer rather than reading it under this codec"
            )
        raise ValueError(f"not a face-native envelope: leads with {blob[:8]!r}")
    body, trailer = blob[:-32], blob[-32:]
    if hashlib.sha256(body).digest() != trailer:
        raise ValueError("face-native envelope failed its digest: the stored bytes are not the bytes written")
    header_len = int.from_bytes(body[len(_MAGIC) : len(_MAGIC) + 4], "little")
    body_at = len(_MAGIC) + 4
    header = json.loads(body[body_at : body_at + header_len].decode("utf-8"))
    payload = memoryview(body)[body_at + header_len :]
    nodes = header["nodes"]
    root = int(header["root"])
    value = _decode(nodes, root, payload, {})
    container = str(header["container"])
    rebuild = _ADAPTERS.get(container)
    # A root node carries its own container when the caller froze the real
    # object, and rebuilding from the header too would wrap it twice. The
    # declared name is honoured only for a root the record left plain.
    if rebuild is not None and "c" not in nodes[root]:
        value = rebuild(value)
    return Native(
        producer=str(header["producer"]),
        producer_version=str(header["producer_version"]),
        container=container,
        value=value,
    )
