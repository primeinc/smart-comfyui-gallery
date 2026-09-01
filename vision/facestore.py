"""The canonical face record: a producer's complete output, frozen whole.

A face pass is the expensive thing in this application, and what it emits is
kilobytes. What a producer emits is therefore captured by ITERATION -- every
key, whatever its name -- and serialized in a typed envelope that records
dtype, shape, byte order and container structure, so the exact values come
back without the producer, the original result object, or the source pixels.

This module knows no field names. `freeze` walks whatever mapping it is
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

Wire layout, versioned by the magic:

    b"sgface1\\n"  u32-LE header length  header JSON (utf-8)  payload bytes

Header: {"producer", "producer_version", "container", "root": <node>}.
`container` records the producer's native result type (for insightface,
`Face` is a dict subclass), so a replay can rebuild the container a consumer
subscripts, not just the values.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Native", "Unpreservable", "freeze", "thaw"]

_MAGIC = b"sgface1\n"

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


def _dtype_str(where: str, value: object, dtype: np.dtype) -> str:
    if dtype.hasobject or dtype.kind not in _FIXED_KINDS:
        raise _refuse(where, value, f"has dtype {dtype!r}, whose bytes do not identify its values")
    return dtype.str


def _encode(value: object, where: str, payload: bytearray) -> dict[str, Any]:
    """One value as a typed node, appending raw bytes to `payload`.

    The isinstance ladder's order is load-bearing: `bool` subclasses `int`
    and `np.generic` scalars would also answer to the plain-number tests, so
    the narrower questions come first.
    """
    if value is None:
        return {"t": "z"}
    if isinstance(value, np.ndarray):
        spec: dict[str, Any] = {
            "t": "nd",
            "d": _dtype_str(where, value, value.dtype),
            "s": [int(one) for one in value.shape],
            "at": len(payload),
            "n": value.nbytes,
        }
        # tobytes always emits C-order; the one layout fact observable through
        # the array itself (`flags`) is F-contiguity, so that flag alone is
        # kept and restored.
        if value.ndim > 1 and value.flags["F_CONTIGUOUS"] and not value.flags["C_CONTIGUOUS"]:
            spec["order"] = "F"
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
        return {"t": "bool", "v": value}
    if isinstance(value, int):
        # json round-trips arbitrary-precision ints exactly; no string form needed
        return {"t": "i", "v": value}
    if isinstance(value, float):
        # hex, not repr: exact for every finite value, and 'nan'/'inf' parse back
        return {"t": "f", "v": value.hex()}
    if isinstance(value, complex):
        # Packed IEEE doubles, which is what a complex IS; hex text would do
        # too, but the payload already exists and struct carries NaN and the
        # infinities without a parser in the loop.
        spec = {"t": "c", "at": len(payload), "n": 16}
        payload.extend(struct.pack("<dd", value.real, value.imag))
        return spec
    if isinstance(value, str):
        return {"t": "s", "v": value}
    if isinstance(value, (bytes, bytearray)):
        spec = {"t": "ba" if isinstance(value, bytearray) else "by", "at": len(payload), "n": len(value)}
        payload.extend(value)
        return spec
    if isinstance(value, (list, tuple)):
        kind = "tu" if isinstance(value, tuple) else "l"
        return {"t": kind, "v": [_encode(one, f"{where}[{i}]", payload) for i, one in enumerate(value)]}
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, (str, int, float, bool, bytes)) and key is not None:
                raise _refuse(f"{where} key {key!r}", key, "cannot key a restored mapping")
        return {
            "t": "d",
            "k": [_encode(key, f"{where} key {key!r}", payload) for key in value],
            "v": [_encode(value[key], f"{where}[{key!r}]", payload) for key in value],
        }
    raise _refuse(where, value, "is not a type this envelope preserves")


def _decode(node: dict[str, Any], payload: memoryview) -> Any:
    kind = node["t"]
    if kind == "z":
        return None
    if kind == "nd":
        raw = payload[node["at"] : node["at"] + node["n"]]
        # copy(): frombuffer views the stored bytes read-only, and a
        # producer's own arrays are writable -- a replay must hand back what
        # a consumer could have mutated.
        values = np.frombuffer(raw, dtype=np.dtype(node["d"])).reshape(node["s"]).copy()
        return np.asfortranarray(values) if node.get("order") == "F" else values
    if kind == "ns":
        raw = payload[node["at"] : node["at"] + node["n"]]
        return np.frombuffer(raw, dtype=np.dtype(node["d"]))[0]
    if kind == "bool":
        return bool(node["v"])
    if kind == "i":
        return int(node["v"])
    if kind == "f":
        return float.fromhex(node["v"])
    if kind == "c":
        real, imag = struct.unpack("<dd", payload[node["at"] : node["at"] + node["n"]])
        return complex(real, imag)
    if kind == "s":
        return str(node["v"])
    if kind in ("by", "ba"):
        raw = bytes(payload[node["at"] : node["at"] + node["n"]])
        return bytearray(raw) if kind == "ba" else raw
    if kind == "l":
        return [_decode(one, payload) for one in node["v"]]
    if kind == "tu":
        return tuple(_decode(one, payload) for one in node["v"])
    if kind == "d":
        return {_decode(key, payload): _decode(value, payload) for key, value in zip(node["k"], node["v"], strict=True)}
    raise ValueError(f"envelope names a node type {kind!r} this build does not know")


@dataclass(frozen=True)
class Native:
    """One thawed record: the producer's complete output plus its provenance."""

    producer: str
    producer_version: str
    container: str
    record: dict[str, Any]


def freeze(record: dict[str, Any], *, producer: str, producer_version: str, container: str) -> bytes:
    """The complete record as one durable blob. Walks by iteration; no field
    is named. Raises `Unpreservable` -- never drops -- on a value the
    envelope cannot carry."""
    payload = bytearray()
    root = _encode(dict(record), "record", payload)
    header = json.dumps(
        {"producer": producer, "producer_version": producer_version, "container": container, "root": root},
        separators=(",", ":"),
    ).encode("utf-8")
    return b"".join((_MAGIC, len(header).to_bytes(4, "little"), header, bytes(payload)))


def thaw(blob: bytes) -> Native:
    """The record back, values bit-exact in their original dtypes and shapes.

    Raises ValueError on bytes that are not this envelope -- a truncated blob
    or a foreign format must never read as an empty record."""
    if blob[: len(_MAGIC)] != _MAGIC:
        raise ValueError(f"not a face-native envelope: leads with {blob[:8]!r}")
    header_len = int.from_bytes(blob[len(_MAGIC) : len(_MAGIC) + 4], "little")
    body_at = len(_MAGIC) + 4
    header = json.loads(blob[body_at : body_at + header_len].decode("utf-8"))
    payload = memoryview(blob)[body_at + header_len :]
    record = _decode(header["root"], payload)
    if not isinstance(record, dict):
        raise TypeError(f"envelope root is {type(record).__name__}, not a mapping")
    return Native(
        producer=str(header["producer"]),
        producer_version=str(header["producer_version"]),
        container=str(header["container"]),
        record=record,
    )
