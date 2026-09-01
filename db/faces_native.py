"""A stored face, replayed and exported from the row alone.

The write path (`db/detect.py`) freezes the producer's complete record into
`derived_face_instance.native`. This module is the read path: it thaws that
record and serves it -- to a caller that wants the producer's own values
back, or to an exporter that writes the file an upstream tool would load.

Nothing here touches the producer, the source image, or any model session.
That is the point of the column: a replay after the original file is gone
must hand back exactly what the pass emitted, and an export must be a file
the upstream tool accepts, produced without re-running anything.

The one upstream export format a stored face currently has is ReActor's
face model: `save_face_model` (Gourieff/ComfyUI-ReActor@6ad6b35a4df2
reactor_utils.py:184-197) builds `torch.tensor(face[key])` for nine named
keys, in source order, and hands the dict to `safetensors.torch.save_file`.
`reactor_face_model_bytes` reproduces that construction from the thawed
record; `compat/consumers/reactor_face_model.py` holds it byte-identical to
the pinned upstream writer, so the names below are upstream's export
contract restated -- never a filter on what this application stores, which
is decided one layer down by iteration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib

    from vision.facestore import Native


class NativeMissing(LookupError):
    """The row holds no native record, so there is nothing to replay.

    The stub backend is the one shipped backend that may write such a row;
    for a real backend's row the fix is re-running the faces job over the
    file, which recaptures the producer's output whole.
    """


#: `save_face_model`'s nine keys -- upstream's export contract restated, per
#: the module docstring. The serializer canonicalises layout itself, so only
#: the set, dtypes and values reach the bytes; dict order does not.
REACTOR_KEYS = ("bbox", "kps", "det_score", "landmark_3d_68", "pose", "landmark_2d_106", "embedding", "gender", "age")


def native_of(conn, face_id: int) -> Native:
    """The producer's complete record for one stored face, thawed.

    Values come back bit-exact in their original dtypes and shapes, keyed
    the producer's own way, with the producer's identity on the envelope.
    """
    from vision import facestore

    row = conn.execute("SELECT native FROM derived_face_instance WHERE id = ?", (face_id,)).fetchone()
    if row is None:
        raise LookupError(f"no stored face has id {face_id}")
    if row[0] is None:
        raise NativeMissing(
            f"face {face_id} carries no native record (a stub backend, or a row written before the"
            f" faces job last looked); re-run the faces job over its file to capture the producer's output"
        )
    return facestore.thaw(row[0])


def reactor_face_model_bytes(native: Native) -> bytes:
    """The exact file ReActor's `save_face_model` writes, from the record.

    Raises KeyError naming the absent keys rather than upstream's
    swallow-and-print: a face model missing a tensor is not a smaller face
    model, it is a file `load_face_model` cannot serve.
    """
    import numpy as np
    from safetensors.numpy import save

    missing = [key for key in REACTOR_KEYS if key not in native.record]
    if missing:
        raise KeyError(f"the {native.producer} record cannot fill ReActor's face model: it lacks {', '.join(missing)}")
    # safetensors' numpy writer: the same Rust serializer, bytes measured
    # identical to upstream's torch save_file -- and `import torch` after
    # onnxruntime is resident dies loading CUDA DLLs (0xc0000139).
    tensors = {key: np.asarray(native.record[key]) for key in REACTOR_KEYS}
    return save(tensors)


def export_reactor_face_model(conn, face_id: int, path: pathlib.Path | str) -> None:
    """One stored face as a ReActor face model file, producer untouched.

    The `.safetensors` suffix is part of upstream's contract -- ReActor
    discovers face models as `models/reactor/faces/*.safetensors` -- so a
    path without it is refused rather than silently renamed.
    """
    import pathlib

    where = pathlib.Path(path)
    if where.suffix != ".safetensors":
        raise ValueError(f"ReActor loads face models by the .safetensors suffix; got {where.name!r}")
    where.write_bytes(reactor_face_model_bytes(native_of(conn, face_id)))
