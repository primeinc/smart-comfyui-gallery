"""What a producer handed over comes back as that shape, or not at all.

Five properties an envelope storing "whatever it actually is" has to hold,
each attacked with the live object as its own control:

    aliasing    two names for one array stay two names for one array
    cycles      a record holding itself is refused by path, not by a
                RecursionError hundreds of frames down
    containers  a container subclass is rebuilt or refused, never widened
                into the builtin it derives from
    protocol    a consumer written against the producer's record type gets
                the same answers from the stored one, absent keys included
    layout      a non-contiguous array keeps its values while its strides
                normalize, which is what the module says happens

Every control here is the LIVE object: each test states the property of the
thing handed in, then demands it of the thing handed back. Nothing reads the
envelope's own header or node tables, so a wire format that agrees with
itself while disagreeing with the producer cannot pass.
"""

from typing import override

import numpy as np
import pytest
from numpy.linalg import norm as l2norm

from vision import facestore

#: The container name upstream's record carries, which is what the capture
#: sites in `vision/faces.py` and `compat/corpus/cache.py` compute and store.
UPSTREAM_FACE = "insightface.app.common.Face"


class UpstreamFace(dict):
    """insightface's own `Face`, transcribed rather than imported.

    deepinsight/insightface@7fadd420c2351d0ffa8cac403421c1a3ed733365
    python-package/insightface/app/common.py:4-48. Importing it would pull
    onnxruntime into the fast lane and tie this suite to the compat
    dependency group, and a transcription is the stronger control anyway:
    the envelope's rebuilt stand-in is measured against upstream's behaviour
    rather than against a copy of itself.
    """

    def __init__(self, d=None, **kwargs):
        super().__init__()
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)

    @override
    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property
    def sex(self):
        if self.gender is None:
            return None
        return "M" if self.gender == 1 else "F"


class Rung(tuple):
    """A tuple subclass, the shape a composite detector's ladder rung has."""

    __slots__ = ()


class Ledger(dict):
    """A mapping subclass with no protocol of its own, only its identity."""


def _dotted(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


facestore.register_container(_dotted(Rung), Rung)
facestore.register_container(_dotted(Ledger), Ledger)


def carried(value, container="dict"):
    """One round trip through the envelope, returning the root as thawed."""
    blob = facestore.freeze(value, producer="probe", producer_version="v1", container=container)
    return facestore.thaw(blob).value


def reads(face):
    """Consumer code against insightface's record, run unchanged on both.

    Attribute reads, a computed property, an absent key, and a subscript --
    the four ways a consumer touches a `Face`, none of which a plain mapping
    answers the same way.
    """
    return (
        int(face.age),
        face.sex,
        face.nose_tip,
        float(np.asarray(face.embedding).sum()),
        float(face.embedding_norm),
        face["bbox"].tolist(),
        sorted(face),
    )


def a_face():
    return UpstreamFace(
        bbox=np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
        embedding=np.linspace(-1.0, 1.0, 8, dtype=np.float32),
        gender=np.int64(1),
        age=np.int64(27),
    )


def test_one_array_under_two_keys_comes_back_as_one_array():
    """Aliasing is a fact about the producer's output, not a detail of how
    it was written down. A record whose consumer mutates one name and reads
    the other is a different record once the envelope has split them."""
    shared = np.arange(4, dtype=np.float32)
    held = {"left": shared, "right": shared, "nested": {"again": shared}}
    assert held["left"] is held["right"]
    assert held["nested"]["again"] is held["left"]

    back = carried(held)

    assert back["left"] is back["right"], "two keys on one array came back holding two arrays"
    assert back["nested"]["again"] is back["left"], "aliasing survived one level and not the next"
    back["left"][0] = np.float32(9)
    assert back["right"][0] == np.float32(9), "a write through one name is invisible through the other"


def test_one_mapping_under_two_names_stays_one_mapping():
    inner = {"count": 1}
    held = {"direct": inner, "listed": [inner]}
    assert held["direct"] is held["listed"][0]

    back = carried(held)

    assert back["direct"] is back["listed"][0], "one nested mapping came back as two"
    back["direct"]["count"] = 2
    assert back["listed"][0]["count"] == 2


def test_a_record_that_holds_itself_is_refused_by_path():
    """`Unpreservable` names where the loop closed. A RecursionError names
    the envelope's own frames, which is a report about this module rather
    than about the producer value that cannot be stored."""
    held: dict = {"outer": {}}
    held["outer"]["back"] = held

    with pytest.raises(facestore.Unpreservable) as refused:
        facestore.freeze(held, producer="probe", producer_version="v1", container="dict")

    assert "result['outer']['back']" in str(refused.value)


def test_a_list_that_holds_itself_is_refused_by_path():
    held: list = []
    held.append(held)

    with pytest.raises(facestore.Unpreservable) as refused:
        facestore.freeze({"loop": held}, producer="probe", producer_version="v1", container="dict")

    assert "result['loop'][0]" in str(refused.value)


def test_a_nested_container_subclass_is_rebuilt_rather_than_widened():
    held = {"rung": Rung((448, 0.91)), "book": Ledger(hits=1)}
    assert type(held["rung"]) is Rung
    assert type(held["book"]) is Ledger

    back = carried(held)

    assert type(back["rung"]) is Rung, "a tuple subclass came back as a plain tuple"
    assert type(back["book"]) is Ledger, "a dict subclass came back as a plain dict"
    assert tuple(back["rung"]) == (448, 0.91)
    assert dict(back["book"]) == {"hits": 1}


def test_a_container_subclass_with_no_adapter_refuses_at_capture():
    """Refusal is the other allowed answer, and it lands at the pass that
    can still re-run."""

    class Unregistered(dict):
        pass

    with pytest.raises(facestore.Unpreservable, match=r"result\['odd'\].*Unregistered"):
        facestore.freeze({"odd": Unregistered(a=1)}, producer="probe", producer_version="v1", container="dict")


def test_a_declared_container_nothing_can_rebuild_refuses_at_capture():
    with pytest.raises(facestore.Unpreservable, match=r"nowhere\.Invented"):
        facestore.freeze({"a": 1}, producer="probe", producer_version="v1", container="nowhere.Invented")


def test_a_consumer_reads_the_stored_record_exactly_as_it_read_the_live_one():
    """The capture sites hand `freeze` a plain mapping and declare the
    producer's own type beside it. A replay that returns the mapping serves
    a consumer worse than the producer did, which is the narrowing this
    field exists to prevent."""
    live = a_face()
    stored = carried({str(key): live[key] for key in live}, UPSTREAM_FACE)

    assert reads(stored) == reads(live)


def test_a_key_the_record_never_carried_reads_as_none_from_the_stored_face():
    """Upstream answers every absent attribute with None. A mapping raises
    AttributeError, so a consumer's `if face.mask is None` becomes a crash
    against the store and a branch against the producer."""
    live = a_face()
    stored = carried({str(key): live[key] for key in live}, UPSTREAM_FACE)

    assert live.nose_tip is None
    assert stored.nose_tip is None
    assert isinstance(stored, dict), "the rebuilt container stopped being a mapping"


def test_the_rebuilt_face_is_named_by_the_record_it_came_from():
    live = a_face()
    thawed = facestore.thaw(
        facestore.freeze(
            {str(key): live[key] for key in live},
            producer="insightface/antelopev2",
            producer_version="v1",
            container=UPSTREAM_FACE,
        )
    )

    assert thawed.container == UPSTREAM_FACE
    assert thawed.record["age"] == live["age"]


def test_a_non_contiguous_array_keeps_its_values_while_its_strides_normalize():
    """A strided view's values are the producer's output; its strides are an
    artifact of how the producer sliced. The envelope keeps the first and
    normalizes the second, and this holds it to exactly that."""
    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    view = base[::2, ::3]
    assert not view.flags["C_CONTIGUOUS"]
    assert not view.flags["F_CONTIGUOUS"]

    back = carried({"view": view})["view"]

    assert back.tobytes() == view.tobytes(), "a strided source lost values"
    assert back.dtype == view.dtype
    assert back.shape == view.shape
    assert back.flags["C_CONTIGUOUS"], "the documented C-contiguous rebuild stopped happening"


def test_a_fortran_ordered_array_comes_back_fortran_ordered():
    held = np.asfortranarray(np.arange(6, dtype=np.float64).reshape(2, 3))
    assert held.flags["F_CONTIGUOUS"]
    assert not held.flags["C_CONTIGUOUS"]

    back = carried({"f": held})["f"]

    assert back.flags["F_CONTIGUOUS"], "the one layout fact the record keeps was dropped"
    assert np.array_equal(back, held)
    assert back.dtype == held.dtype


def test_an_empty_array_survives_with_its_dtype_and_shape():
    """A detector that found nothing emits a zero-length array, which is a
    measurement and not an absence."""
    held = np.zeros((0, 3), dtype=np.float32)

    back = carried({"landmarks": held})["landmarks"]

    assert back.shape == (0, 3)
    assert back.dtype == np.float32
