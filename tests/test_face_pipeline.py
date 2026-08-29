"""The face pipeline from bytes to people, checked at its joints.

Each test is named for the failure it prevents. The pipeline under test is
the production one -- `db.detect` through `db.similarity` and `db.grouping`
into `db.derived` -- not a re-implementation of it, because the last suite
formed clusters by writing the answer down and then checking the answer.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pytest
from PIL import Image

from db import derived, detect, grouping, oriented, scan, similarity
from tests.staging import fresh_schema
from vision import decode


@pytest.fixture
def db():
    """The schema, from the per-process master rather than the DDL.

    A backup from a master built once per process is far cheaper than
    `executescript` of the whole schema (tests/staging.py `fresh_schema`).
    Every test here starts from exactly this.
    """
    return fresh_schema()


def library(conn, files: int, root_path: str = "C:/lib") -> list[int]:
    conn.execute(
        "INSERT INTO root(id,path,kind,created_at) VALUES(1,?, 'library',0)",
        (root_path,),
    )
    folder = scan.mint(conn, "folder", "lib")
    conn.execute(
        "INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(?,1,NULL,'lib',0)",
        (folder,),
    )
    made = []
    for n in range(files):
        file_id = scan.mint(conn, "file", f"p{n}")
        conn.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,content_sha256,"
            "first_seen_at,last_seen_at) VALUES(?,?,?,'image',1,0,'aa',0,0)",
            (file_id, folder, f"p{n}.png"),
        )
        made.append(file_id)
    return made


@dataclass
class Sighting:
    """The shape vision.faces.FaceDetection presents (vision/faces.py)."""

    bbox: tuple
    det_score: float
    embedding: np.ndarray | None
    landmarks: list = field(default_factory=list)
    attributes: dict | None = None


class Peeker:
    """A backend that reports a face only when shown the upright frame."""

    model_id = "fake/peek"
    model_version = "1"

    def __init__(self):
        self.saw = []

    def detect(self, img):
        self.saw.append((img.size, img.getpixel((0, 0))))
        if img.size == (40, 20) and img.getpixel((0, 0)) == (255, 0, 0):
            return [Sighting((0.1, 0.1, 0.2, 0.2), 0.9, np.ones(8, dtype=np.float32))]
        return []


def upright_frame() -> Image.Image:
    image = Image.new("RGB", (40, 20), (0, 0, 255))
    image.putpixel((0, 0), (255, 0, 0))
    return image


# --- orientation is forced, not remembered ---------------------------------


def test_a_sideways_photo_is_turned_before_the_model_sees_it(db, tmp_path):
    """EXIF orientation 6 stores the frame a quarter turn off. The detector
    must be handed the upright picture without the caller doing anything."""
    (file_id,) = library(db, 1)
    stored = upright_frame().transpose(Image.Transpose.ROTATE_90)
    path = tmp_path / "sideways.png"
    stored.save(path)
    db.execute(
        "INSERT INTO capture(file_id, orientation, parsed_at) VALUES(?, 6, 0)",
        (file_id,),
    )
    backend = Peeker()
    written = detect.harvest(db, backend, file_id, path, 0.0)
    assert backend.saw == [((40, 20), (255, 0, 0))]
    assert len(written) == 1


def test_without_the_recorded_tag_the_same_bytes_stay_sideways(db, tmp_path):
    """The positive control's twin: identical bytes, no capture row, and the
    backend sees the sideways frame. Together they prove the turn came from
    the stored tag rather than from anything about the file."""
    (file_id,) = library(db, 1)
    stored = upright_frame().transpose(Image.Transpose.ROTATE_90)
    path = tmp_path / "sideways.png"
    stored.save(path)
    backend = Peeker()
    written = detect.harvest(db, backend, file_id, path, 0.0)
    assert backend.saw == [((20, 40), (0, 0, 255))]
    assert written == []


def test_upright_reads_the_exif_tag_out_of_the_file_itself():
    """A caller with no database row still gets the turned picture: the tag
    travels in the file, and PNG carries it in an eXIf chunk."""
    frame = upright_frame()
    stored = frame.transpose(Image.Transpose.ROTATE_180)
    exif = Image.Exif()
    exif[274] = 3
    buffer = io.BytesIO()
    stored.save(buffer, format="PNG", exif=exif)
    buffer.seek(0)
    turned = oriented.upright(decode.open_bytes(buffer.getvalue()))
    assert turned.size == frame.size
    assert turned.getpixel((0, 0)) == (255, 0, 0)


def test_open_upright_leaves_no_handle_behind(tmp_path):
    """A still closes on load (Pillow file-handling); an animated file does
    not -- open_upright takes its first frame and closes the handle, so no
    ResourceWarning reaches the garbage collector for either."""
    import gc
    import warnings

    still = tmp_path / "still.png"
    upright_frame().save(still)
    moving = tmp_path / "moving.gif"
    frames = [Image.new("RGB", (8, 8), (255, 0, 0)), Image.new("RGB", (8, 8), (0, 255, 0))]
    frames[0].save(moving, save_all=True, append_images=frames[1:], duration=100, loop=0)
    layered = tmp_path / "flat.psd"  # a plugin that keeps its handle after load even with one frame
    from psd_tools import PSDImage

    PSDImage.frompil(Image.new("RGB", (8, 8), (0, 0, 255))).save(str(layered))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        for path, size in ((still, (40, 20)), (moving, (8, 8)), (layered, (8, 8))):
            turned = oriented.open_upright(path, 3)
            assert turned.size == size
            del turned
            gc.collect()
    leaked = [w for w in caught if issubclass(w.category, ResourceWarning) and "unclosed file" in str(w.message)]
    assert leaked == [], [str(w.message) for w in leaked]


def test_every_stored_orientation_comes_back_upright():
    """All seven non-trivial EXIF orientations, each stored by the inverse
    transform and recovered by the mapping Pillow itself uses
    (python-pillow/Pillow@bb1d8e8 src/PIL/ImageOps.py:705-713)."""
    undo = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_90,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_270,
    }
    frame = upright_frame()
    for orientation, inverse in undo.items():
        stored = frame.transpose(inverse)
        recovered = oriented.upright(stored, orientation)
        assert recovered.tobytes() == frame.tobytes(), orientation


def test_faces_below_the_floor_are_recorded_nowhere(db, tmp_path):
    (file_id,) = library(db, 1)
    path = tmp_path / "plain.png"
    upright_frame().save(path)

    class Doubter:
        model_id = "fake/doubt"
        model_version = "1"

        def detect(self, img):
            return [
                Sighting((0.1, 0.1, 0.2, 0.2), 0.69, np.ones(8, dtype=np.float32)),
                Sighting((0.5, 0.5, 0.2, 0.2), 0.71, np.ones(8, dtype=np.float32)),
            ]

    written = detect.harvest(db, Doubter(), file_id, path, 0.0)
    assert len(written) == 1
    kept = db.execute("SELECT det_score FROM derived_face_instance").fetchall()
    assert kept == [(0.71,)]


def test_path_of_composes_from_the_folder_tree(db):
    """A file two folders down resolves to root path + every folder below the
    root folder + its own name. The root folder row IS the root path."""
    library(db, 0, root_path="C:/pictures")
    inner = scan.mint(db, "folder", "a")
    db.execute(
        "INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(?,1,1,'a',1)",
        (inner,),
    )
    deepest = scan.mint(db, "folder", "b")
    db.execute(
        "INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(?,1,?,'b',2)",
        (deepest, inner),
    )
    file_id = scan.mint(db, "file", "x")
    db.execute(
        "INSERT INTO file(id,folder_id,name,kind,size,mtime,content_sha256,"
        "first_seen_at,last_seen_at) VALUES(?,?,?,'image',1,0,'aa',0,0)",
        (file_id, deepest, "x.png"),
    )
    assert detect.path_of(db, file_id).replace("\\", "/") == "C:/pictures/a/b/x.png"


# --- the neighbour graph agrees with itself across backends ----------------


def planted(groups: int = 3, members: int = 6, dim: int = 32) -> np.ndarray:
    """Tight groups around orthogonal directions, deterministic."""
    rng = np.random.default_rng(7)
    centres = np.linalg.qr(rng.standard_normal((dim, groups)))[0].T
    rows = [centre + 0.03 * rng.standard_normal(dim) for centre in centres for _ in range(members)]
    return np.asarray(rows, dtype=np.float32)


# The engine itself -- edges vs the exact oracle, device-policy parity,
# inclusive thresholds -- is proven in tests/test_faiss_index.py. These
# tests consume the oracle directly: grouping is a pure function of a
# graph, and the oracle is the independent way to make one.


# --- grouping --------------------------------------------------------------


def as_groups(labels) -> set:
    groups: dict[int, list[int]] = {}
    for node, label in enumerate(labels):
        groups.setdefault(label, []).append(node)
    return {tuple(sorted(members)) for members in groups.values()}


PLANTED_ANSWER = {tuple(range(6)), tuple(range(6, 12)), tuple(range(12, 18))}


def test_grouping_is_a_pure_function_of_the_graph():
    vectors = planted()
    graph = similarity.numpy_graph(vectors, 0.5)
    first = grouping.group(graph, vectors, "chinese-whispers")
    second = grouping.group(graph, vectors, "chinese-whispers")
    assert first == second
    assert as_groups(first) == PLANTED_ANSWER


def test_connected_components_find_the_planted_groups_at_a_tight_threshold():
    vectors = planted()
    graph = similarity.numpy_graph(vectors, 0.5)
    assert as_groups(grouping.connected_components(graph)) == PLANTED_ANSWER


def test_spherical_kmeans_finds_the_planted_groups():
    vectors = planted()
    graph = similarity.numpy_graph(vectors, 0.5)
    from vision.faiss_index import FALLIBLE

    try:
        labels = grouping.group(graph, vectors, "spherical-kmeans", gpu=False)
    except FALLIBLE as why:
        pytest.skip(f"faiss unavailable here: {why}")
    assert as_groups(labels) == PLANTED_ANSWER


def test_an_unknown_method_is_refused_by_name():
    empty = (np.zeros(1, "int64"), np.zeros(0, "int64"), np.zeros(0, "float32"))
    with pytest.raises(ValueError, match="k-medoids"):
        grouping.group(empty, None, "k-medoids")


# --- judging a run ---------------------------------------------------------


def clustered(conn, vectors, labels, *, model="fake/peek", version="1", threshold=0.5, method="given") -> int:
    """Faces, clusters and memberships written through the public API."""
    files = library(conn, len(vectors))
    return regrouped(
        conn,
        record(conn, files, vectors, model, version),
        labels,
        model=model,
        version=version,
        threshold=threshold,
        method=method,
    )


def record(conn, files, vectors, model, version) -> list[int]:
    face_ids = []
    for file_id, vector in zip(files, vectors, strict=True):
        (face_id,) = derived.record_faces(
            conn,
            file_id,
            model,
            version,
            "aa",
            0.0,
            [
                {
                    "region": derived.region(conn, 0.1, 0.1, 0.2, 0.2),
                    "embedding": np.asarray(vector, dtype=np.float32).tobytes(),
                }
            ],
        )
        face_ids.append(face_id)
    return face_ids


def regrouped(conn, face_ids, labels, *, model="fake/peek", version="1", threshold=0.5, method="given") -> int:
    """One clustering run stated outright, through the public API."""
    distinct = sorted(set(labels))
    made = derived.recluster(
        conn,
        model,
        version,
        0.0,
        [{} for _ in distinct],
        method=method,
        threshold=threshold,
    )
    cluster_of = dict(zip(distinct, made, strict=True))
    for face_id, label in zip(face_ids, labels, strict=True):
        derived.assign_cluster(conn, face_id, cluster_of[label])
    return derived.run_for(conn, model, version, method, threshold, 0.0)


def faces_on_file(conn) -> list[int]:
    return [row[0] for row in conn.execute("SELECT id FROM derived_face_instance ORDER BY id")]


def rousseeuw(unit, labels) -> float:
    """The definition, written independently of the implementation
    (scikit-learn/scikit-learn@bb9d35b sklearn/metrics/cluster/
    _unsupervised.py:211-230): a = mean distance to the rest of the own
    cluster, b = smallest mean distance to another whole cluster."""
    distance = 1.0 - unit @ unit.T
    scores = []
    for i, mine in enumerate(labels):
        same = [j for j, label in enumerate(labels) if label == mine and j != i]
        if not same:
            scores.append(0.0)
            continue
        a = distance[i, same].mean()
        b = min(
            distance[i, [j for j, label in enumerate(labels) if label == other]].mean()
            for other in set(labels) - {mine}
        )
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))


def test_the_silhouette_is_rousseeuws_number_not_a_centroid_shortcut(db):
    """Clusters of unequal spread are where mean-distance-to-members and
    distance-to-centroid disagree, and only the former is the silhouette."""
    rng = np.random.default_rng(3)
    vectors, labels = [], []
    for group, spread in enumerate([0.02, 0.25, 0.05]):
        centre = np.zeros(16)
        centre[group * 5] = 1.0
        for _ in range(8):
            vectors.append(centre + spread * rng.standard_normal(16))
            labels.append(group)
    vectors = np.asarray(vectors, dtype=np.float32)
    run_id = clustered(db, vectors, labels)
    reading = derived.health(db, run_id)
    expected = rousseeuw(similarity.normalise(vectors), labels)
    assert reading["silhouette"] == pytest.approx(expected, abs=1e-5)


def test_a_chained_run_is_disqualified_by_shape_alone(db):
    """Half the library in one cluster is chaining, whatever the scores say."""
    vectors = planted(groups=2, members=10)
    run_id = clustered(db, vectors, [0] * 12 + [1] * 8)
    reading = derived.health(db, run_id)
    assert reading["largest_share"] > derived.CHAINED


def name_people(conn, count: int) -> list[int]:
    people = []
    for n in range(count):
        person = scan.mint(conn, "person", f"person-{n}")
        conn.execute(
            "INSERT INTO person(id,name,created_at) VALUES(?,?,0)",
            (person, f"Person {n}"),
        )
        people.append(person)
    return people


def test_agreement_counts_held_split_and_mixed(db):
    """A run that keeps two asserted people apart scores clean; a second run
    that welds them shows exactly one cluster mixing people."""
    vectors = planted(groups=2, members=4, dim=8)
    run_id = clustered(db, vectors, [0] * 4 + [1] * 4)
    alice, bob = name_people(db, 2)
    files = [row[0] for row in db.execute("SELECT id FROM file ORDER BY id")]
    for person, half in ((alice, files[:4]), (bob, files[4:])):
        for file_id in half:
            db.execute(
                "INSERT INTO person_assertion(person_id,file_id,created_at) VALUES(?,?,0)",
                (person, file_id),
            )
    assert derived.agreement(db, run_id) == {
        "asserted_people": 2,
        "held_together": 2,
        "split_apart": 0,
        "clusters_mixing_people": 0,
    }

    welded = regrouped(db, faces_on_file(db), [0] * 8, threshold=0.3)
    assert derived.agreement(db, welded)["clusters_mixing_people"] == 1


def test_choose_primary_prefers_the_run_people_agree_with(db):
    """Five asserted people; run A keeps them apart, run B welds two. B sits
    at the embedder's measured threshold, which the no-assertions rule would
    prefer -- the assertions must outrank it."""
    vectors = planted(groups=5, members=6, dim=32)
    truth = [n // 6 for n in range(30)]
    derived.SAME_PERSON.setdefault("fake/peek", 0.55)
    try:
        a_run = clustered(db, vectors, truth, threshold=0.40)
        welded = [0 if label == 1 else label for label in truth]
        b_run = regrouped(db, faces_on_file(db), welded, threshold=0.55)

        for run_id in (a_run, b_run):
            reading = derived.health(db, run_id)
            assert reading["largest_share"] <= derived.CHAINED
            assert reading["silhouette"] >= derived.GOOD_ENOUGH

        people = name_people(db, 5)
        files = [row[0] for row in db.execute("SELECT id FROM file ORDER BY id")]
        for n, file_id in enumerate(files):
            db.execute(
                "INSERT INTO person_assertion(person_id,file_id,created_at) VALUES(?,?,0)",
                (people[n // 6], file_id),
            )
        assert derived.choose_primary(db) == a_run
        assert derived.primary_run(db) == a_run
    finally:
        derived.SAME_PERSON.pop("fake/peek", None)


def test_choose_primary_without_assertions_takes_the_measured_threshold(db):
    vectors = planted(groups=3, members=6)
    truth = [n // 6 for n in range(18)]
    derived.SAME_PERSON.setdefault("fake/peek", 0.55)
    try:
        clustered(db, vectors, truth, threshold=0.40)
        b_run = regrouped(db, faces_on_file(db), truth, threshold=0.55)
        assert derived.choose_primary(db) == b_run
    finally:
        derived.SAME_PERSON.pop("fake/peek", None)


def test_a_deleted_face_id_is_never_handed_out_again(db):
    """derived_face_instance ids feed the resident face index, and index
    alignment treats an id as a stable identity. Plain rowid tables
    reuse the largest free id, so a re-detect could mint a new face
    wearing a deleted face's id -- same id, different embedding, and an
    aligned index keeps serving the old vector. AUTOINCREMENT is the
    schema-level guarantee this pins."""
    (file_id,) = library(db, 1)
    db.execute("INSERT INTO region(id, x, y, w, h) VALUES(1, 0.1, 0.1, 0.2, 0.2)")
    sid = similarity.space_id(db, similarity.face_space("m", "1", 1), 0.0)

    def face() -> int:
        return db.execute(
            "INSERT INTO derived_face_instance(file_id, region_id, model_id, model_version,"
            " det_score, embedding, dim, space_id, source_sha256, computed_at)"
            " VALUES(?, 1, 'm', '1', 0.9, x'00000000', 1, ?, 'aa', 0)",
            (file_id, sid),
        ).lastrowid

    first = face()
    second = face()
    assert second > first
    db.execute("DELETE FROM derived_face_instance WHERE id = ?", (second,))
    third = face()
    assert third > second, "a deleted face id came back wearing a different embedding"


# --- why a run is or is not the default, in words ------------------------------


def _reading(**facts) -> dict:
    base = {"faces": 100, "clusters": 10, "largest_share": 0.1, "alone_share": 0.2, "silhouette": 0.5}
    return {**base, **facts}


@pytest.mark.parametrize(
    ("reading", "names"),
    [
        (_reading(clusters=0), "grouped nothing"),
        (_reading(largest_share=0.96), "chained"),
        (_reading(alone_share=0.99), "alone"),
        (_reading(silhouette=0.02), "not apart"),
    ],
    ids=["nothing", "chained", "alone", "silhouette"],
)
def test_a_disqualified_run_is_told_why(reading, names):
    why = derived.disqualification(reading)

    assert why is not None
    assert names in why


@pytest.mark.parametrize(
    "reading",
    [_reading(), _reading(faces=3, clusters=1, largest_share=0.9, silhouette=0.0)],
    ids=["sound", "too-small-to-judge"],
)
def test_a_sound_run_has_no_disqualification(reading):
    assert derived.disqualification(reading) is None
