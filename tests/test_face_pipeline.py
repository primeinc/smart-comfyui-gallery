"""The face pipeline from bytes to people, checked at its joints.

Each test is named for the failure it prevents. The pipeline under test is
the production one -- `db.detect` through `db.similarity` and `db.grouping`
into `db.derived` -- not a re-implementation of it, because the last suite
formed clusters by writing the answer down and then checking the answer.
"""

from __future__ import annotations

import io
import pathlib
import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pytest
from PIL import Image

from db import derived, detect, grouping, oriented, scan, similarity

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"


@pytest.fixture(scope="module")
def ddl():
    return SCHEMA.read_text(encoding="utf-8")


@pytest.fixture
def db(ddl):
    conn = sqlite3.connect(":memory:")
    conn.executescript(ddl)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


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
    """The shape vision.faces.FaceDetection presents (vision/faces.py:118-139)."""

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
    turned = oriented.upright(Image.open(buffer))
    assert turned.size == frame.size
    assert turned.getpixel((0, 0)) == (255, 0, 0)


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


def test_no_pixel_reader_in_db_opens_files_behind_oriented():
    """The rule is structural: model-facing code cannot forget the turn if it
    cannot open pixels. `capture` reads only metadata and never pixels."""
    allowed = {"oriented.py", "capture.py"}
    package = pathlib.Path(__file__).resolve().parent.parent / "db"
    offenders = [
        source.name
        for source in sorted(package.glob("*.py"))
        if source.name not in allowed and "Image.open(" in source.read_text(encoding="utf-8")
    ]
    assert offenders == []


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


def edges(graph) -> set:
    indptr, cols, _ = graph
    return {
        (int(row), int(cols[edge]))
        for row in range(len(indptr) - 1)
        for edge in range(int(indptr[row]), int(indptr[row + 1]))
    }


@pytest.mark.parametrize("backend", ["faiss-cpu", "faiss-gpu"])
def test_faiss_returns_the_same_edges_as_the_exact_numpy_path(backend):
    vectors = planted()
    baseline, ran = similarity.graph(vectors, 0.5, backend="numpy")
    assert ran == "numpy"
    try:
        contender, ran = similarity.graph(vectors, 0.5, backend=backend)
    except similarity.FALLIBLE as why:
        pytest.skip(f"{backend} unavailable here: {why}")
    assert ran == backend
    assert edges(contender) == edges(baseline)


@pytest.mark.parametrize("backend", ["numpy", "faiss-cpu"])
def test_a_pair_sitting_exactly_on_the_threshold_is_kept(backend):
    """FAISS range_search keeps strictly-above for inner product; the numpy
    path keeps at-or-above. The nextafter step makes them agree, and a pair
    at similarity exactly 1.0 against threshold 1.0 is the sharpest case."""
    vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    try:
        graph, _ = similarity.graph(vectors, 1.0, backend=backend)
    except similarity.FALLIBLE as why:
        pytest.skip(f"{backend} unavailable here: {why}")
    assert edges(graph) == {(0, 1), (1, 0)}


def test_a_named_backend_that_does_not_exist_raises_instead_of_substituting():
    with pytest.raises(ValueError, match="cuda-magic"):
        similarity.graph(planted(), 0.5, backend="cuda-magic")


# --- grouping --------------------------------------------------------------


def as_groups(labels) -> set:
    groups: dict[int, list[int]] = {}
    for node, label in enumerate(labels):
        groups.setdefault(label, []).append(node)
    return {tuple(sorted(members)) for members in groups.values()}


PLANTED_ANSWER = {tuple(range(6)), tuple(range(6, 12)), tuple(range(12, 18))}


def test_grouping_is_a_pure_function_of_the_graph():
    vectors = planted()
    graph, _ = similarity.graph(vectors, 0.5, backend="numpy")
    first = grouping.group(graph, vectors, "chinese-whispers")
    second = grouping.group(graph, vectors, "chinese-whispers")
    assert first == second
    assert as_groups(first) == PLANTED_ANSWER


def test_connected_components_find_the_planted_groups_at_a_tight_threshold():
    vectors = planted()
    graph, _ = similarity.graph(vectors, 0.5, backend="numpy")
    assert as_groups(grouping.connected_components(graph)) == PLANTED_ANSWER


def test_spherical_kmeans_finds_the_planted_groups():
    vectors = planted()
    graph, _ = similarity.graph(vectors, 0.5, backend="numpy")
    try:
        labels = grouping.group(graph, vectors, "spherical-kmeans", gpu=False)
    except similarity.FALLIBLE as why:
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
