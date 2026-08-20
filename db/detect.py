"""Faces out of a file, through one door.

Every detector sees the picture through `oriented.for_model` here, and only
here. Measured on a 105-photo set, 14 photos were stored sideways, and a
detector shown a sideways face finds wallpaper instead -- which then became
its own "person". Forcing the turn plus a 0.7 score floor took that set's
pair-F1 from 0.946 to 1.000. Leaving orientation to each caller means the
next caller forgets, so the rule is structural: model-facing code does not
open image files, it calls this.

The backend contract is `detect(PIL.Image) -> list[FaceDetection]` with
normalized boxes (vision/faces.py:118-139): `bbox` (x, y, w, h) in
0..1, `det_score` in 0..1, `embedding` float32 or None, `landmarks` a list
of (x, y) pairs, `attributes` an optional dict with `age` / `sex`.
"""

from __future__ import annotations

from . import derived, oriented

#: Detections below this confidence are recorded nowhere. YuNet reports down
#: to its own floor of 0.5, and on the 105-photo labelled set the band
#: between 0.5 and 0.7 contained only non-faces -- wallpaper and hair --
#: while every real face cleared 0.7.
FLOOR = 0.7


def harvest(
    conn, backend, file_id: int, path, now: float, *, floor: float = FLOOR, sample_id=None, image=None
) -> list[int]:
    """Detect and record every face in one file, replacing earlier answers.

    `image` is for callers that already hold decoded pixels -- a video
    sampler with a frame in hand. It must already be upright; a file path is
    the normal case and is turned here.

    Faces without an embedding are dropped: a face that cannot be compared
    cannot be clustered, asserted against, or shown as a person, and rows
    with NULL vectors made every downstream count lie about what the
    clusterer had to work with.
    """
    if image is None:
        image = oriented.for_model(conn, file_id, path)
    sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()[0]

    faces = []
    for found in backend.detect(image):
        if found.embedding is None or float(found.det_score) < floor:
            continue
        record = {
            "region": derived.region(conn, *found.bbox),
            "det_score": float(found.det_score),
            "embedding": found.embedding.astype("float32").tobytes(),
        }
        if found.landmarks:
            import numpy as np

            record["landmarks"] = np.asarray(found.landmarks, dtype=np.float32).tobytes()
        traits = found.attributes or {}
        if "age" in traits:
            record["age"] = int(traits["age"])
        if "sex" in traits:
            record["sex"] = str(traits["sex"])
        faces.append(record)

    return derived.record_faces(
        conn,
        file_id,
        backend.model_id,
        backend.model_version,
        sha,
        now,
        faces,
        sample_id=sample_id,
    )


def harvest_video(conn, backend, file_id: int, path, now: float, *, floor: float = FLOOR) -> dict[str, int]:
    """Faces across one video: choose the moments, look at each, record.

    The moments come from `db.sample` -- persisted `derived_media_sample`
    rows, so every face found here says which moment it was looking at and
    a re-run recognises its own work. Frames arrive upright through the
    decoder door, display matrix applied, for the same reason stills go
    through `oriented`.
    """
    from vision import decode

    from . import sample

    sample.frames(conn, file_id, path)
    chosen = sample.taken(conn, file_id)
    by_offset = {offset: sample_id for sample_id, offset, _ in chosen}
    found = 0
    for offset_ms, image in decode.frames_at(path, sorted(by_offset)):
        written = harvest(
            conn,
            backend,
            file_id,
            path,
            now,
            floor=floor,
            sample_id=by_offset[offset_ms],
            image=image,
        )
        found += len(written)
    return {"moments": len(chosen), "faces": found}


def path_of(conn, file_id: int) -> str:
    """Where this file's bytes are, composed from the folder tree.

    The root folder row IS the root path -- `observe_tree` creates it from
    the root's basename and walks down from there (db/scan.py:383-386) --
    so the on-disk path is the root's recorded path plus every folder name
    strictly below the root folder, plus the file's name. Composed by
    parent, never by splitting stored text: there is no stored text.
    """
    import os

    chain = conn.execute(
        """
        WITH RECURSIVE up(id, parent_id, name, root_id, lvl) AS (
          SELECT fo.id, fo.parent_id, fo.name, fo.root_id, 0
            FROM file f JOIN folder fo ON fo.id = f.folder_id WHERE f.id = ?
          UNION ALL
          SELECT fo.id, fo.parent_id, fo.name, fo.root_id, up.lvl + 1
            FROM folder fo JOIN up ON fo.id = up.parent_id
        )
        SELECT parent_id, name, root_id FROM up ORDER BY lvl DESC
        """,
        (file_id,),
    ).fetchall()
    if not chain:
        raise ValueError(f"file {file_id} is not in any folder the library knows")
    base = conn.execute("SELECT path FROM root WHERE id = ?", (chain[0][2],)).fetchone()[0]
    name = conn.execute("SELECT name FROM file WHERE id = ?", (file_id,)).fetchone()[0]
    below = [row[1] for row in chain if row[0] is not None]
    return os.path.join(base, *below, name)


def harvest_all(conn, backend, now: float, *, floor: float = FLOOR) -> dict[str, int]:
    """Every present image file the library knows, at its real path.

    The loop a job runs. Returns counts rather than printing, because the
    caller owns the reporting surface. Files marked missing are skipped:
    their bytes are not there to read, and their last detections stand.
    """
    files = with_faces = found = 0
    for (file_id,) in conn.execute(
        "SELECT id FROM file WHERE kind = 'image' AND missing_since IS NULL ORDER BY id"
    ).fetchall():
        files += 1
        got = harvest(conn, backend, file_id, path_of(conn, file_id), now, floor=floor)
        found += len(got)
        with_faces += 1 if got else 0
    return {"files": files, "with_faces": with_faces, "faces": found}
