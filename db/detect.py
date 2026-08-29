"""Faces out of a file, through one entry point.

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
of (x, y) pairs, `attributes` an optional dict of whatever else the model
produced -- open, not a fixed set, and recorded whole.

A detection pass is the expensive thing in this application and the bytes
it produces are not: antelopev2 loads a 143 MB session per worker to derive
head pose and two dense landmark sets, and what it emits is kilobytes per
face to keep. Anything dropped here can only be recovered by reading the whole
library again, and only while the originals are still on disk. So the rule
is: whatever a backend emits is persisted, and columns are promotions out
of that record rather than a filter in front of it.
"""

from __future__ import annotations

from . import derived, oriented

#: Detections below this confidence are recorded nowhere. YuNet reports down to
#: its own floor of 0.5, and on the 105-photo labelled set the band between 0.5
#: and 0.7 held only non-faces -- wallpaper and hair -- while real faces cleared.
FLOOR = 0.7


def harvest(
    conn,
    backend,
    file_id: int,
    path,
    now: float,
    *,
    floor: float = FLOOR,
    sample_id=None,
    image=None,
    thumbs_dir=None,
) -> list[int]:
    """Detect and record every face in one file, replacing earlier answers.

    `image` is for callers that already hold decoded pixels -- a video
    sampler with a frame in hand. It must already be upright; a file path is
    the normal case and is turned here.

    `thumbs_dir` asks for the decoded frame to be cached as the file's
    thumbnail on the way past (vision/thumbs.py) -- the pixels are in hand,
    and decoding them again later to make a thumbnail makes no sense.

    Faces without an embedding are dropped: a face that cannot be compared
    cannot be clustered, asserted against, or shown as a person, and rows
    with NULL vectors made every downstream count lie about what the
    clusterer had to work with.
    """
    if image is None:
        image = oriented.for_model(conn, file_id, path)
    sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()[0]
    if sha is None:
        # A file can reach detection before anything hashed it, and every derived
        # row keys staleness on the content hash (`source_sha256 IS NOT
        # content_sha256`), so the sha lands on the file row in this transaction.
        from . import scan

        sha = scan.sha256_of(path)
        conn.execute("UPDATE file SET content_sha256 = ? WHERE id = ?", (sha, file_id))
    if thumbs_dir is not None:
        import pathlib

        from vision import thumbs

        thumbs.put_all(pathlib.Path(thumbs_dir), sha, image)
    # The frame is decoded and upright; hashing it later means decoding again.
    # Whole file only: a video sample's frame is one moment, not the file's
    # identity, and `record_hash` needs the sha the row may not have yet.
    if sample_id is None:
        from vision import dupes

        phash64, dhash64 = dupes.perceptual(image)
        derived.record_hash(conn, file_id, sha, now, phash64=phash64, dhash64=dhash64)

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

            # float64, not float32: the blob holds NORMALIZED coordinates that
            # a reader multiplies by the frame size, and float32 loses up to
            # 2.4e-4 source pixels (compat/consumers/gallery_storage.py).
            record["landmarks"] = np.asarray(found.landmarks, dtype=np.float64).tobytes()
        # Everything the backend said, then promotions out of it. Not an
        # allowlist: what a backend emits and nobody thought to name is
        # recoverable only by reading the whole library again.
        traits = found.attributes or {}
        if traits:
            record["attributes"] = traits
        # Promoted because a facet filters on them and JSON extraction is not an
        # index. `pose` unpacks by key: the source array is [pitch, yaw, roll]
        # and the columns are yaw-first, so a positional copy swaps two of three.
        if "age" in traits:
            record["age"] = int(traits["age"])
        if "sex" in traits:
            record["sex"] = str(traits["sex"])
        pose = traits.get("pose")
        if isinstance(pose, dict):
            record["pose"] = {axis: float(pose[axis]) for axis in ("yaw", "pitch", "roll") if axis in pose}
        faces.append(record)

    written = derived.record_faces(
        conn,
        file_id,
        backend.model_id,
        backend.model_version,
        sha,
        now,
        faces,
        sample_id=sample_id,
    )
    if sample_id is None:
        # a whole still was looked at; a video's frames are summed by harvest_video
        derived.record_face_scan(conn, file_id, backend.model_id, backend.model_version, sha, now, len(written))
    return written


#: How many extra moments a face-free video is granted beyond its cadence.
#: Refinement bisects the widest gaps first, so the budget spreads across the
#: whole video; when it runs out, the cadence's answer stands.
REFINE_MOST = 32


def harvest_video(
    conn, backend, file_id: int, path, now: float, *, floor: float = FLOOR, thumbs_dir=None
) -> dict[str, int]:
    """Faces across one video: choose the moments, look at each, record.

    The moments come from `db.sample` -- persisted `derived_media_sample`
    rows, so every face found here says which moment it was looking at and
    a re-run recognises its own work. Frames arrive upright through the
    decoder, display matrix applied, for the same reason stills go
    through `oriented`.

    **A face-free cadence is a hint, not a verdict.** A fixed interval can
    land every moment on the establishing shot of a clip that is otherwise
    all people, so when the cadence finds nothing, `sample.refine` bisects
    the gaps between looked-at moments -- widest first, up to REFINE_MOST
    extra frames -- and stops the moment faces appear. The face-free rows
    are kept: where nothing was found is evidence, and a re-run resumes
    from it instead of looking again.

    With `thumbs_dir`, the video's thumbnail is written from the sampled
    frame where the most faces were found -- a clip of somebody should be
    represented by them, not by whatever the first moment happened to show.
    Ties and face-free videos fall back to the earliest frame. The choice
    is deterministic: persisted moments, deterministic detection.
    """
    from vision import decode

    from . import sample

    found = 0
    poster = None  # (faces found, image) -- strict > keeps the earliest on ties

    def look(by_offset: dict[int, int]) -> None:
        nonlocal found, poster
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
            if poster is None or len(written) > poster[0]:
                poster = (len(written), image)

    sample.frames(conn, file_id, path)
    chosen = sample.taken(conn, file_id)
    look({offset: sample_id for sample_id, offset, _ in chosen})
    looked = sorted(offset for _, offset, _ in chosen)

    # The budget counts every bisect row this file has been granted, not this
    # run's: otherwise each re-run of a face-free video deepens the refinement
    # by another REFINE_MOST and converges on decoding the whole file.
    budget = REFINE_MOST - sum(1 for _, _, policy in chosen if policy == "bisect")
    while found == 0 and budget > 0 and looked:
        fresh = sample.refine(conn, file_id, path, looked, budget=budget)
        if not fresh:
            break
        look({offset: sample_id for sample_id, offset in fresh})
        budget -= len(fresh)
        looked = sorted(set(looked) | {offset for _, offset in fresh})

    sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()[0]
    if sha is None:
        # no frame reached harvest (nothing decodable): hash here so the
        # pass is still recorded against the bytes it looked at
        from . import scan

        sha = scan.sha256_of(path)
        conn.execute("UPDATE file SET content_sha256 = ? WHERE id = ?", (sha, file_id))
    derived.record_face_scan(conn, file_id, backend.model_id, backend.model_version, sha, now, found)
    if thumbs_dir is not None and poster is not None:
        import pathlib

        from vision import thumbs

        thumbs.put_all(pathlib.Path(thumbs_dir), sha, poster[1])
    return {"moments": len(looked), "faces": found}


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
