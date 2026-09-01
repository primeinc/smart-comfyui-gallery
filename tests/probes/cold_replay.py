"""Cold replay at its coldest, as one fresh interpreter.

Run by `just probes`, never by pytest: a test may not start a program
(sglint SG006), so the interpreter boundary lives in the recipe and this
child spawns nothing. The proof is unchanged from its pytest ancestor:
write a face through the application's own harvest with a fake backend,
close the database, read it back through replay and export, and assert
the producer stack was never imported -- if any of torch, insightface or
onnxruntime is resident at the end, the stored record did not carry the
replay on its own.
"""

import hashlib
import sys
import tempfile
from pathlib import Path

from PIL import Image


def main() -> int:
    import numpy as np

    from db import authored, connect, detect, faces_native, library, scan
    from tests.staging import NOW
    from tests.test_a_stored_face_replays_and_exports_without_the_producer import producer_record, replaying

    with tempfile.TemporaryDirectory() as scratch:
        where = Path(scratch) / "library.sgly"
        connect.create(where)
        conn = connect.connect(where)
        try:
            root = Path(scratch) / "lib"
            root.mkdir()
            root_id = library.add_root(conn, root, "library", NOW)
            folder_id = scan.ensure_folder(conn, root_id, None, "lib")
            file_id = scan.mint(conn, "file", "cold")
            conn.execute(
                "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256, width, height,"
                " first_seen_at, last_seen_at) VALUES(?, ?, 'cold.png', 'image', 10, 0, 'aa', 640, 480, ?, ?)",
                (file_id, folder_id, NOW, NOW),
            )
            authored.add_user(conn, "will", "hash", "ADMIN", NOW)
            (face_id,) = detect.harvest(
                conn, replaying([producer_record()]), file_id, None, NOW, image=Image.new("RGB", (640, 480))
            )
            warm = hashlib.sha256(
                faces_native.reactor_face_model_bytes(faces_native.native_of(conn, face_id))
            ).hexdigest()
            conn.commit()
        finally:
            connect.close(conn)

        conn = connect.connect(where)
        try:
            native = faces_native.native_of(conn, face_id)
            cold = hashlib.sha256(faces_native.reactor_face_model_bytes(native)).hexdigest()
        finally:
            connect.close(conn)

    for producer_module in ("torch", "insightface", "onnxruntime"):
        assert producer_module not in sys.modules, f"replay or export leaned on {producer_module}"
    assert cold == warm, "the cold read exported different bytes than the writing pass"
    record = native.record
    assert isinstance(record["embedding"], np.ndarray)
    print(f"cold replay ok: {cold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
