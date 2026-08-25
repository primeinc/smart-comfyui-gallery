"""A bug in the reader is not a chore for the person who hit it.

The sniffer called every .m4a a video, ingest wrote the wrong `kind`,
and a folder of album tracks drew a wall of broken pictures. Fixing the
sniffer fixed nothing already recorded: `ingested_sha256` says which
BYTES were read, so a file is stale when its bytes change and never when
the READER improves. The only repair was re-reading the whole library by
hand -- a defect turned into a task, handed to the person it happened to.

So the reader signs its work. `db/ingest.py READER` is bumped whenever a
change would make it write something different for the same bytes, every
file read by an older one is stale by the ORDINARY rule, and the
ordinary sweep -- the one a worker already runs for what is missing --
repairs the library with nobody asked to do anything.

Which puts one obligation on whoever changes the reader, and it is the
only one: bump it when the answer may differ, and not for a refactor.
Bumping it needlessly re-reads every file in the library; not bumping it
leaves a library holding what a reader nobody would ship now decided.
"""

from __future__ import annotations

import pytest

from db import ingest, runner
from tests.staging import fresh_schema

pytestmark = pytest.mark.slow

NOW = 1_700_000_000.0
FILES = 3


@pytest.fixture
def library():
    conn = fresh_schema()
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'C:/x','library',0)")
    conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(1,?,'folder','lib')", (b"\x01" * 16,))
    conn.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(1,1,NULL,'lib',0)")
    for at in range(2, 2 + FILES):
        conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(?,?,'file',?)", (at, bytes([at]) * 16, f"f{at}"))
        conn.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,content_sha256,first_seen_at,last_seen_at)"
            " VALUES(?,1,?,'audio',1,0,?,0,0)",
            (at, f"t{at}.m4a", f"{at:064d}"),
        )
    conn.commit()
    yield conn
    conn.close()


def _read_by(conn, reader: str | None) -> None:
    """Mark every file as read for its current bytes by `reader`."""
    conn.execute("UPDATE file SET ingested_sha256 = content_sha256, ingested_by = ?", (reader,))
    conn.commit()


def _due(conn) -> int:
    job = runner.submit_ingest(conn, NOW)
    return 0 if job is None else conn.execute("SELECT count(*) FROM job_item WHERE job_id = ?", (job,)).fetchone()[0]


def test_a_file_read_by_this_reader_for_these_bytes_is_not_due(library):
    """The control. Without it every test below passes over a rule that
    simply queues everything."""
    _read_by(library, ingest.READER)
    assert _due(library) == 0


def test_a_file_read_by_an_older_reader_is_due(library):
    """The whole point. Nothing about the file changed -- same bytes,
    same row -- and it is stale because the thing that read it has been
    fixed since."""
    _read_by(library, "ingest/1970-01-01")
    assert _due(library) == FILES


def test_a_file_read_before_the_column_existed_is_due(library):
    """NULL is not "read by the current reader". Those rows are exactly
    the population that may carry what an old reader decided, which is
    why the migration leaves them NULL rather than claiming them fresh."""
    _read_by(library, None)
    assert _due(library) == FILES


def test_changed_bytes_are_still_due_on_their_own(library):
    """The half that already worked keeps working: a reader signature
    that matches does not excuse bytes that moved."""
    _read_by(library, ingest.READER)
    library.execute("UPDATE file SET content_sha256 = 'ffff' WHERE id = 2")
    library.commit()
    assert _due(library) == 1


def test_the_repair_needs_nobody_to_ask_for_it(library):
    """`everything=True` is the sledgehammer and it is still there. The
    point is that it is not NEEDED: the ordinary sweep, the one a worker
    runs by itself, is what repairs a library after a reader is fixed."""
    _read_by(library, "ingest/1970-01-01")
    ordinary = runner.submit_ingest(library, NOW)
    assert ordinary is not None, "the ordinary sweep found nothing to repair"
    assert library.execute("SELECT count(*) FROM job_item WHERE job_id = ?", (ordinary,)).fetchone()[0] == FILES


def test_reading_records_both_the_bytes_and_the_reader(library, tmp_path):
    """Together and in one write: a row claiming to be current for bytes
    it was not read for, or for a reader that did not read it, is the
    freshness rule lying."""
    from PIL import Image

    path = tmp_path / "one.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
    library.execute("UPDATE file SET content_sha256 = NULL WHERE id = 2")
    library.commit()

    ingest.one(library, 2, path, NOW)
    library.commit()
    held = library.execute("SELECT ingested_sha256, ingested_by, content_sha256 FROM file WHERE id = 2").fetchone()
    assert held[1] == ingest.READER
    assert held[0] == held[2], "it claimed to be read for bytes it was not read for"


def test_the_console_counts_what_the_sweep_would_queue(library):
    """Counted differently from what is queued, the console says "0
    missing" beside a sweep that is about to read the whole library."""
    from db import inspecting

    _read_by(library, "ingest/1970-01-01")
    assert inspecting.coverage(library)["missing"]["ingest"] == _due(library) == FILES

    _read_by(library, ingest.READER)
    assert inspecting.coverage(library)["missing"]["ingest"] == 0


def test_the_reader_is_dated_not_numbered():
    """Two branches that both bump a number silently agree on "4" and
    leave one of their libraries unrepaired."""
    assert ingest.READER.startswith("ingest/")
    stamp = ingest.READER.split("/", 1)[1]
    assert len(stamp) == 10, ingest.READER
    assert stamp.count("-") == 2, ingest.READER
