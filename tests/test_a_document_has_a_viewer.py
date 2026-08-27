"""A PDF is displayed, not handed back as a link.

The data layer held its half: a document states its page count and gets
one sample per page (test_every_claimed_suffix_is_supported). The web
layer then served all of it as one `<a>download</a>` -- the viewer's
document arm was a dead end, so "PDFs don't work" was the truthful user
report even though every row underneath was right.

The browser ships a PDF renderer. /media types the bytes
`application/pdf` from the sniff (vision/sniff.py), never from the
suffix -- proven at the sniff seam here, at the route by
test_the_bytes_are_served's content-type coverage. The stage hosts that
renderer in a frame; these tests hold the seam and the stage.
"""

from __future__ import annotations

import pytest
from litestar.testing import TestClient

from db import connect
from sg_web.app import build_app

pytestmark = pytest.mark.slow


def test_pdf_bytes_are_typed_application_pdf(tmp_path):
    """The seam the route reads: the magic bytes, never the suffix,
    decide the type -- and `application/pdf` is what lets a browser
    render the frame instead of downloading it."""
    from vision import sniff

    held = tmp_path / "held.pdf"
    held.write_bytes(b"%PDF-1.7\n")
    assert sniff.content_type(sniff.sniff_path(held)) == "application/pdf"


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """One read-only world: no snapshot, no restore -- nothing here
    writes, so the staging machinery's backup would be pure setup cost."""
    import pypdf

    base = tmp_path_factory.mktemp("document-viewer")
    root = base / "lib"
    root.mkdir()
    writer = pypdf.PdfWriter()
    writer.add_blank_page(612, 792)
    writer.add_blank_page(612, 792)
    with (root / "manual.pdf").open("wb") as handle:
        writer.write(handle)
    with TestClient(app=build_app(str(base / "run"))) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        yield client


def _slug(client) -> str:
    conn = connect.connect(client.app.state.db_path)
    try:
        return conn.execute(
            "SELECT e.slug FROM entity e JOIN file f ON f.id = e.id WHERE f.kind = 'document'"
        ).fetchone()[0]
    finally:
        connect.close(conn)


def test_the_viewer_stages_the_document_in_a_frame(served):
    """The stage is the browser's own renderer over the original, not a
    bare link -- the arm that made every PDF a dead end."""
    slug = _slug(served)
    page = served.get(f"/i/{slug}", headers={"accept": "text/html"})
    assert page.status_code == 200
    assert f'<iframe class="viewer-document" data-stage-media src="/media/{slug}"' in page.text
    assert "download manual.pdf" not in page.text


def test_the_stage_contract_names_its_source(served):
    """The typed stage a machine reads: a document arm that carries `src`
    like every other kind, so nothing measures the DOM to find it."""
    slug = _slug(served)
    told = served.get(f"/i/{slug}").json()
    assert told["stage"] == {"kind": "document", "src": f"/media/{slug}", "original": f"/media/{slug}"}
