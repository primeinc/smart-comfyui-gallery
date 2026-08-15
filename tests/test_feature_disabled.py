"""WI-31 acceptance: with the AI DAM layer disabled (the default), the
feature must be fully inert -- every /galleryout/api/aidam/* route (except
/status, which always reports) responds {'enabled': False}, normal gallery
browsing routes are unaffected, no background worker is running, and no
heavy model runtime (torch) is ever imported on that path.
"""

from __future__ import annotations

import sys

import smartgallery_ai
from smartgallery_ai.service import create_ai_resolvers

_PREFIX = "/galleryout/api/aidam"


def test_ai_config_disabled_by_default(smartgallery_app):
    assert isinstance(smartgallery_app.AI_CONFIG, smartgallery_ai.AIConfig)
    assert smartgallery_app.AI_CONFIG.enabled is False


def test_no_worker_thread_running_by_default(smartgallery_app):
    worker = smartgallery_app.ai_dam_service.get_worker()
    assert worker is None or not worker.is_running
    assert not any(t.name == "AIWorker" for t in __import__("threading").enumerate())


def test_status_route_always_reports(smartgallery_app):
    client = smartgallery_app.app.test_client()
    resp = client.get(f"{_PREFIX}/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["enabled"] is False
    assert "backends" in data
    assert "counts" in data
    assert "worker" in data


def test_every_other_route_short_circuits_to_disabled(smartgallery_app):
    client = smartgallery_app.app.test_client()
    checks = [
        ("get", "/duplicates/f1"),
        ("get", "/similar/f1"),
        ("get", "/faces/f1"),
        ("get", "/faces/clusters"),
        ("get", "/faces/clusters/1"),
        ("post", "/faces/recluster"),
        ("get", "/review/f1"),
        ("get", "/review/mask/1"),
        ("post", "/review/feedback"),
        ("get", "/review/feedback/export"),
        ("post", "/index/f1"),
    ]
    for method, path in checks:
        resp = getattr(client, method)(f"{_PREFIX}{path}")
        assert resp.status_code == 200, f"{method.upper()} {path} -> {resp.status_code}"
        assert resp.get_json() == {"enabled": False}, f"{method.upper()} {path} leaked data"


def test_gallery_view_still_works(smartgallery_app):
    client = smartgallery_app.app.test_client()
    resp = client.get("/")
    assert resp.status_code in (200, 302)

    resp2 = client.get("/galleryout/")
    assert resp2.status_code in (200, 302)


def test_ai_resolvers_disabled_config_still_returns_lists_without_crashing(smartgallery_app):
    # create_ai_resolvers() itself doesn't gate on config.enabled (the AST
    # validator does, via requires_ai/ai_enabled) -- it must still open/close
    # its own connection cleanly against the real monolith DB and return [].
    resolvers = create_ai_resolvers(smartgallery_app.AI_CONFIG)
    assert resolvers["near_dup_of"]("no-such-file") == []
    assert resolvers["similar_to_semantic"]("no-such-file") == []
    assert resolvers["similar_to_visual"]({"file_id": "no-such-file", "k": 5}) == []


def test_normal_browsing_never_imports_torch(smartgallery_app):
    client = smartgallery_app.app.test_client()
    client.get("/")
    client.get(f"{_PREFIX}/status")
    client.post(f"{_PREFIX}/index/f1")
    assert "torch" not in sys.modules
