"""Tests for sg_auth (Argon2id password hashing / legacy Fernet migration)
and the smartgallery.py routes that consume it."""

import os
import re
import sqlite3
import tempfile

import pytest
from cryptography.fernet import Fernet

import sg_auth

USERS_DDL = """
    CREATE TABLE users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        full_name TEXT NOT NULL,
        email TEXT,
        phone_number TEXT,
        role TEXT CHECK(role IN ('USER', 'STAFF', 'MANAGER', 'CUSTOMER', 'FRIEND', 'GUEST', 'ADMIN')) DEFAULT 'GUEST',
        start_date DATE DEFAULT CURRENT_DATE,
        expiry_date DATE,
        is_active BOOLEAN DEFAULT 1,
        last_login REAL
    );
"""


# --- hash_password / verify_password ------------------------------------

def test_hash_password_produces_argon2id():
    h = sg_auth.hash_password("correct horse battery staple")
    assert h.startswith("$argon2id$")


def test_verify_password_accepts_correct_rejects_wrong():
    h = sg_auth.hash_password("s3cret-password")
    valid, needs_rehash = sg_auth.verify_password(h, "s3cret-password")
    assert valid is True
    assert needs_rehash is False

    valid, needs_rehash = sg_auth.verify_password(h, "wrong-password")
    assert valid is False
    assert needs_rehash is False


@pytest.mark.parametrize("stored", [
    None,
    "",
    sg_auth.UNUSABLE_PASSWORD,
    "gAAAAABqf6eGg-H7hFSKDW6uZxD0W-XTz9URdPFytuDO7uiPED9ujScO3FIBXT1-vZU8OrzCZYXHAbUcrsCRxW9fBKIxoXDyWw==",
    "plaintext-not-a-hash-at-all",
    "$2b$12$notanargon2hash..................",
    12345,
])
def test_verify_password_never_raises_on_garbage(stored):
    valid, needs_rehash = sg_auth.verify_password(stored, "whatever")
    assert (valid, needs_rehash) == (False, False)


def test_verify_password_never_raises_on_none_candidate():
    h = sg_auth.hash_password("some-password")
    assert sg_auth.verify_password(h, None) == (False, False)
    assert sg_auth.verify_password(h, "") == (False, False)


def test_is_legacy_ciphertext():
    assert sg_auth.is_legacy_ciphertext("gAAAAABqf6eGg-H7hFSKDW6u") is True
    assert sg_auth.is_legacy_ciphertext(sg_auth.hash_password("x")) is False
    assert sg_auth.is_legacy_ciphertext(sg_auth.UNUSABLE_PASSWORD) is False
    assert sg_auth.is_legacy_ciphertext(None) is False
    assert sg_auth.is_legacy_ciphertext("") is False


# --- migrate_legacy_passwords --------------------------------------------

def _make_legacy_db(tmp_path):
    """3 users with real Fernet ciphertext + 1 with corrupt ciphertext."""
    db_path = os.path.join(tmp_path, "legacy.sqlite")
    key_path = os.path.join(tmp_path, "system.key")

    key = Fernet.generate_key()
    with open(key_path, "wb") as f:
        f.write(key)
    fernet = Fernet(key)

    conn = sqlite3.connect(db_path)
    conn.execute(USERS_DDL)

    plaintexts = {
        "alice": "alice-password-1",
        "bob": "bob-password-2",
        "carol": "carol-password-3",
    }
    for username, pw in plaintexts.items():
        token = fernet.encrypt(pw.encode()).decode()
        conn.execute(
            "INSERT INTO users (username, password, full_name, role) VALUES (?, ?, ?, 'USER')",
            (username, token, username.title()),
        )
    # Corrupt ciphertext: looks legacy (gAAAA-prefixed) but won't decrypt.
    conn.execute(
        "INSERT INTO users (username, password, full_name, role) VALUES (?, ?, ?, 'USER')",
        ("dave", "gAAAAABnotarealtoken====", "Dave"),
    )
    conn.commit()
    return conn, key_path, plaintexts


def test_migrate_legacy_passwords(tmp_path):
    conn, key_path, plaintexts = _make_legacy_db(str(tmp_path))

    report = sg_auth.migrate_legacy_passwords(conn, key_path)

    assert report["migrated"] == 3
    assert report["failed"] == 1
    assert report["key_deleted"] is True
    assert not os.path.exists(key_path)

    rows = {r[0]: r[1] for r in conn.execute("SELECT username, password FROM users")}

    for username, plaintext in plaintexts.items():
        assert rows[username].startswith("$argon2id$")
        valid, _ = sg_auth.verify_password(rows[username], plaintext)
        assert valid is True

    assert rows["dave"] == sg_auth.UNUSABLE_PASSWORD

    # No Fernet ciphertext survives migration anywhere in the table.
    all_passwords = " ".join(rows.values())
    assert "gAAAA" not in all_passwords


def test_migrate_legacy_passwords_is_idempotent(tmp_path):
    conn, key_path, plaintexts = _make_legacy_db(str(tmp_path))

    first = sg_auth.migrate_legacy_passwords(conn, key_path)
    assert first["migrated"] == 3
    assert first["failed"] == 1

    hashes_after_first = dict(conn.execute("SELECT username, password FROM users"))

    second = sg_auth.migrate_legacy_passwords(conn, key_path)
    # Nothing legacy remains: everything is either an argon2 hash or the
    # sentinel, both skipped on the second pass.
    assert second["migrated"] == 0
    assert second["failed"] == 0
    assert second["skipped"] == 4
    assert second["key_deleted"] is False  # already deleted by the first pass

    hashes_after_second = dict(conn.execute("SELECT username, password FROM users"))
    assert hashes_after_first == hashes_after_second


def test_migrate_legacy_passwords_missing_key_file_marks_failed(tmp_path):
    """Without a key file, legacy ciphertext is unrecoverable dead weight."""
    conn, key_path, _ = _make_legacy_db(str(tmp_path))
    os.remove(key_path)

    report = sg_auth.migrate_legacy_passwords(conn, key_path)

    assert report["migrated"] == 0
    assert report["failed"] == 4
    assert report["key_deleted"] is False

    rows = [r[0] for r in conn.execute("SELECT password FROM users")]
    assert all(p == sg_auth.UNUSABLE_PASSWORD for p in rows)


def test_migrate_legacy_passwords_safe_with_no_legacy_rows(tmp_path):
    db_path = os.path.join(str(tmp_path), "clean.sqlite")
    conn = sqlite3.connect(db_path)
    conn.execute(USERS_DDL)
    conn.execute(
        "INSERT INTO users (username, password, full_name, role) VALUES (?, ?, ?, 'USER')",
        ("erin", sg_auth.hash_password("already-migrated"), "Erin"),
    )
    conn.commit()

    key_path = os.path.join(str(tmp_path), "system.key")  # never created
    report = sg_auth.migrate_legacy_passwords(conn, key_path)

    assert report == {"migrated": 0, "failed": 0, "skipped": 1, "key_deleted": False}


# --- Recoverability: no decrypt path exists ------------------------------

def test_sg_auth_exposes_no_public_decrypt_api():
    public_names = [n for n in dir(sg_auth) if not n.startswith("_")]
    decrypt_like = [n for n in public_names if re.search("decrypt", n, re.IGNORECASE)]
    assert decrypt_like == []


def test_smartgallery_source_has_no_reversible_password_traces():
    with open(os.path.join(os.path.dirname(__file__), "..", "smartgallery.py")) as f:
        source = f.read()
    assert "decrypt_password" not in source
    assert "plain_password" not in source


# --- End-to-end via Flask test client -------------------------------------

def _insert_user(smartgallery_app, username, password_column_value, role="USER"):
    with smartgallery_app.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO users (username, password, full_name, role, is_active) "
            "VALUES (?, ?, ?, ?, 1)",
            (username, password_column_value, username.title(), role),
        )
        conn.commit()


def test_login_success_and_failure(smartgallery_app):
    _insert_user(smartgallery_app, "e2e_login_user", sg_auth.hash_password("hunter2secret"))

    client = smartgallery_app.app.test_client()

    resp = client.post("/galleryout/login", json={
        "username": "e2e_login_user", "password": "hunter2secret",
    })
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "success"

    client2 = smartgallery_app.app.test_client()
    resp = client2.post("/galleryout/login", json={
        "username": "e2e_login_user", "password": "wrong-password",
    })
    assert resp.status_code == 401
    assert resp.get_json()["status"] == "error"


def test_login_rejects_unusable_sentinel(smartgallery_app):
    _insert_user(smartgallery_app, "e2e_sentinel_user", sg_auth.UNUSABLE_PASSWORD)

    client = smartgallery_app.app.test_client()
    resp = client.post("/galleryout/login", json={
        "username": "e2e_sentinel_user", "password": "anything-at-all",
    })
    assert resp.status_code == 401
    assert resp.get_json()["status"] == "error"


def test_admin_users_endpoint_never_leaks_password_fields(smartgallery_app):
    _insert_user(smartgallery_app, "e2e_admin_list_user", sg_auth.hash_password("another-secret-1"))

    client = smartgallery_app.app.test_client()
    resp = client.get("/galleryout/api/admin/users")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert len(body["users"]) >= 1
    for user in body["users"]:
        assert "password" not in user
        assert "plain_password" not in user
