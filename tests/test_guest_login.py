"""A guest identity must be minted, never claimed.

The passwordless path takes whatever the browser offers as its identity.
Claiming `41` used to hand a stranger account 41's comments and let them
replace that account's rating, which is aggregated -- so one visitor could
move a public average and delete someone else's words.

A well-formed browser UUID IS accepted as-is: the exhibition and main
templates mint their own with crypto.randomUUID and store it under the key
the login sends, so rejecting that shape would silently orphan every
existing visitor's ratings. Anything else is discarded and a fresh id
issued.

Note what is NOT claimed. A guest id is a bearer token -- 64 bits from
secrets.token_hex(8) -- so presenting a well-formed one legitimately grants
that guest identity; that IS the continuity feature. The line that matters
is between guessable account ids and unguessable guest ones.

Each case used to run a fresh interpreter because the flags come from argv
at import. They are three module attributes, read at request time, so
monkeypatch sets them on the loaded gallery instead
(pytest doc/en/how-to/monkeypatch.rst:243-247) -- twelve of these cases are
one parametrized identity check, which was twelve interpreters.
"""

from __future__ import annotations

import secrets

import pytest

# Generated per run rather than written down: these are throwaway
# credentials for this file's fixtures, and a literal one is something
# somebody eventually pastes into a real config.
_PASSWORD = secrets.token_urlsafe(16)


@pytest.fixture
def guests_welcome(smartgallery_app, monkeypatch):
    """The gallery as `--enable-guest-login --force-login --admin-pass` leaves it."""
    force_login, missing, _short = smartgallery_app.derive_login_policy(_PASSWORD, exhibition=False, force_login=True)
    monkeypatch.setattr(smartgallery_app, "ENABLE_GUEST_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "ADMIN_PASS_INPUT", _PASSWORD)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", force_login)
    monkeypatch.setattr(smartgallery_app, "ADMIN_CONFIG_MISSING", missing)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    return smartgallery_app


@pytest.fixture
def a_file(smartgallery_app, request):
    """One file to comment on and rate, removed afterwards along with
    anything attached to it."""
    file_id = f"guest_{request.node.name[:24]}"
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO files (id, path, mtime, name, type, size) VALUES (?, ?, 1.0, 'a.png', 'image', 1)",
            (file_id, f"/x/{file_id}.png"),
        )
        conn.commit()
    finally:
        conn.close()

    yield file_id

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM file_comments WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM file_ratings WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        conn.commit()
    finally:
        conn.close()


def test_a_guest_cannot_claim_a_real_accounts_identity(guests_welcome, a_file):
    """The regression: claiming user_id 41 used to hand a stranger that
    account's comments."""
    conn = guests_welcome.get_db_connection()
    try:
        conn.execute(
            "INSERT INTO file_comments (file_id, client_uuid, author_name, "
            "comment_text, target_audience, created_at) "
            "VALUES (?, '41', 'RealUser', 'private words', 'public', 1.0)",
            (a_file,),
        )
        comment_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
    finally:
        conn.close()

    client = guests_welcome.app.test_client()
    issued = client.post("/galleryout/login", json={"username": "guest", "provided_uuid": "41"})
    identity = issued.get_json()["client_uuid"]

    assert identity != "41", f"the server handed out the claimed identity: {identity}"
    assert identity.startswith("guest_"), identity

    resp = client.post("/galleryout/api/exhibition/delete_comment", json={"comment_id": comment_id})
    assert resp.status_code == 403, f"a guest deleted another account comment ({resp.status_code})"

    conn = guests_welcome.get_db_connection()
    try:
        alive = conn.execute("SELECT 1 FROM file_comments WHERE id = ?", (comment_id,)).fetchone() is not None
    finally:
        conn.close()
    assert alive, "the victim's comment was destroyed"


def test_a_guest_cannot_overwrite_an_accounts_rating(guests_welcome, a_file):
    """The same identity check protects ratings, which are aggregated: a
    guest claiming account 41 would have replaced that account's vote and
    moved the average."""
    conn = guests_welcome.get_db_connection()
    try:
        conn.execute(
            "INSERT INTO file_ratings (file_id, client_uuid, rating, created_at) VALUES (?, '41', 5, 1.0)", (a_file,)
        )
        conn.commit()
    finally:
        conn.close()

    client = guests_welcome.app.test_client()
    client.post("/galleryout/login", json={"username": "guest", "provided_uuid": "41"})
    client.post("/galleryout/api/exhibition/rate", json={"file_id": a_file, "rating": 1})

    conn = guests_welcome.get_db_connection()
    try:
        rows = [
            (r[0], r[1])
            for r in conn.execute(
                "SELECT client_uuid, rating FROM file_ratings WHERE file_id = ?", (a_file,)
            ).fetchall()
        ]
    finally:
        conn.close()

    victim = [r for r in rows if r[0] == "41"]
    assert victim, f"the account's rating was overwritten: {rows}"
    assert victim[0][1] == 5, f"the account's rating was overwritten: {rows}"


@pytest.mark.parametrize(
    "claimed",
    [
        "41",
        "admin",
        "1",
        "guest",
        "GUEST_ABC",
        "guest_",
        "guest_zzzz",
        "guest_dead beef",
        "../guest_deadbeef",
        "guest_deadbeef' OR '1'='1",
        "3f2b1c4d-aaaa-bbbb",
        "not-a-uuid-at-all",
    ],
)
def test_guessable_identities_are_never_accepted(guests_welcome, claimed):
    issued = guests_welcome.app.test_client().post(
        "/galleryout/login", json={"username": "guest", "provided_uuid": claimed}
    )
    identity = issued.get_json()["client_uuid"]

    assert identity != claimed, "a malformed identity was accepted verbatim"
    assert identity.startswith("guest_"), identity


def test_a_browser_generated_uuid_is_honoured(guests_welcome):
    """The exhibition and main templates mint their own identity with
    crypto.randomUUID when they have never been issued one, and store it
    under the same key the guest login sends. Rejecting that shape would
    silently orphan every existing visitor's ratings, so a well-formed
    UUID -- unguessable, and never colliding with an integer account id --
    is accepted as-is."""
    existing = "3f2b1c4d-9e7a-4b21-8f6c-1a2b3c4d5e6f"

    issued = guests_welcome.app.test_client().post(
        "/galleryout/login", json={"username": "guest", "provided_uuid": existing}
    )

    assert issued.get_json()["client_uuid"] == existing, "a browser-generated identity was discarded"


def test_a_returning_guest_keeps_their_own_id(guests_welcome):
    """The feature still has to work: a guest who comes back with the id
    this server minted keeps it, so their ratings remain theirs."""
    client = guests_welcome.app.test_client()
    first = client.post("/galleryout/login", json={"username": "guest"}).get_json()["client_uuid"]
    assert first.startswith("guest_"), first

    again = client.post("/galleryout/login", json={"username": "guest", "provided_uuid": first}).get_json()[
        "client_uuid"
    ]

    assert again == first, f"a returning guest lost their identity: {first} -> {again}"


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/galleryout/delete_batch", {"file_ids": ["x"]}),
        ("/galleryout/delete_folder/_root_", {}),
        ("/galleryout/rename_file/x", {"new_name": "y.png"}),
    ],
)
def test_a_guest_is_still_refused_the_management_apis(guests_welcome, path, payload):
    client = guests_welcome.app.test_client()
    client.post("/galleryout/login", json={"username": "guest"})

    resp = client.post(path, json=payload)

    assert resp.status_code == 403, f"a guest reached {path}: {resp.status_code}"


def test_guest_login_is_refused_when_the_flag_is_off(guests_welcome, monkeypatch):
    """Without --enable-guest-login the passwordless path must not exist."""
    monkeypatch.setattr(guests_welcome, "ENABLE_GUEST_LOGIN", False)
    client = guests_welcome.app.test_client()

    resp = client.post("/galleryout/login", json={"username": "guest"})

    assert resp.get_json().get("status") != "success", "guest login worked without the flag"
    with client.session_transaction() as session:
        assert not session.get("user_id"), "a session was created anyway"
