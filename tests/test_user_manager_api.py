"""User-manager API behavior around one-way password storage.

The edit form can no longer prefill the current password (hashes are not
reversible), so a PUT with an empty password must keep the stored hash and
a PUT with a new password must replace it.
"""

import sg_auth


def _fresh_user(smartgallery, username, password):
    with smartgallery.get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.execute(
            "INSERT INTO users (username, password, full_name, role, is_active) "
            "VALUES (?, ?, ?, 'USER', 1)",
            (username, sg_auth.hash_password(password), "Edit Target"),
        )
        conn.commit()
        return conn.execute(
            "SELECT user_id, password FROM users WHERE username = ?", (username,)
        ).fetchone()


def test_put_with_empty_password_keeps_existing_hash(smartgallery_app):
    sg = smartgallery_app
    row = _fresh_user(sg, "edituser", "originalpass1")
    client = sg.app.test_client()

    resp = client.put(
        "/galleryout/api/admin/users",
        json={
            "user_id": row["user_id"],
            "username": "edituser",
            "password": "",
            "full_name": "Edited Name",
            "role": "STAFF",
            "is_active": 1,
        },
    )
    assert resp.status_code == 200

    with sg.get_db_connection() as conn:
        after = conn.execute(
            "SELECT password, full_name, role FROM users WHERE user_id = ?",
            (row["user_id"],),
        ).fetchone()
    assert after["password"] == row["password"]  # hash untouched
    assert after["full_name"] == "Edited Name"
    assert after["role"] == "STAFF"
    assert sg_auth.verify_password(after["password"], "originalpass1")[0]


def test_put_with_new_password_replaces_hash(smartgallery_app):
    sg = smartgallery_app
    row = _fresh_user(sg, "edituser2", "originalpass2")
    client = sg.app.test_client()

    resp = client.put(
        "/galleryout/api/admin/users",
        json={
            "user_id": row["user_id"],
            "username": "edituser2",
            "password": "brandnewpass9",
            "full_name": "Edit Target",
            "role": "USER",
            "is_active": 1,
        },
    )
    assert resp.status_code == 200

    with sg.get_db_connection() as conn:
        after = conn.execute(
            "SELECT password FROM users WHERE user_id = ?", (row["user_id"],)
        ).fetchone()
    assert after["password"] != row["password"]
    assert after["password"].startswith("$argon2")
    assert sg_auth.verify_password(after["password"], "brandnewpass9")[0]
    assert not sg_auth.verify_password(after["password"], "originalpass2")[0]


def test_put_with_short_password_rejected(smartgallery_app):
    sg = smartgallery_app
    row = _fresh_user(sg, "edituser3", "originalpass3")
    client = sg.app.test_client()

    resp = client.put(
        "/galleryout/api/admin/users",
        json={
            "user_id": row["user_id"],
            "username": "edituser3",
            "password": "short",
            "full_name": "Edit Target",
            "role": "USER",
            "is_active": 1,
        },
    )
    assert resp.status_code == 400

    with sg.get_db_connection() as conn:
        after = conn.execute(
            "SELECT password FROM users WHERE user_id = ?", (row["user_id"],)
        ).fetchone()
    assert after["password"] == row["password"]
