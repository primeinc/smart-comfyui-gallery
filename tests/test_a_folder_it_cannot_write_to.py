"""When the pictures folder refuses a write, say what that means.

Being unable to write to the pictures folder is one of the ordinary ways
this program is set up wrong, not a rare accident:

  * the folder mounted read-only, which people do on purpose -- the
    supplied compose file already mounts the model folders `:ro`
  * a container running as a different user than the one that owns the
    files, which is the whole reason WANTED_UID, WANTED_GID and
    FORCE_CHOWN exist in that compose file
  * a network share that has gone away
  * a file another program still has open

Every one of the six actions that touch the folder answered with the
error as the operating system phrased it:

    {"message": "Error: [Errno 13] Permission denied:
                 'C:/.../output\\picture.png'", "status": "error"}

which tells somebody who has not met errno before nothing at all about
what to do. Uploading was worse: it named which files failed and never
said why, because the reason was collected into a dictionary and then
dropped when the message was built.

Two things this deliberately does NOT change, both measured first and
both already right:

  * every one of these actions already fails loudly. None of them
    reported success while the write was refused.
  * the library still matched the disk afterwards. Nothing recorded a
    rename or a deletion that had not happened.

So this is about what the person is told, not about what was done. Only
permission and read-only failures are put into words; anything else is
left exactly as it was rather than dressed up as something it might not
be.
"""

from __future__ import annotations

import builtins
import errno
import io as _io
import json
import os
import shutil
import struct
import zlib

import pytest

import smartgallery


def a_real_png():
    def chunk(kind, payload):
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def a_folder_that_refuses_writes(smartgallery_app, tmp_path, monkeypatch):
    """A library whose pictures folder raises PermissionError on any
    write, which is what a read-only mount raises (EACCES/EROFS)."""
    sg = smartgallery_app
    root = tmp_path / "read_only_pictures"
    root.mkdir()
    (root / "picture.png").write_bytes(a_real_png())
    (root / "a_folder").mkdir()

    monkeypatch.setattr(sg, "BASE_OUTPUT_PATH", str(root))
    with sg.get_db_connection() as conn:
        conn.execute("DELETE FROM files")
        conn.commit()
        sg.full_sync_database(conn)
        row = conn.execute("SELECT id FROM files WHERE name = 'picture.png'").fetchone()
    assert row, "the scan did not record the picture this rests on"

    folders = sg.get_dynamic_folder_config(force_refresh=True)
    folder_key = next(k for k, v in folders.items() if str(v.get("path", "")).endswith("a_folder"))

    client = sg.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"

    forbidden = os.path.realpath(str(root)).lower()

    def inside(path):
        try:
            return os.path.realpath(str(path)).lower().startswith(forbidden)
        except (TypeError, ValueError, OSError):
            return False

    def refusing(original, index=0):
        def wrapper(*args, **kwargs):
            target = args[index] if len(args) > index else ""
            if inside(target):
                raise PermissionError(errno.EACCES, "Permission denied", str(target))
            return original(*args, **kwargs)

        return wrapper

    real_open = builtins.open

    def refusing_open(*args, **kwargs):
        mode = kwargs.get("mode", args[1] if len(args) > 1 else "r")
        target = args[0] if args else ""
        if any(c in str(mode) for c in "wax+") and inside(target):
            raise PermissionError(errno.EACCES, "Permission denied", str(target))
        return real_open(*args, **kwargs)

    def arm():
        for name in ("rename", "replace", "makedirs", "remove", "unlink", "rmdir"):
            monkeypatch.setattr(os, name, refusing(getattr(os, name)))
        for name in ("move", "copy2", "copy"):
            monkeypatch.setattr(shutil, name, refusing(getattr(shutil, name), 1))
        monkeypatch.setattr(shutil, "rmtree", refusing(shutil.rmtree))
        monkeypatch.setattr(builtins, "open", refusing_open)

    yield sg, client, row["id"], folder_key, str(root), arm

    with smartgallery.get_db_connection() as conn:
        conn.execute("DELETE FROM files")
        conn.commit()


def every_action(file_id, folder_key):
    return [
        ("rename a picture", "/galleryout/rename_file/" + file_id, {"json": {"new_name": "other.png"}}),
        (
            "create a folder",
            "/galleryout/create_folder",
            {"json": {"folder_name": "brand_new", "parent_key": "_root_"}},
        ),
        ("rename a folder", "/galleryout/rename_folder/" + folder_key, {"json": {"new_name": "renamed_folder"}}),
        ("delete a picture", "/galleryout/delete/" + file_id, {}),
        ("delete a folder", "/galleryout/delete_folder/" + folder_key, {}),
        (
            "upload a picture",
            "/galleryout/upload",
            {
                "data": {"files": (_io.BytesIO(a_real_png()), "new.png"), "folder_key": "_root_"},
                "content_type": "multipart/form-data",
            },
        ),
    ]


def post(client, url, kwargs):
    return client.open(url, method="POST", headers={"Sec-Fetch-Site": "same-origin"}, **kwargs)


def test_the_folder_really_does_refuse(a_folder_that_refuses_writes):
    """Control. If the refusal were not in force every check below would
    pass by measuring an ordinary working gallery."""
    _sg, _client, _file_id, _folder_key, root, arm = a_folder_that_refuses_writes
    arm()

    with pytest.raises(PermissionError):
        os.makedirs(os.path.join(root, "nope"))
    # The open itself is what must raise, so the body never runs; written as
    # a `with` so a build where it unexpectedly succeeds still closes it.
    with pytest.raises(PermissionError), builtins.open(os.path.join(root, "nope.txt"), "w"):
        pass

    # and somewhere else is still perfectly writable
    elsewhere = os.path.join(os.path.dirname(root), "elsewhere.txt")
    with builtins.open(elsewhere, "w") as handle:
        handle.write("fine")


def test_none_of_them_claims_to_have_worked(a_folder_that_refuses_writes):
    """Control that passes on both builds: this was already right, and
    the change must not break it."""
    _sg, client, file_id, folder_key, _root, arm = a_folder_that_refuses_writes
    arm()

    lied = []
    for label, url, kwargs in every_action(file_id, folder_key):
        answer = post(client, url, kwargs)
        if answer.status_code == 200:
            lied.append(f"{label} -> 200 {answer.get_data(as_text=True)}")
    assert not lied, "\n  ".join(lied)


def test_the_library_still_matches_the_disk(a_folder_that_refuses_writes):
    """The other control that passes on both: a refused rename or delete
    must not be recorded as though it happened."""
    sg, client, file_id, folder_key, _root, arm = a_folder_that_refuses_writes
    arm()

    for _label, url, kwargs in every_action(file_id, folder_key):
        post(client, url, kwargs)

    with sg.get_db_connection() as conn:
        rows = conn.execute("SELECT name, path FROM files").fetchall()
    assert rows, "the library forgot the picture that is still on the disk"
    for row in rows:
        assert os.path.exists(row["path"]), "the library points at {}, which is not on the disk".format(row["path"])


def test_each_one_says_what_is_actually_wrong(a_folder_that_refuses_writes):
    """The defect. Every one of them answered with errno."""
    _sg, client, file_id, folder_key, _root, arm = a_folder_that_refuses_writes
    arm()

    unhelpful = []
    for label, url, kwargs in every_action(file_id, folder_key):
        said = post(client, url, kwargs).get_data(as_text=True)
        if "not allowed to change" not in said:
            unhelpful.append(f"{label} said {said[:160]}")

    assert not unhelpful, "did not say the gallery cannot write to the folder:\n  " + "\n  ".join(unhelpful)


def test_none_of_them_hands_back_errno(a_folder_that_refuses_writes):
    """No error numbers and no operating-system phrasing.

    The folder itself IS named, on purpose -- somebody has to know which
    folder to go and fix, and these are the management screens, where the
    library's own path is on display anyway. What has no business being
    there is `[Errno 13]` and a traceback."""
    _sg, client, file_id, folder_key, _root, arm = a_folder_that_refuses_writes
    arm()

    machine_speak = []
    for label, url, kwargs in every_action(file_id, folder_key):
        answer = post(client, url, kwargs)
        said = answer.get_data(as_text=True)
        machine_speak.extend(
            f"{label} leaked {giveaway!r}: {said[:160]}"
            for giveaway in ("Errno", "errno", "Traceback", "Permission denied")
            if giveaway in said
        )
    assert not machine_speak, "\n  ".join(machine_speak)


def test_an_upload_says_why_it_failed(a_folder_that_refuses_writes):
    """It named the files and never said why: the reason was collected
    and then dropped when the message was built."""
    _sg, client, file_id, folder_key, _root, arm = a_folder_that_refuses_writes
    arm()

    _label, url, kwargs = every_action(file_id, folder_key)[-1]
    answer = post(client, url, kwargs)

    said = json.loads(answer.get_data(as_text=True))["message"]
    assert "new.png" in said, said
    assert "not allowed to change" in said, said


def test_everything_still_works_when_the_folder_is_writable(a_folder_that_refuses_writes):
    """Over-reach guard, and the case that matters most: none of this may
    make an ordinary gallery refuse or complain."""
    _sg, client, file_id, _folder_key, root, _arm = a_folder_that_refuses_writes
    # deliberately not armed

    made = post(client, "/galleryout/create_folder", {"json": {"folder_name": "brand_new", "parent_key": "_root_"}})
    assert made.status_code == 200, made.get_data(as_text=True)
    assert os.path.isdir(os.path.join(root, "brand_new"))

    renamed = post(client, "/galleryout/rename_file/" + file_id, {"json": {"new_name": "other.png"}})
    assert renamed.status_code == 200, renamed.get_data(as_text=True)
    assert os.path.exists(os.path.join(root, "other.png"))


class TestTheExplanationItself:
    def test_it_speaks_only_for_permission_failures(self):
        """Anything else is left as it was rather than dressed up as a
        permission problem it might not be."""
        say = smartgallery.explain_a_refused_write

        assert say(PermissionError(errno.EACCES, "denied", "/x/y.png"))
        assert say(OSError(errno.EROFS, "read-only", "/x/y.png"))
        assert say(OSError(errno.EPERM, "not permitted", "/x/y.png"))

        assert say(OSError(errno.ENOSPC, "no space", "/x/y.png")) is None
        assert say(FileNotFoundError(errno.ENOENT, "gone", "/x/y.png")) is None
        assert say(ValueError("not an OS error at all")) is None
        assert say(OSError("no errno at all")) is None

    def test_it_names_the_folder_not_the_file(self):
        say = smartgallery.explain_a_refused_write
        said = say(PermissionError(errno.EACCES, "denied", "/pictures/output/holiday.png"))

        assert "/pictures/output" in said
        assert "holiday.png" not in said, "named the file, which is not the thing that refused"

    def test_it_offers_both_of_the_usual_causes(self):
        said = smartgallery.explain_a_refused_write(PermissionError(errno.EACCES, "denied", "/x/y.png"))

        assert "read-only" in said
        assert "different user" in said
        assert "Nothing was changed" in said
