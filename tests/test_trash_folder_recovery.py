"""Losing the trash folder must not stop deleting until a restart.

DELETE_TO is the setting that makes deletions recoverable. The gallery
makes a `SmartGallery` folder inside it once, at startup, and every delete
from then on assumes it is still there. People point DELETE_TO at a place
they empty from time to time -- that is what it is for -- so the folder
goes away.

Reproduced before the fix: with the folder removed, deleting raised

    FileNotFoundError [WinError 3] The system cannot find the path specified

the screen said only "Failed to delete N files", and every later delete
failed the same way until the process was restarted. Nothing was lost, but
nothing could be deleted either, and nothing on screen said why.

It is put back now. What the tests care about most is the case where it
cannot be: when DELETE_TO itself has gone, the delete is REFUSED and the
file stays.

That case was worse than it looked, and not the same for files and
folders. With DELETE_TO removed, deleting a FOLDER did not fail: shutil.move
falls back to copytree, copytree calls makedirs, so the whole trash tree
was rebuilt at that path, the folder was moved into it, and the gallery
reported success. Where DELETE_TO is a drive or a share, its mount point
is an ordinary empty directory while the thing is away -- so the media
landed on the filesystem underneath, and the moment the real one came back
it was mounted over the top and those files were gone. Files escaped that
only by accident, because copy2 cannot create parent directories and so
failed loudly.
"""

from __future__ import annotations

import os
import shutil

import pytest


@pytest.fixture()
def trash(smartgallery_app, tmp_path, monkeypatch):
    """A configured DELETE_TO, which the suite otherwise runs without."""
    delete_to = tmp_path / "recycle"
    folder = delete_to / "SmartGallery"
    folder.mkdir(parents=True)
    monkeypatch.setattr(smartgallery_app, "DELETE_TO", str(delete_to))
    monkeypatch.setattr(smartgallery_app, "TRASH_FOLDER", str(folder))
    return delete_to, folder


def _victim(tmp_path, name="doomed.png"):
    path = tmp_path / name
    path.write_bytes(b"not really a picture")
    return path


def test_a_file_goes_to_the_trash_when_it_is_all_there(smartgallery_app,
                                                       trash, tmp_path):
    """Control. Everything below removes something, so this pins the
    ordinary path and proves the fixture reaches the real code."""
    _delete_to, folder = trash
    victim = _victim(tmp_path)

    smartgallery_app.safe_delete_file(str(victim))

    assert not victim.exists()
    assert len(list(folder.iterdir())) == 1


def test_the_trash_folder_is_put_back_when_it_has_gone(smartgallery_app,
                                                       trash, tmp_path):
    """The bug: from here on every delete failed until a restart."""
    _delete_to, folder = trash
    shutil.rmtree(folder)
    victim = _victim(tmp_path)

    smartgallery_app.safe_delete_file(str(victim))

    assert folder.is_dir(), "the trash folder was not put back"
    assert not victim.exists()
    assert len(list(folder.iterdir())) == 1


def test_a_file_is_never_destroyed_when_the_trash_is_unreachable(
        smartgallery_app, trash, tmp_path):
    """The one that matters. DELETE_TO is set precisely so that deleting is
    recoverable; a delete that cannot be recovered must not happen.

    Falling back to os.remove here would satisfy "deleting works again"
    and quietly destroy the file."""
    delete_to, _folder = trash
    shutil.rmtree(delete_to)
    victim = _victim(tmp_path)

    with pytest.raises(OSError) as raised:
        smartgallery_app.safe_delete_file(str(victim))

    assert victim.exists(), "the file was destroyed with no trash to hold it"
    assert not delete_to.exists(), "a trash tree was built on a missing root"
    assert "nothing was deleted" in str(raised.value), raised.value


def test_a_folder_is_never_destroyed_when_the_trash_is_unreachable(
        smartgallery_app, trash, tmp_path):
    """The worst of the set, and it did not behave like the file case.

    shutil.move falls back to copytree when the rename fails, and copytree
    calls makedirs -- so with DELETE_TO gone, deleting a folder did not
    fail at all. It rebuilt the whole trash tree at that path, moved the
    folder in, and reported success. Verified against the shipped code.

    Where DELETE_TO is a drive or a share, its mount point is an ordinary
    empty directory while the thing is away. The media therefore lands on
    whatever filesystem sits underneath, and the moment the real one comes
    back it is mounted over the top and those files are unreachable, with
    the gallery having said the delete worked.

    Files behaved differently only by accident: copy2 cannot create parent
    directories, so they failed loudly instead."""
    delete_to, _folder = trash
    shutil.rmtree(delete_to)
    doomed = tmp_path / "a_folder"
    doomed.mkdir()
    (doomed / "keepme.png").write_bytes(b"x")

    with pytest.raises(OSError):
        smartgallery_app.safe_delete_tree(str(doomed))

    assert (doomed / "keepme.png").exists(), "an entire folder was destroyed"
    assert not delete_to.exists(), (
        "the trash tree was rebuilt on top of a missing DELETE_TO; the "
        "folder would be hidden the moment the real one came back")


def test_a_folder_goes_to_the_trash_once_it_is_back(smartgallery_app,
                                                    trash, tmp_path):
    """Folder deletion recovers through the same path as file deletion."""
    _delete_to, folder = trash
    shutil.rmtree(folder)
    doomed = tmp_path / "a_folder"
    doomed.mkdir()
    (doomed / "inside.png").write_bytes(b"x")

    smartgallery_app.safe_delete_tree(str(doomed))

    assert folder.is_dir()
    assert not doomed.exists()
    assert len(list(folder.iterdir())) == 1


def test_deleting_stays_permanent_when_no_trash_is_configured(
        smartgallery_app, tmp_path, monkeypatch):
    """Control against over-reach. Most installs set no DELETE_TO at all,
    and refusing there -- or demanding a folder that was never configured
    -- would break deleting for the majority to protect a minority."""
    monkeypatch.setattr(smartgallery_app, "DELETE_TO", None)
    monkeypatch.setattr(smartgallery_app, "TRASH_FOLDER", None)
    victim = _victim(tmp_path)

    smartgallery_app.safe_delete_file(str(victim))

    assert not victim.exists()


def test_the_helper_answers_rather_than_raising(smartgallery_app, trash):
    """It is consulted on every delete, so it has to return a verdict
    whatever the filesystem is doing."""
    delete_to, folder = trash

    assert smartgallery_app.ensure_trash_folder() is True
    shutil.rmtree(folder)
    assert smartgallery_app.ensure_trash_folder() is True
    shutil.rmtree(delete_to)
    assert smartgallery_app.ensure_trash_folder() is False
