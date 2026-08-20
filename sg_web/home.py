"""Where a run of this application lives: one directory, wholly contained.

Everything a running instance owns -- the database, its model weights, any
cache it grows -- sits under a single home directory, so redirecting or
duplicating an entire run is one argument, and deleting the directory
deletes the run. Nothing is scattered into OS application-data folders:
a gallery you cannot pick up and move is the old application's disease.

Media is NOT under here. Libraries are `root` rows in the database --
any number of directories, anywhere, registered through the application
-- because the pictures belong to the person, not to a run of this app.

The home directory is the one value that cannot be a setting row, because
it says where the database holding the rows is. It arrives as an explicit
argument (`python -m sg_web --home`, `build_app(home_dir=...)`) and
defaults to `~/.smartgallery`. Everything else configurable lives in the
`setting` table (db/settings.py), changeable while the app runs.
"""

from __future__ import annotations

import os
import pathlib

#: The default home's name, under the user's own home directory.
DIRNAME = ".smartgallery"


def home(chosen: str | os.PathLike[str] | None = None) -> pathlib.Path:
    """The directory this run lives in, created on first ask."""
    where = pathlib.Path(chosen) if chosen else pathlib.Path.home() / DIRNAME
    where.mkdir(parents=True, exist_ok=True)
    return where


def db_path(base: pathlib.Path) -> pathlib.Path:
    return base / "gallery.db"


def thumbs_dir(base: pathlib.Path) -> pathlib.Path:
    """The thumbnail cache, keyed on content -- safe to delete whole."""
    where = base / "thumbs"
    where.mkdir(parents=True, exist_ok=True)
    return where


def models_dir(base: pathlib.Path, chosen: str = "") -> pathlib.Path:
    """Where model weights are read from: the `models_dir` setting when
    one is set, else `<home>/models`. Several runs may point one place,
    but nothing requires it."""
    where = pathlib.Path(chosen) if chosen else base / "models"
    where.mkdir(parents=True, exist_ok=True)
    return where
