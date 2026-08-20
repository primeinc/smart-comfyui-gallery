"""Where a run of this application lives: one directory, wholly contained.

Everything a running instance owns -- the database, its model weights, any
cache it grows -- sits under a single home directory, so redirecting or
duplicating an entire run is one setting, and deleting the directory
deletes the run. Nothing is scattered into OS application-data folders:
a gallery you cannot pick up and move is the old application's disease.

Media is NOT under here. Libraries are `root` rows in the database --
any number of directories, anywhere, registered through the application
-- because the pictures belong to the person, not to a run of this app.

Resolution order, most explicit first:

  SMARTGALLERY_HOME     the whole run, relocated
  ~/.smartgallery       the default

Models default to `<home>/models` and may be pointed anywhere with
SMARTGALLERY_MODELS -- several runs may share one weights directory, but
nothing requires it.
"""

from __future__ import annotations

import os
import pathlib

#: The default home's name, under the user's own home directory.
DIRNAME = ".smartgallery"


def home() -> pathlib.Path:
    """The directory this run lives in, created on first ask."""
    chosen = os.environ.get("SMARTGALLERY_HOME")
    where = pathlib.Path(chosen) if chosen else pathlib.Path.home() / DIRNAME
    where.mkdir(parents=True, exist_ok=True)
    return where


def db_path() -> pathlib.Path:
    return home() / "gallery.db"


def models_dir() -> pathlib.Path:
    """Where model weights are read from. Shared if pointed somewhere
    shared; private to the run otherwise."""
    chosen = os.environ.get("SMARTGALLERY_MODELS")
    where = pathlib.Path(chosen) if chosen else home() / "models"
    where.mkdir(parents=True, exist_ok=True)
    return where
