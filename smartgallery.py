# SmartGallery DAM for ComfyUI
# Author: Biagio Maffettone © 2025-2026 — Free to use/modify with credit. Provided "as is". See license on GitHub.
#
# Version: 2.22 - August 12, 2026
# Check the GitHub repository for updates, bug fixes, and contributions.
#
# Contact: biagiomaf@gmail.com
# GitHub: https://github.com/biagiomaf/smart-comfyui-gallery

import argparse
import base64
import colorsys
import concurrent.futures
import contextlib
import difflib
import errno
import glob
import hashlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import socket
import sqlite3
import string
import struct
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from functools import wraps
from typing import Any

import cv2
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
from PIL import Image, ImageDraw, ImageFont, ImageSequence
from tqdm import tqdm

# Flask's own guidance for a generic handler (docs/errorhandling.rst,
# "Generic Exception Handlers"): it fires for things you did not cause,
# such as routing's own 404 and 405, so "be sure to craft your handler
# carefully so you don't lose information about the HTTP error". That is
# why HTTPException is handed straight back untouched.
from werkzeug.exceptions import HTTPException, InternalServerError

import metaparse
import sg_auth
import smartgallery_ai
import smartgallery_ai.schema
from metaparse import typed as metaparse_typed
from omniquery.engine import OmniQueryEngine
from omniquery.sqlexec import run_readonly_select
from omniquery.validation import AuthContext
from smartgallery_ai import service as ai_dam_service
from smartgallery_ai.worker import AIWorker

# Same convention as smartgallery_ai.*, which has had one of these per
# module all along; the monolith was the one place still throwing every
# traceback away. The console messages below are unchanged -- they are what
# the operator reads, and several are asserted on. What this adds is the
# traceback behind them, at DEBUG, so a failure that today prints one line
# can be opened up without editing the source to find out what raised.
_logger = logging.getLogger(__name__)

try:
    from waitress import serve

    WAITRESS_AVAILABLE = True
except ImportError:
    WAITRESS_AVAILABLE = False

# tkinter is imported for the dialog helpers further down but never enabled:
# the gallery has to behave the same in a container, over SSH and on a
# desktop, so the console path is the only one that runs.
try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    pass
TKINTER_AVAILABLE = False


def make_output_carry_any_filename(streams=None):
    """Let the gallery print the names of the files it actually indexes.

    It reports damaged files, unreachable folders and offline mounts by
    name, and those names are frequently not written in English -- the
    whole point of the unicode handling elsewhere. Whether that name can
    be printed depends on where the output is going. Attached to a console
    Python writes wide characters and anything prints; redirected to a
    file or a pipe -- a launcher keeping a log, ComfyUI starting the
    gallery itself, anything reading its output -- the encoding is the
    machine's code page instead, which on most Windows installs is cp1252.

    Printing a Japanese filename to cp1252 raises UnicodeEncodeError, and
    it raises inside the message rather than inside the work: the scan is
    built so one damaged file costs that file alone, and then the line
    saying so killed the process. Measured on a library holding one good
    picture and one damaged one, both named in CJK, with the output
    redirected:

        STARTUP DIED: 'charmap' codec can't encode characters in
        position 38-42: character maps to <undefined>

    So the streams are asked for UTF-8, and asked to substitute rather
    than raise if they cannot give it. A name that arrives unreadable is a
    bad line of output; a name that raises is a gallery that will not
    start.

    This sits directly under the imports and above everything else. None
    of the modules above it writes anything while loading, so it still
    runs before the first line of output the gallery produces.
    """
    for stream in streams if streams is not None else (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Not every stream is reconfigurable -- a captured one, a
            # replaced one, a plain file object. Nothing here is worth
            # failing over; the point is to try before anything prints.
            _logger.debug("ignored a failure in make_output_carry_any_filename", exc_info=True)


make_output_carry_any_filename()


# ============================================================================
# CONFIGURATION GUIDE - PLEASE READ BEFORE SETTING UP
# ============================================================================
#
# CONFIGURATION PRIORITY:
# All settings below first check for environment variables. If an environment
# variable is set, its value will be used automatically.
# If you have NOT set environment variables, you only need to modify the
# values AFTER the comma in the os.environ.get() statements.
#
# Example: os.environ.get('BASE_OUTPUT_PATH', 'C:/your/path/here')
#          - If BASE_OUTPUT_PATH environment variable exists → it will be used
#          - If NOT → the value 'C:/your/path/here' will be used instead
#          - ONLY CHANGE 'C:/your/path/here' if you haven't set environment variables
#
# ----------------------------------------------------------------------------
# HOW TO SET ENVIRONMENT VARIABLES (before running python smartgallery.py):
# ----------------------------------------------------------------------------
#
# IMPORTANT: If your paths contain SPACES, you MUST use quotes around them!
#            Replace the example paths below with YOUR actual paths!
#
# Windows (Command Prompt):
#   call venv\Scripts\activate.bat
#   set "BASE_OUTPUT_PATH=C:/ComfyUI/output"
#   set BASE_INPUT_PATH=C:/sm/Data/Packages/ComfyUI/input
#   set "BASE_SMARTGALLERY_PATH=C:/ComfyUI/output"
#   set "FFPROBE_MANUAL_PATH=C:/ffmpeg/bin/ffprobe.exe"
#   set SERVER_PORT=8189
#   set THUMBNAIL_WIDTH=300
#   set WEBP_ANIMATED_FPS=16.0
#   set PAGE_SIZE=100
#   set BATCH_SIZE=500
#   set ENABLE_AI_SEARCH=false
#   REM Leave MAX_PARALLEL_WORKERS empty to use all CPU cores (recommended)
#   set "MAX_PARALLEL_WORKERS="
#   python smartgallery.py
#
# Windows (PowerShell):
#   venv\Scripts\Activate.ps1
#   $env:BASE_OUTPUT_PATH="C:/ComfyUI/output"
#   $env:BASE_INPUT_PATH="C:/sm/Data/Packages/ComfyUI/input"
#   $env:BASE_SMARTGALLERY_PATH="C:/ComfyUI/output"
#   $env:FFPROBE_MANUAL_PATH="C:/ffmpeg/bin/ffprobe.exe"
#   $env:SERVER_PORT="8189"
#   $env:THUMBNAIL_WIDTH="300"
#   $env:WEBP_ANIMATED_FPS="16.0"
#   $env:PAGE_SIZE="100"
#   $env:BATCH_SIZE="500"
#   $env:ENABLE_AI_SEARCH="false"
#   # Leave MAX_PARALLEL_WORKERS empty to use all CPU cores (recommended)
#   $env:MAX_PARALLEL_WORKERS=""
#   python smartgallery.py
#
# Linux/Mac (bash/zsh):
#   source venv/bin/activate
#   export BASE_OUTPUT_PATH="$HOME/ComfyUI/output"
#   export BASE_INPUT_PATH="/path/to/ComfyUI/input"
#   export BASE_SMARTGALLERY_PATH="$HOME/ComfyUI/output"
#   export FFPROBE_MANUAL_PATH="/usr/bin/ffprobe"
#   export DELETE_TO="/path/to/trash" # Optional, set to disable permanent delete
#   export SERVER_PORT=8189
#   export THUMBNAIL_WIDTH=300
#   export WEBP_ANIMATED_FPS=16.0
#   export PAGE_SIZE=100
#   export BATCH_SIZE=500
#   export ENABLE_AI_SEARCH=false
#   # Leave MAX_PARALLEL_WORKERS empty to use all CPU cores (recommended)
#   export MAX_PARALLEL_WORKERS=""
#   python smartgallery.py
#
#
# IMPORTANT NOTES:
# - Write paths however you like. Backslashes or forward slashes, with or
#   without a trailing slash, a leading ~, or the quotes Explorer's "Copy
#   as path" wraps around them -- normalize_configured_path handles it.
# - Use QUOTES around paths containing spaces to avoid errors.
# - Replace example paths (C:/ComfyUI/, $HOME/ComfyUI/) with YOUR actual paths!
# - Set MAX_PARALLEL_WORKERS="" (empty string) to use all available CPU cores.
#   Set it to a number (e.g., 4) to limit CPU usage.
# - It is strongly recommended to have ffmpeg installed,
#   since some features depend on it.
#
# ============================================================================


# ============================================================================
# USER CONFIGURATION
# ============================================================================
# Adjust the parameters below to customize the gallery.
# Remember: environment variables take priority over these default values.
# ============================================================================


# --- CONSOLE STYLING ---
# Defined before the configuration block below, which uses it: the
# DELETE_TO validation prints coloured diagnostics at import time, and
# with this class further down the file every one of those paths raised
# NameError instead -- including the ordinary first run, which has to
# create the trash folder.
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


def env_or(name, default):
    """Environment value for `name`, falling back to `default` when the
    variable is unset OR set to nothing.

    A blank is a fallback, not a value. `set "BASE_OUTPUT_PATH="` in a .bat
    (and `export X=` in a shell, and an empty field in a Docker/Unraid
    template) defines the variable as an empty string, which
    `os.environ.get(name, default)` returns AS the value -- so clearing a
    path in the launcher template, the obvious way to say "use the
    default", silently set the gallery root to "" and scattered cache
    folders into the working directory instead.
    """
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def normalize_configured_path(raw):
    """One spelling for a path, whatever the person typed.

    Every way of writing the same folder used to be a different setting
    to look at, and the docs carried a rule to work around it: "Use
    forward slashes even on Windows." That is the wrong way round -- a
    Windows user copying a path out of Explorer gets backslashes, and
    Explorer's own "Copy as path" wraps it in quotes. Neither should have
    to be corrected by hand.

    So all of these become the same thing:

        C:/ComfyUI/output          C:\\ComfyUI\\output
        C:/ComfyUI/output/         C:\\ComfyUI\\output\\
        "C:\\ComfyUI\\output"        C:/ComfyUI//output
        C:/ComfyUI/sub/../output   ~/ComfyUI/output

    This lands on exactly the string the folder scan already computed for
    itself (os.path.normpath then forward slashes), which is what makes
    it safe: a file is identified by its path, so a canonical form that
    differed by one character would re-identify every file in the library
    and take its ratings with it. Checked against every form above.

    Case is deliberately left alone. On Windows two spellings open the
    same folder, but the rows were written with whichever one was
    configured, so folding it here would re-identify the library of
    anyone whose setting does not match the disk -- the exact damage this
    is meant to avoid. looks_like_a_renamed_root covers that instead.
    """
    text = str(raw or "").strip()
    # Explorer's "Copy as path" hands you the path inside double quotes,
    # and a .bat written as set "X=%~dp0" can leave one behind.
    while len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if not text:
        return ""
    # normpath('') is '.', so the emptiness check above is load-bearing.
    return os.path.normpath(os.path.expanduser(text)).replace("\\", "/")


def env_path(name, default):
    """A path setting, read like any other and then written one way."""
    return normalize_configured_path(env_or(name, default))


def env_num(name, default, cast=int, minimum=None):
    """Numeric environment value, or `default` when unset, blank,
    unparseable, or below `minimum`.

    Most of these are read at module scope, so `int(os.environ.get(...))`
    turned a blank or fat-fingered value into a ValueError during import:
    the gallery refused to start, with a traceback that never named the
    variable at fault. A bad number should cost you that setting, not the
    whole application.

    `minimum` extends that to numbers which parse perfectly and then break
    something further off, where the traceback names Python rather than the
    setting. A zero is the easy slip and every one of these was fatal:

        BATCH_SIZE=0            range() arg 3 must not be zero -- no scan
        MAX_PARALLEL_WORKERS=0  max_workers must be greater than 0 -- no scan
        THUMBNAIL_WIDTH=0       ZeroDivisionError on every thumbnail
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = cast(raw.strip())
        if minimum is not None and value < minimum:
            print(f"WARNING: {name}={raw!r} is below the minimum of {minimum}; using {default}.")
            return default
        return value
    except (TypeError, ValueError):
        print(f"WARNING: {name}={raw!r} is not a valid number; using {default}.")
        return default


def env_flag(name, default=False):
    """Boolean environment flag, or `default` when unset or blank.

    `"1"/"true"/"yes"/"on"` (any case) are True; `"0"/"false"/"no"/"off"`
    are False. Comparing `os.environ.get(name, "true").lower() == "true"`
    made a BLANK read as False, so clearing GENERATE_THUMBNAILS in the
    launcher silently turned thumbnails off rather than restoring the
    documented default. An unrecognized word says so and keeps the default
    instead of quietly meaning False.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    token = raw.strip().lower()
    if token in ("1", "true", "yes", "on"):
        return True
    if token in ("0", "false", "no", "off"):
        return False
    print(f"WARNING: {name}={raw!r} is not a yes/no value; using {default}.")
    return default


# Path to the ComfyUI 'output' folder.
# Common locations:
#   Windows: C:/ComfyUI/output or C:/Users/YourName/ComfyUI/output
#   Linux/Mac: /home/username/ComfyUI/output or ~/ComfyUI/output
BASE_OUTPUT_PATH = env_path("BASE_OUTPUT_PATH", "C:/ComfyUI/output")

# Path to the ComfyUI 'input' folder
BASE_INPUT_PATH = env_path("BASE_INPUT_PATH", "C:/ComfyUI/input")

# --- Granular Paths for Advanced Setups (Docker, Stability Matrix, extra_model_paths.yaml) ---
# If not set, they fallback to the standard ComfyUI relative structure
BASE_MODELS_PATH = env_path(
    "BASE_MODELS_PATH", os.path.join(os.path.dirname(os.path.normpath(BASE_OUTPUT_PATH)), "models")
)
LORAS_PATH = env_path("LORAS_PATH", os.path.join(BASE_MODELS_PATH, "loras"))
CHECKPOINTS_PATH = env_path("CHECKPOINTS_PATH", os.path.join(BASE_MODELS_PATH, "checkpoints"))
UNET_PATH = env_path("UNET_PATH", os.path.join(BASE_MODELS_PATH, "unet"))


# Path for service folders (database, cache, zip files).
# If not specified, the ComfyUI output path will be used.
# These sub-folders won't appear in the gallery.
# Change this if you want the cache stored separately for better performance
# or to keep system files separate from gallery content.
# Leave as-is if you are unsure.
BASE_SMARTGALLERY_PATH = env_path("BASE_SMARTGALLERY_PATH", BASE_OUTPUT_PATH)

# Path to ffprobe executable (part of ffmpeg).
# Common locations:
#   Windows: C:/ffmpeg/bin/ffprobe.exe or C:/Program Files/ffmpeg/bin/ffprobe.exe
#   Linux: /usr/bin/ffprobe or /usr/local/bin/ffprobe
#   Mac: /usr/local/bin/ffprobe or /opt/homebrew/bin/ffprobe
# Required for extracting workflows from .mp4 files.
# NOTE: A full ffmpeg installation is highly recommended.
FFPROBE_MANUAL_PATH = env_path("FFPROBE_MANUAL_PATH", "C:/ffmpeg/bin/ffprobe.exe")

# Port on which the gallery web server will run.
# Must be different from the ComfyUI port (usually 8188).
# The gallery does not require ComfyUI to be running; it works independently.
SERVER_PORT = env_num("SERVER_PORT", 8189, minimum=1)

# Width (in pixels) of the generated thumbnails.
THUMBNAIL_WIDTH = env_num("THUMBNAIL_WIDTH", 300, minimum=1)

# Assumed frame rate for animated WebP files.
# Many tools, including ComfyUI, generate WebP animations at ~16 FPS.
# Adjust this value if your WebPs use a different frame rate,
# so that animation durations are calculated correctly.
WEBP_ANIMATED_FPS = env_num("WEBP_ANIMATED_FPS", 16.0, float, minimum=0.1)

# Maximum number of files to load initially before showing a "Load more" button.
# Use a very large number (e.g., 9999999) for "infinite" loading.
PAGE_SIZE = env_num("PAGE_SIZE", 100, minimum=1)

# Names of special folders (e.g., 'video', 'audio').
# These folders will appear in the menu only if they exist inside BASE_OUTPUT_PATH.
# Leave as-is if unsure.
SPECIAL_FOLDERS = ["video", "audio"]

# Number of files to process at once during database sync.
# Higher values use more memory but may be faster.
# Lower this if you run out of memory.
BATCH_SIZE = env_num("BATCH_SIZE", 500, minimum=1)

# Threshold (in MB) above which videos will be streamed (transcoded)
# instead of loaded natively in the gallery grid preview.
# Default: 50 MB. Set to 0 to force streaming for all supported videos.
STREAM_THRESHOLD_MB = env_num("STREAM_THRESHOLD_MB", 20, minimum=0)
STREAM_THRESHOLD_BYTES = STREAM_THRESHOLD_MB * 1024 * 1024

# Number of parallel processes to use for thumbnail and metadata generation.
# - None or empty string: use all available CPU cores (fastest, recommended)
# - 1: disable parallel processing (slowest, like in previous versions)
# - Specific number (e.g., 4): limit CPU usage on multi-core machines
MAX_PARALLEL_WORKERS = env_num("MAX_PARALLEL_WORKERS", None, minimum=1)
if MAX_PARALLEL_WORKERS is not None:
    pass  # explicit worker count honored as given
# OS-Specific Safety Defaults
# macOS (darwin) often crashes with BrokenProcessPool or runs out of file descriptors
# when maxing out Apple Silicon cores on massive galleries (>3000 files).
# Defaulting to 4 provides excellent speed while maintaining absolute stability.
elif sys.platform == "darwin":
    MAX_PARALLEL_WORKERS = 4
else:
    MAX_PARALLEL_WORKERS = None

# Below this many files a scan stays in this process.
#
# The pool's workers run process_single_file, which lives in this module, so
# each of them has to import it before doing any work. That import is around
# a second, and it is paid whether the batch is one file or ten thousand --
# measured at 2.96s per pooled call before the AI stack was made lazy and
# 1.01s after, against 0.10s for a pool whose task needs no import at all.
# Scanning a handful of newly dropped files is the common case and was
# spending all of its time starting interpreters.
#
# The sequential path below is not a fallback bolted on for this: it already
# existed for a broken pool, and it is the same process_single_file call.
# What is given up under the threshold is crash isolation -- a segfaulting
# image takes the process rather than one worker -- which is why the bound is
# low rather than clever.
PARALLEL_SCAN_MIN_FILES = env_num("PARALLEL_SCAN_MIN_FILES", 12, minimum=1)


class InProcessExecutor:
    """Runs each submission immediately, here, with the executor surface.

    Same `submit` -> Future contract the pool offers, so a small batch can
    skip the pool without every caller growing a second code path.
    """

    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # the pool reports failures the same way
            _logger.debug("handled a failure in submit", exc_info=True)
            future.set_exception(exc)
        return future


def scan_executor(file_count):
    """The process pool for a batch worth starting one, this process below.

    Every scan path went through the pool at any size. The work each worker
    runs lives in this module, so a worker must import the module before it
    can touch a picture -- and that import is the dominant cost of a small
    scan, not the pictures.
    """
    if file_count >= PARALLEL_SCAN_MIN_FILES:
        return concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PARALLEL_WORKERS)
    return InProcessExecutor()


# Flask secret key
# You can set it in the environment variable SECRET_KEY
# If not set, it will be generated randomly
SECRET_KEY = env_or("SECRET_KEY", secrets.token_hex(32))

# Maximum number of items allowed in the "Prefix" dropdown to prevent UI lag.
MAX_PREFIX_DROPDOWN_ITEMS = 100


# Optional path where deleted files will be moved instead of being permanently deleted.
# If set, files will be moved to DELETE_TO/SmartGallery/<timestamp>_<filename>
# If not set (None or empty string), files will be permanently deleted as before.
# The path MUST exist and be writable, or the application will exit with an error.
# Example: /path/to/trash or C:/Trash
DELETE_TO = env_or("DELETE_TO", None)
# Normalised like the rest, but only when there is one: unset stays None,
# which is what the startup banner distinguishes from a configured trash.
if DELETE_TO:
    DELETE_TO = normalize_configured_path(DELETE_TO)
if DELETE_TO and DELETE_TO.strip():
    DELETE_TO = DELETE_TO.strip()
    TRASH_FOLDER = os.path.join(DELETE_TO, "SmartGallery")

    # Validate that DELETE_TO path exists
    if not os.path.exists(DELETE_TO):
        print(f"{Colors.RED}{Colors.BOLD}CRITICAL ERROR: DELETE_TO path does not exist: {DELETE_TO}{Colors.RESET}")
        print(f"{Colors.RED}Please create the directory or unset the DELETE_TO environment variable.{Colors.RESET}")
        sys.exit(1)

    # Validate that DELETE_TO is writable
    if not os.access(DELETE_TO, os.W_OK):
        print(f"{Colors.RED}{Colors.BOLD}CRITICAL ERROR: DELETE_TO path is not writable: {DELETE_TO}{Colors.RESET}")
        print(f"{Colors.RED}Please check permissions or unset the DELETE_TO environment variable.{Colors.RESET}")
        sys.exit(1)

    # Validate that SmartGallery subfolder exists or can be created
    if not os.path.exists(TRASH_FOLDER):
        try:
            os.makedirs(TRASH_FOLDER)
            print(f"{Colors.GREEN}Created trash folder: {TRASH_FOLDER}{Colors.RESET}")
        except OSError as e:
            print(f"{Colors.RED}{Colors.BOLD}CRITICAL ERROR: Cannot create trash folder: {TRASH_FOLDER}{Colors.RESET}")
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
            sys.exit(1)
else:
    DELETE_TO = None
    TRASH_FOLDER = None

# ============================================================================
# WORKFLOW PROMPT EXTRACTION SETTINGS
# ============================================================================
# List of specific text phrases to EXCLUDE from the 'Prompt Keywords' search index.
# Some custom nodes (e.g., Wan2.1, text boxes, primitives) come with long default
# example prompts or placeholder text that gets saved in the workflow metadata
# even if not actually used in the generation.
# Add those specific strings here to prevent them from cluttering your search results.
WORKFLOW_PROMPT_BLACKLIST = {
    (
        "The white dragon warrior stands still, eyes full of determination and "
        "strength. The camera slowly moves closer or circles around the warrior, "
        "highlighting the powerful presence and heroic spirit of the character."
    ),
    "undefined",
    "null",
    "None",
}

# ============================================================================
# RUNTIME FLAGS (Set via command line arguments)
# ============================================================================
# Will be populated in __main__
IS_EXHIBITION_MODE = False

# ============================================================================
# AI SEARCH CONFIGURATION (FUTURE FEATURE)
# ============================================================================
# Enable or disable the AI Search UI features.
#
# IMPORTANT:
# The SmartGallery AI Service (Optional) required for this feature
# is currently UNDER DEVELOPMENT and HAS NOT BEEN RELEASED yet.
#
# SmartGallery works fully out-of-the-box without any AI components.
#
# Advanced features such as AI Search will be provided by a separate,
# optional service that can be installed via Docker or in a separated dedicated Python virtual environment.
#
# PLEASE KEEP THIS SETTING DISABLED (default).
# Do NOT enable this option unless the AI Service has been officially
# released and correctly installed alongside SmartGallery.
#
# Check the GitHub repository for official announcements and
# installation instructions regarding the optional AI Service.
#
#   Windows:     set ENABLE_AI_SEARCH=false
#   Linux / Mac: export ENABLE_AI_SEARCH=false
#   Docker:      -e ENABLE_AI_SEARCH=false
#
ENABLE_AI_SEARCH = env_flag("ENABLE_AI_SEARCH", False)
GENERATE_WAVEFORMS = env_flag("GENERATE_WAVEFORMS", False)
# Default for server-side thumbnail generation; a runtime toggle stored in
# the DB (Tools menu / POST /galleryout/api/site_settings) overrides it.
GENERATE_THUMBNAILS = env_flag("GENERATE_THUMBNAILS", True)
COMFYUI_SERVER_URL = env_or("COMFYUI_SERVER_URL", "http://127.0.0.1:8188")


# ============================================================================
# END OF USER CONFIGURATION
# ============================================================================


# --- CACHE AND FOLDER NAMES ---
THUMBNAIL_CACHE_FOLDER_NAME = ".thumbnails_cache"
SQLITE_CACHE_FOLDER_NAME = ".sqlite_cache"
DATABASE_FILENAME = "gallery_cache.sqlite"
ZIP_CACHE_FOLDER_NAME = ".zip_downloads"
FFMPEG_FOLDER_NAME = ".ffmpeg"

# What a comment may be. Generous for a remark about a picture -- roughly
# two pages -- and small enough that no one visitor can fill the database
# or a reader's browser. Long writing about a collection belongs in the
# collection note, which is a file and is meant for it.
COMMENT_MAX_CHARS = 4000
COMMENT_AUTHOR_MAX_CHARS = 80


def parse_rating(value):
    """(rating, error): a whole 1..5, None to clear, or a reason to refuse.

    The check was `1 <= rating <= 5` against whatever JSON supplied, which
    is two faults at once on a route a visitor reaches.

    Anything not a number raised. Measured -- a rating of "3", "abc", [3]
    or {} answered 500 with an HTML page, so the caller's res.json() got
    nothing it could read:

        TypeError: '<=' not supported between instances of 'int' and 'str'

    And anything that happened to compare was taken. SQLite stores what it
    is handed, so a fraction went into an INTEGER column and passed the
    CHECK(rating >= 1 AND rating <= 5):

        alice   stored 3.5   sqlite type real
        bob     stored 2.25  sqlite type real
        carol   stored 1     sqlite type integer   (sent true)
        average everyone sees: 2.25

    A boolean is not a rating either; True == 1 in Python, so `true`
    quietly became one star, and isinstance(True, int) is why bool is
    turned away before the number check rather than after it.

    3.0 IS accepted as 3: a JSON client that has only one number type is
    not making a mistake, and this is the same principle as taking a path
    in whatever form somebody writes it.
    """
    if value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, "A rating has to be a whole number from 1 to 5."
    if isinstance(value, float) and (not math.isfinite(value) or value != int(value)):
        return None, "A rating has to be a whole number from 1 to 5."
    number = int(value)
    if number == 0:
        return None, None  # clearing a rating
    if not 1 <= number <= 5:
        return None, "A rating has to be from 1 to 5."
    return number, None


def comment_length_error(text, what="comment"):
    """None when it fits, otherwise the sentence to say back.

    One rule in one place because the first version of this was not.
    Posting a comment was capped and EDITING one was not, so the limit
    took one extra request to walk around: post something short, then
    edit it to anything at all. Measured against that build --

        post a short comment          -> 200
        post one over the limit       -> 400  (the cap works)
        EDIT it to 40,000,000 chars   -> 200  stored 40000000 chars

    -- which is the whole cap, undone, by the sibling route two hundred
    lines further down. Every path that writes comment_text asks this.
    """
    if len(text) > COMMENT_MAX_CHARS:
        return f"That {what} is {len(text):,} characters. The limit is {COMMENT_MAX_CHARS:,}."
    return None


AI_MODELS_FOLDER_NAME = ".AImodels"
ENABLE_DAM_MODE = True

# --- APP INFO ---
APP_VERSION = "2.22"
APP_VERSION_DATE = "August 12, 2026"
GITHUB_REPO_URL = "https://github.com/biagiomaf/smart-comfyui-gallery"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/biagiomaf/smart-comfyui-gallery/main/smartgallery.py"

# ============================================================================
# RUNTIME FLAGS (Set via command line arguments)
# ============================================================================
_parser = argparse.ArgumentParser(description="SmartGallery DAM for ComfyUI")
_parser.add_argument("--exhibition", action="store_true", help="Start in Exhibition Mode")
_parser.add_argument(
    "--enable-guest-login", action="store_true", help="Allow anyone to login as Guest without password"
)
_parser.add_argument("--port", type=int, default=None, help="Override server port")
_parser.add_argument("--admin-pass", type=str, help="Set or reset the Admin password")
_parser.add_argument("--force-login", action="store_true", help="Force login for the standard index.html interface")
_parser.add_argument("--blind-rating", action="store_true", help="Hide global average ratings to prevent user bias")


def refuse_unrecognised_flags(unknown, argv0, parser=None):
    """Stop rather than start when a flag was misspelt.

    A mistyped flag used to disappear. parse_known_args collects what it
    does not recognise and the leftovers were dropped without a word, so
    `--forcelogin` or `--force_login` started a gallery with no login at all
    while the operator believed it was shut. The same silence applied to
    --exhibition and --blind-rating: the mode simply did not happen.

    Only when this file is the program being run. Imported -- by the tests,
    or by anything embedding the gallery -- argv belongs to the host, and
    its arguments are none of our business.

    Raises SystemExit(2) when it refuses, and returns the stray flags it
    refused (empty when there is nothing to complain about). A function
    rather than a block at import so it can be asked the question directly:
    reading it any other way costs a whole interpreter per case.
    """
    parser = parser if parser is not None else _parser
    launched_directly = os.path.basename(str(argv0 or "")).lower() in ("smartgallery.py", "smartgallery")
    stray = [a for a in unknown if a.startswith("-")]
    if not (stray and launched_directly):
        return []

    # argparse exposes no public accessor for the flags it knows.
    known = sorted({s for a in parser._actions for s in a.option_strings})
    print(f"{Colors.RED}{Colors.BOLD}Unrecognised option(s): {' '.join(stray)}{Colors.RESET}")
    for flag in stray:
        near = difflib.get_close_matches(flag, known, n=1, cutoff=0.6)
        if near:
            print(f"  {flag}  ->  did you mean {Colors.YELLOW}{near[0]}{Colors.RESET}?")
    print(f"\nValid options: {', '.join(known)}")
    print(
        f"{Colors.YELLOW}Refusing to start: a misspelt --force-login or "
        f"--exhibition would leave the gallery open to anyone who can reach "
        f"it.{Colors.RESET}"
    )
    sys.exit(2)


_args, _unknown = _parser.parse_known_args()
refuse_unrecognised_flags(_unknown, sys.argv[0])

IS_EXHIBITION_MODE = _args.exhibition
if _args.port:
    SERVER_PORT = _args.port
ENABLE_GUEST_LOGIN = _args.enable_guest_login
FORCE_LOGIN = _args.force_login
BLIND_RATING = _args.blind_rating

ADMIN_MIN_PASSWORD_LENGTH = 8


def derive_login_policy(admin_pass, exhibition, force_login):
    """What a password and the two mode flags mean together.

    Three rules that only ever made sense as one decision:

      * a configured admin password enforces login by itself -- nobody
        should have to remember --force-login as well
      * a restricted mode asked for WITHOUT a password is a lockdown, not
        an open gallery: the operator wanted the door shut and there is no
        key, so the gallery refuses rather than serving anonymously
      * a password under ADMIN_MIN_PASSWORD_LENGTH is flagged, not accepted
        quietly

    Returns (force_login, config_missing, password_too_short). A function
    rather than three statements at import so the combinations can be asked
    for directly; reading them any other way costs an interpreter apiece.
    """
    force_login = bool(force_login or admin_pass)
    config_missing = bool((exhibition or force_login) and not admin_pass)
    too_short = bool(admin_pass and len(admin_pass) < ADMIN_MIN_PASSWORD_LENGTH)
    return force_login, config_missing, too_short


# Priority: CLI Param > Environment Variable
ADMIN_PASS_INPUT = _args.admin_pass or env_or("ADMIN_PASSWORD", None)

FORCE_LOGIN, ADMIN_CONFIG_MISSING, ADMIN_PASS_TOO_SHORT = derive_login_policy(
    ADMIN_PASS_INPUT, IS_EXHIBITION_MODE, FORCE_LOGIN
)


# --- HELPER FUNCTIONS (DEFINED FIRST) ---
def path_to_key(relative_path):
    if not relative_path:
        return "_root_"
    return base64.urlsafe_b64encode(relative_path.replace(os.sep, "/").encode()).decode()


def key_to_path(key):
    if key == "_root_":
        return ""
    try:
        return base64.urlsafe_b64decode(key.encode()).decode().replace("/", os.sep)
    except Exception:
        _logger.debug("handled a failure in key_to_path", exc_info=True)
        return None


# --- DERIVED SETTINGS ---
DB_SCHEMA_VERSION = 27
THUMBNAIL_CACHE_DIR = os.path.join(BASE_SMARTGALLERY_PATH, THUMBNAIL_CACHE_FOLDER_NAME)
SQLITE_CACHE_DIR = os.path.join(BASE_SMARTGALLERY_PATH, SQLITE_CACHE_FOLDER_NAME)
# Directory for metadata-stripped files (for client delivery)
CLEAN_CACHE_FOLDER_NAME = ".clean_cache"
CLEAN_CACHE_DIR = os.path.join(BASE_SMARTGALLERY_PATH, CLEAN_CACHE_FOLDER_NAME)
DATABASE_FILE = os.path.join(SQLITE_CACHE_DIR, DATABASE_FILENAME)
ENCRYPTION_KEY_FILE = os.path.join(SQLITE_CACHE_DIR, "system.key")
# Written beside the cache, never inside it, so it survives whatever
# removed the database and can tell a first run from a lost library.
LIBRARY_MARKER_FILE = os.path.join(BASE_SMARTGALLERY_PATH, ".smartgallery_library")
ZIP_CACHE_DIR = os.path.join(BASE_SMARTGALLERY_PATH, ZIP_CACHE_FOLDER_NAME)
IMPORTED_WORKFLOWS_FOLDER_NAME = ".imported_workflows"
IMPORTED_WORKFLOWS_DIR = os.path.join(BASE_SMARTGALLERY_PATH, IMPORTED_WORKFLOWS_FOLDER_NAME)
PROTECTED_FOLDER_KEYS = {path_to_key(f) for f in SPECIAL_FOLDERS}
PROTECTED_FOLDER_KEYS.add("_root_")

# ============================================================================
# AI DAM LAYER CONFIGURATION (NON-BLOCKING, OPT-OUT)
# ============================================================================
# Derived-AI layer (hashing/embeddings/faces/review). ENABLED by default:
# the background worker starts, auto-downloads any missing model weights it
# can actually run (asynchronously; see smartgallery_ai/provision.py), and
# the AI panel appears in the lightbox. No heavy model runtime (e.g. torch)
# is ever imported on the normal browsing path, and every capability whose
# runtime or weights are absent degrades to an actionable message.
# Opt out entirely:            ENABLE_AI_DAM=false
# Opt out of downloads only:   AI_DAM_AUTO_PROVISION=false
AI_CONFIG = smartgallery_ai.AIConfig.from_env(BASE_SMARTGALLERY_PATH, DATABASE_FILE)
# Env-reading consumers (the omniquery model defaults) must resolve the
# SAME models directory provisioning writes to.
# The config anchors it at the gallery root; the process CWD may be
# elsewhere (ComfyUI plugin deployments), so publish the resolved path --
# a user's own AI_DAM_MODELS_DIR still wins.
os.environ.setdefault("AI_DAM_MODELS_DIR", os.path.abspath(AI_CONFIG.models_dir))


def get_omniquery_dictionary(reset=False):
    omni_dir = os.path.join(BASE_SMARTGALLERY_PATH, ".omniquery")
    dict_path = os.path.join(omni_dir, "omniquery_template.txt")

    # --- FACTORY PROMPT BASE ---
    # Edit this variable to permanently alter the default prompt structure in the codebase.
    BA_OU_PA = BASE_OUTPUT_PATH.replace("\\", "/")
    # The backslashes below are line continuations inside the string: they
    # emit nothing, so the prompt the model receives is unchanged while the
    # source stays readable at a normal width.
    FACTORY_PROMPT_BASE = f"""You are an expert SQLite database administrator. \
I will give you a natural language request, and you must return a valid SQLite query.

DATABASE SCHEMA:
- files: id(TEXT), path(TEXT), mtime(REAL unix_ts), last_scanned(REAL unix_ts), \
name(TEXT), type(TEXT: video/image/animated_image/audio), size(INTEGER), \
dimensions(TEXT: are the pixels example '640x480'), is_favorite(INTEGER: 1/0), \
has_workflow(INTEGER: 1/0), workflow_files(TEXT), workflow_prompt(TEXT), \
duration(TEXT: example 03:19 minutes:seconds)
- collections: id(INTEGER), name(TEXT), type(TEXT:'user_album'/'system_flag'), \
is_public(INTEGER 1/0), shared_users(TEXT: comma-separated user_ids), \
parent_id(INTEGER: Unary relationship identifier for nested sub-collections using id key)
- collection_files: collection_id(INTEGER), file_id(TEXT)
- users: user_id(INTEGER), username(TEXT), full_name(TEXT), \
role(TEXT:'ADMIN','MANAGER','STAFF','USER','CUSTOMER','GUEST'), is_active(INTEGER), \
email(TEXT), phone_number(TEXT), start_date(DATE), expiry_date(DATE), last_login(REAL unix_ts)

- file_ratings: file_id(TEXT), client_uuid(TEXT: matches user_id), rating(INTEGER 1-5)
- file_comments: id(INTEGER), file_id(TEXT), client_uuid(TEXT: author), \
comment_text(TEXT), target_audience(TEXT: 'public'/'internal'/'user:{{id}}')

STATUS FLAGS (SYSTEM FLAGS):
Files can be assigned status tags. These are stored in the 'collections' table where type='system_flag'.
To filter by a status, use a subquery or join on collection_files and collections.
Example: SELECT f.id FROM files f JOIN collection_files cf ON f.id = cf.file_id \
JOIN collections c ON cf.collection_id = c.id WHERE c.name='Approved'

CURRENT DATABASE STATUS FLAGS:
- ID: 1 | Name: 'Approved' | Color: Green
- ID: 2 | Name: 'Review' | Color: Yellow
- ID: 3 | Name: 'To Edit' | Color: Cyan
- ID: 4 | Name: 'Rejected' | Color: Red
- ID: 5 | Name: 'Select' | Color: Purple

RULES:
1. You MUST return ONLY the raw SQL query. No markdown formatting (do not wrap in ```sql), no explanations.
2. The query MUST be a SELECT statement returning ONLY the 'id' column from the \
'files' table. Example: SELECT DISTINCT f.id FROM files f LEFT JOIN file_ratings r \
ON f.id = r.file_id WHERE r.rating = 5
3. Use standard SQLite syntax. Do not invent columns or tables.
4. Case sensitivity: By default, assume searches (like folder or file names) are \
case-insensitive and use the standard SQLite LIKE operator (with %). However, if \
the user explicitly specifies that the search must be case-sensitive, you MUST use \
the GLOB operator instead of LIKE, and use asterisks (*) as wildcards (e.g., WHERE \
f.path GLOB '*{BA_OU_PA}/FolderName*').
5. Path handling: The base directory on the disk is '{BA_OU_PA}'. In the database, \
the 'path' column contains the full absolute path starting with this prefix. When \
the user queries for a relative path or treats a folder as the root '/', you must \
map it to '{BA_OU_PA}'. For example, if the user asks for files in '/projects', \
look for paths LIKE '{BA_OU_PA}/projects%'.
6. Megapixel calculation: To filter or calculate megapixels from the 'dimensions' \
field (format 'WIDTHxHEIGHT'), extract the width and height by splitting or parsing \
the string (e.g., using CAST(SUBSTR(...) AS INT)), multiply them, and divide by 1,000,000.

MY REQUEST (I will use my native language):
"""
    # --- END OF FACTORY PROMPT BASE ---
    final_prompt = FACTORY_PROMPT_BASE
    os.makedirs(omni_dir, exist_ok=True)

    if reset or not os.path.exists(dict_path):
        try:
            with open(dict_path, "w", encoding="utf-8") as f:
                f.write(final_prompt)
        except Exception as e:
            _logger.debug("handled a failure in get_omniquery_dictionary", exc_info=True)
            print(f"WARN: Could not write omniquery_template.txt: {e}")
        return final_prompt
    try:
        with open(dict_path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        # An unreadable template falls back to the factory base -- the
        # same thing every other path returns. This branch used to
        # splice in a `dynamic_statuses` that no longer exists
        # anywhere, so a read failure raised NameError instead of
        # degrading.
        _logger.debug("handled a failure in get_omniquery_dictionary", exc_info=True)
        return final_prompt


def run_integrity_check():
    """
    System Health Check with user advice and cross-platform wait.
    Verifies libraries, files, and version consistency.
    """
    print("INFO: Running system integrity check...")

    issues_found = False
    critical_error = False

    # 1. Check Libraries
    required_libs = [
        ("flask", "Flask"),
        ("PIL", "Pillow"),
        ("cv2", "opencv-python"),
        ("waitress", "waitress"),
        ("cryptography", "cryptography"),
    ]

    for lib_imp, lib_name in required_libs:
        try:
            __import__(lib_imp)
        except ImportError:
            print(f"\n{Colors.RED}❌ MISSING LIBRARY: {lib_name}{Colors.RESET}")
            issues_found = True
            critical_error = True

    # 2. Check Files & Version Headers
    critical_files = [
        "templates/index.html",
        "templates/exhibition.html",
        "templates/exhibition_login.html",
        "templates/modals/user_manager_module.html",
        "templates/modals/remix_modal.html",
        "templates/modals/omniquery_palette.html",
        "templates/css/index.css",
        "templates/collections.html",
        "templates/list_view.html",
        "templates/css/collections.css",
    ]

    mismatches = []
    for f_path in critical_files:
        if not os.path.exists(f_path):
            print(f"\n{Colors.RED}❌ CRITICAL FILE MISSING: {f_path}{Colors.RESET}")
            issues_found = True
            critical_error = True
            continue

        try:
            with open(f_path, encoding="utf-8") as f:
                header = "".join([f.readline() for _ in range(15)])
                if APP_VERSION not in header:
                    mismatches.append(f_path)
                    issues_found = True
        except Exception:
            _logger.debug("ignored a failure in run_integrity_check", exc_info=True)

    if mismatches:
        print(f"\n{Colors.YELLOW}⚠️  VERSION WARNING: Some files are outdated or modified:{Colors.RESET}")
        for m in mismatches:
            print(f"   - {m}")
        print(f"{Colors.YELLOW}   Expected Version: {APP_VERSION}.{Colors.RESET}")

    # 3. Advice and "Press Enter" logic
    if issues_found:
        print(f"\n{Colors.CYAN}{Colors.BOLD}💡 ADVICE:{Colors.RESET}")
        print("   Please verify your installation or check for updates at:")
        print(f"   {Colors.BLUE}{Colors.BOLD}{GITHUB_REPO_URL}{Colors.RESET}")

        if critical_error:
            print(f"\n{Colors.RED}The application cannot start due to missing components.{Colors.RESET}")

        # Cross-platform wait that doesn't crash Docker if non-interactive
        try:
            print(f"\n{Colors.DIM}Press Enter to {'exit' if critical_error else 'continue'}...{Colors.RESET}")
            input()
        except (EOFError, KeyboardInterrupt):
            # Fallback for non-interactive environments (Docker/Headless)
            pass

        if critical_error:
            sys.exit(1)

    print(f"{Colors.GREEN}SUCCESS: System integrity verified (v{APP_VERSION}).{Colors.RESET}")


# --- HELPER FOR AI PATH CONSISTENCY ---
def get_standardized_path(filepath):
    """
    Converts path to absolute, forces forward slashes, and handles case sensitivity for Windows.
    Used ONLY for AI Queue uniqueness to prevent loops on mixed-path systems.
    """
    if not filepath:
        return ""
    try:
        # Resolve absolute path (handles .. and current dir)
        abs_path = os.path.abspath(filepath)
        # Force forward slashes (works on Win/Linux/Mac for Python)
        std_path = abs_path.replace("\\", "/")
        # On Windows, filesystem is case-insensitive, so we lower for the DB unique key
        if os.name == "nt":
            return std_path.lower()
        return std_path
    except (TypeError, ValueError, OSError):
        # abspath rejects a non-path argument, an embedded NUL, or a name the
        # platform cannot resolve. The raw text is still a usable queue key.
        return str(filepath)


def _normalize_fuzzy_string(s):
    """Reduce a model name to its letters and digits, for loose matching.

    Applied to BOTH sides of the model comparison -- see model_condition --
    so what it keeps never has to agree with a second list somewhere else.

    isalnum rather than [a-zA-Z0-9]: the old spelling dropped every letter
    outside English, so a LoRA named in Russian or Greek reduced to nothing
    at all. casefold rather than lower, so straße and STRASSE meet.
    """
    if not s:
        return ""
    return "".join(c for c in str(s).casefold() if c.isalnum())


def _word_key(s):
    """The same text as whole words, each padded with spaces.

    Backs the quoted "exact word" form: searching for ' man ' inside
    ' a woman here ' cannot match, which is what the quotes promise.
    """
    if not s:
        return " "
    out, previous_was_word = [], False
    for c in str(s).casefold():
        if c.isalnum():
            out.append(c)
            previous_was_word = True
        elif previous_was_word:
            out.append(" ")
            previous_was_word = False
    return " " + "".join(out).strip() + " "


def day_bounds(typed_date, browser_ts, is_end):
    """The first or last instant of a chosen day, as a timestamp.

    A picture's date is drawn by the browser --
    `new Date(mtime*1000).toLocaleString()` -- while the filter boundary
    was worked out here, from a naive `strptime(...).timestamp()`, which
    is the SERVER's midnight. The two machines are often not in the same
    timezone: the shipped compose file leaves /etc/localtime commented
    out, so the container is UTC while the person looking at it is not.
    Filtering a single day from New Zealand then returned 11 of their 24
    hours and 13 hours belonging to the day before.

    So the browser sends the boundary it means, computed from the same
    clock that wrote the label, and that is used when it arrives.

    Without it -- a bookmark, a typed URL, a script -- this falls back to
    reading the date here, as before, except that the end of the day is
    now the next midnight rather than +86399 seconds. Those differ on the
    day the clocks go back: a 25-hour day ended at 22:59:59 and lost its
    last hour.
    """
    if browser_ts not in (None, ""):
        try:
            given = float(browser_ts)
        except (TypeError, ValueError):
            given = None  # not a number; fall back to the date
        # float('NaN') and float('inf') both parse. A NaN boundary is worse
        # than no boundary: every comparison against it is false, so the
        # gallery empties with the filter still showing the date you asked
        # for.
        if given is not None and math.isfinite(given):
            return given
    if not typed_date:
        return None
    try:
        # A calendar date, kept naive on purpose, then resolved to an instant
        # by .timestamp() -- which reads it as local time.
        #
        # It must stay naive to be right. Adding a day to an AWARE datetime
        # adds exactly 24 hours; adding it to a naive one and resolving
        # afterwards lands on the next local midnight, which is 23 or 25
        # hours away on the days the clocks change. That is the whole point
        # of the end boundary, and tests/test_a_day_means_the_viewers_day.py
        # pins both DST days.
        midnight = datetime.combine(date.fromisoformat(typed_date), datetime.min.time())
    except (TypeError, ValueError):
        return None
    if not is_end:
        return midnight.timestamp()
    return (midnight + timedelta(days=1)).timestamp() - 1


def model_condition(typed, column, is_not):
    """One term of the model/LoRA filter, as SQL plus its parameter.

    Both sides of every comparison go through the same Python function, so
    a name is found by the characters it has rather than by whether its
    punctuation appears on two lists that were written separately. It was
    two lists: the typed text kept only English letters and digits, the
    column kept everything except ` . _ - / \\ ( ) [ ]`, and any character
    in the gap -- a plus, an apostrophe, an at sign, a hash, an exclamation
    mark, the German sharp s -- made the model unfindable by its own name.
    The gallery offers those names in its own suggestion list, so the way
    to meet this was to click what it suggested.

    INSTR rather than LIKE, because LIKE reads `_` and `%` in the typed
    text as wildcards: `"detail_tweaker_xl"` was matching `detail tweaker
    xl` by accident, and a name containing `%` could not be searched for.
    """
    # A file made without any model still answers `!something`, so the
    # column has to read as empty text rather than as nothing at all.
    column = f"COALESCE({column}, '')"
    exact = typed.startswith('"') and typed.endswith('"') and len(typed) > 2
    if exact:
        expression, needle = f"wordkey({column})", _word_key(typed[1:-1])
    else:
        folded = _normalize_fuzzy_string(typed)
        if len(folded) >= 3:
            expression, needle = f"fuzzykey({column})", folded
        else:
            # Too short to fold usefully, or punctuation only: compare the
            # text as it stands, still folding case in every script.
            expression, needle = f"ulower({column})", _unicode_lower(typed)
    test = "= 0" if is_not else "> 0"
    return f"INSTR({expression}, ?) {test}", needle


def normalize_smart_path(path_str):
    """
    Normalizes a path string for search comparison:
    1. Converts to lowercase.
    2. Replaces all backslashes (\\) with forward slashes (/).
    """
    if not path_str:
        return ""
    return str(path_str).lower().replace("\\", "/")


# Every environment variable this program reads, across all its packages.
# tests/test_configuration_doc_matches_code.py holds this to the code, so it
# cannot drift as settings are added.
KNOWN_ENV_VARS = (
    "ADMIN_PASSWORD",
    "AI_DAM_AUTO_PROVISION",
    "AI_DAM_CACHE_DIR",
    "AI_DAM_CRITIC_BACKEND",
    "AI_DAM_DEVICE",
    "AI_DAM_EMBED_BATCH",
    "AI_DAM_EPHEMERAL_INDEX",
    "AI_DAM_FACE_BACKEND",
    "AI_DAM_FACE_CLUSTER_THRESHOLD",
    "AI_DAM_FACE_DETECT_MAX_SIDE",
    "AI_DAM_FACE_EMBEDDER",
    "AI_DAM_FACE_GRAPH_BACKEND",
    "AI_DAM_FACE_MIN_PX",
    "AI_DAM_FAISS_GPU",
    "AI_DAM_ATTN",
    "AI_DAM_CRITIC_MODEL",
    "AI_DAM_MODELS_DIR",
    "AI_DAM_NEAR_DUP_DISTANCE",
    "AI_DAM_ORT_PROVIDERS",
    "AI_DAM_SEGMENTER_BACKEND",
    "AI_DAM_SEMANTIC_BACKEND",
    "AI_DAM_SIMILAR_K",
    "AI_DAM_VECTOR_GPU",
    "AI_DAM_VISUAL_BACKEND",
    "AI_DAM_WORKER_BATCH",
    "AI_DAM_WORKER_POLL",
    "BASE_INPUT_PATH",
    "BASE_MODELS_PATH",
    "BASE_OUTPUT_PATH",
    "BASE_SMARTGALLERY_PATH",
    "BATCH_SIZE",
    "CHECKPOINTS_PATH",
    "COMFYUI_MAX_UPLOAD_MB",
    "COMFYUI_SERVER_URL",
    "DELETE_TO",
    "ENABLE_AI_DAM",
    "ENABLE_AI_SEARCH",
    "FFMPEG_AUTO_DOWNLOAD",
    "FFPROBE_MANUAL_PATH",
    "GENERATE_THUMBNAILS",
    "GENERATE_WAVEFORMS",
    "GENPARAMS_BACKFILL",
    "LORAS_PATH",
    "MAX_PARALLEL_WORKERS",
    "OMNIQUERY_NL2SQL_MODEL",
    "PAGE_SIZE",
    "PARALLEL_SCAN_MIN_FILES",
    "SECRET_KEY",
    "SERVER_PORT",
    "STREAM_THRESHOLD_MB",
    "THUMBNAIL_WIDTH",
    "UNET_PATH",
    "WEBP_ANIMATED_FPS",
)


def find_misspelt_env_vars(environ=None, known=KNOWN_ENV_VARS, cutoff=0.88):
    """Environment variables that closely resemble a setting but are not one.

    A misspelt NAME is the quietest way to configure nothing: the variable
    is simply never read, the default applies, and there is no error to
    notice. `BASE_OUTPUT_PATH` typed with one letter missing shows the
    default folder instead of the library, which reads as "the gallery
    cannot see my files".

    Only close matches are reported. Warning on everything that merely
    starts with a familiar prefix would fire on other programs' variables
    -- ComfyUI's own COMFYUI_* among them -- and a warning that cries wolf
    is worth less than none.

    The cutoff was measured rather than guessed. Real slips of our own
    names score 0.889 and above (DELETE_T0 is the closest call); an
    unrelated PAI_MODEL_DIR found on a live machine scored 0.800 against
    AI_DAM_MODELS_DIR, and AI_MODELS_DIR scores 0.867. 0.88 keeps every
    true slip and drops those. It errs towards saying nothing: a missed
    warning costs what the situation already cost, while a wrong one
    teaches people to ignore the next.

    Returns a list of (name_found, nearest_known).
    """

    env = os.environ if environ is None else environ
    known_set = set(known)
    hits = []
    for name in sorted(env):
        if name in known_set:
            continue
        near = difflib.get_close_matches(name, list(known), n=1, cutoff=cutoff)
        if near:
            hits.append((name, near[0]))
    return hits


def warn_about_misspelt_env_vars():
    """Print a warning for each near-miss. Never fatal: the documented
    contract is that nothing in the environment stops the app starting."""
    for name, suggestion in find_misspelt_env_vars():
        print(
            f"{Colors.YELLOW}WARNING: environment variable {name} is not a "
            f"SmartGallery setting -- did you mean {suggestion}? "
            f"It is being ignored.{Colors.RESET}"
        )


def print_configuration():
    """Prints the current configuration in a neat, aligned table."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}--- CURRENT CONFIGURATION ---{Colors.RESET}")

    # Helper for aligned printing
    def print_row(key, value, is_path=False):
        color = Colors.CYAN if is_path else Colors.GREEN
        print(f" {Colors.BOLD}{key:<25}{Colors.RESET} : {color}{value}{Colors.RESET}")

    print_row("Server Port", SERVER_PORT)
    print_row("Base Output Path", BASE_OUTPUT_PATH, True)
    print_row("Base Input Path", BASE_INPUT_PATH, True)
    print_row("SmartGallery Path", BASE_SMARTGALLERY_PATH, True)
    print_row("FFprobe Path", FFPROBE_MANUAL_PATH, True)
    print_row("Delete To (Trash)", DELETE_TO or "Disabled (Permanent Delete)", DELETE_TO is not None)
    print_row("WebP Animated FPS", WEBP_ANIMATED_FPS)
    print_row("Page Size", PAGE_SIZE)
    print_row("Stream Threshold", f"{STREAM_THRESHOLD_MB} MB")
    print_row("Max Parallel Workers", MAX_PARALLEL_WORKERS or "All Cores")

    # Process Command Line Arguments to display them securely
    cli_args = sys.argv[1:]
    if not cli_args:
        cli_display = "None"
    else:
        masked_args = []
        skip_next = False
        for i, arg in enumerate(cli_args):
            if skip_next:
                skip_next = False
                continue

            # Handle space-separated password argument
            if arg == "--admin-pass":
                masked_args.append("--admin-pass ********")
                # If there is a value after --admin-pass, skip it in the next loop iteration
                if i + 1 < len(cli_args):
                    skip_next = True
            # Handle equals-separated password argument (e.g. --admin-pass=12345)
            elif arg.startswith("--admin-pass="):
                masked_args.append("--admin-pass=********")
            else:
                masked_args.append(arg)

        cli_display = " ".join(masked_args)

    print_row("ComfyUI API URL", COMFYUI_SERVER_URL, True)
    if ENABLE_AI_SEARCH:
        print_row("AI Search", "Enabled" if ENABLE_AI_SEARCH else "Disabled")
    if GENERATE_WAVEFORMS:
        print_row("Audio Waveforms", "Enabled")

    print(f" {Colors.BOLD}{'CLI Parameters':<25}{Colors.RESET} : {Colors.YELLOW}{cli_display}{Colors.RESET}")
    print(f"{Colors.HEADER}-----------------------------{Colors.RESET}")

    # LoRA Synergy Paths (Optional Feature)
    print(f"\n{Colors.YELLOW}{Colors.BOLD}--- LoRA SYNERGY PATHS ---{Colors.RESET}")

    def print_lora_row(key, value):
        print(f" {Colors.BOLD}{key:<25}{Colors.RESET} : {Colors.YELLOW}{value}{Colors.RESET}")

    print_lora_row("LoRAs Path", LORAS_PATH)
    print_lora_row("Checkpoints Path", CHECKPOINTS_PATH)
    print_lora_row("UNET Path", UNET_PATH)
    print(f"{Colors.YELLOW}--------------------------{Colors.RESET}\n")


def management_api_only(f):
    """
    Bulletproof Security Decorator: Blocks access to destructive or management APIs.
    Guarantees that Guests/Customers CANNOT execute these functions under ANY circumstance.
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if IS_EXHIBITION_MODE:
            return jsonify(
                {"status": "error", "message": "Security Lockdown: This API is physically disabled in Exhibition Mode."}
            ), 403

        user_role = session.get("role")
        user_id = session.get("user_id")

        if user_id or user_role:
            if user_role not in ["ADMIN", "MANAGER", "STAFF"]:
                return jsonify(
                    {"status": "error", "message": "Forbidden: Insufficient privileges for this action."}
                ), 403
        elif FORCE_LOGIN:
            return jsonify({"status": "error", "message": "Unauthorized: Authentication required."}), 401

        return f(*args, **kwargs)

    return decorated_function


# --- FLASK APP INITIALIZATION ---
# The templates directory doubles as the static folder, so every template is
# also downloadable under /static/.
app = Flask(__name__, static_folder="templates", static_url_path="/static")
app.secret_key = SECRET_KEY

# Flask leaves SameSite unset by default ("Default: None", docs/config.rst,
# where 'Lax' is the recommended value), which hands the decision to the
# browser: Chrome treats an unset cookie as Lax, Firefox does not. Whether
# a signed-in visitor's cookie rides along with somebody else's form post
# should not depend on which browser they opened. Lax rather than Strict
# so that following a link into the gallery still arrives signed in.
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# Already Flask's default; stated so that it stays true.
app.config["SESSION_COOKIE_HTTPONLY"] = True

folder_config_cache = None
FFPROBE_EXECUTABLE_PATH = None


@app.before_request
def refuse_writes_started_by_another_site():
    """A page you are visiting must not be able to act on your gallery.

    Six routes that change something need no request body at all, so a
    plain form on any web page reaches them -- a form cannot set a JSON
    content type, which is what happens to protect the other thirty-four.
    Two of the six delete things, and neither needs anything guessed:
    delete_folder takes a folder key that is only
    base64(relative path), so a folder called `videos` is `dmlkZW9z`.

    In the default local mode there is no login, so there is no cookie to
    withhold and nothing else stands in the way. The browser will not let
    the other page read the answer, but it does send the request, and the
    folder is gone either way.

    Browsers say where a request came from. Per the Fetch Metadata spec, a
    form submission arrives with Sec-Fetch-Site set to same-origin,
    same-site or cross-site "as appropriate", and something the person did
    themselves -- an address bar, a bookmark -- arrives as `none`, which
    servers may "treat as trusted". Anything not a browser sends no such
    header, so scripts and curl are unaffected.

    Only `cross-site` is refused. `same-site` would mean another subdomain
    of the gallery's own domain, which a reverse proxy can legitimately
    produce, and this compares nothing itself: the browser worked out the
    relationship, so putting a proxy in front changes nothing here.

    What this does not cover, stated rather than implied: the spec asserts
    the request URL is "a potentially trustworthy URL" before setting
    these headers, so a gallery reached over plain http at a LAN address
    -- which the startup banner offers -- gets no Sec-Fetch-Site at all.
    localhost is potentially trustworthy and is covered. Comparing Origin
    against Host would reach the LAN case, and would also refuse the
    proxied setups where those legitimately differ, so it is not done
    here.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None
    if request.headers.get("Sec-Fetch-Site") != "cross-site":
        return None
    return jsonify(
        {
            "status": "error",
            "message": (
                "This request came from another website, so the gallery "
                "refused it. If you meant to do this, do it from the "
                "gallery itself."
            ),
        }
    ), 403


@app.before_request
def restrict_static_to_signed_in_callers():
    """`/static/` is Flask's own route, not one of ours, so none of the
    per-route decisions applied to it and nothing listed it as needing any.

    That made the whole templates directory readable without signing in:
    on a server started with --force-login, /static/index.html returned the
    entire management interface, and /static/modals/user_manager_module.html
    the user manager, to a caller who had never logged in. No gallery data
    goes out that way -- the files are served unrendered, and every endpoint
    the markup calls is gated separately -- but it hands out the shape of
    the whole management side, and it contradicts what --force-login says
    it does.

    The rule is the one gallery_view already applies. Where no login is
    configured nothing changes, which is the common local case. The login
    page and the exhibition portal reference no static asset at all, so
    only index.html's two stylesheets come through here, and only for a
    session that has already signed in to be shown index.html.
    """
    if request.endpoint != "static":
        return
    if not (IS_EXHIBITION_MODE or FORCE_LOGIN):
        return
    if "user_id" in session:
        return
    abort(403)


class ViewSnapshotStore:
    """Per-view result snapshots backing the pagination endpoints.

    Every full gallery render stores its computed file list under an opaque
    token bound to the session owner that produced it; /load_more and
    /api/current_view_ids look the snapshot up by that token. Concurrent
    tabs and users therefore can never be served each other's views (the
    previous single process-wide cache was overwritten by whichever render
    happened last, from any session). A missing token signals the client
    that its view snapshot expired and the page must re-render.

    The LRU is bounded twice over, because counting snapshots bounds
    nothing: a snapshot holds one row per file in the view, and a view can
    be the whole library. Measured at 2278 bytes per file over 26 fields,
    of which workflow_prompt and workflow_files are half:

        10,000 files    21.7 MB per snapshot      695 MB at 32 snapshots
        50,000 files   108.6 MB per snapshot     3475 MB at 32 snapshots
       200,000 files   434.4 MB per snapshot    13902 MB at 32 snapshots

    One snapshot of a large library is what paging through it costs and
    cannot be avoided here. Keeping thirty-two of them is what turns a
    large library into running out of memory, so there is a ceiling on
    rows held as well as on snapshot count, and whichever bites first
    wins.

    The newest snapshot is never evicted, whatever its size: it is the one
    the caller is about to page through, and dropping it would leave a
    library above the ceiling unable to page at all.
    """

    # ~100k rows is about 230 MB at the size measured above. Small
    # libraries never reach it and keep the full thirty-two; a big one
    # keeps fewer; a single view larger than this keeps just itself.
    def __init__(self, capacity=32, max_rows=100_000):
        self._lock = threading.Lock()
        self._snapshots = OrderedDict()  # token -> (owner, files)
        self._capacity = capacity
        self._max_rows = max_rows

    def _rows_held(self):
        return sum(len(files) for _owner, files in self._snapshots.values())

    def put(self, owner, files):
        token = secrets.token_urlsafe(16)
        with self._lock:
            self._snapshots[token] = (owner, files)
            while len(self._snapshots) > self._capacity:
                self._snapshots.popitem(last=False)
            # Oldest first, and never the one just stored.
            while len(self._snapshots) > 1 and self._rows_held() > self._max_rows:
                self._snapshots.popitem(last=False)
        return token

    def get(self, token, owner):
        with self._lock:
            entry = self._snapshots.get(token)
            if entry is None or entry[0] != owner:
                return None
            self._snapshots.move_to_end(token)
            return entry[1]


VIEW_SNAPSHOTS = ViewSnapshotStore()


def _view_owner():
    """Snapshot ownership key: the logged-in user id, or '' for the single
    anonymous local-admin session (no login modes)."""
    return str(session.get("user_id") or "")


# --- AI DAM BLUEPRINT (WI-31, optional; every route no-ops when disabled) ---
# file_access_check applies the gallery's per-file visibility policy to the
# AI read routes so restricted-mode viewers cannot read derived AI metadata
# for files the normal routes would refuse (lambda: is_file_accessible is
# defined later in this module and resolved at request time).
app.register_blueprint(
    ai_dam_service.create_ai_blueprint(
        AI_CONFIG,
        guard=management_api_only,
        file_access_check=lambda fid: is_file_accessible(fid),
        # Seeing a picture and being told how it was made are different
        # permissions here: the same rule that strips prompts out of the
        # files and the listings decides whether a review may quote them.
        generation_metadata_visible=lambda: not should_strip_metadata(),
    ),
    url_prefix="/galleryout/api/aidam",
)


# Data structures for node categorization and analysis
NODE_CATEGORIES_ORDER = ["input", "model", "processing", "output", "others"]
NODE_CATEGORIES = {
    "Load Checkpoint": "input",
    "CheckpointLoaderSimple": "input",
    "Empty Latent Image": "input",
    "CLIPTextEncode": "input",
    "Load Image": "input",
    "ModelMerger": "model",
    "KSampler": "processing",
    "KSamplerAdvanced": "processing",
    "VAEDecode": "processing",
    "VAEEncode": "processing",
    "LatentUpscale": "processing",
    "ConditioningCombine": "processing",
    "PreviewImage": "output",
    "SaveImage": "output",
    "LoadImageOutput": "input",
}
NODE_PARAM_NAMES = {
    "CLIPTextEncode": ["text"],
    "KSampler": ["seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "KSamplerAdvanced": [
        "add_noise",
        "noise_seed",
        "control_after_generate",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "start_at_step",
        "end_at_step",
        "return_with_leftover_noise",
    ],
    "Load Checkpoint": ["ckpt_name"],
    "CheckpointLoaderSimple": ["ckpt_name"],
    "Empty Latent Image": ["width", "height", "batch_size"],
    "LatentUpscale": ["upscale_method", "width", "height"],
    "SaveImage": ["filename_prefix"],
    "ModelMerger": ["ckpt_name1", "ckpt_name2", "ratio"],
    "Load Image": ["image"],
    "LoadImageMask": ["image"],
    "VHS_LoadVideo": ["video"],
    "LoadAudio": ["audio"],
    "AudioLoader": ["audio"],
    "LoadImageOutput": ["image"],
    "WanImageToVideo": ["width", "height", "length", "batch_size"],
    "ImageResize+": ["width", "height", "interpolation", "keep_proportion", "condition", "multiple_of"],
    "VantageI2VDualLooper": [
        "text",
        "seed",
        "control_after_generate",
        "steps",
        "param4",
        "param5",
        "cfg",
        "param7",
        "sampler_name",
        "scheduler",
        "width",
        "height",
        "param12",
        "param13",
        "fps",
        "param15",
        "align",
    ],
    "VantageProject": ["project", "positive_text", "param2", "filename", "param4", "hash"],
}

# Cache for node colors
_node_colors_cache = {}


def get_node_color(node_type):
    """Generates a unique and consistent color for a node type."""
    if node_type not in _node_colors_cache:
        # Use a hash to get a consistent color for the same node type
        hue = (hash(node_type + "a_salt_string") % 360) / 360.0
        rgb = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.7, 0.85)]
        _node_colors_cache[node_type] = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    return _node_colors_cache[node_type]


def filter_enabled_nodes(workflow_data):
    """Filters and returns only active nodes and links (mode=0) from a workflow."""
    if not isinstance(workflow_data, dict):
        return {"nodes": [], "links": []}

    active_nodes = [n for n in workflow_data.get("nodes", []) if n.get("mode", 0) == 0]
    active_node_ids = {str(n["id"]) for n in active_nodes}

    active_links = [
        link
        for link in workflow_data.get("links", [])
        if str(link[1]) in active_node_ids and str(link[3]) in active_node_ids
    ]
    return {"nodes": active_nodes, "links": active_links}


def generate_node_summary(workflow_json_string):
    """
    Analyzes a workflow JSON, extracts active nodes, and identifies input media.
    Robust version: handles ComfyUI specific suffixes like ' [output]'.
    """
    try:
        workflow_data = json.loads(workflow_json_string)
    except json.JSONDecodeError:
        return None

    nodes = []
    is_api_format = False

    if "nodes" in workflow_data and isinstance(workflow_data["nodes"], list):
        active_workflow = filter_enabled_nodes(workflow_data)
        nodes = active_workflow.get("nodes", [])
    else:
        is_api_format = True
        for node_id, node_data in workflow_data.items():
            if isinstance(node_data, dict) and "class_type" in node_data:
                node_entry = node_data.copy()
                node_entry["id"] = node_id
                node_entry["type"] = node_data["class_type"]
                node_entry["inputs"] = node_data.get("inputs", {})
                nodes.append(node_entry)

    if not nodes:
        return []

    def get_id_safe(n):
        raw_id = str(n.get("id", "0"))
        try:
            # Handle nested Node IDs (e.g., "301:297" -> (301, 297)) for perfect sorting
            return tuple(int(x) for x in raw_id.split(":"))
        except Exception:
            # Fallback for pure string IDs (pushes them to the end of the list safely)
            _logger.debug("handled a failure in get_id_safe", exc_info=True)
            return (float("inf"), raw_id)

    sorted_nodes = sorted(
        nodes, key=lambda n: (NODE_CATEGORIES_ORDER.index(NODE_CATEGORIES.get(n.get("type"), "others")), get_id_safe(n))
    )

    summary_list = []

    valid_media_exts = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".jfif",
        ".bmp",
        ".tiff",
        ".mp4",
        ".mov",
        ".webm",
        ".mkv",
        ".avi",
        ".mp3",
        ".wav",
        ".ogg",
        ".flac",
        ".m4a",
        ".aac",
    }

    for node in sorted_nodes:
        node_type = node.get("type", "Unknown")
        params_list = []

        raw_params = {}
        if is_api_format:
            raw_params = node.get("inputs", {})
        else:
            widgets_values = node.get("widgets_values", [])
            param_names_list = NODE_PARAM_NAMES.get(node_type, [])
            for i, value in enumerate(widgets_values):
                name = param_names_list[i] if i < len(param_names_list) else f"param_{i + 1}"
                raw_params[name] = value

        for name, value in raw_params.items():
            display_value = value
            is_input_file = False
            input_url = None

            if isinstance(value, list):
                display_value = f"(Link to {value[0]})" if len(value) == 2 and isinstance(value[0], str) else str(value)

            if isinstance(value, str) and value.strip():
                # 1. Aggressive cleanup to remove suffixes like " [output]" or " [input]"
                clean_value = value.replace("\\", "/").strip()
                # Remove common suffixes in square brackets at the end of the string
                clean_value = re.sub(r"\s*\[.*?\]$", "", clean_value)

                _, ext = os.path.splitext(clean_value)

                if ext.lower() in valid_media_exts:
                    filename_only = os.path.basename(clean_value)

                    candidates = [
                        os.path.join(BASE_INPUT_PATH, clean_value),
                        os.path.join(BASE_INPUT_PATH, filename_only),
                        os.path.normpath(os.path.join(BASE_INPUT_PATH, clean_value)),
                    ]

                    for candidate_path in candidates:
                        try:
                            if os.path.isfile(candidate_path):
                                abs_candidate = os.path.abspath(candidate_path)
                                abs_base = os.path.abspath(BASE_INPUT_PATH)

                                if abs_candidate.startswith(abs_base):
                                    is_input_file = True
                                    rel_path = os.path.relpath(abs_candidate, abs_base).replace("\\", "/")
                                    input_url = f"/galleryout/input_file/{rel_path}"
                                    # Also update the displayed value to clean it up
                                    display_value = clean_value
                                    break
                        except Exception:
                            _logger.debug("handled a failure in generate_node_summary", exc_info=True)
                            continue

            params_list.append(
                {"name": name, "value": display_value, "is_input_file": is_input_file, "input_url": input_url}
            )

        summary_list.append(
            {
                "id": node.get("id", "N/A"),
                "type": node_type,
                "category": NODE_CATEGORIES.get(node_type, "others"),
                "color": get_node_color(node_type),
                "params": params_list,
            }
        )

    return summary_list


# --- ALL UTILITY AND HELPER FUNCTIONS ARE DEFINED HERE, BEFORE ANY ROUTES ---

# ============================================================================
# NEW INTEGRATED TOOLS (ADVANCED METADATA EXTRACTION)
# This section contains the new parsing logic integrated directly into the source
# ============================================================================

# --- Regex Patterns for Prompt Parsing ---
RE_LORA_PROMPT = re.compile(r"<lora:([\w_\s.-]+)(?::([\d.]+))*>", re.IGNORECASE)
RE_LYCO_PROMPT = re.compile(r"<lyco:([\w_\s.]+):([\d.]+)>", re.IGNORECASE)
RE_PARENS = re.compile(r"[\\/\[\](){}]+")
RE_LORA_CLOSE = re.compile(r">\s+")


def clean_prompt_text(x: str) -> dict[str, Any]:
    """
    Cleans a raw prompt string: removes LoRA tags, normalizes whitespace,
    and extracts LoRA usage into a separate list.
    """
    if not x:
        return {"text": "", "loras": []}

    x = re.sub(r"\sBREAK\s", " , BREAK , ", x)
    x = re.sub(RE_LORA_CLOSE, "> , ", x)
    x = x.replace("，", ",").replace("-", " ").replace("_", " ")

    tag_list = [t.strip() for t in x.split(",")]
    lora_list = []
    final_tags = []

    for tag in tag_list:
        if not tag:
            continue

        lora_match = re.search(RE_LORA_PROMPT, tag)
        lyco_match = re.search(RE_LYCO_PROMPT, tag)

        if lora_match:
            val = float(lora_match.group(2)) if lora_match.group(2) else 1.0
            lora_list.append({"name": lora_match.group(1), "value": val})
        elif lyco_match:
            lora_list.append({"name": lyco_match.group(1), "value": float(lyco_match.group(2))})
        else:
            clean_tag = re.sub(RE_PARENS, "", tag).strip()
            if clean_tag:
                final_tags.append(clean_tag)

    return {"text": ", ".join(final_tags), "loras": lora_list}


class ComfyMetadataParser:
    """
    Advanced parser that traces the workflow graph to find real generation parameters.
    Updated to resolve links for Width, Height, and other linked numeric values.
    """

    def __init__(self, workflow_json: dict):
        self.data = workflow_json

    def parse(self) -> dict[str, Any]:
        """
        Main parsing method. Returns a standardized dictionary.
        """
        meta = {
            "seed": None,
            "steps": None,
            "cfg": None,
            "sampler": None,
            "scheduler": None,
            "model": None,
            "positive_prompt": "",
            "negative_prompt": "",
            "positive_prompt_clean": "",
            "width": None,
            "height": None,
            "loras": [],
        }

        # Strategy A: Trace from KSampler (Most accurate for Prompts/Model)
        sampler_node_id = self._find_sampler_node()

        if sampler_node_id:
            self._extract_sampler_params(sampler_node_id, meta)
            self._extract_prompts_from_sampler(sampler_node_id, meta)
            self._extract_model_from_sampler(sampler_node_id, meta)
            self._extract_size_from_sampler(sampler_node_id, meta)

        # Strategy B: Fallback Scan (Scans specific nodes if Strategy A missed data)
        self._fallback_scan(meta)

        # Cleanup
        if meta["positive_prompt"]:
            cleaned = clean_prompt_text(meta["positive_prompt"])
            meta["positive_prompt_clean"] = cleaned["text"]
            for lora in cleaned.get("loras", []):
                if not any(existing.get("name") == lora["name"] for existing in meta["loras"]):
                    meta["loras"].append(lora)

        # Deduplicate Prompts if they are identical due to tracing overlaps
        if meta["negative_prompt"] == meta["positive_prompt"]:
            meta["negative_prompt"] = ""

        return meta

    def _find_sampler_node(self):
        """Finds the main KSampler node ID."""
        if not isinstance(self.data, dict):
            return None
        for node_id, node in self.data.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type", "")
            if "KSampler" in class_type or "SamplerCustom" in class_type:
                return node_id
        return None

    def _get_real_value(self, value):
        """
        Follows links recursively to find the actual value.
        Improved to handle UI format where values are in widgets_values.
        """
        if not isinstance(value, list):
            return value

        try:
            source_id = str(value[0])
            if source_id in self.data:
                node = self.data[source_id]

                # Check Inputs (API Format)
                inputs = node.get("inputs", {})
                for key in ["value", "int", "float", "string", "text"]:
                    if key in inputs:
                        return self._get_real_value(inputs[key])

                # Check Widgets (UI Format)
                widgets = node.get("widgets_values", [])
                if widgets and not isinstance(widgets[0], (list, dict)):
                    return widgets[0]

                # If it's another link in widgets (ComfyUI logic), follow it
                if widgets and isinstance(widgets[0], list):
                    return self._get_real_value(widgets[0])
        except (IndexError, AttributeError, TypeError, RecursionError):
            # An empty link, a node that is not a mapping, or a graph whose
            # links form a cycle. None means "could not trace it", which is
            # what every caller already handles.
            pass
        return None

    def _extract_size_from_sampler(self, node_id, meta):
        """
        Traces the latent image link.
        If direct tracing fails, it attempts to find any 'EmptyLatentImage' node.
        """
        inputs = self.data[node_id].get("inputs", {})
        found_size = False

        if "latent_image" in inputs:
            link = inputs["latent_image"]
            if isinstance(link, list):
                source_id = str(link[0])
                node = self.data.get(source_id, {})
                node_inputs = node.get("inputs", {})

                if "width" in node_inputs:
                    meta["width"] = self._get_real_value(node_inputs["width"])
                    found_size = True
                if "height" in node_inputs:
                    meta["height"] = self._get_real_value(node_inputs["height"])

        # Final attempt: if still no size, scan for any EmptyLatentImage node in the graph
        if not found_size:
            for n in self.data.values():
                if n.get("class_type") == "EmptyLatentImage":
                    meta["width"] = self._get_real_value(n.get("inputs", {}).get("width"))
                    meta["height"] = self._get_real_value(n.get("inputs", {}).get("height"))
                    break

    def _extract_sampler_params(self, node_id, meta):
        """Extracts simple scalar values from the Sampler, resolving links."""
        inputs = self.data[node_id].get("inputs", {})

        # Use the new resolver to get actual values instead of links
        if "seed" in inputs:
            meta["seed"] = self._get_real_value(inputs["seed"])
        if "noise_seed" in inputs:
            meta["seed"] = self._get_real_value(inputs["noise_seed"])
        if "steps" in inputs:
            meta["steps"] = self._get_real_value(inputs["steps"])
        if "cfg" in inputs:
            meta["cfg"] = self._get_real_value(inputs["cfg"])
        if "sampler_name" in inputs:
            meta["sampler"] = self._get_real_value(inputs["sampler_name"])
        if "scheduler" in inputs:
            meta["scheduler"] = self._get_real_value(inputs["scheduler"])
        if "denoise" in inputs:
            meta["denoise"] = self._get_real_value(inputs["denoise"])

    def _extract_prompts_from_sampler(self, node_id, meta):
        """Traces 'positive' and 'negative' links to find text."""
        inputs = self.data[node_id].get("inputs", {})
        if "positive" in inputs:
            meta["positive_prompt"] = self._trace_text(inputs["positive"])
        if "negative" in inputs:
            meta["negative_prompt"] = self._trace_text(inputs["negative"])

    def _trace_text(self, link_info) -> str:
        """Recursive helper to find text content from a link."""
        if not isinstance(link_info, list):
            return ""
        source_id = str(link_info[0])
        if source_id not in self.data:
            return ""

        node = self.data[source_id]
        inputs = node.get("inputs", {})

        # Handle direct text encoders
        if "text" in inputs and isinstance(inputs["text"], str):
            return inputs["text"]

        # Handle SD3/Flux
        if "t5xxl" in inputs and isinstance(inputs["t5xxl"], str):
            return inputs["t5xxl"]

        # Handle concatenated or linked text
        if "text" in inputs and isinstance(inputs["text"], list):
            return self._trace_text(inputs["text"])

        # Handle Conditioning / Guidance nodes
        if "conditioning" in inputs:
            return self._trace_text(inputs["conditioning"])

        # Fallback to widgets for UI format nodes
        widgets = node.get("widgets_values", [])
        for w in widgets:
            if isinstance(w, str) and len(w) > 5:
                return w

        return ""

    def _extract_model_from_sampler(self, node_id, meta):
        """Follows the model wire to find the Checkpoint name."""
        inputs = self.data[node_id].get("inputs", {})
        if "model" in inputs:
            model_link = inputs["model"]
            if isinstance(model_link, list):
                source_id = str(model_link[0])
                if source_id in self.data:
                    node = self.data[source_id]
                    # Check for loader inputs
                    if "ckpt_name" in node.get("inputs", {}):
                        meta["model"] = node["inputs"]["ckpt_name"]
                    # Follow further if it's a LoRA or Model handler
                    elif "model" in node.get("inputs", {}) and isinstance(node["inputs"]["model"], list):
                        self._extract_model_from_sampler(source_id, meta)

    def _fallback_scan(self, meta):
        """Scans all nodes for specific types if direct tracing missed data."""
        if not isinstance(self.data, dict):
            return
        for node in self.data.values():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {})

            if meta["seed"] is None and class_type == "RandomNoise" and "noise_seed" in inputs:
                meta["seed"] = self._get_real_value(inputs["noise_seed"])

            if meta["cfg"] is None and "Guider" in class_type and "cfg" in inputs:
                meta["cfg"] = self._get_real_value(inputs["cfg"])

            if meta["steps"] is None and "Scheduler" in class_type and "steps" in inputs:
                meta["steps"] = self._get_real_value(inputs["steps"])

            if "lora" in class_type.lower() and "loader" in class_type.lower():
                lora_name = None
                weight = 1.0
                if "lora_name" in inputs:
                    lora_name = self._get_real_value(inputs.get("lora_name"))
                    weight = self._get_real_value(inputs.get("strength_model", inputs.get("strength", 1.0)))
                elif "widgets_values" in node:
                    widgets = node.get("widgets_values", [])
                    if len(widgets) > 0 and isinstance(widgets[0], str):
                        lora_name = widgets[0]
                    if len(widgets) > 1 and isinstance(widgets[1], (int, float)):
                        weight = widgets[1]

                if lora_name and isinstance(lora_name, str):
                    try:
                        weight = float(weight)
                    except (TypeError, ValueError):
                        weight = 1.0
                    if not any(lora.get("name") == lora_name for lora in meta["loras"]):
                        meta["loras"].append({"name": lora_name, "value": weight})


# ============================================================================
# END OF INTEGRATED TOOLS
# ============================================================================


def safe_delete_file(filepath):
    """
    Safely delete a file by either moving it to trash (if DELETE_TO is configured)
    or permanently deleting it.

    Args:
        filepath: Path to the file to delete

    Raises:
        OSError: If deletion/move fails
    """
    if DELETE_TO and TRASH_FOLDER:
        # Move to trash (folder already validated at startup)
        trash_path = _trash_destination(os.path.basename(filepath))
        shutil.move(filepath, trash_path)
        print(f"INFO: Moved file to trash: {trash_path}")
    else:
        # Permanently delete
        os.remove(filepath)


def _std_path(path):
    """A path with one separator style, for comparing stored paths."""
    return str(path or "").replace("\\", "/")


def _is_under(path, root):
    """True when `path` sits inside `root`.

    Compared on '/' because stored paths mix separators, and requiring the
    separator so a folder named `box` does not claim `box_archive`.
    """
    candidate = _std_path(path)
    parent = _std_path(root).rstrip("/")
    if not parent:
        return False
    if os.name == "nt":
        candidate, parent = candidate.lower(), parent.lower()
    return candidate.startswith(parent + "/")


# The typed operators a prompt search understands, and the generation_params
# column each text one looks in.
_PROMPT_SEARCH_OPERATOR = re.compile(r"^(neg|tool|model|sampler|scheduler|seed|steps|cfg):(.*)$", re.IGNORECASE)
_PROMPT_SEARCH_TEXT_COLUMNS = {
    "neg": "negative_prompt",
    "tool": "tool",
    "model": "model",
    "sampler": "sampler",
    "scheduler": "scheduler",
}


def prompt_search_condition(term, is_not, prompt_column):
    """One term of a `workflow_prompt=` search as (sql, parameter).

    Shared by the folder view and the collection view because they had a
    copy each and the copies drifted: only the folder one learned the
    typed operators, so `seed:12345` found the picture in a folder and
    found nothing at all inside an album -- which reads as "there are none
    of those here" rather than as a missing feature.

    `prompt_column` is how the caller spells the column (`workflow_prompt`
    or `f.workflow_prompt`); the typed operators join generation_params
    against `f.id`, which both queries alias the same way.

    Returns None for a term with nothing left in it.
    """
    if not term:
        return None

    typed = _PROMPT_SEARCH_OPERATOR.match(term)
    if typed:
        name = typed.group(1).lower()
        value = typed.group(2).strip().strip('"')
        if not value:
            return None
        if name in _PROMPT_SEARCH_TEXT_COLUMNS:
            inner = (
                f"SELECT 1 FROM generation_params gp "
                f"WHERE gp.file_id = f.id "
                f"AND gp.{_PROMPT_SEARCH_TEXT_COLUMNS[name]} LIKE ?"
            )
            parameter = f"%{value}%"
        else:
            try:
                parameter = float(value) if name == "cfg" else int(value)
            except ValueError:
                return None
            inner = f"SELECT 1 FROM generation_params gp WHERE gp.file_id = f.id AND gp.{name} = ?"
        return f"{'NOT ' if is_not else ''}EXISTS ({inner})", parameter

    if term.startswith('"') and term.endswith('"') and len(term) > 2:
        # A quoted phrase matches whole words, so the separators become
        # spaces and the comparison is padded on both sides.
        spaced = (
            f"(' ' || REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE("
            f"REPLACE(REPLACE(REPLACE(REPLACE({prompt_column}, ',', ' '), "
            f"'|', ' '), '.', ' '), '_', ' '), ':', ' '), '(', ' '), "
            f"')', ' '), '[', ' '), ']', ' '), char(10), ' ') || ' ')"
        )
        return (f"ulower({spaced}) {'NOT LIKE' if is_not else 'LIKE'} ulower(?)", f"% {term[1:-1]} %")

    return (f"ulower({prompt_column}) {'NOT LIKE' if is_not else 'LIKE'} ulower(?)", f"%{term}%")


def _descendant_filter(column, folder_path):
    """SQL condition and parameter selecting everything stored under
    `folder_path`, however deep.

    Stored paths mix separators: the scanner joins a folder-config path,
    which uses forward slashes, with `os.sep`. On Windows that makes a file
    sitting directly in a folder `C:/gallery/box\\a.png` and one a level
    deeper `C:/gallery/box/sub\\b.png`, so the obvious prefix of
    `folder + os.sep` matches only the first kind -- while on Linux, where
    os.sep is '/', it matches both. Folder deletes were leaving a row for
    every file in a subfolder, and ComfyUI writes into date-stamped
    subfolders by default.

    Both sides are compared with '/' so depth stops mattering, and the
    trailing separator is required so `box` does not also claim
    `box_archive`. LIKE wildcards in the folder name itself are escaped;
    a folder called `100%_keep` would otherwise match its neighbours.
    """
    prefix = _std_path(folder_path).rstrip("/")
    escaped = prefix.replace("!", "!!").replace("%", "!%").replace("_", "!_")
    return f"REPLACE({column}, '\\', '/') LIKE ? ESCAPE '!'", escaped + "/%"


def ensure_trash_folder():
    """Put the trash folder back if something removed it.

    It is made once, at startup, and every delete from then on assumes it
    is still there. It is a folder called SmartGallery inside whatever the
    person set DELETE_TO to -- somewhere they chose precisely because they
    empty it from time to time. Once it was gone, every delete failed with
    "The system cannot find the path specified", the screen said only
    "Failed to delete N files", and it went on failing that way until the
    next restart.

    Only recreated inside a DELETE_TO that still exists. If that has gone
    as well -- an unplugged drive, a share that stopped answering -- this
    returns False and the caller must refuse the delete. Creating the tree
    on top of an empty mount point would put the one copy of somebody's
    deleted files somewhere they will never think to look.
    """
    if not TRASH_FOLDER:
        return False
    if os.path.isdir(TRASH_FOLDER):
        return True
    if not os.path.isdir(DELETE_TO or ""):
        return False
    try:
        os.makedirs(TRASH_FOLDER, exist_ok=True)
    except OSError:
        return False
    print(f"{Colors.YELLOW}INFO: The trash folder had gone; it has been recreated at {TRASH_FOLDER}.{Colors.RESET}")
    return True


def _trash_destination(name):
    """A free `<timestamp>_<name>` path inside TRASH_FOLDER, disambiguated
    with a counter so same-second deletes never overwrite each other.

    Raises when the trash is unreachable, which both callers let
    propagate. That refusal is the point: deletions are configured to be
    recoverable here, so a delete that cannot be recovered must not
    happen. Neither caller may ever fall back to os.remove or rmtree.
    """
    if not ensure_trash_folder():
        raise OSError(
            f"the trash folder {TRASH_FOLDER} is missing and cannot be "
            f"recreated because {DELETE_TO} is not there either; nothing "
            f"was deleted"
        )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = os.path.join(TRASH_FOLDER, f"{timestamp}_{name}")
    counter = 1
    while os.path.exists(candidate):
        stem, ext = os.path.splitext(name)
        candidate = os.path.join(TRASH_FOLDER, f"{timestamp}_{stem}_{counter}{ext}")
        counter += 1
    return candidate


def safe_delete_tree(folder_path):
    """Delete a real directory, honouring DELETE_TO the way file deletion
    does.

    Deleting a folder used to call shutil.rmtree unconditionally, so an
    install configured for recoverable deletes still lost an entire
    directory of media permanently -- the safety net covered the smallest
    destructive action and not the largest. With DELETE_TO set the folder
    is moved into the trash instead; without it, the previous behaviour is
    unchanged. Links and junctions are the caller's business: unlinking one
    destroys nothing and must not relocate the target.
    """
    if DELETE_TO and TRASH_FOLDER:
        destination = _trash_destination(os.path.basename(folder_path.rstrip(os.sep)) or "folder")
        shutil.move(folder_path, destination)
        print(f"INFO: Moved folder to trash: {destination}")
    else:
        shutil.rmtree(folder_path)


# Two identity shapes a returning guest may legitimately present, both
# unguessable: the `guest_<hex>` this server mints (secrets.token_hex), and
# the RFC-4122 UUID the browser generates for itself when it has never been
# issued one (templates/exhibition.html, templates/index.html both use
# crypto.randomUUID and store it under `sg_exh_uuid`).
_GUEST_ID_RE = re.compile(
    r"^(guest_[0-9a-f]{8,64}"
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$"
)


def _is_guest_uuid(value):
    """True only for an identity that cannot be guessed.

    The guest login accepts a caller-supplied id so a returning visitor
    keeps their own ratings and comments. Ownership is decided by matching
    that id against a row's client_uuid, so the shape check is what keeps
    a passwordless guest from claiming a REAL account: user ids are small
    integers and the admin writes comments as the literal 'admin', all
    trivially guessable. The two accepted shapes each carry 64+ bits of
    entropy, which makes presenting one proof of having been issued it.
    """
    return bool(_GUEST_ID_RE.match(str(value or "").strip().lower()))


# No external tool may block a request, a scan, or startup indefinitely.
# ffmpeg and ffprobe hang readily on a truncated or malformed file, on a
# path that has gone away mid-read, and on network storage that stops
# answering -- and each of these runs somewhere a person is waiting: the
# version check runs before the app starts, the metadata read inside the
# scan, and the strip inside a request a visitor is waiting on.
#
# subprocess.run kills the child when the timeout expires, and every caller
# below already treats a failure as "no metadata" or "no thumbnail", so a
# timeout costs that one file rather than the operation around it.
FFPROBE_TIMEOUT = 30  # version check and metadata reads
FFMPEG_TIMEOUT = 300  # thumbnail extraction and metadata stripping


def content_digest(data):
    """A short stable name for a piece of content. Not a security check.

    File ids, thumbnail cache keys, prompt and workflow hashes are all "the
    same bytes get the same name", and they are stored in the database, so
    the digest must not change. MD5 is kept for that reason and declared
    `usedforsecurity=False`, which says so and keeps the gallery working on
    a FIPS-enforcing build, where an unmarked MD5 raises instead of hashing.

    Accepts bytes, or str which is encoded as UTF-8 -- the call sites spelled
    that out twenty-two different times, some with an explicit encoding and
    some relying on the default.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def no_console_window():
    """The creationflags every ffmpeg and ffprobe call needs on Windows.

    Without it each call flashes a console window over whatever the person
    is looking at, and a scan opens one per file. This was written out at
    fourteen call sites -- thirteen as `sys.platform == "win32"` and one as
    `os.name == "nt"` -- so the answer to "does this call raise a window"
    depended on which copy you happened to read.
    """
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def run_media_tool(cmd, timeout, capture=True):
    """Run ffmpeg or ffprobe the way every call site here needs it run.

    `check=False` is stated rather than left to the default because it is a
    decision, not an oversight: each caller reads `returncode` itself and
    treats a failure as "no metadata" or "no thumbnail" for that one file.
    Raising instead would turn a damaged file into a failed scan.

    The timeout is required, not defaulted -- both tools hang readily, and
    a hang here is a request or a scan that never returns.

    `capture=False` discards the output rather than inheriting the console:
    ffmpeg writes its progress to stderr, and a caller that does not read it
    does not want it interleaved with the gallery's own messages.
    """
    if capture:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=no_console_window(),
            check=False,
        )
    return subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
        creationflags=no_console_window(),
        check=False,
    )


# Playing a video transcodes it as it goes, so it has no total run time to
# bound -- a two hour film legitimately streams for two hours. What it must
# not do is wait forever for a frame that is not coming.
#
# That call reads from a pipe, and a read with nothing on the other end
# blocks. The reading thread is the one serving the request, so it cannot
# notice, and the clean-up only runs when the generator is closed, which a
# blocked thread never reaches. One unreadable file therefore costs a
# server thread for good; a handful of them and nothing loads at all.
#
# So the bound is on silence rather than on length: output resets it, and
# only a gap this long ends the stream.
MEDIA_STREAM_STALL_TIMEOUT = 60

# Pillow refuses to decode an image above twice this, and warns above it.
#
# Its own default is 89 megapixels, which a 16384x16384 upscale (268 Mpx)
# exceeds -- so create_thumbnail turned the guard off entirely with
# `Image.MAX_IMAGE_PIXELS = None`. That fixed a real problem and left no
# ceiling at all, for the whole process and for every later image, since the
# setting is global and one thumbnail switched it off for good. A header
# claiming 65535x65535 then asks for about 13 GB, and the process dies
# rather than the file failing.
#
# 500 Mpx decodes anything up to roughly 22000x22000 in silence and refuses
# above about 31600x31600. A refusal costs that one file its thumbnail --
# create_thumbnail already treats a raised exception as "no thumbnail" --
# instead of costing the gallery.
MAX_DECODED_PIXELS = 500_000_000
Image.MAX_IMAGE_PIXELS = MAX_DECODED_PIXELS


def _is_ffprobe(path_or_name):
    """True only when `-version` runs AND identifies itself as ffprobe.

    Exit status alone cannot tell the tools apart: ffmpeg answers
    `-version` happily, so a FFPROBE_MANUAL_PATH aimed at ffmpeg.exe (an
    easy typo, and what the launcher template used to ship) passed this
    check and then failed every metadata call, since ffmpeg rejects
    ffprobe's arguments. The banner is the discriminator -- "ffprobe
    version ..." vs "ffmpeg version ...".
    """
    try:
        proc = subprocess.run(
            [path_or_name, "-version"],
            capture_output=True,
            check=True,
            timeout=FFPROBE_TIMEOUT,
            creationflags=no_console_window(),
        )
    except Exception:
        _logger.debug("handled a failure in _is_ffprobe", exc_info=True)
        return False
    banner = (proc.stdout or b"").decode("utf-8", "replace").lower()
    return banner.startswith("ffprobe")


_FFPROBE_ANNOUNCED = False


def _announce_ffprobe(message):
    """Say which ffprobe is in use, once per run.

    The configuration table prints FFPROBE_MANUAL_PATH, which is what was
    ASKED for. When that file is absent the app quietly uses whatever is on
    PATH instead, so the table names a tool that is not the one running --
    and someone whose video thumbnails are missing reads that table first.
    Only the failures used to be reported, which left "it is configured, so
    it must be working" as the reasonable conclusion.
    """
    global _FFPROBE_ANNOUNCED
    if _FFPROBE_ANNOUNCED:
        return
    _FFPROBE_ANNOUNCED = True
    print(message)


# --- FFMPEG, FETCHED WHEN THERE IS NONE ------------------------------------
#
# ffprobe is not something anyone installs on purpose, and without it the
# gallery silently loses video duration and dimensions, video thumbnails,
# waveforms and video metadata stripping. Telling somebody to go and
# install ffmpeg is asking them to do work the gallery can do itself.
#
# Pinned to a month-end auto-build rather than the `latest` tag, which is
# rolling: the file behind a name there changes, so a digest recorded
# against it goes stale on the next rebuild. BtbN keeps one build per
# month -- checked, an unbroken monthly series back to 2024-09 -- so a
# month-end tag stays fetchable.
#
# GPL rather than LGPL is forced by what the gallery does with it: the
# video stream route runs `-vcodec libx264`, and x264 is GPL, so an LGPL
# build would download successfully and then fail every playback.
#
# Static rather than shared: one self-contained program, no libraries to
# place beside it. It costs about twice the download and removes a whole
# class of "it downloaded and still does not run".
#
# The sizes and digests below are the release's own published
# checksums.sha256, read from the pinned tag.
FFMPEG_AUTO_DOWNLOAD = env_flag("FFMPEG_AUTO_DOWNLOAD", True)

# Seconds the download may go without receiving a byte. A per-read
# timeout, not a budget for the whole transfer, so a slow connection
# still finishes; without it a stalled mirror blocks startup for ever.
FFMPEG_DOWNLOAD_STALL_TIMEOUT = 60

FFMPEG_RELEASE_TAG = "autobuild-2026-07-31-14-10"
FFMPEG_BUILDS = {
    "win32": {
        "asset": "ffmpeg-n8.1.2-34-g9b6c8969e0-win64-gpl-8.1.zip",
        "bytes": 167405723,
        "sha256": "cc4156d51387566ea8ba653fc3a04897bdf812fddf652428d9030bbf7ae24835",
        "programs": ("ffprobe.exe", "ffmpeg.exe"),
    },
    "linux": {
        "asset": "ffmpeg-n8.1.2-34-g9b6c8969e0-linux64-gpl-8.1.tar.xz",
        "bytes": 124917816,
        "sha256": "09fc77be269c7053e438b7e96548e4af97604faf96a42c4a3c56a1ad74c22c0a",
        "programs": ("ffprobe", "ffmpeg"),
    },
    # No macOS build is published here. Rather than reach for a second,
    # less verifiable source, macOS is told what to run instead -- see
    # find_ffprobe_path.
}

FFMPEG_DOWNLOAD_BASE = f"https://github.com/BtbN/FFmpeg-Builds/releases/download/{FFMPEG_RELEASE_TAG}/"


def ffmpeg_build_for_this_machine():
    """The pinned build for this platform, or None where none is published."""
    if sys.platform.startswith("linux"):
        return FFMPEG_BUILDS.get("linux")
    return FFMPEG_BUILDS.get(sys.platform)


def bundled_ffmpeg_dir():
    """Where a fetched ffmpeg lives: beside the database, not in the library."""
    return os.path.join(BASE_SMARTGALLERY_PATH, FFMPEG_FOLDER_NAME)


def bundled_ffprobe_path():
    """The fetched ffprobe, if one has already been fetched."""
    build = ffmpeg_build_for_this_machine()
    if not build:
        return None
    candidate = os.path.join(bundled_ffmpeg_dir(), build["programs"][0])
    return candidate if os.path.isfile(candidate) else None


def _extract_ffmpeg_programs(archive_path, destination, programs, asset_name=None):
    """Take just the programs out of the archive, flat, into destination.

    The archives nest everything under a versioned folder; only bin/ is
    wanted, and only two files from it. Members are matched by basename
    and written by a path this code builds, so nothing in the archive can
    choose where it lands -- a tar member is allowed to say ../../ and
    tarfile will honour it.
    """
    os.makedirs(destination, exist_ok=True)
    wanted = {name.lower() for name in programs}
    taken = {}

    def _keep(name, reader):
        base = os.path.basename(name).lower()
        if base not in wanted or base in taken:
            return
        target = os.path.join(destination, os.path.basename(name))
        with open(target + ".part", "wb") as out:
            shutil.copyfileobj(reader, out)
        os.replace(target + ".part", target)
        with contextlib.suppress(OSError):
            os.chmod(target, 0o755)
        taken[base] = target

    # The format comes from the ASSET name, not the path on disk: the
    # download lands on `<asset>.part`, so asking the path whether it ends
    # in .zip answered no for every Windows build and then tried to open a
    # zip as tar.xz. Caught by test_a_good_download_is_kept.
    looks_like = (asset_name or archive_path).lower()
    if looks_like.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                with archive.open(member) as reader:
                    _keep(member, reader)
    else:
        with tarfile.open(archive_path, "r:xz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                reader = archive.extractfile(member)
                if reader is not None:
                    _keep(member.name, reader)

    missing = sorted(wanted - set(taken))
    if missing:
        raise OSError(f"the ffmpeg archive did not contain {missing}")
    return taken


def download_progress_reporter(interval=1.0):
    """Say how a download is going, in a way that suits where it is going.

    The gallery blocks on this at startup, and the scan needs ffprobe, so
    it has to block -- but a hundred and seventy megabytes with nothing on
    the console is a startup that looks hung, which is the one thing that
    makes people kill it. It was written that way and shipped that way
    (7d8675d called fetch_ffmpeg with no reporter at all).

    On a terminal the line rewrites itself with \\r. Where output is
    redirected -- a launcher keeping a log, a service manager, ComfyUI
    starting the gallery -- \\r produces one enormous line, so that case
    gets an ordinary line every so often instead.
    """
    try:
        interactive = bool(sys.stdout.isatty())
    except Exception:
        _logger.debug("handled a failure in download_progress_reporter", exc_info=True)
        interactive = False

    state = {"at": 0.0, "shown": False}

    def report(done, total):
        now = time.monotonic()
        finished = total and done >= total
        if not finished and now - state["at"] < interval:
            return
        state["at"] = now
        state["shown"] = True
        megabytes = done / (1024 * 1024)
        if total:
            line = f"      {100.0 * done / total:5.1f}%  {megabytes:,.0f} of {total / (1024 * 1024):,.0f} MB"
        else:
            line = f"      {megabytes:,.0f} MB"
        if interactive:
            sys.stdout.write("\r" + line + "   ")
            if finished:
                sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            print(line)

    return report


def fetch_ffmpeg(progress=None):
    """Download the pinned ffmpeg and keep the two programs from it.

    Verified three ways before anything is kept: the bytes that arrived
    against the length the release publishes, the SHA-256 against the
    digest in that release's checksums.sha256, and finally the extracted
    program is run and has to identify itself as ffprobe. Nothing partial
    is ever promoted -- the download lands on a .part and the archive is
    removed either way.

    Returns the ffprobe path, or None if it could not be done. Never
    raises: no ffmpeg is a gallery without video features, not a gallery
    that will not start.
    """
    build = ffmpeg_build_for_this_machine()
    if not build:
        return None

    destination = bundled_ffmpeg_dir()
    url = FFMPEG_DOWNLOAD_BASE + build["asset"]
    megabytes = build["bytes"] / (1024 * 1024)

    print(
        f"{Colors.BLUE}INFO: no ffmpeg found. Fetching the pinned build ({megabytes:.0f} MB) from {url}{Colors.RESET}"
    )
    print(
        f"{Colors.DIM}      Set FFMPEG_AUTO_DOWNLOAD=false to stop this "
        f"and install ffmpeg yourself instead.{Colors.RESET}"
    )

    try:
        os.makedirs(destination, exist_ok=True)
    except OSError as exc:
        print(f"{Colors.YELLOW}WARNING: cannot create {destination}: {exc}{Colors.RESET}")
        return None

    archive = os.path.join(destination, build["asset"] + ".part")
    digest = hashlib.sha256()
    written = 0
    try:
        with urllib.request.urlopen(url, timeout=FFMPEG_DOWNLOAD_STALL_TIMEOUT) as response, open(archive, "wb") as out:
            declared = response.headers.get("Content-Length")
            expected = int(declared) if declared else build["bytes"]
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                if progress is not None:
                    progress(written, expected)

        if written != build["bytes"]:
            raise OSError(
                f"expected {build['bytes']:,} bytes, got {written:,} "
                f"-- the download stopped early or the pinned build "
                f"changed"
            )
        if digest.hexdigest() != build["sha256"]:
            raise OSError(f"SHA-256 mismatch: expected {build['sha256'][:16]}..., got {digest.hexdigest()[:16]}...")

        taken = _extract_ffmpeg_programs(archive, destination, build["programs"], asset_name=build["asset"])
        ffprobe = taken[build["programs"][0].lower()]
        if not _is_ffprobe(ffprobe):
            raise OSError("the program that came out of the archive does not identify itself as ffprobe")
        print(f"{Colors.GREEN}SUCCESS: ffmpeg is ready at {destination}{Colors.RESET}")
        return ffprobe
    except Exception as exc:
        _logger.debug("handled a failure in fetch_ffmpeg", exc_info=True)
        print(f"{Colors.YELLOW}WARNING: could not fetch ffmpeg: {exc}{Colors.RESET}")
        print(
            f"{Colors.YELLOW}Video features stay unavailable. Install "
            f"ffmpeg yourself and point FFPROBE_MANUAL_PATH at it."
            f"{Colors.RESET}"
        )
        return None
    finally:
        with contextlib.suppress(OSError):
            os.remove(archive)


def ffprobe_candidates_for(setting):
    """Every place ffprobe could be, given what someone pointed at.

    Nobody installs "ffprobe" -- they install ffmpeg, and ffprobe is one
    of the programs in it. So the setting gets aimed at whatever is to
    hand: ffmpeg.exe, the bin folder, the folder the download unpacked
    to. All of those are the right install, and refusing them because the
    filename is not the expected one is the gallery being difficult about
    something it can work out. The old message even told people to point
    it at "the folder holding ffprobe", which the check then rejected for
    not being a file.

    So all of these find it:

        C:/ffmpeg/bin/ffprobe.exe    the exact program
        C:/ffmpeg/bin/ffmpeg.exe     its neighbour
        C:/ffmpeg/bin                the folder
        C:/ffmpeg                    the install, via bin/

    Order matters: the named file is tried first, so pointing straight at
    ffprobe never runs anything else.
    """
    if not setting:
        return []
    names = ("ffprobe.exe", "ffprobe") if sys.platform == "win32" else ("ffprobe",)

    candidates = []
    if os.path.isfile(setting):
        candidates.append(setting)

    folder = setting if os.path.isdir(setting) else os.path.dirname(setting)
    for base in (folder, os.path.join(folder, "bin")):
        if not base:
            continue
        for name in names:
            candidate = os.path.join(base, name)
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def resolve_ffprobe_from(setting):
    """The first candidate that exists and really is ffprobe, or None."""
    for candidate in ffprobe_candidates_for(setting):
        if os.path.isfile(candidate) and _is_ffprobe(candidate):
            return candidate
    return None


def find_ffprobe_path():
    if FFPROBE_MANUAL_PATH:
        found = resolve_ffprobe_from(FFPROBE_MANUAL_PATH)
        if found:
            same = os.path.normcase(os.path.abspath(found)) == os.path.normcase(os.path.abspath(FFPROBE_MANUAL_PATH))
            _announce_ffprobe(
                f"INFO: ffprobe: {found}"
                if same
                else f"INFO: ffprobe: {found} (found from FFPROBE_MANUAL_PATH={FFPROBE_MANUAL_PATH})"
            )
            return found
        print(f"WARNING: no ffprobe at or beside FFPROBE_MANUAL_PATH ({FFPROBE_MANUAL_PATH}); falling back to PATH.")
    base_name = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
    if _is_ffprobe(base_name):
        found = shutil.which(base_name) or base_name
        _announce_ffprobe(f"INFO: ffprobe: {found} (found on PATH, not at FFPROBE_MANUAL_PATH)")
        return base_name

    # One this gallery fetched earlier. Checked before downloading again,
    # so this costs a download once and nothing on every later start.
    already = bundled_ffprobe_path()
    if already and _is_ffprobe(already):
        _announce_ffprobe(f"INFO: ffprobe: {already} (fetched by the gallery)")
        return already

    if FFMPEG_AUTO_DOWNLOAD and ffmpeg_build_for_this_machine():
        # With a reporter: this blocks startup, and the scan cannot index a
        # video without it, so the wait is right -- being silent through it
        # is not.
        fetched = fetch_ffmpeg(progress=download_progress_reporter())
        if fetched:
            _announce_ffprobe(f"INFO: ffprobe: {fetched} (fetched by the gallery)")
            return fetched
        return None

    how = "brew install ffmpeg" if sys.platform == "darwin" else "install ffmpeg"
    print(
        f"{Colors.YELLOW}WARNING: ffprobe not found. Video duration and "
        f"dimensions, video thumbnails, waveforms and video metadata "
        f"stripping are all unavailable. Run `{how}`, then point "
        f"FFPROBE_MANUAL_PATH at it -- the ffprobe program, the ffmpeg "
        f"program beside it, or the folder either lives in.{Colors.RESET}"
    )
    return None


def _validate_and_get_workflow(json_string):
    try:
        data = json.loads(json_string)

        def sanitize_for_json(obj):
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize_for_json(v) for v in obj]
            return obj

        data = sanitize_for_json(data)

        # Check for UI format (has 'nodes')
        workflow_data = data
        if isinstance(data, dict):
            workflow_data = data.get("workflow", data.get("prompt", data))

        if isinstance(workflow_data, dict):
            if "nodes" in workflow_data:
                return json.dumps(workflow_data), "ui"

            # Check for API format (keys are IDs, values have class_type)
            is_api = False
            for v in workflow_data.values():
                if isinstance(v, dict) and "class_type" in v:
                    is_api = True
                    break
            if is_api:
                return json.dumps(workflow_data), "api"

        elif isinstance(workflow_data, list):
            # Check for Array API format
            is_api = False
            for v in workflow_data:
                if isinstance(v, dict) and "class_type" in v:
                    is_api = True
                    break
            if is_api:
                return json.dumps(workflow_data), "api"

    except Exception:
        _logger.debug("ignored a failure in _validate_and_get_workflow", exc_info=True)

    return None, None


def _scan_bytes_for_workflow(content_bytes):
    """
    Generator that yields all valid JSON objects found in the byte stream.
    Searches for matching curly braces.
    """
    try:
        stream_str = content_bytes.decode("utf-8", errors="ignore")
    except Exception:
        _logger.debug("handled a failure in _scan_bytes_for_workflow", exc_info=True)
        return

    start_pos = 0
    while True:
        first_brace = stream_str.find("{", start_pos)
        if first_brace == -1:
            break

        open_braces = 0
        start_index = first_brace

        for i in range(start_index, len(stream_str)):
            char = stream_str[i]
            if char == "{":
                open_braces += 1
            elif char == "}":
                open_braces -= 1

            if open_braces == 0:
                candidate = stream_str[start_index : i + 1]
                # FIX: Use 'except Exception' to allow GeneratorExit to pass through
                try:
                    json.loads(candidate)
                    yield candidate
                except Exception:
                    _logger.debug("ignored a failure in _scan_bytes_for_workflow", exc_info=True)

                # Move start_pos to after this candidate to find the next one
                start_pos = i + 1
                break
        else:
            # If loop finishes without open_braces hitting 0, no more valid JSON here
            break


def extract_workflow(filepath, target_type="ui"):
    """
    Extracts workflow JSON from image/video files.

    Args:
        filepath (str): Path to the file.
        target_type (str): 'ui' (for visual node graph/version) or 'api' (for real execution values like Seed).
                           Defaults to 'ui' to restore original compatibility.
    """
    ext = os.path.splitext(filepath)[1].lower()
    video_exts = [".mp4", ".mkv", ".webm", ".mov", ".avi"]

    found_workflows = {}  # Stores {'ui': json_str, 'api': json_str}

    def analyze_json(json_str):
        # Helper to classify and store found workflows
        wf, wf_type = _validate_and_get_workflow(json_str)
        if wf and wf_type and wf_type not in found_workflows:
            found_workflows[wf_type] = wf

    if ext in video_exts:
        # --- FIX: Path resolution in worker processes ---
        current_ffprobe_path = FFPROBE_EXECUTABLE_PATH
        if not current_ffprobe_path:
            current_ffprobe_path = find_ffprobe_path()

        if current_ffprobe_path:
            try:
                cmd = [current_ffprobe_path, "-v", "quiet", "-print_format", "json", "-show_format", filepath]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    check=True,
                    timeout=FFPROBE_TIMEOUT,
                    creationflags=no_console_window(),
                )
                data = json.loads(result.stdout)
                if "format" in data and "tags" in data["format"]:
                    for value in data["format"]["tags"].values():
                        if isinstance(value, str) and value.strip().startswith("{"):
                            analyze_json(value)
            except Exception:
                _logger.debug("ignored a failure in extract_workflow", exc_info=True)
    else:
        try:
            with Image.open(filepath) as img:
                # Check standard keys
                for key in ["workflow", "prompt"]:
                    val = img.info.get(key)
                    if val:
                        analyze_json(val)

                # Check Exif/UserComment (for WebP/JPG)
                exif_data = img.info.get("exif")
                if exif_data and isinstance(exif_data, bytes):
                    try:
                        exif_str = exif_data.decode("utf-8", errors="ignore")
                        # Fast path: check for workflow marker
                        if "workflow:{" in exif_str:
                            start = exif_str.find("workflow:{") + len("workflow:")
                            for json_candidate in _scan_bytes_for_workflow(exif_str[start:].encode("utf-8")):
                                analyze_json(json_candidate)
                    except Exception:
                        _logger.debug("ignored a failure in extract_workflow", exc_info=True)

                    # Full scan fallback
                    for json_str in _scan_bytes_for_workflow(exif_data):
                        analyze_json(json_str)
        except Exception:
            _logger.debug("ignored a failure in extract_workflow", exc_info=True)

    # Raw byte scan (ultimate fallback)
    if not found_workflows:
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            for json_str in _scan_bytes_for_workflow(content):
                analyze_json(json_str)
                # Optimization: Stop if we found what we wanted
                if target_type in found_workflows:
                    break
        except Exception:
            _logger.debug("ignored a failure in extract_workflow", exc_info=True)

    # Return Logic:
    # 1. Return the requested type if found
    if target_type in found_workflows:
        return found_workflows[target_type]

    # 2. Fallback: If we wanted API but only have UI (or vice versa), return what we have
    if found_workflows:
        return next(iter(found_workflows.values()))

    return None


def is_webp_animated(filepath):
    try:
        with Image.open(filepath) as img:
            return getattr(img, "is_animated", False)
    except (OSError, ValueError):
        # Missing, unreadable, or not an image Pillow recognises. Not animated
        # is the safe answer for anything we cannot open.
        return False


def format_duration(seconds):
    if not seconds or seconds < 0:
        return ""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def analyze_file_metadata(filepath):
    details = {"type": "unknown", "duration": "", "dimensions": "", "has_workflow": 0}
    ext_lower = os.path.splitext(filepath)[1].lower()
    # Extended Type Map for Professional Formats
    type_map = {
        # Images
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".bmp": "image",
        ".tiff": "image",
        ".tif": "image",
        # Animations
        ".gif": "animated_image",
        # Videos (Standard & Pro)
        ".mp4": "video",
        ".webm": "video",
        ".mov": "video",
        ".mkv": "video",
        ".avi": "video",
        ".m4v": "video",
        ".wmv": "video",
        ".flv": "video",
        ".mts": "video",
        ".ts": "video",
        # Audio
        ".mp3": "audio",
        ".wav": "audio",
        ".ogg": "audio",
        ".flac": "audio",
        ".m4a": "audio",
        # Documents / Notes
        ".txt": "document",
        ".md": "document",
    }
    details["type"] = type_map.get(ext_lower, "unknown")
    if details["type"] == "unknown" and ext_lower == ".webp":
        details["type"] = "animated_image" if is_webp_animated(filepath) else "image"
    if "image" in details["type"]:
        try:
            with Image.open(filepath) as img:
                details["dimensions"] = f"{img.width}x{img.height}"
        except Exception:
            _logger.debug("ignored a failure in analyze_file_metadata", exc_info=True)
    if extract_workflow(filepath):
        details["has_workflow"] = 1
    total_duration_sec = 0
    if details["type"] == "video":
        try:
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps > 0 and count > 0:
                    total_duration_sec = count / fps
                details["dimensions"] = (
                    f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                )
                cap.release()
        except Exception:
            _logger.debug("ignored a failure in analyze_file_metadata", exc_info=True)
    elif details["type"] == "audio":
        current_ffprobe = FFPROBE_EXECUTABLE_PATH or find_ffprobe_path()

        if current_ffprobe:
            try:
                cmd_info = [
                    current_ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    filepath,
                ]

                res = run_media_tool(cmd_info, timeout=3)

                if res.stdout.strip():
                    total_duration_sec = float(res.stdout.strip())

            except Exception:
                _logger.debug("ignored a failure in analyze_file_metadata", exc_info=True)
    elif details["type"] == "animated_image":
        try:
            with Image.open(filepath) as img:
                if getattr(img, "is_animated", False):
                    if ext_lower == ".gif":
                        total_duration_sec = (
                            sum(frame.info.get("duration", 100) for frame in ImageSequence.Iterator(img)) / 1000
                        )
                    elif ext_lower == ".webp":
                        total_duration_sec = getattr(img, "n_frames", 1) / WEBP_ANIMATED_FPS
        except Exception:
            _logger.debug("ignored a failure in analyze_file_metadata", exc_info=True)
    if total_duration_sec > 0:
        details["duration"] = format_duration(total_duration_sec)
    return details


def ensure_thumbnail_cache_dir():
    """Put the thumbnail cache back if something removed it, and say so.

    The directory is made once at startup, and everything that writes a
    thumbnail then assumes it is still there. It is called
    `.thumbnails_cache`, it lives wherever the person pointed
    BASE_SMARTGALLERY_PATH -- frequently a synced folder or a second drive
    -- and disk cleaners go looking for exactly that name. Once it was
    gone, every thumbnail failed for the rest of the process's life, and
    the only sign was one console line per file quoting a temporary
    filename that says nothing about a missing folder. Restarting fixed
    it, which is not something anyone would guess.

    Only ever recreated INSIDE an existing gallery root. If the root
    itself is missing -- an unplugged drive, a share that stopped
    answering -- makedirs would cheerfully build the tree on whatever
    filesystem the empty mount point sits on, and thumbnails would pile up
    somewhere the gallery will never look again.
    """
    if os.path.isdir(THUMBNAIL_CACHE_DIR):
        return True
    if not os.path.isdir(BASE_SMARTGALLERY_PATH):
        print(
            f"{Colors.YELLOW}WARNING: The gallery folder {BASE_SMARTGALLERY_PATH} "
            f"is not there, so no thumbnail can be written. Nothing has been "
            f"created in its place.{Colors.RESET}"
        )
        return False
    try:
        os.makedirs(THUMBNAIL_CACHE_DIR, exist_ok=True)
    except OSError as exc:
        print(
            f"{Colors.YELLOW}WARNING: Could not recreate the thumbnail cache {THUMBNAIL_CACHE_DIR}: {exc}{Colors.RESET}"
        )
        return False
    print(
        f"{Colors.YELLOW}INFO: The thumbnail cache folder had gone; it has been "
        f"recreated at {THUMBNAIL_CACHE_DIR}. Thumbnails will be rebuilt as "
        f"files are viewed.{Colors.RESET}"
    )
    return True


_THUMBNAIL_KEY = re.compile(r"^[0-9a-f]{32}")


def prune_thumbnail_cache(conn):
    """Drop cache entries belonging to files the library no longer has.

    A thumbnail is named for md5(path + str(mtime)), so it is tied to one
    version of one file. Nothing ever removed one. Deleting a picture left
    its thumbnail behind, and -- the part that adds up -- so did anything
    that changed a file's mtime, since that is a different name and the
    old one is simply never asked for again. An edit, a sync tool, a
    metadata strip: each one leaves a thumbnail nothing will collect.

    Measured on six pictures, browsed so every thumbnail existed, then
    three deleted and one touched:

        after browsing 6 pics: 6 cache entries
        after deleting 3     : 6 cache entries   files in library: 3
        after touching 1     : 7 cache entries   files in library: 3

    Every name in the cache -- `<key>.jpeg`, `<key>_wave.png`,
    `<key>_wave_1.5.png`, and the `<key>/` directory of storyboard frames
    -- begins with that key, so one rule covers them all.

    An empty library means nothing is removed. A database with no rows is
    indistinguishable from one that has not been read yet, and the cost of
    guessing wrong is every thumbnail in the gallery.

    Returns (files removed, directories removed).
    """
    if not os.path.isdir(THUMBNAIL_CACHE_DIR):
        return 0, 0

    try:
        rows = conn.execute("SELECT path, mtime FROM files").fetchall()
    except sqlite3.DatabaseError as exc:
        print(f"Thumbnail Cleanup: could not read the library: {exc}")
        return 0, 0

    if not rows:
        return 0, 0

    live = {content_digest(row["path"] + str(row["mtime"])) for row in rows}

    removed_files = removed_dirs = 0
    try:
        entries = os.listdir(THUMBNAIL_CACHE_DIR)
    except OSError as exc:
        print(f"Thumbnail Cleanup: could not read {THUMBNAIL_CACHE_DIR}: {exc}")
        return 0, 0

    for entry in entries:
        if entry.startswith("tmp_"):
            continue  # a render in flight; swept on its own at startup
        match = _THUMBNAIL_KEY.match(entry)
        if not match or match.group(0) in live:
            continue
        target = os.path.join(THUMBNAIL_CACHE_DIR, entry)
        try:
            if os.path.isdir(target):
                shutil.rmtree(target)
                removed_dirs += 1
            else:
                os.remove(target)
                removed_files += 1
        except OSError:
            pass  # in use, or already gone; it will be offered again

    return removed_files, removed_dirs


def create_waveform(filepath, file_hash, file_type, amp=1.0):
    if not GENERATE_WAVEFORMS or not FFPROBE_EXECUTABLE_PATH:
        return None
    if not ensure_thumbnail_cache_dir():
        return None
    suffix = f"_{amp}" if amp != 1.0 else ""
    cache_path = os.path.join(THUMBNAIL_CACHE_DIR, f"{file_hash}_wave{suffix}.png")
    if os.path.exists(cache_path):
        return cache_path
    # ffmpeg renders to a tmp_ path, promoted only on success: a timeout
    # kill or disk-full must never leave a partial PNG at the final name,
    # which the existence check above would then serve forever.
    tmp_path = os.path.join(THUMBNAIL_CACHE_DIR, f"tmp_{file_hash}_wave{suffix}.png")

    try:
        ffmpeg_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        ffmpeg_bin = os.path.join(os.path.dirname(FFPROBE_EXECUTABLE_PATH), ffmpeg_name)
        if not os.path.exists(ffmpeg_bin):
            ffmpeg_bin = ffmpeg_name

        # Generates a white waveform on black background
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            filepath,
            "-filter_complex",
            f"volume={amp},showwavespic=s=1000x120:colors=white",
            "-frames:v",
            "1",
            "-c:v",
            "png",
            tmp_path,
        ]
        run_media_tool(cmd, timeout=20, capture=False)
        if os.path.exists(tmp_path):
            os.replace(tmp_path, cache_path)
            return cache_path
    except Exception:
        _logger.debug("ignored a failure in create_waveform", exc_info=True)  # Silently fail if corrupted or timeout
    with contextlib.suppress(OSError):
        os.remove(tmp_path)
    return None


# Site setting cache for thumbnail_generation_enabled(); per-process, short
# TTL so scan workers and the web process converge quickly after a toggle.
_THUMBNAIL_SETTING_CACHE = {"value": None, "read_at": 0.0}


def thumbnail_generation_enabled():
    """Site setting: spend CPU on server-side thumbnail generation?

    Stored in ai_metadata (key 'thumbnail_generation', '1'/'0') so it survives
    restarts; the GENERATE_THUMBNAILS env default applies while unset.
    Toggling never touches existing cached thumbnails: cache keys are
    md5(path + mtime), so cached thumbs keep serving either way and
    regeneration only ever happens for files whose content changed."""
    now = time.time()
    if _THUMBNAIL_SETTING_CACHE["value"] is not None and now - _THUMBNAIL_SETTING_CACHE["read_at"] < 5:
        return _THUMBNAIL_SETTING_CACHE["value"]
    value = GENERATE_THUMBNAILS
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT value FROM ai_metadata WHERE key = 'thumbnail_generation'").fetchone()
        if row is not None:
            value = row[0] != "0"
    except Exception:
        _logger.debug(
            "ignored a failure in thumbnail_generation_enabled", exc_info=True
        )  # missing table (first boot): fall back to the env default
    _THUMBNAIL_SETTING_CACHE["value"] = value
    _THUMBNAIL_SETTING_CACHE["read_at"] = now
    return value


def create_thumbnail(filepath, file_hash, file_type):
    # The pixel ceiling is set once at import (MAX_DECODED_PIXELS). It used
    # to be cleared here, which switched the guard off process-wide the
    # first time any thumbnail was made.

    # Startup made this folder; a cleaner or a sync client may have taken
    # it away since. Every branch below writes into it.
    if not ensure_thumbnail_cache_dir():
        return None

    # --- IMAGES / ANIMATIONS ---
    # All encoders write to a tmp_ path and os.replace() into place: a save
    # that dies mid-write (disk full, process kill) must never leave a
    # partial file at the final name -- the caller's existence check would
    # treat the corrupt thumbnail as done forever. tmp_ does not match the
    # caller's f"{file_hash}.*" glob.
    if file_type in ["image", "animated_image"]:
        tmp_path = None
        try:
            with Image.open(filepath) as img:
                fmt = "gif" if img.format == "GIF" else "webp" if img.format == "WEBP" else "jpeg"
                cache_path = os.path.join(THUMBNAIL_CACHE_DIR, f"{file_hash}.{fmt}")
                tmp_path = os.path.join(THUMBNAIL_CACHE_DIR, f"tmp_{file_hash}.{fmt}")

                # Handle Animations (Animated WebP / GIF)
                if file_type == "animated_image" and getattr(img, "is_animated", False):
                    frames = [fr.copy() for fr in ImageSequence.Iterator(img)]
                    if frames:
                        for frame in frames:
                            frame.thumbnail((THUMBNAIL_WIDTH, THUMBNAIL_WIDTH * 2), Image.Resampling.LANCZOS)

                        processed_frames = [frame.convert("RGBA").convert("RGB") for frame in frames]
                        if processed_frames:
                            processed_frames[0].save(
                                tmp_path,
                                save_all=True,
                                append_images=processed_frames[1:],
                                duration=img.info.get("duration", 100),
                                loop=img.info.get("loop", 0),
                                optimize=True,
                            )
                            os.replace(tmp_path, cache_path)
                            return cache_path

                # Handle Static Images
                else:
                    img.thumbnail((THUMBNAIL_WIDTH, THUMBNAIL_WIDTH * 2), Image.Resampling.LANCZOS)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(tmp_path, "JPEG", quality=85)
                    os.replace(tmp_path, cache_path)
                    return cache_path

        except Exception as e:
            _logger.debug("handled a failure in create_thumbnail", exc_info=True)
            print(f"ERROR (Pillow): Thumbnail failed for {os.path.basename(filepath)}: {e}")
            if tmp_path:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)

    # --- VIDEOS (MP4, MOV, MKV, AVI, etc.) ---
    elif file_type == "video":
        cache_path = os.path.join(THUMBNAIL_CACHE_DIR, f"{file_hash}.jpeg")
        tmp_path = os.path.join(THUMBNAIL_CACHE_DIR, f"tmp_{file_hash}.jpeg")

        # Method A: Try OpenCV first (Fastest)
        try:
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                success, frame = cap.read()
                cap.release()
                if success:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img.thumbnail((THUMBNAIL_WIDTH, THUMBNAIL_WIDTH * 2), Image.Resampling.LANCZOS)
                    img.save(tmp_path, "JPEG", quality=80)
                    os.replace(tmp_path, cache_path)
                    return cache_path
        except Exception:
            _logger.debug("handled a failure in create_thumbnail", exc_info=True)
            with contextlib.suppress(OSError):
                os.remove(tmp_path)
            # Fallback silently to FFmpeg

        # Method B: Fallback to FFmpeg (Most Robust for MKV/AVI/ProRes)
        if FFPROBE_EXECUTABLE_PATH:
            try:
                ffmpeg_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
                ffmpeg_bin = os.path.join(os.path.dirname(FFPROBE_EXECUTABLE_PATH), ffmpeg_name)
                if not os.path.exists(ffmpeg_bin):
                    ffmpeg_bin = ffmpeg_name

                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    filepath,
                    "-ss",
                    "00:00:00",  # Seek to start
                    "-vframes",
                    "1",  # Grab 1 frame
                    "-vf",
                    f"scale={THUMBNAIL_WIDTH}:-1",  # Resize directly
                    "-q:v",
                    "2",  # High Quality
                    tmp_path,
                ]

                creation_flags = no_console_window()
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                    timeout=FFMPEG_TIMEOUT,
                    creationflags=creation_flags,
                )

                if os.path.exists(tmp_path):
                    os.replace(tmp_path, cache_path)
                    return cache_path
            except Exception as e:
                _logger.debug("handled a failure in create_thumbnail", exc_info=True)
                print(f"ERROR (FFmpeg): Thumbnail failed for {os.path.basename(filepath)}: {e}")
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)

    return None


def extract_workflow_files_string(workflow_json_string):
    """
    Parses workflow and returns a normalized string containing ONLY filenames
    (models, images, videos) used in the workflow.

    Robust version: Handles both UI (widgets_values) and API (inputs) formats safely.
    Filters out prompts, settings, and comments based on extensions and path structure.
    """
    if not workflow_json_string:
        return ""

    try:
        data = json.loads(workflow_json_string)
    except (json.JSONDecodeError, TypeError):
        return ""

    # Normalize structure (UI vs API format)
    nodes = []
    if isinstance(data, dict):
        if "nodes" in data and isinstance(data["nodes"], list):
            nodes = data["nodes"]  # UI Format
        else:
            # API Format fallback (Dict of nodes)
            # We convert it to a list for uniform processing
            nodes = list(data.values())
    elif isinstance(data, list):
        nodes = data  # Raw list format

    # 1. Blocklist Nodes (Comments and structural nodes)
    ignored_types = {"Note", "NotePrimitive", "Reroute", "PrimitiveNode"}

    # 2. Whitelist Extensions (The most important filter)
    valid_extensions = {
        # Models
        ".safetensors",
        ".ckpt",
        ".pt",
        ".pth",
        ".bin",
        ".gguf",
        ".lora",
        ".sft",
        # Images
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".bmp",
        ".tiff",
        # Video/Audio
        ".mp4",
        ".mov",
        ".webm",
        ".mkv",
        ".avi",
        ".mp3",
        ".wav",
        ".ogg",
        ".flac",
        ".m4a",
    }

    found_tokens = set()

    for node in nodes:
        if not isinstance(node, dict):
            continue

        node_type = node.get("type", node.get("class_type", ""))

        # Skip comment nodes
        if node_type in ignored_types:
            continue

        # Collect values to check from BOTH formats to be safe
        values_to_check = []

        # UI Format values
        w_vals = node.get("widgets_values")
        if isinstance(w_vals, list):
            values_to_check.extend(w_vals)

        # API Format inputs
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            values_to_check.extend(inputs.values())
        elif isinstance(inputs, list):
            values_to_check.extend(inputs)

        for val in values_to_check:
            # CRITICAL: Only process Strings. API inputs contain Ints/Floats/Lists(links).
            if isinstance(val, str) and val.strip():
                # Normalize immediately
                norm_val = normalize_smart_path(val.strip())

                # --- FILTER LOGIC ---

                # Check A: Valid Extension?
                # We check if the string ends with one of the valid extensions
                has_valid_ext = any(norm_val.endswith(ext) for ext in valid_extensions)

                # Check B: Absolute Path? (For folders or files without standard extensions)
                # Matches "c:/..." or "/home/..."
                # Must be shorter than 260 chars to avoid catching long prompts starting with /
                is_abs_path = (len(norm_val) < 260) and (
                    (len(norm_val) > 2 and norm_val[1] == ":")  # Windows Drive (c:)
                    or norm_val.startswith("/")  # Unix/Linux root
                )

                # Keep ONLY if it looks like a file/path
                if has_valid_ext or is_abs_path:
                    found_tokens.add(norm_val)

    return " ||| ".join(sorted(found_tokens))


# --- Helper to filter out garbage text (Markdown, Stats, Instructions, UI values) ---
def _is_garbage_text(text):
    if not text:
        return True
    t = text.strip()
    # Ignore very short strings
    if len(t) < 3:
        return True

    # 1. Detect Markdown Tables / System Stats
    if "|" in t and ("---" in t or "VRAM" in t or "Model" in t):
        return True
    if "GPU:" in t or "RTX" in t or "it/s" in t:
        return True

    # 2. Detect Instructions / Notes / Shortcuts / UI Trash
    t_lower = t.lower()

    # List of phrases that identify non-prompt text.
    # Simply add or remove strings here to update the filter.
    GARBAGE_MARKERS = (
        "ctrl +",
        "box-select",
        "don't forget to use",
        "partial - execution",
        "creative prompt",
        "bad quality",
        "embedding:",
        "🟢",
        "select wildcard",
        "by percentage",
        "what is art?",
        "send none",
        "you are an ai artist",
        "jpeg压缩残留",
        "/",
        "select the wildcard",
    )

    # If any of the markers are found in the text, it is considered garbage
    if any(marker in t_lower for marker in GARBAGE_MARKERS):
        return True

    # 3. Detect URLs
    if "http://" in t_lower or "https://" in t_lower:
        return True

    # 4. Detect Numbered Lists (common in notes: "1. do this")
    if len(t) > 3 and t[0].isdigit() and t[1] == "." and t[2] == " ":
        return True

    # 5. Detect Technical/UI Parameters (Extended Blacklist)
    ui_keywords = {
        "enable",
        "disable",
        "fixed",
        "randomize",
        "auto",
        "simple",
        "always",
        "center",
        "left",
        "top",
        "bottom",
        "right",
        "nearest",
        "bilinear",
        "bicubic",
        "lanczos",
        "keep proportion",
        "image",
        "default",
        "comfyui",
        "wan",
        "crop",
        "input",
        "output",
        "float",
        "int",
        "boolean",
        # Samplers & Schedulers
        "euler",
        "euler_a",
        "heun",
        "dpm_2",
        "dpmpp_2m",
        "dpmpp_sde",
        "ddim",
        "uni_pc",
        "lms",
        "karras",
        "exponential",
        "sgd",
        "normal",
    }

    # Check exact match or if it looks like a parameter
    if t_lower in ui_keywords:
        return True

    # 6. Detect Unresolved variables
    return bool(t.startswith("%") or "${" in t)


def extract_workflow_prompt_string(workflow_json_string):
    """
    Broad extraction for Database Indexing (Searchable Keywords).
    This function scans ALL nodes to ensure keyword searches work as expected,
    while filtering out known UI noise and technical instructions.
    """
    if not workflow_json_string:
        return ""

    try:
        data = json.loads(workflow_json_string)
    except (json.JSONDecodeError, TypeError):
        return ""

    # Normalize structure (UI vs API format)
    nodes = []
    if isinstance(data, dict):
        if "nodes" in data and isinstance(data["nodes"], list):
            nodes = data["nodes"]  # UI Format
        else:
            nodes = list(data.values())  # API Format
    elif isinstance(data, list):
        nodes = data

    found_texts = set()

    # Nodes to strictly ignore for text extraction
    ignored_types = {
        "Note",
        "NotePrimitive",
        "Reroute",
        "PrimitiveNode",
        "ShowText",
        "Display Text",
        "Simple Text",
        "Text Box",
        "ComfyUI",
        "ExtraMetadata",
        "SaveImage",
        "PreviewImage",
        "VHS_VideoCombine",
        "VHS_LoadVideo",
    }

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = node.get("type", node.get("class_type", "")).strip()

        if node_type in ignored_types:
            continue

        # Collect all possible string values from widgets and inputs
        values_to_check = []
        if "widgets_values" in node and isinstance(node["widgets_values"], list):
            values_to_check.extend(node["widgets_values"])
        if "inputs" in node and isinstance(node["inputs"], dict):
            values_to_check.extend(node["inputs"].values())

        for val in values_to_check:
            if isinstance(val, str) and val.strip():
                text = val.strip()

                # --- BROAD FILTERING FOR SEARCH ACCURACY ---

                # A. Global Blacklist check
                if text in WORKFLOW_PROMPT_BLACKLIST:
                    continue

                # B. Advanced Garbage filtering (Instructions, technical values, etc.)
                if _is_garbage_text(text):
                    continue

                # C. Ignore filenames and short numeric strings
                if text.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".safetensors", ".ckpt", ".pt")):
                    continue

                # D. Minimum length for a searchable keyword
                if len(text) < 3:
                    continue

                found_texts.add(text)

    # Join everything with a separator for the Database field
    return " , ".join(list(found_texts))


def process_single_file(filepath):
    """
    Worker function to perform all heavy processing for a single file.
    Designed to be run in a parallel process pool.
    """
    try:
        mtime = os.path.getmtime(filepath)
        metadata = analyze_file_metadata(filepath)
        file_hash_for_thumbnail = content_digest(filepath + str(mtime))

        if thumbnail_generation_enabled() and not glob.glob(
            os.path.join(THUMBNAIL_CACHE_DIR, f"{file_hash_for_thumbnail}.*")
        ):
            create_thumbnail(filepath, file_hash_for_thumbnail, metadata["type"])

        if GENERATE_WAVEFORMS and metadata["type"] in ["video", "audio"]:
            create_waveform(filepath, file_hash_for_thumbnail, metadata["type"])

        file_id = content_digest(filepath)
        file_size = os.path.getsize(filepath)

        # Extract workflow data
        workflow_files_content = ""
        workflow_prompt_content = ""

        if metadata["has_workflow"]:
            # UPDATED: Request 'api' format for indexing to get real execution values (seeds, clean prompts)
            # If not found, extract_workflow will automatically fallback to 'ui'
            wf_json = extract_workflow(filepath, target_type="api")

            if wf_json:
                workflow_files_content = extract_workflow_files_string(wf_json)
                workflow_prompt_content = extract_workflow_prompt_string(wf_json)

        # Non-ComfyUI generators (A1111/Forge, SwarmUI, Fooocus, InvokeAI, ...):
        # parse their metadata once -- it feeds both the searchable prompt
        # field and the typed generation_params row below.
        # Stealth (LSB) decoding stays off here -- too costly for bulk scans.
        parsed_meta = None
        if not (metadata["has_workflow"] and wf_json) and metadata["type"] in ("image", "animated_image"):
            parsed_meta = metaparse.parse_file(filepath, allow_stealth=False)
        if not workflow_prompt_content and parsed_meta and parsed_meta.positive:
            workflow_prompt_content = parsed_meta.positive

        # The same backfill for the models, which was missing. workflow_files
        # is what the Models search box and the "models used" list read, and
        # only the ComfyUI branch above ever filled it -- so an A1111 or Forge
        # picture was searchable by its prompt and invisible to a search for
        # the checkpoint that made it, while `model:` in the prompt box found
        # it through generation_params. One idea, two fields, two answers.
        #
        # Clustering is unaffected: the foreign hashes come from the parsed
        # metadata, never from this field.
        if not workflow_files_content and parsed_meta:
            foreign_models = []
            model_name = str(parsed_meta.params.get("model") or "").strip()
            if model_name:
                foreign_models.append(model_name)
            for lora in re.finditer(r"<lora:([^:>]+)", parsed_meta.positive or ""):
                name = lora.group(1).strip()
                if name and name not in foreign_models:
                    foreign_models.append(name)
            if foreign_models:
                workflow_files_content = " ||| ".join(foreign_models)

        # First-class typed generation parameters (metaparse.typed):
        # ComfyUI graphs go through the sampler-tracing parser; every
        # other tool through its metaparse adapter.
        gen_row = None
        try:
            if metadata["has_workflow"] and wf_json:
                graph_meta = ComfyMetadataParser(json.loads(wf_json)).parse()
                gp = metaparse_typed.GenerationParams.from_comfy(graph_meta)
                if gp.has_content:
                    gen_row = gp.to_row(file_id, time.time())
            elif parsed_meta is not None:
                gp = metaparse_typed.GenerationParams.from_parsed(parsed_meta)
                if gp.has_content or parsed_meta.params:
                    gen_row = gp.to_row(file_id, time.time())
        except Exception:
            _logger.debug("handled a failure in process_single_file", exc_info=True)
            gen_row = None

        hashable = metadata["has_workflow"] or metadata["type"] in ("image", "animated_image")
        wf_hash, pr_hash, md_hash = compute_workflow_hashes(filepath) if hashable else ("", "", "")
        return (
            file_id,
            filepath,
            mtime,
            os.path.basename(filepath),
            metadata["type"],
            metadata["duration"],
            metadata["dimensions"],
            metadata["has_workflow"],
            file_size,
            time.time(),
            workflow_files_content,
            workflow_prompt_content,
            wf_hash,
            pr_hash,
            md_hash,
            gen_row,
        )
    except Exception as e:
        _logger.debug("handled a failure in process_single_file", exc_info=True)
        print(f"ERROR: Failed to process file {os.path.basename(filepath)} in worker: {e}")
        return None


def split_file_results(results):
    """process_file returns the 15-column files row plus its
    generation_params row (or None); split them for the two upserts."""
    file_rows = [r[:-1] for r in results]
    gen_rows = [r[-1] for r in results if r[-1] is not None]
    gen_deletes = [(r[0],) for r in results if r[-1] is None]
    return file_rows, gen_rows, gen_deletes


_GENPARAMS_UPSERT = (
    "INSERT OR REPLACE INTO generation_params ("
    + ", ".join(metaparse_typed.ROW_COLUMNS)
    + ") VALUES ("
    + ", ".join("?" * len(metaparse_typed.ROW_COLUMNS))
    + ")"
)


def upsert_generation_params(conn, gen_rows, gen_deletes):
    """Write typed generation rows; a re-indexed file whose metadata
    vanished loses its stale row."""
    if gen_deletes:
        conn.executemany("DELETE FROM generation_params WHERE file_id = ?", gen_deletes)
    if gen_rows:
        conn.executemany(_GENPARAMS_UPSERT, gen_rows)


def _unicode_lower(value):
    """Fold case for comparison, in every script rather than just ASCII.

    SQLite folds case for ASCII only, so LIKE matched CAFE against cafe but
    not CAFÉ against café, and never РИСУНОК against рисунок. Searching by
    name quietly worked in English and failed for everyone else, and it
    fails by returning nothing -- which reads as an empty library rather
    than a broken comparison.

    casefold rather than lower: lower() leaves the German ß alone, so
    STRASSE would still miss straße. casefold is the operation meant for
    caseless matching and maps the pair together.
    """
    return value.casefold() if isinstance(value, str) else value


def check_library_continuity(is_new_database):
    """Say so when a library that existed has been replaced by an empty one.

    sqlite3.connect CREATES the file when it is missing, so losing
    gallery_cache.sqlite -- an antivirus quarantine, a sync conflict, a
    tidy-up of a folder named cache -- raises nothing at all. The gallery
    builds a fresh schema, rescans the pictures, and looks exactly like a
    new install. The pictures are all still there, which is what makes it
    convincing; every rating, comment, album, collection and tag is not,
    and nothing said a word.

    A marker beside the cache rather than inside it separates the two
    cases. A genuinely first run has no marker and is told nothing. A
    folder that has held a library before, now holding an empty database,
    is told plainly and early, while restoring a backup is still an option.

    Returns True when it warned, so the tests can assert on the decision
    rather than on console text.
    """
    try:
        had_library = os.path.exists(LIBRARY_MARKER_FILE)
    except OSError:
        return False

    replaced = bool(is_new_database and had_library)
    if replaced:
        print(
            f"\n{Colors.RED}{Colors.BOLD}WARNING: This gallery folder has held "
            f"a library before, but the database is empty.{Colors.RESET}"
        )
        print(f"{Colors.RED}{DATABASE_FILE} was missing, so a new empty one has been created.{Colors.RESET}")
        print(
            f"{Colors.YELLOW}Your picture files are untouched. Ratings, "
            f"comments, albums, collections and tags lived in that database "
            f"and are not in the new one.{Colors.RESET}"
        )
        print(
            f"{Colors.YELLOW}If you have a backup of it, stop the gallery and "
            f"put it back before browsing; a scan will refill the new "
            f"database and there is nothing to merge afterwards.{Colors.RESET}\n"
        )

    if not had_library:
        try:
            with open(LIBRARY_MARKER_FILE, "w", encoding="utf-8") as marker:
                marker.write(
                    "This file records that a SmartGallery library was created "
                    "here.\nIt lets the gallery tell a first run apart from a "
                    "database that has gone missing.\nDeleting it is harmless; "
                    "it only costs you that warning.\n"
                )
        except OSError:
            pass  # a read-only gallery folder must not stop the gallery

    return replaced


def ensure_sqlite_cache_dir():
    """Put the database folder back if something removed it.

    Only inside a gallery root that still exists, for the same reason as
    the thumbnail cache and the trash: an unplugged drive leaves a
    writable empty mount point, and building the tree there would start a
    second, invisible library on the wrong filesystem.
    """
    if os.path.isdir(SQLITE_CACHE_DIR):
        return True
    if not os.path.isdir(BASE_SMARTGALLERY_PATH):
        return False
    try:
        os.makedirs(SQLITE_CACHE_DIR, exist_ok=True)
    except OSError:
        return False
    # Only news if a library was here. On a first start this folder is
    # simply being made, and telling someone it "had gone" would be a
    # false alarm on their very first run.
    if os.path.exists(LIBRARY_MARKER_FILE):
        print(
            f"{Colors.YELLOW}INFO: The database folder had gone; it has been "
            f"recreated at {SQLITE_CACHE_DIR}.{Colors.RESET}"
        )
    return True


def get_db_connection():
    # Timeout increased to 60s to be patient with the Indexer
    try:
        conn = sqlite3.connect(DATABASE_FILE, timeout=60)
    except sqlite3.OperationalError:
        # "unable to open database file" is what a removed .sqlite_cache
        # looks like, and it would otherwise be the answer to every
        # request for the rest of the session. Retried once, and only
        # once: a genuine problem still raises.
        if not ensure_sqlite_cache_dir():
            raise
        conn = sqlite3.connect(DATABASE_FILE, timeout=60)
    conn.row_factory = sqlite3.Row
    # CONCURRENCY OPTIMIZATION:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    # --- CRITICAL FOR DATA CONSISTENCY ---
    # Enables cascading updates/deletes for Categories/Collections
    conn.execute("PRAGMA foreign_keys = ON;")
    # Case folding that covers every script, for the search conditions.
    conn.create_function("ulower", 1, _unicode_lower, deterministic=True)
    # The model/LoRA filter folds the stored name and the typed one with
    # these same two functions, so the two sides cannot disagree.
    conn.create_function("fuzzykey", 1, _normalize_fuzzy_string, deterministic=True)
    conn.create_function("wordkey", 1, _word_key, deterministic=True)

    return conn


def _file_id_tables(conn):
    """Every table carrying a `file_id` that points at `files(id)`."""
    names = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]
    tables = []
    for name in names:
        columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{name}")').fetchall()]
        if "file_id" in columns:
            tables.append(name)
    return tables


def _reassign_file_ids(conn, pairs):
    """Move everything belonging to a file onto its new id.

    A file's id is the md5 of its path, so renaming a file -- or the folder
    above it -- gives it a new one. Nine tables declare a foreign key onto
    files(id) with ON DELETE CASCADE and no ON UPDATE clause, so changing
    files.id on its own raises "FOREIGN KEY constraint failed": renaming
    anything failed outright once a file inside had been rated or commented
    on. Six more tables hold a file_id with no constraint at all, and would
    have been quietly orphaned instead.

    Both are handled by moving the children first, under deferred foreign
    keys so the pair of updates is judged as one at commit. Discovery is by
    column rather than by a hand-kept list, so a table added later comes
    along on its own.

    `UPDATE OR REPLACE` covers the case where the destination id already has
    a row for the same key -- a rating by the same user, say -- which would
    otherwise collide on the primary key.
    """
    pairs = [(old, new) for old, new in pairs if old != new]
    if not pairs:
        return
    # The deferral lasts until the transaction ends, and is cleared by every
    # commit -- so there has to be a transaction for it to belong to. A caller
    # that has only read so far is still in autocommit, where the setting
    # would be dropped again before the updates ran.
    if not conn.in_transaction:
        conn.execute("BEGIN")
    conn.execute("PRAGMA defer_foreign_keys = ON")
    moves = [(new, old) for old, new in pairs]
    for table in _file_id_tables(conn):
        conn.executemany(f'UPDATE OR REPLACE "{table}" SET file_id = ? WHERE file_id = ?', moves)


def init_db(conn=None):
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    # A database with no `files` table has never been used. Everything below
    # adds tables and columns with IF NOT EXISTS, so a first run performed
    # the whole migration history and announced each step: a new user's very
    # first start reported six schema updates and a version upgrade, which
    # reads as "already out of date" rather than "created". The steps still
    # run; only the commentary is withheld.
    try:
        is_new_database = (
            conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='files'").fetchone() is None
        )
    except Exception:
        _logger.debug("handled a failure in init_db", exc_info=True)
        is_new_database = False

    try:
        # 1. CORE TABLE CREATION
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                mtime REAL NOT NULL,
                name TEXT NOT NULL,
                type TEXT,
                duration TEXT,
                dimensions TEXT,
                has_workflow INTEGER,
                is_favorite INTEGER DEFAULT 0,
                size INTEGER DEFAULT 0,
                last_scanned REAL DEFAULT 0,
                workflow_files TEXT DEFAULT '',
                workflow_prompt TEXT DEFAULT '',
                ai_last_scanned REAL DEFAULT 0,
                ai_caption TEXT,
                ai_embedding BLOB,
                ai_error TEXT
            )
        """)

        # First-class generation parameters, one typed row per file
        # (metaparse.typed.GenerationParams.to_row order). Numeric fields
        # carry real SQL types; unmapped tool keys live verbatim in
        # `extra` JSON so no first-party field is ever dropped.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generation_params (
                file_id TEXT PRIMARY KEY REFERENCES files(id)
                    ON DELETE CASCADE ON UPDATE CASCADE,
                tool TEXT NOT NULL,
                detection TEXT NOT NULL,
                positive_prompt TEXT NOT NULL DEFAULT '',
                negative_prompt TEXT NOT NULL DEFAULT '',
                model TEXT,
                model_hash TEXT,
                sampler TEXT,
                scheduler TEXT,
                seed INTEGER,
                steps INTEGER,
                cfg REAL,
                width INTEGER,
                height INTEGER,
                denoise REAL,
                clip_skip INTEGER,
                version TEXT,
                loras TEXT,
                extra TEXT,
                parsed_at REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_genparams_tool ON generation_params(tool)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_genparams_model ON generation_params(model)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_genparams_seed ON generation_params(seed)")

        # 2. AI TABLES
        conn.execute("""
            CREATE TABLE IF NOT EXISTS omniquery_sessions (
                session_id TEXT PRIMARY KEY,
                raw_sql TEXT,
                created_at REAL
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS omniquery_results (
                session_id TEXT,
                file_id TEXT,
                FOREIGN KEY (session_id) REFERENCES omniquery_sessions(session_id) ON DELETE CASCADE
            );
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_omniquery_results ON omniquery_results(session_id);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_search_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                query TEXT NOT NULL,
                limit_results INTEGER DEFAULT 100,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP NULL
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_search_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                score REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES ai_search_queue(session_id)
            );
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_status ON ai_search_queue(status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_session ON ai_search_results(session_id);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_indexing_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                file_id TEXT,
                status TEXT DEFAULT 'pending',
                force_index INTEGER DEFAULT 0,
                params TEXT DEFAULT '{}',
                created_at REAL,
                updated_at REAL,
                error_msg TEXT,
                UNIQUE(file_path) ON CONFLICT REPLACE
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ai_idx_status ON ai_indexing_queue(status);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_watched_folders (
                path TEXT PRIMARY KEY,
                recursive INTEGER DEFAULT 0,
                added_at REAL
            );
        """)

        conn.execute("CREATE TABLE IF NOT EXISTS ai_metadata (key TEXT PRIMARY KEY, value TEXT, updated_at REAL)")

        # MOUNT POINTS TABLE
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mounted_folders (
                path TEXT PRIMARY KEY,
                target_source TEXT,
                created_at REAL
            );
        """)

        # 3. COLLECTIONS SYSTEM
        conn.execute("""
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                color TEXT,
                is_public INTEGER DEFAULT 0,
                parent_id INTEGER DEFAULT NULL,
                created_at REAL
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS collection_files (
                collection_id INTEGER,
                file_id TEXT,
                added_at REAL,
                PRIMARY KEY (collection_id, file_id),
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE ON UPDATE CASCADE,
                FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
            );
        """)

        # 4. EXHIBITION MODE TABLES (Ratings & Comments)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_ratings (
                file_id TEXT,
                client_uuid TEXT,
                rating INTEGER CHECK(rating >= 1 AND rating <= 5),
                created_at REAL,
                PRIMARY KEY (file_id, client_uuid),
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT,
                client_uuid TEXT,
                author_name TEXT,
                comment_text TEXT,
                target_audience TEXT DEFAULT 'public',
                created_at REAL,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            );
        """)

        # Pre-populate Standard Workflow Flags
        system_flags = [
            ("Approved", "system_flag", "#28a745"),
            ("Review", "system_flag", "#ffc107"),
            ("To Edit", "system_flag", "#17a2b8"),
            ("Rejected", "system_flag", "#dc3545"),
            ("Select", "system_flag", "#6f42c1"),
        ]

        existing_cols = conn.execute("SELECT COUNT(*) FROM collections WHERE type='system_flag'").fetchone()[0]
        if existing_cols == 0:
            print(f"{Colors.BLUE}INFO: Initializing standard workflow tags...{Colors.RESET}")
            conn.executemany(
                "INSERT INTO collections (name, type, color, is_public, created_at) VALUES (?, ?, ?, 0, ?)",
                [(n, t, c, time.time()) for n, t, c in system_flags],
            )

        # 5. USER MANAGEMENT (Always required now for messaging target resolution)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,         -- Argon2id hash (sg_auth); never reversible
                full_name TEXT NOT NULL,
                email TEXT,                     -- Communication email
                phone_number TEXT,              -- Optional contact
                role TEXT CHECK(role IN
                    ('USER', 'STAFF', 'MANAGER', 'CUSTOMER', 'FRIEND', 'GUEST', 'ADMIN')
                ) DEFAULT 'GUEST',
                start_date DATE DEFAULT CURRENT_DATE,
                expiry_date DATE,               -- Optional expiration date
                is_active BOOLEAN DEFAULT 1     -- 1 = Active, 0 = Disabled
            );
        """)
        # Index for faster login lookups
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")

        # --- MIGRATION: Add last_login to users table ---
        try:
            cursor_usr = conn.execute("PRAGMA table_info(users)")
            usr_columns = {row["name"] for row in cursor_usr.fetchall()}
            if "last_login" not in usr_columns:
                if not is_new_database:
                    print("INFO: Updating Database Schema... Adding 'last_login' to users")
                conn.execute("ALTER TABLE users ADD COLUMN last_login REAL")
        except Exception as e:
            _logger.debug("handled a failure in init_db", exc_info=True)
            print(f"WARNING: Could not migrate users table: {e}")

        # 6. COLUMN MIGRATION
        required_columns = {
            "size": "INTEGER DEFAULT 0",
            "last_scanned": "REAL DEFAULT 0",
            "workflow_files": "TEXT DEFAULT ''",
            "workflow_prompt": "TEXT DEFAULT ''",
            "ai_last_scanned": "REAL DEFAULT 0",
            "ai_caption": "TEXT",
            "ai_embedding": "BLOB",
            "ai_error": "TEXT",
            "workflow_hash": "TEXT DEFAULT ''",
            "prompt_hash": "TEXT DEFAULT ''",
            "models_hash": "TEXT DEFAULT ''",
            "hash_failed": "INTEGER DEFAULT 0",
        }

        # Handle comments target migration
        try:
            cursor_fc = conn.execute("PRAGMA table_info(file_comments)")
            fc_columns = {row["name"] for row in cursor_fc.fetchall()}
            if "target_audience" not in fc_columns:
                if not is_new_database:
                    print("INFO: Updating Database Schema... Adding 'target_audience' to file_comments")
                conn.execute("ALTER TABLE file_comments ADD COLUMN target_audience TEXT DEFAULT 'public'")
        except Exception as e:
            _logger.debug("handled a failure in init_db", exc_info=True)
            print(f"WARNING: Could not migrate file_comments table: {e}")

        # Handle collections public flag migration
        try:
            cursor_col = conn.execute("PRAGMA table_info(collections)")
            col_columns = {row["name"] for row in cursor_col.fetchall()}
            # Auto-fix existing txt/md files in database from unknown to document
            conn.execute(
                "UPDATE files SET type = 'document'"
                " WHERE (type = 'unknown' OR type IS NULL OR type = '')"
                " AND (LOWER(name) LIKE '%.txt' OR LOWER(name) LIKE '%.md')"
            )
            if "is_public" not in col_columns:
                if not is_new_database:
                    print("INFO: Updating Database Schema... Adding 'is_public' to collections")
                conn.execute("ALTER TABLE collections ADD COLUMN is_public INTEGER DEFAULT 0")
            if "shared_users" not in col_columns:
                if not is_new_database:
                    print("INFO: Updating Database Schema... Adding 'shared_users' to collections")
                conn.execute("ALTER TABLE collections ADD COLUMN shared_users TEXT DEFAULT ''")
            if "parent_id" not in col_columns:
                if not is_new_database:
                    print("INFO: Updating Database Schema... Adding 'parent_id' to collections")
                conn.execute("ALTER TABLE collections ADD COLUMN parent_id INTEGER DEFAULT NULL")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_collections_parent ON collections(parent_id)")
        except Exception as e:
            _logger.debug("handled a failure in init_db", exc_info=True)
            print(f"WARNING: Could not migrate collections table: {e}")

        cursor = conn.execute("PRAGMA table_info(files)")
        existing_columns = {row["name"] for row in cursor.fetchall()}

        for col_name, col_type in required_columns.items():
            if col_name not in existing_columns:
                if not is_new_database:
                    print(f"INFO: Updating Database Schema... Adding missing column '{col_name}'")
                try:
                    conn.execute(f"ALTER TABLE files ADD COLUMN {col_name} {col_type}")
                except Exception as e:
                    _logger.debug("handled a failure in init_db", exc_info=True)
                    print(f"WARNING: Could not add column {col_name}: {e}")

        try:
            cleared = clear_synthetic_prompt_hashes(conn)
            if cleared:
                print(
                    f"INFO: Cleared {cleared} synthetic prompt hashes "
                    f"(promptless files no longer form fake prompt clusters)"
                )
        except Exception as e:
            _logger.debug("handled a failure in init_db", exc_info=True)
            print(f"WARNING: Could not clear synthetic prompt hashes: {e}")

        # 6. SCHEMA VERSION
        try:
            cur = conn.execute("PRAGMA user_version")
            current_ver = cur.fetchone()[0]

            if current_ver > DB_SCHEMA_VERSION:
                # Stamping this DOWN would erase the only record that newer
                # migrations have already run, so the newer build would run
                # them again over its own work. Two installs sharing one
                # gallery folder is how this happens -- a container and a
                # local copy at different versions, or a rollback after an
                # upgrade -- and `!=` treated it as an ordinary migration.
                print(
                    f"{Colors.RED}{Colors.BOLD}WARNING: this database was made by a "
                    f"NEWER SmartGallery (schema {current_ver}; this build knows "
                    f"{DB_SCHEMA_VERSION}).{Colors.RESET}"
                )
                print(
                    f"{Colors.YELLOW}Its version marker is being left alone. Newer "
                    f"data may not appear here, and changes made from this build may "
                    f"not be understood by the newer one. Update SmartGallery, or "
                    f"point BASE_SMARTGALLERY_PATH at a different folder.{Colors.RESET}"
                )
            elif current_ver < DB_SCHEMA_VERSION:
                if not is_new_database:
                    print(f"INFO: Updating Database Schema Version: {current_ver} -> {DB_SCHEMA_VERSION}")
                conn.execute(f"PRAGMA user_version = {DB_SCHEMA_VERSION}")
        except Exception as e:
            _logger.debug("handled a failure in init_db", exc_info=True)
            print(f"WARNING: Could not update DB schema version: {e}")

        # 7. AI DAM SCHEMA (WI-31) -- derived, rebuildable tables; always kept
        # in sync regardless of ENABLE_AI_DAM so enabling the feature later
        # never requires a separate migration step.
        try:
            smartgallery_ai.schema.init_schema(conn)
        except Exception as e:
            _logger.debug("handled a failure in init_db", exc_info=True)
            print(f"WARNING: Could not initialize AI DAM schema: {e}")

        conn.commit()

        # The schema is now sound, so an empty one here is a real answer
        # rather than a half-built database: either this folder is new, or
        # a library that was here has gone.
        try:
            check_library_continuity(is_new_database)
        except Exception as e:
            _logger.debug("handled a failure in init_db", exc_info=True)
            print(f"WARNING: Could not check library continuity: {e}")

    except Exception as e:
        _logger.debug("handled a failure in init_db", exc_info=True)
        print(f"CRITICAL DATABASE ERROR: {e}")

    finally:
        if close_conn:
            conn.close()


def get_dynamic_folder_config(force_refresh=False):
    global folder_config_cache
    if folder_config_cache is not None and not force_refresh:
        return folder_config_cache

    # print("INFO: Refreshing folder configuration by scanning directory tree...")

    base_path_normalized = os.path.normpath(BASE_OUTPUT_PATH).replace("\\", "/")

    try:
        root_mtime = os.path.getmtime(BASE_OUTPUT_PATH)
    except OSError:
        root_mtime = time.time()

    dynamic_config = {
        "_root_": {
            "display_name": "Main",
            "path": base_path_normalized,
            "relative_path": "",
            "parent": None,
            "children": [],
            "mtime": root_mtime,
            "is_watched": False,
            "is_explicitly_watched": False,
            "is_mount": False,  # Root is never a mount
            "file_count": 0,
            "descendant_file_count": 0,
        }
    }

    try:
        # 1. Fetch Watched Status
        watched_rules = []
        if ENABLE_AI_SEARCH:
            try:
                with get_db_connection() as conn:
                    rows = conn.execute("SELECT path, recursive FROM ai_watched_folders").fetchall()
                    for r in rows:
                        w_path = os.path.normpath(r["path"]).replace("\\", "/")
                        watched_rules.append((w_path, bool(r["recursive"])))
            except sqlite3.Error:
                pass

        # 2. Fetch Mounted Folders (New)
        mounted_paths = set()
        try:
            with get_db_connection() as conn:
                rows = conn.execute("SELECT path FROM mounted_folders").fetchall()
                for r in rows:
                    # Normalize for comparison
                    mounted_paths.add(os.path.normpath(r["path"]).replace("\\", "/"))
        except sqlite3.Error:
            pass

        all_folders = {}
        for dirpath, dirnames, _ in os.walk(BASE_OUTPUT_PATH, followlinks=True):
            dirnames[:] = [
                d
                for d in dirnames
                if (not d.startswith(".") or d == ".collection_notes")
                and d
                not in [
                    THUMBNAIL_CACHE_FOLDER_NAME,
                    SQLITE_CACHE_FOLDER_NAME,
                    ZIP_CACHE_FOLDER_NAME,
                    AI_MODELS_FOLDER_NAME,
                ]
            ]
            # The trash is excluded by PATH, not by name, because DELETE_TO can
            # be anywhere -- and pointing it inside the gallery is a reasonable
            # thing to do when you want deletions to stay on the same drive.
            # Indexing it put every deleted file straight back in the gallery
            # under its timestamped trash name, with the ratings and comments
            # gone, since the new path means a new id.
            if DELETE_TO:
                dirnames[:] = [
                    d
                    for d in dirnames
                    if not _is_under(os.path.join(dirpath, d) + "/", DELETE_TO)
                    and _std_path(os.path.join(dirpath, d)).rstrip("/").lower()
                    != _std_path(DELETE_TO).rstrip("/").lower()
                ]
            for dirname in dirnames:
                full_path = os.path.normpath(os.path.join(dirpath, dirname)).replace("\\", "/")
                relative_path = os.path.relpath(full_path, BASE_OUTPUT_PATH).replace("\\", "/")
                try:
                    mtime = os.path.getmtime(full_path)
                except OSError:
                    mtime = time.time()

                all_folders[relative_path] = {"full_path": full_path, "display_name": dirname, "mtime": mtime}

        sorted_paths = sorted(all_folders.keys(), key=lambda x: x.count("/"))

        for rel_path in sorted_paths:
            folder_data = all_folders[rel_path]
            key = path_to_key(rel_path)
            parent_rel_path = os.path.dirname(rel_path).replace("\\", "/")
            parent_key = "_root_" if parent_rel_path in {".", ""} else path_to_key(parent_rel_path)

            if parent_key in dynamic_config:
                dynamic_config[parent_key]["children"].append(key)

            current_path = folder_data["full_path"]

            # Watch Logic
            is_watched_folder = False
            is_explicitly_watched = False
            for w_path, is_recursive in watched_rules:
                if current_path == w_path:
                    is_watched_folder = True
                    is_explicitly_watched = True
                    break
                if is_recursive and current_path.startswith(w_path + "/"):
                    is_watched_folder = True
                    break

            # Mount Logic
            is_mount = current_path in mounted_paths

            # NEW: Resolve the physical path (handles Symlinks/Junctions for subfolders too)
            real_path = os.path.realpath(current_path).replace("\\", "/")

            dynamic_config[key] = {
                "display_name": folder_data["display_name"],
                "path": current_path,
                "real_path": real_path,  # <--- NEW FIELD
                "relative_path": rel_path,
                "parent": parent_key,
                "children": [],
                "mtime": folder_data["mtime"],
                "is_watched": is_watched_folder,
                "is_explicitly_watched": is_explicitly_watched,
                "is_mount": is_mount,
                "is_hidden": folder_data["display_name"] == ".collection_notes",
                "file_count": 0,
                "descendant_file_count": 0,
            }
    except FileNotFoundError:
        print(f"WARNING: The base directory '{BASE_OUTPUT_PATH}' was not found.")

    # Calculate folder file counts from database
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT path FROM files WHERE type != 'document'"
                " AND LOWER(name) NOT LIKE '%.txt' AND LOWER(name) NOT LIKE '%.md'"
            ).fetchall()
            dir_counts = {}
            for r in rows:
                f_dir = os.path.normpath(os.path.dirname(r["path"])).replace("\\", "/").lower().rstrip("/")
                dir_counts[f_dir] = dir_counts.get(f_dir, 0) + 1

            for key, info in dynamic_config.items():
                info_path_norm = os.path.normpath(info["path"]).replace("\\", "/").lower().rstrip("/")
                info["file_count"] = dir_counts.get(info_path_norm, 0)

            def _calc_descendant_count(key):
                info = dynamic_config[key]
                sub_count = 0
                for child_key in info.get("children", []):
                    if child_key in dynamic_config:
                        child_info = dynamic_config[child_key]
                        sub_count += child_info.get("file_count", 0) + _calc_descendant_count(child_key)
                info["descendant_file_count"] = sub_count
                return sub_count

            _calc_descendant_count("_root_")
    except Exception as e:
        _logger.debug("handled a failure in get_dynamic_folder_config", exc_info=True)
        print(f"Error calculating folder file counts: {e}")

    folder_config_cache = dynamic_config
    return dynamic_config


# --- BACKGROUND WATCHER THREAD ---
QUEUE_STALE_SECONDS = 3 * 86400


def sweep_stale_index_queue(conn, now=None):
    """Drop indexing-queue rows older than three days. Returns the count.

    'pending' is swept as well as 'completed' because nothing in the
    gallery drains this queue any more -- the AI layer indexes straight
    from the library (see the ENABLE_AI_SEARCH section of
    docs/CONFIGURATION.md). Sweeping only 'completed', a status nothing
    ever sets, left a row per file for ever.

    Dropping a stale row costs nothing: the scan re-queues on conflict, so
    anything still wanted comes back on the next pass.
    """
    cutoff = (now if now is not None else time.time()) - QUEUE_STALE_SECONDS
    cur = conn.execute(
        "DELETE FROM ai_indexing_queue WHERE status IN ('completed', 'pending') AND created_at < ?", (cutoff,)
    )
    conn.commit()
    return cur.rowcount


def background_watcher_task():
    """
    Periodically scans watched folders.
    Ensures TRUE incremental indexing:
    1. Ignores files currently 'pending' or 'processing'.
    2. Checks 'files' DB: if ai_data is missing or outdated -> queues it.
    3. Revives 'completed'/'error' queue entries back to 'pending' if the file is dirty.
    """
    print("INFO: AI Background Watcher started (Incremental Mode).")
    while True:
        try:
            if ENABLE_AI_SEARCH:
                with get_db_connection() as conn:
                    sweep_stale_index_queue(conn)

                    watched = conn.execute("SELECT path, recursive FROM ai_watched_folders").fetchall()

                    for row in watched:
                        folder_path = row["path"]
                        is_recursive = row["recursive"]

                        valid_exts = {
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".webp",
                            ".gif",
                            ".mp4",
                            ".mov",
                            ".avi",
                            ".webm",
                            ".txt",
                            ".md",
                        }
                        EXCLUDED = {
                            ".thumbnails_cache",
                            ".sqlite_cache",
                            ".zip_downloads",
                            ".AImodels",
                            "venv",
                            "venv-ai",
                            ".git",
                        }

                        files_to_check = []

                        if os.path.isdir(folder_path):
                            if is_recursive:
                                for root, dirs, files in os.walk(folder_path, topdown=True, followlinks=True):
                                    dirs[:] = [
                                        d
                                        for d in dirs
                                        if (not d.startswith(".") or d == ".collection_notes") and d not in EXCLUDED
                                    ]
                                    for f in files:
                                        if os.path.splitext(f)[1].lower() in valid_exts:
                                            files_to_check.append(os.path.join(root, f))
                            else:
                                try:
                                    for f in os.listdir(folder_path):
                                        full = os.path.join(folder_path, f)
                                        if os.path.isfile(full) and os.path.splitext(f)[1].lower() in valid_exts:
                                            files_to_check.append(full)
                                except OSError:
                                    pass

                        # Process Candidates
                        for raw_path in files_to_check:
                            p_key = get_standardized_path(raw_path)

                            # 1. CHECK ACTIVE STATUS
                            # Only skip if it is actively waiting or running.
                            # Do NOT skip if it is 'completed' or 'error' (we might need to retry/update).
                            active_job = conn.execute(
                                """
                                SELECT 1 FROM ai_indexing_queue
                                WHERE file_path = ? AND status IN ('pending', 'processing', 'waiting_gpu')
                            """,
                                (p_key,),
                            ).fetchone()

                            if active_job:
                                continue  # Busy, come back later

                            # 2. CHECK FILE STATE IN DB
                            # We need to find the file ID and its scan timestamp
                            # We use the robust path lookup logic (normalized slash match)
                            # to ensure we find the record even if slashes differ.

                            # Try exact match first
                            file_row = conn.execute(
                                "SELECT id, mtime, ai_last_scanned FROM files WHERE path = ?", (raw_path,)
                            ).fetchone()

                            # Fallback: Normalized Match
                            if not file_row:
                                norm_p = raw_path.replace("\\", "/")
                                file_row = conn.execute(
                                    "SELECT id, mtime, ai_last_scanned FROM files WHERE REPLACE(path, '\\', '/') = ?",
                                    (norm_p,),
                                ).fetchone()

                            if not file_row:
                                # File exists on disk but NOT in DB.
                                # We cannot index it yet (missing metadata/dimensions).
                                # The main 'files' sync must run first. We skip it silently.
                                continue

                            file_id = file_row["id"]
                            last_scan_ts = file_row["ai_last_scanned"] if file_row["ai_last_scanned"] is not None else 0
                            mtime = file_row["mtime"]

                            # 3. DIRTY CHECK (The Core Incremental Logic)
                            needs_index = False

                            if last_scan_ts == 0:
                                needs_index = True  # Never scanned or Reset by user
                            elif last_scan_ts < mtime:
                                needs_index = True  # File modified on disk after last scan

                            if needs_index:
                                # UPSERT: If exists (e.g. 'completed'), revive to 'pending'. If new, insert.
                                # This fixes the issue where completed items were ignored even after reset.
                                conn.execute(
                                    """
                                    INSERT INTO ai_indexing_queue
                                    (file_path, file_id, status, created_at, force_index, params)
                                    VALUES (?, ?, 'pending', ?, 0, '{}')
                                    ON CONFLICT(file_path) DO UPDATE SET
                                        status = 'pending',
                                        file_id = excluded.file_id,
                                        created_at = excluded.created_at
                                """,
                                    (p_key, file_id, time.time()),
                                )

                    conn.commit()

        except Exception as e:
            _logger.debug("handled a failure in background_watcher_task", exc_info=True)
            print(f"Watcher Loop Error: {e}")

        time.sleep(10)  # Faster check cycle (10s instead of 60s) to feel responsive


def looks_like_a_renamed_root(db_paths, to_delete, to_add):
    """True when the library is the same and only its address changed.

    Every row would be deleted, and the very same filenames have just
    turned up. Matching the SET OF NAMES rather than the count is what
    keeps this away from a genuine wholesale replacement -- and if
    somebody really did replace every file with one of the same name,
    keeping the ratings is right anyway.

    An empty database is not a renamed root, and neither is a library
    where anything at all still lines up: one row in common means the
    address is the one on record and the missing files are missing.
    """
    if not db_paths or to_delete != db_paths:
        return False
    return {os.path.basename(p) for p in to_delete} == {os.path.basename(p) for p in to_add}


def full_sync_database(conn):
    print("INFO: Starting full file scan...")
    start_time = time.time()

    all_folders = get_dynamic_folder_config(force_refresh=True)
    db_files = {row["path"]: row["mtime"] for row in conn.execute("SELECT path, mtime FROM files").fetchall()}

    disk_files = {}
    print("INFO: Scanning directories on disk...")

    # Whitelist approach: Only index valid media files
    valid_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".gif",  # Images
        ".mp4",
        ".mov",
        ".webm",
        ".mkv",
        ".avi",
        ".m4v",
        ".wmv",
        ".flv",
        ".mts",
        ".ts",  # Videos
        ".mp3",
        ".wav",
        ".ogg",
        ".flac",
        ".m4a",
        ".aac",
        ".txt",
        ".md",  # Audio & Docs
    }

    # Folders we could not read. "We looked and it is empty" and "we could
    # not look" are the same thing to the diff below, and only the first one
    # means the files were deleted -- so the second is recorded and the rows
    # underneath are left alone.
    unreadable_roots = []

    for folder_data in all_folders.values():
        folder_path = folder_data["path"]
        if not os.path.isdir(folder_path):
            continue
        try:
            for name in os.listdir(folder_path):
                filepath = os.path.join(folder_path, name)

                # Check extension against whitelist
                _, ext = os.path.splitext(name)
                if os.path.isfile(filepath) and ext.lower() in valid_extensions:
                    disk_files[filepath] = os.path.getmtime(filepath)

        except OSError as e:
            print(f"WARNING: Could not access folder {folder_path}: {e}")
            unreadable_roots.append(folder_path)

    # The gallery root is the one path every install has, and the one most
    # likely to be the external drive or network share. If it is not there,
    # nothing under it can be judged missing.
    if not os.path.isdir(BASE_OUTPUT_PATH):
        unreadable_roots.append(BASE_OUTPUT_PATH)

    db_paths = set(db_files.keys())
    disk_paths = set(disk_files.keys())

    to_delete = db_paths - disk_paths
    to_add = disk_paths - db_paths

    # A file is identified by its path, so a library reached by a
    # different spelling of the same folder is a library of strangers.
    # Every row is missing and every picture is new, and the scan does
    # exactly what it is told: deletes all the rows, which cascades the
    # ratings, comments, album membership and tags away, then indexes the
    # same pictures again as if they had just arrived.
    #
    # Measured on three rated pictures, restarting with only the case of
    # BASE_OUTPUT_PATH changed and then changed back:
    #
    #     forward  seed   3 files, 3 favourites, 3 ratings
    #     forward  check  3 files, 3 favourites, 3 ratings
    #     upper    check  3 files, 0 favourites, 0 ratings
    #     forward  check  3 files, 0 favourites, 0 ratings
    #
    # One start under the other spelling, and putting the setting back
    # does not bring any of it back. Nothing looks wrong afterwards: the
    # pictures are all there, which is what makes it convincing.
    #
    # The existing guard above covers a root that is not THERE. This one
    # is reachable and full; only the name it is reached by has changed --
    # a Docker mount moved, a library copied to another drive, a path
    # retyped in another case on Windows, where both spellings open the
    # same folder.
    #
    # The signature is unmistakable: every row would go, and the very same
    # filenames just arrived. Matching on the set of names rather than the
    # count is what keeps this from firing on a genuine wholesale
    # replacement -- and if somebody really did replace every file with
    # one of the same name, keeping the ratings is right anyway.
    if looks_like_a_renamed_root(db_paths, to_delete, to_add):
        was = os.path.dirname(min(to_delete))
        now = os.path.dirname(min(to_add))
        print(
            f"\n{Colors.RED}{Colors.BOLD}WARNING: every file in the library "
            f"is at a different address than the one recorded.{Colors.RESET}"
        )
        print(f"{Colors.RED}recorded: {was}{Colors.RESET}")
        print(f"{Colors.RED}found   : {now}{Colors.RESET}")
        print(
            f"{Colors.YELLOW}The same {len(to_add)} file(s) are all there, so "
            f"this is the folder being reached by a different name rather "
            f"than anything being deleted -- a changed mount, a moved "
            f"library, or the same path typed in another case.{Colors.RESET}"
        )
        print(
            f"{Colors.YELLOW}Nothing has been removed. Ratings, comments, "
            f"albums and tags are keyed to the recorded address, so they "
            f"would all have gone. Point BASE_OUTPUT_PATH back at the "
            f"recorded spelling to see them again.{Colors.RESET}"
        )
        print(
            f"{Colors.YELLOW}Until then the gallery lists both sets, so "
            f"every picture appears twice. That is the cost of keeping "
            f"them; it goes away as soon as the address matches."
            f"{Colors.RESET}\n"
        )
        to_delete = set()
    to_check = disk_paths & db_paths
    to_update = {path for path in to_check if int(disk_files.get(path, 0)) > int(db_files.get(path, 0))}

    files_to_process = list(to_add.union(to_update))
    # debug if files_to_process: print(f"{Colors.YELLOW}DEBUG - File to process: {files_to_process}{Colors.RESET}")
    if files_to_process:
        print(
            f"INFO: Processing {len(files_to_process)} files in parallel"
            f" using up to {MAX_PARALLEL_WORKERS or 'all'} CPU cores..."
        )

        results = []
        attempted = set()
        pool_failure = None
        # A single corrupted file can take a worker down with a C-level
        # segfault (OpenCV/Pillow), which surfaces as BrokenProcessPool on
        # EVERY pending future -- so one bad file must not cost the scan.
        # The pool can also be unusable outright (frozen builds, sandboxes
        # and containers that forbid process spawning, memory pressure);
        # that used to be reported as "likely a corrupted file" while the
        # scan indexed nothing and still announced completion, leaving an
        # empty gallery with no actionable message. Either way, whatever
        # the pool did not finish is processed in this process below.
        try:
            with scan_executor(len(files_to_process)) as executor:
                futures = {executor.submit(process_single_file, path): path for path in files_to_process}

                with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        file_path = futures[future]
                        try:
                            result = future.result()
                            if result:
                                results.append(result)
                            attempted.add(file_path)
                        except concurrent.futures.process.BrokenProcessPool as e:
                            pool_failure = e
                            break  # the pool is gone; the rest runs sequentially
                        except Exception as e:
                            _logger.debug("handled a failure in full_sync_database", exc_info=True)
                            attempted.add(file_path)  # tried and failed on its own merits
                            print(f"\nWARNING: Unhandled error processing {os.path.basename(file_path)}: {e}")
                        pbar.update(1)
        except Exception as e:  # pool creation, or shutdown after a break
            _logger.debug("handled a failure in full_sync_database", exc_info=True)
            pool_failure = pool_failure or e

        leftover = [p for p in files_to_process if p not in attempted]
        if leftover:
            if pool_failure is not None:
                print(
                    f"\nWARNING: parallel processing stopped ({type(pool_failure).__name__}: "
                    f"{pool_failure}). Processing the remaining {len(leftover)} file(s) "
                    f"one at a time -- slower, but nothing is skipped."
                )
            with tqdm(total=len(leftover), desc="Processing files (sequential)") as pbar:
                for path in leftover:
                    try:
                        result = process_single_file(path)
                        if result:
                            results.append(result)
                    except Exception as e:
                        _logger.debug("handled a failure in full_sync_database", exc_info=True)
                        print(f"\nWARNING: Unhandled error processing {os.path.basename(path)}: {e}")
                    pbar.update(1)

        if results:
            print(f"INFO: Inserting {len(results)} processed records into the database...")
            for i in range(0, len(results), BATCH_SIZE):
                batch, gen_rows, gen_deletes = split_file_results(results[i : i + BATCH_SIZE])
                conn.executemany(
                    """
                    INSERT INTO files (
                        id, path, mtime, name, type, duration, dimensions,
                        has_workflow, size, last_scanned, workflow_files,
                        workflow_prompt, workflow_hash, prompt_hash, models_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        path = excluded.path,
                        name = excluded.name,
                        type = excluded.type,
                        duration = excluded.duration,
                        dimensions = excluded.dimensions,
                        has_workflow = excluded.has_workflow,
                        size = excluded.size,
                        last_scanned = excluded.last_scanned,
                        workflow_files = excluded.workflow_files,
                        workflow_prompt = excluded.workflow_prompt,
                        workflow_hash = excluded.workflow_hash,
                        prompt_hash = excluded.prompt_hash,
                        models_hash = excluded.models_hash,
                        hash_failed = CASE WHEN ABS(files.mtime - excluded.mtime) > 0.1
                                THEN 0 ELSE files.hash_failed END,

                        -- CONDITIONAL LOGIC: everything below is DERIVED from
                        -- the file's content, so a changed mtime invalidates
                        -- it. is_favorite is not derived from anything -- it
                        -- is what the person chose -- and used to be cleared
                        -- here with them. Ratings, comments, albums and tags
                        -- all survive a rescan; the favourite was the only
                        -- one that did not, because it sits in this block
                        -- rather than in a table of its own. mtime moves for
                        -- reasons that have nothing to do with content: a
                        -- restored backup, a copy to another drive, a sync
                        -- client rewriting files. Any of those wiped every
                        -- favourite in the library at once, silently.
                        ai_caption = CASE
                            WHEN ABS(files.mtime - excluded.mtime) > 0.1 THEN NULL
                            ELSE files.ai_caption
                        END,

                        ai_embedding = CASE
                            WHEN ABS(files.mtime - excluded.mtime) > 0.1 THEN NULL
                            ELSE files.ai_embedding
                        END,

                        ai_last_scanned = CASE
                            WHEN ABS(files.mtime - excluded.mtime) > 0.1 THEN 0
                            ELSE files.ai_last_scanned
                        END,

                        -- Update mtime at the end
                        mtime = excluded.mtime
                """,
                    batch,
                )
                upsert_generation_params(conn, gen_rows, gen_deletes)
                conn.commit()

    # SAFETY GUARD FOR DISCONNECTED DRIVES
    if to_delete:
        print("INFO: Detecting disconnected drives before cleanup...")

        # 1. Identify Offline Mounts
        # We fetch all configured mount points to check if their root is accessible
        mount_rows = conn.execute("SELECT path FROM mounted_folders").fetchall()
        offline_prefixes = []

        for row in mount_rows:
            m_path = row["path"]
            # If the mount root itself is missing, assume the drive is offline.
            # note: os.path.exists returns False for broken symlinks/junctions
            if not os.path.exists(m_path):
                print(f"{Colors.YELLOW}WARN: Mount point seems offline: {m_path}{Colors.RESET}")
                offline_prefixes.append(m_path)

        # A folder we failed to read counts the same as an offline mount:
        # its files are unaccounted for, not gone.
        for unreadable in unreadable_roots:
            print(f"{Colors.YELLOW}WARN: Folder could not be read, keeping its entries: {unreadable}{Colors.RESET}")
            offline_prefixes.append(unreadable)

        # 2. Filter files to delete
        # Only delete files if they do NOT belong to an offline mount
        safe_to_delete = []
        protected_count = 0

        for path_to_remove in to_delete:
            is_protected = False
            for offline_root in offline_prefixes:
                # Containment, not a bare prefix: an offline "box" must not
                # also protect the files of a "box_archive" that is present.
                if _is_under(path_to_remove, offline_root):
                    is_protected = True
                    break

            if is_protected:
                protected_count += 1
            else:
                safe_to_delete.append(path_to_remove)

        if protected_count > 0:
            print(
                f"{Colors.YELLOW}PROTECTION ACTIVE: Skipped deletion of {protected_count}"
                f" files because their source drive appears offline.{Colors.RESET}"
            )

        # 3. Proceed with safe deletion (protecting collection notes from being purged)
        if safe_to_delete:
            notes_dir_smart = os.path.normpath(os.path.join(BASE_SMARTGALLERY_PATH, ".collection_notes")).lower()
            notes_dir_out = os.path.normpath(os.path.join(BASE_OUTPUT_PATH, ".collection_notes")).lower()

            real_to_delete = []
            for p in safe_to_delete:
                p_norm = os.path.normpath(p).lower()
                if p_norm.startswith((notes_dir_smart, notes_dir_out)):
                    if not os.path.exists(p):
                        real_to_delete.append(p)
                else:
                    real_to_delete.append(p)
            safe_to_delete = real_to_delete

        if safe_to_delete:
            print(f"INFO: Removing {len(safe_to_delete)} obsolete file entries from the database...")

            paths_to_remove = [(p,) for p in safe_to_delete]
            conn.executemany("DELETE FROM files WHERE path = ?", paths_to_remove)

            # Clean AI Queue for validly deleted files
            std_paths_to_remove = [(get_standardized_path(p),) for p in safe_to_delete]
            conn.executemany("DELETE FROM ai_indexing_queue WHERE file_path = ?", std_paths_to_remove)

            conn.commit()

    print(f"INFO: Full scan completed in {time.time() - start_time:.2f} seconds.")


def sse(**payload):
    """One server-sent event.

    The `data: ` prefix and the blank-line terminator were written out at
    every yield site; getting either wrong silently produces an event the
    browser never delivers.
    """
    return f"data: {json.dumps(payload)}\n\n"


def sync_folder_on_demand(folder_path):
    yield sse(message="Checking folder for changes...", current=0, total=1)

    try:
        with get_db_connection() as conn:
            disk_files, valid_extensions = (
                {},
                {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".webp",
                    ".mp4",
                    ".mkv",
                    ".webm",
                    ".mov",
                    ".avi",
                    ".mp3",
                    ".wav",
                    ".ogg",
                    ".flac",
                    ".txt",
                    ".md",
                },
            )
            if os.path.isdir(folder_path):
                for name in os.listdir(folder_path):
                    filepath = os.path.join(folder_path, name)
                    if os.path.isfile(filepath) and os.path.splitext(name)[1].lower() in valid_extensions:
                        disk_files[filepath] = os.path.getmtime(filepath)

            db_files_query = conn.execute(
                "SELECT path, mtime FROM files WHERE path LIKE ?", (folder_path + os.sep + "%",)
            ).fetchall()
            db_files = {
                row["path"]: row["mtime"]
                for row in db_files_query
                if os.path.normpath(os.path.dirname(row["path"])) == os.path.normpath(folder_path)
            }

            disk_filepaths, db_filepaths = set(disk_files.keys()), set(db_files.keys())
            files_to_add = disk_filepaths - db_filepaths
            files_to_delete = db_filepaths - disk_filepaths
            files_to_update = {
                path for path in (disk_filepaths & db_filepaths) if int(disk_files[path]) > int(db_files[path])
            }

            if not files_to_add and not files_to_update and not files_to_delete:
                yield sse(message="Folder is up-to-date.", status="no_changes", current=1, total=1)
                return

            files_to_process = list(files_to_add.union(files_to_update))
            total_files = len(files_to_process)

            if total_files > 0:
                yield sse(
                    message=f"Found {total_files} new/modified files. Processing...",
                    current=0,
                    total=total_files,
                )

                data_to_upsert = []
                processed_count = 0

                with scan_executor(total_files) as executor:
                    futures = {executor.submit(process_single_file, path): path for path in files_to_process}

                    for future in concurrent.futures.as_completed(futures):
                        # --- FAULT TOLERANCE FIX FOR SYNC ---
                        try:
                            result = future.result()
                            if result:
                                data_to_upsert.append(result)
                        except concurrent.futures.process.BrokenProcessPool as e:
                            print(
                                f"\nWARNING: A worker process crashed (likely due to a"
                                f" corrupted file). Recovering... Error: {e}"
                            )
                        except Exception as e:
                            _logger.debug("handled a failure in sync_folder_on_demand", exc_info=True)
                            file_path_failed = futures[future]
                            print(f"\nWARNING: Unhandled error processing {os.path.basename(file_path_failed)}: {e}")

                        processed_count += 1
                        path = futures[future]
                        progress_data = {
                            "message": f"Processing: {os.path.basename(path)}",
                            "current": processed_count,
                            "total": total_files,
                        }
                        yield sse(**progress_data)

                if data_to_upsert:
                    file_rows_2, gen_rows_2, gen_deletes_2 = split_file_results(data_to_upsert)
                    conn.executemany(
                        """
                        INSERT INTO files (
                        id, path, mtime, name, type, duration, dimensions,
                        has_workflow, size, last_scanned, workflow_files,
                        workflow_prompt, workflow_hash, prompt_hash, models_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            path = excluded.path,
                            name = excluded.name,
                            type = excluded.type,
                            duration = excluded.duration,
                            dimensions = excluded.dimensions,
                            has_workflow = excluded.has_workflow,
                            size = excluded.size,
                            last_scanned = excluded.last_scanned,
                            workflow_files = excluded.workflow_files,
                            workflow_prompt = excluded.workflow_prompt,
                            workflow_hash = excluded.workflow_hash,
                            prompt_hash = excluded.prompt_hash,
                        models_hash = excluded.models_hash,
                            hash_failed = CASE WHEN ABS(files.mtime - excluded.mtime) > 0.1
                                THEN 0 ELSE files.hash_failed END,

                            -- CONDITIONAL LOGIC: derived-from-content fields
                            -- only. is_favorite is the person's own choice
                            -- and is left alone (see the note on the same
                            -- statement in the full scan).
                            ai_caption = CASE
                                WHEN ABS(files.mtime - excluded.mtime) > 0.1 THEN NULL
                                ELSE files.ai_caption
                            END,

                            ai_embedding = CASE
                                WHEN ABS(files.mtime - excluded.mtime) > 0.1 THEN NULL
                                ELSE files.ai_embedding
                            END,

                            ai_last_scanned = CASE
                                WHEN ABS(files.mtime - excluded.mtime) > 0.1 THEN 0
                                ELSE files.ai_last_scanned
                            END,

                            -- Update mtime at the end
                            mtime = excluded.mtime
                    """,
                        file_rows_2,
                    )
                    upsert_generation_params(conn, gen_rows_2, gen_deletes_2)

            if files_to_delete:
                conn.executemany("DELETE FROM files WHERE path IN (?)", [(p,) for p in files_to_delete])

            conn.commit()
            yield sse(
                message="Sync complete. Reloading...",
                status="reloading",
                current=total_files,
                total=total_files,
            )

    except Exception as e:
        _logger.debug("handled a failure in sync_folder_on_demand", exc_info=True)
        error_message = f"Error during sync: {e}"
        print(f"ERROR: {error_message}")
        yield sse(message=error_message, current=1, total=1, error=True)


def scan_folder_and_extract_options(folder_path, recursive=True):
    """
    Scans the physical folder to count files and extract metadata.
    Supports recursive mode to include subfolders in the count.
    """
    extensions, prefixes = set(), set()
    file_count = 0
    try:
        if not os.path.isdir(folder_path):
            return 0, [], []

        if recursive:
            # Recursive scan using os.walk
            for _root, dirs, files in os.walk(folder_path, followlinks=True):
                # Filter out hidden/protected folders in-place
                dirs[:] = [
                    d
                    for d in dirs
                    if not d.startswith(".")
                    and d
                    not in [
                        THUMBNAIL_CACHE_FOLDER_NAME,
                        SQLITE_CACHE_FOLDER_NAME,
                        ZIP_CACHE_FOLDER_NAME,
                        AI_MODELS_FOLDER_NAME,
                    ]
                ]
                for filename in files:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext and ext not in [".json", ".sqlite"]:
                        file_count += 1
                        extensions.add(ext.lstrip("."))
                        if "_" in filename:
                            prefixes.add(filename.split("_")[0])
        else:
            # Single folder scan using os.scandir (faster)
            for entry in os.scandir(folder_path):
                if entry.is_file():
                    filename = entry.name
                    ext = os.path.splitext(filename)[1].lower()
                    if ext and ext not in [".json", ".sqlite"]:
                        file_count += 1
                        extensions.add(ext.lstrip("."))
                        if "_" in filename:
                            prefixes.add(filename.split("_")[0])

    except Exception as e:
        _logger.debug("handled a failure in scan_folder_and_extract_options", exc_info=True)
        print(f"ERROR: Could not scan folder '{folder_path}': {e}")

    return file_count, sorted(extensions), sorted(prefixes)


def cleanup_invalid_watched_folders(conn):
    """
    Checks if watched folders still exist on disk.
    [SAFE MODE]: If a folder is missing, we assumes it might be a disconnected drive
    and we DO NOT remove it automatically to prevent config loss.
    """
    try:
        rows = conn.execute("SELECT path FROM ai_watched_folders").fetchall()

        for row in rows:
            path = row["path"]
            if not os.path.exists(path) or not os.path.isdir(path):
                # We just WARN the user, we do NOT delete the config.
                print(f"{Colors.YELLOW}WARN: Watched folder not found (Offline or Deleted): {path}")
                print(f"      Skipping AI checks for this folder. Config preserved.{Colors.RESET}")

    except Exception as e:
        _logger.debug("handled a failure in cleanup_invalid_watched_folders", exc_info=True)
        print(f"ERROR checking watched folders: {e}")


def initialize_gallery_fast_no_db_check():
    print("INFO: Initializing gallery...")
    global FFPROBE_EXECUTABLE_PATH
    FFPROBE_EXECUTABLE_PATH = find_ffprobe_path()
    os.makedirs(THUMBNAIL_CACHE_DIR, exist_ok=True)
    os.makedirs(SQLITE_CACHE_DIR, exist_ok=True)
    # See initialize_gallery: sweep tmp_* strandings from killed encoders.
    for stale in glob.glob(os.path.join(THUMBNAIL_CACHE_DIR, "tmp_*")):
        with contextlib.suppress(OSError):
            os.remove(stale)
    # Prepared downloads are full copies of what went into them, and their
    # only sweep ran at the end of building another one -- so one download
    # and no more left that copy in the gallery folder for good.
    _removed_zips, _forgotten_jobs = prune_zip_cache()
    if _removed_zips:
        print(f"INFO: Cleared {_removed_zips} prepared download(s) older than {ZIP_RETENTION_SECONDS // 3600} hours.")

    with get_db_connection() as conn:
        try:
            init_db(conn)
            # 4. Fallback check for empty DB on existing install
            file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            if file_count == 0:
                print(f"{Colors.BLUE}INFO: Database file exists but is empty. Scanning...{Colors.RESET}")
                full_sync_database(conn)

        except sqlite3.DatabaseError as e:
            print(f"ERROR initializing database: {e}")


def pregenerate_exhibition_cache():
    """
    Pre-generates metadata-stripped files for all items in public collections.
    Runs only in Exhibition mode to ensure fast, secure delivery to guests.
    Utilizes parallel thread processing for speed, skipping already processed files.
    Safe for Windows, macOS, Linux, and Docker environments. Handles mixed slashes.
    """
    if not IS_EXHIBITION_MODE:
        return

    print(f"{Colors.BLUE}INFO: Checking Exhibition Cache (Metadata-stripped files)...{Colors.RESET}")

    files_to_process = []
    with get_db_connection() as conn:
        # Fetch all distinct files that belong to public user albums
        query = """
            SELECT DISTINCT f.id, f.path, f.mtime, f.type, f.name
            FROM files f
            JOIN collection_files cf ON f.id = cf.file_id
            JOIN collections c ON cf.collection_id = c.id
            WHERE c.type = 'user_album' AND (c.is_public = 1 OR c.shared_users != '')
        """
        rows = conn.execute(query).fetchall()

        for row in rows:
            filepath = row["path"]
            mtime = row["mtime"]
            file_type = row["type"]

            # CRITICAL: Calculate hash using the EXACT path string from the DB
            # to match the retrieval logic in serve_cleaned_file().
            cache_hash = content_digest(filepath + str(mtime))
            _, ext = os.path.splitext(filepath)
            clean_path = os.path.join(CLEAN_CACHE_DIR, f"{cache_hash}{ext}")

            # NORMALIZE PATHS FOR OS (fixes Windows mixed slashes like c:/folder\subfolder/file.jpg)
            # This ensures FFmpeg and Pillow receive perfectly valid native paths.
            safe_input_path = os.path.normpath(filepath)
            safe_output_path = os.path.normpath(clean_path)

            # Only process if missing or corrupted (0 bytes)
            if not os.path.exists(safe_output_path) or os.path.getsize(safe_output_path) == 0:
                files_to_process.append(
                    {
                        "input_path": safe_input_path,
                        "output_path": safe_output_path,
                        "type": file_type,
                        "name": row["name"],
                    }
                )

    if not files_to_process:
        print(f"{Colors.GREEN}INFO: Exhibition cache is up to date.{Colors.RESET}")
        return

    print(
        f"INFO: Pre-generating {len(files_to_process)} clean files"
        f" using up to {MAX_PARALLEL_WORKERS or 'all'} CPU cores..."
    )

    success_count = 0

    # We use ThreadPoolExecutor to prevent OS-specific multiprocessing issues (like Windows pickling)
    # while allowing I/O and external FFmpeg calls to run concurrently safely across all platforms.
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        # Submit jobs using the safely normalized OS paths
        futures = {
            executor.submit(strip_media_metadata, f["input_path"], f["output_path"], f["type"]): f
            for f in files_to_process
        }

        with tqdm(total=len(files_to_process), desc="Cleaning files") as pbar:
            for future in concurrent.futures.as_completed(futures):
                file_info = futures[future]
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    _logger.debug("handled a failure in pregenerate_exhibition_cache", exc_info=True)
                    print(f"\nWARNING: Failed to clean {file_info['name']}: {e}")
                pbar.update(1)

    print(
        f"{Colors.GREEN}INFO: Successfully pre-generated"
        f" {success_count}/{len(files_to_process)} clean files.{Colors.RESET}"
    )


def check_exhibition_requirements():
    """
    Strict Pre-Flight Check for Exhibition Mode.
    Ensures that the Main gallery has been run before, the database exists,
    and at least one public or user-shared collection is configured.
    Exits the application if requirements are not met to prevent ghost databases.
    """
    if not IS_EXHIBITION_MODE:
        return

    print(f"{Colors.BLUE}INFO: Performing Pre-Flight Checks for Exhibition Mode...{Colors.RESET}")

    db_exists = os.path.exists(DATABASE_FILE)

    if not db_exists:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ CRITICAL ERROR: Database Not Found{Colors.RESET}")
        print(f"{Colors.RED}Exhibition Mode cannot run because the main database does not exist at:{Colors.RESET}")
        print(f"{Colors.YELLOW}{DATABASE_FILE}{Colors.RESET}\n")
        print(f"{Colors.CYAN}{Colors.BOLD}💡 HOW TO FIX IT:{Colors.RESET}")
        print("1. Ensure 'BASE_SMARTGALLERY_PATH' is configured correctly.")
        print("2. You must run the standard gallery AT LEAST ONCE before using Exhibition Mode.")
        print(f"   Launch without flags: {Colors.YELLOW}python smartgallery.py{Colors.RESET}")
        print("   Create your collections there, then restart with --exhibition.\n")
        sys.exit(1)

    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            conn.row_factory = sqlite3.Row

            # Check if collections table exists
            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='collections'"
            ).fetchone()
            if not table_check:
                print(f"\n{Colors.RED}{Colors.BOLD}❌ CRITICAL ERROR: Collections Table Missing{Colors.RESET}")
                print(f"{Colors.RED}The database exists, but it's empty or outdated.{Colors.RESET}")
                print(f"\n{Colors.CYAN}{Colors.BOLD}💡 HOW TO FIX IT:{Colors.RESET}")
                print("Run the standard gallery first to initialize the database tables:")
                print(f"   {Colors.YELLOW}python smartgallery.py{Colors.RESET}\n")
                sys.exit(1)

            # Check if at least one user album is public or shared with a user.
            accessible_colls = conn.execute("""
                SELECT COUNT(*)
                FROM collections
                WHERE type = 'user_album'
                  AND (is_public = 1 OR TRIM(COALESCE(shared_users, '')) != '')
            """).fetchone()[0]

            if accessible_colls == 0:
                print(f"\n{Colors.RED}{Colors.BOLD}❌ CRITICAL ERROR: No Exhibition Collections Found{Colors.RESET}")
                print(
                    f"{Colors.RED}Exhibition Mode displays collections that are"
                    f" public or shared with at least one user.{Colors.RESET}"
                )
                print(
                    f"{Colors.RED}Currently, your database has no accessible Exhibition"
                    f" collections, so the Exhibition would be completely empty.{Colors.RESET}"
                )
                print(f"\n{Colors.CYAN}{Colors.BOLD}💡 HOW TO FIX IT:{Colors.RESET}")
                print(f"1. Start the standard gallery: {Colors.YELLOW}python smartgallery.py{Colors.RESET}")
                print("2. Log in, select some files, and click the 📚️ Add/Remove from collection button.")
                print("3. Mark a collection as Exhibition Ready or share it with at least one user.")
                print("4. Restart with --exhibition.\n")
                sys.exit(1)

    except sqlite3.DatabaseError as e:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ CRITICAL ERROR: Database corrupted or inaccessible: {e}{Colors.RESET}")
        sys.exit(1)


def initialize_gallery():
    print("INFO: Initializing gallery...")

    # --- STRICT CHECK FOR EXHIBITION MODE ---
    # Will exit(1) immediately if db/collections are missing, preventing ghost DB creation
    check_exhibition_requirements()

    global FFPROBE_EXECUTABLE_PATH
    FFPROBE_EXECUTABLE_PATH = find_ffprobe_path()
    os.makedirs(THUMBNAIL_CACHE_DIR, exist_ok=True)
    os.makedirs(SQLITE_CACHE_DIR, exist_ok=True)
    os.makedirs(CLEAN_CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_SMARTGALLERY_PATH, ".collection_notes"), exist_ok=True)
    os.makedirs(IMPORTED_WORKFLOWS_DIR, exist_ok=True)
    # Thumbnail/waveform encoders write tmp_* then os.replace() into place;
    # a hard process kill can strand the tmp_ file. Sweep leftovers here --
    # nothing references them, and their hashes retry naturally.
    for stale in glob.glob(os.path.join(THUMBNAIL_CACHE_DIR, "tmp_*")):
        with contextlib.suppress(OSError):
            os.remove(stale)
    # Prepared downloads are full copies of what went into them, and their
    # only sweep ran at the end of building another one -- so one download
    # and no more left that copy in the gallery folder for good.
    _removed_zips, _forgotten_jobs = prune_zip_cache()
    if _removed_zips:
        print(f"INFO: Cleared {_removed_zips} prepared download(s) older than {ZIP_RETENTION_SECONDS // 3600} hours.")

    with get_db_connection() as conn:
        try:
            init_db(conn)
            # Auto-migrate collection notes paths if directory moved to BASE_SMARTGALLERY_PATH
            try:
                old_notes_prefix = os.path.join(BASE_OUTPUT_PATH, ".collection_notes")
                new_notes_prefix = os.path.join(BASE_SMARTGALLERY_PATH, ".collection_notes")
                if old_notes_prefix != new_notes_prefix:
                    rows = conn.execute(
                        "SELECT id, path FROM files WHERE path LIKE ?", (old_notes_prefix + "%",)
                    ).fetchall()
                    for r in rows:
                        old_p = r["path"]
                        new_p = old_p.replace(old_notes_prefix, new_notes_prefix, 1)
                        new_id = content_digest(new_p)
                        # Everything referencing the old id moves with it. Updating
                        # files.id first used to raise a foreign key error that this
                        # block's own except swallowed into a one-line notice, so the
                        # migration never ran for anyone with a separate gallery path.
                        _reassign_file_ids(conn, [(r["id"], new_id)])
                        conn.execute("UPDATE files SET id = ?, path = ? WHERE id = ?", (new_id, new_p, r["id"]))
                    conn.commit()
            except Exception as e:
                _logger.debug("handled a failure in initialize_gallery", exc_info=True)
                print(f"Notes path migration notice: {e}")
            # Cleanup invalid watched folders before full sync
            if ENABLE_AI_SEARCH:
                cleanup_invalid_watched_folders(conn)
            # Force full sync on every startup to clean external deletions
            print(f"{Colors.BLUE}INFO: Performing startup consistency check...{Colors.RESET}")
            full_sync_database(conn)
            check_and_update_workflow_hashes(conn)
            backfill_audio_durations(conn)
            ensure_genparams_backfill_async(conn)
            migration_report = sg_auth.migrate_legacy_passwords(conn, ENCRYPTION_KEY_FILE)
            if migration_report["migrated"] or migration_report["failed"]:
                print(
                    f"{Colors.BLUE}INFO: Password migration - "
                    f"{migration_report['migrated']} migrated, "
                    f"{migration_report['failed']} failed (unusable, needs admin reset), "
                    f"key file deleted: {migration_report['key_deleted']}{Colors.RESET}"
                )
            ensure_admin_user(conn)

            # After full_sync_database, the library rows are what is on
            # disk -- which is the only moment the leftovers below can be
            # told apart from the thumbnails still in use.
            gone, gone_dirs = prune_thumbnail_cache(conn)
            if gone or gone_dirs:
                print(
                    f"{Colors.BLUE}INFO: Cleared {gone + gone_dirs} thumbnail "
                    f"cache entries for files the library no longer has."
                    f"{Colors.RESET}"
                )

            # Pre-generate clean files for Exhibition Mode (safe cross-platform call)
            pregenerate_exhibition_cache()

        except sqlite3.DatabaseError as e:
            print(f"ERROR initializing database: {e}")


def get_filter_options_from_db(conn, scope, folder_path=None, recursive=True):
    """
    Extracts extensions and prefixes for dropdowns using a robust
    Python-side path filtering to handle mixed slashes and cross-platform issues.
    """
    extensions, prefixes = set(), set()
    prefix_limit_reached = False

    # Identical helper to gallery_view for consistency
    def safe_path_norm(p):
        if not p:
            return ""
        return os.path.normpath(str(p).replace("\\", "/")).replace("\\", "/").lower().rstrip("/")

    try:
        # We fetch all names and paths. For very large DBs (100k+ files),
        # this is still faster than failing with a wrong SQL LIKE.
        cursor = conn.execute("SELECT name, path FROM files")

        target_norm = safe_path_norm(folder_path)

        # FILTERING LOGIC (Same as Gallery View), and normalising the same
        # way it does: only what the branch taken actually reads. In global
        # scope no path is looked at at all, and outside the strict-local
        # branch the directory never is -- both were computed for every
        # row regardless.
        recursive_prefix = target_norm + "/"
        for row in cursor:
            f_name = row["name"]

            if scope == "global":
                show_file = True
            elif recursive:
                # Check if it's inside the target folder tree
                show_file = safe_path_norm(row["path"]).startswith(recursive_prefix)
            else:
                # Strict local: must be in this exact folder
                f_path_norm = safe_path_norm(row["path"])
                show_file = safe_path_norm(os.path.dirname(f_path_norm)) == target_norm

            if show_file:
                # 1. Extensions
                _, ext = os.path.splitext(f_name)
                if ext:
                    ext_clean = ext.lstrip(".").lower()
                    if ext_clean not in ["txt", "md"]:
                        extensions.add(ext_clean)

                # 2. Prefixes
                if not prefix_limit_reached and "_" in f_name:
                    pfx = f_name.split("_")[0]
                    if pfx:
                        prefixes.add(pfx)
                        if len(prefixes) > MAX_PREFIX_DROPDOWN_ITEMS:
                            prefix_limit_reached = True
                            prefixes.clear()

    except Exception as e:
        _logger.debug("handled a failure in get_filter_options_from_db", exc_info=True)
        print(f"Error extracting options: {e}")

    return sorted(extensions), sorted(prefixes), prefix_limit_reached


# --- USER SECURITY ---
# Passwords are one-way hashed by sg_auth (Argon2id); there is no decrypt
# path. ENCRYPTION_KEY_FILE (defined above) is retained only as the path to
# the legacy Fernet key consumed by sg_auth.migrate_legacy_passwords().


def ensure_admin_user(conn):
    """Checks for admin user and applies password from startup config."""
    # --- Ensure Admin is updated for BOTH Exhibition and Force Login modes ---
    if not (IS_EXHIBITION_MODE or FORCE_LOGIN) or ADMIN_CONFIG_MISSING:
        return

    admin = conn.execute("SELECT password FROM users WHERE username = 'admin'").fetchone()

    if not admin:
        conn.execute(
            """
            INSERT INTO users (username, password, full_name, role, is_active)
            VALUES ('admin', ?, 'System Administrator', 'ADMIN', 1)
        """,
            (sg_auth.hash_password(ADMIN_PASS_INPUT),),
        )
        print(f"{Colors.GREEN}USER SETUP: Admin account initialized.{Colors.RESET}")
    else:
        # Avoid rewriting the hash on every boot when the password hasn't changed.
        valid, _ = sg_auth.verify_password(admin["password"], ADMIN_PASS_INPUT)
        if valid:
            print(f"{Colors.CYAN}USER SETUP: Admin password verified.{Colors.RESET}")
        else:
            conn.execute(
                "UPDATE users SET password = ? WHERE username = 'admin'", (sg_auth.hash_password(ADMIN_PASS_INPUT),)
            )
            print(f"{Colors.CYAN}USER SETUP: Admin password updated.{Colors.RESET}")
    conn.commit()


def is_file_accessible(file_id):
    """Checks if the current user has permission to access this specific file."""
    if not IS_EXHIBITION_MODE and not FORCE_LOGIN:
        return True

    user_role = session.get("role", "GUEST")
    user_id = str(session.get("user_id", ""))

    # Privileged roles always have full access to all files
    if user_role in ["ADMIN", "MANAGER", "STAFF"]:
        return True

    if not IS_EXHIBITION_MODE:
        # If in standard mode with FORCE_LOGIN, non-staff users should not be able to access files
        return False

    # Exhibition Mode: File MUST belong to a public collection OR a collection shared with this specific user
    with get_db_connection() as conn:
        query = """
            SELECT 1
            FROM collection_files cf
            JOIN collections c ON cf.collection_id = c.id
            WHERE cf.file_id = ?
            AND c.type = 'user_album'
        """
        params = [file_id]
        if user_id:
            # Bound, not escaped by hand. This read `user_id.replace("'",
            # "''")` pasted into the statement, which is correct only for as
            # long as the value stays a plain string and nobody adds a
            # second one beside it.
            query += " AND (c.is_public = 1 OR (',' || c.shared_users || ',') LIKE ?)"
            params.append(f"%,{user_id},%")
        else:
            query += " AND c.is_public = 1"

        result = conn.execute(query, params).fetchone()
        return bool(result)


def should_strip_metadata():
    """Helper to determine if metadata stripping is required based on session and flags."""
    user_role = session.get("role", "GUEST")  # Default to GUEST if not set
    privileged_roles = ["ADMIN", "MANAGER", "STAFF", "FRIEND"]

    is_guest = user_role not in privileged_roles
    # The protection is ACTIVE if we are in Exhibition mode OR Force Login is on
    # AND the user is NOT staff/admin.
    return (FORCE_LOGIN or IS_EXHIBITION_MODE) and is_guest

    # console log
    # print(f"--- SECURITY CHECK ---")
    # print(f"User Role in Session: {user_role}")
    # print(f"Force Login: {FORCE_LOGIN} | Exhibition Mode: {IS_EXHIBITION_MODE}")
    # print(f"Result: {'!!! STRIPPING ACTIVE !!!' if active else 'Serving Original'}")


# Fields a visitor has no business reading: how the picture was made, and
# where it lives on the server. The exhibition interface uses none of them.
_HIDDEN_FROM_GUESTS = (
    "path",
    "workflow_prompt",
    "workflow_files",
    "workflow_hash",
    "prompt_hash",
    "models_hash",
    "ai_embedding",
)


def redact_file_listing(files):
    """Strip generation details out of a file listing bound for a guest.

    Serving each file through a rewritten copy is only half the job: the
    listing that describes those files is JSON the same visitor can read,
    and it carried workflow_prompt verbatim. Cleaning the file while
    publishing its prompt beside it protects nothing.
    """
    if not should_strip_metadata():
        return files
    return [{key: value for key, value in dict(row).items() if key not in _HIDDEN_FROM_GUESTS} for row in files]


def strip_media_metadata(input_path, output_path, file_type):
    """
    Strips metadata.
    - Images & Animated Images (WebP/GIF): Rebuilt frame-by-frame via Pillow (safest for privacy).
    - Videos: Stripped via FFmpeg stream copy (fastest).
    """
    try:
        # --- CASE A: IMAGES & ANIMATIONS (PNG, JPG, WebP, GIF) ---
        if file_type in ["image", "animated_image"]:
            with Image.open(input_path) as img:
                # Check if it's an animation (Animated WebP or GIF)
                if getattr(img, "is_animated", False):
                    frames = []
                    durations = []
                    # Logic: We extract pixels frame by frame to a NEW list.
                    # This completely discards any metadata chunks (EXIF, XMP, Comfy workflow).
                    for frame in ImageSequence.Iterator(img):
                        # Create a fresh copy of the pixel data only
                        new_frame = frame.copy().convert(frame.mode)
                        frames.append(new_frame)
                        # Keep the original timing
                        durations.append(frame.info.get("duration", 100))

                    # Save the new reconstructed animation
                    frames[0].save(
                        output_path,
                        save_all=True,
                        append_images=frames[1:],
                        duration=durations,
                        loop=img.info.get("loop", 0),
                        optimize=True,
                        exif=b"",  # Extra safety
                        xmp=b"",  # Extra safety
                    )
                else:
                    # Static image: Save pixel data only, explicitly stripping EXIF/XMP
                    img.save(output_path, img.format, optimize=True, exif=b"", xmp=b"")
            return True

        # --- CASE C: DOCUMENTS (Bypass stripping, just copy safely) ---
        if file_type == "document" or input_path.lower().endswith((".txt", ".md")):
            shutil.copy2(input_path, output_path)
            return True

        # --- CASE B: REAL VIDEOS & AUDIO (MP4, MOV, MKV, MP3, WAV...) ---
        if file_type in ["video", "audio"] and FFPROBE_EXECUTABLE_PATH:
            ffmpeg_dir = os.path.dirname(FFPROBE_EXECUTABLE_PATH)
            ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
            ffmpeg_path = os.path.join(ffmpeg_dir, ffmpeg_name)
            if not os.path.exists(ffmpeg_path):
                ffmpeg_path = ffmpeg_name

            cmd = [
                ffmpeg_path,
                "-y",
                "-i",
                input_path,
                "-map_metadata",
                "-1",  # Strips global metadata
                "-map_metadata:s:v",
                "-1",  # Strips video stream metadata
                "-map_metadata:s:a",
                "-1",  # Strips audio stream metadata
                "-c",
                "copy",  # Fast stream copy (safe for these formats)
                output_path,
            ]

            result = run_media_tool(cmd, timeout=FFMPEG_TIMEOUT)

            if result.returncode == 0 and os.path.exists(output_path):
                return True
            print(f"FFMPEG VIDEO STRIP ERROR: {result.stderr}")

    except Exception as e:
        _logger.debug("handled a failure in strip_media_metadata", exc_info=True)
        print(f"RECONSTRUCTION STRIP ERROR: {e}")
    return False


# --- FLASK ROUTES ---
@app.route("/galleryout/")
@app.route("/")
def gallery_redirect_base():
    return redirect(url_for("gallery_view", folder_key="_root_"))


@app.route("/galleryout/aidam")
@management_api_only
def aidam_dashboard():
    """The AI DAM dashboard: pipeline status, generation-parameter
    analytics, face workspace, and the detector comparison tool as a
    full page (the per-file modal keeps only quick context)."""
    return render_template(
        "aidam_dashboard.html", file_id=request.args.get("file", ""), view=request.args.get("view", "status")
    )


@app.route("/galleryout/api/genparams/summary")
@management_api_only
def genparams_summary():
    """Aggregates over the first-class generation_params table: tracked
    coverage, per-tool counts, top models/samplers, steps and cfg
    distributions. Everything computed in SQL on the typed columns."""
    with get_db_connection() as conn:
        total_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        tracked = conn.execute("SELECT COUNT(*) FROM generation_params").fetchone()[0]
        tools = [
            dict(r)
            for r in conn.execute("SELECT tool, COUNT(*) AS n FROM generation_params GROUP BY tool ORDER BY n DESC")
        ]
        models = [
            dict(r)
            for r in conn.execute(
                "SELECT model, COUNT(*) AS n FROM generation_params "
                "WHERE model IS NOT NULL GROUP BY model ORDER BY n DESC LIMIT 15"
            )
        ]
        samplers = [
            dict(r)
            for r in conn.execute(
                "SELECT sampler, COUNT(*) AS n FROM generation_params "
                "WHERE sampler IS NOT NULL GROUP BY sampler ORDER BY n DESC LIMIT 12"
            )
        ]
        steps = [
            dict(r)
            for r in conn.execute(
                "SELECT steps, COUNT(*) AS n FROM generation_params "
                "WHERE steps IS NOT NULL GROUP BY steps ORDER BY n DESC LIMIT 12"
            )
        ]
        cfg = [
            dict(r)
            for r in conn.execute(
                "SELECT ROUND(cfg * 2) / 2.0 AS cfg, COUNT(*) AS n "
                "FROM generation_params WHERE cfg IS NOT NULL "
                "GROUP BY ROUND(cfg * 2) ORDER BY n DESC LIMIT 12"
            )
        ]
        sizes = [
            dict(r)
            for r in conn.execute(
                "SELECT width || 'x' || height AS size, COUNT(*) AS n "
                "FROM generation_params WHERE width IS NOT NULL "
                "GROUP BY width, height ORDER BY n DESC LIMIT 10"
            )
        ]
        negatives = conn.execute("SELECT COUNT(*) FROM generation_params WHERE negative_prompt != ''").fetchone()[0]
    return jsonify(
        {
            "total_files": total_files,
            "tracked": tracked,
            "with_negative": negatives,
            "tools": tools,
            "models": models,
            "samplers": samplers,
            "steps": steps,
            "cfg": cfg,
            "sizes": sizes,
        }
    )


def account_expired_on(user_row):
    """When this account's access lapsed, or None if it has not.

    `expiry_date` is written by the user manager and drawn there in red,
    with an hourglass, once the date has passed -- and nothing ever read it
    back. An operator handing a client access until Friday saw the account
    marked expired on Saturday and it still logged in, which is the worst
    arrangement: the interface asserts a rule the server does not keep.

    A value that cannot be read as a date is treated as no expiry and said
    so on the console. Locking someone out of their own gallery over a
    malformed field is a worse failure than the one being fixed, and the
    field is optional to begin with.
    """
    try:
        raw = (user_row["expiry_date"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return None
    if not raw:
        return None
    try:
        # Both branches end up aware, in this machine's zone. fromisoformat
        # returns an AWARE datetime when the stored value carries an offset,
        # and comparing one of those with a naive now() raises TypeError --
        # so an expiry_date written with a timezone took the account down
        # rather than expiring it.
        when = datetime.fromisoformat(raw).astimezone()
    except ValueError:
        try:
            when = datetime.strptime(raw[:10], "%Y-%m-%d").astimezone()
        except ValueError:
            print(
                f"WARNING: user {user_row['username']!r} has an unreadable "
                f"expiry_date ({raw!r}); treating the account as never expiring."
            )
            return None
    return when if when < datetime.now().astimezone() else None


@app.route("/galleryout/login", methods=["POST"])
def exhibition_login():

    # Use silent=True to prevent 400 Bad Request if headers/content are malformed
    data = request.get_json(silent=True) or {}
    username = data.get("username", "")
    password = data.get("password", "")
    provided_uuid = data.get("provided_uuid")

    # Guest Login Logic
    if ENABLE_GUEST_LOGIN and username and str(username).lower() == "guest":
        session.permanent = False

        # A returning guest may carry their previous id so their own
        # ratings and comments stay theirs -- but the value is supplied by
        # the caller, so it may only ever be a GUEST id.
        #
        # Without this check a guest could log in (no password) claiming
        # any identity, including a real account's user_id: ownership of
        # comments and ratings is decided by comparing that session id to
        # the row's client_uuid, so claiming "41" handed a stranger the
        # power to edit and delete user 41's comments and overwrite their
        # ratings. Anything that is not a well-formed guest id is ignored
        # and a fresh one is minted instead.
        candidate = str(provided_uuid) if provided_uuid is not None else ""
        guest_uuid = candidate if _is_guest_uuid(candidate) else f"guest_{secrets.token_hex(8)}"

        session["user_id"] = guest_uuid
        session["username"] = "guest"
        session["role"] = "GUEST"
        session["full_name"] = "Guest User"
        return jsonify({"status": "success", "role": "GUEST", "client_uuid": guest_uuid})

    # Standard User Login Logic
    with get_db_connection() as conn:
        user = conn.execute("SELECT * FROM users WHERE username = ? AND is_active = 1", (username,)).fetchone()
        is_valid = False
        if user:
            if username == "admin" and ADMIN_PASS_INPUT:
                # Total constant-time compare: never raises on non-ASCII or
                # non-string (JSON int/list) password input, so a crafted
                # request can't 500 this path and a non-ASCII admin passphrase
                # still authenticates.
                is_valid = sg_auth.constant_time_equals(password, ADMIN_PASS_INPUT)
            else:
                is_valid, needs_rehash = sg_auth.verify_password(user["password"], password)
                if is_valid and needs_rehash:
                    conn.execute(
                        "UPDATE users SET password = ? WHERE user_id = ?",
                        (sg_auth.hash_password(password), user["user_id"]),
                    )
                    conn.commit()
        else:
            # No active user row: still perform one Argon2id verification
            # against a decoy so login latency does not reveal whether the
            # username exists (account-enumeration mitigation).
            sg_auth.dummy_verify()

        # Checked only after the password has been verified, so a wrong
        # password still gets the same generic answer as an unknown user and
        # nothing here tells a stranger which accounts exist.
        #
        # The admin's env/CLI password is exempt. That credential belongs to
        # whoever controls the machine rather than to the account, and it is
        # the documented way back in; letting an expiry date disable it would
        # mean an operator who put a date on their own admin account had no
        # route to their gallery except editing the database by hand.
        if user and is_valid:
            expired_on = account_expired_on(user)
            if expired_on and not (username == "admin" and ADMIN_PASS_INPUT):
                return jsonify(
                    {
                        "status": "error",
                        "message": f"This account expired on "
                        f"{expired_on.strftime('%Y-%m-%d %H:%M')}. "
                        f"Ask an administrator to extend it.",
                    }
                ), 403

        if user and is_valid:
            try:
                conn.execute("UPDATE users SET last_login = ? WHERE user_id = ?", (time.time(), user["user_id"]))
                conn.commit()
            except Exception as e:
                _logger.debug("handled a failure in exhibition_login", exc_info=True)
                print(f"Login timestamp update error: {e}")

            session.permanent = False
            session["user_id"] = str(user["user_id"])
            session["username"] = user["username"]
            session["role"] = user["role"]
            session["full_name"] = user["full_name"]
            return jsonify({"status": "success", "role": user["role"]})

    return jsonify({"status": "error", "message": "Invalid credentials"}), 401


@app.route("/galleryout/logout")
def exhibition_logout():
    session.clear()
    return redirect(url_for("gallery_view", folder_key="_root_"))


def _user_write_error(exc):
    """What went wrong with a user, said so somebody can act on it.

    The database enforces both of these, and the raw text it raises --
    "UNIQUE constraint failed: users.username" -- is not something to show
    a person who has just typed a name that is already taken.
    """
    text = str(exc)
    if "users.username" in text:
        return "That username is already taken."
    if "role IN" in text:
        return "That is not a role this gallery knows."
    return text


@app.route("/galleryout/api/admin/users", methods=["GET", "POST", "PUT", "DELETE"])
def admin_manage_users():
    user_role = session.get("role")
    user_id = session.get("user_id")

    if user_id or user_role:
        if user_role not in ["ADMIN", "MANAGER"]:
            abort(403)
    elif IS_EXHIBITION_MODE or FORCE_LOGIN:
        abort(401)

    with get_db_connection() as conn:
        if request.method == "GET":
            rows = conn.execute("SELECT * FROM users WHERE username != 'admin' ORDER BY user_id DESC").fetchall()
            users = []
            for r in rows:
                d = dict(r)
                d.pop("password", None)
                users.append(d)
            return jsonify({"status": "success", "users": users})

        data = request.json

        # --- SECURITY CHECK: Enforce 8-char minimum for all users ---
        # Passwords can never be displayed back (one-way hashes), so an edit
        # (PUT) may omit the password to keep the current one unchanged.
        if request.method in ["POST", "PUT"]:
            password_input = data.get("password", "").strip()
            password_optional = request.method == "PUT" and not password_input
            if not password_optional and len(password_input) < 8:
                return jsonify({"status": "error", "message": "Password must be at least 8 characters long."}), 400

        if request.method == "POST":
            # CREATE
            hashed_pass = sg_auth.hash_password(data["password"])
            try:
                conn.execute(
                    """
                    INSERT INTO users (username, password, full_name, role, email, phone_number, expiry_date, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        data["username"],
                        hashed_pass,
                        data["full_name"],
                        data["role"],
                        data.get("email"),
                        data.get("phone_number"),
                        data.get("expiry_date"),
                        data.get("is_active", 1),
                    ),
                )
                conn.commit()
                return jsonify({"status": "success"})
            except Exception as e:
                _logger.debug("handled a failure in admin_manage_users", exc_info=True)
                return jsonify({"status": "error", "message": _user_write_error(e)}), 400

        if request.method == "PUT":
            # EDIT (empty password = keep the currently stored hash)
            # The same guard as creating one. Without it, renaming somebody
            # to a username that already exists -- an ordinary slip -- came
            # back as a 500 with no body at all, so the screen could not
            # even say what was wrong.
            user_id = data.get("user_id")
            try:
                if password_input:
                    conn.execute(
                        """
                        UPDATE users SET
                            username=?, password=?, full_name=?, role=?, email=?,
                            phone_number=?, expiry_date=?, is_active=?
                        WHERE user_id=? AND username != 'admin'
                    """,
                        (
                            data["username"],
                            sg_auth.hash_password(password_input),
                            data["full_name"],
                            data["role"],
                            data.get("email"),
                            data.get("phone_number"),
                            data.get("expiry_date"),
                            data.get("is_active"),
                            user_id,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE users SET
                            username=?, full_name=?, role=?, email=?,
                            phone_number=?, expiry_date=?, is_active=?
                        WHERE user_id=? AND username != 'admin'
                    """,
                        (
                            data["username"],
                            data["full_name"],
                            data["role"],
                            data.get("email"),
                            data.get("phone_number"),
                            data.get("expiry_date"),
                            data.get("is_active"),
                            user_id,
                        ),
                    )
                conn.commit()
            except Exception as e:
                _logger.debug("handled a failure in admin_manage_users", exc_info=True)
                return jsonify({"status": "error", "message": _user_write_error(e)}), 400
            return jsonify({"status": "success"})

        if request.method == "DELETE":
            # DELETE
            data = request.json
            user_id = data.get("user_id")
            if not user_id:
                return jsonify({"status": "error", "message": "Missing User ID"}), 400

            # Perform physical deletion
            conn.execute("DELETE FROM users WHERE user_id = ? AND username != 'admin'", (user_id,))
            conn.commit()
            return jsonify({"status": "success"})
    return None


# AI QUEUE SUBMISSION ROUTE
@app.route("/galleryout/ai_queue", methods=["POST"])
@management_api_only
def ai_queue_search():
    """
    Receives a search query from the frontend and adds it to the DB queue.
    Also performs basic housekeeping (cleaning old requests).
    """
    data = request.json
    query = data.get("query", "").strip()
    # FIX: Leggi il limite dal JSON (default 100 se non presente)
    limit = int(data.get("limit", 100))

    if not query:
        return jsonify({"status": "error", "message": "Query cannot be empty"}), 400

    session_id = str(uuid.uuid4())

    try:
        with get_db_connection() as conn:
            # 1. Housekeeping
            conn.execute("DELETE FROM ai_search_queue WHERE created_at < datetime('now', '-1 hour')")
            conn.execute(
                "DELETE FROM ai_search_results WHERE session_id NOT IN (SELECT session_id FROM ai_search_queue)"
            )

            # 2. Insert new request WITH LIMIT
            # Assicurati che la query SQL includa la colonna limit_results
            conn.execute(
                """
                INSERT INTO ai_search_queue (session_id, query, limit_results, status)
                VALUES (?, ?, ?, 'pending')
            """,
                (session_id, query, limit),
            )
            conn.commit()

        return jsonify({"status": "queued", "session_id": session_id})
    except Exception as e:
        _logger.debug("handled a failure in ai_queue_search", exc_info=True)
        print(f"AI Queue Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# AI STATUS CHECK ROUTE (POLLING)
@app.route("/galleryout/ai_check/<session_id>", methods=["GET"])
@management_api_only
def ai_check_status(session_id):
    """Checks the status of a specific search session."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT status FROM ai_search_queue WHERE session_id = ?", (session_id,)).fetchone()

        if not row:
            return jsonify({"status": "not_found"})

        return jsonify({"status": row["status"]})


@app.route("/galleryout/sync_status/<string:folder_key>")
@management_api_only
def sync_status(folder_key):
    # This does not report on a scan, it runs one: it walks the folder,
    # processes what changed and writes to the database, naming each file in
    # the stream as it goes. With no gate at all, an unidentified caller on a
    # login-protected gallery could set that going as often as they liked and
    # read the filenames back.
    # --- FIX: SILENT RESPONSE FOR VIRTUAL COLLECTIONS ---
    if folder_key.startswith("collection_"):
        # Return a dummy SSE stream that does nothing but prevents 404
        def dummy_stream():
            yield sse(status="no_changes", message="Virtual collection")

        return Response(dummy_stream(), mimetype="text/event-stream")

    folders = get_dynamic_folder_config()
    if folder_key not in folders:
        abort(404, description="That folder is not in the gallery.")
    folder_path = folders[folder_key]["path"]
    return Response(sync_folder_on_demand(folder_path), mimetype="text/event-stream")


def node_error_html(ctype, nid, err):
    """One ComfyUI node failure, as a line for the error panel.

    Built by hand at four call sites, each repeating the markup and the
    colour; a fix to one of them reached the other three only by accident.
    """
    message = err.get("message", "")
    return f"<br>• [{ctype} {nid}]: <span style='color:#ffaaaa;'>{message}</span>"


# width * height, parsed out of the "WIDTHxHEIGHT" dimensions column.
MEGAPIXELS_SQL = (
    "(CAST(SUBSTR(f.dimensions, 1, INSTR(f.dimensions, 'x') - 1) AS INTEGER)"
    " * CAST(SUBSTR(f.dimensions, INSTR(f.dimensions, 'x') + 1) AS INTEGER))"
)


def collections_in_scope(coll_id, recursive, visibility, identity):
    """(sql, params) selecting the collection ids a request covers.

    `visibility` says who the answer is for, and is spelled out rather than
    inferred so the caller's policy is visible at the call site:

        "any"              every collection in scope
        "public_or_mine"   public, or shared with `identity`
        "public_or_shared" public, or shared with anyone

    Every value is bound. This query was built twice by pasting values into
    an f-string, and the caller's identity was made safe for that by hand:

        # Raw, not escaped: it is bound as a parameter now, and doubling the
    # quotes here would make an identity containing one fail to match.
    client_identity = _current_client_identity()
        ... LIKE '%,{safe_uid},%'

    Doubling quotes is the job parameters already do, and it is only correct
    for as long as nobody adds a second value that needs different
    treatment. The identity is a bound parameter here and the LIKE pattern
    is built around it in Python, so no caller-supplied text ever reaches
    the statement.
    """
    shared_with_me = "(is_public = 1 OR (',' || shared_users || ',') LIKE ?)"
    shared_with_anyone = "(is_public = 1 OR shared_users != '')"

    if coll_id is None:
        sql = "SELECT id FROM collections WHERE type='user_album'"
        params = []
        if visibility == "public_or_mine":
            sql += f" AND {shared_with_me}"
            params.append(f"%,{identity},%")
        elif visibility == "public_or_shared":
            sql += f" AND {shared_with_anyone}"
        return sql, params

    if not recursive:
        return "SELECT ?", [int(coll_id)]

    sql = """
        WITH RECURSIVE children AS (
            SELECT id, is_public, shared_users FROM collections WHERE id = ?
            UNION ALL
            SELECT c.id, c.is_public, c.shared_users
            FROM collections c INNER JOIN children p ON c.parent_id = p.id
        )
        SELECT id FROM children
    """
    params = [int(coll_id)]
    if visibility == "public_or_mine":
        sql += f" WHERE {shared_with_me}"
        params.append(f"%,{identity},%")
    elif visibility == "public_or_shared":
        sql += f" WHERE {shared_with_anyone}"
    return sql, params


def collection_file_names(conn, coll_id, recursive, visibility, identity):
    """Every filename in the collections a request covers."""
    scope_sql, params = collections_in_scope(coll_id, recursive, visibility, identity)
    sql = (
        "SELECT DISTINCT f.name FROM files f "
        "JOIN collection_files cf ON f.id = cf.file_id "
        f"WHERE cf.collection_id IN ({scope_sql})"
    )
    return conn.execute(sql, params).fetchall()


@app.route("/galleryout/api/search_options")
def api_search_options():
    scope = request.args.get("scope", "local")
    folder_key = request.args.get("folder_key", "_root_")
    is_rec = request.args.get("recursive", "true").lower() != "false"

    exts, pfxs, limit_reached = [], [], False
    user_role = session.get("role", "GUEST")
    # Raw, not escaped: it is bound as a parameter now, and doubling the
    # quotes here would make an identity containing one fail to match.
    client_identity = _current_client_identity()
    is_local_admin = not FORCE_LOGIN and not IS_EXHIBITION_MODE
    is_privileged = is_local_admin or user_role in ["ADMIN", "MANAGER", "STAFF"]

    with get_db_connection() as conn:
        if folder_key.startswith("collection_") or IS_EXHIBITION_MODE:
            coll_id_str = folder_key.replace("collection_", "") if folder_key.startswith("collection_") else "all"
            if scope == "global":
                coll_id_str = "all"

            is_all_mode = coll_id_str == "all"
            if not is_all_mode and coll_id_str.isdigit():
                # NOTE: this path applies the shared-with-me filter only in
                # exhibition mode, while collection_view applies it whenever
                # the caller is not privileged. The two were built by hand in
                # two places and have drifted; the difference is preserved
                # here rather than picked, because choosing one changes who
                # can see which album.
                visibility = "public_or_mine" if IS_EXHIBITION_MODE and not is_privileged else "any"
                ext_rows = collection_file_names(conn, coll_id_str, is_rec, visibility, client_identity)
            elif is_all_mode:
                if not IS_EXHIBITION_MODE:
                    visibility = "any"
                elif is_privileged:
                    visibility = "public_or_shared"
                else:
                    visibility = "public_or_mine"
                ext_rows = collection_file_names(conn, None, is_rec, visibility, client_identity)
            else:
                ext_rows = []

            extensions = set()
            prefixes = set()
            for r in ext_rows:
                fname = r["name"]
                if "." in fname:
                    ext_clean = fname.split(".")[-1].lower()
                    if ext_clean not in ["txt", "md"]:
                        extensions.add(ext_clean)
                if not limit_reached and "_" in fname:
                    pfx = fname.split("_")[0]
                    if pfx:
                        prefixes.add(pfx)
                        if len(prefixes) > MAX_PREFIX_DROPDOWN_ITEMS:
                            limit_reached = True
                            prefixes.clear()
            exts = sorted(extensions)
            pfxs = sorted(prefixes) if not limit_reached else []
        else:
            folders = get_dynamic_folder_config()
            folder_path = folders.get(folder_key, {}).get("path", BASE_OUTPUT_PATH)
            exts, pfxs, limit_reached = get_filter_options_from_db(conn, scope, folder_path, recursive=is_rec)

    return jsonify({"extensions": exts, "prefixes": pfxs, "prefix_limit_reached": limit_reached})


@app.route("/galleryout/api/compare_files", methods=["POST"])
def compare_files_api():
    if should_strip_metadata():
        return jsonify(
            {"status": "error", "message": "Security Policy: Metadata comparison is restricted for your role."}
        ), 403

    data = request.json
    id_a = data.get("id_a")
    id_b = data.get("id_b")

    if not id_a or not id_b:
        return jsonify({"status": "error", "message": "Missing file IDs"}), 400

    def get_flat_params(file_id):
        try:
            info = get_file_info_from_db(file_id)
            wf_json = extract_workflow(info["path"])
            if not wf_json:
                return {}

            summary = generate_node_summary(wf_json)
            if not summary:
                return {}

            flat_params = {}
            for node in summary:
                node_type = node["type"]
                for p in node["params"]:
                    key = f"{node_type} > {p['name']}"
                    flat_params[key] = str(p["value"])
            return flat_params
        except Exception:
            # Three helpers deep -- database, workflow extraction, node
            # summarising -- and the comparison view degrades to "no
            # parameters" for whatever any of them raises.
            _logger.debug("handled a failure in get_flat_params", exc_info=True)
            return {}

    try:
        params_a = get_flat_params(id_a)
        params_b = get_flat_params(id_b)

        all_keys = sorted(set(params_a.keys()) | set(params_b.keys()))

        diff_table = []
        for key in all_keys:
            val_a = params_a.get(key, "N/A")
            val_b = params_b.get(key, "N/A")

            is_diff = str(val_a).lower() != str(val_b).lower()

            diff_table.append({"key": key, "val_a": val_a, "val_b": val_b, "is_diff": is_diff})

        diff_table.sort(key=lambda x: (not x["is_diff"], x["key"]))

        return jsonify({"status": "success", "diff": diff_table})

    except Exception as e:
        _logger.debug("handled a failure in compare_files_api", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# --- AI MANAGER API ROUTES ---
@app.route("/galleryout/ai_indexing/reset", methods=["POST"])
@management_api_only
def ai_indexing_reset():
    """
    Resets AI metadata (caption, embedding, timestamp) for specific files or a whole folder.
    CRITICAL: Also removes these files from the indexing queue to prevent re-processing.
    """
    if not ENABLE_AI_SEARCH:
        return jsonify({"status": "error"})
    data = request.json

    # Mode 1: Batch IDs
    file_ids = data.get("file_ids", [])

    # Mode 2: Folder Path
    folder_key = data.get("folder_key")
    recursive = data.get("recursive", True)

    count = 0

    try:
        with get_db_connection() as conn:
            ids_to_wipe = []

            # Case A: Specific File IDs (Selection or Lightbox)
            if file_ids:
                ids_to_wipe = file_ids

            # Case B: Folder (Recursive or Flat)
            elif folder_key:
                folders = get_dynamic_folder_config()
                if folder_key in folders:
                    folder_path = folders[folder_key]["path"]
                    # Normalize for robust DB lookup
                    target_norm = os.path.normpath(folder_path).replace("\\", "/").lower()
                    if not target_norm.endswith("/"):
                        target_norm += "/"

                    # Fetch candidates to wipe
                    cursor = conn.execute(
                        "SELECT id, path FROM files WHERE ai_caption IS NOT NULL OR ai_embedding IS NOT NULL"
                    )
                    for row in cursor:
                        f_path = row["path"]
                        # Normalize DB path
                        f_path_norm = os.path.normpath(f_path).replace("\\", "/").lower()

                        is_match = False
                        if recursive:
                            if f_path_norm.startswith(target_norm):
                                is_match = True
                        else:
                            # Strict parent check
                            parent_norm = os.path.dirname(f_path_norm).replace("\\", "/").lower() + "/"
                            if parent_norm == target_norm:
                                is_match = True

                        if is_match:
                            ids_to_wipe.append(row["id"])

            if ids_to_wipe:
                # Process in chunks to avoid SQL limits
                chunk_size = 500
                for i in range(0, len(ids_to_wipe), chunk_size):
                    chunk = ids_to_wipe[i : i + chunk_size]
                    placeholders = ",".join(["?"] * len(chunk))

                    # 1. WIPE METADATA (Instant)
                    conn.execute(
                        f"""
                        UPDATE files
                        SET ai_caption=NULL, ai_embedding=NULL, ai_last_scanned=0, ai_error=NULL
                        WHERE id IN ({placeholders})
                    """,
                        chunk,
                    )

                    # 2. REMOVE FROM PROCESSING QUEUE (Critical fix)
                    # We must delete pending jobs for these files to stop the worker from indexing them
                    conn.execute(
                        f"""
                        DELETE FROM ai_indexing_queue
                        WHERE file_id IN ({placeholders})
                    """,
                        chunk,
                    )

                count = len(ids_to_wipe)
                conn.commit()

        return jsonify(
            {"status": "success", "count": count, "message": f"AI data erased and queue cleared for {count} files."}
        )

    except Exception as e:
        _logger.debug("handled a failure in ai_indexing_reset", exc_info=True)
        print(f"AI Reset Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/ai_indexing/add_files", methods=["POST"])
@management_api_only
def ai_indexing_add_files():
    if not ENABLE_AI_SEARCH:
        return jsonify({"status": "error"})
    data = request.json
    file_ids = data.get("file_ids", [])
    force_index = data.get("force", False)
    params = json.dumps({"beams": data.get("beams", 3), "precision": data.get("precision", "fp16")})

    count = 0
    skipped = 0

    with get_db_connection() as conn:
        # --- NEW: WIPE DATA IF FORCED ---
        if force_index and file_ids:
            # We must wipe database fields before queuing
            placeholders = ",".join(["?"] * len(file_ids))
            conn.execute(
                f"""
                UPDATE files
                SET ai_caption=NULL, ai_embedding=NULL, ai_last_scanned=0, ai_error=NULL
                WHERE id IN ({placeholders})
            """,
                file_ids,
            )

        for fid in file_ids:
            # Check current status
            row = conn.execute("SELECT path, ai_last_scanned FROM files WHERE id=?", (fid,)).fetchone()
            if row:
                # --- INCREMENTAL LOGIC ---
                has_ai_data = row["ai_last_scanned"] and row["ai_last_scanned"] > 0

                if not force_index and has_ai_data:
                    skipped += 1
                    continue

                p_key = get_standardized_path(row["path"])
                # FIX: Use "ON CONFLICT DO UPDATE" to reset status to 'pending'
                conn.execute(
                    """
                    INSERT INTO ai_indexing_queue (file_path, file_id, status, created_at, force_index, params)
                    VALUES (?, ?, 'pending', ?, ?, ?)
                    ON CONFLICT(file_path) DO UPDATE SET
                        status = 'pending',
                        force_index = excluded.force_index,
                        created_at = excluded.created_at,
                        params = excluded.params
                """,
                    (p_key, fid, time.time(), 1 if force_index else 0, params),
                )
                count += 1
        conn.commit()

    # --- FEEDBACK MESSAGES ---
    if count == 0 and skipped > 0:
        return jsonify(
            {
                "status": "warning",
                "message": "All selected files are already indexed. Enable 'Force Re-Index' to overwrite.",
                "count": 0,
            }
        )

    msg = f"Queued {count} files."
    if skipped > 0:
        msg += f" (Skipped {skipped} already indexed)"

    return jsonify({"status": "success", "count": count, "message": msg})


@app.route("/galleryout/ai_indexing/add_folder", methods=["POST"])
@management_api_only
def ai_indexing_add_folder():
    if not ENABLE_AI_SEARCH:
        return jsonify({"status": "error"})
    data = request.json

    folder_key = data.get("folder_key")
    recursive = data.get("recursive", True)
    watch = data.get("watch", False)
    force = data.get("force", False)

    folders = get_dynamic_folder_config()
    if folder_key not in folders:
        return jsonify({"status": "error", "message": "Folder not found"}), 404

    raw_path = folders[folder_key]["path"]
    std_path = get_standardized_path(raw_path)

    params = json.dumps({"beams": data.get("beams", 3), "precision": data.get("precision", "fp16")})
    msg = "Indexing queued."

    # 1. HANDLE WATCH LIST UPDATE
    with get_db_connection() as conn:
        if watch:
            existing = conn.execute("SELECT path, recursive FROM ai_watched_folders").fetchall()
            should_add = True
            for row in existing:
                exist_std = get_standardized_path(row["path"])
                if exist_std == std_path:
                    # Update recursion if needed
                    if recursive and not row["recursive"]:
                        conn.execute("UPDATE ai_watched_folders SET recursive=1 WHERE path=?", (row["path"],))
                    should_add = False
                    break
                if std_path.startswith(exist_std + "/") and row["recursive"]:
                    should_add = False
                    msg = "Covered by parent watcher."
                    break
            if should_add:
                conn.execute(
                    "INSERT OR REPLACE INTO ai_watched_folders (path, recursive, added_at) VALUES (?, ?, ?)",
                    (raw_path, 1 if recursive else 0, time.time()),
                )
                msg = "Folder added to Watch List & Queued."
        conn.commit()

    # --- CRITICAL FIX: REFRESH SERVER CACHE IMMEDIATELY ---
    # This ensures that subsequent UI calls see 'is_watched=True' right away.
    if watch:
        get_dynamic_folder_config(force_refresh=True)

    # 2. BACKGROUND SCAN & QUEUE
    def _scan():
        valid = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".avi", ".webm"}
        exc = {".thumbnails_cache", ".sqlite_cache", ".zip_downloads", ".AImodels", "venv", ".git"}
        files_found = []
        try:
            if recursive:
                for r, d, f in os.walk(raw_path, topdown=True, followlinks=True):
                    d[:] = [x for x in d if (not x.startswith(".") or x == ".collection_notes") and x not in exc]
                    for x in f:
                        if os.path.splitext(x)[1].lower() in valid:
                            files_found.append(os.path.join(r, x))
            else:
                for entry in os.scandir(raw_path):
                    if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid:
                        files_found.append(entry.path)
        except OSError:
            return

        # Optimize: Batch Operations
        with get_db_connection() as conn:
            ids_to_wipe = []
            queue_entries = []

            for fp in files_found:
                pk = get_standardized_path(fp)

                # --- ROBUST LOOKUP START (YOUR LOGIC) ---
                # 1. Try exact match
                row = conn.execute("SELECT id, mtime, ai_last_scanned FROM files WHERE path=?", (fp,)).fetchone()

                # 2. Try standardized match (case insensitive on Windows)
                if not row:
                    row = conn.execute("SELECT id, mtime, ai_last_scanned FROM files WHERE path=?", (pk,)).fetchone()

                # 3. Try Normalized Slash match (Fixes subfolder mismatch issues)
                if not row:
                    norm_p = fp.replace("\\", "/")
                    row = conn.execute(
                        "SELECT id, mtime, ai_last_scanned FROM files WHERE REPLACE(path, '\\', '/') = ?", (norm_p,)
                    ).fetchone()
                # --- ROBUST LOOKUP END ---

                should_queue = False
                fid = None

                if row:
                    fid = row["id"]
                    if force:
                        ids_to_wipe.append(fid)
                        should_queue = True
                    elif (row["ai_last_scanned"] or 0) < row["mtime"]:
                        should_queue = True  # Needs update (Incremental logic)
                else:
                    # New file not in DB yet - queue it, worker will retry later
                    should_queue = True

                if should_queue:
                    # Prepare for batch insertion
                    queue_entries.append((pk, fid, time.time(), 1 if force else 0, params))

            # 3. WIPE OLD DATA IF FORCED
            if ids_to_wipe:
                chunk_size = 500
                for i in range(0, len(ids_to_wipe), chunk_size):
                    chunk = ids_to_wipe[i : i + chunk_size]
                    placeholders = ",".join(["?"] * len(chunk))
                    conn.execute(
                        f"""
                        UPDATE files
                        SET ai_caption=NULL, ai_embedding=NULL, ai_last_scanned=0, ai_error=NULL
                        WHERE id IN ({placeholders})
                    """,
                        chunk,
                    )

            # 4. BATCH INSERT INTO QUEUE (UPSERT)
            if queue_entries:
                conn.executemany(
                    """
                    INSERT INTO ai_indexing_queue (file_path, file_id, status, created_at, force_index, params)
                    VALUES (?, ?, 'pending', ?, ?, ?)
                    ON CONFLICT(file_path) DO UPDATE SET
                        status = 'pending',
                        force_index = excluded.force_index,
                        created_at = excluded.created_at,
                        params = excluded.params
                """,
                    queue_entries,
                )

            conn.commit()

    threading.Thread(target=_scan, daemon=True).start()
    return jsonify({"status": "success", "message": msg})


@app.route("/galleryout/ai_indexing/watched", methods=["GET", "DELETE"])
@management_api_only
def ai_watched_folders():
    if not ENABLE_AI_SEARCH:
        return jsonify({})
    with get_db_connection() as conn:
        if request.method == "DELETE":
            path = request.json.get("folder_path")
            if not path:
                key = request.json.get("folder_key")
                folders = get_dynamic_folder_config()
                if key in folders:
                    path = folders[key]["path"]

            if path:
                # 1. Stop Watching
                conn.execute("DELETE FROM ai_watched_folders WHERE path=?", (path,))

                # 2. CLEAR QUEUE (Critical Fix)
                # When stopping watch, we ALWAYS clear pending jobs for this folder to stop immediate processing.
                # We use LIKE for path matching.
                # Ensure we handle OS separators robustly.
                std_path = get_standardized_path(path)
                # Remove exact match or subfiles
                conn.execute(
                    "DELETE FROM ai_indexing_queue WHERE file_path = ? OR file_path LIKE ?", (std_path, std_path + "/%")
                )

                # 3. WIPE DATA (Optional User Choice)
                if request.json.get("reset_data"):
                    std_target = get_standardized_path(path)
                    rows = conn.execute(
                        "SELECT id, path FROM files WHERE ai_caption IS NOT NULL OR ai_embedding IS NOT NULL"
                    ).fetchall()
                    ids_to_wipe = []
                    for r in rows:
                        p_std = get_standardized_path(r["path"])
                        if p_std == std_target or p_std.startswith(std_target + "/"):
                            ids_to_wipe.append(r["id"])

                    if ids_to_wipe:
                        # Chunk processing for huge folders
                        chunk_size = 500
                        for i in range(0, len(ids_to_wipe), chunk_size):
                            chunk = ids_to_wipe[i : i + chunk_size]
                            ph = ",".join(["?"] * len(chunk))
                            conn.execute(
                                "UPDATE files SET ai_caption=NULL, ai_embedding=NULL,"
                                f" ai_last_scanned=0, ai_error=NULL WHERE id IN ({ph})",
                                chunk,
                            )
                            # (Queue already cleared above by path, but redundant check by ID is safe)
                            conn.execute(f"DELETE FROM ai_indexing_queue WHERE file_id IN ({ph})", chunk)

                conn.commit()
                # --- FORCE CONFIG REFRESH TO UPDATE UI COLORS IMMEDIATELY ---
                get_dynamic_folder_config(force_refresh=True)

                return jsonify({"status": "success"})
            return jsonify({"status": "error"})

        rows = conn.execute("SELECT path, recursive FROM ai_watched_folders").fetchall()
        folders = get_dynamic_folder_config()
        pmap = {info["path"]: {"key": k, "name": info["display_name"]} for k, info in folders.items()}
        res = []
        for r in rows:
            m = pmap.get(r["path"])
            rel = r["path"]
            try:
                rel = os.path.relpath(r["path"], BASE_OUTPUT_PATH)
            except (TypeError, ValueError):
                # A watched folder on another Windows drive has no path
                # relative to the output root. Show the absolute one.
                pass
            if m:
                res.append(
                    {
                        "path": r["path"],
                        "rel_path": rel,
                        "key": m["key"],
                        "display_name": m["name"],
                        "recursive": bool(r["recursive"]),
                    }
                )
            else:
                res.append(
                    {
                        "path": r["path"],
                        "rel_path": rel,
                        "key": "_unknown",
                        "display_name": os.path.basename(r["path"]),
                        "recursive": bool(r["recursive"]),
                    }
                )
        return jsonify({"folders": res})


@app.route("/galleryout/ai_indexing/status")
@management_api_only
def ai_indexing_status():
    if not ENABLE_AI_SEARCH:
        return jsonify({})
    try:
        with get_db_connection() as conn:
            pending = conn.execute("SELECT COUNT(*) FROM ai_indexing_queue WHERE status='pending'").fetchone()[0]
            processing = conn.execute("SELECT file_path FROM ai_indexing_queue WHERE status='processing'").fetchone()

            # Preview Next 10 files with PRIORITY INFO
            next_rows = conn.execute(
                "SELECT file_path, force_index FROM ai_indexing_queue"
                " WHERE status='pending' ORDER BY force_index DESC, created_at ASC LIMIT 10"
            ).fetchall()

            avg = conn.execute("SELECT value FROM ai_metadata WHERE key='avg_processing_time'").fetchone()
            paused = conn.execute("SELECT value FROM ai_metadata WHERE key='indexing_paused'").fetchone()
            waiting = conn.execute("SELECT COUNT(*) FROM ai_indexing_queue WHERE status='waiting_gpu'").fetchone()[0]

            status = "Idle"
            if paused and paused["value"] == "1":
                status = "Paused"
            elif waiting > 0:
                status = "waiting_gpu"
            elif processing:
                status = "Indexing"
            elif pending > 0:
                status = "Queued"

            curr_file = ""
            if processing:
                try:
                    curr_file = os.path.relpath(processing["file_path"], BASE_OUTPUT_PATH)
                except (TypeError, ValueError):
                    curr_file = os.path.basename(processing["file_path"])

            next_files = []
            for r in next_rows:
                try:
                    p = os.path.relpath(r["file_path"], BASE_OUTPUT_PATH)
                except (TypeError, ValueError):
                    p = os.path.basename(r["file_path"])

                next_files.append({"path": p, "is_priority": bool(r["force_index"])})

            return jsonify(
                {
                    "global_status": status,
                    "pending_count": pending,
                    "current_file": curr_file,
                    "gpu_usage": 0,
                    "avg_time": float(avg["value"]) if avg else 0.0,
                    "current_job_progress": 0,
                    "current_job_total": pending + (1 if processing else 0),
                    "next_files": next_files,
                }
            )
    except Exception as e:
        _logger.debug("handled a failure in ai_indexing_status", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/galleryout/ai_indexing/control", methods=["POST"])
@management_api_only
def ai_indexing_control():
    if not ENABLE_AI_SEARCH:
        return jsonify({"status": "error"})
    action = request.json.get("action")
    with get_db_connection() as conn:
        if action == "pause":
            conn.execute("INSERT OR REPLACE INTO ai_metadata (key, value) VALUES ('indexing_paused', '1')")
        elif action == "resume":
            conn.execute("INSERT OR REPLACE INTO ai_metadata (key, value) VALUES ('indexing_paused', '0')")
            conn.execute("UPDATE ai_indexing_queue SET status='pending' WHERE status='waiting_gpu'")
        elif action == "clear":
            conn.execute("DELETE FROM ai_indexing_queue WHERE status != 'processing'")
        conn.commit()
    return jsonify({"status": "success", "message": f"Queue {action}d"})


def _current_client_identity():
    """The client_uuid this caller's ratings and comments are stored under.

    It has to agree with what the page sends, or a rating written through
    the API cannot be found again by the queries that build the grid. The
    templates mint the fixed identity 'admin' whenever there is no login to
    take one from (see index.html), so the default single-user install must
    resolve to that and not to the empty string.
    """
    user_id = session.get("user_id")
    if user_id:
        return str(user_id)
    if not FORCE_LOGIN and not IS_EXHIBITION_MODE:
        return "admin"
    return ""


def process_clustering(current_files, cluster_mode, cluster_sort, cluster_target_id, cluster_scope):
    if not cluster_mode:
        return current_files

    # Never hash inline: a first clustering click on a large un-hashed library
    # used to block this request for minutes. Kick the background worker and
    # serve honest partial clusters; the banner shows hashing progress and the
    # stale-refresh below picks up rows the worker finishes mid-request.
    with get_db_connection() as conn_check:
        unhashed_cnt = conn_check.execute(
            """SELECT COUNT(*) FROM files
               WHERE (has_workflow = 1 OR type IN ('image', 'animated_image'))
               AND (workflow_hash IS NULL OR workflow_hash = '')
               AND (prompt_hash IS NULL OR prompt_hash = '')
               AND hash_failed = 0"""
        ).fetchone()[0]
    if unhashed_cnt > 0:
        ensure_cluster_backfill_async()

    # The hash column that defines cluster identity for this mode. Files
    # missing it cannot belong to any cluster of this kind.
    primary_hash_key = {
        "prompt": "prompt_hash",
        "models": "models_hash",
    }.get(cluster_mode, "workflow_hash")

    result_files = []

    if cluster_scope == "global":
        with get_db_connection() as conn_target:
            safe_uuid = _current_client_identity().replace("'", "''")
            user_role = session.get("role", "GUEST")
            is_local_admin = not FORCE_LOGIN and not IS_EXHIBITION_MODE

            if is_local_admin or user_role in ["ADMIN", "MANAGER", "STAFF"]:
                comment_sub_filter = ""
            else:
                comment_sub_filter = (
                    f" AND (target_audience = 'public'"
                    f" OR target_audience = 'user:{safe_uuid}'"
                    f" OR client_uuid = '{safe_uuid}')"
                )

            if cluster_target_id:
                target_row = conn_target.execute(
                    "SELECT workflow_hash, prompt_hash, models_hash FROM files WHERE id = ?", (cluster_target_id,)
                ).fetchone()
                target_wf = target_row[0] if target_row else None
                target_pr = target_row[1] if target_row else None
                target_md = target_row[2] if target_row else None

                where_clause = ""
                params_t = ()
                if cluster_mode == "combo" and target_wf and target_pr:
                    where_clause = "f.workflow_hash = ? AND f.prompt_hash = ?"
                    params_t = (target_wf, target_pr)
                elif cluster_mode == "prompt" and target_pr:
                    where_clause = "f.prompt_hash = ?"
                    params_t = (target_pr,)
                elif cluster_mode == "models" and target_md:
                    where_clause = "f.models_hash = ?"
                    params_t = (target_md,)
                elif cluster_mode not in ("prompt", "models") and target_wf:
                    where_clause = "f.workflow_hash = ?"
                    params_t = (target_wf,)

                if where_clause:
                    rows = conn_target.execute(
                        f"""
                        SELECT DISTINCT f.*,
                        (SELECT c.color FROM collections c
                         JOIN collection_files cf ON c.id = cf.collection_id
                         WHERE cf.file_id = f.id AND c.type = 'system_flag' LIMIT 1) as status_color,
                        (SELECT AVG(rating) FROM file_ratings WHERE file_id = f.id) as avg_rating,
                        (SELECT COUNT(*) FROM file_ratings WHERE file_id = f.id) as vote_count,
                        (SELECT rating FROM file_ratings
                         WHERE file_id = f.id AND client_uuid = '{safe_uuid}') as my_rating,
                        (SELECT COUNT(*) FROM file_comments WHERE file_id = f.id {comment_sub_filter}) as comment_count,
                        (SELECT MAX(created_at) FROM file_comments
                         WHERE file_id = f.id {comment_sub_filter}) as latest_comment_time
                        FROM files f
                        WHERE {where_clause}
                    """,
                        params_t,
                    ).fetchall()
                    result_files = [dict(r) for r in rows]
                else:
                    result_files = []
            else:
                rows = conn_target.execute(f"""
                    SELECT DISTINCT f.*,
                    (SELECT c.color FROM collections c
                         JOIN collection_files cf ON c.id = cf.collection_id
                         WHERE cf.file_id = f.id AND c.type = 'system_flag' LIMIT 1) as status_color,
                    (SELECT AVG(rating) FROM file_ratings WHERE file_id = f.id) as avg_rating,
                    (SELECT COUNT(*) FROM file_ratings WHERE file_id = f.id) as vote_count,
                    (SELECT rating FROM file_ratings
                         WHERE file_id = f.id AND client_uuid = '{safe_uuid}') as my_rating,
                    (SELECT COUNT(*) FROM file_comments WHERE file_id = f.id {comment_sub_filter}) as comment_count,
                    (SELECT MAX(created_at) FROM file_comments
                         WHERE file_id = f.id {comment_sub_filter}) as latest_comment_time
                    FROM files f
                    WHERE f.{primary_hash_key} IS NOT NULL AND f.{primary_hash_key} != ''
                """).fetchall()
                result_files = [dict(r) for r in rows]

            for d in result_files:
                d.pop("ai_embedding", None)
    else:
        result_files = [dict(f) for f in current_files]

        # The caller fetched these rows before the backfill above ran, so a
        # first-ever clustering request would otherwise silently drop files
        # whose hashes were computed moments ago. Re-read hashes for any row
        # that still looks unhashed.
        stale_ids = [
            f["id"]
            for f in result_files
            if not str(f.get("workflow_hash") or "").strip() and not str(f.get("prompt_hash") or "").strip()
        ]
        if stale_ids:
            fresh_hashes = {}
            with get_db_connection() as conn_refresh:
                for i in range(0, len(stale_ids), 500):
                    chunk = stale_ids[i : i + 500]
                    placeholders = ",".join("?" * len(chunk))
                    for r in conn_refresh.execute(
                        f"SELECT id, workflow_hash, prompt_hash, models_hash FROM files WHERE id IN ({placeholders})",
                        chunk,
                    ).fetchall():
                        fresh_hashes[r["id"]] = (r["workflow_hash"], r["prompt_hash"], r["models_hash"])
            for f in result_files:
                if f["id"] in fresh_hashes:
                    f["workflow_hash"], f["prompt_hash"], f["models_hash"] = fresh_hashes[f["id"]]

        if cluster_target_id:
            with get_db_connection() as conn_t:
                t_row = conn_t.execute(
                    "SELECT workflow_hash, prompt_hash, models_hash FROM files WHERE id = ?", (cluster_target_id,)
                ).fetchone()
                if t_row:
                    t_wf, t_pr, t_md = t_row[0], t_row[1], t_row[2]
                    if cluster_mode == "combo":
                        result_files = [
                            f for f in result_files if f.get("workflow_hash") == t_wf and f.get("prompt_hash") == t_pr
                        ]
                    elif cluster_mode == "prompt":
                        result_files = [f for f in result_files if f.get("prompt_hash") == t_pr]
                    elif cluster_mode == "models":
                        result_files = [f for f in result_files if f.get("models_hash") == t_md]
                    else:
                        result_files = [f for f in result_files if f.get("workflow_hash") == t_wf]
                else:
                    result_files = []

    result_files = [f for f in result_files if str(f.get(primary_hash_key) or "").strip()]

    def get_inner_sort_key(item):
        if cluster_sort == "date_asc":
            return item.get("mtime") or 0
        if cluster_sort == "rating_desc":
            return -(item.get("avg_rating") or 0)
        return -(item.get("mtime") or 0)

    result_files.sort(
        key=lambda x: (x.get(primary_hash_key) or "", x.get("workflow_hash") or "", get_inner_sort_key(x))
    )
    return result_files


@app.route("/galleryout/view/<string:folder_key>")
def gallery_view(folder_key):
    # 1. SECURITY LOCKDOWN CHECK
    if ADMIN_CONFIG_MISSING:
        # f-string, not a plain one. The mode was interpolated here with
        # {...} into a string carrying no f prefix, so the lockdown page
        # printed the Python expression itself -- "python smartgallery.py
        # { '--exhibition' if IS_EXHIBITION_MODE else '--force-login' }
        # --admin-pass YOUR_PASSWORD" -- as the command to copy.
        mode_flag = "--exhibition" if IS_EXHIBITION_MODE else "--force-login"
        return (
            f"""
        <body style="background:#0a0a0a; color:#eee; font-family:sans-serif;
                     display:flex; align-items:center; justify-content:center;
                     height:100vh; text-align:center;">
            <div style="border:1px solid #dc3545; padding:40px;
                        border-radius:16px; background:#1a1a1a; max-width:500px;">
                <h1 style="color:#dc3545;">🔒 Security Lockdown</h1>
                <p>Restricted modes (--exhibition or --force-login) require an
                   Administrator Password to start.</p>
                <div style="background:#000; padding:15px; border-radius:8px;
                            font-family:monospace; margin:20px 0;">
                    python smartgallery.py {mode_flag} --admin-pass YOUR_PASSWORD
                </div>
                <p style="color:#888; font-size:0.9rem;">Please restart the server
                   with the password parameter or set the ADMIN_PASSWORD
                   environment variable.</p>
            </div>
        </body>
        """,
            403,
        )

    # 2. AUTHENTICATION & PERMISSIONS LOGIC
    is_management_side = not IS_EXHIBITION_MODE
    is_logged_in = "user_id" in session

    must_authenticate = IS_EXHIBITION_MODE or FORCE_LOGIN

    if must_authenticate:
        if not is_logged_in:
            return render_template(
                "exhibition_login.html",
                app_version=APP_VERSION,
                enable_guest_login=ENABLE_GUEST_LOGIN if IS_EXHIBITION_MODE else False,
                admin_side=is_management_side,
            )

        # --- NEW GRACEFUL ROLE PROTECTION ---
        if is_management_side:
            user_role = session.get("role")
            if user_role not in ["ADMIN", "MANAGER", "STAFF"]:
                # Block GUESTs or CUSTOMERs from management interface
                session.clear()
                return render_template(
                    "exhibition_login.html",
                    app_version=APP_VERSION,
                    enable_guest_login=False,
                    admin_side=True,
                    error_msg="Unauthorized: Your role does not have management privileges.",
                )

    # 3. REDIRECT VIRTUAL COLLECTIONS
    if folder_key.startswith("collection_"):
        try:
            # Extract ID part (can be 'all' or a numeric string)
            coll_id_raw = folder_key.split("_", 1)[1]
            if coll_id_raw == "all" or coll_id_raw.isdigit():
                # FIX: Convert MultiDict to dict with lists to preserve multi-select filters
                args_dict = request.args.to_dict(flat=False)
                # A search answers with a list of files from across the
                # library, and the collection view cannot show one: it
                # reads neither omniquery_id nor ai_session_id. Forwarding
                # a search there dropped it without a word -- the person
                # got their album back exactly as it was, as though
                # nothing had been searched for. Searching from inside an
                # album is the ordinary way to arrive here, because the
                # palette navigates to whichever key the page is showing
                # and a collection's key is `collection_<id>`.
                if "omniquery_id" in args_dict or "ai_session_id" in args_dict:
                    return redirect(url_for("gallery_view", folder_key="_root_", **args_dict))
                return redirect(url_for("collection_view", coll_id=coll_id_raw, **args_dict))
        except IndexError:
            pass

    view_files = []

    # 4. EXHIBITION MODE SECURITY CHECK
    # Prevent browsing physical folders if in Exhibition mode
    if IS_EXHIBITION_MODE and folder_key != "_root_":
        return redirect(url_for("gallery_view", folder_key="_root_"))

    # 5. FOLDER CONFIGURATION
    folders = get_dynamic_folder_config(force_refresh=True)

    # If root not found or invalid key
    if folder_key not in folders:
        return redirect(url_for("gallery_view", folder_key="_root_"))

    current_folder_info = folders[folder_key]
    folder_path = current_folder_info["path"]

    # 1. Capture All Request Parameters
    # Subfolders are included by default (ComfyUI routinely scatters outputs
    # into date-stamped subfolders); only an explicit recursive=false narrows
    # the view to the folder itself.
    is_recursive = request.args.get("recursive", "true").lower() != "false"
    search_scope = request.args.get("scope", "local")
    is_global_search = search_scope == "global"
    ai_session_id = request.args.get("ai_session_id")
    omniquery_id = request.args.get("omniquery_id")

    # Text filters
    search_term = request.args.get("search", "").strip()
    wf_files = request.args.get("workflow_files", "").strip()
    wf_prompt = request.args.get("workflow_prompt", "").strip()
    comment_search = request.args.get("comment_search", "").strip()
    start_date = request.args.get("start_date", "").strip()
    end_date = request.args.get("end_date", "").strip()
    selected_exts = request.args.getlist("extension")
    selected_prefixes = request.args.getlist("prefix")
    selected_raters = request.args.getlist("rated_by")
    selected_rating_ranges = request.args.getlist("rating_range")

    is_ai_search = False
    is_omniquery = False
    omniquery_sql = ""

    # --- PATH OMNIQUERY RESULTS ---
    if omniquery_id:
        with get_db_connection() as conn:
            try:
                session_info = conn.execute(
                    "SELECT raw_sql FROM omniquery_sessions WHERE session_id = ?", (omniquery_id,)
                ).fetchone()
                if session_info:
                    is_omniquery = True
                    omniquery_sql = session_info["raw_sql"]
                    safe_uuid = _current_client_identity().replace("'", "''")
                    rows = conn.execute(
                        f"""
                        SELECT f.*,
                        (SELECT c.color FROM collections c
                         JOIN collection_files cf2 ON c.id = cf2.collection_id
                         WHERE cf2.file_id = f.id AND c.type = 'system_flag' LIMIT 1) as status_color,
                        (SELECT AVG(rating) FROM file_ratings WHERE file_id = f.id) as avg_rating,
                        (SELECT COUNT(*) FROM file_ratings WHERE file_id = f.id) as vote_count,
                        (SELECT rating FROM file_ratings
                         WHERE file_id = f.id AND client_uuid = '{safe_uuid}') as my_rating,
                        (SELECT COUNT(*) FROM file_comments WHERE file_id = f.id) as comment_count,
                        (SELECT MAX(created_at) FROM file_comments WHERE file_id = f.id) as latest_comment_time
                        FROM omniquery_results r
                        JOIN files f ON r.file_id = f.id
                        WHERE r.session_id = ?
                        ORDER BY r.rowid ASC
                    """,
                        (omniquery_id,),
                    ).fetchall()

                    files_list = []
                    for row in rows:
                        d = dict(row)
                        d.pop("ai_embedding", None)
                        files_list.append(d)

                    # --- FIX: Apply UI Sorting ONLY if query doesn't have custom ORDER BY ---
                    # `re` is imported at module level. Importing it again HERE
                    # made the name local to the whole of gallery_view, and this
                    # branch only runs for an OmniQuery request -- so every
                    # ordinary prompt search reached the re.match below with the
                    # name unbound and answered 500.
                    has_custom_order = False
                    if omniquery_sql:
                        has_custom_order = bool(re.search(r"\bORDER\s+BY\b", omniquery_sql, re.IGNORECASE))

                    if not has_custom_order:
                        omni_sort_by = request.args.get("sort_by", "date")
                        omni_sort_desc = request.args.get("sort_order", "desc").lower() != "asc"

                        if omni_sort_by == "name":
                            files_list.sort(key=lambda x: (x.get("name") or "").lower(), reverse=omni_sort_desc)
                        elif omni_sort_by == "rating":
                            if is_effectively_blind():
                                files_list.sort(key=lambda x: x.get("my_rating") or 0, reverse=omni_sort_desc)
                            else:
                                files_list.sort(key=lambda x: x.get("avg_rating") or 0, reverse=omni_sort_desc)
                        elif omni_sort_by == "comments":
                            files_list.sort(key=lambda x: x.get("comment_count") or 0, reverse=omni_sort_desc)
                        elif omni_sort_by in ["latest_comment", "latestcomment"]:
                            files_list.sort(key=lambda x: x.get("latest_comment_time") or 0, reverse=omni_sort_desc)
                        elif omni_sort_by in ["date", "mtime"]:
                            files_list.sort(key=lambda x: x.get("mtime") or 0, reverse=omni_sort_desc)
                        else:
                            files_list.sort(key=lambda x: x.get("mtime") or 0, reverse=True)
                    # --------------------------------------------------

                    view_files = files_list
            except Exception as e:
                _logger.debug("handled a failure in gallery_view", exc_info=True)
                print(f"OmniQuery Search Error: {e}")
                is_omniquery = False

    # --- PATH A: AI SEARCH RESULTS ---
    if ENABLE_AI_SEARCH and ai_session_id:
        with get_db_connection() as conn:
            try:
                queue_info = conn.execute(
                    "SELECT query, status FROM ai_search_queue WHERE session_id = ?", (ai_session_id,)
                ).fetchone()
                if queue_info and queue_info["status"] == "completed":
                    is_ai_search = True
                    rows = conn.execute(
                        """
                        SELECT f.*, r.score FROM ai_search_results r
                        JOIN files f ON r.file_id = f.id
                        WHERE r.session_id = ? ORDER BY r.score DESC
                    """,
                        (ai_session_id,),
                    ).fetchall()

                    files_list = []
                    for row in rows:
                        d = dict(row)
                        if "ai_embedding" in d:
                            del d["ai_embedding"]
                        files_list.append(d)

                    view_files = files_list
            except Exception as e:
                _logger.debug("handled a failure in gallery_view", exc_info=True)
                print(f"AI Search Error: {e}")
                is_ai_search = False

    # --- PATH B: STANDARD VIEW / SEARCH ---
    if not is_ai_search and not is_omniquery:
        with get_db_connection() as conn:
            conditions, params = (
                ["f.type != 'document' AND LOWER(f.name) NOT LIKE '%.txt' AND LOWER(f.name) NOT LIKE '%.md'"],
                [],
            )

            if search_term:
                conditions.append("ulower(name) LIKE ulower(?)")
                params.append(f"%{search_term}%")

            if wf_files:
                for kw in [k.strip() for k in wf_files.split(",") if k.strip()]:
                    sub_kws = [s.strip() for s in kw.split(";") if s.strip()]
                    if not sub_kws:
                        continue

                    or_conds = []
                    not_conds = []
                    for s in sub_kws:
                        is_not = False
                        if s.startswith("!"):
                            is_not = True
                            s = s[1:].strip()
                        if not s:
                            continue

                        cond_str, param_val = model_condition(s, "f.workflow_files", is_not)

                        if is_not:
                            not_conds.append((cond_str, param_val))
                        else:
                            or_conds.append((cond_str, param_val))

                    if or_conds:
                        if len(or_conds) > 1:
                            conditions.append("(" + " OR ".join([c[0] for c in or_conds]) + ")")
                        elif len(or_conds) == 1:
                            conditions.append(or_conds[0][0])
                        params.extend([c[1] for c in or_conds])

                    for cond, param in not_conds:
                        conditions.append(cond)
                        params.append(param)
            if wf_prompt:
                for kw in [k.strip() for k in wf_prompt.split(",") if k.strip()]:
                    sub_kws = [s.strip() for s in kw.split(";") if s.strip()]
                    if not sub_kws:
                        continue

                    or_conds = []
                    not_conds = []
                    for s in sub_kws:
                        is_not = False
                        if s.startswith("!"):
                            is_not = True
                            s = s[1:].strip()
                        if not s:
                            continue

                        built = prompt_search_condition(s, is_not, "workflow_prompt")
                        if built is None:
                            continue
                        cond_str, param_val = built

                        if is_not:
                            not_conds.append((cond_str, param_val))
                        else:
                            or_conds.append((cond_str, param_val))

                    if or_conds:
                        if len(or_conds) > 1:
                            conditions.append("(" + " OR ".join([c[0] for c in or_conds]) + ")")
                        elif len(or_conds) == 1:
                            conditions.append(or_conds[0][0])
                        params.extend([c[1] for c in or_conds])

                    for cond, param in not_conds:
                        conditions.append(cond)
                        params.append(param)
            if comment_search:
                for kw in [k.strip() for k in comment_search.split(",") if k.strip()]:
                    sub_kws = [s.strip() for s in kw.split(";") if s.strip()]
                    if not sub_kws:
                        continue

                    or_conds = []
                    not_conds = []
                    for s in sub_kws:
                        is_not = False
                        if s.startswith("!"):
                            is_not = True
                            s = s[1:].strip()
                        if not s:
                            continue

                        op_in = "NOT IN" if is_not else "IN"
                        if s.startswith('"') and s.endswith('"') and len(s) > 2:
                            clean_s = s[1:-1]
                            col_expr = (
                                "(' ' || REPLACE(REPLACE(REPLACE(REPLACE(REPLACE("
                                "comment_text, ',', ' '), '?', ' '), '.', ' '),"
                                " '!', ' '), char(10), ' ') || ' ')"
                            )
                            cond_str = (
                                f"f.id {op_in} (SELECT file_id FROM file_comments"
                                f" WHERE ulower({col_expr}) LIKE ulower(?))"
                            )
                            param_val = f"% {clean_s} %"
                        else:
                            cond_str = (
                                f"f.id {op_in} (SELECT file_id FROM file_comments"
                                f" WHERE ulower(comment_text) LIKE ulower(?))"
                            )
                            param_val = f"%{s}%"

                        if is_not:
                            not_conds.append((cond_str, param_val))
                        else:
                            or_conds.append((cond_str, param_val))

                    if or_conds:
                        if len(or_conds) > 1:
                            conditions.append("(" + " OR ".join([c[0] for c in or_conds]) + ")")
                        elif len(or_conds) == 1:
                            conditions.append(or_conds[0][0])
                        params.extend([c[1] for c in or_conds])

                    for cond, param in not_conds:
                        conditions.append(cond)
                        params.append(param)
            if request.args.get("favorites") == "true":
                conditions.append("is_favorite = 1")
            if request.args.get("no_workflow") == "true":
                conditions.append("has_workflow = 0")
            if request.args.get("no_ai_caption") == "true":
                conditions.append("(ai_caption IS NULL OR ai_caption = '')")

            # Counted further down with the rest of them; active_filters_count
            # does not exist yet here, and the bare `except: pass` this
            # replaced was swallowing the UnboundLocalError that said so.
            day_from = day_bounds(start_date, request.args.get("start_ts"), False)
            if day_from is not None:
                conditions.append("mtime >= ?")
                params.append(day_from)
            day_to = day_bounds(end_date, request.args.get("end_ts"), True)
            if day_to is not None:
                conditions.append("mtime <= ?")
                params.append(day_to)

            if selected_rating_ranges:
                r_conds = []
                avg_sql = "IFNULL((SELECT AVG(rating) FROM file_ratings WHERE file_id = f.id), 0)"
                for rr in selected_rating_ranges:
                    if rr == "0 stars":
                        r_conds.append(f"{avg_sql} = 0")
                    elif rr == "1 star":
                        r_conds.append(f"ROUND({avg_sql}) = 1")
                    elif rr == "2 stars":
                        r_conds.append(f"ROUND({avg_sql}) = 2")
                    elif rr == "3 stars":
                        r_conds.append(f"ROUND({avg_sql}) = 3")
                    elif rr == "4 stars":
                        r_conds.append(f"ROUND({avg_sql}) = 4")
                    elif rr == "5 stars":
                        r_conds.append(f"ROUND({avg_sql}) = 5")
                    # Legacy support for old URLs/bookmarks
                    elif rr == "1-2 stars":
                        r_conds.append(f"({avg_sql} > 0 AND {avg_sql} <= 2)")
                    elif rr == "2-3 stars":
                        r_conds.append(f"({avg_sql} > 2 AND {avg_sql} <= 3)")
                    elif rr == "3-4 stars":
                        r_conds.append(f"({avg_sql} > 3 AND {avg_sql} <= 4)")
                    elif rr == "4-5 stars":
                        r_conds.append(f"({avg_sql} > 4 AND {avg_sql} <= 5)")
                if r_conds:
                    conditions.append(f"({' OR '.join(r_conds)})")

            if selected_raters:
                expanded_raters = list(selected_raters)
                if "admin" in expanded_raters:
                    try:
                        admin_id = conn.execute("SELECT user_id FROM users WHERE username = 'admin'").fetchone()
                        if admin_id and str(admin_id[0]) not in expanded_raters:
                            expanded_raters.append(str(admin_id[0]))
                    except sqlite3.Error:
                        pass
                placeholders = ",".join(["?"] * len(expanded_raters))
                conditions.append(f"f.id IN (SELECT file_id FROM file_ratings WHERE client_uuid IN ({placeholders}))")
                params.extend(expanded_raters)

            if selected_exts:
                e_cond = ["name LIKE ?" for e in selected_exts if e.strip()]
                params.extend([f"%.{e.lstrip('.').lower()}" for e in selected_exts if e.strip()])
                if e_cond:
                    conditions.append(f"({' OR '.join(e_cond)})")

            if selected_prefixes:
                p_cond = ["name LIKE ?" for p in selected_prefixes if p.strip()]
                params.extend([f"{p.strip()}_%" for p in selected_prefixes if p.strip()])
                if p_cond:
                    conditions.append(f"({' OR '.join(p_cond)})")

            req_sort_by = request.args.get("sort_by", "date")
            sort_order = "ASC" if request.args.get("sort_order", "desc").lower() == "asc" else "DESC"

            # --- COMMENT VISIBILITY FILTER FOR SORTING ---
            user_role = session.get("role", "GUEST")
            safe_uuid = _current_client_identity().replace("'", "''")

            # Allow Local Admin (no force login) to see all comments during sort
            is_local_admin = not FORCE_LOGIN and not IS_EXHIBITION_MODE

            if is_local_admin or user_role in ["ADMIN", "MANAGER", "STAFF"]:
                comment_sub_filter = ""
                comment_exists_filter = "SELECT file_id FROM file_comments"
            else:
                # Regular users only consider public comments or comments involving them
                comment_sub_filter = (
                    f" AND (target_audience = 'public'"
                    f" OR target_audience = 'user:{safe_uuid}'"
                    f" OR client_uuid = '{safe_uuid}')"
                )
                comment_exists_filter = (
                    f"SELECT file_id FROM file_comments"
                    f" WHERE (target_audience = 'public'"
                    f" OR target_audience = 'user:{safe_uuid}'"
                    f" OR client_uuid = '{safe_uuid}')"
                )

            if req_sort_by == "name":
                order_clause = f"ulower(f.name) {sort_order}"
            elif req_sort_by == "rating":
                if is_effectively_blind():
                    conditions.append(f"f.id IN (SELECT file_id FROM file_ratings WHERE client_uuid = '{safe_uuid}')")

                    order_clause = f"my_rating {sort_order}, f.mtime DESC"

                else:
                    conditions.append("f.id IN (SELECT file_id FROM file_ratings)")

                    order_clause = f"avg_rating {sort_order}, f.mtime DESC"
            elif req_sort_by == "size":
                order_clause = f"f.size {sort_order}"

            elif req_sort_by == "type":
                order_clause = f"f.type {sort_order}, ulower(f.name) ASC"

            elif req_sort_by == "duration":
                order_clause = f"f.duration {sort_order}"

            elif req_sort_by == "dimensions":
                order_clause = f"{MEGAPIXELS_SQL} {sort_order}"
            elif req_sort_by == "unrated":
                if is_effectively_blind():
                    conditions.append(
                        f"f.id NOT IN (SELECT file_id FROM file_ratings WHERE client_uuid = '{safe_uuid}')"
                    )
                else:
                    conditions.append("f.id NOT IN (SELECT file_id FROM file_ratings)")
                order_clause = f"f.mtime {sort_order}"
            elif req_sort_by == "uncommented":
                if is_effectively_blind():
                    conditions.append(
                        f"f.id NOT IN (SELECT file_id FROM file_comments WHERE client_uuid = '{safe_uuid}')"
                    )
                else:
                    conditions.append("f.id NOT IN (SELECT file_id FROM file_comments)")
                order_clause = f"f.mtime {sort_order}"
            elif req_sort_by == "comments":
                conditions.append(f"f.id IN ({comment_exists_filter})")
                order_clause = f"comment_count {sort_order}, f.mtime DESC"
            elif req_sort_by == "latest_comment":
                conditions.append(f"f.id IN ({comment_exists_filter})")
                order_clause = f"latest_comment_time {sort_order}, f.mtime DESC"
            else:
                order_clause = f"f.mtime {sort_order}"

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            query = f"""
                SELECT f.*,
                (
                    SELECT c.color
                    FROM collections c
                    JOIN collection_files cf ON c.id = cf.collection_id
                    WHERE cf.file_id = f.id AND c.type = 'system_flag'
                    LIMIT 1
                ) as status_color,
                (
                    SELECT AVG(rating) FROM file_ratings WHERE file_id = f.id
                ) as avg_rating,
                (
                    SELECT COUNT(*) FROM file_ratings WHERE file_id = f.id
                ) as vote_count,
            (SELECT rating FROM file_ratings
                         WHERE file_id = f.id AND client_uuid = '{safe_uuid}') as my_rating,
                (
                    SELECT COUNT(*) FROM file_comments WHERE file_id = f.id {comment_sub_filter}
                ) as comment_count,
                (
                    SELECT MAX(created_at) FROM file_comments WHERE file_id = f.id {comment_sub_filter}
                ) as latest_comment_time
                FROM files f
                {where_clause}
                ORDER BY {order_clause}
            """

            rows = conn.execute(query, params).fetchall()

            final_files = []

            def safe_path_norm(p):
                if not p:
                    return ""
                return os.path.normpath(str(p).replace("\\", "/")).replace("\\", "/").lower().rstrip("/")

            target_norm = safe_path_norm(folder_path)

            # Both norms used to be computed for every row before anything
            # looked at them, and only one branch of three uses the second.
            # Same decisions, same results; the work each branch does not
            # need is simply not done. At 100,000 files this loop and its
            # twin in get_filter_options_from_db called safe_path_norm
            # 200,001 times per page load.
            recursive_prefix = target_norm + "/"
            for row in rows:
                f_data = dict(row)
                f_data.pop("ai_embedding", None)

                if is_global_search:
                    final_files.append(f_data)
                elif is_recursive:
                    if safe_path_norm(f_data["path"]).startswith(recursive_prefix):
                        final_files.append(f_data)
                else:
                    f_path_norm = safe_path_norm(f_data["path"])
                    if safe_path_norm(os.path.dirname(f_path_norm)) == target_norm:
                        final_files.append(f_data)

            view_files = final_files

    active_filters_count = 0
    if search_term:
        active_filters_count += 1
    if wf_files:
        active_filters_count += 1
    if wf_prompt:
        active_filters_count += 1
    if request.args.get("comment_search", "").strip():
        active_filters_count += 1
    # A date range narrows the view like anything else here, and was the
    # one filter the badge never counted.
    if start_date:
        active_filters_count += 1
    if end_date:
        active_filters_count += 1

    if selected_exts:
        active_filters_count += 1
    if selected_prefixes:
        active_filters_count += 1
    if selected_raters:
        active_filters_count += 1
    if selected_rating_ranges:
        active_filters_count += 1
    if request.args.get("favorites") == "true":
        active_filters_count += 1
    if request.args.get("no_workflow") == "true":
        active_filters_count += 1
    if ENABLE_AI_SEARCH and request.args.get("no_ai_caption") == "true":
        active_filters_count += 1
    # Subtree inclusion is the browsing default; the narrowing states are
    # global search and the explicit folder-only opt-out.
    if is_global_search or not is_recursive:
        active_filters_count += 1

    # --- CLUSTER MODE OVERRIDE LOGIC & SCOPE SEARCH ---
    # Exhibition mode ships no clustering UI (banner/exit controls), so
    # crafted cluster URLs must not switch the view into an inescapable mode.
    cluster_mode = None if IS_EXHIBITION_MODE else request.args.get("cluster_mode")
    cluster_sort = request.args.get("cluster_sort", "date_desc")
    cluster_target_id = request.args.get("cluster_target_id")
    cluster_scope = request.args.get("cluster_scope", "global")

    view_files = process_clustering(view_files, cluster_mode, cluster_sort, cluster_target_id, cluster_scope)

    total_folder_files, _, _ = scan_folder_and_extract_options(folder_path, recursive=is_recursive)
    total_db_files = 0
    with get_db_connection() as conn_opts:
        try:
            total_db_files = conn_opts.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        except sqlite3.Error:
            total_db_files = 0

        scope_for_opts = "global" if is_global_search else "local"
        extensions, prefixes, pfx_limit = get_filter_options_from_db(
            conn_opts, scope_for_opts, folder_path, recursive=is_recursive
        )
        try:
            users_rows = conn_opts.execute(
                "SELECT user_id, full_name FROM users WHERE is_active=1 AND username != 'admin'"
            ).fetchall()
            available_raters = [{"id": str(r["user_id"]), "name": r["full_name"]} for r in users_rows]
        except sqlite3.Error:
            available_raters = []
        available_raters.insert(0, {"id": "admin", "name": "System Admin"})

    breadcrumbs, ancestor_keys = [], set()

    # In Exhibition Mode, don't show full physical breadcrumbs
    if not IS_EXHIBITION_MODE:
        curr = folder_key
        while curr and curr in folders:
            f_info = folders[curr]
            breadcrumbs.append({"key": curr, "display_name": f_info["display_name"]})
            ancestor_keys.add(curr)
            curr = f_info.get("parent")
        breadcrumbs.reverse()
    else:
        breadcrumbs.append({"key": "_root_", "display_name": "Exhibition Home"})

    # --- TEMPLATE SELECTION ---
    try:
        with get_db_connection() as conn_opts:
            users_rows = conn_opts.execute(
                "SELECT user_id, full_name FROM users WHERE is_active=1 AND username != 'admin'"
            ).fetchall()
            available_raters = [{"id": str(r["user_id"]), "name": r["full_name"]} for r in users_rows]
    except sqlite3.Error:
        available_raters = []
    available_raters.insert(0, {"id": "admin", "name": "System Admin"})
    template_name = "exhibition.html" if IS_EXHIBITION_MODE else "index.html"

    return render_template(
        template_name,
        files=view_files[:PAGE_SIZE],
        total_files=len(view_files),
        view_token=VIEW_SNAPSHOTS.put(_view_owner(), view_files),
        total_folder_files=total_folder_files,
        total_db_files=total_db_files,
        folders=folders,
        current_folder_key=folder_key,
        current_folder_info=current_folder_info,
        breadcrumbs=breadcrumbs,
        ancestor_keys=list(ancestor_keys),
        available_extensions=extensions,
        available_prefixes=prefixes,
        prefix_limit_reached=pfx_limit,
        selected_extensions=selected_exts,
        selected_prefixes=selected_prefixes,
        available_raters=available_raters,
        selected_raters=selected_raters,
        selected_rating_ranges=selected_rating_ranges,
        protected_folder_keys=list(PROTECTED_FOLDER_KEYS),
        show_favorites=request.args.get("favorites", "false").lower() == "true",
        generate_waveforms=GENERATE_WAVEFORMS,
        enable_ai_search=ENABLE_AI_SEARCH,
        enable_ai_dam=AI_CONFIG.enabled,
        is_ai_search=False,
        ai_query="",
        is_omniquery=is_omniquery,
        omniquery_sql=omniquery_sql,
        omniquery_dictionary=get_omniquery_dictionary(),
        is_global_search=is_global_search,
        active_filters_count=active_filters_count,
        current_scope=search_scope,
        is_recursive=is_recursive,
        server_dam_default=ENABLE_DAM_MODE,
        is_exhibition_mode=IS_EXHIBITION_MODE,
        blind_rating=is_effectively_blind(),
        global_blind_active=BLIND_RATING,  # Pass flag to template
        app_version=APP_VERSION,
        github_url=GITHUB_REPO_URL,
        update_available=UPDATE_AVAILABLE,
        remote_version=REMOTE_VERSION,
        ffmpeg_available=(FFPROBE_EXECUTABLE_PATH is not None),
        comfy_server_url=COMFYUI_SERVER_URL,
        stream_threshold=STREAM_THRESHOLD_BYTES,
        page_size_from_backend=PAGE_SIZE,
        force_login=FORCE_LOGIN,
        session_username=session.get("username", "Guest"),
        session_user_id=session.get("user_id"),
        session_role=session.get("role"),
        session_full_name=session.get("full_name"),
        has_notes=False,
        note_files=[],
    )


@app.route("/galleryout/upload", methods=["POST"])
@management_api_only
def upload_files():
    folder_key = request.form.get("folder_key")
    if not folder_key:
        return jsonify({"status": "error", "message": "No destination folder provided."}), 400
    folders = get_dynamic_folder_config()
    if folder_key not in folders:
        return jsonify({"status": "error", "message": "Destination folder not found."}), 404
    destination_path = folders[folder_key]["path"]
    if "files" not in request.files:
        return jsonify({"status": "error", "message": "No files were uploaded."}), 400
    uploaded_files, errors, success_count = request.files.getlist("files"), {}, 0
    ALLOWED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".gif",
        ".mp4",
        ".mov",
        ".webm",
        ".mkv",
        ".avi",
        ".m4v",
        ".wmv",
        ".flv",
        ".mts",
        ".ts",
        ".mp3",
        ".wav",
        ".ogg",
        ".flac",
        ".m4a",
        ".aac",
        ".json",
        ".txt",
        ".md",
    }
    for file in uploaded_files:
        if file and file.filename:
            filename = safe_media_filename(file.filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                errors[filename] = "Security Policy: File extension not allowed."
                continue
            try:
                # Never overwrite: an upload whose name collides with a file
                # already in the folder used to replace it silently and
                # still report success, destroying media the user never
                # chose to delete. The move path has always renamed instead
                # (_get_unique_filepath -> "name(1).png"); uploads do the
                # same, so the two ways a file can arrive in a folder agree.
                file.save(_get_unique_filepath(destination_path, filename))
                success_count += 1
            except Exception as e:
                _logger.debug("handled a failure in upload_files", exc_info=True)
                print(f"ERROR: Upload of {filename} failed: {e}")
                errors[filename] = e
    if success_count > 0:
        sync_folder_on_demand(destination_path)
    if errors:
        # It named which files failed and never said why: the reason was
        # collected here and then dropped.
        said = f"Successfully uploaded {success_count} files. The following files failed: {', '.join(errors.keys())}"
        why = next(
            (
                explain_a_refused_write(e, destination_path)
                for e in errors.values()
                if explain_a_refused_write(e, destination_path)
            ),
            None,
        )
        return jsonify({"status": "partial_success", "message": f"{said} {why}" if why else said}), 207
    return jsonify({"status": "success", "message": f"Successfully uploaded {success_count} files."})


# Global dictionary to track active background jobs
# Structure: { 'job_id': {'status': 'processing', 'current': 0, 'total': 100, 'folder_key': '...'} }
rescan_jobs = {}


def background_rescan_worker(job_id, files_to_process):
    """
    Background worker that updates a global job status so the UI can poll for progress.
    """
    if not files_to_process:
        rescan_jobs[job_id]["status"] = "done"
        return

    print(f"INFO: [Background] Job {job_id}: Rescanning {len(files_to_process)} files...")

    try:
        total = len(files_to_process)
        rescan_jobs[job_id]["total"] = total

        with get_db_connection() as conn:
            processed_count = 0
            results = []

            with scan_executor(total) as executor:
                futures = {executor.submit(process_single_file, path): path for path in files_to_process}

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)

                        processed_count += 1
                        # UPDATE PROGRESS
                        rescan_jobs[job_id]["current"] = processed_count

                    except Exception as e:
                        _logger.debug("handled a failure in background_rescan_worker", exc_info=True)
                        print(f"ERROR: Worker failed for a file: {e}")

            if results:
                file_rows_3, gen_rows_3, gen_deletes_3 = split_file_results(results)
                conn.executemany(
                    """
                    INSERT INTO files (
                        id, path, mtime, name, type, duration, dimensions,
                        has_workflow, size, last_scanned, workflow_files,
                        workflow_prompt, workflow_hash, prompt_hash, models_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        path = excluded.path,
                        name = excluded.name,
                        type = excluded.type,
                        duration = excluded.duration,
                        dimensions = excluded.dimensions,
                        has_workflow = excluded.has_workflow,
                        size = excluded.size,
                        last_scanned = excluded.last_scanned,
                        workflow_files = excluded.workflow_files,
                        workflow_prompt = excluded.workflow_prompt,
                        workflow_hash = excluded.workflow_hash,
                        prompt_hash = excluded.prompt_hash,
                        models_hash = excluded.models_hash,
                        hash_failed = CASE WHEN ABS(files.mtime - excluded.mtime) > 0.1
                                THEN 0 ELSE files.hash_failed END,
                        -- is_favorite is deliberately absent: it is the
                        -- person's own choice, not derived from the file
                        -- (see the note on the same statement in the full scan).
                        ai_caption = CASE WHEN ABS(files.mtime - excluded.mtime) > 0.1
                            THEN NULL ELSE files.ai_caption END,
                        ai_embedding = CASE WHEN ABS(files.mtime - excluded.mtime) > 0.1
                            THEN NULL ELSE files.ai_embedding END,
                        ai_last_scanned = CASE WHEN ABS(files.mtime - excluded.mtime) > 0.1
                            THEN 0 ELSE files.ai_last_scanned END,
                        mtime = excluded.mtime
                """,
                    file_rows_3,
                )
                upsert_generation_params(conn, gen_rows_3, gen_deletes_3)
                conn.commit()

        print(f"INFO: [Background] Job {job_id} finished.")
        rescan_jobs[job_id]["status"] = "done"

    except Exception as e:
        _logger.debug("handled a failure in background_rescan_worker", exc_info=True)
        print(f"CRITICAL ERROR in Background Rescan: {e}")
        rescan_jobs[job_id]["status"] = "error"
        rescan_jobs[job_id]["error"] = str(e)


@app.route("/galleryout/rescan_folder", methods=["POST"])
@management_api_only
def rescan_folder():
    data = request.json
    folder_key = data.get("folder_key")
    mode = data.get("mode", "all")

    if not folder_key:
        return jsonify({"status": "error", "message": "No folder provided."}), 400
    folders = get_dynamic_folder_config()
    if folder_key not in folders:
        return jsonify({"status": "error", "message": "Folder not found."}), 404

    folder_path = folders[folder_key]["path"]
    folder_name = folders[folder_key]["display_name"]

    try:
        files_to_process = []
        with get_db_connection() as conn:
            query = "SELECT path, last_scanned FROM files WHERE path LIKE ?"
            rows = conn.execute(query, (folder_path + os.sep + "%",)).fetchall()

            folder_path_norm = os.path.normpath(folder_path)
            files_in_folder = [
                {"path": row["path"], "last_scanned": row["last_scanned"]}
                for row in rows
                if os.path.normpath(os.path.dirname(row["path"])) == folder_path_norm
            ]

            current_time = time.time()
            if mode == "recent":
                cutoff_time = current_time - 3600
                files_to_process = [f["path"] for f in files_in_folder if (f["last_scanned"] or 0) < cutoff_time]
            else:
                files_to_process = [f["path"] for f in files_in_folder]

        if not files_to_process:
            return jsonify({"status": "success", "message": "No files needed rescanning.", "count": 0})

        # --- JOB CREATION ---
        job_id = str(uuid.uuid4())
        rescan_jobs[job_id] = {
            "status": "processing",
            "current": 0,
            "total": len(files_to_process),
            "folder_key": folder_key,
            "folder_name": folder_name,
        }

        # Start Worker with Job ID
        threading.Thread(target=background_rescan_worker, args=(job_id, files_to_process), daemon=True).start()

        return jsonify(
            {
                "status": "started",
                "job_id": job_id,
                "total": len(files_to_process),
                "message": "Background process started.",
            }
        )

    except Exception as e:
        _logger.debug("handled a failure in rescan_folder", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/check_rescan_status/<job_id>")
@management_api_only
def check_rescan_status(job_id):
    job = rescan_jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"})

    # Return copy of job data
    return jsonify(job)


@app.route("/galleryout/create_folder", methods=["POST"])
@management_api_only
def create_folder():
    data = request.json
    parent_key = data.get("parent_key", "_root_")

    raw_name = data.get("folder_name", "").strip()
    folder_name = re.sub(r'[\\/:*?"<>|]', "", raw_name)

    if not folder_name or folder_name in [".", ".."]:
        return jsonify({"status": "error", "message": "Invalid folder name provided."}), 400

    folders = get_dynamic_folder_config()
    if parent_key not in folders:
        return jsonify({"status": "error", "message": "Parent folder not found."}), 404
    parent_path = folders[parent_key]["path"]
    new_folder_path = os.path.join(parent_path, folder_name)
    try:
        os.makedirs(new_folder_path, exist_ok=False)
        sync_folder_on_demand(parent_path)
        return jsonify({"status": "success", "message": f'Folder "{folder_name}" created successfully.'})
    except FileExistsError:
        return jsonify({"status": "error", "message": "Folder already exists."}), 400
    except Exception as e:
        _logger.debug("handled a failure in create_folder", exc_info=True)
        said = explain_a_refused_write(e, new_folder_path)
        return jsonify({"status": "error", "message": said or str(e)}), 500


@app.route("/galleryout/mount_folder", methods=["POST"])
@management_api_only
def mount_folder():
    data = request.json
    link_name_raw = data.get("link_name", "").strip()
    target_path_raw = data.get("target_path", "").strip()

    # Sanitize name
    link_name = re.sub(r'[\\/:*?"<>|]', "", link_name_raw)

    if not link_name or not target_path_raw:
        return jsonify({"status": "error", "message": "Missing name or target path."}), 400

    # Security: Normalize target path
    target_path = os.path.normpath(target_path_raw)

    if not os.path.exists(target_path) or not os.path.isdir(target_path):
        return jsonify({"status": "error", "message": f"Target path does not exist: {target_path}"}), 404

    # Construct link path inside BASE_OUTPUT_PATH
    link_full_path = os.path.join(BASE_OUTPUT_PATH, link_name)

    if os.path.exists(link_full_path):
        return jsonify({"status": "error", "message": "A folder with this name already exists."}), 409

    try:
        if os.name == "nt":
            # --- WINDOWS ROBUST LOGIC ---

            # 1. Force Windows-style backslashes for cmd.exe compatibility
            # (Fixes issues with mixed slashes like Z:/path\folder)
            win_link = link_full_path.replace("/", "\\")
            win_target = target_path.replace("/", "\\")

            # Attempt 1: Junction (/J)
            # Ideal for local drives, does not require Admin usually.
            #
            # mklink is a cmd builtin, so cmd has to run it -- but the
            # arguments are passed as a list rather than pasted into one
            # string. They used to be interpolated into
            #   f'mklink /J "{win_link}" "{win_target}"'
            # and handed to shell=True, and win_link is built from a folder
            # name the caller supplies. A name containing a double quote
            # closed the quoting and everything after it ran as its own
            # command, as whoever the gallery runs as.
            cmd_junction = ["cmd", "/c", "mklink", "/J", win_link, win_target]

            # Linking a target on unreachable network storage can sit there;
            # the caller is a request waiting on the answer.
            result = subprocess.run(cmd_junction, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT, check=False)

            if result.returncode != 0:
                # Capture the actual error (e.g. "Local volumes are required...")
                err_junction = result.stderr.strip() or result.stdout.strip() or "Unknown Error"

                print(f"WARN: Junction failed ({err_junction}). Trying Symlink fallback...")

                # Attempt 2: Symbolic Link (/D)
                # Necessary for Network Shares, Virtual Drives, or Cross-Volume links.
                # NOTE: This usually requires Developer Mode enabled OR running ComfyUI as Administrator.
                cmd_symlink = ["cmd", "/c", "mklink", "/D", win_link, win_target]
                result_sym = subprocess.run(
                    cmd_symlink, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT, check=False
                )

                if result_sym.returncode != 0:
                    err_sym = result_sym.stderr.strip() or result_sym.stdout.strip()

                    # Create a detailed error message for the user
                    error_msg = (
                        f"Failed to create link.\n\n"
                        f"Attempt 1 (Junction): {err_junction}\n"
                        f"Attempt 2 (Symlink): {err_sym}\n\n"
                        f"TIP: If using Virtual Drives or Network Shares, try running ComfyUI as Administrator."
                    )
                    raise Exception(error_msg)

        else:
            # LINUX/MAC: Standard symlink
            os.symlink(target_path, link_full_path)

        # Register in DB
        with get_db_connection() as conn:
            norm_link_path = os.path.normpath(link_full_path).replace("\\", "/")
            conn.execute(
                "INSERT OR REPLACE INTO mounted_folders (path, target_source, created_at) VALUES (?, ?, ?)",
                (norm_link_path, target_path, time.time()),
            )
            conn.commit()

        # Refresh Cache
        get_dynamic_folder_config(force_refresh=True)

        return jsonify({"status": "success", "message": f'Successfully linked "{link_name}".'})

    except Exception as e:
        _logger.debug("handled a failure in mount_folder", exc_info=True)
        print(f"Mount Error: {e}")
        # Clean up if partially created
        if os.path.exists(link_full_path):
            with contextlib.suppress(OSError):
                os.rmdir(link_full_path)
            with contextlib.suppress(OSError):
                os.unlink(link_full_path)

        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/unmount_folder", methods=["POST"])
@management_api_only
def unmount_folder():
    data = request.json
    folder_key = data.get("folder_key")

    folders = get_dynamic_folder_config()
    if folder_key not in folders:
        return jsonify({"status": "error", "message": "Folder not found"}), 404

    folder_info = folders[folder_key]
    path_to_remove = folder_info["path"]

    # Security Check: Ensure it is actually in the mounted_folders table
    # This prevents users from deleting real folders via this API
    is_safe_mount = False
    with get_db_connection() as conn:
        norm_path = os.path.normpath(path_to_remove).replace("\\", "/")
        row = conn.execute("SELECT path FROM mounted_folders WHERE path = ?", (norm_path,)).fetchone()
        if row:
            is_safe_mount = True

    if not is_safe_mount:
        return jsonify({"status": "error", "message": "This folder is not a managed mount point. Cannot unmount."}), 403

    try:
        # Remove the Link (Not the content)
        if os.name == "nt":
            # On Windows, rmdir removes the Junction point safely without deleting content
            os.rmdir(path_to_remove)
        else:
            # On Linux/Mac, unlink removes the symlink
            os.unlink(path_to_remove)

        # Cleanup DB
        with get_db_connection() as conn:
            # 1. Remove from Mounts registry
            conn.execute("DELETE FROM mounted_folders WHERE path = ?", (norm_path,))

            # 2. Remove from AI Watch list (if present)
            conn.execute("DELETE FROM ai_watched_folders WHERE path = ?", (path_to_remove,))

            # 3. CRITICAL: Remove the file records associated with this path
            # from the Gallery DB -- the folder and everything inside it,
            # at any depth.
            cond, param = _descendant_filter("path", path_to_remove)
            conn.execute(f"DELETE FROM files WHERE {cond}", (param,))

            # 4. Also clean pending AI jobs for these files
            q_cond, q_param = _descendant_filter("file_path", path_to_remove)
            conn.execute(f"DELETE FROM ai_indexing_queue WHERE {q_cond}", (q_param,))

            conn.commit()

        get_dynamic_folder_config(force_refresh=True)
        return jsonify({"status": "success", "message": "Folder unmounted successfully."})

    except Exception as e:
        _logger.debug("handled a failure in unmount_folder", exc_info=True)
        print(f"Unmount Error: {e}")
        return jsonify({"status": "error", "message": f"Error unmounting: {e}"}), 500


@app.route("/galleryout/api/browse_filesystem", methods=["POST"])
@management_api_only
def browse_filesystem():
    data = request.json
    # Get path safely, handling None
    raw_path = data.get("path", "")
    if raw_path is None:
        raw_path = ""
    current_path = str(raw_path).strip()

    response_data = {"current_path": "", "parent_path": "", "folders": [], "error": None}

    # --- BLOCK 1: LIST DRIVES (WINDOWS) OR ROOT ---
    # If path is empty or 'Computer', list drives only and EXIT immediately.
    if not current_path or current_path == "Computer":
        response_data["current_path"] = "Computer"

        if os.name == "nt":
            drives = []
            # Iterate from A to Z
            for letter in string.ascii_uppercase:
                drive_path = f"{letter}:\\"
                try:
                    # Use isdir which is specific for drives
                    # Fault-tolerant check inside its own try/except block
                    if os.path.isdir(drive_path):
                        drives.append({"name": f"Drive ({letter}:)", "path": drive_path, "is_drive": True})
                except Exception:
                    # If a specific drive hangs, is not ready, or errors,
                    # skip it and continue to the next letter.
                    _logger.debug("handled a failure in browse_filesystem", exc_info=True)
                    continue

            response_data["folders"] = drives
            # Return JSON immediately. Do not execute further code.
            return jsonify(response_data)

        # On Linux/Mac, root is simply '/'
        current_path = "/"

    # --- BLOCK 2: SCAN FOLDER CONTENT ---
    # We reach here only if browsing inside a specific drive or folder
    try:
        current_path = os.path.normpath(current_path)
        items = []

        # Scandir is faster and allows skipping unreadable files individually
        with os.scandir(current_path) as it:
            for entry in it:
                try:
                    if entry.is_dir() and not entry.name.startswith("."):
                        items.append({"name": entry.name, "path": entry.path, "is_drive": False})
                except Exception:
                    # Skip individual unreadable folders without breaking the list
                    _logger.debug("handled a failure in browse_filesystem", exc_info=True)
                    continue

        items.sort(key=lambda x: x["name"].lower())
        response_data["folders"] = items
        response_data["current_path"] = current_path

        # Calculate "Up" button (Parent)
        parent = os.path.dirname(current_path)
        if parent == current_path:
            # If at drive root (e.g. C:\), parent is Computer list
            parent = "" if os.name == "nt" else ""

        response_data["parent_path"] = parent

    except Exception as e:
        # Catch errors accessing the specific folder (not the drives)
        _logger.debug("handled a failure in browse_filesystem", exc_info=True)
        response_data["error"] = f"Error accessing folder: {e!s}"

    return jsonify(response_data)


# --- ZIP BACKGROUND JOB MANAGEMENT ---
def zip_entry_names(rows):
    """One name per file, none of them repeated, in the order given.

    Every file went into the zip under its bare name, and a zip holding
    two entries with one name is not an error: both are written, and
    whoever opens it keeps whichever came last. ComfyUI numbers its
    output per folder, so ComfyUI_00001_.png exists once in every folder
    somebody has, and selecting across folders is the ordinary way to use
    this. Measured on three folders holding that name: three entries went
    in, one file came out, and the job reported ready.

    A name nothing else in the selection shares is kept exactly as it is,
    so downloading from a single folder is unchanged. A name that repeats
    is qualified on EVERY copy, not on the second onwards -- otherwise one
    arbitrary file keeps the bare name and there is no telling which
    folder that one came from. The qualifier is the containing folder,
    which is both the reason for the clash and the thing worth knowing;
    where that still is not enough, a number.
    """
    seen = {}
    for _path, name in rows:
        seen[name] = seen.get(name, 0) + 1
    repeated = {name for name, count in seen.items() if count > 1}

    taken = set()
    names = []
    for path, name in rows:
        stem, extension = os.path.splitext(name)
        candidates = []
        if name in repeated:
            folder = os.path.basename(os.path.dirname(path))
            if folder:
                candidates.append(f"{stem} ({folder}){extension}")
        else:
            candidates.append(name)

        chosen = next((c for c in candidates if c not in taken), None)
        if chosen is None:
            counter = 2
            while f"{stem} ({counter}){extension}" in taken:
                counter += 1
            chosen = f"{stem} ({counter}){extension}"
        taken.add(chosen)
        names.append(chosen)
    return names


zip_jobs = {}

# A prepared download is a second copy of everything that went into it,
# sitting in the gallery folder -- which is often a synced drive or the
# same disk the library is on. A day is long enough to click the link.
ZIP_RETENTION_SECONDS = 24 * 3600


def prune_zip_cache(now=None):
    """Delete prepared downloads past their day, and forget the jobs.

    This used to run only at the end of building a zip, so a gallery where
    somebody prepared one download and never prepared another kept that
    copy for good. It now also runs at startup, which is the one moment
    every gallery reaches.

    The job entries are pruned with the files. Nothing removed them
    before, so `zip_jobs` grew for the life of the process and, worse,
    outlived what it described: after the file was deleted the entry still
    said ready, check_zip_status still handed out a download link, and
    following it returned an HTML 404 page.

    Returns (files removed, jobs forgotten).
    """
    now = time.time() if now is None else now
    removed = 0
    try:
        for entry in os.listdir(ZIP_CACHE_DIR):
            path = os.path.join(ZIP_CACHE_DIR, entry)
            try:
                if os.path.isfile(path) and os.stat(path).st_mtime < now - ZIP_RETENTION_SECONDS:
                    os.remove(path)
                    removed += 1
            except OSError:
                pass  # being written, held open, or already gone
    except FileNotFoundError:
        pass  # no downloads have been prepared yet
    except OSError as exc:
        print(f"Zip Cleanup: could not read {ZIP_CACHE_DIR}: {exc}")

    forgotten = []
    for job_id, job in list(zip_jobs.items()):
        if job.get("status") == "processing":
            continue  # still being built; its file does not exist yet
        filename = job.get("filename")
        if filename:
            if not os.path.exists(os.path.join(ZIP_CACHE_DIR, filename)):
                forgotten.append(job_id)
        elif now - job.get("created", now) > ZIP_RETENTION_SECONDS:
            forgotten.append(job_id)  # a failure, with no file to check
    for job_id in forgotten:
        zip_jobs.pop(job_id, None)

    return removed, len(forgotten)


def background_zip_task(job_id, file_ids):
    try:
        if not os.path.exists(ZIP_CACHE_DIR):
            try:
                os.makedirs(ZIP_CACHE_DIR, exist_ok=True)
            except Exception as e:
                _logger.debug("handled a failure in background_zip_task", exc_info=True)
                print(f"ERROR: Could not create zip directory: {e}")
                zip_jobs[job_id] = {
                    "status": "error",
                    "created": time.time(),
                    "message": f"Server permission error: {e}",
                }
                return

        zip_filename = f"smartgallery_{job_id}.zip"
        zip_filepath = os.path.join(ZIP_CACHE_DIR, zip_filename)

        with get_db_connection() as conn:
            placeholders = ",".join(["?"] * len(file_ids))
            query = f"SELECT path, name FROM files WHERE id IN ({placeholders})"
            files_to_zip = conn.execute(query, file_ids).fetchall()

        if not files_to_zip:
            zip_jobs[job_id] = {"status": "error", "created": time.time(), "message": "No valid files found."}
            return

        entry_names = zip_entry_names([(row["path"], row["name"]) for row in files_to_zip])
        with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_row, file_name in zip(files_to_zip, entry_names, strict=False):
                file_path = file_row["path"]
                # Check the file esists
                if os.path.exists(file_path):
                    # Add file to zip
                    zf.write(file_path, file_name)

        # Job completed succesfully
        zip_jobs[job_id] = {"status": "ready", "filename": zip_filename, "created": time.time()}

        prune_zip_cache()

    except Exception as e:
        _logger.debug("handled a failure in background_zip_task", exc_info=True)
        print(f"Zip Error: {e}")
        zip_jobs[job_id] = {"status": "error", "message": str(e), "created": time.time()}


@app.route("/galleryout/prepare_batch_zip", methods=["POST"])
@management_api_only
def prepare_batch_zip():
    data = request.json
    file_ids = data.get("file_ids", [])
    if not file_ids:
        return jsonify({"status": "error", "message": "No files specified."}), 400

    job_id = str(uuid.uuid4())
    zip_jobs[job_id] = {"status": "processing", "created": time.time()}

    thread = threading.Thread(target=background_zip_task, args=(job_id, file_ids))
    thread.daemon = True
    thread.start()

    return jsonify({"status": "success", "job_id": job_id, "message": "Zip generation started."})


@app.route("/galleryout/check_zip_status/<job_id>")
@management_api_only
def check_zip_status(job_id):
    job = zip_jobs.get(job_id)
    if not job:
        return jsonify(
            {"status": "error", "message": "That download is no longer available. Please prepare it again."}
        ), 404
    response_data = job.copy()
    response_data.pop("created", None)
    if job["status"] == "ready" and "filename" in job:
        # Prepared downloads are cleared after a day, and the entry saying
        # so was never cleared with them: this answered ready and handed
        # out a link that returned an HTML 404. Ask the disk, not the note.
        if not os.path.exists(os.path.join(ZIP_CACHE_DIR, job["filename"])):
            zip_jobs.pop(job_id, None)
            return jsonify(
                {"status": "error", "message": "That download has expired and been cleared. Please prepare it again."}
            ), 404
        response_data["download_url"] = url_for("serve_zip_file", filename=job["filename"])

    return jsonify(response_data)


@app.route("/galleryout/serve_zip/<filename>")
@management_api_only
def serve_zip_file(filename):
    # A zip holds the original files, prompts and all. Only staff can make
    # one, so only staff may fetch one; the unguessable name was the whole
    # of the protection before.
    return send_from_directory(ZIP_CACHE_DIR, filename, as_attachment=True)


@app.route("/galleryout/rename_folder/<string:folder_key>", methods=["POST"])
@management_api_only
def rename_folder(folder_key):
    if folder_key in PROTECTED_FOLDER_KEYS:
        return jsonify({"status": "error", "message": "This folder cannot be renamed."}), 403

    raw_name = request.json.get("new_name", "").strip()
    # Separators and the rest of what Windows forbids come out here; the
    # test above this route pins that, because what matters is that a
    # rename cannot land outside the gallery.
    new_name = re.sub(r'[\\/:*?"<>|]', "", raw_name)
    # What was missing is the other kind: the characters Windows does not
    # refuse but silently removes. .strip() above catches a plain trailing
    # space; a trailing dot got through, and so did "holiday. ",
    # "holiday.." and "holiday  .". Windows then creates the folder
    # without the dot while the rows are written with it:
    #
    #     database says : holiday./ComfyUI_00001_.png
    #     disk actually : holiday/ComfyUI_00001_.png
    #
    # A file's id is its path, so those are different files. The next scan
    # walks the disk, finds a picture with no row and a row with no
    # picture, and deletes the row -- taking its ratings, comments and
    # album membership with it. The rename reported success.
    #
    # safe_media_filename is deliberately NOT used whole here: it takes a
    # basename, which is right for an upload arriving as a path and wrong
    # for someone typing a folder name, where it would turn "a/b" into "b"
    # and quietly drop half of what they typed.
    new_name = strip_what_windows_drops(new_name)

    if not new_name:
        return jsonify({"status": "error", "message": "Invalid name."}), 400

    folders = get_dynamic_folder_config()
    if folder_key not in folders:
        return jsonify({"status": "error", "message": "Folder not found."}), 400

    # 1. GET EXACT FOLDER PATH FROM CONFIG (Usually has forward slashes '/')
    old_folder_path = folders[folder_key]["path"]

    # 2. CONSTRUCT NEW FOLDER PATH (Preserving forward slashes structure)
    # We do NOT use os.path.join here for the folder part because it might force backslashes on Windows,
    # breaking consistency with get_dynamic_folder_config which enforces '/'.
    # We strip the last segment and append the new name.
    if "/" in old_folder_path:
        parent_dir = old_folder_path.rsplit("/", 1)[0]
        new_folder_path = f"{parent_dir}/{new_name}"
    else:
        # Fallback for systems strictly using backslash (unlikely given your logs, but safe)
        parent_dir = os.path.dirname(old_folder_path)
        new_folder_path = os.path.join(parent_dir, new_name)

    # Check existence (using normpath for OS safety check)
    if os.path.exists(os.path.normpath(new_folder_path)):
        return jsonify({"status": "error", "message": "A folder with this name already exists."}), 400

    try:
        with get_db_connection() as conn:
            all_files_cursor = conn.execute("SELECT id, path FROM files")

            update_data = []
            ids_to_clean_collisions = []

            # Prepare check. Compare on '/' alone: stored paths mix
            # separators, so a raw comparison misses everything nested.
            is_windows = os.name == "nt"
            std_old = _std_path(old_folder_path).rstrip("/")
            check_old = std_old.lower() if is_windows else std_old

            for row in all_files_cursor:
                current_path = row["path"]
                norm_curr = _std_path(current_path)
                check_curr = norm_curr.lower() if is_windows else norm_curr

                # The trailing separator is the whole check: without it,
                # renaming "box" also drags in every file under a sibling
                # called "box_archive".
                if not check_curr.startswith(check_old + "/"):
                    continue

                # 1. KEEP THE TAIL, NOT JUST THE FILENAME
                # A file one level deeper is stored as ".../box/2026-08-15\a.png";
                # rebuilding from the basename alone would move it to the top
                # of the renamed folder, where no such file exists.
                relative = norm_curr[len(check_old) + 1 :]
                subdirs, filename = relative.rsplit("/", 1) if "/" in relative else ("", relative)

                # 2. CONSTRUCT NEW PATH EXACTLY LIKE THE SCANNER DOES
                # Scanner logic: os.path.join(folder_path_from_config, filename),
                # where the config path keeps '/' separators. That produces
                # "C:/.../NewName/sub\filename.ext" on Windows.
                new_dir = f"{new_folder_path}/{subdirs}" if subdirs else new_folder_path
                new_file_path = os.path.join(new_dir, filename)

                # 3. GENERATE ID
                new_id = content_digest(new_file_path)

                update_data.append((new_id, new_file_path, row["id"]))
                ids_to_clean_collisions.append(new_id)

            # Cleanup Ghost records
            if ids_to_clean_collisions:
                placeholders = ",".join(["?"] * len(ids_to_clean_collisions))
                conn.execute(f"DELETE FROM files WHERE id IN ({placeholders})", ids_to_clean_collisions)

            # Atomic DB Update. The rows move first and the folder second, so
            # a failure on either side leaves both untouched: the physical
            # rename used to run first, and when the database update then hit
            # a foreign key it left the folder renamed on disk while every row
            # still pointed at the old path -- entries the next scan treats as
            # deleted, taking their ratings and comments with them.
            if update_data:
                _reassign_file_ids(conn, [(row[2], row[0]) for row in update_data])
                conn.executemany("UPDATE files SET id = ?, path = ? WHERE id = ?", update_data)

            # Physical Rename (Use normpath for OS call to be safe)
            os.rename(os.path.normpath(old_folder_path), os.path.normpath(new_folder_path))

            # Update Watch List
            watched_folders = conn.execute("SELECT path FROM ai_watched_folders").fetchall()
            for row in watched_folders:
                w_path = row["path"]
                w_std = _std_path(w_path)
                w_check = w_std.lower() if is_windows else w_std

                if w_check == check_old:
                    conn.execute("UPDATE ai_watched_folders SET path = ? WHERE path = ?", (new_folder_path, w_path))
                elif w_check.startswith(check_old + "/"):
                    # Subfolder logic: keep the tail below the renamed folder.
                    # The separator above matters here too -- otherwise a
                    # watched "box_archive" is rewritten when "box" is renamed.
                    if is_windows:
                        # Case insensitive replace is tricky, let's assume structure holds
                        # We reconstruct the tail
                        suffix = w_std[len(check_old) :]
                        new_w_path = new_folder_path + suffix
                        conn.execute("UPDATE ai_watched_folders SET path = ? WHERE path = ?", (new_w_path, w_path))
                    else:
                        new_w_path = w_path.replace(old_folder_path, new_folder_path, 1)
                        conn.execute("UPDATE ai_watched_folders SET path = ? WHERE path = ?", (new_w_path, w_path))

            conn.commit()

        get_dynamic_folder_config(force_refresh=True)
        return jsonify({"status": "success", "message": "Folder renamed."})

    except Exception as e:
        _logger.debug("handled a failure in rename_folder", exc_info=True)
        print(f"Rename Error: {e}")
        said = explain_a_refused_write(e)
        return jsonify({"status": "error", "message": said or f"Error: {e}"}), 500


@app.route("/galleryout/delete_folder/<string:folder_key>", methods=["POST"])
@management_api_only
def delete_folder(folder_key):
    if folder_key in PROTECTED_FOLDER_KEYS:
        return jsonify({"status": "error", "message": "This folder cannot be deleted."}), 403
    folders = get_dynamic_folder_config()
    if folder_key not in folders:
        return jsonify({"status": "error", "message": "Folder not found."}), 404
    try:
        folder_path = folders[folder_key]["path"]
        with get_db_connection() as conn:
            # 1. Remove files from DB, including everything in subfolders
            cond, param = _descendant_filter("path", folder_path)
            conn.execute(f"DELETE FROM files WHERE {cond}", (param,))

            # 2. AI WATCHED FOLDERS CLEANUP (Logic added)
            # Remove the folder itself from watched list
            conn.execute("DELETE FROM ai_watched_folders WHERE path = ?", (folder_path,))
            # Remove any subfolders that might be in the watched list
            w_cond, w_param = _descendant_filter("path", folder_path)
            conn.execute(f"DELETE FROM ai_watched_folders WHERE {w_cond}", (w_param,))

            conn.commit()

        # 3. Physical deletion (Safe for Symlinks/Junctions). A link is only
        # unlinked -- that destroys nothing, so it never goes to the trash;
        # a real folder full of media is relocated when DELETE_TO is set
        # (safe_delete_tree), because the safety net has to cover the most
        # destructive action, not just the least.
        if os.path.islink(folder_path):
            os.unlink(folder_path)
        elif os.name == "nt" and os.path.isdir(folder_path) and not os.path.exists(os.path.join(folder_path, "..")):
            # Fallback for Windows Junctions acting weirdly with islink
            try:
                os.rmdir(folder_path)
            except OSError:
                safe_delete_tree(folder_path)
        else:
            try:
                # Extra check: if it's a junction, rmtree throws an error in some python versions.
                # Let's try rmdir first for junctions, fallback to rmtree for real folders.
                os.rmdir(folder_path)
            except OSError:
                safe_delete_tree(folder_path)

        get_dynamic_folder_config(force_refresh=True)
        return jsonify({"status": "success", "message": "Folder deleted/unlinked."})
    except Exception as e:
        _logger.debug("handled a failure in delete_folder", exc_info=True)
        print(f"Delete Folder Error: {e}")
        said = explain_a_refused_write(e)
        return jsonify({"status": "error", "message": said or f"Error: {e}"}), 500


@app.route("/galleryout/api/current_view_ids")
def get_current_view_ids():
    """Returns all file IDs in the caller's view snapshot (by view_token)."""
    if (IS_EXHIBITION_MODE or FORCE_LOGIN) and not session.get("user_id"):
        return jsonify({"status": "error", "message": "Authentication required"}), 401
    snapshot = VIEW_SNAPSHOTS.get(request.args.get("view_token", ""), _view_owner())
    if snapshot is None:
        return jsonify({"status": "error", "message": "View expired. Reload the page.", "stale": True}), 410
    return jsonify({"status": "success", "ids": [f["id"] for f in snapshot]})


@app.route("/galleryout/load_more")
def load_more():
    if (IS_EXHIBITION_MODE or FORCE_LOGIN) and not session.get("user_id"):
        return jsonify({"files": []}), 401
    snapshot = VIEW_SNAPSHOTS.get(request.args.get("view_token", ""), _view_owner())
    if snapshot is None:
        return jsonify({"files": [], "stale": True})
    offset = request.args.get("offset", 0, type=int)
    if offset >= len(snapshot):
        return jsonify(files=[])
    # Same rows as the album listing, so the same audience rule applies.
    return jsonify(files=redact_file_listing(snapshot[offset : offset + PAGE_SIZE]))


def get_file_info_from_db(file_id, column="*"):
    with get_db_connection() as conn:
        row = conn.execute(f"SELECT {column} FROM files WHERE id = ?", (file_id,)).fetchone()
    # Said in words, because the default text for a 404 is "The requested
    # URL was not found on the server", which is untrue here: the address
    # is a real one, the picture behind it is not.
    if not row:
        abort(404, description="That file is not in the gallery.")
    return dict(row) if column == "*" else row[0]


def explain_a_refused_write(error, path=None):
    """Say what a failed write means, or hand back None to say nothing.

    Being unable to write to the pictures folder is one of the ordinary
    ways this program is set up wrong, not a rare accident: the folder
    mounted read-only, which people do on purpose; a container running as
    a different user than the one that owns the files, which is why the
    supplied compose file carries WANTED_UID and FORCE_CHOWN at all; a
    share that has gone away; a file another program still has open.

    All of it arrived as `Error: [Errno 13] Permission denied:
    'C:/.../output\\picture.png'`, which says nothing to somebody who has
    not met errno before, and the same for every one of the six actions
    that touch the folder.

    Only permission and read-only failures are put into words. Anything
    else is left exactly as it was rather than dressed up as something it
    might not be.
    """
    if not isinstance(error, OSError):
        return None
    if error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
        return None
    where = path or getattr(error, "filename", None)
    folder = os.path.dirname(str(where)) if where else None
    return (
        ("The gallery is not allowed to change %s." % (("anything in " + folder) if folder else "that file"))
        + " The folder may be mounted read-only, or owned by a different"
        " user than the one the gallery runs as. Nothing was changed."
    )


def answer_an_abort_readably(stop):
    """Hand back the refusal a route made about itself, not a fault.

    A route that wraps its body in `except Exception` catches its own
    abort() -- the 404 raised for a picture that is not there arrives as
    an exception like any other and was answered as 500. Measured across
    the 24 addresses that take a file id, asking for one that does not
    exist: twelve said 404 and three said the server had failed.

    That difference is the whole meaning of the answer. The page cannot
    tell "this picture is gone" from "the gallery is broken", so it
    reports a fault for something perfectly ordinary -- a stale tab, a
    bookmarked picture since deleted -- and a real fault looks the same
    as that.

    Flask's docs say this outright (errorhandling.rst, "Generic Exception
    Handlers"): a handler that catches everything must be written so it
    does not lose the HTTP error.

    In JSON, because every caller of these addresses reads JSON.
    """
    return jsonify({"status": "error", "message": stop.description or stop.name}), stop.code


# Path separators, control characters, and the characters Windows forbids
# in a name. Everything else -- including every non-Latin script -- is a
# legitimate part of a filename.
_UNSAFE_NAME_CHARS = re.compile(r'[\x00-\x1f<>:"/\\|?*]')
_WINDOWS_DEVICE_NAMES = (
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
)


def strip_what_windows_drops(name):
    """Remove the trailing dots and spaces Windows silently removes itself.

    These are the characters that do not FAIL, which is what makes them
    worth a function. Windows creates `holiday.` as `holiday`, so anything
    that records the name it asked for ends up describing a path that is
    not there -- and here a path is a file's identity, so the next scan
    sees a picture with no row and a row with no picture, and deletes the
    row with its ratings and comments.

    Only trailing. A leading dot is how a hidden file is spelled.

    os.path.exists will not show you this: Windows resolves `holiday.` to
    `holiday` when asked whether it exists, so a lookup agrees and only a
    directory listing disagrees. Scans list.
    """
    return str(name or "").rstrip(". ")


def safe_media_filename(filename, fallback="upload"):
    """Make a filename safe to write while keeping the name the user gave it.

    secure_filename() transliterates to ASCII and drops whatever will not
    convert, which is fine for a token and wrong for someone's media: it
    turns "测试.png" into "png" -- no extension left, so the upload is then
    refused as an unsupported type -- and quietly shortens "한글_0001_.png"
    to "0001_.png". Chinese, Korean, Japanese, Cyrillic, Greek, Hebrew and
    Arabic names all lose part or all of themselves.

    What is actually dangerous is removed instead: any directory part, the
    characters Windows forbids, control characters, the trailing dots and
    spaces Windows silently strips, and the reserved device names (CON,
    COM1, ...) which are not usable as files even with an extension.
    """
    raw = str(filename or "").replace("\\", "/")
    name = os.path.basename(raw.rstrip("/"))
    name = _UNSAFE_NAME_CHARS.sub("_", name).strip()
    # Only TRAILING dots and spaces: Windows drops them silently, so a name
    # ending in one never round-trips. A leading dot is left alone -- it is
    # how a hidden file is spelled, and stripping it would recreate the very
    # bug this function exists to fix.
    name = strip_what_windows_drops(name)
    # Whatever is left must be more than dots: "." and ".." are directories.
    if not name.strip("."):
        return fallback

    stem, ext = os.path.splitext(name)
    if stem.upper() in _WINDOWS_DEVICE_NAMES:
        stem = f"_{stem}"
    return f"{stem}{ext}"


def _get_unique_filepath(destination_folder, filename):
    """
    Generates a unique filepath using the NATIVE OS separator.
    This ensures that the path matches exactly what the Scanner generates,
    preventing duplicate records in the database.
    """
    base, ext = os.path.splitext(filename)
    counter = 1

    # Use standard os.path.join.
    # On Windows with base path "C:/A", it produces "C:/A\file.txt" (Matches your DB).
    # On Linux, it produces "C:/A/file.txt" (Matches Linux DB).
    full_path = os.path.join(destination_folder, filename)

    while os.path.exists(full_path):
        new_filename = f"{base}({counter}){ext}"
        full_path = os.path.join(destination_folder, new_filename)
        counter += 1

    return full_path


@app.route("/galleryout/move_batch", methods=["POST"])
@management_api_only
def move_batch():
    data = request.json
    file_ids = data.get("file_ids", [])
    dest_key = data.get("destination_folder")

    folders = get_dynamic_folder_config()

    if not all([file_ids, dest_key, dest_key in folders]):
        return jsonify({"status": "error", "message": "Invalid data provided."}), 400

    moved_count, renamed_count, skipped_count = 0, 0, 0
    failed_files = []

    # Get destination path from config
    dest_path_raw = folders[dest_key]["path"]

    with get_db_connection() as conn:
        for file_id in file_ids:
            source_path = None
            try:
                # 1. Fetch Source Data + AI Metadata
                query_fetch = """
                    SELECT
                        path, name, size, has_workflow, is_favorite, type, duration, dimensions,
                        ai_last_scanned, ai_caption, ai_embedding, ai_error, workflow_files, workflow_prompt,
                        workflow_hash, prompt_hash, models_hash
                    FROM files WHERE id = ?
                """
                file_info_row = conn.execute(query_fetch, (file_id,)).fetchone()

                if not file_info_row:
                    failed_files.append(f"ID {file_id} not found in DB")
                    continue

                file_info = dict(file_info_row)

                source_path = file_info["path"]
                source_filename = file_info["name"]

                # Metadata Pack
                meta = {
                    "size": file_info["size"],
                    "has_workflow": file_info["has_workflow"],
                    "is_favorite": file_info["is_favorite"],
                    "type": file_info["type"],
                    "duration": file_info["duration"],
                    "dimensions": file_info["dimensions"],
                    "ai_last_scanned": file_info["ai_last_scanned"],
                    "ai_caption": file_info["ai_caption"],
                    "ai_embedding": file_info["ai_embedding"],
                    "ai_error": file_info["ai_error"],
                    "workflow_files": file_info["workflow_files"],
                    "workflow_prompt": file_info["workflow_prompt"],
                    "workflow_hash": file_info.get("workflow_hash", ""),
                    "prompt_hash": file_info.get("prompt_hash", ""),
                    "models_hash": file_info.get("models_hash", ""),
                }

                # Check Source vs Dest (OS Agnostic comparison)
                source_dir_norm = os.path.normpath(os.path.dirname(source_path))
                dest_dir_norm = os.path.normpath(dest_path_raw)
                is_same_folder = (
                    (source_dir_norm.lower() == dest_dir_norm.lower())
                    if os.name == "nt"
                    else (source_dir_norm == dest_dir_norm)
                )

                if is_same_folder:
                    skipped_count += 1
                    continue

                if not os.path.exists(source_path):
                    failed_files.append(f"{source_filename} (not found on disk)")
                    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
                    continue

                # 2. Calculate unique path NATIVELY (No separator forcing)
                # This guarantees the path string matches what the Scanner will see.
                final_dest_path = _get_unique_filepath(dest_path_raw, source_filename)
                final_filename = os.path.basename(final_dest_path)

                if final_filename != source_filename:
                    renamed_count += 1

                # 3. Calculate New ID based on the NATIVE path
                new_id = content_digest(final_dest_path)

                # 4. DB Update / Merge Logic, then the file itself.
                # A file's id is derived from its path, so moving one changes
                # it and everything keyed to it has to come along. The rows go
                # first and the file second, inside a savepoint, so a file that
                # fails changes neither -- moving the file first meant a failure
                # here left it moved on disk with its row still pointing at the
                # old folder, and this loop reports that as a partial success
                # rather than an error, so the next scan quietly deleted the row
                # and the rating with it.
                conn.execute("SAVEPOINT move_one")
                existing_target = conn.execute("SELECT id FROM files WHERE id = ?", (new_id,)).fetchone()

                if existing_target:
                    # MERGE: Target exists (e.g. ghost record). Overwrite with source metadata.
                    query_merge = """
                        UPDATE files
                        SET path = ?, name = ?, mtime = ?,
                            size = ?, has_workflow = ?, is_favorite = ?,
                            type = ?, duration = ?, dimensions = ?,
                            ai_last_scanned = ?, ai_caption = ?, ai_embedding = ?, ai_error = ?,
                            workflow_files = ?, workflow_prompt = ?,
                            workflow_hash = ?, prompt_hash = ?, models_hash = ?
                        WHERE id = ?
                    """
                    conn.execute(
                        query_merge,
                        (
                            final_dest_path,
                            final_filename,
                            time.time(),
                            meta["size"],
                            meta["has_workflow"],
                            meta["is_favorite"],
                            meta["type"],
                            meta["duration"],
                            meta["dimensions"],
                            meta["ai_last_scanned"],
                            meta["ai_caption"],
                            meta["ai_embedding"],
                            meta["ai_error"],
                            meta["workflow_files"],
                            meta["workflow_prompt"],
                            meta["workflow_hash"],
                            meta["prompt_hash"],
                            meta["models_hash"],
                            new_id,
                        ),
                    )
                    # The ratings and comments belong to the file being moved,
                    # so they move onto the surviving row before the old one is
                    # deleted -- the delete cascades them away otherwise.
                    _reassign_file_ids(conn, [(file_id, new_id)])
                    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
                else:
                    # STANDARD: Update existing record path/name.
                    _reassign_file_ids(conn, [(file_id, new_id)])
                    conn.execute(
                        "UPDATE files SET id = ?, path = ?, name = ? WHERE id = ?",
                        (new_id, final_dest_path, final_filename, file_id),
                    )

                shutil.move(source_path, final_dest_path)
                conn.execute("RELEASE move_one")
                moved_count += 1

            except Exception as e:
                _logger.debug("handled a failure in move_batch", exc_info=True)
                try:
                    conn.execute("ROLLBACK TO move_one")
                    conn.execute("RELEASE move_one")
                except sqlite3.Error:
                    pass  # the savepoint was never opened for this file
                filename_for_error = os.path.basename(source_path) if source_path else f"ID {file_id}"
                failed_files.append(filename_for_error)
                print(f"ERROR: Failed to move file {filename_for_error}. Reason: {e}")
                continue
        conn.commit()

    message = f"Successfully moved {moved_count} file(s)."
    if skipped_count > 0:
        message += f" {skipped_count} skipped (same folder)."
    if renamed_count > 0:
        message += f" {renamed_count} renamed."
    if failed_files:
        message += f" Failed: {len(failed_files)}."

    status = "success"
    if failed_files or (skipped_count > 0 and moved_count == 0):
        status = "partial_success"

    return jsonify({"status": status, "message": message})


@app.route("/galleryout/copy_batch", methods=["POST"])
@management_api_only
def copy_batch():
    data = request.json
    file_ids = data.get("file_ids", [])
    dest_key = data.get("destination_folder")
    keep_favorites = data.get("keep_favorites", False)

    folders = get_dynamic_folder_config()

    if not all([file_ids, dest_key, dest_key in folders]):
        return jsonify({"status": "error", "message": "Invalid data provided."}), 400

    dest_path_raw = folders[dest_key]["path"]
    copied_count = 0
    failed_files = []

    with get_db_connection() as conn:
        for file_id in file_ids:
            try:
                # 1. Fetch Source info
                file_info_row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
                if not file_info_row:
                    continue
                file_info = dict(file_info_row)

                source_path = file_info["path"]
                source_filename = file_info["name"]

                if not os.path.exists(source_path):
                    failed_files.append(f"{source_filename} (not found)")
                    continue

                # 2. Determine Destination Path (Auto-rename logic)
                # Helper function _get_unique_filepath handles (1), (2) etc.
                final_dest_path = _get_unique_filepath(dest_path_raw, source_filename)
                final_filename = os.path.basename(final_dest_path)

                # 3. Physical Copy (Metadata preserved via copy2)
                shutil.copy2(source_path, final_dest_path)

                # 4. Create DB Record
                new_id = content_digest(final_dest_path)
                new_mtime = time.time()  # New file gets new import time

                # Logic for Favorites
                is_fav = file_info["is_favorite"] if keep_favorites else 0

                # Insert Copy
                # We copy AI data too because the image content is identical!
                conn.execute(
                    """
                    INSERT INTO files (
                        id, path, mtime, name, type, duration, dimensions, has_workflow,
                        size, is_favorite, last_scanned, workflow_files, workflow_prompt,
                        ai_last_scanned, ai_caption, ai_embedding, ai_error,
                        workflow_hash, prompt_hash, models_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        new_id,
                        final_dest_path,
                        new_mtime,
                        final_filename,
                        file_info["type"],
                        file_info["duration"],
                        file_info["dimensions"],
                        file_info["has_workflow"],
                        file_info["size"],
                        is_fav,  # User Choice
                        file_info["last_scanned"],
                        file_info["workflow_files"],
                        file_info["workflow_prompt"],
                        file_info["ai_last_scanned"],
                        file_info["ai_caption"],
                        file_info["ai_embedding"],
                        file_info["ai_error"],
                        file_info.get("workflow_hash", ""),
                        file_info.get("prompt_hash", ""),
                        file_info.get("models_hash", ""),
                    ),
                )

                copied_count += 1

            except Exception as e:
                _logger.debug("handled a failure in copy_batch", exc_info=True)
                print(f"COPY ERROR: {e}")
                failed_files.append(source_filename)

        conn.commit()

    msg = f"Successfully copied {copied_count} files."
    status = "success"
    if failed_files:
        status = "partial_success"
        msg += f" Failed: {len(failed_files)}"

    return jsonify({"status": status, "message": msg})


@app.route("/galleryout/delete_batch", methods=["POST"])
@management_api_only
def delete_batch():
    try:
        # Preveniamo il crash gestendo tutto in un blocco try/except
        data = request.json
        file_ids = data.get("file_ids", [])

        if not file_ids:
            return jsonify({"status": "error", "message": "No files selected."}), 400

        deleted_count = 0
        failed_files = []
        ids_to_remove_from_db = []

        with get_db_connection() as conn:
            # 1. Generazione corretta e sicura dei placeholder SQL (?,?,?)
            # Usiamo una lista esplicita per evitare errori di sintassi python
            placeholders = ",".join(["?"] * len(file_ids))

            # Selezioniamo i file per verificare i percorsi
            query_select = f"SELECT id, path FROM files WHERE id IN ({placeholders})"
            files_to_delete = conn.execute(query_select, file_ids).fetchall()

            for row in files_to_delete:
                file_path = row["path"]
                file_id = row["id"]

                try:
                    # Cancellazione Fisica (o spostamento nel cestino)
                    if os.path.exists(file_path):
                        safe_delete_file(file_path)

                    # Se l'operazione su disco riesce (o il file non c'era già più),
                    # segniamo l'ID per la rimozione dal DB
                    ids_to_remove_from_db.append(file_id)
                    deleted_count += 1

                except Exception as e:
                    # Se fallisce la cancellazione fisica di un file, lo annotiamo ma continuiamo
                    _logger.debug("handled a failure in delete_batch", exc_info=True)
                    print(f"ERROR: Could not delete {file_path}: {e}")
                    failed_files.append(os.path.basename(file_path))

            # 2. Pulizia Database (Massiva)
            if ids_to_remove_from_db:
                # Generiamo nuovi placeholder solo per gli ID effettivamente cancellati
                db_placeholders = ",".join(["?"] * len(ids_to_remove_from_db))
                query_delete = f"DELETE FROM files WHERE id IN ({db_placeholders})"
                conn.execute(query_delete, ids_to_remove_from_db)
                conn.commit()

        # Costruzione messaggio finale
        action = "moved to trash" if DELETE_TO else "deleted"
        message = f"Successfully {action} {deleted_count} files."

        status = "success"
        if failed_files:
            message += f" Failed to delete {len(failed_files)} files."
            status = "partial_success"

        return jsonify({"status": status, "message": message})

    except Exception as e:
        # THIS solves the "doctype is not json" issue:
        # If there is a critical error, return an error JSON instead of a broken HTML page.
        _logger.debug("handled a failure in delete_batch", exc_info=True)
        print(f"CRITICAL ERROR in delete_batch: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/favorite_batch", methods=["POST"])
@management_api_only
def favorite_batch():
    data = request.json
    file_ids, status = data.get("file_ids", []), data.get("status", False)
    if not file_ids:
        return jsonify({"status": "error", "message": "No files selected"}), 400
    with get_db_connection() as conn:
        placeholders = ",".join("?" * len(file_ids))
        conn.execute(f"UPDATE files SET is_favorite = ? WHERE id IN ({placeholders})", [1 if status else 0] + file_ids)
        conn.commit()
    return jsonify({"status": "success", "message": f"Updated favorites for {len(file_ids)} files."})


@app.route("/galleryout/toggle_favorite/<string:file_id>", methods=["POST"])
@management_api_only
def toggle_favorite(file_id):
    with get_db_connection() as conn:
        current = conn.execute("SELECT is_favorite FROM files WHERE id = ?", (file_id,)).fetchone()
        if not current:
            abort(404, description="That file is not in the gallery.")
        new_status = 1 - current["is_favorite"]
        conn.execute("UPDATE files SET is_favorite = ? WHERE id = ?", (new_status, file_id))
        conn.commit()
        return jsonify({"status": "success", "is_favorite": bool(new_status)})


# --- FIX: ROBUST DELETE ROUTE ---
@app.route("/galleryout/delete/<string:file_id>", methods=["POST"])
@management_api_only
def delete_file(file_id):
    with get_db_connection() as conn:
        file_info = conn.execute("SELECT path FROM files WHERE id = ?", (file_id,)).fetchone()
        if not file_info:
            return jsonify({"status": "success", "message": "File already deleted from database."})

        filepath = file_info["path"]

        try:
            if os.path.exists(filepath):
                safe_delete_file(filepath)
            # If file doesn't exist on disk, we still proceed to remove the DB entry, which is the desired state.
        except OSError as e:
            # A real OS error occurred (e.g., permissions).
            print(f"ERROR: Could not delete file {filepath} from disk: {e}")
            said = explain_a_refused_write(e, filepath)
            return jsonify({"status": "error", "message": said or f"Could not delete file from disk: {e}"}), 500

        # Whether the file was deleted now or was already gone, we clean up the DB.
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        conn.commit()
        action = "moved to trash" if DELETE_TO else "deleted"
        return jsonify({"status": "success", "message": f"File {action} successfully."})


# --- RENAME FILE ---
@app.route("/galleryout/rename_file/<string:file_id>", methods=["POST"])
@management_api_only
def rename_file(file_id):
    data = request.json
    new_name = data.get("new_name", "").strip()

    if not new_name or len(new_name) > 250:
        return jsonify({"status": "error", "message": "Invalid filename."}), 400
    if re.search(r'[\\/:"*?<>|]', new_name):
        return jsonify({"status": "error", "message": "Invalid characters."}), 400

    try:
        with get_db_connection() as conn:
            # 1. Fetch All Metadata
            query_fetch = """
                SELECT
                    path, name, size, has_workflow, is_favorite, type, duration, dimensions,
                    ai_last_scanned, ai_caption, ai_embedding, ai_error, workflow_files, workflow_prompt,
                    workflow_hash, prompt_hash, models_hash
                FROM files WHERE id = ?
            """
            file_info_row = conn.execute(query_fetch, (file_id,)).fetchone()

            if not file_info_row:
                return jsonify({"status": "error", "message": "File not found."}), 404

            # A sqlite3.Row has no .get(), and the metadata pack below uses
            # it -- so every rename raised AttributeError and returned 500.
            # move_batch/copy_batch already convert here for the same reason.
            file_info = dict(file_info_row)

            old_path = file_info["path"]
            old_name = file_info["name"]

            # Metadata Pack
            meta = {
                "size": file_info["size"],
                "has_workflow": file_info["has_workflow"],
                "is_favorite": file_info["is_favorite"],
                "type": file_info["type"],
                "duration": file_info["duration"],
                "dimensions": file_info["dimensions"],
                "ai_last_scanned": file_info["ai_last_scanned"],
                "ai_caption": file_info["ai_caption"],
                "ai_embedding": file_info["ai_embedding"],
                "ai_error": file_info["ai_error"],
                "workflow_files": file_info["workflow_files"],
                "workflow_prompt": file_info["workflow_prompt"],
                "workflow_hash": file_info.get("workflow_hash", ""),
                "prompt_hash": file_info.get("prompt_hash", ""),
                "models_hash": file_info.get("models_hash", ""),
            }

            # Extension logic
            _, old_ext = os.path.splitext(old_name)
            _new_name_base, new_ext = os.path.splitext(new_name)
            final_new_name = new_name if new_ext else new_name + old_ext

            if final_new_name == old_name:
                return jsonify({"status": "error", "message": "Name unchanged."}), 400

            # 2. Construct Path NATIVELY using os.path.join
            # This respects the OS separator (Mixed on Win, Forward on Linux)
            # ensuring the Hash ID matches future Scans.
            dir_name = os.path.dirname(old_path)
            new_path = os.path.join(dir_name, final_new_name)

            if os.path.exists(new_path):
                return jsonify({"status": "error", "message": f'File "{final_new_name}" already exists.'}), 409

            new_id = content_digest(new_path)
            existing_db = conn.execute("SELECT id FROM files WHERE id = ?", (new_id,)).fetchone()

            if existing_db:
                # MERGE SCENARIO
                query_merge = """
                    UPDATE files
                    SET path = ?, name = ?, mtime = ?,
                        size = ?, has_workflow = ?, is_favorite = ?,
                        type = ?, duration = ?, dimensions = ?,
                        ai_last_scanned = ?, ai_caption = ?, ai_embedding = ?, ai_error = ?,
                        workflow_files = ?, workflow_prompt = ?,
                        workflow_hash = ?, prompt_hash = ?, models_hash = ?
                    WHERE id = ?
                """
                conn.execute(
                    query_merge,
                    (
                        new_path,
                        final_new_name,
                        time.time(),
                        meta["size"],
                        meta["has_workflow"],
                        meta["is_favorite"],
                        meta["type"],
                        meta["duration"],
                        meta["dimensions"],
                        meta["ai_last_scanned"],
                        meta["ai_caption"],
                        meta["ai_embedding"],
                        meta["ai_error"],
                        meta["workflow_files"],
                        meta["workflow_prompt"],
                        meta["workflow_hash"],
                        meta["prompt_hash"],
                        meta["models_hash"],
                        new_id,
                    ),
                )
                # The ratings, comments and album membership belong to the
                # file being renamed, so they move onto the row that survives
                # -- deleting the old row cascades them away otherwise.
                _reassign_file_ids(conn, [(file_id, new_id)])
                conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
            else:
                # STANDARD SCENARIO
                _reassign_file_ids(conn, [(file_id, new_id)])
                conn.execute(
                    "UPDATE files SET id = ?, path = ?, name = ? WHERE id = ?",
                    (new_id, new_path, final_new_name, file_id),
                )

            # The file moves last: if this raises, the rows roll back with it
            # rather than leaving the database describing a file that is no
            # longer at that path.
            os.rename(old_path, new_path)

            conn.commit()

            return jsonify(
                {"status": "success", "message": "File renamed.", "new_name": final_new_name, "new_id": new_id}
            )

    except Exception as e:
        _logger.debug("handled a failure in rename_file", exc_info=True)
        print(f"ERROR: Rename failed: {e}")
        said = explain_a_refused_write(e)
        return jsonify({"status": "error", "message": said or f"Error: {e}"}), 500


@app.route("/galleryout/file_clean/<string:file_id>")
def serve_cleaned_file(file_id):
    if not is_file_accessible(file_id):
        abort(403, description="Access Denied.")
    """
    Serves the cleaned file from cache.
    If the cached file is corrupted (0 bytes) or the client specifically
    requests a retry, it deletes the cache and regenerates it.
    """
    # Check if the frontend is forcing a regeneration
    force_retry = request.args.get("retry") == "true"

    info = get_file_info_from_db(file_id)
    filepath, mtime, file_type = info["path"], info["mtime"], info["type"]

    # Calculate unique cache filename
    cache_hash = content_digest(filepath + str(mtime))
    _, ext = os.path.splitext(filepath)
    clean_filename = f"{cache_hash}{ext}"
    clean_path = os.path.join(CLEAN_CACHE_DIR, clean_filename)

    # --- AUTO-HEALING LOGIC ---
    if os.path.exists(clean_path):
        # 1. Check if file is empty (often happens after a crash)
        # 2. Or if the client explicitly asked for a retry due to loading errors
        if os.path.getsize(clean_path) == 0 or force_retry:
            print(f"DEBUG: Cache corrupted or retry requested for {clean_filename}. Regenerating...")
            try:
                os.remove(clean_path)
            except Exception as e:
                _logger.debug("handled a failure in serve_cleaned_file", exc_info=True)
                print(f"DEBUG: Could not remove corrupted cache: {e}")

    # Generate if not exists (either new or just deleted above)
    if not os.path.exists(clean_path):
        print(f"ACTION: Generating clean version for: {info['name']}")
        os.makedirs(CLEAN_CACHE_DIR, exist_ok=True)
        success = strip_media_metadata(filepath, clean_path, file_type)
        if not success:
            if should_strip_metadata():
                # Refuse. Serving the original here hands the visitor exactly
                # the prompt and workflow this mode exists to withhold, and the
                # commonest way to get here is not exotic: video and audio are
                # stripped with ffmpeg, and that branch is only entered when an
                # ffmpeg was found at all. The gallery kept working and quietly
                # published the prompts, with a warning on a console nobody
                # reads.
                print(
                    f"{Colors.RED}SECURITY: refusing to serve {info['name']} -- "
                    f"its metadata could not be removed.{Colors.RESET}"
                )
                message = (
                    "This file cannot be shown right now: its metadata could "
                    "not be removed, and showing it unchanged would reveal the "
                    "prompt and workflow stored inside it."
                )
                if file_type in ("video", "audio"):
                    message += (
                        " Removing metadata from video and audio needs ffmpeg:"
                        " install it on PATH, or point FFPROBE_MANUAL_PATH at"
                        " the folder holding ffmpeg and ffprobe."
                    )
                abort(503, description=message)

            # Not a protected request -- this caller is entitled to the
            # original, so an unstrippable file is served as it is.
            print(f"WARNING: Metadata stripping failed for {info['name']}. Falling back to original file.")
            if filepath.lower().endswith(".webp"):
                return send_file(filepath, mimetype="image/webp")
            return send_file(filepath)

    # Serve the file with correct mimetype for WebP
    if filepath.lower().endswith(".webp"):
        return send_file(clean_path, mimetype="image/webp")
    return send_file(clean_path)


@app.route("/galleryout/file/<string:file_id>")
def serve_file(file_id):
    if not is_file_accessible(file_id):
        abort(403, description="Access Denied.")
    if should_strip_metadata():
        return serve_cleaned_file(file_id)

    # Default: serve original
    filepath = get_file_info_from_db(file_id, "path")
    if filepath.lower().endswith(".webp"):
        return send_file(filepath, mimetype="image/webp")
    return send_file(filepath)


@app.route("/galleryout/download/<string:file_id>")
def download_file(file_id):
    if not is_file_accessible(file_id):
        abort(403, description="Access Denied.")
    if should_strip_metadata():
        # Logic for download is identical but we ensure serve_cleaned_file handles the cache
        info = get_file_info_from_db(file_id)
        filepath, mtime, file_type = info["path"], info["mtime"], info["type"]

        cache_hash = content_digest(filepath + str(mtime))
        _, ext = os.path.splitext(filepath)
        clean_path = os.path.join(CLEAN_CACHE_DIR, f"{cache_hash}{ext}")

        if not os.path.exists(clean_path):
            os.makedirs(CLEAN_CACHE_DIR, exist_ok=True)
            success = strip_media_metadata(filepath, clean_path, file_type)
            if not success:
                # Same refusal as the view route, and it matters more here:
                # a download hands the visitor the file itself, prompt and
                # workflow included, to keep and read at leisure.
                print(
                    f"{Colors.RED}SECURITY: refusing to hand out {info['name']} -- "
                    f"its metadata could not be removed.{Colors.RESET}"
                )
                message = (
                    "This file cannot be downloaded right now: its metadata "
                    "could not be removed, and the original carries the prompt "
                    "and workflow used to make it."
                )
                if file_type in ("video", "audio"):
                    message += (
                        " Removing metadata from video and audio needs ffmpeg:"
                        " install it on PATH, or point FFPROBE_MANUAL_PATH at"
                        " the folder holding ffmpeg and ffprobe."
                    )
                abort(503, description=message)

        return send_file(clean_path, as_attachment=True, download_name=info["name"])

    # Admin/Staff: serve original
    filepath = get_file_info_from_db(file_id, "path")
    return send_file(filepath, as_attachment=True)


@app.route("/galleryout/workflow/<string:file_id>")
def download_workflow(file_id):
    if not is_file_accessible(file_id):
        abort(403, description="Access Denied.")
    if should_strip_metadata():
        abort(403, description="Security Policy: Access to raw workflow metadata is restricted for your role.")

    info = get_file_info_from_db(file_id)
    filepath = info["path"]
    original_filename = info["name"]

    workflow_json = extract_workflow(filepath, target_type="ui")

    if workflow_json:
        base_name, _ = os.path.splitext(original_filename)
        new_filename = f"{base_name}.json"
        headers = {"Content-Disposition": f'attachment;filename="{new_filename}"'}
        return Response(workflow_json, mimetype="application/json", headers=headers)
    abort(404, description="That file has no workflow saved in it.")
    return None


@app.route("/galleryout/node_summary/<string:file_id>")
def get_node_summary(file_id):
    if should_strip_metadata():
        return jsonify(
            {"status": "error", "message": "Security Policy: Access to node summary is restricted for your role."}
        ), 403

    try:
        file_info = get_file_info_from_db(file_id)
        filepath = file_info["path"]
        db_dimensions = file_info.get("dimensions")

        ui_json = extract_workflow(filepath, target_type="ui")
        if not ui_json:
            return jsonify({"status": "error", "message": "Workflow not found for this file."}), 404

        summary_data = generate_node_summary(ui_json)

        api_json = extract_workflow(filepath, target_type="api")
        meta_data = {}

        try:
            json_source = api_json or ui_json
            wf_data = json.loads(json_source)
            if isinstance(wf_data, list):
                wf_data = {str(i): n for i, n in enumerate(wf_data)}

            parser = ComfyMetadataParser(wf_data)
            parsed_meta = parser.parse()

            tech_count = 0
            if parsed_meta.get("seed"):
                tech_count += 1
            if parsed_meta.get("model"):
                tech_count += 1
            if parsed_meta.get("steps"):
                tech_count += 1
            if parsed_meta.get("sampler"):
                tech_count += 1

            has_prompt = len(parsed_meta.get("positive_prompt", "")) > 5
            has_loras = len(parsed_meta.get("loras", [])) > 0

            if (has_prompt and tech_count >= 2) or has_loras:
                meta_data = parsed_meta
                if meta_data.get("loras"):
                    enriched_loras = []
                    for lora in meta_data["loras"]:
                        l_name = lora.get("name", "")
                        l_val = lora.get("value", 1.0)
                        l_dict = {
                            "name": l_name,
                            "value": l_val,
                            "preview_image": None,
                            "civitai_url": None,
                            "civitai_id": None,
                        }
                        try:
                            norm_name = os.path.normpath(l_name).replace(chr(92), "/")
                            clean_name = os.path.splitext(norm_name)[0]
                            base_path = os.path.join(LORAS_PATH, clean_name)
                            raw_path = os.path.join(LORAS_PATH, norm_name)

                            img_paths = [
                                base_path + ".preview.png",
                                base_path + ".png",
                                base_path + ".jpg",
                                base_path + ".jpeg",
                                raw_path + ".preview.png",
                                raw_path + ".png",
                                raw_path + ".jpg",
                                raw_path + ".jpeg",
                            ]
                            for ip in img_paths:
                                if os.path.exists(ip):
                                    with open(ip, "rb") as f:
                                        encoded = base64.b64encode(f.read()).decode("utf-8")
                                        mime = "image/png" if "png" in ip.lower() else "image/jpeg"
                                        l_dict["preview_image"] = f"data:{mime};base64," + encoded
                                    break

                            json_paths = [
                                base_path + ".civitai.info",
                                base_path + ".metadata.json",
                                base_path + ".info",
                                base_path + ".json",
                                raw_path + ".civitai.info",
                                raw_path + ".metadata.json",
                                raw_path + ".info",
                                raw_path + ".json",
                            ]
                            for jp in json_paths:
                                if os.path.exists(jp):
                                    with open(jp, encoding="utf-8") as f:
                                        jdata = json.load(f)
                                    cid = jdata.get("modelId") or jdata.get("id")
                                    if not cid and "civitai" in jdata:
                                        cid = jdata["civitai"].get("modelId") or jdata["civitai"].get("id")
                                    if cid:
                                        l_dict["civitai_id"] = cid
                                        l_dict["civitai_url"] = "https://civitai.com/models/" + str(cid)
                                        break
                        except Exception:
                            # Sidecar preview/metadata is decoration: a missing,
                            # unreadable, or malformed file leaves the LoRA
                            # listed without its thumbnail, not the page broken.
                            _logger.debug("ignored a failure in get_node_summary", exc_info=True)
                        enriched_loras.append(l_dict)
                    meta_data["loras"] = enriched_loras
                if not meta_data.get("width") or not meta_data.get("height"):
                    if db_dimensions and "x" in db_dimensions:
                        w, h = db_dimensions.split("x")
                        meta_data["width"], meta_data["height"] = w.strip(), h.strip()
            else:
                meta_data = {}

        except Exception as e:
            _logger.debug("handled a failure in get_node_summary", exc_info=True)
            print(f"Metadata Validation Warning: {e}")
            meta_data = {}

        return jsonify({"status": "success", "summary": summary_data, "meta": meta_data})

    except HTTPException as stop:
        # Answered 200 with the 404's own text pasted into the message, so
        # the page was told the request had succeeded.
        return answer_an_abort_readably(stop)
    except Exception as e:
        _logger.debug("handled a failure in get_node_summary", exc_info=True)
        print(f"ERROR generating node summary: {e}")
        return jsonify({"status": "error", "message": str(e)})


@app.route("/galleryout/waveform/<string:file_id>")
def serve_waveform(file_id):
    if not is_file_accessible(file_id):
        abort(403, description="Access Denied.")
    if not GENERATE_WAVEFORMS:
        abort(404, description="Waveforms are switched off in the settings.")
    info = get_file_info_from_db(file_id)
    filepath = info["path"]
    file_type = info["type"]
    file_hash = content_digest(filepath + str(info["mtime"]))

    try:
        amp = float(request.args.get("amp", "1.0"))
    except ValueError:
        amp = 1.0

    suffix = f"_{amp}" if amp != 1.0 else ""
    cache_path = os.path.join(THUMBNAIL_CACHE_DIR, f"{file_hash}_wave{suffix}.png")

    # 1. Return cached waveform if it exists
    if os.path.exists(cache_path):
        return send_file(cache_path, mimetype="image/png")

    # 2. On-the-fly generation for old/existing files
    if file_type in ["video", "audio"]:
        new_cache_path = create_waveform(filepath, file_hash, file_type, amp)
        if new_cache_path and os.path.exists(new_cache_path):
            return send_file(new_cache_path, mimetype="image/png")

    abort(404, description="No waveform could be made for that file.")
    return None


@app.route("/galleryout/thumbnail/<string:file_id>")
def serve_thumbnail(file_id):
    if not is_file_accessible(file_id):
        abort(403, description="Access Denied.")
    info = get_file_info_from_db(file_id)
    filepath, mtime = info["path"], info["mtime"]
    file_hash = content_digest(filepath + str(mtime))
    existing_thumbnails = glob.glob(os.path.join(THUMBNAIL_CACHE_DIR, f"{file_hash}.*"))
    if existing_thumbnails:
        return send_file(existing_thumbnails[0])
    if (
        not thumbnail_generation_enabled()
        and info["type"] in ("image", "animated_image")
        and not should_strip_metadata()
    ):
        # Site setting says no thumbnail compute: serve the original and let
        # the browser downscale. Videos fall through — a raw video file can't
        # act as an <img> tile, so a single poster frame is still rendered
        # on demand.
        #
        # Not for a caller who is owed a cleaned copy. This is the route every
        # tile in the grid asks for, so with the setting off it handed a
        # visitor the original of every picture, prompt and all, without them
        # opening any of it. They fall through instead and get a thumbnail,
        # which is re-encoded from pixels and carries no metadata; the setting
        # exists to save CPU, and that does not outrank the guarantee.
        if os.path.exists(filepath):
            return send_file(filepath)
        abort(404, description="That thumbnail is not in the cache.")
    print(f"WARN: Thumbnail not found for {os.path.basename(filepath)}, generating...")
    cache_path = create_thumbnail(filepath, file_hash, info["type"])
    if cache_path and os.path.exists(cache_path):
        return send_file(cache_path)
    return "Thumbnail generation failed", 404


# --- STORYBOARD (GRID SYSTEM) - FAST + SMART CORRUPTION DETECTION ---
@app.route("/galleryout/storyboard/<string:file_id>")
@management_api_only
def get_storyboard(file_id):
    # One request here is a probe, sometimes a transcode, and eleven ffmpeg
    # frame extractions. It is a management-side tool -- only index.html
    # asks for it -- but it was reachable by anyone who could see the video,
    # who could walk an album and set that going for every file in it.
    if not is_file_accessible(file_id):
        abort(403, description="Access Denied.")
    # 1. Validation
    has_ffmpeg = FFPROBE_EXECUTABLE_PATH is not None

    try:
        info = get_file_info_from_db(file_id)
        if info["type"] not in ["video", "animated_image"]:
            return jsonify({"status": "error", "message": "Not a video or animated file"}), 400

        if info["type"] == "video" and not has_ffmpeg:
            return jsonify({"status": "error", "message": "FFmpeg not available"}), 501

        filepath = info["path"]
        mtime = info["mtime"]

        # 2. Cache Strategy
        file_hash = content_digest(filepath + str(mtime))
        cache_subdir = os.path.join(THUMBNAIL_CACHE_DIR, file_hash)

        # Return cached results immediately if available
        if os.path.exists(cache_subdir):
            cached_files = sorted(glob.glob(os.path.join(cache_subdir, "frame_*.jpg")))
            if len(cached_files) > 0:
                urls = [f"/galleryout/storyboard_frame/{file_hash}/{os.path.basename(f)}" for f in cached_files]
                return jsonify({"status": "success", "cached": True, "frames": urls})

        os.makedirs(cache_subdir, exist_ok=True)

        # 3. Get Duration + FPS + Frame Count
        duration = 0
        fps = 0
        total_video_frames = 0

        if info["type"] == "video" and has_ffmpeg:
            # Get duration, fps, and frame count in ONE call
            try:
                cmd_info = [
                    FFPROBE_EXECUTABLE_PATH,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=duration,r_frame_rate,nb_frames",
                    "-of",
                    "csv=p=0",
                    filepath,
                ]
                res = run_media_tool(cmd_info, timeout=3)
                if res.stdout.strip():
                    parts = res.stdout.strip().split(",")

                    if len(parts) > 0 and parts[0]:
                        fps_str = parts[0]
                        if "/" in fps_str:
                            num, den = fps_str.split("/")
                            fps = float(num) / float(den)
                        else:
                            fps = float(fps_str)

                    if len(parts) > 1 and parts[1]:
                        duration = float(parts[1])

                    if len(parts) > 2 and parts[2]:
                        total_video_frames = int(parts[2])

            except Exception as e:
                _logger.debug("handled a failure in get_storyboard", exc_info=True)
                print(f"Info probe error: {e}")

            # Fallback: Try DB duration
            if duration <= 0 and info.get("duration"):
                try:
                    parts = info["duration"].split(":")
                    parts.reverse()
                    duration += float(parts[0])
                    if len(parts) > 1:
                        duration += int(parts[1]) * 60
                    if len(parts) > 2:
                        duration += int(parts[2]) * 3600
                except ValueError:
                    # A stored duration that is not h:mm:ss. Fall through to
                    # the ffprobe fallbacks below.
                    pass

            # Fallback: Try format duration
            if duration <= 0:
                try:
                    cmd_dur2 = [
                        FFPROBE_EXECUTABLE_PATH,
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        filepath,
                    ]
                    res2 = run_media_tool(cmd_dur2, timeout=3)
                    if res2.stdout.strip():
                        duration = float(res2.stdout.strip())
                except (OSError, subprocess.SubprocessError, ValueError):
                    # ffprobe missing, killed by the timeout, or answering
                    # with something that is not a number.
                    pass

            # Calculate missing values
            if total_video_frames == 0 and duration > 0 and fps > 0:
                total_video_frames = int(duration * fps)
            elif fps == 0 and duration > 0 and total_video_frames > 0:
                fps = total_video_frames / duration

        # Final fallback
        if duration <= 0 and info["type"] == "video":
            duration = 60
        if fps <= 0 and info["type"] == "video":
            fps = 25

        # 4. SMART CORRUPTION TEST - Test at 50% instead of end (faster + reliable)
        needs_transcode = False

        if info["type"] == "video" and has_ffmpeg and duration > 15:
            print("🔍 Quick test...")

            ffmpeg_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
            ffmpeg_bin = os.path.join(os.path.dirname(FFPROBE_EXECUTABLE_PATH), ffmpeg_name)
            if not os.path.exists(ffmpeg_bin):
                ffmpeg_bin = ffmpeg_name

            test_path = os.path.join(cache_subdir, "test.jpg")
            # Test at 50% - faster seek and still detects corruption
            test_timestamp = duration * 0.5

            creation_flags = no_console_window()

            # Adaptive timeout based on duration
            test_timeout = min(20, max(8, int(duration / 100)))  # 8-20s range

            cmd_test = [
                ffmpeg_bin,
                "-y",
                "-ss",
                f"{test_timestamp:.3f}",
                "-i",
                filepath,
                "-frames:v",
                "1",
                "-vf",
                "scale=-2:240:flags=fast_bilinear",
                "-q:v",
                "5",
                test_path,
            ]

            try:
                subprocess.run(
                    cmd_test,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=test_timeout,
                    creationflags=creation_flags,
                )

                if os.path.exists(test_path) and os.path.getsize(test_path) > 100:
                    print("✅ Healthy")
                    needs_transcode = False
                else:
                    print("⚠️ Corrupted!")
                    needs_transcode = True

            except subprocess.TimeoutExpired:
                # Timeout on healthy files = just slow, not corrupted
                print("⏱️ Slow seek (normal for large files)")
                needs_transcode = False
            except Exception as e:
                _logger.debug("handled a failure in get_storyboard", exc_info=True)
                print(f"⚠️ Corrupted: {e}")
                needs_transcode = True

            if os.path.exists(test_path):
                with contextlib.suppress(OSError):
                    os.remove(test_path)

        # 5. TRANSCODING if needed
        source_for_extraction = filepath
        temp_transcoded = None

        if needs_transcode:
            print("🔧 Transcoding...")

            try:
                ffmpeg_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
                ffmpeg_bin = os.path.join(os.path.dirname(FFPROBE_EXECUTABLE_PATH), ffmpeg_name)
                if not os.path.exists(ffmpeg_bin):
                    ffmpeg_bin = ffmpeg_name

                temp_transcoded = os.path.join(cache_subdir, f"temp_proxy_{uuid.uuid4().hex}.mp4")

                cmd_transcode = [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    filepath,
                    "-vf",
                    "scale=-2:480",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "28",
                    "-an",
                    "-movflags",
                    "+faststart",
                    temp_transcoded,
                ]

                creation_flags = no_console_window()

                subprocess.run(cmd_transcode, capture_output=True, text=True, timeout=300, creationflags=creation_flags)

                if os.path.exists(temp_transcoded) and os.path.getsize(temp_transcoded) > 1000:
                    print("✅ Transcoded")
                    source_for_extraction = temp_transcoded

                    # Get corrected info
                    try:
                        cmd_info = [
                            FFPROBE_EXECUTABLE_PATH,
                            "-v",
                            "error",
                            "-select_streams",
                            "v:0",
                            "-show_entries",
                            "stream=duration,r_frame_rate,nb_frames",
                            "-of",
                            "csv=p=0",
                            temp_transcoded,
                        ]
                        res = run_media_tool(cmd_info, timeout=2)
                        if res.stdout.strip():
                            parts = res.stdout.strip().split(",")

                            if len(parts) > 0 and parts[0]:
                                fps_str = parts[0]
                                if "/" in fps_str:
                                    num, den = fps_str.split("/")
                                    fps = float(num) / float(den)
                                else:
                                    fps = float(fps_str)

                            if len(parts) > 1 and parts[1]:
                                duration = float(parts[1])

                            if len(parts) > 2 and parts[2]:
                                total_video_frames = int(parts[2])
                    except (OSError, subprocess.SubprocessError, ValueError, ZeroDivisionError):
                        # ffprobe missing, timed out, or reporting a frame rate
                        # this cannot divide (a 0 denominator) or parse.
                        pass

            except Exception as e:
                _logger.debug("handled a failure in get_storyboard", exc_info=True)
                print(f"❌ Transcode failed: {e}")
                if temp_transcoded and os.path.exists(temp_transcoded):
                    with contextlib.suppress(OSError):
                        os.remove(temp_transcoded)
                temp_transcoded = None

        # 6. Worker Function (OPTIMIZED)
        def extract_and_save_frame(index, timestamp):
            out_filename = f"frame_{index:02d}.jpg"
            out_path = os.path.join(cache_subdir, out_filename)

            try:
                img = None
                actual_timestamp = timestamp
                actual_frame_number = None

                # A. Video Extraction
                if info["type"] == "video" and has_ffmpeg:
                    actual_timestamp = timestamp

                    if fps > 0:
                        actual_frame_number = int(timestamp * fps)

                    ffmpeg_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
                    ffmpeg_bin = os.path.join(os.path.dirname(FFPROBE_EXECUTABLE_PATH), ffmpeg_name)
                    if not os.path.exists(ffmpeg_bin):
                        ffmpeg_bin = ffmpeg_name

                    creation_flags = no_console_window()

                    # Fast extraction
                    cmd = [
                        ffmpeg_bin,
                        "-y",
                        "-ss",
                        f"{timestamp:.3f}",
                        "-i",
                        source_for_extraction,
                        "-frames:v",
                        "1",
                        "-vf",
                        "scale=-2:360:flags=fast_bilinear",
                        "-q:v",
                        "4",
                        "-preset",
                        "ultrafast",
                        out_path,
                    ]

                    try:
                        subprocess.run(
                            cmd,
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=8,
                            creationflags=creation_flags,
                        )

                        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
                            img = Image.open(out_path)

                    except Exception:
                        _logger.debug("handled a failure in extract_and_save_frame", exc_info=True)
                        if os.path.exists(out_path):
                            with contextlib.suppress(OSError):
                                os.remove(out_path)

                        # Slow seek fallback
                        cmd_slow = [
                            ffmpeg_bin,
                            "-y",
                            "-i",
                            source_for_extraction,
                            "-ss",
                            f"{timestamp:.3f}",
                            "-frames:v",
                            "1",
                            "-vf",
                            "scale=-2:360:flags=fast_bilinear",
                            "-q:v",
                            "4",
                            out_path,
                        ]

                        try:
                            subprocess.run(
                                cmd_slow,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=40,
                                creationflags=creation_flags,
                            )

                            if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
                                img = Image.open(out_path)
                        except (OSError, subprocess.SubprocessError, ValueError):
                            # The slow-seek fallback is the last attempt at this
                            # frame; a missing ffmpeg, a timeout, or an
                            # unreadable output leaves this thumbnail empty.
                            pass

                # B. Animation Extraction
                elif info["type"] == "animated_image":
                    with Image.open(filepath) as source_img:
                        is_anim = getattr(source_img, "is_animated", False)
                        total_frames = source_img.n_frames if is_anim else 1
                        pct = index / 10.0
                        target_frame_idx = int(pct * (total_frames - 1))
                        source_img.seek(target_frame_idx)
                        img = source_img.copy().convert("RGB")
                        img.thumbnail((640, 360))

                        actual_timestamp = None
                        actual_frame_number = target_frame_idx + 1

                # C. Professional Overlay
                if img:
                    draw = ImageDraw.Draw(img)

                    # Calculate text
                    if actual_timestamp is None:
                        # Animation
                        with Image.open(filepath) as temp_img:
                            total_frames = temp_img.n_frames if getattr(temp_img, "is_animated", False) else 1
                        time_str = f"#{actual_frame_number}/{total_frames}"
                    else:
                        # Video: timestamp + frame
                        display_ts = round(actual_timestamp)
                        m, s = int(display_ts // 60), int(display_ts % 60)

                        if actual_frame_number is not None and total_video_frames > 0:
                            display_frame_number = actual_frame_number + 1
                            time_str = f"{m:02d}:{s:02d} | #{display_frame_number}/{total_video_frames}"
                        else:
                            time_str = f"{m:02d}:{s:02d}"

                    # Font
                    font_size = 24
                    font = None
                    try:
                        font = ImageFont.load_default(size=font_size)
                    except TypeError:
                        # Pillow before 10.1 has no size argument here.
                        font = ImageFont.load_default()

                    # Measure
                    left, top, right, bottom = draw.textbbox((0, 0), time_str, font=font)
                    txt_w = right - left
                    txt_h = bottom - top

                    # Box
                    pad_x = 6
                    pad_y = 4
                    box_w = txt_w + (pad_x * 2)
                    box_h = txt_h + (pad_y * 2)

                    # Draw
                    draw.rectangle([0, 0, box_w, box_h], fill="black", outline=None)
                    draw.text((pad_x - left, pad_y - top), time_str, font=font, fill="#ffffff")

                    # Save
                    img.save(out_path, quality=85)
                    img.close()

                    return f"/galleryout/storyboard_frame/{file_hash}/{out_filename}"

            except Exception as e:
                _logger.debug("handled a failure in extract_and_save_frame", exc_info=True)
                print(f"Worker error {index}: {e}")

            return None

        # 7. Parallel Execution
        timestamps = []

        if info["type"] == "video":
            safe_end = max(0, duration - 0.1)
            # Generate 11 evenly spaced timestamps, but force the last one (index 10) to be the exact last frame
            base_timestamps = [(i, (safe_end / 10) * i) for i in range(11)]
            # Override the last timestamp to point to the very end (or last frame if frame count is known)
            if total_video_frames > 0 and fps > 0:
                # Use exact last frame position
                last_frame_timestamp = (total_video_frames - 1) / fps
                # Ensure it doesn't exceed duration
                last_frame_timestamp = min(last_frame_timestamp, duration - 0.001)
                base_timestamps[-1] = (10, last_frame_timestamp)
            else:
                # Fallback: use end of video minus a tiny epsilon
                base_timestamps[-1] = (10, duration - 0.001)
            timestamps = base_timestamps
        else:
            timestamps = [(i, 0) for i in range(11)]

        frame_urls = [None] * 11

        print("🎬 Extracting...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=11) as executor:
            futures = {executor.submit(extract_and_save_frame, i, ts): i for i, ts in timestamps}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                res = future.result()
                if res:
                    frame_urls[idx] = res

        success_count = sum(1 for url in frame_urls if url is not None)
        print(f"✅ {success_count}/11")

        # Cleanup
        if temp_transcoded and os.path.exists(temp_transcoded):
            with contextlib.suppress(OSError):
                os.remove(temp_transcoded)

        final_urls = [url for url in frame_urls if url is not None]

        if not final_urls:
            return jsonify({"status": "error", "message": "Extraction failed completely."}), 500

        return jsonify({"status": "success", "cached": False, "frames": final_urls})

    except HTTPException as stop:
        # Its own 404 or 403, not a fault to report as one.
        return answer_an_abort_readably(stop)
    except Exception as e:
        _logger.debug("handled a failure in get_storyboard", exc_info=True)
        print(f"Storyboard error: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/storyboard_frame/<string:file_hash>/<string:filename>")
def serve_storyboard_frame(file_hash, filename):
    # send_from_directory guards the filename but takes the directory on
    # trust, and the default converter matches any segment without a slash
    # -- `..` included. That made this route read the gallery root, which is
    # where the pictures are, for a caller with no session at all.
    if (IS_EXHIBITION_MODE or FORCE_LOGIN) and not session.get("user_id"):
        abort(403, description="Access Denied.")
    if not re.fullmatch(r"[0-9a-f]{32}", file_hash or ""):
        abort(404, description="That is not the name of a stored frame.")

    # Frames are written by this app as frame_NN.jpg, so nothing is lost
    # here -- but the same helper is used everywhere a name becomes a
    # path, so there is one rule to check rather than a judgement call
    # per call site about whether that particular name could be someone's.
    safe_name = safe_media_filename(filename)
    directory = os.path.join(THUMBNAIL_CACHE_DIR, file_hash)
    # Belt and braces: whatever the segment was, the folder it names has to
    # sit inside the frame cache.
    if not _is_under(os.path.realpath(directory), os.path.realpath(THUMBNAIL_CACHE_DIR)):
        abort(404, description="That is not the name of a stored frame.")
    return send_from_directory(directory, safe_name)


# Route to serve the cached frames


@app.route("/galleryout/api/remix/object_info", methods=["POST"])
@management_api_only
def api_remix_object_info():
    try:
        target_url = request.json.get("target_url", COMFYUI_SERVER_URL).strip()
        if not target_url:
            target_url = COMFYUI_SERVER_URL
        req = urllib.request.Request(
            f"{target_url.rstrip('/')}/object_info", headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return Response(r.read(), mimetype="application/json")
    except Exception as e:
        _logger.debug("handled a failure in api_remix_object_info", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/input_file/<path:filename>")
def serve_input_file(filename):
    user_role = session.get("role", "GUEST")
    if (IS_EXHIBITION_MODE or FORCE_LOGIN) and user_role not in ["ADMIN", "MANAGER", "STAFF"]:
        abort(403, description="Access Denied.")
    """Serves input files directly from the ComfyUI Input folder."""
    try:
        # secure_filename() used to run here, and it is the wrong tool: it
        # is for naming a file you are about to WRITE, not for finding one
        # you are about to READ. It strips every non-ASCII character, so
        # 测试.png arrived as "png" and 404'd, and Ordner-Größe.png lost its
        # umlaut -- the same fault that was fixed for uploads. It also
        # flattens separators, so ComfyUI's own clipspace/ images, which
        # every paste and mask produces, became clipspace_x.png and 404'd
        # for everyone.
        #
        # Containment is the thing that matters, and it is checked twice:
        # here on the resolved path, and again inside send_from_directory,
        # whose safe_join refuses '..', absolute paths and drive-relative
        # ones. Compared with _is_under rather than startswith, so a
        # sibling folder named input_backup cannot pass for input.
        input_root = os.path.realpath(BASE_INPUT_PATH)
        resolved = os.path.realpath(os.path.join(BASE_INPUT_PATH, filename))
        if resolved != input_root and not _is_under(resolved, input_root):
            abort(403)

        # For webp, frocing the correct mimetype
        if filename.lower().endswith(".webp"):
            return send_from_directory(BASE_INPUT_PATH, filename, mimetype="image/webp", as_attachment=False)

        # For all the other files, I let Flask guessing the mimetype, but disable the attachment, just a lil trick
        return send_from_directory(BASE_INPUT_PATH, filename, as_attachment=False)
    except Exception:
        # Whatever went wrong -- missing, unreadable, outside the folder --
        # the answer to the person asking is the same one sentence.
        _logger.debug("handled a failure in serve_input_file", exc_info=True)
        abort(404, description="That file is not in the input folder.")


@app.route("/galleryout/check_metadata/<string:file_id>")
def check_metadata(file_id):
    """
    Lightweight endpoint to check real-time status of metadata.
    Now includes Real Path resolution for mounted folders.
    """
    if not is_file_accessible(file_id):
        return jsonify({"status": "error", "message": "Access Denied"}), 403
    try:
        with get_db_connection() as conn:
            # Added 'path' to selection to resolve symlinks
            row = conn.execute(
                "SELECT path, has_workflow, ai_caption, ai_last_scanned FROM files WHERE id = ?", (file_id,)
            ).fetchone()

        if not row:
            return jsonify({"status": "error", "message": "File not found"}), 404

        # Resolve Real Path (Handles Windows Junctions and Linux Symlinks)
        internal_path = row["path"]
        real_path_resolved = os.path.realpath(internal_path)

        # Check if they differ (ignore case on Windows for safety)
        is_different = False
        if os.name == "nt":
            if internal_path.lower() != real_path_resolved.lower():
                is_different = True
        elif internal_path != real_path_resolved:
            is_different = True

        return jsonify(
            {
                "status": "success",
                "has_workflow": bool(row["has_workflow"]),
                "has_ai_caption": bool(row["ai_caption"]),
                "ai_caption": row["ai_caption"] or "",
                "ai_last_scanned": row["ai_last_scanned"] or 0,
                # Send real_path only if it's actually different (a link), and
                # never to someone who is not meant to know where the library
                # lives on disk.
                "real_path": (real_path_resolved if is_different and not should_strip_metadata() else None),
            }
        )
    except Exception as e:
        _logger.debug("handled a failure in check_metadata", exc_info=True)
        print(f"Metadata Check Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


def stream_media_process(process, label="", stall_timeout=None):
    """Hand a transcoder's output to the caller, and give up if it goes quiet.

    The reading thread is the one serving the request, so it cannot watch
    itself; a separate thread holds the clock and kills the child if no
    output arrives for stall_timeout seconds. Killing it closes the pipe,
    the blocked read returns empty, and the loop below ends normally --
    which is also what runs the clean-up.
    """
    if stall_timeout is None:
        stall_timeout = MEDIA_STREAM_STALL_TIMEOUT

    last_output = [time.monotonic()]
    delivered = [0]
    stopped = threading.Event()
    gave_up = []

    def watch():
        # Polls rather than re-arming a timer per chunk: a healthy stream
        # produces thousands of chunks and each one would be a new thread.
        while not stopped.wait(min(1.0, max(stall_timeout / 4.0, 0.05))):
            if time.monotonic() - last_output[0] > stall_timeout:
                gave_up.append(True)
                print(f"Stream Stalled: no output for {stall_timeout}s, stopping the transcode of {label or 'a file'}.")
                with contextlib.suppress(Exception):
                    process.kill()
                return

    guard = threading.Thread(target=watch, daemon=True)
    guard.start()

    try:
        # read1 rather than read: read() waits for the full 16KB before it
        # returns anything, which holds finished frames back from the player
        # and, worse, makes a stream that is producing steadily look silent
        # to the clock above until a whole chunk has piled up.
        while True:
            data = process.stdout.read1(16384)
            if not data:
                break
            last_output[0] = time.monotonic()
            delivered[0] += len(data)
            yield data
    finally:
        stopped.set()
        # Clean up: ensure the process is killed when the request ends
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        # A stream that produced nothing at all reaches the player as an
        # empty video with no explanation, so say so where it can be read.
        if not delivered[0] and not gave_up:
            print(
                f"Stream Empty: the transcode of {label or 'a file'} "
                f"produced no output (exit code {process.returncode})."
            )


@app.route("/galleryout/stream/<string:file_id>")
def stream_video(file_id):
    if not is_file_accessible(file_id):
        abort(403, description="Access Denied.")
    """
    Streams video files by transcoding them on-the-fly using FFmpeg.
    This allows professional formats like ProRes to be viewed in any browser.
    Includes a safety scale filter to ensure smooth playback even for 4K+ sources.
    """
    filepath = get_file_info_from_db(file_id, "path")

    if not FFPROBE_EXECUTABLE_PATH:
        abort(404, description="FFmpeg/FFprobe not found on system.")

    # Determine ffmpeg executable path based on ffprobe location
    ffmpeg_dir = os.path.dirname(FFPROBE_EXECUTABLE_PATH)
    ffmpeg_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    ffmpeg_path = os.path.join(ffmpeg_dir, ffmpeg_name) if ffmpeg_dir else ffmpeg_name

    # FFmpeg command for fast on-the-fly transcoding
    # -preset ultrafast: minimal CPU usage
    # -vf scale: ensures the stream is not larger than 720p for performance
    # -movflags frag_keyframe+empty_moov: required for fragmented MP4 streaming

    # FFmpeg command for fast on-the-fly transcoding
    # ADDED: -map_metadata -1 to ensure NO workflow info is streamed to the client
    cmd = [
        ffmpeg_path,
        "-i",
        filepath,
        "-map_metadata",
        "-1",  # <--- STRIP METADATA FROM STREAM
        "-map_metadata:s:v",
        "-1",  # <--- STRIP VIDEO STREAM DATA
        "-map_metadata:s:a",
        "-1",  # <--- STRIP AUDIO STREAM DATA
        "-vcodec",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-vf",
        "scale='min(1280,iw)':-2",
        "-acodec",
        "aac",
        "-b:a",
        "128k",
        "-f",
        "mp4",
        "-movflags",
        "frag_keyframe+empty_moov+default_base_moof",
        "pipe:1",
    ]

    def generate():
        # Start ffmpeg process with specific flags to avoid console windows on Windows
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=no_console_window(),
        )
        return stream_media_process(process, os.path.basename(filepath or ""))

    return Response(generate(), mimetype="video/mp4")


# --- COLLECTIONS / CATEGORIES API ---


def get_descendant_file_counts(conn, countable_collection_ids):
    """Count unique files in descendant collections, excluding direct membership."""
    if not countable_collection_ids:
        return {}

    placeholders = ",".join("?" for _ in countable_collection_ids)
    rows = conn.execute(
        f"""
        WITH RECURSIVE counted_descendants(id) AS (
            SELECT DISTINCT collection_id
            FROM collection_files
            WHERE collection_id IN ({placeholders})
        ), ancestry(ancestor_id, descendant_id) AS (
            SELECT c.parent_id, c.id
            FROM collections c
            JOIN counted_descendants d ON d.id = c.id
            WHERE c.parent_id IS NOT NULL
            UNION
            SELECT c.parent_id, a.descendant_id
            FROM ancestry a
            JOIN collections c ON c.id = a.ancestor_id
            WHERE c.parent_id IS NOT NULL
        )
        SELECT a.ancestor_id, COUNT(DISTINCT cf.file_id) AS descendant_file_count
        FROM ancestry a
        JOIN collection_files cf ON cf.collection_id = a.descendant_id
        WHERE a.ancestor_id != a.descendant_id
        GROUP BY a.ancestor_id
    """,
        countable_collection_ids,
    ).fetchall()
    return {row["ancestor_id"]: row["descendant_file_count"] for row in rows}


@app.route("/galleryout/api/collections", methods=["GET"])
def get_collections():
    if (IS_EXHIBITION_MODE or FORCE_LOGIN) and not session.get("user_id"):
        return jsonify({"status": "error", "message": "Authentication required"}), 401
    user_id = str(session.get("user_id", "")).strip()
    user_role = session.get("role", "GUEST")
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT c.*,
                   (SELECT COUNT(*) FROM collection_files cf WHERE cf.collection_id = c.id) AS file_count,
                   (SELECT COUNT(*) FROM collection_files cf
                        JOIN files f ON cf.file_id = f.id
                        WHERE cf.collection_id = c.id
                          AND (f.type = 'document' OR LOWER(f.name) LIKE '%.txt'
                               OR LOWER(f.name) LIKE '%.md')) AS note_count
            FROM collections c
            ORDER BY ulower(c.name)
        """).fetchall()

        all_cols = [dict(r) for r in rows]
        flags = [c for c in all_cols if c["type"] == "system_flag"]
        albums = [c for c in all_cols if c["type"] == "user_album"]
        for flag in flags:
            flag.pop("file_count", None)

        filtered_albums = []

        # Who is asking, not which mode the server is in. This filtering was
        # written as a property of exhibition mode, so the same non-staff
        # account under --force-login fell through to the branch that returns
        # every album, its file counts, and the shared_users list naming who
        # each private album is shared with. With no login configured there is
        # one person and they own the library, so they are privileged.
        is_local_admin = not FORCE_LOGIN and not IS_EXHIBITION_MODE
        is_privileged = is_local_admin or user_role in ["ADMIN", "MANAGER", "STAFF"]

        if not is_privileged:
            explicit_access_ids = set()
            album_dict = {c["id"]: c for c in albums}

            # Step 1: Identify explicitly accessible collections
            for c in albums:
                is_public = int(c.get("is_public", 0)) == 1
                shared_raw = str(c.get("shared_users", "")).split(",")
                shared_list = [str(uid).strip() for uid in shared_raw if uid.strip()]

                if is_public or str(user_id) in shared_list:
                    explicit_access_ids.add(c["id"])
                    if str(user_id) in shared_list:
                        c["is_shared_access"] = True

            # Step 2: Traverse upwards to unlock ancestor paths for the tree
            required_ancestors = set()
            for cid in explicit_access_ids:
                curr = album_dict.get(cid)
                while curr and curr.get("parent_id"):
                    pid = curr["parent_id"]
                    required_ancestors.add(pid)
                    curr = album_dict.get(pid)

            # Step 3: Build final list with locked/unlocked flags
            for c in albums:
                if c["id"] in explicit_access_ids:
                    c["restricted_access"] = False
                    filtered_albums.append(c)
                elif c["id"] in required_ancestors:
                    c["restricted_access"] = True
                    filtered_albums.append(c)

        elif IS_EXHIBITION_MODE:
            for c in albums:
                shared_raw = str(c.get("shared_users", "")).split(",")
                shared_list = [str(uid).strip() for uid in shared_raw if uid.strip()]
                if shared_list:
                    c["is_shared_access"] = True
                c["restricted_access"] = False
            filtered_albums = albums

        else:
            for c in albums:
                c["restricted_access"] = False
            filtered_albums = albums

        if not is_privileged:
            # Ancestors are listed so the tree can be drawn, but their
            # contents are not this caller's to count.
            count_album_ids = [
                c["id"]
                for c in filtered_albums
                if not c.get("restricted_access") and (c.get("is_public") or c.get("is_shared_access"))
            ]
        else:
            count_album_ids = [c["id"] for c in filtered_albums]

        all_count = 0
        if count_album_ids:
            placeholders = ",".join("?" for _ in count_album_ids)
            all_count = conn.execute(
                f"SELECT COUNT(DISTINCT file_id) FROM collection_files WHERE collection_id IN ({placeholders})",
                count_album_ids,
            ).fetchone()[0]

        # Map user names for shared collections tooltip
        user_rows = conn.execute("SELECT user_id, full_name FROM users").fetchall()
        user_map = {str(r["user_id"]): r["full_name"] for r in user_rows}

        descendant_counts = get_descendant_file_counts(conn, count_album_ids)
        for c in filtered_albums:
            c["descendant_file_count"] = descendant_counts.get(c["id"], 0)
            if c.get("restricted_access"):
                c["file_count"] = None
            if c.get("shared_users"):
                uids = [u.strip() for u in str(c["shared_users"]).split(",") if u.strip()]
                names = [user_map.get(u, "Unknown User") for u in uids]
                c["shared_user_names"] = ", ".join(names)

        return jsonify({"flags": flags, "albums": filtered_albums, "all_count": all_count})


@app.route("/galleryout/api/sidebar_state")
@management_api_only
def get_sidebar_state():
    """Returns the current state of folders and collections for real-time sync.

    Asking only for a session was not enough: this lists every album without
    regard to who is asking, private ones included, and resolves shared_users
    into the full names of the people each was shared with. It belongs to the
    management sidebar (collections.html, included only by index.html), so it
    is management-only.
    """
    folders = get_dynamic_folder_config(force_refresh=True)
    with get_db_connection() as conn:
        flags = conn.execute("SELECT * FROM collections WHERE type='system_flag' ORDER BY id").fetchall()
        if IS_EXHIBITION_MODE:
            albums = conn.execute("""
                SELECT c.*,
                       (SELECT COUNT(*) FROM collection_files cf WHERE cf.collection_id = c.id) AS file_count,
                       (SELECT COUNT(*) FROM collection_files cf
                        JOIN files f ON cf.file_id = f.id
                        WHERE cf.collection_id = c.id
                          AND (f.type = 'document' OR LOWER(f.name) LIKE '%.txt'
                               OR LOWER(f.name) LIKE '%.md')) AS note_count
                FROM collections c
                WHERE c.type='user_album'
                ORDER BY ulower(c.name)
            """).fetchall()
            all_count = None
        else:
            albums = conn.execute("""
                SELECT c.*,
                       (SELECT COUNT(*) FROM collection_files cf WHERE cf.collection_id = c.id) AS file_count,
                       (SELECT COUNT(*) FROM collection_files cf
                        JOIN files f ON cf.file_id = f.id
                        WHERE cf.collection_id = c.id
                          AND (f.type = 'document' OR LOWER(f.name) LIKE '%.txt'
                               OR LOWER(f.name) LIKE '%.md')) AS note_count
                FROM collections c
                WHERE c.type='user_album'
                ORDER BY ulower(c.name)
            """).fetchall()
            all_count = conn.execute("""
                SELECT COUNT(DISTINCT cf.file_id)
                FROM collection_files cf
                JOIN collections c ON c.id = cf.collection_id
                WHERE c.type='user_album'
            """).fetchone()[0]
        album_dicts = [dict(r) for r in albums]

        user_rows = conn.execute("SELECT user_id, full_name FROM users").fetchall()
        user_map = {str(r["user_id"]): r["full_name"] for r in user_rows}

        for album in album_dicts:
            if album.get("shared_users"):
                uids = [u.strip() for u in str(album["shared_users"]).split(",") if u.strip()]
                names = [user_map.get(u, "Unknown User") for u in uids]
                album["shared_user_names"] = ", ".join(names)

        if not IS_EXHIBITION_MODE:
            descendant_counts = get_descendant_file_counts(conn, [album["id"] for album in album_dicts])
            for album in album_dicts:
                album["descendant_file_count"] = descendant_counts.get(album["id"], 0)

    collections = {"flags": [dict(r) for r in flags], "albums": album_dicts}
    if all_count is not None:
        collections["all_count"] = all_count

    return jsonify({"folders": folders, "collections": collections})


@app.route("/galleryout/api/collections/rename", methods=["POST"])
@management_api_only
def rename_collection_api():
    data = request.json
    coll_id = data.get("id")
    new_name = data.get("name", "").strip()

    if not coll_id or not new_name:
        return jsonify({"status": "error", "message": "ID and Name required"}), 400

    try:
        with get_db_connection() as conn:
            # Prevent renaming system flags
            row = conn.execute("SELECT type FROM collections WHERE id=?", (coll_id,)).fetchone()
            if not row or row["type"] == "system_flag":
                return jsonify({"status": "error", "message": "Cannot rename system tags"}), 403

            conn.execute("UPDATE collections SET name = ? WHERE id = ?", (new_name, coll_id))
            conn.commit()

        return jsonify({"status": "success"})
    except Exception as e:
        _logger.debug("handled a failure in rename_collection_api", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/collections/create", methods=["POST"])
@management_api_only
def create_collection():
    data = request.json
    name = data.get("name", "").strip()
    is_public = data.get("is_public", False)
    parent_id = data.get("parent_id", None)
    shared_users = data.get("shared_users", "")

    if not name:
        return jsonify({"status": "error", "message": "Name required"}), 400

    try:
        with get_db_connection() as conn:
            # Execute insert and get the cursor to retrieve the lastrowid
            cursor = conn.execute(
                "INSERT INTO collections"
                " (name, type, color, is_public, shared_users, parent_id, created_at)"
                " VALUES (?, 'user_album', '#ffffff', ?, ?, ?, ?)",
                (name, 1 if is_public else 0, shared_users, parent_id, time.time()),
            )
            new_id = cursor.lastrowid  # <--- Get the newly created ID
            conn.commit()

        return jsonify({"status": "success", "id": new_id})  # <--- Return the ID
    except Exception as e:
        _logger.debug("handled a failure in create_collection", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/collections/delete", methods=["POST"])
@management_api_only
def delete_collection():
    coll_id = request.json.get("id")
    with get_db_connection() as conn:
        row = conn.execute("SELECT type FROM collections WHERE id=?", (coll_id,)).fetchone()
        if not row or row["type"] == "system_flag":
            return jsonify({"status": "error", "message": "Cannot delete system tags"}), 403
        # Use Recursive CTE to safely delete the collection and all its nested sub-collections
        conn.execute(
            """
            WITH RECURSIVE cols_to_delete AS (
                SELECT id FROM collections WHERE id = ?
                UNION ALL
                SELECT c.id FROM collections c
                INNER JOIN cols_to_delete cd ON c.parent_id = cd.id
            )
            DELETE FROM collections WHERE id IN cols_to_delete;
        """,
            (coll_id,),
        )
        # The collection_files table relationships are handled automatically by SQLite ON DELETE CASCADE
        conn.commit()
    return jsonify({"status": "success"})


@app.route("/galleryout/api/collections/toggle_public", methods=["POST"])
@management_api_only
def toggle_collection_public():
    try:
        data = request.json
        coll_id = int(data.get("id", 0))

        if not coll_id:
            return jsonify({"status": "error", "message": "ID required"}), 400

        with get_db_connection() as conn:
            row = conn.execute("SELECT is_public FROM collections WHERE id=?", (coll_id,)).fetchone()
            if not row:
                return jsonify({"status": "error", "message": "Collection not found"}), 404

            current_val = row["is_public"] if row["is_public"] is not None else 0
            new_state = 0 if current_val else 1

            conn.execute("UPDATE collections SET is_public = ? WHERE id = ?", (new_state, coll_id))
            conn.commit()

        return jsonify({"status": "success", "new_state": bool(new_state)})

    except Exception as e:
        _logger.debug("handled a failure in toggle_collection_public", exc_info=True)
        print(f"Toggle Public Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/collections/share", methods=["POST"])
@management_api_only
def share_collection():
    data = request.json
    coll_id = data.get("id")
    user_ids = data.get("user_ids", [])  # List of user ID strings

    if not coll_id:
        return jsonify({"status": "error", "message": "ID required"}), 400

    try:
        # Join IDs with commas
        shared_str = ",".join(str(uid) for uid in user_ids)

        with get_db_connection() as conn:
            # Force is_public to 0 if we are setting specific users (to avoid logic conflicts)
            if shared_str:
                conn.execute(
                    "UPDATE collections SET shared_users = ?, is_public = 0 WHERE id = ?", (shared_str, coll_id)
                )
            else:
                conn.execute("UPDATE collections SET shared_users = ? WHERE id = ?", (shared_str, coll_id))
            conn.commit()

        return jsonify({"status": "success"})
    except Exception as e:
        _logger.debug("handled a failure in share_collection", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/file_full_details/<string:file_id>")
def get_file_full_details(file_id):
    if not is_file_accessible(file_id):
        return jsonify({"status": "error", "message": "Access Denied"}), 403

    try:
        with get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT f.*,
                (SELECT c.color FROM collections c
                         JOIN collection_files cf ON c.id = cf.collection_id
                         WHERE cf.file_id = f.id AND c.type = 'system_flag' LIMIT 1) as status_color,
                (SELECT c.name FROM collections c
                         JOIN collection_files cf ON c.id = cf.collection_id
                         WHERE cf.file_id = f.id AND c.type = 'system_flag' LIMIT 1) as status_name,
                (SELECT AVG(rating) FROM file_ratings WHERE file_id = f.id) as avg_rating,
                (SELECT COUNT(*) FROM file_ratings WHERE file_id = f.id) as vote_count,
                (SELECT COUNT(*) FROM file_comments WHERE file_id = f.id) as comment_count
                FROM files f WHERE f.id = ?
            """,
                (file_id,),
            ).fetchone()

            if not row:
                return jsonify({"status": "error", "message": "File not found"}), 404

            file_data = dict(row)
            file_data.pop("ai_embedding", None)

            if should_strip_metadata():
                file_data["has_workflow"] = 0

            generation_metadata_text = None
            generation_tool = None
            if not should_strip_metadata() and file_data.get("type") in ("image", "animated_image"):
                parsed = metaparse.parse_file(file_data["path"], allow_stealth=True)
                if parsed:
                    generation_tool = parsed.tool
                    generation_metadata_text = metaparse.render_report(parsed)
            file_data["generation_metadata"] = generation_metadata_text
            file_data["generation_tool"] = generation_tool

            # First-class stored data: the typed generation_params row and
            # per-face attributes, so the details modal reflects what is
            # actually tracked instead of only a live re-parse.
            file_data["generation_params"] = None
            if not should_strip_metadata():
                gen_row = conn.execute("SELECT * FROM generation_params WHERE file_id = ?", (file_id,)).fetchone()
                if gen_row:
                    g = dict(gen_row)
                    for k in ("loras", "extra"):
                        if g.get(k):
                            with contextlib.suppress(Exception):
                                g[k] = json.loads(g[k])
                    file_data["generation_params"] = g
            try:
                # Face rows are provenance-scoped per pipeline; serve the
                # most recently computed model's rows (after a backend
                # switch that is the active one) instead of mixing models.
                face_rows = conn.execute(
                    "SELECT face_id, model_id, det_score, age, sex, pose_pitch, "
                    "pose_yaw, pose_roll, cluster_id FROM ai_face_instances "
                    "WHERE file_id = ? AND model_id = ("
                    "  SELECT model_id FROM ai_face_instances WHERE file_id = ? "
                    "  ORDER BY computed_at DESC LIMIT 1) "
                    "ORDER BY face_id",
                    (file_id, file_id),
                ).fetchall()
                file_data["faces"] = [dict(r) for r in face_rows]
            except Exception:
                _logger.debug("handled a failure in get_file_full_details", exc_info=True)
                file_data["faces"] = []

            folders_config = get_dynamic_folder_config()
            abs_path = file_data["path"]
            real_path = os.path.realpath(abs_path).replace("\\", "/")
            is_link = os.path.normpath(abs_path).lower() != os.path.normpath(real_path).lower()

            parent_dir = os.path.dirname(abs_path)
            folder_hierarchy = []

            curr_key = None
            for fk, finfo in folders_config.items():
                if os.path.normpath(finfo["path"]).lower() == os.path.normpath(parent_dir).lower():
                    curr_key = fk
                    break

            if curr_key:
                chain = []
                k = curr_key
                while k and k in folders_config:
                    chain.append(folders_config[k]["display_name"])
                    k = folders_config[k].get("parent")
                chain.reverse()
                folder_hierarchy = chain
            else:
                try:
                    rel_dir = os.path.relpath(parent_dir, BASE_OUTPUT_PATH).replace("\\", "/")
                    if rel_dir and rel_dir != ".":
                        folder_hierarchy = ["Main"] + [p for p in rel_dir.split("/") if p]
                    else:
                        folder_hierarchy = ["Main"]
                except Exception:
                    _logger.debug("handled a failure in get_file_full_details", exc_info=True)
                    folder_hierarchy = [os.path.basename(parent_dir)]

            coll_rows = conn.execute(
                """
                SELECT c.id, c.name, c.color, c.type, c.parent_id, c.is_public
                FROM collections c
                JOIN collection_files cf ON c.id = cf.collection_id
                WHERE cf.file_id = ? AND c.type = 'user_album'
                ORDER BY ulower(c.name)
            """,
                (file_id,),
            ).fetchall()

            all_colls = conn.execute("SELECT id, name, parent_id FROM collections WHERE type = 'user_album'").fetchall()
            coll_map = {c["id"]: c for c in all_colls}

            collections_list = []
            for cr in coll_rows:
                cid = cr["id"]
                chain = []
                curr_cid = cid
                visited = set()
                while curr_cid and curr_cid in coll_map and curr_cid not in visited:
                    visited.add(curr_cid)
                    chain.append(coll_map[curr_cid]["name"])
                    curr_cid = coll_map[curr_cid]["parent_id"]
                chain.reverse()

                collections_list.append(
                    {
                        "id": cid,
                        "name": cr["name"],
                        "color": cr["color"],
                        "is_public": bool(cr["is_public"]),
                        "hierarchy": chain,
                    }
                )

            cluster_wf_count = 0
            cluster_pr_count = 0
            cluster_md_count = 0
            nodes_pipeline = []
            models_used = []

            wf_h = file_data.get("workflow_hash")
            pr_h = file_data.get("prompt_hash")
            md_h = file_data.get("models_hash")
            if wf_h:
                cluster_wf_count = conn.execute(
                    "SELECT COUNT(*) FROM files WHERE workflow_hash = ?", (wf_h,)
                ).fetchone()[0]
            if pr_h:
                cluster_pr_count = conn.execute("SELECT COUNT(*) FROM files WHERE prompt_hash = ?", (pr_h,)).fetchone()[
                    0
                ]
            if md_h:
                cluster_md_count = conn.execute("SELECT COUNT(*) FROM files WHERE models_hash = ?", (md_h,)).fetchone()[
                    0
                ]

            if file_data.get("has_workflow"):
                wf_json = extract_workflow(abs_path, target_type="ui")
                if not wf_json:
                    wf_json = extract_workflow(abs_path, target_type="api")
                if wf_json:
                    try:
                        summary = generate_node_summary(wf_json)
                        if summary:
                            for n in summary:
                                nodes_pipeline.append(
                                    {
                                        "id": n.get("id"),
                                        "type": n.get("type"),
                                        "category": n.get("category"),
                                        "color": n.get("color"),
                                    }
                                )
                    except Exception:
                        _logger.debug("ignored a failure in get_file_full_details", exc_info=True)

                if file_data.get("workflow_files"):
                    for item in file_data["workflow_files"].split(" ||| "):
                        if item.strip():
                            models_used.append(os.path.basename(item.strip()))

            # This one answer carried everything the mode exists to hide:
            # the prompt and model names out of the file row, the workflow's
            # node pipeline, and the file's path on the server. Stripping the
            # picture, the album listing and the cluster views achieves
            # nothing while this hands it over on request. The panel still
            # gets its ratings, comments and collections.
            if should_strip_metadata():
                file_data = redact_file_listing([file_data])[0]
                real_path = None
                is_link = False
                nodes_pipeline = []
                models_used = []

            return jsonify(
                {
                    "status": "success",
                    "file": file_data,
                    "is_link": is_link,
                    "real_path": real_path if is_link else None,
                    "folder_hierarchy": folder_hierarchy,
                    "collections": collections_list,
                    "cluster_wf_count": cluster_wf_count,
                    "cluster_pr_count": cluster_pr_count,
                    "cluster_md_count": cluster_md_count,
                    "nodes_pipeline": nodes_pipeline,
                    "models_used": sorted(set(models_used)),
                }
            )

    except Exception as e:
        _logger.debug("handled a failure in get_file_full_details", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/file_collections/<string:file_id>")
def get_file_collections(file_id):
    """Returns a list of all collections and status flags associated with a file."""
    if not is_file_accessible(file_id):
        return jsonify({"status": "error", "message": "Access Denied"}), 403
    # Check if frontend specifically requested only public collections (Exhibition mode)
    public_only = request.args.get("public_only", "false").lower() == "true"

    query = """
        SELECT c.name, c.type, c.color, c.is_public
        FROM collections c
        JOIN collection_files cf ON c.id = cf.collection_id
        WHERE cf.file_id = ?
    """

    # Exhibition Security: Only return public user albums. Hide system flags and private albums.
    if public_only:
        query += " AND c.is_public = 1 AND c.type = 'user_album'"

    query += " ORDER BY c.type DESC, ulower(c.name) ASC"

    try:
        with get_db_connection() as conn:
            rows = conn.execute(query, (file_id,)).fetchall()

        return jsonify({"status": "success", "collections": [dict(r) for r in rows]})
    except Exception as e:
        _logger.debug("handled a failure in get_file_collections", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/collections/tag_batch", methods=["POST"])
@management_api_only
def tag_batch():
    """
    Handles batch assignment/removal of files to/from collections and status tags.
    Ensures data consistency and provides precise results for UI updates.
    """
    data = request.json
    file_ids = data.get("file_ids", [])
    collection_id = data.get("collection_id")
    action = data.get("action", "add")  # 'add', 'remove', 'toggle', 'remove_all_status'

    if not file_ids:
        return jsonify({"status": "error", "message": "No files selected"}), 400

    results_map = {}

    try:
        with get_db_connection() as conn:
            # --- CASE 1: REMOVE ALL STATUS (Shortcut '0') ---
            # This doesn't need a specific collection_id check, it targets all 'system_flag' types
            if action == "remove_all_status":
                placeholders = ",".join(["?"] * len(file_ids))
                conn.execute(
                    f"""
                    DELETE FROM collection_files
                    WHERE file_id IN ({placeholders})
                    AND collection_id IN (SELECT id FROM collections WHERE type='system_flag')
                """,
                    file_ids,
                )

                for fid in file_ids:
                    results_map[fid] = "removed"

                conn.commit()
                return jsonify({"status": "success", "results": results_map})

            # --- PRE-REQUISITE: FETCH COLLECTION TYPE ---
            if not collection_id:
                return jsonify({"status": "error", "message": "Missing collection ID"}), 400

            coll_row = conn.execute("SELECT type FROM collections WHERE id=?", (collection_id,)).fetchone()
            if not coll_row:
                return jsonify({"status": "error", "message": "Collection not found"}), 404

            coll_type = coll_row["type"]

            # --- CASE 2: SMART LOGIC FOR STATUS COLORS (system_flag) ---
            if coll_type == "system_flag" and action == "toggle":
                # NEW LOGIC:
                # If multiple files are selected, we ALWAYS 'add' (overwrite) to prevent
                # accidental desaturation of files that were already in that state.
                # If only ONE file is selected, we 'toggle' (add or remove).
                is_multiple = len(file_ids) > 1

                for fid in file_ids:
                    # Check current status for this specific file
                    exists = conn.execute(
                        "SELECT 1 FROM collection_files WHERE collection_id=? AND file_id=?", (collection_id, fid)
                    ).fetchone()

                    if exists and not is_multiple:
                        # SCENARIO A: Single file and already this color -> REMOVE
                        conn.execute(
                            "DELETE FROM collection_files WHERE collection_id=? AND file_id=?", (collection_id, fid)
                        )
                        results_map[fid] = "removed"
                    else:
                        # SCENARIO B: Multi-select OR file is not this color -> ASSIGN/OVERWRITE
                        # First, clear any OTHER system flags (mutual exclusivity)
                        conn.execute(
                            """
                            DELETE FROM collection_files
                            WHERE file_id = ?
                            AND collection_id IN (SELECT id FROM collections WHERE type='system_flag')
                        """,
                            (fid,),
                        )

                        # Add the new color
                        conn.execute(
                            "INSERT INTO collection_files (collection_id, file_id, added_at) VALUES (?, ?, ?)",
                            (collection_id, fid, time.time()),
                        )
                        results_map[fid] = "added"

            # --- CASE 3: EXPLICIT ADD/REMOVE (For User Collections/Albums) ---
            else:
                # If adding a system flag explicitly, still maintain mutual exclusivity
                if coll_type == "system_flag" and action == "add":
                    placeholders = ",".join(["?"] * len(file_ids))
                    conn.execute(
                        f"""
                        DELETE FROM collection_files
                        WHERE file_id IN ({placeholders})
                        AND collection_id IN (SELECT id FROM collections WHERE type='system_flag')
                    """,
                        file_ids,
                    )

                for fid in file_ids:
                    if action == "add":
                        try:
                            conn.execute(
                                "INSERT INTO collection_files (collection_id, file_id, added_at) VALUES (?, ?, ?)",
                                (collection_id, fid, time.time()),
                            )
                            results_map[fid] = "added"
                        except sqlite3.IntegrityError:
                            results_map[fid] = "added"  # Already exists

                    elif action == "remove":
                        conn.execute(
                            "DELETE FROM collection_files WHERE collection_id=? AND file_id=?", (collection_id, fid)
                        )
                        results_map[fid] = "removed"

                    elif action == "toggle":
                        # Generic toggle for albums (multi-assignment allowed)
                        exists = conn.execute(
                            "SELECT 1 FROM collection_files WHERE collection_id=? AND file_id=?", (collection_id, fid)
                        ).fetchone()
                        if exists:
                            conn.execute(
                                "DELETE FROM collection_files WHERE collection_id=? AND file_id=?", (collection_id, fid)
                            )
                            results_map[fid] = "removed"
                        else:
                            conn.execute(
                                "INSERT INTO collection_files (collection_id, file_id, added_at) VALUES (?, ?, ?)",
                                (collection_id, fid, time.time()),
                            )
                            results_map[fid] = "added"

            conn.commit()
            return jsonify({"status": "success", "results": results_map})

    except Exception as e:
        _logger.debug("handled a failure in tag_batch", exc_info=True)
        print(f"ERROR in tag_batch: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# Updated route to accept both integer IDs and the string "all"
@app.route("/galleryout/collection/<coll_id>")
def collection_view(coll_id):
    # AUTHENTICATION CHECK FOR EXHIBITION / FORCE_LOGIN MODES
    is_management_side = not IS_EXHIBITION_MODE
    is_logged_in = "user_id" in session
    must_authenticate = IS_EXHIBITION_MODE or FORCE_LOGIN

    if must_authenticate and not is_logged_in:
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "error", "message": "Authentication required"}), 401
        return render_template(
            "exhibition_login.html",
            app_version=APP_VERSION,
            enable_guest_login=ENABLE_GUEST_LOGIN if IS_EXHIBITION_MODE else False,
            admin_side=is_management_side,
        )

    # Decided once, because several of the query builders below read them.
    # is_privileged used to be set only inside the branch that handles a
    # PRIVATE collection, and read whenever a specific album is rendered --
    # so opening a public album, which is what exhibition mode is for,
    # reached that read with the name unbound and answered 500. `not
    # is_privileged` is evaluated as soon as exhibition mode is on, so staff
    # were locked out of albums as thoroughly as visitors.
    user_role = session.get("role", "GUEST")
    is_local_admin = not FORCE_LOGIN and not IS_EXHIBITION_MODE
    is_privileged = is_local_admin or (user_role in ["ADMIN", "MANAGER", "STAFF"])
    # Raw, not escaped: it is bound as a parameter now, and doubling the
    # quotes here would make an identity containing one fail to match.
    client_identity = _current_client_identity()

    # 1. Handle Virtual "All Categories" vs Specific Collection
    coll_info = None
    is_all_mode = coll_id == "all"

    if is_all_mode:
        # Create virtual metadata for the "All Categories" view
        coll_info = {
            "id": "all",
            "name": "All Collections",
            "type": "user_album",
            "is_public": 1,
            "color": "#ffffff",  # Default white for the virtual category
        }
    else:
        # Standard logic: Fetch specific Collection Metadata from DB
        try:
            target_id = int(coll_id)
            with get_db_connection() as conn:
                row = conn.execute("SELECT * FROM collections WHERE id=?", (target_id,)).fetchone()
                if row:
                    coll_info = dict(row)
        except ValueError:
            return redirect(url_for("gallery_view", folder_key="_root_"))

    if not coll_info:
        return redirect(url_for("gallery_view", folder_key="_root_"))

    # --- ACCESS: only public or shared content, for anyone not staff ---
    # This was written as `if IS_EXHIBITION_MODE`, so the same non-staff
    # account under --force-login skipped the check entirely and could read
    # any private album's listing. Whether a caller may see an album depends
    # on the caller.
    if not is_privileged:
        # "all" is allowed and filtered below; a specific collection must be
        # public or shared with this user.
        if not is_all_mode and coll_info["type"] == "system_flag":
            return redirect(url_for("gallery_view", folder_key="_root_"))

        if not is_all_mode and not coll_info["is_public"]:
            user_id = str(session.get("user_id", ""))
            shared_list = [u.strip() for u in str(coll_info.get("shared_users", "")).split(",") if u.strip()]

            if user_id not in shared_list:
                return redirect(url_for("gallery_view", folder_key="_root_"))

    # 2. Capture Filter Parameters
    search_term = request.args.get("search", "").strip()
    wf_files = request.args.get("workflow_files", "").strip()
    wf_prompt = request.args.get("workflow_prompt", "").strip()
    comment_search = request.args.get("comment_search", "").strip()
    start_date = request.args.get("start_date", "").strip()
    end_date = request.args.get("end_date", "").strip()
    selected_exts = request.args.getlist("extension")
    selected_prefixes = request.args.getlist("prefix")
    selected_raters = request.args.getlist("rated_by")
    selected_rating_ranges = request.args.getlist("rating_range")

    req_sort_by = request.args.get("sort_by")
    req_sort_order = request.args.get("sort_order", "desc").upper()
    if req_sort_order not in ["ASC", "DESC"]:
        req_sort_order = "DESC"

    active_filters_count = 0

    # 3. Build Dynamic Query Conditions
    conditions = [
        (
            "f.id IN (SELECT id FROM files WHERE type != 'document'"
            " AND LOWER(name) NOT LIKE '%.txt' AND LOWER(name) NOT LIKE '%.md')"
        )
    ]
    params = []

    if is_all_mode:
        # Logic for "All Categories": Select files belonging to any user album.
        # "all" must not become a way to read the albums a specific request
        # would have been refused, in either mode.
        if not is_privileged:
            visibility = "public_or_mine"
        elif IS_EXHIBITION_MODE:
            visibility = "public_or_shared"
        else:
            visibility = "any"
        scope_sql, scope_params = collections_in_scope(None, False, visibility, client_identity)
        conditions.append(f"cf.collection_id IN ({scope_sql})")
        params.extend(scope_params)
    else:
        # Sub-collections are included by default; only an explicit
        # recursive=false narrows to this collection alone (counted as a
        # filter, since it is the non-default narrowing state).
        is_recursive = request.args.get("recursive", "true").lower() != "false"
        if not is_recursive:
            active_filters_count += 1
        if is_recursive:
            # Staff see all nested collections; nobody else sees one they
            # were not given, whichever mode the server is in.
            visibility = "any" if is_privileged else "public_or_mine"
            scope_sql, scope_params = collections_in_scope(coll_id, True, visibility, client_identity)
            conditions.append(f"cf.collection_id IN ({scope_sql})")
            params.extend(scope_params)
        else:
            conditions.append("cf.collection_id = ?")
            params.append(int(coll_id))

    # --- Apply common filters ---

    if search_term:
        active_filters_count += 1
        for kw in [k.strip() for k in search_term.split(",") if k.strip()]:
            sub_kws = [s.strip() for s in kw.split(";") if s.strip()]
            if not sub_kws:
                continue

            or_conds = []
            not_conds = []
            for s in sub_kws:
                is_not = False
                if s.startswith("!"):
                    is_not = True
                    s = s[1:].strip()
                if not s:
                    continue

                if s.startswith('"') and s.endswith('"') and len(s) > 2:
                    cond_str = f"ulower(f.name) {'NOT LIKE' if is_not else 'LIKE'} ulower(?)"
                    param_val = f"%{s[1:-1]}%"
                else:
                    cond_str = f"ulower(f.name) {'NOT LIKE' if is_not else 'LIKE'} ulower(?)"
                    param_val = f"%{s}%"

                if is_not:
                    not_conds.append((cond_str, param_val))
                else:
                    or_conds.append((cond_str, param_val))

            if or_conds:
                if len(or_conds) > 1:
                    conditions.append("(" + " OR ".join([c[0] for c in or_conds]) + ")")
                elif len(or_conds) == 1:
                    conditions.append(or_conds[0][0])
                params.extend([c[1] for c in or_conds])

            for cond, param in not_conds:
                conditions.append(cond)
                params.append(param)

    if wf_files:
        active_filters_count += 1
        for kw in [k.strip() for k in wf_files.split(",") if k.strip()]:
            sub_kws = [s.strip() for s in kw.split(";") if s.strip()]
            if not sub_kws:
                continue

            or_conds = []
            not_conds = []
            for s in sub_kws:
                is_not = False
                if s.startswith("!"):
                    is_not = True
                    s = s[1:].strip()
                if not s:
                    continue

                cond_str, param_val = model_condition(s, "f.workflow_files", is_not)

                if is_not:
                    not_conds.append((cond_str, param_val))
                else:
                    or_conds.append((cond_str, param_val))

            if or_conds:
                if len(or_conds) > 1:
                    conditions.append("(" + " OR ".join([c[0] for c in or_conds]) + ")")
                elif len(or_conds) == 1:
                    conditions.append(or_conds[0][0])
                params.extend([c[1] for c in or_conds])

            for cond, param in not_conds:
                conditions.append(cond)
                params.append(param)

    if wf_prompt:
        active_filters_count += 1
        for kw in [k.strip() for k in wf_prompt.split(",") if k.strip()]:
            sub_kws = [s.strip() for s in kw.split(";") if s.strip()]
            if not sub_kws:
                continue

            or_conds = []
            not_conds = []
            for s in sub_kws:
                is_not = False
                if s.startswith("!"):
                    is_not = True
                    s = s[1:].strip()
                if not s:
                    continue

                built = prompt_search_condition(s, is_not, "f.workflow_prompt")
                if built is None:
                    continue
                cond_str, param_val = built

                if is_not:
                    not_conds.append((cond_str, param_val))
                else:
                    or_conds.append((cond_str, param_val))

            if or_conds:
                if len(or_conds) > 1:
                    conditions.append("(" + " OR ".join([c[0] for c in or_conds]) + ")")
                elif len(or_conds) == 1:
                    conditions.append(or_conds[0][0])
                params.extend([c[1] for c in or_conds])

            for cond, param in not_conds:
                conditions.append(cond)
                params.append(param)

    if comment_search:
        active_filters_count += 1
        for kw in [k.strip() for k in comment_search.split(",") if k.strip()]:
            sub_kws = [s.strip() for s in kw.split(";") if s.strip()]
            if not sub_kws:
                continue

            or_conds = []
            not_conds = []
            for s in sub_kws:
                is_not = False
                if s.startswith("!"):
                    is_not = True
                    s = s[1:].strip()
                if not s:
                    continue

                op_in = "NOT IN" if is_not else "IN"
                if s.startswith('"') and s.endswith('"') and len(s) > 2:
                    clean_s = s[1:-1]
                    col_expr = (
                        "(' ' || REPLACE(REPLACE(REPLACE(REPLACE(REPLACE("
                        "comment_text, ',', ' '), '?', ' '), '.', ' '),"
                        " '!', ' '), char(10), ' ') || ' ')"
                    )
                    cond_str = (
                        f"f.id {op_in} (SELECT file_id FROM file_comments WHERE ulower({col_expr}) LIKE ulower(?))"
                    )
                    param_val = f"% {clean_s} %"
                else:
                    cond_str = (
                        f"f.id {op_in} (SELECT file_id FROM file_comments WHERE ulower(comment_text) LIKE ulower(?))"
                    )
                    param_val = f"%{s}%"

                if is_not:
                    not_conds.append((cond_str, param_val))
                else:
                    or_conds.append((cond_str, param_val))

            if or_conds:
                if len(or_conds) > 1:
                    conditions.append("(" + " OR ".join([c[0] for c in or_conds]) + ")")
                elif len(or_conds) == 1:
                    conditions.append(or_conds[0][0])
                params.extend([c[1] for c in or_conds])

            for cond, param in not_conds:
                conditions.append(cond)
                params.append(param)

    if request.args.get("favorites") == "true":
        conditions.append("f.is_favorite = 1")
        active_filters_count += 1

    if request.args.get("no_workflow") == "true":
        conditions.append("f.has_workflow = 0")
        active_filters_count += 1

    if ENABLE_AI_SEARCH and request.args.get("no_ai_caption") == "true":
        conditions.append("(f.ai_caption IS NULL OR f.ai_caption = '')")
        active_filters_count += 1

    day_from = day_bounds(start_date, request.args.get("start_ts"), False)
    if day_from is not None:
        conditions.append("f.mtime >= ?")
        params.append(day_from)
        active_filters_count += 1
    day_to = day_bounds(end_date, request.args.get("end_ts"), True)
    if day_to is not None:
        conditions.append("f.mtime <= ?")
        params.append(day_to)
        active_filters_count += 1

    if selected_rating_ranges:
        active_filters_count += 1
        r_conds = []
        avg_sql = "IFNULL((SELECT AVG(rating) FROM file_ratings WHERE file_id = f.id), 0)"
        for rr in selected_rating_ranges:
            if rr == "0 stars":
                r_conds.append(f"{avg_sql} = 0")
            elif rr == "1 star":
                r_conds.append(f"ROUND({avg_sql}) = 1")
            elif rr == "2 stars":
                r_conds.append(f"ROUND({avg_sql}) = 2")
            elif rr == "3 stars":
                r_conds.append(f"ROUND({avg_sql}) = 3")
            elif rr == "4 stars":
                r_conds.append(f"ROUND({avg_sql}) = 4")
            elif rr == "5 stars":
                r_conds.append(f"ROUND({avg_sql}) = 5")
            # Legacy support for old URLs/bookmarks
            elif rr == "1-2 stars":
                r_conds.append(f"({avg_sql} > 0 AND {avg_sql} <= 2)")
            elif rr == "2-3 stars":
                r_conds.append(f"({avg_sql} > 2 AND {avg_sql} <= 3)")
            elif rr == "3-4 stars":
                r_conds.append(f"({avg_sql} > 3 AND {avg_sql} <= 4)")
            elif rr == "4-5 stars":
                r_conds.append(f"({avg_sql} > 4 AND {avg_sql} <= 5)")
        if r_conds:
            conditions.append(f"({' OR '.join(r_conds)})")

    if selected_raters:
        active_filters_count += 1
        expanded_raters = list(selected_raters)
        if "admin" in expanded_raters:
            try:
                # FIX: Ensure a database connection is explicitly opened to fetch the admin ID.
                # Solves the missing ID bug in "All Collections" mode where 'conn' is not yet defined.
                with get_db_connection() as temp_conn:
                    admin_id = temp_conn.execute("SELECT user_id FROM users WHERE username = 'admin'").fetchone()
                    if admin_id and str(admin_id[0]) not in expanded_raters:
                        expanded_raters.append(str(admin_id[0]))
            except sqlite3.Error:
                pass
        placeholders = ",".join(["?"] * len(expanded_raters))
        conditions.append(f"f.id IN (SELECT file_id FROM file_ratings WHERE client_uuid IN ({placeholders}))")
        params.extend(expanded_raters)

    if selected_exts:
        active_filters_count += 1
        e_cond = ["f.name LIKE ?" for e in selected_exts if e.strip()]
        params.extend([f"%.{e.lstrip('.').lower()}" for e in selected_exts if e.strip()])
        if e_cond:
            conditions.append(f"({' OR '.join(e_cond)})")

    if selected_prefixes:
        active_filters_count += 1
        p_cond = ["f.name LIKE ?" for p in selected_prefixes if p.strip()]
        params.extend([f"{p.strip()}_%" for p in selected_prefixes if p.strip()])
        if p_cond:
            conditions.append(f"({' OR '.join(p_cond)})")

    # --- SORTING LOGIC ---
    safe_uuid = _current_client_identity().replace("'", "''")
    if req_sort_by == "name":
        order_clause = f"ulower(f.name) {req_sort_order}"
    elif req_sort_by == "rating":
        if is_effectively_blind():
            conditions.append(f"f.id IN (SELECT file_id FROM file_ratings WHERE client_uuid = '{safe_uuid}')")

            order_clause = f"my_rating {req_sort_order}, f.mtime DESC"

        else:
            conditions.append("f.id IN (SELECT file_id FROM file_ratings)")

            order_clause = f"avg_rating {req_sort_order}, f.mtime DESC"
    elif req_sort_by == "unrated":
        if is_effectively_blind():
            conditions.append(f"f.id NOT IN (SELECT file_id FROM file_ratings WHERE client_uuid = '{safe_uuid}')")
        else:
            conditions.append("f.id NOT IN (SELECT file_id FROM file_ratings)")
        order_clause = f"f.mtime {req_sort_order}"
    elif req_sort_by == "uncommented":
        if is_effectively_blind():
            conditions.append(f"f.id NOT IN (SELECT file_id FROM file_comments WHERE client_uuid = '{safe_uuid}')")
        else:
            conditions.append("f.id NOT IN (SELECT file_id FROM file_comments)")
        order_clause = f"f.mtime {req_sort_order}"
    elif req_sort_by in ["comments", "latest_comment", "latestcomment"]:
        conditions.append("f.id IN (SELECT file_id FROM file_comments)")
        if req_sort_by == "comments":
            order_clause = f"comment_count {req_sort_order}, f.mtime DESC"
        else:
            order_clause = f"latest_comment_time {req_sort_order}, f.mtime DESC"
    elif req_sort_by == "size":
        order_clause = f"f.size {req_sort_order}"
    elif req_sort_by == "type":
        order_clause = f"f.type {req_sort_order}, ulower(f.name) ASC"
    elif req_sort_by == "duration":
        order_clause = f"f.duration {req_sort_order}"
    elif req_sort_by == "dimensions":
        order_clause = f"{MEGAPIXELS_SQL} {req_sort_order}"
    elif req_sort_by in {"date", "mtime"}:
        order_clause = f"f.mtime {req_sort_order}"
    else:
        order_clause = "f.mtime DESC"

    final_files = []
    total_db_files = 0
    total_folder_files = 0

    with get_db_connection() as conn:
        # Calculate total files in this view (without search/filters)
        if is_all_mode:
            if not is_privileged:
                visibility = "public_or_mine"
            elif IS_EXHIBITION_MODE:
                visibility = "public_or_shared"
            else:
                visibility = "any"
            scope_sql, scope_params = collections_in_scope(None, False, visibility, client_identity)
            total_folder_files = conn.execute(
                f"SELECT COUNT(DISTINCT file_id) FROM collection_files WHERE collection_id IN ({scope_sql})",
                scope_params,
            ).fetchone()[0]
        elif is_recursive:
            sub_query = f"""
                    WITH RECURSIVE children AS (
                        SELECT id FROM collections WHERE id = {int(coll_id)}
                        UNION ALL
                        SELECT c.id FROM collections c INNER JOIN children p ON c.parent_id = p.id
                    )
                    SELECT id FROM children
                """
            total_folder_files = conn.execute(
                f"SELECT COUNT(DISTINCT file_id) FROM collection_files WHERE collection_id IN ({sub_query})"
            ).fetchone()[0]
        else:
            total_folder_files = conn.execute(
                "SELECT COUNT(*) FROM collection_files WHERE collection_id = ?", (int(coll_id),)
            ).fetchone()[0]

        with contextlib.suppress(sqlite3.Error):
            total_db_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

        where_clause = " AND ".join(conditions)

        # We use DISTINCT to avoid showing the same file twice if it's in multiple albums
        user_role = session.get("role", "GUEST")
        safe_uuid = _current_client_identity().replace("'", "''")

        # Allow Local Admin (no force login) to see all comments during sort
        is_local_admin = not FORCE_LOGIN and not IS_EXHIBITION_MODE

        if is_local_admin or user_role in ["ADMIN", "MANAGER", "STAFF"]:
            comment_sub_filter = ""
        else:
            comment_sub_filter = (
                f" AND (target_audience = 'public'"
                f" OR target_audience = 'user:{safe_uuid}'"
                f" OR client_uuid = '{safe_uuid}')"
            )

        query = f"""
            SELECT DISTINCT f.*,
            (SELECT c.color FROM collections c
                         JOIN collection_files cf2 ON c.id = cf2.collection_id
                         WHERE cf2.file_id = f.id AND c.type = 'system_flag' LIMIT 1) as status_color,
            (SELECT AVG(rating) FROM file_ratings WHERE file_id = f.id) as avg_rating,
            (SELECT COUNT(*) FROM file_ratings WHERE file_id = f.id) as vote_count,
            (SELECT rating FROM file_ratings
                         WHERE file_id = f.id AND client_uuid = '{safe_uuid}') as my_rating,
            (SELECT COUNT(*) FROM file_comments WHERE file_id = f.id {comment_sub_filter}) as comment_count,
            (SELECT MAX(created_at) FROM file_comments
                         WHERE file_id = f.id {comment_sub_filter}) as latest_comment_time
            FROM files f
            JOIN collection_files cf ON f.id = cf.file_id
            WHERE {where_clause}
            ORDER BY {order_clause}
        """

        rows = conn.execute(query, params).fetchall()

        for r in rows:
            d = dict(r)
            d.pop("ai_embedding", None)
            final_files.append(d)

        try:
            users_rows = conn.execute(
                "SELECT user_id, full_name FROM users WHERE is_active=1 AND username != 'admin'"
            ).fetchall()
            available_raters = [{"id": str(r["user_id"]), "name": r["full_name"]} for r in users_rows]
        except sqlite3.Error:
            available_raters = []
        available_raters.insert(0, {"id": "admin", "name": "System Admin"})

    # --- CLUSTER MODE OVERRIDE LOGIC & SCOPE SEARCH ---
    # Exhibition mode ships no clustering UI; ignore crafted cluster URLs.
    cluster_mode = None if IS_EXHIBITION_MODE else request.args.get("cluster_mode")
    cluster_sort = request.args.get("cluster_sort", "date_desc")
    cluster_target_id = request.args.get("cluster_target_id")
    cluster_scope = request.args.get("cluster_scope", "global")

    final_files = process_clustering(final_files, cluster_mode, cluster_sort, cluster_target_id, cluster_scope)

    fake_folder_key = f"collection_{coll_id}"

    # Fetch notes dynamically for the current collection
    has_notes = False
    note_files = []
    try:
        with get_db_connection() as conn_notes:
            query = """
                SELECT DISTINCT f.id, f.name, f.path, f.mtime, f.type
                FROM files f
                JOIN collection_files cf ON f.id = cf.file_id
                WHERE (cf.collection_id = ? OR ? = 'all')
                AND (f.type = 'document' OR LOWER(f.name) LIKE '%.txt' OR LOWER(f.name) LIKE '%.md')
                ORDER BY f.mtime DESC
            """
            rows = conn_notes.execute(query, (coll_id if coll_id != "all" else -1, coll_id)).fetchall()
            note_files = [dict(r) for r in rows]
            has_notes = len(note_files) > 0
    except Exception:
        _logger.debug("ignored a failure in collection_view", exc_info=True)

    # Standard metadata extraction for UI filters
    extensions = set()
    prefixes = set()
    prefix_limit_reached = False

    # Extract all extensions independently from filters to populate the dropdowns fully
    with get_db_connection() as conn_ext:
        if not is_all_mode:
            visibility = "any" if is_privileged else "public_or_mine"
            ext_rows = collection_file_names(conn_ext, coll_id, is_recursive, visibility, client_identity)
        else:
            if not is_privileged:
                visibility = "public_or_mine"
            elif IS_EXHIBITION_MODE:
                visibility = "public_or_shared"
            else:
                visibility = "any"
            # No collection id, so there is nothing to recurse into --
            # is_recursive is not even bound on this branch.
            ext_rows = collection_file_names(conn_ext, None, False, visibility, client_identity)
        for r in ext_rows:
            fname = r["name"]
            if "." in fname:
                ext_clean = fname.split(".")[-1].lower()
                if ext_clean not in ["txt", "md"]:
                    extensions.add(ext_clean)
            if not prefix_limit_reached and "_" in fname:
                pfx = fname.split("_")[0]
                if pfx:
                    prefixes.add(pfx)
                    if len(prefixes) > MAX_PREFIX_DROPDOWN_ITEMS:
                        prefix_limit_reached = True
                        prefixes.clear()

    # --- JSON RESPONSE FOR AJAX/EXHIBITION ---
    if request.headers.get("Accept") == "application/json":
        return jsonify(
            {
                "status": "success",
                "collection_name": coll_info["name"],
                "files": redact_file_listing(final_files),
                "total_count": total_folder_files,
                "has_notes": has_notes,
                "note_files": note_files,
                "available_extensions": sorted(extensions),
            }
        )

    # --- TEMPLATE RENDERING ---
    is_system_flag = coll_info.get("type") == "system_flag"
    parent_name = "Status" if is_system_flag else "Collections"

    breadcrumbs = []
    if not IS_EXHIBITION_MODE:
        breadcrumbs = [
            {"key": "_root_", "display_name": "Main"},
            {"key": None, "display_name": parent_name},
            {"key": fake_folder_key, "display_name": coll_info["name"]},
        ]
    else:
        breadcrumbs = [
            {"key": "_root_", "display_name": "Exhibition Home"},
            {"key": fake_folder_key, "display_name": coll_info["name"]},
        ]

    current_folder_info = {
        "display_name": coll_info["name"],
        "path": f"{parent_name}: {coll_info['name']}",
        "is_watched": False,
        "is_mount": False,
        "is_collection": True,
        "collection_id": coll_id,  # Can be 'all' or int
        "collection_color": coll_info.get("color", "#ffffff"),
        "collection_type": coll_info.get("type", "user_album"),
    }

    folders = get_dynamic_folder_config()

    try:
        with get_db_connection() as conn_opts:
            users_rows = conn_opts.execute(
                "SELECT user_id, full_name FROM users WHERE is_active=1 AND username != 'admin'"
            ).fetchall()
            available_raters = [{"id": str(r["user_id"]), "name": r["full_name"]} for r in users_rows]
    except sqlite3.Error:
        available_raters = []
    available_raters.insert(0, {"id": "admin", "name": "System Admin"})
    template_name = "exhibition.html" if IS_EXHIBITION_MODE else "index.html"

    return render_template(
        template_name,
        files=final_files[:PAGE_SIZE],
        total_files=len(final_files),
        view_token=VIEW_SNAPSHOTS.put(_view_owner(), final_files),
        total_folder_files=total_folder_files,
        total_db_files=total_db_files,
        folders=folders,
        current_folder_key=fake_folder_key,
        current_folder_info=current_folder_info,
        breadcrumbs=breadcrumbs,
        ancestor_keys=[],
        available_extensions=sorted(extensions),
        available_prefixes=sorted(prefixes),
        prefix_limit_reached=prefix_limit_reached,
        selected_extensions=selected_exts,
        selected_prefixes=selected_prefixes,
        available_raters=available_raters,
        selected_raters=selected_raters,
        selected_rating_ranges=selected_rating_ranges,
        protected_folder_keys=list(PROTECTED_FOLDER_KEYS),
        show_favorites=request.args.get("favorites", "false").lower() == "true",
        generate_waveforms=GENERATE_WAVEFORMS,
        enable_ai_search=ENABLE_AI_SEARCH,
        enable_ai_dam=AI_CONFIG.enabled,
        is_ai_search=False,
        ai_query="",
        is_omniquery=False,
        omniquery_sql="",
        omniquery_dictionary=get_omniquery_dictionary(),
        is_global_search=False,
        active_filters_count=active_filters_count,
        current_scope="local",
        is_recursive=True,
        server_dam_default=ENABLE_DAM_MODE,
        is_exhibition_mode=IS_EXHIBITION_MODE,
        blind_rating=is_effectively_blind(),
        global_blind_active=BLIND_RATING,
        app_version=APP_VERSION,
        github_url=GITHUB_REPO_URL,
        update_available=UPDATE_AVAILABLE,
        remote_version=REMOTE_VERSION,
        ffmpeg_available=(FFPROBE_EXECUTABLE_PATH is not None),
        comfy_server_url=COMFYUI_SERVER_URL,
        stream_threshold=STREAM_THRESHOLD_BYTES,
        has_notes=has_notes,
        note_files=note_files,
    )


# --- EXHIBITION API: RATINGS & COMMENTS ---


@app.route("/galleryout/api/exhibition/rating_details", methods=["GET"])
@management_api_only
def get_rating_details():
    file_id = request.args.get("file_id")
    if not file_id:
        return jsonify({"status": "error", "message": "Missing file ID"}), 400

    try:
        with get_db_connection() as conn:
            # Join ratings with users to get real names
            query = """
                SELECT r.rating, r.client_uuid, u.full_name
                FROM file_ratings r
                LEFT JOIN users u ON r.client_uuid = CAST(u.user_id AS TEXT)
                WHERE r.file_id = ?
                ORDER BY r.rating DESC, r.created_at DESC
            """
            rows = conn.execute(query, (file_id,)).fetchall()

            details = []
            for row in rows:
                name = "Guest (Anonymous)"
                if row["client_uuid"] == "admin":
                    name = "System Admin"
                elif row["full_name"]:
                    name = row["full_name"]

                details.append({"rating": row["rating"], "name": name})

            return jsonify({"status": "success", "details": details})
    except Exception as e:
        _logger.debug("handled a failure in get_rating_details", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/exhibition/rate", methods=["POST"])
def exhibition_rate_file():
    if (IS_EXHIBITION_MODE or FORCE_LOGIN) and not session.get("user_id"):
        return jsonify({"status": "error", "message": "Authentication required"}), 401
    data = request.json
    file_id = data.get("file_id")

    current_user_id = session.get("user_id")
    # Spoofing Protection: Force server-side ID if authenticated
    client_uuid = str(current_user_id) if current_user_id else data.get("client_uuid")

    rating = data.get("rating")  # 1-5 integer, or None/0 to delete

    if not all([file_id, client_uuid]):
        return jsonify({"status": "error", "message": "Missing data"}), 400

    rating, rating_error = parse_rating(rating)
    if rating_error:
        return jsonify({"status": "error", "message": rating_error}), 400

    # Rating asked only whether the file existed, never whether this caller
    # was allowed to see it -- so a visitor could score a picture that was
    # never shared with them, and the owner's average for a private
    # picture was set by someone who had never laid eyes on it. The reply
    # is the same as for a file that is not there at all, so that the two
    # cannot be told apart: answering 404 for one and 200 for the other let
    # a visitor discover which ids exist.
    if not is_file_accessible(file_id):
        return jsonify({"status": "error", "message": "File not found"}), 404

    try:
        with get_db_connection() as conn:
            if not conn.execute("SELECT 1 FROM files WHERE id=?", (file_id,)).fetchone():
                return jsonify({"status": "error", "message": "File not found"}), 404

            if rating is None or rating == 0:
                conn.execute(
                    """
                    DELETE FROM file_ratings
                    WHERE file_id = ? AND client_uuid = ?
                """,
                    (file_id, client_uuid),
                )
                conn.commit()
            else:
                conn.execute(
                    """
                    INSERT INTO file_ratings (file_id, client_uuid, rating, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(file_id, client_uuid) DO UPDATE SET
                        rating = excluded.rating,
                        created_at = excluded.created_at
                """,
                    (file_id, client_uuid, rating, time.time()),
                )
                conn.commit()

            result = conn.execute(
                """
                SELECT AVG(rating), COUNT(*)
                FROM file_ratings
                WHERE file_id=?
            """,
                (file_id,),
            ).fetchone()

            avg = result[0] if result[0] is not None else 0.0
            vote_count = result[1] if result[1] is not None else 0

        # Blind rating exists so a rater is not anchored by the crowd. The
        # interface honoured that and the reply did not: the average came back
        # in the JSON on every vote, where anyone with the network tab open
        # could read it. A guarantee that holds only in the markup is not one.
        if is_effectively_blind():
            return jsonify({"status": "success", "new_average": None, "vote_count": None})

        return jsonify({"status": "success", "new_average": avg, "vote_count": vote_count})
    except Exception as e:
        _logger.debug("handled a failure in exhibition_rate_file", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/exhibition/rate_batch", methods=["POST"])
def exhibition_rate_batch():
    # The single-file route has always checked this; the batch one did not,
    # so an unauthenticated caller could write or clear ratings in bulk under
    # any identity it named.
    if (IS_EXHIBITION_MODE or FORCE_LOGIN) and not session.get("user_id"):
        return jsonify({"status": "error", "message": "Authentication required"}), 401
    data = request.json
    file_ids = data.get("file_ids", [])

    current_user_id = session.get("user_id")
    # Spoofing Protection: Force server-side ID if authenticated
    client_uuid = str(current_user_id) if current_user_id else data.get("client_uuid")

    rating = data.get("rating")

    if not file_ids or not client_uuid:
        return jsonify({"status": "error", "message": "Missing data"}), 400

    rating, rating_error = parse_rating(rating)
    if rating_error:
        return jsonify({"status": "error", "message": rating_error}), 400

    # The same rule as the single-file route above, and for the same
    # reason. Also the reason this route no longer answers with a raw
    # database error: it used to hand whatever ids it was given straight to
    # the insert, so one that did not exist came back as a 500 quoting
    # "FOREIGN KEY constraint failed".
    allowed = [fid for fid in file_ids if is_file_accessible(fid)]
    if not allowed:
        return jsonify({"status": "error", "message": "File not found"}), 404

    try:
        with get_db_connection() as conn:
            placeholders = ",".join(["?"] * len(allowed))
            present = {
                row[0] for row in conn.execute(f"SELECT id FROM files WHERE id IN ({placeholders})", allowed).fetchall()
            }
            file_ids = [fid for fid in allowed if fid in present]
            if not file_ids:
                return jsonify({"status": "error", "message": "File not found"}), 404

            if rating is None or rating == 0:
                placeholders = ",".join(["?"] * len(file_ids))
                query = f"""
                    DELETE FROM file_ratings
                    WHERE file_id IN ({placeholders}) AND client_uuid = ?
                """
                params = [*file_ids, client_uuid]
                conn.execute(query, params)
            else:
                current_time = time.time()
                records = [(fid, client_uuid, rating, current_time) for fid in file_ids]

                conn.executemany(
                    """
                    INSERT INTO file_ratings (file_id, client_uuid, rating, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(file_id, client_uuid) DO UPDATE SET
                        rating = excluded.rating,
                        created_at = excluded.created_at
                """,
                    records,
                )

            conn.commit()

        return jsonify({"status": "success", "message": f"Successfully updated {len(file_ids)} files."})

    except Exception as e:
        _logger.debug("handled a failure in exhibition_rate_batch", exc_info=True)
        print(f"Batch Rating Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/exhibition/comments", methods=["GET"])
def exhibition_get_comments():
    if (IS_EXHIBITION_MODE or FORCE_LOGIN) and not session.get("user_id"):
        return jsonify({"status": "error", "message": "Authentication required"}), 401
    file_id = request.args.get("file_id")
    current_user_id = session.get("user_id")
    current_role = session.get("role", "GUEST")
    client_uuid = str(current_user_id) if current_user_id else request.args.get("client_uuid", "")

    if not file_id:
        return jsonify({"status": "error", "message": "File ID missing"}), 400

    # Reading comments asked only for a session, never whether this caller
    # may see the picture the comments are about -- so a visitor given one
    # album could read the owner's notes on every other picture in the
    # library, including ones the gallery refuses to send them. The answer
    # matches a file that is not there, so the two cannot be told apart.
    if not is_file_accessible(file_id):
        return jsonify({"status": "error", "message": "File not found"}), 404

    with get_db_connection() as conn:
        # --- FIX: LOCAL ADMIN EQUIVALENCE ---
        # If FORCE_LOGIN is False and we are in the main interface, the user is implicitly Admin
        is_local_admin = not FORCE_LOGIN and not IS_EXHIBITION_MODE
        is_privileged = is_local_admin or (current_role in ["ADMIN", "MANAGER", "STAFF"])

        if is_privileged:
            # Admins, Managers, and Staff see EVERYTHING
            query = """
                SELECT fc.*, u.full_name as target_user_name
                FROM file_comments fc
                LEFT JOIN users u ON fc.target_audience = 'user:' || u.user_id
                WHERE fc.file_id=? ORDER BY fc.created_at DESC
            """
            params = (file_id,)
        else:
            # Regular users (GUEST, CUSTOMER, FRIEND, USER) see only:
            # 1. Public comments
            # 2. Comments specifically directed to their UUID/User_ID
            # 3. Comments authored by themselves
            query = """
                SELECT fc.*, u.full_name as target_user_name
                FROM file_comments fc
                LEFT JOIN users u ON fc.target_audience = 'user:' || u.user_id
                WHERE fc.file_id=?
                AND (
                    fc.target_audience = 'public'
                    OR fc.target_audience = ?
                    OR fc.client_uuid = ?
                )
                ORDER BY fc.created_at DESC
            """
            params = (file_id, f"user:{client_uuid}", client_uuid)

        comments = conn.execute(query, params).fetchall()

        # 2. PERSONAL RATING
        my_rating = 0
        if client_uuid:
            r = conn.execute(
                "SELECT rating FROM file_ratings WHERE file_id=? AND client_uuid=?", (file_id, client_uuid)
            ).fetchone()
            if r:
                my_rating = r["rating"]

        # 3. GLOBAL STATS (Fresh Calculation for Real-Time Polling)
        stats = conn.execute("SELECT AVG(rating), COUNT(*) FROM file_ratings WHERE file_id=?", (file_id,)).fetchone()
        avg_rating = stats[0] if stats[0] is not None else 0.0
        vote_count = stats[1] if stats[1] is not None else 0

    return jsonify(
        {
            "status": "success",
            "comments": [dict(c) for c in comments],
            "my_rating": my_rating,
            # Send fresh stats to frontend
            "avg_rating": avg_rating,
            "vote_count": vote_count,
        }
    )


@app.route("/galleryout/api/users/simple_list", methods=["GET"])
def get_users_simple_list():
    # --- FIX: LOCAL ADMIN EQUIVALENCE ---
    is_local_admin = not FORCE_LOGIN and not IS_EXHIBITION_MODE
    if not is_local_admin and session.get("role") not in ["ADMIN", "MANAGER", "STAFF"]:
        return jsonify({"status": "error", "message": "Unauthorized"}), 403

    exclude_staff = request.args.get("exclude_staff", "false").lower() == "true"
    try:
        with get_db_connection() as conn:
            query = "SELECT user_id, full_name, username FROM users WHERE is_active = 1 AND username != 'admin'"
            if exclude_staff:
                query += " AND role NOT IN ('ADMIN', 'MANAGER', 'STAFF')"
            query += " ORDER BY full_name ASC"
            rows = conn.execute(query).fetchall()
            return jsonify({"status": "success", "users": [dict(r) for r in rows]})
    except Exception as e:
        _logger.debug("handled a failure in get_users_simple_list", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/exhibition/post_comment", methods=["POST"])
def exhibition_post_comment():
    if (IS_EXHIBITION_MODE or FORCE_LOGIN) and not session.get("user_id"):
        return jsonify({"status": "error", "message": "Authentication required"}), 401
    data = request.json
    file_id = data.get("file_id")
    text = data.get("text", "").strip()

    # Nothing bounded this. Measured: a 40,000,000 character comment was
    # accepted with a 200 and stored whole, and four posts put 45 MB into
    # the database. In Exhibition mode -- the portal built to be handed to
    # family, friends and clients, optionally with guest login -- that is
    # a visitor, and whatever they send is then read back to everyone who
    # opens that picture and rendered in their browser.
    #
    # Refused rather than truncated: quietly keeping the first few
    # thousand characters of what somebody wrote is worse than telling
    # them it was too long. Long text about a collection has a place
    # already, which is the collection note.
    too_long = comment_length_error(text)
    if too_long:
        return jsonify({"status": "error", "message": too_long}), 400

    # The same for the name shown beside it. Only a real visitor can set
    # this -- a signed-in user gets their account name and the local owner
    # gets "System Admin" -- so it is exactly the unauthenticated path.
    if len(str(data.get("author", ""))) > COMMENT_AUTHOR_MAX_CHARS:
        return jsonify(
            {
                "status": "error",
                "message": (f"That name is too long. The limit is {COMMENT_AUTHOR_MAX_CHARS} characters."),
            }
        ), 400

    # The same rule as reading them: a visitor may only write about a
    # picture that was actually shared with them. Without this, a comment
    # could be attached to any picture in the library by anyone who had
    # been given a single album.
    if file_id and not is_file_accessible(file_id):
        return jsonify({"status": "error", "message": "File not found"}), 404

    target_audience = data.get("target_audience", "public").strip()
    if not target_audience:
        # Se sono un Admin "locale" (non force_login), default = internal
        target_audience = "internal" if not FORCE_LOGIN and not IS_EXHIBITION_MODE else "public"
    # Get User Context from Session
    user_id = session.get("user_id")
    role = session.get("role", "GUEST")
    real_full_name = session.get("full_name", "Guest")

    # --- FIX: LOCAL ADMIN EQUIVALENCE ---
    is_local_admin = not FORCE_LOGIN and not IS_EXHIBITION_MODE
    is_privileged = is_local_admin or (role in ["ADMIN", "MANAGER", "STAFF"])

    # Security: Non-privileged users can ONLY post 'public' or 'internal' (Staff Only).
    # They cannot DM specific users (e.g., 'user:123').
    if not is_privileged and target_audience not in ["public", "internal"]:
        target_audience = "public"

    client_uuid = str(user_id) if user_id else data.get("client_uuid")

    if role != "GUEST" and user_id:
        author = real_full_name
    elif is_local_admin:
        # If we are the Local Admin (no login), override the author name to System Admin
        # and force the UUID to 'admin' so the UI highlights it properly with the shield 🛡️
        author = "System Admin"
        client_uuid = "admin"
    else:
        author = data.get("author", "Guest").strip()

    if not all([file_id, client_uuid, text]):
        return jsonify({"status": "error", "message": "Missing data"}), 400

    try:
        with get_db_connection() as conn:
            # --- SECURITY CHECK: Ensure target user actually exists ---
            if target_audience.startswith("user:"):
                target_user_id = target_audience.split(":")[1]
                # We check if it's a registered user ID (guests don't have integer IDs)
                if target_user_id.isdigit():
                    user_exists = conn.execute("SELECT 1 FROM users WHERE user_id = ?", (target_user_id,)).fetchone()
                    if not user_exists:
                        return jsonify(
                            {"status": "error", "message": "Target user has been deleted or does not exist."}
                        ), 404

            conn.execute(
                """
                INSERT INTO file_comments (file_id, client_uuid, author_name, comment_text, target_audience, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (file_id, client_uuid, author, text, target_audience, time.time()),
            )
            conn.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        _logger.debug("handled a failure in exhibition_post_comment", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/exhibition/delete_comment", methods=["POST"])
def exhibition_delete_comment():
    # Ownership below is decided by comparing the row's client_uuid against
    # one taken from the request body, which is only meaningful once we know
    # a caller. Without this, an anonymous request supplied both halves --
    # and comment ids are AUTOINCREMENT integers, so they can be counted
    # through until one matches the identity being claimed.
    if (IS_EXHIBITION_MODE or FORCE_LOGIN) and not session.get("user_id"):
        return jsonify({"status": "error", "message": "Authentication required"}), 401
    data = request.json
    comment_id = data.get("comment_id")
    current_user_id = session.get("user_id")
    current_role = session.get("role")

    client_uuid = str(current_user_id) if current_user_id else data.get("client_uuid")

    is_local_admin = (not FORCE_LOGIN and not IS_EXHIBITION_MODE) and not current_user_id
    is_privileged = is_local_admin or (current_role in ["ADMIN", "MANAGER", "STAFF"])

    try:
        with get_db_connection() as conn:
            if not is_privileged:
                if not client_uuid:
                    return jsonify({"status": "error", "message": "Auth required"}), 403
                res = conn.execute("DELETE FROM file_comments WHERE id=? AND client_uuid=?", (comment_id, client_uuid))
                if res.rowcount == 0:
                    return jsonify({"status": "error", "message": "Permission denied: Not your comment."}), 403
            else:
                conn.execute("DELETE FROM file_comments WHERE id=?", (comment_id,))

            conn.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        _logger.debug("handled a failure in exhibition_delete_comment", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/exhibition/edit_comment", methods=["POST"])
def exhibition_edit_comment():
    # Same as delete: ownership is checked against an identity supplied by
    # the caller, so there has to be a caller.
    if (IS_EXHIBITION_MODE or FORCE_LOGIN) and not session.get("user_id"):
        return jsonify({"status": "error", "message": "Authentication required"}), 401
    data = request.json
    comment_id = data.get("comment_id")
    new_text = data.get("new_text", "").strip()

    current_user_id = session.get("user_id")
    current_role = session.get("role")
    client_uuid = str(current_user_id) if current_user_id else data.get("client_uuid")

    if not all([comment_id, client_uuid, new_text]):
        return jsonify({"status": "error", "message": "Missing data"}), 400

    # The same rule as posting one. Without this the cap on posting was a
    # cap on nothing: post something short, then edit it to any size.
    too_long = comment_length_error(new_text)
    if too_long:
        return jsonify({"status": "error", "message": too_long}), 400

    is_local_admin = (not FORCE_LOGIN and not IS_EXHIBITION_MODE) and not current_user_id
    is_privileged = is_local_admin or (current_role in ["ADMIN", "MANAGER", "STAFF"])

    try:
        with get_db_connection() as conn:
            if not is_privileged:
                res = conn.execute(
                    """
                    UPDATE file_comments
                    SET comment_text = ?
                    WHERE id = ? AND client_uuid = ?
                """,
                    (new_text, comment_id, client_uuid),
                )

                if res.rowcount == 0:
                    return jsonify(
                        {"status": "error", "message": "Permission denied: Cannot edit this comment (Not owner)"}
                    ), 403
            else:
                conn.execute(
                    """
                    UPDATE file_comments
                    SET comment_text = ?
                    WHERE id = ?
                """,
                    (new_text, comment_id),
                )

            conn.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        _logger.debug("handled a failure in exhibition_edit_comment", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# --- OMNIQUERY API ---
@app.route("/galleryout/api/omniquery/execute", methods=["POST"])
@management_api_only
def execute_omniquery():
    data = request.json
    raw_sql = data.get("sql", "").strip()
    raw_sql = raw_sql.replace("\r\n", "\n").replace("\n", " \n")
    if not raw_sql:
        return jsonify({"status": "error", "message": "SQL query cannot be empty."}), 400

    # The ONE sandboxed SQL gate (omniquery.sqlexec): SELECT-prefix check,
    # true read-only URI connection, and the C-engine authorizer. The
    # nl2sql model's generated queries run through this exact function too.

    session_id = str(uuid.uuid4())

    exec_result = run_readonly_select(DATABASE_FILE, raw_sql)
    if not exec_result.ok:
        status = 403 if "only SELECT" in (exec_result.error or "") else 400
        return jsonify({"status": "error", "message": exec_result.error}), status
    result_ids = exec_result.ids

    if not result_ids:
        return jsonify(
            {"status": "success", "session_id": None, "message": "Query executed successfully, but returned 0 results."}
        )

    # Save results to main Read-Write connection
    try:
        with get_db_connection() as rw_conn:
            # Housekeeping: delete sessions older than 2 hours
            rw_conn.execute("DELETE FROM omniquery_sessions WHERE created_at < ?", (time.time() - 7200,))

            rw_conn.execute(
                "INSERT INTO omniquery_sessions (session_id, raw_sql, created_at) VALUES (?, ?, ?)",
                (session_id, raw_sql, time.time()),
            )

            records = [(session_id, fid) for fid in result_ids]
            rw_conn.executemany("INSERT INTO omniquery_results (session_id, file_id) VALUES (?, ?)", records)
            rw_conn.commit()

        return jsonify({"status": "success", "session_id": session_id, "count": len(result_ids)})
    except Exception as e:
        _logger.debug("handled a failure in execute_omniquery", exc_info=True)
        return jsonify({"status": "error", "message": f"Database Error: {e!s}"}), 500


# --- OMNIQUERY SEARCH (the LLM pretending to be a search field) ---
# Fusion of two answerers, each used where it is strong:
#   - the deterministic nlq parser (sub-ms): exact for queries whose every
#     term its rules consume, and the instant live-typing path
#   - the nl2sql model (SqlSearch): natural language -> SQL, run in an
#     agentic generate/execute/read-results loop -- for queries that carry
#     free language. Its SQL executes ONLY through the same sandbox as the
#     manual endpoint (omniquery.sqlexec).
# Model failure falls back to the deterministic answer; no response ever
# carries SQL or an AST.
_omniquery_parser = None
_omniquery_sqlsearch = None
_omniquery_lock = threading.Lock()


def _get_omniquery_parser():
    """Lazy singleton for the deterministic nlq parser (zero-dependency,
    always available)."""
    global _omniquery_parser
    if _omniquery_parser is None:
        with _omniquery_lock:
            if _omniquery_parser is None:
                import omniquery.parsers as omniquery_parsers

                _omniquery_parser = omniquery_parsers.make_search_parser()
    return _omniquery_parser


def _get_omniquery_sqlsearch():
    """Lazy singleton for the nl2sql model search. Constructing never
    loads the model; available() is re-checked per request so weights
    provisioned mid-session start working without a restart."""
    global _omniquery_sqlsearch
    if _omniquery_sqlsearch is None:
        with _omniquery_lock:
            if _omniquery_sqlsearch is None:
                from omniquery.parsers.nl2sql import SqlSearch

                _omniquery_sqlsearch = SqlSearch(db_path=DATABASE_FILE)
    return _omniquery_sqlsearch


def _omniquery_run_ast(ast_dict, query_text):
    """Execute a validated AST through the typed engine; returns
    (ids_or_None, count_or_None, error)."""

    # Same role-derivation formula used elsewhere for the local/no-auth
    # admin case (see is_effectively_blind / management_api_only): when
    # neither FORCE_LOGIN nor exhibition mode is active, an unauthenticated
    # session is the local admin; otherwise it's an unauthenticated guest.
    role = session.get("role", "ADMIN" if not (FORCE_LOGIN or IS_EXHIBITION_MODE) else "GUEST")
    user_id = session.get("user_id")
    ctx = AuthContext(role=role, user_id=user_id, client_uuid=user_id, ai_enabled=AI_CONFIG.enabled)
    engine = OmniQueryEngine(
        DATABASE_FILE,
        BASE_OUTPUT_PATH,
        ai_resolvers=ai_dam_service.create_ai_resolvers(AI_CONFIG),
    )
    try:
        result = engine.run(ast_dict, ctx)
    except Exception as e:
        _logger.debug("handled a failure in _omniquery_run_ast", exc_info=True)
        return None, None, str(e)
    if not result.ok:
        return None, None, result.error
    if result.kind == "count":
        return None, result.count, None
    return (result.ids or []), None, None


def _omniquery_store_session(query_text, result_ids, bookkeeping_sql):
    """Persist a result set as an omniquery session for gallery
    navigation; returns the session id."""
    session_id = str(uuid.uuid4())
    safe_nl_text = query_text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    raw_sql = f"-- OmniQuery: {safe_nl_text}\n{bookkeeping_sql}"
    with get_db_connection() as rw_conn:
        rw_conn.execute("DELETE FROM omniquery_sessions WHERE created_at < ?", (time.time() - 7200,))
        rw_conn.execute(
            "INSERT INTO omniquery_sessions (session_id, raw_sql, created_at) VALUES (?, ?, ?)",
            (session_id, raw_sql, time.time()),
        )
        rw_conn.executemany(
            "INSERT INTO omniquery_results (session_id, file_id) VALUES (?, ?)",
            [(session_id, fid) for fid in result_ids],
        )
        rw_conn.commit()
    return session_id


@app.route("/galleryout/api/omniquery/nlq", methods=["POST"])
@management_api_only
def omniquery_nlq():
    """The search palette's engine: question in, tiles out.

      live=true  -- keystroke path: deterministic parse only (sub-ms, no
                    model, no DB writes); instant approximate tiles.
      live=false -- the real answer: rules when they consume the whole
                    query (exact), else the nl2sql model's agentic SQL
                    loop; results land in an omniquery session.

    The response never carries SQL or an AST; interpretation chips are
    the only explanation surface.
    """

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Request body must be JSON."}), 400

    query_text = str(data.get("query", "") or "").strip()
    if not query_text:
        return jsonify({"status": "error", "message": "Query text cannot be empty."}), 400
    live = bool(data.get("live"))

    outcome = _get_omniquery_parser().parse(query_text, time.time())
    if outcome.ast is None:
        return jsonify({"status": "error", "message": outcome.reason or "parse failed"}), 500
    raw = outcome.raw or {}
    chips = list(raw.get("interpretation", []))
    base = {"status": "success", "backend": outcome.backend, "interpretation": chips, "query": query_text}

    def _card_payload(ids=None, stat_value=None, session_id=None):
        """Classify a result into one of the palette's result cards:
        stat (aggregate value), spotlight (single hit), empty, or tiles."""
        if stat_value is not None:
            return {"kind": "count", "card": "stat", "count": stat_value}
        ids = ids or []
        payload = {"kind": "ids", "count": len(ids), "preview_ids": ids[:24], "session_id": session_id}
        if not ids:
            payload["card"] = "empty"
        elif len(ids) == 1:
            payload["card"] = "spotlight"
        else:
            payload["card"] = "tiles"
        return payload

    _AGG_RE = re.compile(r"\s*SELECT\s+(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)

    # The model answers when the query carries free language the rules
    # could not type precisely -- and only on the non-live path.
    rules_exact = not raw.get("text_terms")
    if not live and not rules_exact:
        sqlsearch = _get_omniquery_sqlsearch()
        if sqlsearch.available():
            ids, model_sql, err = sqlsearch.search(query_text)
            if ids is not None:
                base["backend"] = "nl2sql"
                base["interpretation"] = [*chips, {"label": "ai search", "field": None}]
                if model_sql and _AGG_RE.match(model_sql):
                    value = float(ids[0]) if ids else 0
                    value = int(value) if value == int(value) else value
                    return jsonify({**base, **_card_payload(stat_value=value)})
                session_id = None
                if ids:
                    try:
                        session_id = _omniquery_store_session(query_text, ids, model_sql)
                    except Exception as e:
                        _logger.debug("handled a failure in omniquery_nlq", exc_info=True)
                        return jsonify({"status": "error", "message": f"Database Error: {e!s}"}), 500
                return jsonify({**base, **_card_payload(ids=ids, session_id=session_id)})
            # Model produced nothing usable: the deterministic answer stands.

    result_ids, count, err = _omniquery_run_ast(outcome.ast, query_text)
    if err is not None:
        return jsonify({"status": "error", "message": err}), 400
    if count is not None:
        return jsonify({**base, **_card_payload(stat_value=count)})
    session_id = None
    if not live and result_ids:
        try:
            session_id = _omniquery_store_session(query_text, result_ids, "typed-engine result set")
        except Exception as e:
            _logger.debug("handled a failure in omniquery_nlq", exc_info=True)
            return jsonify({"status": "error", "message": f"Database Error: {e!s}"}), 500
    return jsonify({**base, **_card_payload(ids=result_ids, session_id=session_id)})


# --- ADMIN BLIND RATING OVERRIDE ---
def is_effectively_blind():
    """Determines if blind rating should be applied for the current user session."""
    if not BLIND_RATING:
        # User opt-in if server doesn't enforce it globally
        return session.get("my_ratings_only", False)
    # Check if user is privileged
    role = session.get("role", "GUEST")
    is_local_admin = not FORCE_LOGIN and not IS_EXHIBITION_MODE
    if is_local_admin or role in ["ADMIN", "MANAGER", "STAFF"]:
        # If they toggled the override, disable blind mode
        if session.get("override_blind", False):
            return False
    return True


@app.route("/galleryout/api/exhibition/toggle_my_ratings", methods=["POST"])
def toggle_my_ratings():
    session["my_ratings_only"] = not session.get("my_ratings_only", False)
    return jsonify({"status": "success"})


@app.route("/galleryout/api/exhibition/toggle_blind", methods=["POST"])
def toggle_blind_override():
    role = session.get("role", "GUEST")
    is_local_admin = not FORCE_LOGIN and not IS_EXHIBITION_MODE
    if not is_local_admin and role not in ["ADMIN", "MANAGER", "STAFF"]:
        return jsonify({"status": "error", "message": "Unauthorized"}), 403

    session["override_blind"] = not session.get("override_blind", False)
    return jsonify({"status": "success"})


def print_startup_banner():
    banner = rf"""
{Colors.GREEN}{Colors.BOLD}   _____                      _      _____       _ _
  / ____|                    | |    / ____|     | | |
 | (___  _ __ ___   __ _ _ __| |_  | |  __  __ _| | | ___ _ __ _   _
  \___ \| '_ ` _ \ / _` | '__| __| | | |_ |/ _` | | |/ _ \ '__| | | |
  ____) | | | | | | (_| | |  | |_  | |__| | (_| | | |  __/ |  | |_| |
 |_____/|_| |_| |_|\__,_|_|   \__|  \_____|\__,_|_|_|\___|_|   \__, |
                                                                __/ |
                                                               |___/ {Colors.RESET}"""

    exh_banner = rf"""
{Colors.YELLOW}{Colors.BOLD}   ______      _     _ _     _ _   _
  |  ____|    | |   (_) |   (_) | (_)
  | |__  __  _| |__  _| |__  _| |_ _  ___  _ __
  |  __| \ \/ / '_ \| | '_ \| | __| |/ _ \| '_ \
  | |____ >  <| | | | | |_) | | |_| | (_) | | | |
  |______/_/\_\_| |_|_|_.__/|_|\__|_|\___/|_| |_|{Colors.RESET}"""

    print(banner)

    if IS_EXHIBITION_MODE:
        print(exh_banner)
        print()
    else:
        print("\n")

    print(f"   {Colors.BOLD}SmartGallery DAM for ComfyUI{Colors.RESET}")
    print(f"   Author     : {Colors.BLUE}Biagio Maffettone{Colors.RESET}")
    print(f"   Version    : {Colors.YELLOW}{APP_VERSION}{Colors.RESET} ({APP_VERSION_DATE})")
    print(f"   GitHub     : {Colors.CYAN}{GITHUB_REPO_URL}{Colors.RESET}")
    print(f"   Contributor: {Colors.CYAN}Martial Michel (Docker & Codebase){Colors.RESET}")
    print()


# --- GLOBAL STATE FOR UPDATES ---
UPDATE_AVAILABLE = False
REMOTE_VERSION = None  # New global variable


def check_for_updates():
    """Checks the GitHub repo for a newer version without external libs."""
    global UPDATE_AVAILABLE, REMOTE_VERSION
    print("Checking for updates...", end=" ", flush=True)
    try:
        # Timeout (3s) not blocking start if no internet connection
        with urllib.request.urlopen(GITHUB_RAW_URL, timeout=3) as response:
            content = response.read().decode("utf-8")

            # Regex modified to handle APP_VERSION="1.41" (string) or APP_VERSION=1.41 (number)
            match = re.search(r'APP_VERSION\s*=\s*["\']?([0-9.]+)["\']?', content)

            remote_version_str = None
            if match:
                remote_version_str = match.group(1)
            else:
                match_header = re.search(r"#\s*Version:\s*([0-9.]+)", content)
                if match_header:
                    remote_version_str = match_header.group(1)

            if remote_version_str:
                local_clean = re.sub(r"[^0-9.]", "", str(APP_VERSION))
                remote_clean = re.sub(r"[^0-9.]", "", str(remote_version_str))

                local_dots = local_clean.count(".")
                remote_dots = remote_clean.count(".")

                is_update_available = False

                if local_dots <= 1 and remote_dots <= 1:
                    with contextlib.suppress(ValueError):
                        is_update_available = float(remote_clean) > float(local_clean)

                if not is_update_available:
                    local_v = tuple(map(int, local_clean.split("."))) if local_clean else (0,)
                    remote_v = tuple(map(int, remote_clean.split("."))) if remote_clean else (0,)
                    is_update_available = remote_v > local_v

                if is_update_available:
                    UPDATE_AVAILABLE = True
                    REMOTE_VERSION = remote_version_str  # Store the version string
                    print(
                        f"\n{Colors.YELLOW}{Colors.BOLD}NOTICE: A new version"
                        f" ({remote_version_str}) is available!{Colors.RESET}"
                    )
                else:
                    print("You are up to date.")
            else:
                print("Could not parse remote version.")

    except Exception:
        _logger.debug("handled a failure in check_for_updates", exc_info=True)
        print("Skipped (Offline or GitHub unreachable).")


# --- STARTUP CHECKS AND MAIN ENTRY POINT ---
def show_config_error_and_exit(path):
    """Shows a critical error message and exits the program.

    The message says which of the two mistakes it is, because "does not
    exist" is untrue for the second one and sends people looking for a
    folder that is sitting right where they put it.
    """
    if os.path.exists(path):
        # Reached because the check asks isdir, not exists. A file passes
        # exists and then nothing works: the scan walks it, finds nothing,
        # and the gallery comes up empty with everything looking normal.
        # "Copy as path" on a file rather than its folder is one keystroke
        # away, and the FFPROBE setting directly below this one in the
        # launcher IS a file, which makes it an easy line to copy.
        what = (
            f"❌ CRITICAL ERROR: BASE_OUTPUT_PATH is a file, not a folder:"
            f"\n\n👉 {path}\n\n"
            f"The gallery shows the contents of a folder, so this needs "
            f"to be the folder your images are in -- most likely the one "
            f"containing the file above.\n\n"
        )
    else:
        what = f"❌ CRITICAL ERROR: The specified path does not exist or is not accessible:\n\n👉 {path}\n\n"

    msg = (
        f"{what}"
        f"INSTRUCTIONS:\n"
        "1. If you are launching via a script (e.g., .bat file), please edit it"
        " and set the correct 'BASE_OUTPUT_PATH' variable.\n"
        "2. Or edit 'smartgallery.py' (USER CONFIGURATION section) and ensure"
        " the path points to an existing folder.\n\n"
        f"The program cannot continue and will now exit."
    )

    if TKINTER_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showerror("SmartGallery - Configuration Error", msg)
        root.destroy()
    else:
        # Fallback for headless environments (Docker, etc.)
        print(f"\n{Colors.RED}{Colors.BOLD}" + "=" * 70 + f"{Colors.RESET}")
        print(f"{Colors.RED}{Colors.BOLD}{msg}{Colors.RESET}")
        print(f"{Colors.RED}{Colors.BOLD}" + "=" * 70 + f"{Colors.RESET}\n")

    sys.exit(1)


def show_ffmpeg_warning():
    """Shows a non-blocking warning message for missing FFmpeg."""
    msg = (
        "WARNING: FFmpeg/FFprobe not found\n\n"
        "The system uses the 'ffprobe' utility to analyze video files. "
        "It seems it is missing or not configured correctly.\n\n"
        "CONSEQUENCES:\n"
        "❌ You will NOT be able to extract ComfyUI workflows from video files (.mp4, .mov, etc).\n"
        "✅ Gallery browsing, playback, and image features will still work perfectly.\n\n"
        "To fix this, install FFmpeg or check the 'FFPROBE_MANUAL_PATH' in the configuration."
    )

    if TKINTER_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showwarning("SmartGallery - Feature Limitation", msg)
        root.destroy()
    else:
        # Fallback for headless environments (Docker, etc.)
        print(f"\n{Colors.YELLOW}{Colors.BOLD}" + "=" * 70 + f"{Colors.RESET}")
        print(f"{Colors.YELLOW}{msg}{Colors.RESET}")
        print(f"{Colors.YELLOW}{Colors.BOLD}" + "=" * 70 + f"{Colors.RESET}\n")


def install_shutdown_signals(handler, names=None):
    """Take the shutdown signals, except any the launcher told us to ignore.

    nohup is `signal (SIGHUP, SIG_IGN); execvp (...)` -- coreutils
    src/nohup.c -- so a gallery started that way inherits an ignored
    SIGHUP, which is the whole point of starting it that way. Installing a
    handler replaces the ignore, and this handler kills the process group,
    so `nohup python smartgallery.py &` over SSH died the moment the
    session ended: the one arrangement made specifically to survive that.

    A shell that backgrounds a job without job control ignores SIGINT for
    it in the same way, for the same reason.

    getsignal returns SIG_IGN when "the signal was previously ignored"
    (Doc/library/signal.rst), so the ones to leave alone can simply be
    asked for. Nothing here decides which signals matter -- whoever
    started the gallery already did.

    Returns (installed, left alone), by name.
    """

    if names is None:
        names = ["SIGINT", "SIGTERM"]
        if os.name != "nt":
            names.append("SIGHUP")

    installed, left_alone = [], []
    for name in names:
        number = getattr(signal, name, None)
        if number is None:
            continue  # SIGHUP does not exist on Windows
        try:
            if signal.getsignal(number) == signal.SIG_IGN:
                left_alone.append(name)
                continue
            signal.signal(number, handler)
            installed.append(name)
        except Exception as exc:
            _logger.debug("handled a failure in install_shutdown_signals", exc_info=True)
            print(f"WARN: Could not bind {name}: {exc}")
    return installed, left_alone


def check_port_available(port):
    """
    Checks if the specified port is available on the host machine.
    Returns True if available, False if already in use.

    It binds the way the server is about to bind, which is the only thing
    that makes the answer worth having. waitress calls set_reuse_addr()
    on its listening socket before binding it, and werkzeug's server sets
    allow_reuse_address as well, so both ask for SO_REUSEADDR. A check
    that binds without it is stricter than the server it stands in front
    of, and the gap is not hypothetical: on POSIX a port still holding
    connections from a crashed run refuses a plain bind and accepts the
    server's. Refusing there means the gallery announces PORT ALREADY IN
    USE and waits for a keypress, over a port it could have listened on.

    Matching the server also means this cannot drift the other way. It
    asks for exactly what the server asks for, so whatever the option
    means on a given system, both get the same answer.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, s.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) | 1
            )
        except OSError:
            pass  # same tolerance waitress has; the bind is what matters
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


# --- OS FILE DESCRIPTOR BOOSTER (macOS/Linux) ---
# Safely attempts to increase the open file limit to prevent 'Too many open files'
# or 'ValueError: filedescriptor out of range in select()' during heavy grid loads.
# Windows ignores this block automatically.
try:
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target_soft = 4096
    if soft < target_soft:
        new_soft = min(target_soft, hard) if hard != resource.RLIM_INFINITY else target_soft
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
except Exception:
    _logger.debug("ignored a failure in module scope", exc_info=True)

# --- EXPERIMENTAL REMIX API (INLINE) ---

UPLOAD_TRANSPORT_HEADROOM_BYTES = 32 * 1024 * 1024


def derive_upload_ceilings(max_upload_mb):
    """The two limits an upload has to clear, from the one setting.

    The web server in front of Flask keeps its own ceiling, and refuses a
    request before Flask ever sees it. That ceiling was written in as a
    fixed 2 GiB, so raising COMFYUI_MAX_UPLOAD_MB past 2048 did nothing at
    all: the app agreed to a 4 GB video and the server underneath it
    refused one at 2048 MB, with a message naming a number the setting does
    not contain.

    The transport ceiling sits ABOVE the app's rather than level with it,
    for two reasons. An upload is a multipart body, so it is bigger than
    the file inside it by the boundaries and headers; and waitress refuses
    at >= its ceiling while Flask refuses above its own. Keeping the
    transport limit clear of both means the app is always the one that
    decides, and can say so in words.

    Returns (app_ceiling_bytes, transport_ceiling_bytes).
    """
    app_ceiling = max_upload_mb * 1024 * 1024
    return app_ceiling, app_ceiling + UPLOAD_TRANSPORT_HEADROOM_BYTES


# Increase max request body size to handle large workflow JSON payloads (if users
# manually upload them, though we now read from disk)
MAX_UPLOAD_MB = env_num("COMFYUI_MAX_UPLOAD_MB", 2000, minimum=1)
app.config["MAX_CONTENT_LENGTH"], MAX_REQUEST_BODY_BYTES = derive_upload_ceilings(MAX_UPLOAD_MB)


def _caller_is_reading_json():
    """Whether the thing waiting on this reply parses JSON.

    A page navigation wants Flask's error page. Everything the interface
    calls with fetch() does res.json(), and an HTML page there shows the
    person nothing at all.
    """
    try:
        if request.path.startswith("/galleryout/api/"):
            return True
        if request.is_json:
            return True
        best = request.accept_mimetypes.best_match(["application/json", "text/html"])
        return best == "application/json" and request.accept_mimetypes[best] > request.accept_mimetypes["text/html"]
    except Exception:
        _logger.debug("handled a failure in _caller_is_reading_json", exc_info=True)
        return False


@app.errorhandler(Exception)
def _unexpected_error_still_answers_readably(error):
    """Give an unhandled fault a body the screen can read.

    Swept every JSON route with values of the wrong kind -- a string where
    a number was expected, a list where a string was, an object where a
    list was -- and 235 of those answers were a 500 carrying an HTML page,
    across 63 route-and-field pairs. Every one of them reaches a caller
    doing res.json(), so what the person sees is nothing: not the fault,
    not a reason, not even that it was refused.

    Fixing the sixty-three by hand would be sixty-three chances to get it
    wrong, and the next route added would start again. What they have in
    common is the answer, so that is what is fixed.

    Two things this deliberately does NOT do. It does not swallow the
    fault: the traceback still goes to the console exactly as before, so
    nothing becomes harder to diagnose. And it does not repeat the
    exception's message, which is how a filesystem path or a fragment of
    SQL would reach an exhibition visitor -- the console has that, and the
    console is the owner's.
    """
    if isinstance(error, HTTPException):
        return error  # 404, 405, 403 and the rest keep their meaning

    traceback.print_exc()

    if not _caller_is_reading_json():
        return InternalServerError()

    return jsonify(
        {
            "status": "error",
            "message": (
                "Something went wrong handling that request "
                f"({type(error).__name__}). The gallery console has the "
                f"details."
            ),
        }
    ), 500


@app.errorhandler(413)
def _upload_too_large(error):
    """Say which limit was hit and what it is set to.

    Without this the reply is an HTML error page, and the upload screen has
    nothing to show but the number 413.
    """
    return jsonify(
        {
            "status": "error",
            "message": (
                f"That file is larger than this gallery accepts. The "
                f"limit is {MAX_UPLOAD_MB} MB, and is set by "
                f"COMFYUI_MAX_UPLOAD_MB."
            ),
        }
    ), 413


# Experimental Remix API Module


def clean_workflow_paths(data):
    """
    Recursively cleans multiple forward slashes (//) and backslashes (\\)
    from string values in the workflow data, reducing them to a single slash,
    without normalizing all slashes to a single type.
    Preserves UNC paths (\\server or //server) and skips URLs (://).
    """
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str):
                if "://" not in v:
                    is_unc_slash = v.startswith("//")
                    is_unc_back = v.startswith(r"\\")

                    cleaned = re.sub(r"/+", "/", v)
                    cleaned = re.sub(r"\\+", lambda m: "\\", cleaned)

                    if is_unc_slash and not cleaned.startswith("//"):
                        cleaned = "/" + cleaned
                    if is_unc_back and not cleaned.startswith(r"\\"):
                        cleaned = "\\" + cleaned

                    data[k] = cleaned
            else:
                clean_workflow_paths(v)
    elif isinstance(data, list):
        for i in range(len(data)):
            if isinstance(data[i], str):
                if "://" not in data[i]:
                    is_unc_slash = data[i].startswith("//")
                    is_unc_back = data[i].startswith(r"\\")

                    cleaned = re.sub(r"/+", "/", data[i])
                    cleaned = re.sub(r"\\+", lambda m: "\\", cleaned)

                    if is_unc_slash and not cleaned.startswith("//"):
                        cleaned = "/" + cleaned
                    if is_unc_back and not cleaned.startswith(r"\\"):
                        cleaned = "\\" + cleaned

                    data[i] = cleaned
            else:
                clean_workflow_paths(data[i])
    return data


# --- OMNIQUERY EXTENDED APIs ---
@app.route("/galleryout/api/omniquery/prompts/factory", methods=["GET"])
@management_api_only
def get_factory_omniquery_prompt():
    try:
        new_text = get_omniquery_dictionary(reset=True)
        return jsonify({"status": "success", "template": new_text})
    except Exception as e:
        _logger.debug("handled a failure in get_factory_omniquery_prompt", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/omniquery/prompts/list", methods=["GET"])
@management_api_only
def list_omniquery_prompts():
    try:
        p_dir = os.path.join(BASE_SMARTGALLERY_PATH, ".omniquery", "saved_prompts")
        os.makedirs(p_dir, exist_ok=True)
        prompts = []
        for f in os.listdir(p_dir):
            if f.lower().endswith(".txt"):
                path = os.path.join(p_dir, f)
                mtime = os.path.getmtime(path)
                desc = ""
                try:
                    with open(path, encoding="utf-8") as pf:
                        first_line = pf.readline().strip()
                        if first_line.startswith("-- Description:"):
                            desc = first_line.replace("-- Description:", "", 1).strip()
                except Exception:
                    _logger.debug("ignored a failure in list_omniquery_prompts", exc_info=True)
                prompts.append({"name": f, "mtime": mtime, "description": desc})
        return jsonify({"status": "success", "prompts": prompts})
    except Exception as e:
        _logger.debug("handled a failure in list_omniquery_prompts", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/omniquery/prompts/save", methods=["POST"])
@management_api_only
def save_omniquery_prompt():
    data = request.json
    name = data.get("name", "").strip()
    text = data.get("text", "").strip()

    if not name or not text:
        return jsonify({"status": "error", "message": "Name and Prompt text required."}), 400
    if not name.lower().endswith(".txt"):
        name += ".txt"
    safe_name = safe_media_filename(name, fallback="untitled")

    try:
        p_dir = os.path.join(BASE_SMARTGALLERY_PATH, ".omniquery", "saved_prompts")
        os.makedirs(p_dir, exist_ok=True)
        with open(os.path.join(p_dir, safe_name), "w", encoding="utf-8") as f:
            f.write(text)
        return jsonify({"status": "success", "message": "Prompt saved."})
    except Exception as e:
        _logger.debug("handled a failure in save_omniquery_prompt", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/omniquery/prompts/load", methods=["POST"])
@management_api_only
def load_omniquery_prompt():
    name = request.json.get("name")
    safe_name = safe_media_filename(name, fallback="untitled")
    try:
        p_dir = os.path.join(BASE_SMARTGALLERY_PATH, ".omniquery", "saved_prompts")
        with open(os.path.join(p_dir, safe_name), encoding="utf-8") as f:
            text = f.read()
        return jsonify({"status": "success", "text": text})
    except Exception as e:
        _logger.debug("handled a failure in load_omniquery_prompt", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/omniquery/prompts/delete", methods=["POST"])
@management_api_only
def delete_omniquery_prompt():
    name = request.json.get("name")
    safe_name = safe_media_filename(name, fallback="untitled")
    try:
        p_dir = os.path.join(BASE_SMARTGALLERY_PATH, ".omniquery", "saved_prompts")
        path = os.path.join(p_dir, safe_name)
        if os.path.exists(path):
            os.remove(path)
        return jsonify({"status": "success"})
    except Exception as e:
        _logger.debug("handled a failure in delete_omniquery_prompt", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/omniquery/prompts/rename", methods=["POST"])
@management_api_only
def rename_omniquery_prompt():
    data = request.json
    old_name = safe_media_filename(data.get("old_name", ""), fallback="untitled")
    new_name = data.get("new_name", "").strip()
    if not new_name.lower().endswith(".txt"):
        new_name += ".txt"
    safe_new = safe_media_filename(new_name, fallback="untitled")
    try:
        p_dir = os.path.join(BASE_SMARTGALLERY_PATH, ".omniquery", "saved_prompts")
        old_path = os.path.join(p_dir, old_name)
        new_path = os.path.join(p_dir, safe_new)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            return jsonify({"status": "success"})
        return jsonify({"status": "error", "message": "File not found or destination exists."}), 400
    except Exception as e:
        _logger.debug("handled a failure in rename_omniquery_prompt", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/omniquery/queries/list", methods=["GET"])
@management_api_only
def list_omniquery_queries():
    try:
        q_dir = os.path.join(BASE_SMARTGALLERY_PATH, ".omniquery", "saved_queries")
        os.makedirs(q_dir, exist_ok=True)
        queries = []
        for f in os.listdir(q_dir):
            if f.lower().endswith(".txt"):
                path = os.path.join(q_dir, f)
                mtime = os.path.getmtime(path)
                desc = ""
                # Read description and prompt from SQL comment
                try:
                    with open(path, encoding="utf-8") as qf:
                        file_content = qf.read()
                        desc_match = re.search(r"^--\s*Description:\s*(.*)", file_content, re.IGNORECASE | re.MULTILINE)
                        desc = desc_match.group(1).strip() if desc_match else ""

                        prompt_match = re.search(
                            r"/\*\s*Prompt Request:\s*(.*?)\s*\*/", file_content, re.IGNORECASE | re.DOTALL
                        )
                        if prompt_match:
                            prompt_text = prompt_match.group(1).strip()
                            if prompt_text:
                                desc = desc + f" 💡 {prompt_text}" if desc else f"💡 {prompt_text}"
                except Exception:
                    _logger.debug("ignored a failure in list_omniquery_queries", exc_info=True)
                queries.append({"name": f, "mtime": mtime, "description": desc})
        return jsonify({"status": "success", "queries": queries})
    except Exception as e:
        _logger.debug("handled a failure in list_omniquery_queries", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/omniquery/queries/save", methods=["POST"])
@management_api_only
def save_omniquery_query():
    data = request.json
    name = data.get("name", "").strip()
    desc = data.get("description", "").strip()
    sql = data.get("sql", "").strip()

    if not name or not sql:
        return jsonify({"status": "error", "message": "Name and SQL required."}), 400
    if not name.lower().endswith(".txt"):
        name += ".txt"
    safe_name = safe_media_filename(name, fallback="untitled")

    # /list advertises a description read back out of a leading
    # `-- Description:` comment, so that is where a supplied one has to go --
    # accepting the field and dropping it left every saved query blank.
    # Flattened to one line first: a newline would end the comment and the
    # rest would be executed as SQL.
    final_sql = sql
    if desc:
        final_sql = "-- Description: " + " ".join(desc.split()) + "\n" + sql

    try:
        q_dir = os.path.join(BASE_SMARTGALLERY_PATH, ".omniquery", "saved_queries")
        os.makedirs(q_dir, exist_ok=True)
        with open(os.path.join(q_dir, safe_name), "w", encoding="utf-8") as f:
            f.write(final_sql)
        return jsonify({"status": "success", "message": "Query saved."})
    except Exception as e:
        _logger.debug("handled a failure in save_omniquery_query", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/omniquery/queries/load", methods=["POST"])
@management_api_only
def load_omniquery_query():
    name = request.json.get("name")
    safe_name = safe_media_filename(name, fallback="untitled")
    try:
        q_dir = os.path.join(BASE_SMARTGALLERY_PATH, ".omniquery", "saved_queries")
        with open(os.path.join(q_dir, safe_name), encoding="utf-8") as f:
            sql = f.read()
        return jsonify({"status": "success", "sql": sql})
    except Exception as e:
        _logger.debug("handled a failure in load_omniquery_query", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/omniquery/queries/delete", methods=["POST"])
@management_api_only
def delete_omniquery_query():
    name = request.json.get("name")
    safe_name = safe_media_filename(name, fallback="untitled")
    try:
        q_dir = os.path.join(BASE_SMARTGALLERY_PATH, ".omniquery", "saved_queries")
        path = os.path.join(q_dir, safe_name)
        if os.path.exists(path):
            os.remove(path)
        return jsonify({"status": "success"})
    except Exception as e:
        _logger.debug("handled a failure in delete_omniquery_query", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/omniquery/queries/rename", methods=["POST"])
@management_api_only
def rename_omniquery_query():
    data = request.json
    old_name = safe_media_filename(data.get("old_name", ""), fallback="untitled")
    new_name = data.get("new_name", "").strip()
    if not new_name.lower().endswith(".txt"):
        new_name += ".txt"
    safe_new = safe_media_filename(new_name, fallback="untitled")
    try:
        q_dir = os.path.join(BASE_SMARTGALLERY_PATH, ".omniquery", "saved_queries")
        old_path = os.path.join(q_dir, old_name)
        new_path = os.path.join(q_dir, safe_new)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            return jsonify({"status": "success"})
        return jsonify({"status": "error", "message": "File not found or destination exists."}), 400
    except Exception as e:
        _logger.debug("handled a failure in rename_omniquery_query", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/galleryout/api/omniquery/preview", methods=["POST"])
@management_api_only
def preview_omniquery_query():
    data = request.json
    raw_sql = data.get("sql", "").strip()
    if not raw_sql:
        return jsonify({"status": "error", "message": "Empty query."}), 400

    clean_sql = re.sub(r"(/\*.*?\*/)|(--.*?\n)", "", raw_sql, flags=re.DOTALL).strip()
    if not re.match(r"^SELECT\b", clean_sql, re.IGNORECASE):
        return jsonify({"status": "error", "message": "Security Block: Only SELECT statements are allowed."}), 403

    db_uri = f"file:{os.path.abspath(DATABASE_FILE)}?mode=ro"
    try:

        def query_authorizer(action, arg1, arg2, dbname, source):
            if action in (21, 20, 31):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        with sqlite3.connect(db_uri, uri=True) as ro_conn:
            ro_conn.set_authorizer(query_authorizer)
            cursor = ro_conn.execute(raw_sql)
            rows = cursor.fetchall()
            cols = [description[0] for description in cursor.description] if cursor.description else []
            result = [dict(zip(cols, row, strict=False)) for row in rows]
            return jsonify({"status": "success", "columns": cols, "rows": result, "count": len(result)})
    except Exception as e:
        _logger.debug("handled a failure in preview_omniquery_query", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 400


def _register_remix_routes_inline():

    def parse_workflow(raw_json, wf_type, raw_ui_json=None):
        wf_data = json.loads(raw_json)
        ui_data = json.loads(raw_ui_json) if raw_ui_json else (wf_data if wf_type == "ui" else {})

        extract = {
            "workflow_type": wf_type,
            "texts": [],
            "seeds": [],
            "numbers": [],
            "images": [],
            "save_prefix": None,
            "save_node_type": None,
            "default_comfy_url": COMFYUI_SERVER_URL,
            "has_app_mode": False,
        }

        app_params = []
        if isinstance(ui_data, dict):
            extra = ui_data.get("extra", {})
            linear_data = extra.get("linearData", {})
            if isinstance(linear_data, dict) and "inputs" in linear_data:
                app_params = linear_data["inputs"]
                extract["has_app_mode"] = True
            else:
                app_data = ui_data.get("app", extra.get("app", {}))
                if isinstance(app_data, dict) and "parameters" in app_data:
                    app_params = app_data["parameters"]
                    extract["has_app_mode"] = True

        def check_is_app(n_id, k, n_type, n_title):
            for p in app_params:
                if isinstance(p, list) and len(p) >= 2:
                    if str(p[0]) == str(n_id) and str(p[1]) == k:
                        return True
                elif isinstance(p, dict):
                    if str(p.get("node_id")) == str(n_id) and (not p.get("widget_name") or p.get("widget_name") == k):
                        return True
            return bool(n_type == "PrimitiveNode" and n_title and n_title != "PrimitiveNode")

        def check_is_app_by_index(n_id, widget_idx):
            # Match by position: count app_params entries for this node_id;
            # the Nth entry corresponds to the Nth app widget for that node.
            # This handles nodes not in NODE_PARAM_NAMES where k='widget_N'
            # but app_params stores the real widget name.
            # If this node has ANY app_params entries, check if widget_idx
            # falls within the range of defined app params for this node.
            # We also do a direct index match against the linearData order.
            all_node_widgets = []
            for p in app_params:
                if (isinstance(p, list) and len(p) >= 2 and str(p[0]) == str(n_id)) or (
                    isinstance(p, dict) and str(p.get("node_id", "")) == str(n_id)
                ):
                    all_node_widgets.append(p)
            return len(all_node_widgets) > 0 and widget_idx < len(all_node_widgets)

        def get_widget_name_by_index(n_type, idx):
            param_names = NODE_PARAM_NAMES.get(n_type, [])
            if idx < len(param_names):
                return param_names[idx]
            return f"widget_{idx}"

        save_classes = [
            "SaveImage",
            "SaveAnimatedWEBP",
            "SaveAnimatedPNG",
            "VHS_VideoCombine",
            "SaveVideo",
            "SaveLatent",
        ]

        if wf_type == "api":
            if isinstance(wf_data, list):
                wf_data = {str(i): n for i, n in enumerate(wf_data)}
            for node_id, node in wf_data.items():
                if not isinstance(node, dict):
                    continue
                inputs = node.get("inputs", {})
                n_type = node.get("class_type", "Unknown")
                n_title = node.get("_meta", {}).get("title", n_type)
                type_title_lower = (n_type + " " + n_title).lower()

                if n_type in save_classes or any(c in n_type for c in save_classes):
                    fp = inputs.get("filename_prefix")
                    if fp and isinstance(fp, str):
                        extract["save_prefix"] = fp
                        extract["save_node_type"] = n_title

                for key, val in inputs.items():
                    key_l = key.lower()
                    is_app_field = check_is_app(node_id, key, n_type, n_title)
                    if is_app_field:
                        extract["has_app_mode"] = True

                    if isinstance(val, str) and any(
                        val.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".jfif", ".bmp"]
                    ):
                        extract["images"].append(
                            {
                                "node_id": node_id,
                                "value": val,
                                "key": key,
                                "node_type": n_type,
                                "title": n_title,
                                "is_app_field": is_app_field,
                                "label": n_title if is_app_field else "Image",
                            }
                        )
                    elif isinstance(val, bool):
                        is_target = is_app_field or any(x in key_l for x in ["enable", "keep", "save", "preview"])
                        if is_target:
                            extract["numbers"].append(
                                {
                                    "node_id": node_id,
                                    "key": key,
                                    "value": val,
                                    "node_type": n_type,
                                    "title": n_title,
                                    "label": n_title if is_app_field else key.capitalize(),
                                    "is_app_field": is_app_field,
                                    "is_bool": True,
                                }
                            )
                    elif (
                        ("seed" in key_l or "seed" in type_title_lower)
                        and isinstance(val, (int, float))
                        and not isinstance(val, bool)
                    ):
                        extract["seeds"].append(
                            {
                                "node_id": node_id,
                                "key": key,
                                "value": val,
                                "node_type": n_type,
                                "title": n_title,
                                "is_app_field": is_app_field,
                                "label": n_title if is_app_field else "Seed",
                            }
                        )
                    elif isinstance(val, int) and not isinstance(val, bool) and val > 10000:
                        if (
                            "width" not in key_l
                            and "height" not in key_l
                            and "width" not in type_title_lower
                            and "height" not in type_title_lower
                        ):
                            if not any(s["node_id"] == node_id and s["key"] == key for s in extract["seeds"]):
                                extract["seeds"].append(
                                    {
                                        "node_id": node_id,
                                        "key": key,
                                        "value": val,
                                        "node_type": n_type,
                                        "title": n_title,
                                        "is_app_field": is_app_field,
                                        "label": n_title if is_app_field else "Seed",
                                    }
                                )
                    elif isinstance(val, (int, float)) and not isinstance(val, bool):
                        is_target = is_app_field or any(
                            x in key_l or x in type_title_lower
                            for x in [
                                "step",
                                "cfg",
                                "guidance",
                                "denoise",
                                "width",
                                "height",
                                "batch",
                                "literal",
                                "scale",
                                "length",
                                "frame",
                                "total_second",
                                "num_frame",
                                "video_length",
                                "num_frames",
                                "fps",
                                "frame_rate",
                                "duration",
                                "second",
                            ]
                        )
                        if is_target:
                            label = n_title if is_app_field else "Param"
                            if not is_app_field:
                                if "step" in key_l or "step" in type_title_lower:
                                    label = "Steps"
                                elif "cfg" in key_l or "cfg" in type_title_lower or "scale" in type_title_lower:
                                    label = "CFG"
                                elif "width" in key_l or "width" in type_title_lower:
                                    label = "Width"
                                elif "height" in key_l or "height" in type_title_lower:
                                    label = "Height"
                                elif "denoise" in key_l or "denoise" in type_title_lower:
                                    label = "Denoise"
                                elif any(x in key_l for x in ["fps", "frame_rate"]):
                                    label = "FPS"
                                elif (
                                    any(
                                        x in key_l
                                        for x in ["length", "num_frame", "video_length", "num_frames", "frame_count"]
                                    )
                                    or "frame" in key_l
                                ):
                                    label = "Frames"
                                elif n_title and n_title != n_type:
                                    label = n_title
                            if not any(s["node_id"] == node_id and s["key"] == key for s in extract["seeds"]):
                                extract["numbers"].append(
                                    {
                                        "node_id": node_id,
                                        "key": key,
                                        "value": val,
                                        "node_type": n_type,
                                        "title": n_title,
                                        "label": label,
                                        "is_app_field": is_app_field,
                                        "orig_type": "number",
                                    }
                                )
                    elif isinstance(val, str):
                        num_keys = [
                            "fps",
                            "frame_rate",
                            "steps",
                            "length",
                            "num_frames",
                            "width",
                            "height",
                            "seed",
                            "cfg",
                            "denoise",
                            "overlap",
                            "batch",
                        ]
                        if any(x in key_l for x in num_keys) and val.strip().lstrip("-").replace(".", "", 1).isdigit():
                            num_val = float(val) if "." in val else int(val)
                            label = (
                                n_title
                                if is_app_field
                                else (
                                    "FPS"
                                    if any(x in key_l for x in ["fps", "frame_rate"])
                                    else "Frames"
                                    if any(x in key_l for x in ["length", "num_frames"])
                                    else "Steps"
                                    if "step" in key_l
                                    else "Width"
                                    if "width" in key_l
                                    else "Height"
                                    if "height" in key_l
                                    else n_title
                                    if (n_title and n_title != n_type)
                                    else "Param"
                                )
                            )
                            if not any(s["node_id"] == node_id and s["key"] == key for s in extract["seeds"]):
                                extract["numbers"].append(
                                    {
                                        "node_id": node_id,
                                        "key": key,
                                        "value": num_val,
                                        "node_type": n_type,
                                        "title": n_title,
                                        "label": label,
                                        "is_app_field": is_app_field,
                                        "orig_type": "string",
                                    }
                                )
                        else:
                            is_explicit_prompt = any(
                                x in type_title_lower or x in key_l for x in ["prompt", "positive", "negative"]
                            )
                            if (
                                (is_app_field or is_explicit_prompt)
                                and not val.endswith((".ckpt", ".safetensors", ".pth", ".bin", ".gguf", ".pt", ".json"))
                                and "|" not in val
                            ):
                                label = n_title if is_app_field else "Text"
                                if not is_app_field:
                                    if "positive" in type_title_lower or "positive" in key_l:
                                        label = "Positive Prompt"
                                    elif "negative" in type_title_lower or "negative" in key_l:
                                        label = "Negative Prompt"
                                    elif "system" in type_title_lower:
                                        label = "System Prompt"
                                    elif "wildcard" in type_title_lower:
                                        label = "Wildcard Text"
                                    elif n_title and n_title != n_type:
                                        label = n_title
                                extract["texts"].append(
                                    {
                                        "node_id": node_id,
                                        "value": val,
                                        "key": key,
                                        "node_type": n_type,
                                        "title": n_title,
                                        "label": label,
                                        "is_app_field": is_app_field,
                                    }
                                )
        else:
            # Use filter_enabled_nodes to exclude disabled/muted nodes (mode!=0)
            # and Note/Reroute nodes — same filtering used elsewhere in the codebase.
            _filtered = filter_enabled_nodes(wf_data) if isinstance(wf_data, dict) else {"nodes": wf_data}
            nodes = _filtered.get("nodes", [])
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                n_id = str(node.get("id", ""))
                n_type = node.get("type", node.get("class_type", "Unknown"))
                n_title = node.get("title", n_type)
                widgets = node.get("widgets_values", [])
                if not widgets:
                    continue
                if ("Note" in n_type and "Project" not in n_type) or n_type == "Reroute":
                    continue
                type_title_lower = (n_type + " " + n_title).lower()
                for i, w in enumerate(widgets):
                    k = f"widget_{i}"
                    w_name = get_widget_name_by_index(n_type, i)
                    is_app_field = (
                        check_is_app(n_id, w_name, n_type, n_title)
                        or check_is_app(n_id, k, n_type, n_title)
                        or check_is_app_by_index(n_id, i)
                    )
                    if is_app_field:
                        extract["has_app_mode"] = True
                    if isinstance(w, str) and any(
                        w.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".jfif", ".bmp"]
                    ):
                        extract["images"].append(
                            {
                                "node_id": n_id,
                                "value": w,
                                "widget_index": i,
                                "node_type": n_type,
                                "title": n_title,
                                "is_app_field": is_app_field,
                                "label": n_title if is_app_field else "Image",
                            }
                        )
                    elif isinstance(w, bool):
                        is_target = is_app_field or any(
                            x in type_title_lower for x in ["enable", "keep", "save", "preview"]
                        )
                        if is_target:
                            extract["numbers"].append(
                                {
                                    "node_id": n_id,
                                    "key": k,
                                    "value": w,
                                    "widget_index": i,
                                    "node_type": n_type,
                                    "title": n_title,
                                    "label": n_title if is_app_field else k.capitalize(),
                                    "is_app_field": is_app_field,
                                    "is_bool": True,
                                }
                            )
                    elif (
                        isinstance(w, (int, float))
                        and not isinstance(w, bool)
                        and (w > 10000 or (n_type.endswith("Looper") and i == 1) or "seed" in type_title_lower)
                    ):
                        if not any(s["node_id"] == n_id for s in extract["seeds"]):
                            extract["seeds"].append(
                                {
                                    "node_id": n_id,
                                    "key": k,
                                    "value": w,
                                    "widget_index": i,
                                    "node_type": n_type,
                                    "title": n_title,
                                    "is_app_field": is_app_field,
                                    "label": n_title if is_app_field else "Seed",
                                }
                            )
                    elif isinstance(w, (int, float)) and not isinstance(w, bool):
                        is_target_param = is_app_field or any(
                            x in type_title_lower or x in n_type.lower()
                            for x in [
                                "sampler",
                                "noise",
                                "step",
                                "cfg",
                                "guidance",
                                "detailer",
                                "scale",
                                "denoise",
                                "literal",
                                "width",
                                "height",
                                "resolution",
                                "video",
                                "latent",
                                "looper",
                                "wan",
                                "hunyuan",
                                "mochi",
                                "framepack",
                                "frame",
                                "vantage",
                                "i2v",
                                "t2v",
                                "combine",
                                "fps",
                            ]
                        )
                        if is_target_param:
                            label = n_title if is_app_field else "Param"
                            if not is_app_field:
                                if "step" in type_title_lower:
                                    label = "Steps"
                                elif "cfg" in type_title_lower or "scale" in type_title_lower:
                                    label = "CFG"
                                elif "denoise" in type_title_lower:
                                    label = "Denoise"
                                elif "width" in type_title_lower:
                                    label = "Width"
                                elif "height" in type_title_lower:
                                    label = "Height"
                                elif any(x in n_type.lower() for x in ["combine", "save"]) and 1 <= w <= 240:
                                    label = "FPS"
                                elif (
                                    any(
                                        x in n_type.lower()
                                        for x in [
                                            "video",
                                            "latent",
                                            "looper",
                                            "wan",
                                            "hunyuan",
                                            "mochi",
                                            "framepack",
                                            "vantage",
                                            "i2v",
                                            "t2v",
                                        ]
                                    )
                                    and isinstance(w, int)
                                    and 1 < w < 10000
                                ):
                                    label = "Frames"
                                elif n_title and n_title != n_type:
                                    label = n_title
                            if not any(s["node_id"] == n_id and s["key"] == k for s in extract["seeds"]):
                                extract["numbers"].append(
                                    {
                                        "node_id": n_id,
                                        "key": k,
                                        "value": w,
                                        "widget_index": i,
                                        "node_type": n_type,
                                        "title": n_title,
                                        "label": label,
                                        "is_app_field": is_app_field,
                                        "orig_type": "number",
                                    }
                                )
                    elif isinstance(w, str):
                        is_explicit_prompt = any(x in type_title_lower for x in ["prompt", "positive", "negative"])
                        if (
                            (is_app_field or is_explicit_prompt)
                            and not w.endswith((".ckpt", ".safetensors", ".pth", ".bin", ".gguf", ".pt", ".json"))
                            and "|" not in w
                        ):
                            label = n_title if is_app_field else "Text"
                            if not is_app_field:
                                if "positive" in type_title_lower:
                                    label = "Positive Prompt"
                                elif "negative" in type_title_lower:
                                    label = "Negative Prompt"
                                elif "system" in type_title_lower:
                                    label = "System Prompt"
                                elif "wildcard" in type_title_lower:
                                    label = "Wildcard Text"
                                elif n_title and n_title != n_type:
                                    label = n_title
                            extract["texts"].append(
                                {
                                    "node_id": n_id,
                                    "value": w,
                                    "widget_index": i,
                                    "node_type": n_type,
                                    "title": n_title,
                                    "label": label,
                                    "is_app_field": is_app_field,
                                }
                            )

        # Sort app-flagged fields by their position in app_params (App Builder order)
        if app_params and extract.get("has_app_mode"):

            def _app_order(field):
                n_id = str(field.get("node_id", ""))
                k = str(field.get("key", ""))
                wi = field.get("widget_index")
                for idx, p in enumerate(app_params):
                    if isinstance(p, list) and len(p) >= 2:
                        if str(p[0]) == n_id:
                            if str(p[1]) == k:
                                return idx
                            if wi is not None and str(p[1]) == str(wi):
                                return idx
                    elif isinstance(p, dict) and str(p.get("node_id", "")) == n_id:
                        wname = str(p.get("widget_name", ""))
                        if not wname or wname == k:
                            return idx
                        if wi is not None and str(p.get("widget_index", "")) == str(wi):
                            return idx
                # Positional fallback for index-matched app fields
                if wi is not None:
                    count = 0
                    for idx, p in enumerate(app_params):
                        p_nid = str(p[0]) if isinstance(p, list) else str(p.get("node_id", ""))
                        if p_nid == n_id:
                            if count == wi:
                                return idx
                            count += 1
                return 99999

            extract["texts"] = sorted(extract["texts"], key=_app_order)
            extract["seeds"] = sorted(extract["seeds"], key=_app_order)
            extract["numbers"] = sorted(extract["numbers"], key=_app_order)
            extract["images"] = sorted(extract["images"], key=_app_order)

        return extract

    @app.route("/galleryout/api/remix/workflows", methods=["GET"])
    @management_api_only
    def api_remix_list_workflows():
        try:
            if not os.path.exists(IMPORTED_WORKFLOWS_DIR):
                os.makedirs(IMPORTED_WORKFLOWS_DIR, exist_ok=True)
            files = []
            for f in os.listdir(IMPORTED_WORKFLOWS_DIR):
                if not f.lower().endswith(".json"):
                    continue
                path = os.path.join(IMPORTED_WORKFLOWS_DIR, f)
                if os.path.isfile(path):
                    mtime = os.path.getmtime(path)
                    has_app = False
                    has_custom = False
                    source_file_id = None
                    try:
                        with open(path, encoding="utf-8") as jf:
                            data = json.load(jf)
                            ui_data = data.get("ui", {})
                            extra = ui_data.get("extra", {})
                            if "linearData" in extra or "app" in extra or "app" in ui_data:
                                has_app = True
                            source_file_id = data.get("sg_meta", {}).get("source_file_id")
                            has_custom = bool(data.get("sg_meta", {}).get("custom_app"))
                    except (OSError, json.JSONDecodeError, AttributeError):
                        # An unreadable or non-JSON template still gets listed,
                        # just without its app/custom flags.
                        pass
                    files.append(
                        {
                            "name": f,
                            "mtime": mtime,
                            "has_app_mode": has_app,
                            "has_custom_mode": has_custom,
                            "source_file_id": source_file_id,
                        }
                    )
            return jsonify({"status": "success", "workflows": files})
        except Exception as e:
            _logger.debug("handled a failure in api_remix_list_workflows", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/galleryout/api/remix/workflows/save_template", methods=["POST"])
    @management_api_only
    def api_remix_save_template():
        try:
            data = request.json
            file_id = data.get("file_id")
            name = data.get("name", "").strip()
            companion_path = data.get("companion_path")
            custom_app = data.get("custom_app", [])
            workflow_file = data.get("workflow_file")  # PATCH: Get workflow file name
            save_mode = data.get("save_mode", "modified")
            modifications = data.get("modifications", {})
            raw_override = data.get("raw_override")
            override_type = data.get("override_type", "api")
            favorite_nodes = data.get("favorite_nodes", {})

            if not name:
                return jsonify({"status": "error", "message": "Missing name"}), 400

            safe_name = safe_media_filename(name, fallback="untitled")
            if not safe_name.lower().endswith(".json"):
                safe_name += ".json"

            raw_api, raw_ui = None, None
            source_file_id = file_id

            if workflow_file:
                # PATCH: Read workflow data directly from the existing template instead of extracting from media
                tpl_path = os.path.join(IMPORTED_WORKFLOWS_DIR, safe_media_filename(workflow_file, fallback="untitled"))
                if os.path.exists(tpl_path):
                    with open(tpl_path, encoding="utf-8") as f:
                        tpl_data = json.load(f)
                        raw_api = json.dumps(tpl_data.get("api", {}))
                        raw_ui = json.dumps(tpl_data.get("ui", {}))
                        source_file_id = tpl_data.get("sg_meta", {}).get("source_file_id", file_id)
            else:
                # Standard extraction from media file
                if not file_id:
                    return jsonify({"status": "error", "message": "Missing file ID"}), 400
                info = get_file_info_from_db(file_id)
                target_path = companion_path if companion_path and os.path.exists(companion_path) else info["path"]

                raw_api = extract_workflow(target_path, target_type="api")
                raw_ui = extract_workflow(target_path, target_type="ui")

                if not raw_api or not raw_ui:
                    stem = os.path.splitext(target_path)[0]
                    for ext in (".png", ".PNG"):
                        companion = stem + ext
                        if os.path.isfile(companion):
                            raw_api = extract_workflow(companion, target_type="api")
                            raw_ui = extract_workflow(companion, target_type="ui")
                            break

            if not raw_api or not raw_ui or raw_api == "{}" or raw_ui == "{}":
                return jsonify(
                    {
                        "status": "error",
                        "message": "Media lacks complete workflow data (API + UI). Cannot save template.",
                    }
                ), 400

            # --- APPLY USER MODIFICATIONS BEFORE SAVING ---
            if save_mode == "modified":
                if raw_override:
                    if override_type == "api":
                        raw_api = raw_override
                    else:
                        raw_ui = raw_override
                elif modifications:
                    if raw_api and raw_api != "{}":
                        try:
                            api_data = json.loads(raw_api)
                            for mod in modifications.get("texts", []):
                                if mod["node_id"] in api_data:
                                    api_data[mod["node_id"]]["inputs"][mod.get("key", "text")] = mod["value"]
                            for mod in modifications.get("seeds", []) + modifications.get("numbers", []):
                                if mod["node_id"] in api_data:
                                    api_data[mod["node_id"]]["inputs"][mod["key"]] = mod["value"]
                            raw_api = json.dumps(api_data)
                        except Exception as e:
                            _logger.debug("handled a failure in api_remix_save_template", exc_info=True)
                            print(f"Save API mod error: {e}")

                    if raw_ui and raw_ui != "{}":
                        try:
                            ui_data = json.loads(raw_ui)
                            nodes = ui_data.get("nodes", []) if isinstance(ui_data, dict) else ui_data
                            all_mods = (
                                modifications.get("texts", [])
                                + modifications.get("seeds", [])
                                + modifications.get("numbers", [])
                            )
                            for mod in all_mods:
                                target_node = next((n for n in nodes if str(n.get("id")) == str(mod["node_id"])), None)
                                if target_node and "widgets_values" in target_node:
                                    orig_val_str = str(mod.get("orig_value", ""))
                                    is_numeric = False
                                    orig_val_float = 0.0
                                    try:
                                        orig_val_float = float(orig_val_str)
                                        is_numeric = True
                                    except ValueError:
                                        pass
                                    matched = False
                                    for i, w in enumerate(target_node["widgets_values"]):
                                        if str(w) == orig_val_str:
                                            target_node["widgets_values"][i] = mod["value"]
                                            matched = True
                                            break
                                        if is_numeric and isinstance(w, (int, float)):
                                            if abs(w - orig_val_float) < 0.0001:
                                                target_node["widgets_values"][i] = mod["value"]
                                                matched = True
                                                break
                                    if not matched:
                                        idx = mod.get("widget_index")
                                        if idx is not None and 0 <= idx < len(target_node["widgets_values"]):
                                            target_node["widgets_values"][idx] = mod["value"]
                            raw_ui = json.dumps(ui_data)
                        except Exception as e:
                            _logger.debug("handled a failure in api_remix_save_template", exc_info=True)
                            print(f"Save UI mod error: {e}")
            # --- END MODIFICATIONS ---

            api_data_to_save = json.loads(raw_api)
            ui_data_to_save = json.loads(raw_ui)

            clean_workflow_paths(api_data_to_save)
            clean_workflow_paths(ui_data_to_save)

            template_data = {
                "api": api_data_to_save,
                "ui": ui_data_to_save,
                "sg_meta": {
                    "source_file_id": source_file_id,
                    "custom_app": custom_app,
                    "favorite_nodes": favorite_nodes,
                },
            }

            os.makedirs(IMPORTED_WORKFLOWS_DIR, exist_ok=True)
            with open(os.path.join(IMPORTED_WORKFLOWS_DIR, safe_name), "w", encoding="utf-8") as f:
                json.dump(template_data, f)

            return jsonify({"status": "success", "message": "Template saved!"})
        except Exception as e:
            _logger.debug("handled a failure in api_remix_save_template", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/galleryout/api/remix/workflows/rename", methods=["POST"])
    @management_api_only
    def api_remix_rename_workflow():
        try:
            old_name = request.json.get("old_name")
            new_name = request.json.get("new_name")
            if not old_name or not new_name:
                return jsonify({"status": "error"}), 400
            safe_old = safe_media_filename(old_name, fallback="untitled")
            safe_new = safe_media_filename(new_name, fallback="untitled")
            if not safe_new.lower().endswith(".json"):
                safe_new += ".json"
            old_path = os.path.join(IMPORTED_WORKFLOWS_DIR, safe_old)
            new_path = os.path.join(IMPORTED_WORKFLOWS_DIR, safe_new)
            if not os.path.exists(old_path):
                return jsonify({"status": "error", "message": "Original template not found"}), 404
            if os.path.exists(new_path):
                return jsonify({"status": "error", "message": "A template with this name already exists"}), 400
            os.rename(old_path, new_path)
            return jsonify({"status": "success"})
        except Exception as e:
            _logger.debug("handled a failure in api_remix_rename_workflow", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/galleryout/api/remix/workflows/delete", methods=["POST"])
    @management_api_only
    def api_remix_delete_workflow():
        try:
            filename = request.json.get("filename")
            if not filename or not filename.lower().endswith(".json"):
                return jsonify({"status": "error"}), 400
            path = os.path.join(IMPORTED_WORKFLOWS_DIR, safe_media_filename(filename, fallback="untitled"))
            if os.path.exists(path):
                os.remove(path)
                return jsonify({"status": "success"})
            return jsonify({"status": "error", "message": "Not found"}), 404
        except Exception as e:
            _logger.debug("handled a failure in api_remix_delete_workflow", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    def _convert_ui_to_api(ui_data, object_info):
        nodes = ui_data.get("nodes", [])
        links = ui_data.get("links", [])
        link_map = {}
        for lnk in links:
            if len(lnk) >= 4:
                link_map[lnk[0]] = [str(lnk[1]), lnk[2]]
        api = {}
        for node in nodes:
            node_id = str(node.get("id"))
            node_type = node.get("type", "")
            if node_type in ("Note", "PrimitiveNode", "Reroute"):
                continue

            node_inputs_connected = node.get("inputs", [])
            widgets_values = node.get("widgets_values", [])
            inputs = {}
            linked_names = set()
            for inp in node_inputs_connected:
                link_id = inp.get("link")
                if link_id is not None and link_id in link_map:
                    inputs[inp["name"]] = link_map[link_id]
                    linked_names.add(inp["name"])

            if isinstance(widgets_values, dict):
                for k, v in widgets_values.items():
                    if isinstance(v, (str, int, float, bool)):
                        inputs[k] = v
            else:
                param_names = NODE_PARAM_NAMES.get(node_type, [])
                if param_names:
                    wv_list = list(widgets_values)
                    for i, w_val in enumerate(wv_list):
                        wname = param_names[i] if i < len(param_names) else f"widget_{i}"
                        if wname not in linked_names:
                            inputs[wname] = w_val
                elif node_type in object_info:
                    node_def = object_info[node_type]
                    required = node_def.get("input", {}).get("required", {})
                    optional = node_def.get("input", {}).get("optional", {})
                    hidden = node_def.get("input", {}).get("hidden", {})
                    all_inputs = list(required.items()) + list(optional.items())
                    hidden_names = set(hidden.keys())
                    widget_names = []
                    for inp_name, inp_def in all_inputs:
                        if inp_name in linked_names:
                            continue
                        inp_type = inp_def[0] if (isinstance(inp_def, (list, tuple)) and inp_def) else inp_def
                        is_connection = (
                            isinstance(inp_type, str)
                            and inp_type == inp_type.upper()
                            and inp_type not in ("INT", "FLOAT", "STRING", "BOOLEAN")
                        )
                        if not is_connection:
                            widget_names.append((inp_name, inp_name in hidden_names))
                    for inp_name in hidden_names:
                        if inp_name not in linked_names and not any(n == inp_name for n, _ in widget_names):
                            widget_names.append((inp_name, True))
                    wv_list = list(widgets_values)
                    for i, (wname, is_hidden) in enumerate(widget_names):
                        if i < len(wv_list) and not is_hidden:
                            inputs[wname] = wv_list[i]
                else:
                    wv_list = list(widgets_values)
                    for i, w_val in enumerate(wv_list):
                        wname = f"widget_{i}"
                        if wname not in linked_names:
                            inputs[wname] = w_val

            title = node.get("title") or node_type
            api[node_id] = {"class_type": node_type, "_meta": {"title": title}, "inputs": inputs}
        return api

    def _get_unified_workflow(file_id, workflow_override, companion_override, target_comfy_url=COMFYUI_SERVER_URL):
        if workflow_override:
            override_path = os.path.join(
                IMPORTED_WORKFLOWS_DIR, safe_media_filename(workflow_override, fallback="untitled")
            )
            if os.path.isfile(override_path):
                with open(override_path, encoding="utf-8") as wf_f:
                    tpl_data = json.load(wf_f)
                    raw_api = json.dumps(tpl_data.get("api", {}))
                    raw_ui = json.dumps(tpl_data.get("ui", {}))

                    # Convert UI to API if the template was an old JSON lacking API data
                    if raw_ui and raw_api == "{}":
                        try:
                            ui_data = json.loads(raw_ui)
                            object_info = {}
                            try:
                                info_req = urllib.request.Request(
                                    f"{target_comfy_url.rstrip('/')}/object_info",
                                    headers={"Content-Type": "application/json"},
                                )
                                with urllib.request.urlopen(info_req, timeout=3) as r:
                                    object_info = json.loads(r.read().decode("utf-8"))
                            except Exception:
                                _logger.debug("ignored a failure in _get_unified_workflow", exc_info=True)
                            converted_api = _convert_ui_to_api(ui_data, object_info)
                            if converted_api:
                                raw_api = json.dumps(converted_api)
                        except Exception:
                            _logger.debug("ignored a failure in _get_unified_workflow", exc_info=True)

                    sg_meta = tpl_data.get("sg_meta", {})
                    return raw_api, raw_ui, sg_meta, None
            return None, None, {}, "Template file not found."

        target_path = (
            companion_override
            if companion_override and os.path.isfile(companion_override)
            else get_file_info_from_db(file_id)["path"]
        )
        raw_api = extract_workflow(target_path, target_type="api")
        raw_ui = extract_workflow(target_path, target_type="ui")

        if not raw_api and not raw_ui:
            stem = os.path.splitext(target_path)[0]
            for ext in (".png", ".PNG"):
                companion = stem + ext
                if os.path.isfile(companion):
                    raw_api = extract_workflow(companion, target_type="api")
                    raw_ui = extract_workflow(companion, target_type="ui")
                    break
        return raw_api, raw_ui, {}, None

    @app.route("/galleryout/api/remix/info/<string:file_id>")
    def api_remix_info(file_id):
        if should_strip_metadata():
            return jsonify(
                {"status": "error", "message": "Security Policy: Access to remix workflow is restricted for your role."}
            ), 403

        try:
            workflow_override = request.args.get("workflow_file")
            companion_override = request.args.get("companion")

            raw_api, raw_ui, sg_meta, err = _get_unified_workflow(file_id, workflow_override, companion_override)
            if err:
                return jsonify({"status": "error", "message": err}), 404

            extract = None
            if raw_api:
                extract = parse_workflow(raw_api, "api", raw_ui)
                if (
                    len(extract["texts"]) == 0
                    and len(extract["seeds"]) == 0
                    and len(extract["images"]) == 0
                    and len(extract["numbers"]) == 0
                ):
                    extract = None

            if not extract and raw_ui:
                extract = parse_workflow(raw_ui, "ui", raw_ui)

            if not extract or (
                len(extract["texts"]) == 0 and len(extract["seeds"]) == 0 and len(extract["images"]) == 0
            ):
                return jsonify({"status": "error", "message": "No editable fields found in this file format."}), 404

            try:
                _api_check = json.loads(raw_api) if raw_api else None
                if isinstance(_api_check, dict):
                    _api_nodes = [v for v in _api_check.values() if isinstance(v, dict) and "class_type" in v]
                    extract["has_api"] = len(_api_nodes) > 0
                else:
                    extract["has_api"] = False
            except Exception:
                _logger.debug("handled a failure in api_remix_info", exc_info=True)
                extract["has_api"] = False

            try:
                _ui_check = json.loads(raw_ui) if raw_ui else None
                _ui_nodes = (
                    _ui_check.get("nodes", [])
                    if isinstance(_ui_check, dict)
                    else (_ui_check if isinstance(_ui_check, list) else [])
                )
                extract["has_ui"] = bool(raw_ui) and len(_ui_nodes) > 0
            except Exception:
                _logger.debug("handled a failure in api_remix_info", exc_info=True)
                extract["has_ui"] = False

            extract["default_comfy_url"] = COMFYUI_SERVER_URL
            extract["custom_app"] = sg_meta.get("custom_app", [])
            extract["favorite_nodes"] = sg_meta.get("favorite_nodes", {})
            if extract["custom_app"]:
                extract["has_custom_mode"] = True
            extract["raw_api_json"] = raw_api or ""
            extract["raw_ui_json"] = raw_ui or ""
            if "raw_json" in extract:
                del extract["raw_json"]
            return jsonify({"status": "success", "data": extract})
        except HTTPException as stop:
            return answer_an_abort_readably(stop)
        except Exception as e:
            _logger.debug("handled a failure in api_remix_info", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    def _read_safetensors_metadata(filepath):
        try:
            with open(filepath, "rb") as f:
                header_size_bytes = f.read(8)
                if len(header_size_bytes) < 8:
                    return {}
                header_size = struct.unpack("<Q", header_size_bytes)[0]
                if header_size > 10000000:
                    return {}
                header_json = f.read(header_size).decode("utf-8", errors="ignore")
                header = json.loads(header_json)
                return header.get("__metadata__", {})
        except Exception:
            _logger.debug("handled a failure in _read_safetensors_metadata", exc_info=True)
            return {}

    def _guess_architecture(filename, metadata):
        arch = "unknown"
        if metadata:
            ss_base = metadata.get("ss_base_model_version", "").lower()
            modelspec = metadata.get("modelspec.architecture", "").lower()
            if "sdxl" in ss_base or "sdxl" in modelspec:
                arch = "sdxl"
            elif "flux" in ss_base or "flux" in modelspec:
                arch = "flux"
            elif "sd_1_5" in ss_base or "sd15" in modelspec or "sd1.5" in ss_base:
                arch = "sd1.5"
            elif "sd3" in ss_base or "sd3" in modelspec:
                arch = "sd3"
            elif "hunyuan" in ss_base or "hunyuan" in modelspec:
                arch = "hunyuan"
            elif "wan" in ss_base or "wan" in modelspec:
                arch = "wan"

        if arch == "unknown" and filename:
            name_lower = filename.lower()
            if "flux" in name_lower:
                arch = "flux"
            elif "sdxl" in name_lower or "/xl/" in name_lower or "xl_" in name_lower or "_xl" in name_lower:
                arch = "sdxl"
            elif "sd3" in name_lower:
                arch = "sd3"
            elif "1.5" in name_lower or "15" in name_lower or "v1-5" in name_lower:
                arch = "sd1.5"
        return arch

    @app.route("/galleryout/api/remix/job_status/<string:job_id>", methods=["POST"])
    @management_api_only
    def api_remix_job_status(job_id):
        target_url = request.json.get("target_url", COMFYUI_SERVER_URL).strip()
        try:
            # 1. Check History
            try:
                req = urllib.request.Request(
                    f"{target_url.rstrip('/')}/history/{job_id}", headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=3) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    if job_id in data:
                        status_obj = data[job_id].get("status", {})
                        if status_obj.get("status_str") == "error":
                            msgs = status_obj.get("messages", [])
                            err_detail = "Execution Error"
                            for m in msgs:
                                if m[0] == "execution_error":
                                    err_detail = (
                                        f"[Node {m[1].get('node_id')}]"
                                        f" {m[1].get('exception_type')}:"
                                        f" {m[1].get('exception_message')}"
                                    )
                            return jsonify({"status": "error", "message": err_detail})
                        if status_obj.get("completed"):
                            return jsonify({"status": "completed"})
            except Exception:
                _logger.debug("ignored a failure in api_remix_job_status", exc_info=True)

            # 2. Check Queue (Using item[1] to match prompt_id correctly)
            req_queue = urllib.request.Request(
                f"{target_url.rstrip('/')}/queue", headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req_queue, timeout=3) as response:
                q_data = json.loads(response.read().decode("utf-8"))

                for item in q_data.get("queue_running", []):
                    if len(item) > 1 and str(item[1]) == str(job_id):
                        return jsonify({"status": "running"})

                for idx, item in enumerate(q_data.get("queue_pending", [])):
                    if len(item) > 1 and str(item[1]) == str(job_id):
                        return jsonify({"status": "pending", "position": idx + 1})

            return jsonify({"status": "vanished"})
        except Exception as e:
            _logger.debug("handled a failure in api_remix_job_status", exc_info=True)
            print(f"Remix job-status poll error: {e}")
            return jsonify({"status": "connection_error"})

    @app.route("/galleryout/api/remix/comfy_console_peek", methods=["POST"])
    @management_api_only
    def api_remix_console_peek():
        target_url = request.json.get("target_url", COMFYUI_SERVER_URL).strip()
        try:
            req = urllib.request.Request(
                f"{target_url.rstrip('/')}/history", headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3) as response:
                history_data = json.loads(response.read().decode("utf-8"))

            if not history_data:
                return jsonify({"status": "empty", "message": "ComfyUI history is empty."})

            last_job_id = list(history_data.keys())[-1]
            last_job = history_data[last_job_id]
            status_obj = last_job.get("status", {})

            if status_obj.get("status_str") == "error":
                msgs = status_obj.get("messages", [])
                err_detail = "Unknown execution error."
                for m in msgs:
                    if m[0] == "execution_error":
                        err_detail = (
                            f"[Node {m[1].get('node_id')}]"
                            f" {m[1].get('exception_type')}:"
                            f" {m[1].get('exception_message')}"
                        )
                return jsonify({"status": "found_error", "message": err_detail})

            return jsonify({"status": "clean"})
        except Exception as e:
            _logger.debug("handled a failure in api_remix_console_peek", exc_info=True)
            print(f"Remix console peek error: {e}")
            return jsonify({"status": "error"})

    @app.route("/galleryout/api/remix/lora_intelligence", methods=["POST"])
    @management_api_only
    def api_remix_lora_intelligence():
        try:
            data = request.json or {}
            action = data.get("action")

            if not action:
                action = "triggers" if "lora" in data else "matchmaker"

            # --- ACTION 1: TRIGGER MINING & CIVITAI METADATA ---
            if action == "triggers":
                lora_name = data.get("lora")
                res = {"db_triggers": [], "civitai_triggers": [], "preview_image": None, "civitai_url": None}
                if lora_name:
                    try:
                        if os.path.exists(LORAS_PATH):
                            # Normalize path to handle subfolders perfectly
                            lora_norm = os.path.normpath(lora_name)
                            clean_lora_name = os.path.splitext(lora_norm)[0]

                            full_base_path = os.path.join(LORAS_PATH, clean_lora_name)
                            full_raw_path = os.path.join(LORAS_PATH, lora_norm)

                            json_paths = [
                                full_base_path + ".civitai.info",
                                full_base_path + ".metadata.json",
                                full_base_path + ".info",
                                full_base_path + ".json",
                                full_raw_path + ".civitai.info",
                                full_raw_path + ".metadata.json",
                                full_raw_path + ".info",
                                full_raw_path + ".json",
                            ]

                            civitai_id = None
                            for jp in json_paths:
                                if os.path.exists(jp):
                                    try:
                                        with open(jp, encoding="utf-8") as f:
                                            meta = json.load(f)
                                        if isinstance(meta, dict):

                                            def _get_all_tw(d):
                                                found = []
                                                if isinstance(d, dict):
                                                    tw = (
                                                        d.get("trainedWords")
                                                        or d.get("trained_words")
                                                        or d.get("activation_text")
                                                    )
                                                    if tw:
                                                        if isinstance(tw, list):
                                                            found.extend(tw)
                                                        elif isinstance(tw, str):
                                                            found.append(tw)
                                                    if "modelVersions" in d and isinstance(d["modelVersions"], list):
                                                        for mv in d["modelVersions"]:
                                                            found.extend(_get_all_tw(mv))
                                                    if "civitai" in d and isinstance(d["civitai"], dict):
                                                        found.extend(_get_all_tw(d["civitai"]))
                                                return found

                                            if not res["civitai_triggers"]:
                                                raw_tw = _get_all_tw(meta)
                                                for t in raw_tw:
                                                    if isinstance(t, str):
                                                        res["civitai_triggers"].extend(
                                                            [x.strip() for x in t.split(",") if x.strip()]
                                                        )
                                                res["civitai_triggers"] = list(
                                                    dict.fromkeys(res["civitai_triggers"])
                                                )  # deduplicate

                                            if not civitai_id:
                                                civitai_id = meta.get("modelId") or meta.get("id")
                                                if (
                                                    not civitai_id
                                                    and "civitai" in meta
                                                    and isinstance(meta["civitai"], dict)
                                                ):
                                                    civitai_id = meta["civitai"].get("modelId") or meta["civitai"].get(
                                                        "id"
                                                    )
                                    except Exception:
                                        _logger.debug("ignored a failure in api_remix_lora_intelligence", exc_info=True)

                                if res["civitai_triggers"] and civitai_id:
                                    break

                            if not civitai_id:
                                st_path = os.path.join(LORAS_PATH, lora_norm)
                                if os.path.exists(st_path):
                                    st_meta = _read_safetensors_metadata(st_path)
                                    if st_meta:
                                        civitai_id = st_meta.get("modelspec.civitai_model_id") or st_meta.get(
                                            "ss_civitai_model_id"
                                        )

                            if civitai_id:
                                res["civitai_url"] = f"https://civitai.com/models/{civitai_id}"

                            img_paths = [
                                full_base_path + ".preview.png",
                                full_base_path + ".png",
                                full_base_path + ".jpg",
                                full_base_path + ".jpeg",
                                full_raw_path + ".preview.png",
                                full_raw_path + ".png",
                                full_raw_path + ".jpg",
                                full_raw_path + ".jpeg",
                            ]
                            for ip in img_paths:
                                if os.path.exists(ip):
                                    with open(ip, "rb") as f:
                                        encoded = base64.b64encode(f.read()).decode("utf-8")
                                        mime = "image/png" if "png" in ip.lower() else "image/jpeg"
                                        res["preview_image"] = f"data:{mime};base64,{encoded}"
                                    break
                    except Exception as e:
                        _logger.debug("handled a failure in api_remix_lora_intelligence", exc_info=True)
                        print(f"LoRA Synergy Error (Passive): {e}")

                    with get_db_connection() as conn:
                        norm_lora_query = lora_name.lower().replace("\\", "/")
                        rows = conn.execute(
                            "SELECT workflow_prompt FROM files WHERE workflow_files LIKE ?", (f"%{norm_lora_query}%",)
                        ).fetchall()
                        word_counts = {}
                        ignore_words = {
                            "masterpiece",
                            "best",
                            "quality",
                            "highres",
                            "high",
                            "resolution",
                            "intricate",
                            "details",
                            "1girl",
                            "solo",
                            "text",
                            "watermark",
                        }
                        for r in rows:
                            prompt = r["workflow_prompt"]
                            if prompt:
                                tags = [t.strip().lower() for t in prompt.split(",")]
                                for t in tags:
                                    clean_t = re.sub(r"[()\[\]:]|[0-9.]+", "", t).strip()
                                    if len(clean_t) > 3 and clean_t not in ignore_words and len(clean_t.split()) <= 3:
                                        word_counts[clean_t] = word_counts.get(clean_t, 0) + 1

                        civitai_lower = [t.lower() for t in res["civitai_triggers"]]
                        sorted_triggers = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
                        res["db_triggers"] = [k for k, v in sorted_triggers[:15] if k.lower() not in civitai_lower]

                return jsonify({"status": "success", "data": res})

            # --- ACTION 2: MATCHMAKER ---
            if action == "matchmaker":
                ckpt_name = data.get("checkpoint")
                all_loras = data.get("all_loras", [])

                res = {"perfect_match": [], "proven_match": [], "possible_match": [], "incompatible": []}

                ckpt_arch = "unknown"
                if ckpt_name and ckpt_name != "Unknown":
                    ckpt_paths = [os.path.join(CHECKPOINTS_PATH, ckpt_name), os.path.join(UNET_PATH, ckpt_name)]
                    for cp in ckpt_paths:
                        if os.path.exists(cp) and cp.endswith(".safetensors"):
                            meta = _read_safetensors_metadata(cp)
                            ckpt_arch = _guess_architecture(ckpt_name, meta)
                            break
                    if ckpt_arch == "unknown":
                        ckpt_arch = _guess_architecture(ckpt_name, {})

                proven_loras_norm = set()
                with get_db_connection() as conn:
                    if ckpt_name and ckpt_name != "Unknown":
                        norm_ckpt = ckpt_name.lower().replace("\\", "/")
                        rows = conn.execute(
                            "SELECT workflow_files FROM files WHERE workflow_files LIKE ?", (f"%{norm_ckpt}%",)
                        ).fetchall()
                        for r in rows:
                            wf_files = r["workflow_files"].split(" ||| ")
                            for f in wf_files:
                                if (
                                    f != norm_ckpt
                                    and f.endswith((".safetensors", ".pt", ".ckpt"))
                                    and ("lora" in f or "/" in f)
                                ):
                                    proven_loras_norm.add(f.replace("\\", "/"))

                for lora in all_loras:
                    norm_lora = lora.lower().replace("\\", "/")
                    if norm_lora in proven_loras_norm:
                        res["proven_match"].append(lora)
                        continue

                    lora_path = os.path.join(LORAS_PATH, norm_lora)
                    lora_arch = "unknown"

                    if os.path.exists(lora_path) and norm_lora.endswith(".safetensors"):
                        meta = _read_safetensors_metadata(lora_path)
                        lora_arch = _guess_architecture(norm_lora, meta)
                    else:
                        lora_arch = _guess_architecture(norm_lora, {})

                    if ckpt_arch != "unknown" and lora_arch == ckpt_arch:
                        res["perfect_match"].append(lora)
                    elif lora_arch != "unknown" and ckpt_arch not in ("unknown", lora_arch):
                        res["incompatible"].append(lora)
                    else:
                        res["possible_match"].append(lora)

                return jsonify({"status": "success", "data": res, "detected_arch": ckpt_arch.upper()})

            return jsonify({"status": "error", "message": "Unknown action."}), 400

        except Exception as e:
            _logger.debug("handled a failure in api_remix_lora_intelligence", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/galleryout/api/remix/submit", methods=["POST"])
    @management_api_only
    def api_remix_submit():
        try:
            file_id = request.form.get("file_id")
            action_req = request.form.get("action_type", "api")
            modifications = json.loads(request.form.get("modifications", "{}"))
            wf_type = request.form.get("workflow_type", "api")
            target_comfy_url = request.form.get("target_url", COMFYUI_SERVER_URL).strip()

            file_name = request.form.get("file_name", "workflow.json")
            base_name = os.path.splitext(file_name)[0]

            workflow_override = request.form.get("workflow_file")
            companion_path = request.form.get("companion_path")

            raw_api, raw_ui, _, err = _get_unified_workflow(
                file_id, workflow_override, companion_path, target_comfy_url
            )
            if err:
                return jsonify({"status": "error", "message": err}), 400
            raw_json = raw_api if wf_type == "api" else raw_ui
            if not raw_json:
                raw_json = raw_ui if wf_type == "api" else raw_api

            if not raw_json:
                return jsonify({"status": "error", "message": "Could not read workflow data."}), 400
            raw_override = request.form.get("raw_override")
            if raw_override:
                wf_data = json.loads(raw_override)
                # Nodepad Patch: Keep media node data so backend saves the file physically!
                modifications["texts"] = []
                modifications["seeds"] = []
                modifications["numbers"] = []
            else:
                wf_data = json.loads(raw_json)

            if wf_type == "api":
                for mod in modifications.get("texts", []):
                    if mod["node_id"] in wf_data:
                        wf_data[mod["node_id"]]["inputs"][mod.get("key", "text")] = mod["value"]
                for mod in modifications.get("seeds", []) + modifications.get("numbers", []):
                    if mod["node_id"] in wf_data:
                        wf_data[mod["node_id"]]["inputs"][mod["key"]] = mod["value"]
                if "image_upload" in request.files and modifications.get("image_node_id"):
                    img_file = request.files["image_upload"]
                    if img_file.filename:
                        filename = safe_media_filename("remix_" + img_file.filename)
                        # Never clobber an input already queued for another
                        # generation: two remixes of same-named sources would
                        # otherwise overwrite each other's bytes, and ComfyUI
                        # would render whichever landed last. The workflow is
                        # pointed at what was actually written.
                        saved_path = _get_unique_filepath(BASE_INPUT_PATH, filename)
                        img_file.save(saved_path)
                        filename = os.path.basename(saved_path)
                        img_node_id = modifications["image_node_id"]
                        if img_node_id in wf_data:
                            wf_data[img_node_id]["inputs"][modifications.get("image_key", "image")] = filename
            else:
                nodes = wf_data.get("nodes", []) if isinstance(wf_data, dict) else wf_data
                all_mods = (
                    modifications.get("texts", []) + modifications.get("seeds", []) + modifications.get("numbers", [])
                )
                for mod in all_mods:
                    target_node = next((n for n in nodes if str(n.get("id")) == str(mod["node_id"])), None)
                    if target_node and "widgets_values" in target_node:
                        orig_val_str = str(mod.get("orig_value", ""))
                        is_numeric = False
                        orig_val_float = 0.0
                        try:
                            orig_val_float = float(orig_val_str)
                            is_numeric = True
                        except ValueError:
                            pass
                        matched = False
                        for i, w in enumerate(target_node["widgets_values"]):
                            if str(w) == orig_val_str:
                                target_node["widgets_values"][i] = mod["value"]
                                matched = True
                                break
                            if is_numeric and isinstance(w, (int, float)):
                                if abs(w - orig_val_float) < 0.0001:
                                    target_node["widgets_values"][i] = mod["value"]
                                    matched = True
                                    break
                        if not matched:
                            idx = mod.get("widget_index")
                            if idx is not None and 0 <= idx < len(target_node["widgets_values"]):
                                target_node["widgets_values"][idx] = mod["value"]
                if "image_upload" in request.files and modifications.get("image_node_id"):
                    img_file = request.files["image_upload"]
                    if img_file.filename:
                        filename = safe_media_filename("remix_" + img_file.filename)
                        # Never clobber an input already queued for another
                        # generation: two remixes of same-named sources would
                        # otherwise overwrite each other's bytes, and ComfyUI
                        # would render whichever landed last. The workflow is
                        # pointed at what was actually written.
                        saved_path = _get_unique_filepath(BASE_INPUT_PATH, filename)
                        img_file.save(saved_path)
                        filename = os.path.basename(saved_path)
                        target_node = next(
                            (n for n in nodes if str(n.get("id")) == str(modifications["image_node_id"])), None
                        )
                        if target_node and "widgets_values" in target_node:
                            orig_img = str(modifications.get("image_orig_value", ""))
                            matched = False
                            for i, w in enumerate(target_node["widgets_values"]):
                                if str(w) == orig_img:
                                    target_node["widgets_values"][i] = filename
                                    matched = True
                                    break
                            if not matched:
                                idx = modifications.get("image_widget_index")
                                if idx is not None and 0 <= idx < len(target_node["widgets_values"]):
                                    target_node["widgets_values"][idx] = filename

            # Clean multiple slashes from all string values before queuing
            clean_workflow_paths(wf_data)

            if action_req in ["copy", "download"]:
                modified_json_string = json.dumps(wf_data, indent=2)
                headers = {"X-Workflow-Type": wf_type}
                if action_req == "download":
                    headers["Content-Disposition"] = f'attachment;filename="remixed_{base_name}.json"'
                return Response(modified_json_string, mimetype="application/json", headers=headers)

            if action_req == "api":
                if wf_type == "ui":
                    return jsonify(
                        {
                            "status": "error",
                            "message": "Cannot queue UI-format workflow via API. Use Copy/Download instead.",
                        }
                    ), 400
                if not target_comfy_url:
                    return jsonify({"status": "error", "message": "ComfyUI URL is required."}), 400
                try:
                    ping_req = urllib.request.Request(
                        f"{target_comfy_url.rstrip('/')}/system_stats", headers={"Content-Type": "application/json"}
                    )
                    urllib.request.urlopen(ping_req, timeout=4)
                except urllib.error.URLError:
                    return jsonify({"status": "error", "message": f"Cannot reach ComfyUI at {target_comfy_url}."}), 502
                except Exception:
                    _logger.debug("ignored a failure in api_remix_submit", exc_info=True)

                invalid_nodes = []
                for node_id, node_val in wf_data.items():
                    if not isinstance(node_val, dict) or "class_type" not in node_val or "inputs" not in node_val:
                        invalid_nodes.append(str(node_id))
                if invalid_nodes:
                    return jsonify(
                        {"status": "error", "message": "Workflow data is not in valid ComfyUI API format."}
                    ), 400

                payload = json.dumps({"prompt": wf_data}).encode("utf-8")
                req = urllib.request.Request(
                    f"{target_comfy_url.rstrip('/')}/prompt", data=payload, headers={"Content-Type": "application/json"}
                )
                try:
                    with urllib.request.urlopen(req, timeout=5) as response:
                        resp_data = json.loads(response.read().decode("utf-8"))

                        node_errors = resp_data.get("node_errors")
                        error_obj = resp_data.get("error")

                        if (node_errors and isinstance(node_errors, dict) and len(node_errors) > 0) or error_obj:
                            err_msg = "<strong>ComfyUI Validation Error (Partial Execution Blocked):</strong>"
                            if node_errors and isinstance(node_errors, dict):
                                for nid, ndata in node_errors.items():
                                    ctype = ndata.get("class_type", "Unknown Node")
                                    errs = ndata.get("errors", [{}])
                                    for err in errs:
                                        err_msg += node_error_html(ctype, nid, err)
                            elif error_obj:
                                err_msg += (
                                    f" {error_obj.get('message', 'Unknown Error')}"
                                    if isinstance(error_obj, dict)
                                    else f" {error_obj!s}"
                                )
                            return jsonify({"status": "error", "message": err_msg}), 400

                        job_id = resp_data.get("prompt_id", "Unknown")
                        return jsonify(
                            {
                                "status": "success",
                                "action": "queued",
                                "job_id": job_id,
                                "message": f"Job queued! ID: {job_id}",
                            }
                        )
                except urllib.error.HTTPError as e:
                    err_msg = f"ComfyUI rejected the workflow: HTTP {e.code}"
                    try:
                        raw_body = e.read().decode("utf-8")
                        err_body = json.loads(raw_body)
                        if "error" in err_body or "node_errors" in err_body:
                            err_msg = (
                                "<strong>ComfyUI Validation Error:</strong> "
                                f"{err_body.get('error', {}).get('message', '')}"
                            )
                            node_errors = err_body.get("node_errors") or err_body.get("error", {}).get(
                                "node_errors", {}
                            )
                            if node_errors and isinstance(node_errors, dict):
                                for nid, ndata in node_errors.items():
                                    ctype = ndata.get("class_type", "Unknown Node")
                                    errs = ndata.get("errors", [{}])
                                    for err in errs:
                                        err_msg += node_error_html(ctype, nid, err)
                    except Exception:
                        _logger.debug("ignored a failure in api_remix_submit", exc_info=True)
                    return jsonify({"status": "error", "message": err_msg}), 400
                except urllib.error.URLError:
                    return jsonify({"status": "error", "message": "Failed to connect to ComfyUI."}), 502

        except Exception as e:
            _logger.debug("handled a failure in api_remix_submit", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    def _convert_ui_to_api(ui_data, object_info):
        nodes = ui_data.get("nodes", [])
        links = ui_data.get("links", [])
        link_map = {}
        for lnk in links:
            if len(lnk) >= 4:
                link_map[lnk[0]] = [str(lnk[1]), lnk[2]]
        api = {}
        for node in nodes:
            node_id = str(node.get("id"))
            node_type = node.get("type", "")
            if node_type in ("Note", "PrimitiveNode", "Reroute"):
                continue
            if node_type not in object_info:
                continue
            node_inputs_connected = node.get("inputs", [])
            widgets_values = node.get("widgets_values", [])
            inputs = {}
            linked_names = set()
            for inp in node_inputs_connected:
                link_id = inp.get("link")
                if link_id is not None and link_id in link_map:
                    inputs[inp["name"]] = link_map[link_id]
                    linked_names.add(inp["name"])
            if isinstance(widgets_values, dict):
                for k, v in widgets_values.items():
                    if isinstance(v, (str, int, float, bool)):
                        inputs[k] = v
            elif node_type in object_info:
                node_def = object_info[node_type]
                required = node_def.get("input", {}).get("required", {})
                optional = node_def.get("input", {}).get("optional", {})
                hidden = node_def.get("input", {}).get("hidden", {})
                all_inputs = list(required.items()) + list(optional.items())
                hidden_names = set(hidden.keys())
                widget_names = []
                for inp_name, inp_def in all_inputs:
                    if inp_name in linked_names:
                        continue
                    inp_type = inp_def[0] if (isinstance(inp_def, (list, tuple)) and inp_def) else inp_def
                    is_connection = (
                        isinstance(inp_type, str)
                        and inp_type == inp_type.upper()
                        and inp_type not in ("INT", "FLOAT", "STRING", "BOOLEAN")
                    )
                    if not is_connection:
                        widget_names.append((inp_name, inp_name in hidden_names))
                for inp_name in hidden_names:
                    if inp_name not in linked_names and not any(n == inp_name for n, _ in widget_names):
                        widget_names.append((inp_name, True))
                wv_list = list(widgets_values)
                for i, (wname, is_hidden) in enumerate(widget_names):
                    if i < len(wv_list) and not is_hidden:
                        inputs[wname] = wv_list[i]
            title = node.get("title") or node_type
            api[node_id] = {"class_type": node_type, "_meta": {"title": title}, "inputs": inputs}
        return api

    @app.route("/galleryout/api/remix/companion/<string:file_id>")
    @management_api_only
    def api_remix_companion(file_id):
        try:
            info = get_file_info_from_db(file_id)
            file_path = info["path"]
            stem = os.path.splitext(file_path)[0]
            for ext in (".png", ".PNG"):
                companion_path = stem + ext
                if os.path.isfile(companion_path):
                    candidate_api = extract_workflow(companion_path, target_type="api")
                    if candidate_api:
                        try:
                            _check = json.loads(candidate_api)
                            _nodes = [v for v in _check.values() if isinstance(v, dict) and "class_type" in v]
                            if _nodes:
                                companion_name = os.path.basename(companion_path)
                                return jsonify(
                                    {
                                        "status": "success",
                                        "companion_path": companion_path,
                                        "companion_name": companion_name,
                                    }
                                )
                        except Exception:
                            _logger.debug("ignored a failure in api_remix_companion", exc_info=True)
            return jsonify({"status": "error", "message": "No companion PNG found."}), 404
        except HTTPException as stop:
            return answer_an_abort_readably(stop)
        except Exception as e:
            _logger.debug("handled a failure in api_remix_companion", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/galleryout/api/remix/autofix", methods=["POST"])
    @management_api_only
    def api_remix_autofix():
        try:
            file_id = request.form.get("file_id")
            companion_path = request.form.get("companion_path")
            target_comfy_url = request.form.get("target_url", COMFYUI_SERVER_URL).strip()
            workflow_override = request.form.get("workflow_file")

            if not target_comfy_url:
                return jsonify({"status": "error", "message": "ComfyUI URL is required."}), 400

            raw_api, raw_ui, _, err = _get_unified_workflow(
                file_id, workflow_override, companion_path, target_comfy_url
            )
            if err:
                return jsonify({"status": "error", "message": err}), 404

            if not raw_ui and not raw_api:
                return jsonify({"status": "error", "message": "No workflow data found."}), 400
            object_info = {}
            try:
                info_req = urllib.request.Request(
                    f"{target_comfy_url.rstrip('/')}/object_info", headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(info_req, timeout=5) as r:
                    object_info = json.loads(r.read().decode("utf-8"))
            except Exception:
                _logger.debug("ignored a failure in api_remix_autofix", exc_info=True)
            if raw_api:
                api_wf = json.loads(raw_api)
            else:
                ui_data = json.loads(raw_ui)
                nodes = ui_data.get("nodes", []) if isinstance(ui_data, dict) else []
                if not nodes:
                    return jsonify({"status": "error", "message": "UI workflow contains no nodes."}), 400
                api_wf = _convert_ui_to_api(ui_data, object_info)
            clean_workflow_paths(api_wf)
            if request.form.get("debug") == "1":
                return jsonify({"status": "debug", "converted": api_wf, "object_info_keys": list(object_info.keys())})
            payload = json.dumps({"prompt": api_wf}).encode("utf-8")
            req = urllib.request.Request(
                f"{target_comfy_url.rstrip('/')}/prompt", data=payload, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    resp_data = json.loads(response.read().decode("utf-8"))
                    job_id = resp_data.get("prompt_id", "Unknown")
                    return jsonify({"status": "success", "message": f"Autofix succeeded! Job ID: {job_id}"})
            except urllib.error.HTTPError as e:
                return jsonify({"status": "error", "message": f"ComfyUI rejected it: HTTP {e.code}"}), 400
            except urllib.error.URLError:
                return jsonify({"status": "error", "message": "Failed to connect to ComfyUI."}), 502
        except Exception as e:
            _logger.debug("handled a failure in api_remix_autofix", exc_info=True)
            return jsonify({"status": "error", "message": f"Autofix error: {e!s}"}), 500

    @app.route("/galleryout/api/remix/autofix_apply", methods=["POST"])
    @management_api_only
    def api_remix_autofix_apply():
        try:
            file_id = request.form.get("file_id")
            companion_path = request.form.get("companion_path")
            choices_json = request.form.get("choices")
            target_comfy_url = request.form.get("target_url", COMFYUI_SERVER_URL).strip()
            workflow_override = request.form.get("workflow_file")

            if not choices_json:
                return jsonify({"status": "error", "message": "Missing data."}), 400

            raw_api, _raw_ui, _, err = _get_unified_workflow(
                file_id, workflow_override, companion_path, target_comfy_url
            )
            if err:
                return jsonify({"status": "error", "message": err}), 404

            if not raw_api:
                return jsonify({"status": "error", "message": "Could not read API workflow."}), 400
            api_wf = json.loads(raw_api)
            choices = json.loads(choices_json)
            applied = []
            for c in choices:
                n_id = c["node_id"]
                inp_name = c["inp_name"]
                chosen = c["chosen"]
                if n_id in api_wf:
                    api_wf[n_id]["inputs"][inp_name] = chosen
                    applied.append(f"[Node {n_id} ({c.get('node_title', '')})] '{inp_name}' -> {chosen!r}")

            clean_workflow_paths(api_wf)
            payload = json.dumps({"prompt": api_wf}).encode("utf-8")
            req = urllib.request.Request(
                f"{target_comfy_url.rstrip('/')}/prompt", data=payload, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    resp_data = json.loads(response.read().decode("utf-8"))
                    node_errors = resp_data.get("node_errors")
                    error_obj = resp_data.get("error")

                    if (node_errors and isinstance(node_errors, dict) and len(node_errors) > 0) or error_obj:
                        err_msg = "<strong>ComfyUI Validation Error:</strong>"
                        if node_errors and isinstance(node_errors, dict):
                            for nid, ndata in node_errors.items():
                                ctype = ndata.get("class_type", "Unknown Node")
                                errs = ndata.get("errors", [{}])
                                for err in errs:
                                    err_msg += node_error_html(ctype, nid, err)
                        elif error_obj:
                            err_msg += (
                                f" {error_obj.get('message', 'Unknown Error')}"
                                if isinstance(error_obj, dict)
                                else f" {error_obj!s}"
                            )
                        return jsonify({"status": "error", "message": err_msg}), 400

                    job_id = resp_data.get("prompt_id", "Unknown")
                    return jsonify(
                        {
                            "status": "success",
                            "job_id": job_id,
                            "message": (
                                f"Queued successfully! Job ID: {job_id}"
                                f"<br><small>Applied: {' | '.join(applied)}</small>"
                            ),
                        }
                    )
            except urllib.error.HTTPError as e:
                err_msg = f"ComfyUI rejected even after corrections: HTTP {e.code}"
                try:
                    raw_body = e.read().decode("utf-8")
                    err_body = json.loads(raw_body)
                    if "error" in err_body or "node_errors" in err_body:
                        err_msg = (
                            f"<strong>ComfyUI Validation Error:</strong> {err_body.get('error', {}).get('message', '')}"
                        )
                        node_errors = err_body.get("node_errors") or err_body.get("error", {}).get("node_errors", {})
                        if node_errors and isinstance(node_errors, dict):
                            for nid, ndata in node_errors.items():
                                ctype = ndata.get("class_type", "Unknown Node")
                                errs = ndata.get("errors", [{}])
                                for err in errs:
                                    err_msg += node_error_html(ctype, nid, err)
                except Exception:
                    _logger.debug("ignored a failure in api_remix_autofix_apply", exc_info=True)
                return jsonify({"status": "error", "message": err_msg}), 400
            except urllib.error.URLError:
                return jsonify({"status": "error", "message": "Cannot reach ComfyUI."}), 502
        except Exception as e:
            _logger.debug("handled a failure in api_remix_autofix_apply", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500


_register_remix_routes_inline()


@app.route("/galleryout/api/collections/upload_note", methods=["POST"])
@management_api_only
def upload_collection_note():
    coll_id = request.form.get("collection_id")
    if not coll_id:
        return jsonify({"status": "error", "message": "Missing collection ID"}), 400

    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    file = request.files["file"]

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".txt", ".md"]:
        return jsonify({"status": "error", "message": "Only .txt and .md files are allowed as notes."}), 400

    notes_dir = os.path.join(BASE_SMARTGALLERY_PATH, ".collection_notes")
    os.makedirs(notes_dir, exist_ok=True)

    safe_name = f"note_c{coll_id}_{int(time.time())}_{safe_media_filename(file.filename)}"
    dest_path = os.path.join(notes_dir, safe_name)

    try:
        file.save(dest_path)
        mtime = os.path.getmtime(dest_path)
        file_id = content_digest(dest_path)
        file_size = os.path.getsize(dest_path)

        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO files (id, path, mtime, name, type, size)
                VALUES (?, ?, ?, ?, 'document', ?)
            """,
                (file_id, dest_path, mtime, file.filename, file_size),
            )

            conn.execute(
                """
                INSERT OR IGNORE INTO collection_files (collection_id, file_id, added_at)
                VALUES (?, ?, ?)
            """,
                (int(coll_id), file_id, time.time()),
            )

            conn.commit()

        return jsonify({"status": "success", "message": "Note added successfully to the collection."})
    except Exception as e:
        _logger.debug("handled a failure in upload_collection_note", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# --- SMART WORKFLOW FILES SEARCH, SUGGESTIONS & CLUSTERING HASHES ---

# Per-image knobs that never define pipeline identity; mirrors the ComfyUI
# graph hash's policy of ignoring seeds, steps, CFG, prompts, and ephemeral
# widget values. Keys containing "time" (generation_time, prep_time) are
# excluded by pattern.
_FOREIGN_EPHEMERAL_KEYS = {
    "seed",
    "variationseed",
    "variationseedstrength",
    "steps",
    "cfg",
    "cfgscale",
    "images",
    "batchsize",
    "denoise",
    "clip_skip",
    "size",
    "model_hash",
    "version",
    "swarm_version",
    "date",
    "original_prompt",
}
_FOREIGN_MODEL_KEY_RE = re.compile(r"model|lora|vae|refiner|controlnet", re.IGNORECASE)


def _foreign_cluster_hashes(parsed):
    """Cluster identities for images whose metadata names a non-ComfyUI tool
    (SwarmUI, A1111/Forge, Fooocus, ...). There is no node graph, but SwarmUI
    and friends forward generation to backends like ComfyUI, so the parameter
    set shapes the workflow they generate. Returns (workflow_hash,
    prompt_hash, models_hash):
      workflow_hash -> which parameters are present + model-valued parameters
                       (fine-grained pipeline identity)
      models_hash   -> checkpoint model + LoRA set (coarse identity)
    The prompt hash uses the same normalization as the ComfyUI path so
    identical prompts cluster across tools."""
    if parsed is None:
        return "", "", ""
    prompt_hash = ""
    if parsed.positive:
        prompt_hash = content_digest(parsed.positive.strip().lower())
    loras = {m.group(1).strip().lower() for m in re.finditer(r"<lora:([^:>]+)", parsed.positive or "")}
    for key in ("loras", "used_loras", "lora_hashes", "Lora hashes", "used models"):
        value = parsed.extra.get(key)
        if value:
            loras.add(str(value).strip().lower())

    keys = []
    model_values = set()
    for key, value in {**parsed.extra, **parsed.params}.items():
        k = str(key).strip().lower()
        if k in _FOREIGN_EPHEMERAL_KEYS or "time" in k:
            continue
        keys.append(k)
        if _FOREIGN_MODEL_KEY_RE.search(k):
            model_values.add(str(value).strip().lower())
    workflow_hash = ""
    if keys or loras:
        identity = json.dumps(
            {"keys": sorted(keys), "models": sorted(model_values), "loras": sorted(loras)},
            sort_keys=True,
        )
        workflow_hash = content_digest(identity)

    models_hash = ""
    model = str(parsed.params.get("model") or "").strip().lower()
    if model:
        identity = json.dumps({"model": model, "loras": sorted(loras)}, sort_keys=True)
        models_hash = content_digest(identity)
    return workflow_hash, prompt_hash, models_hash


def compute_workflow_hashes(filepath):
    """
    Computes (workflow_hash, prompt_hash, models_hash):
      workflow_hash -> canonical structural architecture (node graph, or the
                       parameter-set shape for graph-less foreign images)
      prompt_hash   -> normalized positive prompt
      models_hash   -> the set of model/LoRA files used (coarse identity)
    Ignores folder paths, seeds, steps, CFG, prompts, and ephemeral widget values.
    Files without an embedded ComfyUI graph fall back to metaparse identities.
    """
    if not filepath or not os.path.exists(filepath):
        return "", "", ""
    try:
        # Marker-detected non-ComfyUI files skip extract_workflow: its raw
        # byte-scan fallback reads the whole file twice, which is what made
        # bulk hashing of a SwarmUI-dominated tree prohibitive.
        foreign = None
        raw_meta = metaparse.load_raw(filepath)
        if raw_meta is not None and not metaparse.adapters.ComfyUIAdapter.match(raw_meta):
            foreign = metaparse.parse_raw(raw_meta)

        wf_json = None
        if foreign is None:
            wf_json = extract_workflow(filepath, target_type="api")
            if not wf_json:
                wf_json = extract_workflow(filepath, target_type="ui")
        if not wf_json:
            return _foreign_cluster_hashes(foreign)

        data = json.loads(wf_json)
        prompt_text = extract_workflow_prompt_string(wf_json)
        prompt_hash = content_digest(prompt_text.strip().lower()) if prompt_text else ""

        MODEL_EXTENSIONS = (
            ".safetensors",
            ".ckpt",
            ".pt",
            ".pth",
            ".bin",
            ".gguf",
            ".lora",
            ".sft",
            ".vae",
            ".onnx",
            ".engine",
        )
        IGNORED_NODES = {
            "Note",
            "NotePrimitive",
            "Reroute",
            "ShowText",
            "Display Text",
            "SaveImage",
            "PreviewImage",
            "VHS_VideoCombine",
            "PrimitiveNode",
        }

        node_descriptors = []

        # CASE A: UI Format ({'nodes': [...], 'links': [...]})
        if isinstance(data, dict) and "nodes" in data and isinstance(data["nodes"], list):
            node_type_by_id = {}
            for node in data["nodes"]:
                if isinstance(node, dict):
                    nid = str(node.get("id"))
                    ntype = str(node.get("type", "")).strip()
                    node_type_by_id[nid] = ntype

            links_map = {}
            if "links" in data and isinstance(data["links"], list):
                for link in data["links"]:
                    if isinstance(link, list) and len(link) >= 4:
                        link_id = link[0]
                        from_id = str(link[1])
                        from_type = node_type_by_id.get(from_id, "Unknown")
                        links_map[link_id] = from_type

            for node in data["nodes"]:
                if not isinstance(node, dict):
                    continue
                node_type = str(node.get("type", "")).strip()
                if not node_type or node_type in IGNORED_NODES:
                    continue

                connections = []
                inputs = node.get("inputs", [])
                if isinstance(inputs, list):
                    for inp in inputs:
                        if isinstance(inp, dict):
                            link_id = inp.get("link")
                            inp_name = str(inp.get("name", "")).lower()
                            if link_id is not None and link_id in links_map:
                                connections.append((inp_name, links_map[link_id]))

                models = []
                widgets = node.get("widgets_values", [])
                if isinstance(widgets, list):
                    for w in widgets:
                        if isinstance(w, str) and w.strip():
                            w_clean = w.strip().replace(chr(92), "/")
                            base = os.path.basename(w_clean).lower()
                            if any(base.endswith(ext) for ext in MODEL_EXTENSIONS):
                                models.append(base)
                elif isinstance(widgets, dict):
                    for k, v in widgets.items():
                        if isinstance(v, str) and v.strip():
                            v_clean = v.strip().replace(chr(92), "/")
                            base = os.path.basename(v_clean).lower()
                            if any(base.endswith(ext) for ext in MODEL_EXTENSIONS):
                                models.append(base)

                descriptor = {"type": node_type, "connections": sorted(connections), "models": sorted(set(models))}
                node_descriptors.append(descriptor)

        # CASE B: API Format ({ '1': {'class_type': '...', 'inputs': {...}}, ... })
        elif isinstance(data, dict):
            node_type_by_id = {}
            for nid, node in data.items():
                if isinstance(node, dict):
                    ntype = str(node.get("class_type", node.get("type", ""))).strip()
                    node_type_by_id[str(nid)] = ntype

            for nid, node in data.items():
                if not isinstance(node, dict):
                    continue
                node_type = str(node.get("class_type", node.get("type", ""))).strip()
                if not node_type or node_type in IGNORED_NODES:
                    continue

                connections = []
                models = []
                inputs = node.get("inputs", {})

                if isinstance(inputs, dict):
                    for k, v in inputs.items():
                        k_str = str(k).lower()
                        if isinstance(v, list) and len(v) >= 1:
                            from_id = str(v[0])
                            from_type = node_type_by_id.get(from_id, "Unknown")
                            connections.append((k_str, from_type))
                        elif isinstance(v, str) and v.strip():
                            v_clean = v.strip().replace(chr(92), "/")
                            base = os.path.basename(v_clean).lower()
                            if any(base.endswith(ext) for ext in MODEL_EXTENSIONS):
                                models.append(base)

                descriptor = {"type": node_type, "connections": sorted(connections), "models": sorted(set(models))}
                node_descriptors.append(descriptor)

        node_descriptors.sort(key=lambda d: (d["type"], json.dumps(d["connections"]), json.dumps(d["models"])))

        if not node_descriptors:
            return "", prompt_hash, ""

        struct_str = json.dumps(node_descriptors, sort_keys=True)
        workflow_hash = content_digest(struct_str)

        # Coarse identity: just the set of model files the graph loads.
        all_models = sorted({m for d in node_descriptors for m in d["models"]})
        models_hash = ""
        if all_models:
            models_hash = content_digest(json.dumps(all_models))

        # prompt_hash stays empty when the workflow has no positive prompt:
        # prompt clusters mean "identical prompt text", and a synthesized
        # value would group files that share no prompt at all.
        return workflow_hash, prompt_hash, models_hash
    except Exception:
        _logger.debug("handled a failure in compute_workflow_hashes", exc_info=True)
        return "", "", ""


GENPARAMS_SCHEMA = "genparams-v1"


def _genparams_backfill_worker(item):
    """Parse one file into a typed generation_params row (or None).
    Module-level so worker pools can pickle it."""
    file_id, path, ftype, has_workflow = item
    try:
        if not os.path.exists(path):
            return file_id, None
        gp = None
        if has_workflow:
            wf_json = extract_workflow(path, target_type="api")
            if wf_json:
                graph_meta = ComfyMetadataParser(json.loads(wf_json)).parse()
                gp = metaparse_typed.GenerationParams.from_comfy(graph_meta)
        if (gp is None or not gp.has_content) and ftype in ("image", "animated_image"):
            parsed = metaparse.parse_file(path, allow_stealth=False)
            if parsed is not None:
                candidate = metaparse_typed.GenerationParams.from_parsed(parsed)
                if candidate.has_content or parsed.params:
                    gp = candidate
        if gp is not None and gp.has_content:
            return file_id, gp.to_row(file_id, time.time())
        return file_id, None
    except Exception:
        _logger.debug("handled a failure in _genparams_backfill_worker", exc_info=True)
        return file_id, None


def ensure_genparams_backfill_async(conn):
    """One-time typed generation-params backfill for files indexed before
    the generation_params table existed. Marker-gated like the cluster
    hash migration: recorded only on completion, so an interrupted run
    resumes next startup (already-written rows are skipped by query)."""
    if not env_flag("GENPARAMS_BACKFILL", True):
        print("INFO: [GenParams] Backfill disabled by GENPARAMS_BACKFILL=0; new scans still track parameters.")
        return
    stored = conn.execute("SELECT value FROM ai_metadata WHERE key = 'genparams_schema'").fetchone()
    if (stored[0] if stored else None) == GENPARAMS_SCHEMA:
        return

    def run():
        try:
            with get_db_connection() as bconn:
                todo = bconn.execute(
                    "SELECT f.id, f.path, f.type, f.has_workflow FROM files f "
                    "LEFT JOIN generation_params gp ON gp.file_id = f.id "
                    "WHERE gp.file_id IS NULL "
                    "AND (f.has_workflow = 1 OR f.type IN ('image', 'animated_image'))"
                ).fetchall()
                written = 0
                if todo:
                    print(
                        f"{Colors.BLUE}INFO: [GenParams] Backfilling typed generation "
                        f"parameters for {len(todo)} files...{Colors.RESET}",
                        flush=True,
                    )
                    done = 0
                    batch = []
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        for _file_id, row in pool.map(_genparams_backfill_worker, [tuple(r) for r in todo]):
                            done += 1
                            if row is not None:
                                batch.append(row)
                            if len(batch) >= 500:
                                bconn.executemany(_GENPARAMS_UPSERT, batch)
                                bconn.commit()
                                written += len(batch)
                                batch = []
                            if done % 2000 == 0:
                                print(
                                    f"INFO: [GenParams] {done}/{len(todo)} scanned, "
                                    f"{written + len(batch)} with parameters...",
                                    flush=True,
                                )
                    if batch:
                        bconn.executemany(_GENPARAMS_UPSERT, batch)
                        bconn.commit()
                        written += len(batch)
                total = bconn.execute("SELECT COUNT(*) FROM generation_params").fetchone()[0]
                bconn.execute(
                    "INSERT OR REPLACE INTO ai_metadata (key, value, updated_at) VALUES ('genparams_schema', ?, ?)",
                    (GENPARAMS_SCHEMA, time.time()),
                )
                bconn.commit()
                print(
                    f"{Colors.GREEN}SUCCESS: [GenParams] Typed generation parameters "
                    f"tracked for {total} files.{Colors.RESET}",
                    flush=True,
                )
        except Exception as e:
            _logger.debug("handled a failure in run", exc_info=True)
            print(f"ERROR in genparams backfill: {e}")

    threading.Thread(target=run, daemon=True, name="genparams-backfill").start()


def backfill_audio_durations(conn=None):
    """
    Auto-migrates existing audio files in DB to populate their duration.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        rows = conn.execute(
            "SELECT id, path FROM files WHERE type = 'audio' AND (duration IS NULL OR duration = '')"
        ).fetchall()
        if not rows:
            if close_conn:
                conn.close()
            return 0

        uncalculated = [(r["id"], r["path"]) for r in rows if os.path.exists(r["path"])]
        total_uncalc = len(uncalculated)
        if total_uncalc == 0:
            if close_conn:
                conn.close()
            return 0

        print(
            f"{Colors.BLUE}INFO: [Audio] Starting duration calculation for {total_uncalc} audio files...{Colors.RESET}",
            flush=True,
        )

        def _work(item):
            fid, fpath = item
            dur = ""
            current_ffprobe = FFPROBE_EXECUTABLE_PATH or find_ffprobe_path()
            if current_ffprobe:
                try:
                    cmd_info = [
                        current_ffprobe,
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        fpath,
                    ]
                    res = run_media_tool(cmd_info, timeout=3)
                    if res.stdout.strip():
                        total_duration_sec = float(res.stdout.strip())
                        dur = format_duration(total_duration_sec)
                except Exception:
                    _logger.debug("ignored a failure in _work", exc_info=True)
            return (dur, fid) if dur else None

        results = []
        completed = 0
        last_reported_pct = -1

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            futures = [executor.submit(_work, item) for item in uncalculated]
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                except Exception:
                    _logger.debug("ignored a failure in backfill_audio_durations", exc_info=True)

                pct = int((completed / total_uncalc) * 100)
                if pct % 5 == 0 and pct != last_reported_pct:
                    last_reported_pct = pct
                    print(
                        f"\r   [Audio Progress] Calculated {completed}/{total_uncalc} files ({pct}%)...",
                        end="",
                        flush=True,
                    )

        if results:
            batch_size = 500
            for i in range(0, len(results), batch_size):
                batch = results[i : i + batch_size]
                conn.executemany("UPDATE files SET duration = ? WHERE id = ?", batch)
                conn.commit()

        print()
        print(
            f"{Colors.GREEN}SUCCESS: [Audio] Successfully calculated durations"
            f" for {len(results)}/{total_uncalc} files!{Colors.RESET}",
            flush=True,
        )
        return len(results)
    except Exception as e:
        _logger.debug("handled a failure in backfill_audio_durations", exc_info=True)
        print(f"ERROR in backfill_audio_durations: {e}")
        return 0
    finally:
        if close_conn and conn:
            conn.close()


def clear_synthetic_prompt_hashes(conn):
    """One-time data repair: prompt_hash used to be synthesized as
    md5(workflow_hash + '_prompt') for promptless workflows, which made
    'identical prompt text' clusters group files sharing no prompt. Detect
    exactly those synthetic values and clear them. Idempotent."""
    rows = conn.execute(
        "SELECT id, workflow_hash, prompt_hash FROM files WHERE prompt_hash != '' AND workflow_hash != ''"
    ).fetchall()
    synthetic = [(r["id"],) for r in rows if r["prompt_hash"] == content_digest(r["workflow_hash"] + "_prompt")]
    if synthetic:
        conn.executemany("UPDATE files SET prompt_hash = '' WHERE id = ?", synthetic)
        conn.commit()
    return len(synthetic)


def _backfill_hash_worker(item):
    """Hash one file for backfill_unhashed_workflows. Module-level so
    ProcessPoolExecutor can pickle it (Windows spawn)."""
    fid, fpath, had_prompt = item
    wf_h, pr_h, md_h = compute_workflow_hashes(fpath)
    # Rows indexed before foreign-metadata support have no searchable
    # prompt; the file is already open here, so fill it in one pass.
    prompt_text = ""
    if not had_prompt:
        parsed = metaparse.parse_file(fpath)
        if parsed and parsed.positive:
            prompt_text = parsed.positive
    return (wf_h, pr_h, md_h, prompt_text, fid)


# Runs at or above this size hash in worker PROCESSES. The hashing is mostly
# pure-Python parsing, so a large thread pool of it inside the web process
# monopolizes the GIL and starves the request threads (observed live: the
# site froze during the 42k-file migration despite the "background" thread).
# Small incremental runs stay on threads: no spawn cost, and in-process
# monkeypatching keeps working for tests.
_BACKFILL_PROCESS_THRESHOLD = 64


def backfill_unhashed_workflows(conn=None, force_all=False):
    """
    Auto-migrates existing files in DB with real-time console progress.
    Fast parallel execution, safe, non-destructive. Sets the state 'aborted'
    flag on any incomplete run so completion hooks don't fire.
    """
    _CLUSTER_BACKFILL_STATE["aborted"] = False
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        # Hashable population: ComfyUI-graph carriers plus any image whose
        # embedded metadata metaparse can identify (SwarmUI, A1111, ...).
        hashable = "(has_workflow = 1 OR type IN ('image', 'animated_image'))"
        if force_all:
            # Algorithm change: every file gets a fresh attempt, including
            # ones previously marked failed.
            conn.execute(f"UPDATE files SET hash_failed = 0 WHERE {hashable} AND hash_failed = 1")
            conn.commit()
            rows = conn.execute(f"SELECT id, path, workflow_prompt FROM files WHERE {hashable}").fetchall()
        else:
            # Both hashes empty: a file that yielded only a prompt hash (or
            # only an architecture hash) is done, not pending — selecting on
            # one column alone would re-scan it forever.
            rows = conn.execute(
                f"""SELECT id, path, workflow_prompt FROM files WHERE {hashable}
                    AND (workflow_hash IS NULL OR workflow_hash = '')
                    AND (prompt_hash IS NULL OR prompt_hash = '')
                    AND hash_failed = 0"""
            ).fetchall()

        if not rows:
            if close_conn:
                conn.close()
            return 0

        unhashed = [
            (r["id"], r["path"], bool(str(r["workflow_prompt"] or "").strip()))
            for r in rows
            if os.path.exists(r["path"])
        ]

        # Rows whose file vanished can never hash; mark them failed so they
        # stop re-triggering this backfill on every clustered page view.
        missing_ids = [(r["id"],) for r in rows if not os.path.exists(r["path"])]
        if missing_ids:
            conn.executemany("UPDATE files SET hash_failed = 1 WHERE id = ?", missing_ids)
            conn.commit()

        total_unhashed = len(unhashed)
        if total_unhashed == 0:
            if close_conn:
                conn.close()
            return 0

        _CLUSTER_BACKFILL_STATE["total"] = total_unhashed
        _CLUSTER_BACKFILL_STATE["done"] = 0
        print(
            f"{Colors.BLUE}INFO: [Clustering] Starting hash indexing for {total_unhashed} files...{Colors.RESET}",
            flush=True,
        )

        results = []
        failed_ids = []
        completed = 0
        last_reported_pct = -1

        executor_cls = (
            concurrent.futures.ProcessPoolExecutor
            if total_unhashed >= _BACKFILL_PROCESS_THRESHOLD
            else concurrent.futures.ThreadPoolExecutor
        )
        try:
            with executor_cls(max_workers=MAX_PARALLEL_WORKERS) as executor:
                futures = {executor.submit(_backfill_hash_worker, item): item[0] for item in unhashed}
                for future in concurrent.futures.as_completed(futures):
                    completed += 1
                    try:
                        res = future.result()
                        if res and (res[0] or res[1] or res[2]):
                            results.append(res)
                        else:
                            failed_ids.append((futures[future],))
                    except concurrent.futures.process.BrokenProcessPool:
                        raise
                    except Exception:
                        _logger.debug("handled a failure in backfill_unhashed_workflows", exc_info=True)
                        failed_ids.append((futures[future],))

                    _CLUSTER_BACKFILL_STATE["done"] = completed
                    pct = int((completed / total_unhashed) * 100)
                    if pct % 5 == 0 and pct != last_reported_pct:
                        last_reported_pct = pct
                        print(
                            f"\r   [Clustering Progress] Indexed {completed}/{total_unhashed} files ({pct}%)...",
                            end="",
                            flush=True,
                        )
        except concurrent.futures.process.BrokenProcessPool:
            # A crashed worker pool must not mass-mark the uncollected rows as
            # failed: write what was collected, leave the rest pending so the
            # next trigger resumes them.
            _CLUSTER_BACKFILL_STATE["aborted"] = True
            print(
                f"\n{Colors.YELLOW}WARN: [Clustering] Worker pool died mid-run; "
                f"{len(results)} results kept, the rest stay pending.{Colors.RESET}",
                flush=True,
            )

        batch_size = 500
        if results:
            hash_rows = [(wf, pr, md, fid) for wf, pr, md, _, fid in results]
            prompt_rows = [(text, fid) for _, _, _, text, fid in results if text]
            for i in range(0, len(hash_rows), batch_size):
                conn.executemany(
                    "UPDATE files SET workflow_hash = ?, prompt_hash = ?,"
                    " models_hash = ?, hash_failed = 0 WHERE id = ?",
                    hash_rows[i : i + batch_size],
                )
                conn.commit()
            for i in range(0, len(prompt_rows), batch_size):
                conn.executemany(
                    "UPDATE files SET workflow_prompt = ? WHERE id = ?"
                    " AND (workflow_prompt IS NULL OR workflow_prompt = '')",
                    prompt_rows[i : i + batch_size],
                )
                conn.commit()

        # Unhashable files (corrupt/promptless-and-graphless workflows) are
        # marked failed; a rescan that changes the file's mtime resets the
        # flag and earns them another attempt.
        if failed_ids:
            for i in range(0, len(failed_ids), batch_size):
                conn.executemany("UPDATE files SET hash_failed = 1 WHERE id = ?", failed_ids[i : i + batch_size])
                conn.commit()

        print()
        print(
            f"{Colors.GREEN}SUCCESS: [Clustering] Successfully indexed"
            f" {len(results)}/{total_unhashed} files!{Colors.RESET}",
            flush=True,
        )
        return len(results)
    except Exception as e:
        _logger.debug("handled a failure in backfill_unhashed_workflows", exc_info=True)
        _CLUSTER_BACKFILL_STATE["aborted"] = True
        print(f"ERROR in backfill_unhashed_workflows: {e}")
        return 0
    finally:
        if close_conn and conn:
            conn.close()


# Live progress of the cluster-hash backfill, readable from any thread.
# 'running' is owned by ensure_cluster_backfill_async(); done/total are
# written by backfill_unhashed_workflows() itself, so synchronous callers
# (tests, scripts) report progress the same way.
_CLUSTER_BACKFILL_STATE = {"running": False, "done": 0, "total": 0}
_CLUSTER_BACKFILL_LOCK = threading.Lock()


def ensure_cluster_backfill_async(force_all=False, on_complete=None):
    """Run the cluster-hash backfill on a daemon thread unless one is already
    in flight; returns True when a new run was started. The backfill is
    idempotent and convergent (pending = both hashes empty and not failed),
    so a request that arrives while a run is active can simply be dropped —
    the active run picks those rows up, and the next trigger catches any
    stragglers.

    `on_complete` fires only after a run that finished WITHOUT aborting
    (no worker-pool death, no top-level error) — callers use it to record
    "this migration actually completed" markers. A marker written before
    completion turns an interrupted migration into a silently skipped one
    (observed live: models_hash empty across the gallery with the schema
    marker claiming done)."""
    with _CLUSTER_BACKFILL_LOCK:
        if _CLUSTER_BACKFILL_STATE["running"]:
            return False
        _CLUSTER_BACKFILL_STATE["running"] = True
        _CLUSTER_BACKFILL_STATE["done"] = 0
        _CLUSTER_BACKFILL_STATE["total"] = 0

    def _run():
        try:
            backfill_unhashed_workflows(force_all=force_all)
            if on_complete is not None and not _CLUSTER_BACKFILL_STATE.get("aborted"):
                on_complete()
        finally:
            with _CLUSTER_BACKFILL_LOCK:
                _CLUSTER_BACKFILL_STATE["running"] = False

    threading.Thread(target=_run, name="ClusterHashBackfill", daemon=True).start()
    return True


# Version of the cluster-hash column scheme. Bump when a hash's meaning or
# coverage changes (not just the ComfyUI graph algorithm, which the sampling
# check below detects on its own) so existing rows are recomputed once.
# 3 = three-hash scheme: workflow_hash (graph / foreign pipeline shape),
#     prompt_hash, models_hash. ('2' was the same scheme, but the old code
#     recorded the marker BEFORE the migration ran, so a '2' marker can sit
#     on a DB whose models_hash was never filled — '3' re-runs those once.)
CLUSTER_HASH_SCHEMA = "3"


def _write_cluster_schema_marker():
    """Record that the CLUSTER_HASH_SCHEMA migration ran to completion.
    Only ever called from the backfill's on_complete hook."""
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ai_metadata (key, value, updated_at) VALUES ('cluster_hash_schema', ?, ?)",
                (CLUSTER_HASH_SCHEMA, time.time()),
            )
            conn.commit()
        print(
            f"{Colors.GREEN}SUCCESS: [Clustering] Hash scheme {CLUSTER_HASH_SCHEMA} migration complete.{Colors.RESET}",
            flush=True,
        )
    except Exception as e:
        _logger.debug("handled a failure in _write_cluster_schema_marker", exc_info=True)
        print(f"ERROR recording cluster schema marker: {e}")


def check_and_update_workflow_hashes(conn):
    """
    Tests a few existing hashes against the current algorithm.
    If the algorithm was updated, it forces a complete recalculation of all hashes.
    A CLUSTER_HASH_SCHEMA bump forces the same full recalculation; the marker
    is recorded only when that recalculation completes, so an interrupted
    migration re-runs on the next startup instead of being silently skipped.
    """
    stored = conn.execute("SELECT value FROM ai_metadata WHERE key = 'cluster_hash_schema'").fetchone()
    if (stored[0] if stored else None) != CLUSTER_HASH_SCHEMA:
        conn.execute("DELETE FROM ai_metadata WHERE key = 'foreign_arch_identity_mode'")
        conn.commit()
        print(
            f"{Colors.YELLOW}INFO: Cluster hash scheme updated. Re-indexing all"
            f" cluster identities in the background...{Colors.RESET}"
        )
        ensure_cluster_backfill_async(force_all=True, on_complete=_write_cluster_schema_marker)
        return

    sample = conn.execute(
        "SELECT id, path, workflow_hash FROM files WHERE has_workflow = 1"
        " AND workflow_hash IS NOT NULL AND workflow_hash != '' LIMIT 3"
    ).fetchall()

    needs_update = False
    for row in sample:
        if os.path.exists(row["path"]):
            new_wf_hash, _, _ = compute_workflow_hashes(row["path"])
            # If the newly computed hash differs from the one in the DB, the algorithm changed!
            if new_wf_hash and new_wf_hash != row["workflow_hash"]:
                needs_update = True
                break

    if needs_update:
        print(
            f"{Colors.YELLOW}INFO: Workflow clustering algorithm updated."
            f" Re-indexing architectures in the background...{Colors.RESET}"
        )
        ensure_cluster_backfill_async(force_all=True)
    else:
        # Standard check for newly added files that missed the hash
        ensure_cluster_backfill_async(force_all=False)


@app.route("/galleryout/api/workflow_files_suggestions", methods=["GET"])
def api_workflow_files_suggestions():
    """
    Returns unique model/LoRA filenames from DB and calculates 'Did you mean?'
    fuzzy suggestions using Python's native difflib without any external AI model.
    """
    if (IS_EXHIBITION_MODE or FORCE_LOGIN) and not session.get("user_id"):
        return jsonify({"status": "error", "message": "Authentication required"}), 401

    if should_strip_metadata():
        return jsonify(
            {"status": "error", "message": "Security Policy: Access to workflow metadata is restricted for your role."}
        ), 403

    query = request.args.get("q", "").strip()
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT workflow_files FROM files WHERE workflow_files IS NOT NULL AND workflow_files != ''"
        ).fetchall()

    all_files = set()
    for r in rows:
        wf_str = r["workflow_files"]
        if wf_str:
            for item in wf_str.split(" ||| "):
                item_clean = item.strip()
                if item_clean:
                    all_files.add(item_clean)

    all_files_list = sorted(all_files)
    if not query:
        return jsonify({"status": "success", "suggestions": all_files_list[:20], "did_you_mean": None})

    norm_query = _normalize_fuzzy_string(query)
    query_tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", query) if len(t) > 1]

    matching_suggestions = []
    for f in all_files_list:
        norm_f = _normalize_fuzzy_string(f)
        if (norm_query and norm_query in norm_f) or (
            query_tokens and all(_normalize_fuzzy_string(tok) in norm_f for tok in query_tokens)
        ):
            matching_suggestions.append(f)

    did_you_mean = None
    if query and query not in all_files_list:
        file_map = {os.path.basename(f): f for f in all_files_list}
        close_matches = difflib.get_close_matches(query, list(file_map.keys()), n=1, cutoff=0.3)
        if close_matches:
            did_you_mean = file_map[close_matches[0]]
        else:
            close_full = difflib.get_close_matches(query, all_files_list, n=1, cutoff=0.25)
            if close_full:
                did_you_mean = close_full[0]

    return jsonify({"status": "success", "suggestions": matching_suggestions[:15], "did_you_mean": did_you_mean})


@app.route("/galleryout/api/site_settings", methods=["GET"])
def api_site_settings_get():
    if (IS_EXHIBITION_MODE or FORCE_LOGIN) and not session.get("user_id"):
        return jsonify({"status": "error", "message": "Authentication required"}), 401
    return jsonify(
        {
            "status": "success",
            "thumbnail_generation": thumbnail_generation_enabled(),
        }
    )


@app.route("/galleryout/api/site_settings", methods=["POST"])
@management_api_only
def api_site_settings_set():
    data = request.get_json(silent=True) or {}
    if "thumbnail_generation" not in data:
        return jsonify({"status": "error", "message": "No known setting in payload"}), 400
    enabled = bool(data["thumbnail_generation"])
    with get_db_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO ai_metadata (key, value, updated_at) VALUES ('thumbnail_generation', ?, ?)",
            ("1" if enabled else "0", time.time()),
        )
        conn.commit()
    _THUMBNAIL_SETTING_CACHE["value"] = None  # re-read on next check
    return jsonify({"status": "success", "thumbnail_generation": enabled})


@app.route("/galleryout/api/cluster_hash_status")
def api_cluster_hash_status():
    """Progress of the background cluster-hash backfill, for the banner."""
    if (IS_EXHIBITION_MODE or FORCE_LOGIN) and not session.get("user_id"):
        return jsonify({"status": "error", "message": "Authentication required"}), 401
    with get_db_connection() as conn:
        pending = conn.execute(
            """SELECT COUNT(*) FROM files
               WHERE (has_workflow = 1 OR type IN ('image', 'animated_image'))
               AND (workflow_hash IS NULL OR workflow_hash = '')
               AND (prompt_hash IS NULL OR prompt_hash = '')
               AND hash_failed = 0"""
        ).fetchone()[0]
    return jsonify(
        {
            "status": "success",
            "running": _CLUSTER_BACKFILL_STATE["running"],
            "done": _CLUSTER_BACKFILL_STATE["done"],
            "total": _CLUSTER_BACKFILL_STATE["total"],
            "pending": pending,
        }
    )


# --- CLUSTER INFO INSPECTOR API ROUTE ---
@app.route("/galleryout/api/cluster_info/<string:hash_type>/<string:hash_val>")
def api_cluster_info(hash_type, hash_val):
    if (IS_EXHIBITION_MODE or FORCE_LOGIN) and not session.get("user_id"):
        return jsonify({"status": "error", "message": "Authentication required"}), 401

    if should_strip_metadata():
        return jsonify(
            {"status": "error", "message": "Security Policy: Access to cluster metadata is restricted for your role."}
        ), 403

    if hash_type not in ("workflow", "prompt", "models"):
        return jsonify({"status": "error", "message": "Invalid hash type"}), 400

    col_name = {"workflow": "workflow_hash", "prompt": "prompt_hash", "models": "models_hash"}[hash_type]
    requested_file_id = request.args.get("file_id")

    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT id, name, path, type, mtime, dimensions, workflow_files,"
                f" workflow_prompt FROM files WHERE {col_name} = ? ORDER BY mtime DESC",
                (hash_val,),
            ).fetchall()

            if not rows:
                return jsonify({"status": "error", "message": "No matching cluster assets found"}), 404

            matching_files = [dict(r) for r in rows]

            if IS_EXHIBITION_MODE:
                matching_files = [f for f in matching_files if is_file_accessible(f["id"])]
                if not matching_files:
                    return jsonify({"status": "error", "message": "Access Denied"}), 403

            total_count = len(matching_files)

            sample = matching_files[0]
            if requested_file_id:
                matched_sample = next((f for f in matching_files if f["id"] == requested_file_id), None)
                if matched_sample and is_file_accessible(matched_sample["id"]):
                    sample = matched_sample

            nodes_pipeline = []
            models_used = []
            sample_prompt = sample.get("workflow_prompt", "")

            wf_json = extract_workflow(sample["path"], target_type="ui")
            if not wf_json:
                wf_json = extract_workflow(sample["path"], target_type="api")

            if wf_json:
                try:
                    summary = generate_node_summary(wf_json)
                    if summary:
                        for n in summary:
                            nodes_pipeline.append(
                                {
                                    "id": n.get("id"),
                                    "type": n.get("type"),
                                    "category": n.get("category"),
                                    "color": n.get("color"),
                                }
                            )
                except Exception:
                    _logger.debug("ignored a failure in api_cluster_info", exc_info=True)

            if sample.get("workflow_files"):
                for item in sample["workflow_files"].split(" ||| "):
                    if item.strip():
                        models_used.append(os.path.basename(item.strip()))

            distinct_other_hashes = 0
            if hash_type == "prompt":
                distinct_other_hashes = conn.execute(
                    "SELECT COUNT(DISTINCT workflow_hash) FROM files WHERE prompt_hash = ? AND workflow_hash != ''",
                    (hash_val,),
                ).fetchone()[0]
            elif hash_type == "models":
                distinct_other_hashes = conn.execute(
                    "SELECT COUNT(DISTINCT workflow_hash) FROM files WHERE models_hash = ? AND workflow_hash != ''",
                    (hash_val,),
                ).fetchone()[0]
            else:
                distinct_other_hashes = conn.execute(
                    "SELECT COUNT(DISTINCT prompt_hash) FROM files WHERE workflow_hash = ? AND prompt_hash != ''",
                    (hash_val,),
                ).fetchone()[0]

            return jsonify(
                {
                    "status": "success",
                    "hash_type": hash_type,
                    "hash_val": hash_val,
                    "total_count": total_count,
                    "sample_file": sample,
                    "nodes_pipeline": nodes_pipeline,
                    "models_used": sorted(set(models_used)),
                    "sample_prompt": sample_prompt,
                    "distinct_counterparts": distinct_other_hashes,
                }
            )
    except Exception as e:
        _logger.debug("handled a failure in api_cluster_info", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    # --- OS SIGNAL HANDLER FOR TMUX/LINUX/MAC PORT RELEASE ---
    import socket

    def force_hard_kill(signum, frame):
        """
        Aggressive shutdown sequence to prevent 'Port already in use' errors
        in persistent terminal multiplexers like tmux/screen on Linux/macOS.
        """
        print(
            f"\n{Colors.YELLOW}INFO: Shutdown signal ({signum}) received. Releasing port {SERVER_PORT}...{Colors.RESET}"
        )

        # 1. Kill the entire Process Group (terminates zombie Waitress threads and ProcessPools)
        try:
            if os.name != "nt":
                import signal

                os.killpg(os.getpgrp(), signal.SIGKILL)
        except Exception:
            # Best effort on the way out. If the group kill is unavailable or
            # refused, os._exit below still ends this process.
            _logger.debug("ignored a failure in force_hard_kill", exc_info=True)

        # 2. Absolute final fallback
        os._exit(0)

    _installed, _left_alone = install_shutdown_signals(force_hard_kill)
    if _left_alone:
        print(
            f"INFO: Leaving {', '.join(_left_alone)} as the launcher set "
            f"it; the gallery keeps running when the terminal closes."
        )
    # ---------------------------------------------------------

    run_integrity_check()
    # --- CHECK: PORT AVAILABILITY ---
    print(f"INFO: Checking port {SERVER_PORT} availability...")
    if not check_port_available(SERVER_PORT):
        print(f"\n{Colors.RED}{Colors.BOLD}❌ CRITICAL ERROR: PORT ALREADY IN USE{Colors.RESET}")
        print(f"{Colors.RED}The port {SERVER_PORT} is currently being used by another application.{Colors.RESET}")
        print(f"\n{Colors.CYAN}{Colors.BOLD}💡 HOW TO FIX IT:{Colors.RESET}")
        print("  1. Ensure you don't have another instance of SmartGallery already running.")
        print("  2. If using Docker, check if another container is bound to this port.")
        print("  3. You can start SmartGallery on a different port using the --port argument:")
        print(f"     {Colors.YELLOW}python smartgallery.py --port 8190{Colors.RESET}\n")

        # Cross-platform wait
        try:
            print(f"{Colors.DIM}Press Enter to exit...{Colors.RESET}")
            input()
        except (EOFError, KeyboardInterrupt):
            pass

        sys.exit(1)

    print_startup_banner()
    # --- CRITICAL SECURITY CHECK ---
    # Stops the server immediately if login is forced but no admin credentials are provided.
    if ADMIN_CONFIG_MISSING:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ CRITICAL SECURITY ERROR: Missing Admin Password{Colors.RESET}")
        if IS_EXHIBITION_MODE:
            print(
                f"{Colors.RED}You started the server with '--exhibition',"
                f" which requires an admin account.{Colors.RESET}"
            )
        else:
            print(
                f"{Colors.RED}You started the server with '--force-login',"
                f" which requires an admin account.{Colors.RESET}"
            )

        print(f"\n{Colors.CYAN}{Colors.BOLD}💡 HOW TO FIX IT:{Colors.RESET}")
        print("Please restart the application and provide the password using one of these methods:")
        print(
            f"  1. CLI Argument: {Colors.YELLOW}python smartgallery.py "
            f"{'--exhibition' if IS_EXHIBITION_MODE else '--force-login'}"
            f" --admin-pass YOUR_PASSWORD{Colors.RESET}"
        )
        print(
            f"  2. Environment Variable: Set {Colors.YELLOW}ADMIN_PASSWORD=YOUR_PASSWORD{Colors.RESET} before running."
        )
        print("\nThe server cannot start in this state and will now exit.\n")

        # Cross-platform safe wait (Docker friendly)
        try:
            print(f"{Colors.DIM}Press Enter to exit...{Colors.RESET}")
            input()
        except (EOFError, KeyboardInterrupt):
            pass

        sys.exit(1)

    # --- CRITICAL SECURITY CHECK: PASSWORD LENGTH ---
    # Stops the server if the user provided a password but it is too weak (under 8 chars).
    if ADMIN_PASS_TOO_SHORT:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ CRITICAL SECURITY ERROR: Weak Admin Password{Colors.RESET}")
        print(
            f"{Colors.RED}The provided admin password is too short."
            f" It must be at least 8 characters long.{Colors.RESET}"
        )

        print(f"\n{Colors.CYAN}{Colors.BOLD}💡 HOW TO FIX IT:{Colors.RESET}")
        print(
            "Please restart the application and provide a stronger password (8+ characters) using one of these methods:"
        )
        print(
            f"  1. CLI Argument: {Colors.YELLOW}python smartgallery.py "
            f"{'--exhibition' if IS_EXHIBITION_MODE else '--force-login'}"
            f" --admin-pass YOUR_STRONG_PASSWORD{Colors.RESET}"
        )
        print(
            f"  2. Environment Variable: Set"
            f" {Colors.YELLOW}ADMIN_PASSWORD=YOUR_STRONG_PASSWORD{Colors.RESET}"
            f" before running."
        )
        print("\nThe server cannot start in this state and will now exit.\n")

        # Cross-platform safe wait (Docker friendly)
        try:
            print(f"{Colors.DIM}Press Enter to exit...{Colors.RESET}")
            input()
        except (EOFError, KeyboardInterrupt):
            pass

        sys.exit(1)

    # --- MODE ANNOUNCEMENTS ---
    if IS_EXHIBITION_MODE:
        print(f"{Colors.YELLOW}{Colors.BOLD}*** EXHIBITION MODE ACTIVE ***{Colors.RESET}")
        print("Restricted view enabled. Granular messaging (Public/Private/Direct) is active.")
    elif FORCE_LOGIN:
        print(f"{Colors.YELLOW}{Colors.BOLD}*** SECURE TEAM MODE ACTIVE (--force-login) ***{Colors.RESET}")
        print("Index view is protected. Users must log in to view or manage files.")

    check_for_updates()
    print_configuration()
    # After the table, so a near-miss is read next to the value it failed to
    # set -- "Base Output Path: C:/ComfyUI/output" above the warning explains
    # itself in a way the warning alone does not.
    warn_about_misspelt_env_vars()

    # --- CHECK: CRITICAL OUTPUT PATH CHECK (Blocking) ---
    # isdir, not exists: a file passes exists and then nothing works.
    if not os.path.isdir(BASE_OUTPUT_PATH):
        show_config_error_and_exit(BASE_OUTPUT_PATH)

    # --- CHECK: INPUT PATH CHECK (Non-Blocking / Warning) ---
    if not os.path.isdir(BASE_INPUT_PATH):
        print(f"{Colors.YELLOW}{Colors.BOLD}WARNING: Input Path not found!{Colors.RESET}")
        print(f"{Colors.YELLOW}   The path '{BASE_INPUT_PATH}' does not exist.{Colors.RESET}")
        print(f"{Colors.YELLOW}   > Source media visualization in Node Summary will be DISABLED.{Colors.RESET}")
        print(f"{Colors.YELLOW}   > The gallery will still function normally for output files.{Colors.RESET}\n")

    # Initialize the gallery (Creates DB, Migrations, etc.)
    initialize_gallery()

    # --- CHECK: FFMPEG WARNING ---
    if not FFPROBE_EXECUTABLE_PATH:
        if os.environ.get("DISPLAY") or os.name == "nt":
            try:
                show_ffmpeg_warning()
            except Exception:
                # A dialog toolkit that is absent, misconfigured, or has no
                # display to draw on. The console still gets told.
                _logger.debug("handled a failure in module scope", exc_info=True)
                print(f"{Colors.RED}WARNING: FFmpeg not found.{Colors.RESET}")
        else:
            print(f"{Colors.RED}WARNING: FFmpeg not found.{Colors.RESET}")

    # --- START BACKGROUND WATCHER ---
    # In exhibition mode, watcher might not be needed, but it's safe to run it (it reads DB config)
    if ENABLE_AI_SEARCH and not IS_EXHIBITION_MODE:
        try:
            watcher = threading.Thread(target=background_watcher_task, daemon=True)
            watcher.start()
            print(f"{Colors.BLUE}INFO: AI Background Watcher started.{Colors.RESET}")
        except Exception as e:
            _logger.debug("handled a failure in module scope", exc_info=True)
            print(f"{Colors.RED}ERROR: Failed to start AI Watcher: {e}{Colors.RESET}")

    # --- START AI DAM BACKGROUND WORKER (Optional, WI-31) ---
    # Disabled by default (ENABLE_AI_DAM unset/false): this block is skipped
    # entirely, so normal startup/browsing is completely unchanged.
    if AI_CONFIG.enabled and not IS_EXHIBITION_MODE:
        try:
            ai_dam_worker = AIWorker(
                AI_CONFIG,
                DATABASE_FILE,
                poll_interval=env_num("AI_DAM_WORKER_POLL", 25.0, float),
                batch_size=env_num("AI_DAM_WORKER_BATCH", 150),
            )
            ai_dam_worker.start()
            ai_dam_service.set_worker(ai_dam_worker)
            print(f"{Colors.BLUE}INFO: AI DAM background worker started.{Colors.RESET}")
        except Exception as e:
            _logger.debug("handled a failure in module scope", exc_info=True)
            print(f"{Colors.RED}ERROR: Failed to start AI DAM worker: {e}{Colors.RESET}")

    print(f"{Colors.GREEN}{Colors.BOLD}🚀 Gallery started successfully!{Colors.RESET}")
    url_host = "localhost" if SERVER_PORT == 80 else "127.0.0.1"
    print(f"👉 Local Access:   {Colors.CYAN}{Colors.BOLD}http://{url_host}:{SERVER_PORT}/galleryout/{Colors.RESET}")

    # Safely attempt to discover the local network IP
    try:
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # We don't actually send any data. We just use UDP to ask the OS
        # which network interface it would use to reach an external IP.
        # This works on Windows, Linux, macOS, and inside Docker containers.
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()

        # Only print if it's a valid LAN IP (ignoring local loopback)
        if local_ip and local_ip != "127.0.0.1":
            print(
                f"👉 Network Access: {Colors.CYAN}{Colors.BOLD}http://{local_ip}:{SERVER_PORT}/galleryout/{Colors.RESET}"
            )
    except Exception:
        # If the machine is completely offline or strict Docker network rules apply,
        # fail silently to prevent application crashes.
        _logger.debug("ignored a failure in module scope", exc_info=True)

    print("   (Press CTRL+C to stop)")

    # Socket reuse is the server's own doing and needs nothing here.
    # waitress calls set_reuse_addr() on its listening socket before
    # binding it; werkzeug's server sets allow_reuse_address. There was a
    # block here that made a socket of its own, set SO_REUSEADDR and
    # SO_REUSEPORT on it, and closed it again without ever binding it --
    # options belong to a socket, not to the kernel, so it changed
    # nothing about the one the server went on to open. What it was
    # reaching for was real, and it belongs in check_port_available,
    # which is the thing that was refusing to start.

    if WAITRESS_AVAILABLE:
        # PRODUCTION MODE: Launching with Waitress WSGI Server
        # threads=8 allows handling multiple concurrent requests (images/video thumbnails)
        # channel_timeout avoids drops during heavy video streaming
        print(f"{Colors.GREEN}INFO: Starting Production WSGI Server (Waitress)...{Colors.RESET}")
        serve(
            app,
            host="0.0.0.0",
            port=SERVER_PORT,
            threads=8,
            connection_limit=150,
            channel_timeout=120,
            asyncore_use_poll=True,
            max_request_body_size=MAX_REQUEST_BODY_BYTES,
            _quiet=True,
        )
    else:
        # DEVELOPMENT MODE: Falling back to Flask built-in server
        print(f"{Colors.YELLOW}WARNING: 'waitress' not found. Using Flask development server.{Colors.RESET}")
        print("INFO: For better performance, install it with: pip install waitress")
        app.run(host="0.0.0.0", port=SERVER_PORT, debug=False)
