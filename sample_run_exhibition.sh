#!/usr/bin/env bash
# ======================================================================
# SMARTGALLERY DAM -- CONFIGURATION -- EXHIBITION MODE
# Instructions: copy this to "run_exhibition.sh", make it executable
# (chmod +x run_exhibition.sh) and run it with ./run_exhibition.sh
# Exhibition is the read-only portal you share with family, friends or
# clients. It runs alongside the main gallery on its own port, so keep
# run_smartgallery.sh as it is.
# ======================================================================
set -euo pipefail
cd "$(dirname "$0")"

# --- CORE PATHS ---
# Write these however you like. A trailing slash, a ~, quotes around the
# whole thing -- the gallery normalises them.
export BASE_OUTPUT_PATH="$HOME/ComfyUI/output"
export BASE_INPUT_PATH="$HOME/ComfyUI/input"

# --- CACHE & DATABASE PATH ---
# Directory for the SQLite database and thumbnail cache.
# You can store this anywhere; it does NOT need to be inside the ComfyUI
# output folder.
export BASE_SMARTGALLERY_PATH="$HOME/ComfyUI/output"

# --- FFMPEG CONFIGURATION (Highly Recommended) ---
# Leave unset to let the gallery find ffprobe on PATH.
# export FFPROBE_MANUAL_PATH="/usr/bin/ffprobe"

# --- NETWORK ---
# Exhibition runs on its own port so it can sit beside the main gallery.
export SERVER_PORT="8190"

# --- ADMIN PASSWORD (REQUIRED) ---
# Exhibition always needs an admin account, so choose your own password
# here. Minimum 8 characters. Until you set one the gallery refuses to
# start and says so -- it will not fall back to a default, because a
# password shipped in this file would be the same one for everybody who
# downloaded it.
# Set it here rather than on the launch line below: command lines are
# visible to other programs on the machine, which is why the gallery
# masks the password out of its own startup log.
export ADMIN_PASSWORD=

# ======================================================================
# OPTIONAL LAUNCH PARAMETERS
# ======================================================================
# Add any of the following to the launch line at the bottom:
#
#   --enable-guest-login        Show a "Login as Guest" button, so people
#                                 can browse without an account
#   --blind-rating              Hide global averages to prevent user bias

# ======================================================================
# STARTUP -- nothing below here needs editing
# ======================================================================

# --- WHERE IS THE APP? ---
if [ -f "smartgallery.py" ]; then
    APP_DIR="."
elif [ -f "app/smartgallery.py" ]; then
    APP_DIR="app"
else
    echo "ERROR: could not find smartgallery.py." >&2
    echo "Looked in '$PWD' and '$PWD/app'." >&2
    echo "Put this file in the same folder as smartgallery.py." >&2
    exit 1
fi

# --- WHICH PYTHON? ---
# A virtual environment named .venv or venv is used when there is one.
PYTHON=""
for candidate in \
    ".venv/bin/python" \
    "venv/bin/python" \
    "$APP_DIR/.venv/bin/python" \
    "$APP_DIR/venv/bin/python"
do
    if [ -x "$candidate" ]; then
        PYTHON="$(cd "$(dirname "$candidate")" && pwd)/$(basename "$candidate")"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: no Python environment found." >&2
    echo "Looked for .venv/bin/python and venv/bin/python under '$PWD'." >&2
    echo >&2
    echo "Create one, either way:" >&2
    echo >&2
    echo "  With uv - installs the exact versions pinned in uv.lock." >&2
    echo "  If you do not have uv yet:" >&2
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "  then, in this folder:" >&2
    echo "    uv sync" >&2
    echo >&2
    echo "  Or with the Python you already have:" >&2
    echo "    python3 -m venv .venv" >&2
    echo "    .venv/bin/python -m pip install -r requirements.txt" >&2
    exit 1
fi

echo "Using Python: $PYTHON"

# Open a browser after a moment, if this desktop has a way to.
if command -v xdg-open >/dev/null 2>&1; then
    ( sleep 3; xdg-open "http://127.0.0.1:${SERVER_PORT}/galleryout/" >/dev/null 2>&1 || true ) &
elif command -v open >/dev/null 2>&1; then
    ( sleep 3; open "http://127.0.0.1:${SERVER_PORT}/galleryout/" >/dev/null 2>&1 || true ) &
fi

# Move to the app folder so Flask finds the templates correctly
cd "$APP_DIR"
exec "$PYTHON" smartgallery.py --port "$SERVER_PORT" --exhibition
