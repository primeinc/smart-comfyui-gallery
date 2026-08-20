#!/usr/bin/env bash
# ======================================================================
# SMARTGALLERY DAM -- CONFIGURATION
# Instructions: copy this to "run_smartgallery.sh", make it executable
# (chmod +x run_smartgallery.sh) and run it with ./run_smartgallery.sh
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
# FFmpeg is required to enable all video features. Leave this unset to
# let the gallery find ffprobe on PATH, which is usually what you want on
# Linux and macOS.
# export FFPROBE_MANUAL_PATH="/usr/bin/ffprobe"

# --- NETWORK ---
export SERVER_PORT="8189"

# ======================================================================
# OPTIONAL LAUNCH PARAMETERS
# ======================================================================
# Add any of the following to the launch line at the bottom:
#
#   --admin-pass yourpassword   Set the admin password (log in as: admin / yourpassword)
#   --force-login               Require login on the Main Interface (use with --admin-pass)
#   --exhibition                Start in Exhibition Mode instead of the Main Interface
#   --enable-guest-login        Allow anonymous guest access in Exhibition Mode
#   --blind-rating              Hide global averages to prevent user bias
#
# To run Exhibition Mode alongside this one, use the file next to this:
# copy sample_run_exhibition.sh to run_exhibition.sh and set your paths
# and admin password in it. It uses its own port, so both can run at once.

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
exec "$PYTHON" smartgallery.py --port "$SERVER_PORT"
