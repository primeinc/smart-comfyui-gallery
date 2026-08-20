@echo off
setlocal EnableExtensions
cd /d "%~dp0"

:: ======================================================================
:: SMARTGALLERY DAM -- CONFIGURATION -- EXHIBITION MODE
:: Instructions: Rename this file to "run_exhibition.bat" to use it.
:: Exhibition is the read-only portal you share with family, friends or
:: clients. It runs alongside the main gallery on its own port, so keep
:: run_smartgallery.bat as it is.
:: ======================================================================

:: --- CORE PATHS ---
:: Write these however you like. Backslashes or forward slashes, with or
:: without a trailing slash, and the quotes Explorer's "Copy as path"
:: wraps around a path are all fine -- the gallery normalises them.
set "BASE_OUTPUT_PATH=C:\Path\To\ComfyUI\output"
set "BASE_INPUT_PATH=C:\Path\To\ComfyUI\input"

:: --- CACHE & DATABASE PATH ---
:: Directory for the SQLite database and thumbnail cache.
:: You can store this anywhere; it does NOT need to be inside the ComfyUI output folder.
set "BASE_SMARTGALLERY_PATH=C:\Path\To\ComfyUI\output"

:: --- FFMPEG CONFIGURATION (Highly Recommended) ---
:: FFmpeg is required to enable all video features. Point this at your
:: ffmpeg install and the gallery works out the rest: the ffprobe program,
:: the ffmpeg program beside it, or the folder either of them lives in --
:: any of those is enough. Leave it blank to use whatever is on PATH.
set "FFPROBE_MANUAL_PATH=C:\Path\To\ffmpeg\bin"

:: --- NETWORK ---
:: Exhibition runs on its own port so it can sit beside the main gallery.
set "SERVER_PORT=8190"

:: --- ADMIN PASSWORD (REQUIRED) ---
:: Exhibition always needs an admin account, so choose your own password
:: here. Minimum 8 characters. Until you set one the gallery refuses to
:: start and says so -- it will not fall back to a default, because a
:: password shipped in this file would be the same one for everybody who
:: downloaded it.
:: Set it here rather than on the launch line below: command lines are
:: visible to other programs on the machine, which is why the gallery
:: masks the password out of its own startup log.
set "ADMIN_PASSWORD="


:: ======================================================================
:: OPTIONAL LAUNCH PARAMETERS
:: ======================================================================
:: Add any of the following to the launch line at the bottom:
::
::   --enable-guest-login        Show a "Login as Guest" button, so people
::                                 can browse without an account
::   --blind-rating              Hide global averages to prevent user bias


:: ======================================================================
:: STARTUP -- nothing below here needs editing
:: ======================================================================

:: --- WHERE IS THE APP? ---
:: Next to this file in a normal download; inside app\ in the portable build.
set "APP_DIR="
if exist "smartgallery.py" set "APP_DIR=."
if not defined APP_DIR if exist "app\smartgallery.py" set "APP_DIR=app"
if not defined APP_DIR (
    echo.
    echo ERROR: could not find smartgallery.py.
    echo Looked in "%CD%" and "%CD%\app".
    echo Put this file in the same folder as smartgallery.py.
    echo.
    pause
    exit /b 1
)

:: --- WHICH PYTHON? ---
:: A virtual environment named .venv or venv is used when there is one.
set "PYTHON="
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
if not defined PYTHON if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"
if not defined PYTHON if exist "%APP_DIR%\.venv\Scripts\python.exe" set "PYTHON=%APP_DIR%\.venv\Scripts\python.exe"
if not defined PYTHON if exist "%APP_DIR%\venv\Scripts\python.exe" set "PYTHON=%APP_DIR%\venv\Scripts\python.exe"
if not defined PYTHON if exist "python\python.exe" set "PYTHON=python\python.exe"
if not defined PYTHON (
    echo.
    echo ERROR: no Python environment found.
    echo Looked for .venv\Scripts\python.exe, venv\Scripts\python.exe and
    echo python\python.exe under "%CD%".
    echo.
    echo Create one, either way:
    echo.
    echo   With uv - installs the exact versions pinned in uv.lock.
    echo   If you do not have uv yet:
    echo     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo   then, in this folder:
    echo     uv sync
    echo.
    echo   Or with the Python you already have:
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: Make it absolute before moving into the app folder, or a relative
:: interpreter path stops resolving the moment we change directory.
for %%I in ("%PYTHON%") do set "PYTHON=%%~fI"
echo Using Python: %PYTHON%

:: Open browser automatically in the background after a 3-second delay
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:%SERVER_PORT%/galleryout/"

:: Move to the app folder so Flask finds the templates correctly
pushd "%APP_DIR%"
"%PYTHON%" smartgallery.py --port %SERVER_PORT% --exhibition
popd

pause
