@echo off
cd /d "%~dp0"

:: ======================================================================
:: SMARTGALLERY DAM PORTABLE CONFIGURATION -- EXHIBITION MODE
:: Instructions: Rename this file to "run_exhibition.bat" to use it.
:: Exhibition is the read-only portal you share with family, friends or
:: clients. It runs alongside the main gallery on its own port, so keep
:: run_smartgallery.bat as it is.
:: ======================================================================

:: --- CORE PATHS ---
:: IMPORTANT: Use forward slashes (/) even on Windows!
set "BASE_OUTPUT_PATH=C:/Path/To/ComfyUI/output"
set "BASE_INPUT_PATH=C:/Path/To/ComfyUI/input"

:: --- CACHE & DATABASE PATH ---
:: Directory for the SQLite database and thumbnail cache.
:: You can store this anywhere; it does NOT need to be inside the ComfyUI output folder.
set "BASE_SMARTGALLERY_PATH=C:/Path/To/ComfyUI/output"

:: --- FFMPEG CONFIGURATION (Highly Recommended) ---
:: FFmpeg is required to enable all video features. 
:: Provide the exact path to the 'ffprobe.exe' utility below.
set "FFPROBE_MANUAL_PATH=C:/Path/To/ffmpeg/bin/ffprobe.exe"

:: --- NETWORK ---
:: Exhibition runs on its own port so it can sit beside the main gallery.
set "SERVER_PORT=8190"

:: --- ADMIN PASSWORD (REQUIRED) ---
:: Exhibition always needs an admin account, so choose your own password
:: here. Minimum 8 characters. Until you set one the gallery refuses to
:: start and says so -- it will not fall back to a default, because a
:: password shipped in this file would be the same one for everybody who
:: downloaded it.
:: Set it here rather than on the command line below: command lines are
:: visible to other programs on the machine, which is why the gallery
:: masks the password out of its own startup log.
set "ADMIN_PASSWORD="


:: ======================================================================
:: OPTIONAL LAUNCH PARAMETERS
:: ======================================================================
:: Add any of the following to the python command at the bottom:
::
::   --enable-guest-login        Show a "Login as Guest" button, so people
::                                 can browse without an account
::   --blind-rating              Hide global averages to prevent user bias
::
:: Example - Exhibition with guest access and unbiased rating:
::   ..\python\python.exe smartgallery.py --port %SERVER_PORT% --exhibition --enable-guest-login --blind-rating


:: --- STARTUP SEQUENCE ---
echo Starting SmartGallery Portable Edition...

:: Open browser automatically in the background after a 3-second delay
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:%SERVER_PORT%/galleryout/"

:: Move to the app folder so Flask finds the templates correctly
cd app

:: Launch the server
..\python\python.exe smartgallery.py --port %SERVER_PORT% --exhibition

pause