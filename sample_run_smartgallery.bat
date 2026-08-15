@echo off
cd /d "%~dp0"

:: ======================================================================
:: SMARTGALLERY DAM PORTABLE CONFIGURATION
:: Instructions: Rename this file to "run_smartgallery.bat" to use it.
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
set "SERVER_PORT=8189"


:: ======================================================================
:: OPTIONAL LAUNCH PARAMETERS
:: ======================================================================
:: Add any of the following to the python command at the bottom depending on your scenario:
::
::   --admin-pass yourpassword   Set the admin password (log in as: admin / yourpassword)
::   --force-login               Require login on the Main Interface (use with --admin-pass)
::   --exhibition                Start in Exhibition Mode instead of the Main Interface
::   --enable-guest-login        Allow anonymous guest access in Exhibition Mode
::   --blind-rating              Hide global averages to prevent user bias
::
:: Example 1 - Main Interface with login enforced:
::   ..\python\python.exe smartgallery.py --port %SERVER_PORT% --admin-pass yourpassword --force-login
::
:: ----------------------------------------------------------------------
:: 🌐 OPTIONAL: HOW TO RUN EXHIBITION MODE
:: Exhibition Mode is completely optional. It is a safe, read-only portal 
:: designed specifically to share your work with family, friends, or clients.
:: If you want to run it alongside your main gallery:
::
:: 1. Copy this file and rename it to "run_exhibition.bat"
:: 2. Open it and change the SERVER_PORT variable above to 8190
:: 3. Change the python command at the bottom of the file to look like this:
::    ..\python\python.exe smartgallery.py --port %SERVER_PORT% --exhibition --admin-pass yourpassword
:: ----------------------------------------------------------------------


:: --- STARTUP SEQUENCE ---
echo Starting SmartGallery Portable Edition...

:: Open browser automatically in the background after a 3-second delay
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:%SERVER_PORT%/galleryout/"

:: Move to the app folder so Flask finds the templates correctly
cd app

:: Launch the server
..\python\python.exe smartgallery.py --port %SERVER_PORT%

pause