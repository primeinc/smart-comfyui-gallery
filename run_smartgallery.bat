@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat

:: --- CONFIGURATION: replace with your real paths ---
:: Use forward slashes (/) even on Windows
set "BASE_OUTPUT_PATH="
set "BASE_INPUT_PATH="
set "BASE_SMARTGALLERY_PATH="
set "FFPROBE_MANUAL_PATH=C:/ffmpeg/bin/ffprobe.exe"
set SERVER_PORT=8190
set ENABLE_AI_DAM=true
set GENERATE_THUMBNAILS=Off

:: --- OPTIONAL LAUNCH PARAMETERS ---
:: Add any of the following to the python command below depending on your scenario:
::
::   --admin-pass yourpassword   Set the admin password (log in as: admin / yourpassword)
::   --force-login               Require login on the Main Interface (use with --admin-pass)
::   --exhibition                Start in Exhibition Mode instead of the Main Interface
::   --port 8190                 Use a different port (default: 8189)
::   --enable-guest-login        Allow anonymous guest access in Exhibition
::   --blind-rating              Hide global averages to prevent user bias
::
:: Example – Main Interface with login enforced:
::   python smartgallery.py --port 8189 --admin-pass yourpassword --force-login
::
:: Example – Exhibition on port 8190 with Blind Rating:
::   python smartgallery.py --exhibition --port 8190 --admin-pass yourpassword --blind-rating

:: --- START ---
python smartgallery.py
pause