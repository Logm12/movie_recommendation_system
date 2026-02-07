@echo off
echo ========================================================
echo  Setting up Linux Dependencies for Docker (in backend/libs)
echo ========================================================

REM 1. Create the directory if it doesn't exist
if not exist "backend\libs" mkdir "backend\libs"

REM 2. Run a temporary Python container to install Linux-compatible wheels
REM    We mount the current directory to access requirements.txt
REM    We mount backend/libs to target the installation
echo.
echo Downloading Linux binaries... (This caches them on E: drive)
echo.

docker run --rm -v "%cd%/backend:/app" -v "%cd%/backend/libs:/app/libs" python:3.9-slim /bin/bash -c "pip install -t /app/libs --no-cache-dir -r /app/requirements.txt"

echo.
echo ========================================================
echo  Cleaning up C: Drive Pip Cache
echo ========================================================
python -m pip cache purge

echo.
echo ========================================================
echo  DONE!
echo  Dependencies are installed in: %cd%\backend\libs
echo  You can now run 'docker-compose up --build'
echo ========================================================
pause
