@echo off
setlocal EnableExtensions

echo ==================================================
echo  ComfyUI NVIDIA DLSS Frame Interpolation - setup
echo ==================================================
echo.

set "PYTHON_EXE="

rem 1) Portable ComfyUI: ComfyUI_windows_portable\python_embeded\python.exe
if exist "%~dp0..\..\..\python_embeded\python.exe" set "PYTHON_EXE=%~dp0..\..\..\python_embeded\python.exe"

rem 2) venv beside ComfyUI: ComfyUI\venv\Scripts\python.exe
if not defined PYTHON_EXE if exist "%~dp0..\..\venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0..\..\venv\Scripts\python.exe"

rem 3) python on PATH (the WindowsApps store stub fails --version and is skipped)
if not defined PYTHON_EXE python --version >nul 2>nul && set "PYTHON_EXE=python"

rem 4) py launcher
if not defined PYTHON_EXE py --version >nul 2>nul && set "PYTHON_EXE=py"

if not defined PYTHON_EXE (
    echo [ERROR] Could not find a Python interpreter.
    echo.
    echo Tried:
    echo   - ComfyUI_windows_portable\python_embeded\python.exe
    echo   - ComfyUI\venv\Scripts\python.exe
    echo   - python and py on PATH
    echo.
    echo Install ComfyUI first, then re-run this script.
    pause
    exit /b 1
)

echo Python: %PYTHON_EXE%
"%PYTHON_EXE%" --version
echo.

echo Installing Python dependencies from requirements.txt ...
"%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo.
    echo [ERROR] pip install failed. Review the messages above.
    pause
    exit /b 1
)
echo.
echo [OK] Python dependencies installed.
echo.

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [WARN] ffmpeg was not found on PATH.
    echo        The DLSS nodes need FFmpeg and FFprobe.
    echo        Install FFmpeg, add it to PATH, or set DLSS_FFMPEG_PATH
    echo        and DLSS_FFPROBE_PATH before starting ComfyUI.
) else (
    echo [OK] ffmpeg found.
)

where ffprobe >nul 2>nul
if errorlevel 1 (
    echo [WARN] ffprobe was not found on PATH.
) else (
    echo [OK] ffprobe found.
)
echo.

where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [WARN] nvidia-smi was not found.
    echo        An NVIDIA RTX GPU with a current driver is required
    echo        for DLSS processing.
) else (
    echo [OK] NVIDIA driver detected:
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
)

echo.
echo ==================================================
echo  Setup finished. Restart ComfyUI and look for
echo  the "NVIDIA DLSS" nodes in the node menu.
echo ==================================================
pause
