@echo off
REM Setup script for cc-voice on Windows
REM Detects GPU VRAM and installs appropriate PyTorch variant

echo === cc-voice Setup ===
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ from https://python.org
    pause & exit /b 1
)

REM Check for NVIDIA GPU and VRAM
set USE_CUDA=0
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [INFO] NVIDIA GPU detected. Checking VRAM...
    REM Query VRAM in MiB
    for /f "tokens=1" %%a in ('nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2^>nul') do set VRAM_MiB=%%a
    echo [INFO] VRAM: %VRAM_MiB% MiB
    if defined VRAM_MiB (
        REM 4096 MiB = 4 GB required for large-v3
        if %VRAM_MiB% GEQ 4096 (
            echo [INFO] Sufficient VRAM for GPU inference. Will install PyTorch with CUDA 12.1.
            set USE_CUDA=1
        ) else (
            echo [INFO] Insufficient VRAM for large-v3 ^(need 4096 MiB, have %VRAM_MiB% MiB^). Using CPU.
        )
    )
) else (
    echo [INFO] No NVIDIA GPU found. Using CPU.
)

REM Install PyTorch
if "%USE_CUDA%"=="1" (
    echo [INFO] Installing PyTorch with CUDA 12.1...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo [INFO] Installing PyTorch ^(CPU only^)...
    pip install torch torchvision torchaudio
)

REM Install other requirements
echo [INFO] Installing other dependencies...
pip install openai-whisper sounddevice soundfile numpy pyperclip pyautogui pynput

REM ffmpeg check
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARN] ffmpeg not found. Whisper requires ffmpeg.
    echo        Install via: winget install ffmpeg
    echo        Or download from https://ffmpeg.org/download.html and add to PATH.
    echo.
)

echo.
echo === Setup complete ===
echo Run with:  python voice_input.py
echo Shortcut:  Hold Ctrl+Alt+Shift+S to record, release to transcribe
pause
