# Setup script for cc-voice on Windows
# Detects GPU VRAM and installs appropriate PyTorch variant

Write-Host "=== cc-voice Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python not found. Please install Python 3.9+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Detect GPU and VRAM
$useCuda = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "[INFO] NVIDIA GPU detected. Checking VRAM..."
    $vramMiB = nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null |
               ForEach-Object { $_.Trim() } |
               Select-Object -First 1
    if ($vramMiB -match '^\d+$') {
        $vramMiB = [int]$vramMiB
        Write-Host "[INFO] VRAM: $vramMiB MiB"
        if ($vramMiB -ge 4096) {
            Write-Host "[INFO] Sufficient VRAM for GPU inference. Installing PyTorch with CUDA 12.1." -ForegroundColor Green
            $useCuda = $true
        } else {
            Write-Host "[INFO] Insufficient VRAM for large-v3 (need 4096 MiB, have $vramMiB MiB). Using CPU." -ForegroundColor Yellow
        }
    } else {
        Write-Host "[WARN] Could not parse VRAM value. Using CPU." -ForegroundColor Yellow
    }
} else {
    Write-Host "[INFO] No NVIDIA GPU found. Using CPU."
}

# Install PyTorch
if ($useCuda) {
    Write-Host "[INFO] Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "[INFO] Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio
}

# Install other dependencies
Write-Host "[INFO] Installing other dependencies..."
pip install openai-whisper sounddevice soundfile numpy pyperclip pyautogui pynput

# Check ffmpeg
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "[WARN] ffmpeg not found. Whisper requires ffmpeg." -ForegroundColor Yellow
    Write-Host "       Install via:  winget install ffmpeg"
    Write-Host "       Then restart your terminal."
    Write-Host ""
}

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Cyan
Write-Host "Run with:  python voice_input.py"
Write-Host "Shortcut:  Hold Ctrl+Alt+Shift+S to record, release to transcribe"
