"""
Voice input program triggered by Ctrl+Alt+Shift+S.
Hold the shortcut to record, release to transcribe and type into the active window.
Uses OpenAI Whisper (local) for Chinese+English mixed speech recognition.
"""

import threading
import time
import tempfile
import os
import sys
import queue
import msvcrt

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import pyautogui
from pynput import keyboard

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000          # Whisper expects 16 kHz
CHANNELS = 1
WHISPER_MODEL = "large-v3"   # Best for Chinese+English mixing
HOTKEY_KEY = keyboard.Key.f10
HOTKEY_MODIFIERS = {keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
SILENCE_THRESHOLD = 0.01     # RMS below this is considered silence
SILENCE_CHUNK_MS = 20        # Chunk size in ms for silence detection

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
recording = False
audio_frames: list[np.ndarray] = []
record_lock = threading.Lock()
model = None
status_queue: queue.Queue = queue.Queue()
pressed_keys: set = set()

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device() -> str:
    """Return 'cuda' if enough VRAM available, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if vram >= 4.0:
                print(f"[INFO] GPU detected with {vram:.1f} GB VRAM — using CUDA")
                return "cuda"
            else:
                print(f"[INFO] GPU has only {vram:.1f} GB VRAM (need 4 GB for large-v3) — using CPU")
                return "cpu"
    except Exception:
        pass
    print("[INFO] No CUDA available — using CPU")
    return "cpu"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: str):
    global model
    print(f"[INFO] Loading Whisper {WHISPER_MODEL} on {device} ...")
    print("[INFO] First run will download the model (~1.5 GB). Please wait.")
    model = whisper.load_model(WHISPER_MODEL, device=device)
    print(f"[INFO] Model loaded. Ready. Press Ctrl+Alt+Shift+S to start recording.")


# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------

stream: sd.InputStream | None = None


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[WARN] Audio status: {status}")
    audio_frames.append(indata.copy())


def start_recording():
    global recording, audio_frames, stream
    with record_lock:
        if recording:
            return
        audio_frames = []
        recording = True
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=audio_callback,
        )
        stream.start()
    print("[REC] Recording started — press Ctrl+F10 to stop...")


def trim_silence(audio: np.ndarray) -> np.ndarray:
    """Remove leading and trailing silence based on RMS energy."""
    chunk = int(SAMPLE_RATE * SILENCE_CHUNK_MS / 1000)
    rms = lambda a: np.sqrt(np.mean(a ** 2))

    start = 0
    while start + chunk < len(audio):
        if rms(audio[start:start + chunk]) >= SILENCE_THRESHOLD:
            break
        start += chunk

    end = len(audio)
    while end - chunk > start:
        if rms(audio[end - chunk:end]) >= SILENCE_THRESHOLD:
            break
        end -= chunk

    return audio[start:end]


def stop_recording_and_transcribe():
    global recording, stream
    with record_lock:
        if not recording:
            return
        recording = False
        if stream is not None:
            stream.stop()
            stream.close()
            stream = None

    if not audio_frames:
        print("[WARN] No audio captured.")
        return

    audio_data = np.concatenate(audio_frames, axis=0).flatten()
    audio_data = trim_silence(audio_data)
    if len(audio_data) == 0:
        print("[WARN] Audio is silent after trimming.")
        return
    duration = len(audio_data) / SAMPLE_RATE
    print(f"[REC] Stopped. Captured {duration:.1f}s of audio. Transcribing...")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    sf.write(tmp_path, audio_data, SAMPLE_RATE)

    try:
        result = model.transcribe(
            tmp_path,
            language=None,       # auto-detect; handles mixed Chinese+English
            task="transcribe",
            fp16=False,          # fp16 only safe on CUDA; CPU needs fp32
        )
        text = result["text"].strip()
        if text:
            print(f"[TEXT] {text}")
            type_text(text)
        else:
            print("[WARN] Transcription returned empty text.")
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
    finally:
        os.unlink(tmp_path)


def type_text(text: str):
    """Type text into the active window."""
    try:
        pyautogui.write(text, interval=0.01)
    except Exception as e:
        print(f"[ERROR] Failed to type text: {e}")


# ---------------------------------------------------------------------------
# Key state tracking
# ---------------------------------------------------------------------------

def on_press(key):
    global recording
    # Ctrl+C is intercepted by pynput's low-level hook on Windows before the
    # terminal can raise KeyboardInterrupt — handle it explicitly here.
    if getattr(key, 'char', None) == '\x03':
        print("\n[INFO] Exiting.")
        os._exit(0)
    pressed_keys.add(key)
    ctrl = pressed_keys & HOTKEY_MODIFIERS
    if key == HOTKEY_KEY and ctrl:
        if not recording:
            start_recording()
        else:
            threading.Thread(target=stop_recording_and_transcribe, daemon=True).start()


def on_release(key):
    pressed_keys.discard(key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if sys.platform != "win32":
        print("[WARN] This program is designed for Windows.")

    # Single-instance lock using an exclusive file lock
    lock_path = os.path.join(tempfile.gettempdir(), "cc-voice.lock")
    try:
        lock_file = open(lock_path, "w")
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("[ERROR] Another instance of voice_input.py is already running. Exiting.")
        sys.exit(1)

    device = select_device()

    # Load model in background so keyboard listener starts immediately
    loader_thread = threading.Thread(target=load_model, args=(device,), daemon=True)
    loader_thread.start()

    print("[INFO] Starting keyboard listener...")
    print("[INFO] Shortcut: Ctrl+F10 to start recording, Ctrl+F10 again to stop and transcribe")

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            loader_thread.join()  # Wait for model to finish loading before accepting input
            print("[READY] Listening for hotkey. Press Ctrl+F10 to toggle recording. Press Ctrl+C to quit.")
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\n[INFO] Exiting.")
                listener.stop()


if __name__ == "__main__":
    main()
