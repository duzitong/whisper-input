"""
Voice input program triggered by Ctrl+Alt+Shift+S.
Hold the hotkey to record, release to transcribe and type into the active window.
Uses OpenAI Whisper (local) for Chinese+English mixed speech recognition.
"""

import threading
import time
import tempfile
import os
import sys
import queue
import msvcrt
import signal
import tkinter as tk

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import pyperclip
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
SILENCE_PAD_MS = 200         # Padding to keep around detected speech (ms)

# Optional HTTP/HTTPS proxy for downloading the Whisper model.
# Examples: "http://127.0.0.1:7890"  or  "http://user:pass@proxy.corp:8080"
# Leave empty string "" to use no proxy (direct connection).
PROXY = ""

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
# Status indicator overlay
# ---------------------------------------------------------------------------

STATES = {
    "idle":         ("IDLE",         "#555555", "white"),
    "loading":      ("LOADING...",   "#555555", "white"),
    "recording":    ("RECORDING",    "#cc2222", "white"),
    "transcribing": ("TRANSCRIBING", "#ccaa00", "black"),
    "done":         ("DONE",         "#22aa44", "white"),
}


class Indicator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)       # no title bar
        self.root.attributes("-topmost", True) # always on top
        self.root.attributes("-alpha", 0.85)

        self.label = tk.Label(
            self.root,
            text="",
            font=("Segoe UI", 11, "bold"),
            padx=12, pady=6,
        )
        self.label.pack()

        # Position: bottom-right of primary monitor
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = self.root.winfo_reqwidth()
        h = self.root.winfo_reqheight()
        self.root.geometry(f"+{sw - w - 20}+{sh - h - 60}")

        self._after_id = None
        self.set("loading")

    def set(self, state: str):
        text, bg, fg = STATES[state]
        self.label.config(text=text, bg=bg, fg=fg)
        self.root.config(bg=bg)
        # Refit window to label size
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = self.root.winfo_reqwidth()
        h = self.root.winfo_reqheight()
        self.root.geometry(f"+{sw - w - 20}+{sh - h - 60}")

    def set_done_then_idle(self):
        self.set("done")
        if self._after_id:
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(2000, lambda: self.set("idle"))

    def run(self):
        self.root.mainloop()


indicator: Indicator | None = None


def set_status(state: str):
    """Thread-safe status update — schedules onto the tkinter main thread."""
    if indicator is None:
        return
    if state == "done":
        indicator.root.after(0, indicator.set_done_then_idle)
    else:
        indicator.root.after(0, indicator.set, state)

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

    # Apply proxy settings for the model download if configured.
    _proxy_keys = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
    _saved_env: dict[str, str | None] = {}
    if PROXY:
        print(f"[INFO] Using proxy for model download: {PROXY}")
        for _k in _proxy_keys:
            _saved_env[_k] = os.environ.get(_k)
            os.environ[_k] = PROXY
    try:
        model = whisper.load_model(WHISPER_MODEL, device=device)
    finally:
        if PROXY:
            for _k, _v in _saved_env.items():
                if _v is None:
                    os.environ.pop(_k, None)
                else:
                    os.environ[_k] = _v

    print(f"[INFO] Model loaded. Ready. Hold Ctrl+F10 to record, release to transcribe.")
    set_status("idle")


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
    set_status("recording")
    print("[REC] Recording started — release Ctrl+F10 to stop...")


def trim_silence(audio: np.ndarray) -> np.ndarray:
    """Remove leading and trailing silence, keeping a small pad around speech."""
    chunk = int(SAMPLE_RATE * SILENCE_CHUNK_MS / 1000)
    pad = int(SAMPLE_RATE * SILENCE_PAD_MS / 1000)
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

    start = max(0, start - pad)
    end = min(len(audio), end + pad)
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
        set_status("idle")
        return

    audio_data = np.concatenate(audio_frames, axis=0).flatten()
    audio_data = trim_silence(audio_data)
    if len(audio_data) == 0:
        print("[WARN] Audio is silent after trimming.")
        set_status("idle")
        return
    duration = len(audio_data) / SAMPLE_RATE
    print(f"[REC] Stopped. Captured {duration:.1f}s of audio. Transcribing...")
    set_status("transcribing")

    # Pass numpy array directly — no temp file, no ffmpeg dependency.
    try:
        result = model.transcribe(
            audio_data,
            language=None,       # auto-detect; handles mixed Chinese+English
            task="transcribe",
            fp16=False,          # fp16 only safe on CUDA; CPU needs fp32
        )
        text = result["text"].strip()
        if text:
            print(f"[TEXT] {text}")
            type_text(text)
            set_status("done")
        else:
            print("[WARN] Transcription returned empty text.")
            set_status("idle")
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        set_status("idle")


def type_text(text: str):
    """Type text into the active window via clipboard paste for Unicode/CJK support."""
    try:
        pyperclip.copy(text)
        time.sleep(0.05)
        pyautogui.hotkey("ctrl", "v")
    except Exception as e:
        print(f"[ERROR] Failed to type text: {e}")


# ---------------------------------------------------------------------------
# Key state tracking
# ---------------------------------------------------------------------------

def on_press(key):
    pressed_keys.add(key)
    ctrl = pressed_keys & HOTKEY_MODIFIERS
    if key == HOTKEY_KEY and ctrl and not recording:
        start_recording()


def on_release(key):
    pressed_keys.discard(key)
    if key == HOTKEY_KEY and recording:
        threading.Thread(target=stop_recording_and_transcribe, daemon=True).start()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global indicator

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

    indicator = Indicator()

    loader_thread = threading.Thread(target=load_model, args=(device,), daemon=True)
    loader_thread.start()

    print("[INFO] Starting keyboard listener...")
    print("[INFO] Shortcut: Hold Ctrl+F10 to record, release to transcribe")

    def run_listener():
        loader_thread.join()
        print("[READY] Listening for hotkey. Hold Ctrl+F10 to record. Press Ctrl+C to quit.")
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
        indicator.root.after(0, indicator.root.destroy)

    def handle_sigint(sig, frame):
        print("\n[INFO] Exiting.")
        os._exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    # Tkinter on Windows blocks signal delivery unless we periodically yield.
    def poll_signals():
        indicator.root.after(200, poll_signals)

    threading.Thread(target=run_listener, daemon=True).start()
    indicator.root.after(200, poll_signals)
    indicator.run()  # tkinter mainloop on main thread


if __name__ == "__main__":
    main()
