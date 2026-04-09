# Speech-to-Text Setup

The speech dictation feature uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with the Whisper small model.

When running as a Python script, Faster-Whisper can still use the Hugging Face cache normally.

When building the EXE, the Whisper model is bundled from:

`hand-gesture-recognition-mediapipe\whisper-small\`

On this machine that resolves to:

`C:\Users\Harnoor\Capstone\Capstone-MS07\hand-gesture-recognition-mediapipe\whisper-small`

---

## Quick Start

```bash
pip install -r requirements.txt
python download_whisper_model.py
pyinstaller --clean -y hand_gesture_app.spec
```

---

## EXE Behavior

The EXE now prefers the bundled model inside:

`dist\HandGestureRecognition\_internal\speech_models\whisper-small`

If that bundled copy is missing, it falls back to:

`%APPDATA%\HandGestureApp\models\whisper-small`

---

## If Speech Is Unavailable

If speech still cannot load, the app reports the exact model path it tried to use in the runtime log:

`%APPDATA%\HandGestureApp\logs\hand_gesture_app.log`

On this machine that resolves to:

`C:\Users\Harnoor\AppData\Roaming\HandGestureApp\logs\hand_gesture_app.log`

---

## Notes

- The old `vosk-model-small-en-us` folder is no longer used by speech.
- The bundled build now takes its Whisper model from the repo-local `whisper-small` folder.
- The Whisper model is not stored in git.
- `faster-whisper`, `ctranslate2`, `av`, and `sounddevice` must be installed in the build environment.
