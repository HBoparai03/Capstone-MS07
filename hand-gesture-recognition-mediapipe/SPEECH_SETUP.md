# Speech-to-Text Setup

The speech dictation feature uses `faster-whisper` for fully offline, local speech recognition after a one-time model download.

## Quick Start

```bash
pip install -r requirements.txt
python download_whisper_model.py
python app.py
```

That is all you need for the speech path.

## What `download_whisper_model.py` does

- Downloads `Systran/faster-whisper-small`
- Places it as `faster-whisper-small/` next to `app.py`
- Safe to re-run and skips the download when the local model folder is already complete

```text
hand-gesture-recognition-mediapipe/
    app.py
    download_whisper_model.py
    faster-whisper-small/    <-- created by the script
```

The model folder is in `.gitignore`, so each developer runs the script once on their machine.

## Building the EXE

Run the download script before building. The spec auto-detects the local model folder and bundles it.

```bash
python download_whisper_model.py
pyinstaller hand_gesture_app.spec
```

The built EXE in `dist/HandGestureRecognition/` is self-contained for speech dictation as long as the model was present when you built it.

## What happens if the model folder is missing

The app still launches and gesture recognition still works. Speech mode shows:

```text
Speech: Unavailable (Speech model not found ...)
```

No crash. Run `python download_whisper_model.py` and relaunch.

## Notes

- Speech dictation stays offline after the one-time model download
- The bundled speech model is larger than the old Vosk model, but transcription quality is noticeably better
- Dictation still depends on `sounddevice`, a working microphone, and `pyautogui` for typing the transcript into the active window
