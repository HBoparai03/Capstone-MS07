# Speech-to-Text Setup

The speech dictation feature uses [Vosk](https://alphacephei.com/vosk/) for fully offline, local speech recognition (~50 MB model, no internet after setup).

---

## Quick Start

```
pip install -r requirements.txt
python download_vosk_model.py
python app.py
```

That's it. The download script handles everything.

---

## What `download_vosk_model.py` does

- Downloads `vosk-model-small-en-us-0.15` (~50 MB) from the official Vosk site
- Extracts and places it as `vosk-model-small-en-us/` next to `app.py`
- Safe to re-run — skips download if the folder already exists

```
hand-gesture-recognition-mediapipe/
    app.py
    download_vosk_model.py
    vosk-model-small-en-us/        <-- created by the script
```

The model folder is in `.gitignore` — each developer runs the script once.

---

## Building the EXE

Run the download script **before** building. The spec auto-detects the model folder and bundles it.

```
python download_vosk_model.py
pyinstaller hand_gesture_app.spec
```

The built EXE in `dist/HandGestureRecognition/` is fully self-contained — no internet required on the end user's machine.

---

## What happens if the model folder is missing

The app launches normally, gesture recognition works. Speech mode shows:

```
Speech: Unavailable (Speech model not found)
```

No crash. Run `python download_vosk_model.py` and relaunch.

---

## Notes

- Vosk runs fully offline after the one-time download
- The model folder is excluded from git — run the script once per machine
- Accuracy is lower than Whisper but sufficient for short dictation phrases
