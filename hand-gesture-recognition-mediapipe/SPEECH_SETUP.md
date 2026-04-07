# Speech-to-Text Setup

The speech dictation feature uses [Vosk](https://alphacephei.com/vosk/) for fully offline, local speech recognition.
The small English model is ~50 MB and works without internet after initial setup.

---

## Step 1 — Download the Vosk model

Download the small English model and extract it so the folder is named exactly `vosk-model-small-en-us` in the same directory as `app.py`:

```
hand-gesture-recognition-mediapipe/
    app.py
    vosk-model-small-en-us/       <-- place it here
        am/
        conf/
        graph/
        ...
```

Download link: https://alphacephei.com/vosk/models
Model to use: **vosk-model-small-en-us-0.15** (or latest small English model)

---

## Step 2 — Install dependencies

```
pip install vosk sounddevice pyautogui
```

---

## Running as a Python Script

```
python app.py
```

Speech activates via gesture. The model loads from `vosk-model-small-en-us/` next to `app.py`.

---

## Building and Running the EXE

### Build

Place the `vosk-model-small-en-us/` folder next to `app.py` **before** building.
The spec automatically detects and bundles it:

```
pyinstaller hand_gesture_app.spec
```

The model is bundled inside `dist/HandGestureRecognition/` — no internet required on the end user's machine.

### Run

```
dist/HandGestureRecognition/HandGestureRecognition.exe
```

---

## What happens if the model folder is missing

The app launches normally and gesture recognition works. Speech mode will show:

```
Speech: Unavailable (Speech model not found)
```

No crash. Add the model folder and relaunch to enable speech.

---

## Notes

- The `vosk-model-small-en-us/` folder is excluded from git (too large). Each developer downloads it once.
- Vosk runs fully offline — no HuggingFace downloads, no internet dependency.
- Accuracy is lower than Whisper but sufficient for short dictation commands.
