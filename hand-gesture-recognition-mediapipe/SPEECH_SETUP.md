# Speech-to-Text Setup

The speech dictation feature uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper small model, ~244 MB).

When running as a **Python script**, the model is downloaded automatically to the HuggingFace cache on first use — no setup needed.

When running as a **built EXE**, the model must be downloaded separately to a local folder before speech will work. The EXE cannot download it on its own.

---

## Running as a Python Script (dev/team)

No extra steps. On first speech activation the model downloads automatically via HuggingFace. Subsequent runs load from cache instantly.

Requirements:
```
pip install -r requirements.txt
python app.py
```

---

## Running as the EXE (distributed build)

### Step 1 — Build the EXE

```
pyinstaller hand_gesture_app.spec
```

The output is in `dist/HandGestureRecognition/`.

### Step 2 — Download the Whisper model

Run this **once** on the machine that will use the EXE:

```python
python download_whisper_model.py
```

This saves the model to:
```
%APPDATA%\HandGestureApp\models\whisper-small\
```

> If `download_whisper_model.py` does not exist yet, you can trigger the download manually from a Python shell:
> ```python
> from huggingface_hub import snapshot_download
> import os
> model_dir = os.path.join(os.environ["APPDATA"], "HandGestureApp", "models", "whisper-small")
> os.makedirs(model_dir, exist_ok=True)
> snapshot_download(repo_id="Systran/faster-whisper-small", local_dir=model_dir)
> ```

### Step 3 — Run the EXE

```
dist/HandGestureRecognition/HandGestureRecognition.exe
```

The model is loaded from `%APPDATA%` on every launch — no internet required after Step 2.

---

## What happens if the model is missing (EXE)

The app launches normally and gesture recognition works. Speech mode will show:

```
Speech: Unavailable (Speech model not downloaded)
```

No crash. Run Step 2 above and relaunch to enable speech.

---

## Model location reference

| Mode       | Model source                                              |
|------------|-----------------------------------------------------------|
| Script     | `~/.cache/huggingface/hub/` (managed automatically)      |
| EXE        | `%APPDATA%\HandGestureApp\models\whisper-small\`          |

---

## Notes

- The model download is ~244 MB and only needs to happen once per machine.
- The model folder is not included in the git repo (too large for GitHub). It lives only on the local machine.
- `huggingface_hub` must be installed: `pip install huggingface_hub`
