#!/usr/bin/env python3
"""Download the Vosk small English model used for speech dictation.

Run this once before launching the app or building the EXE:
    python download_vosk_model.py

The model (~50 MB) is saved as:
    vosk-model-small-en-us/   (next to this script / app.py)
"""

import os
import sys
import shutil
import tempfile
import urllib.request
import zipfile

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
ARCHIVE_NAME = "vosk-model-small-en-us-0.15"
DEST_NAME = "vosk-model-small-en-us"
DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEST_NAME)


def _progress_hook(count, block_size, total_size):
    if total_size <= 0:
        return
    pct = min(count * block_size * 100 // total_size, 100)
    bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
    sys.stdout.write(f"\r  [{bar}] {pct}%")
    sys.stdout.flush()


def download():
    if os.path.isdir(DEST_DIR):
        print(f"Model already exists at: {DEST_DIR}")
        print("Delete that folder and re-run if you want to re-download.")
        return

    print(f"Downloading Vosk small English model from:\n  {MODEL_URL}\n")

    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "vosk_model.zip")

    try:
        urllib.request.urlretrieve(MODEL_URL, zip_path, reporthook=_progress_hook)
        print()  # newline after progress bar

        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        extracted = os.path.join(tmp_dir, ARCHIVE_NAME)
        if not os.path.isdir(extracted):
            # Fallback: find whatever folder was extracted
            entries = [e for e in os.listdir(tmp_dir)
                       if os.path.isdir(os.path.join(tmp_dir, e)) and e != "__MACOSX"]
            if not entries:
                raise RuntimeError("Could not find extracted model folder in zip.")
            extracted = os.path.join(tmp_dir, entries[0])

        shutil.move(extracted, DEST_DIR)
        print(f"Model saved to: {DEST_DIR}")
        print("Done. You can now run app.py or build the EXE.")

    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    download()
