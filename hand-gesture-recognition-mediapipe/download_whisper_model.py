#!/usr/bin/env python3
"""Download the faster-whisper small model used for speech dictation.

Run this once before launching the app or building the EXE:
    python download_whisper_model.py

The model is saved as:
    faster-whisper-small/   (next to this script / app.py)
"""

import os
import shutil
import sys
import tempfile

MODEL_REPO_ID = "Systran/faster-whisper-small"
DEST_NAME = "faster-whisper-small"
REQUIRED_FILES = ("config.json", "model.bin")
DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEST_NAME)


def _is_model_ready(path):
    return os.path.isdir(path) and all(
        os.path.isfile(os.path.join(path, filename))
        for filename in REQUIRED_FILES
    )


def download():
    if _is_model_ready(DEST_DIR):
        print(f"Model already exists at: {DEST_DIR}")
        print("Delete that folder and re-run if you want to re-download.")
        return

    if os.path.exists(DEST_DIR):
        print(f"Removing incomplete model folder at: {DEST_DIR}")
        shutil.rmtree(DEST_DIR, ignore_errors=True)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        print(f"Error importing huggingface_hub: {exc}", file=sys.stderr)
        print("Install dependencies with: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading faster-whisper model from:\n  {MODEL_REPO_ID}\n")
    print("This may take a while the first time.\n")

    tmp_dir = tempfile.mkdtemp()
    tmp_model_dir = os.path.join(tmp_dir, DEST_NAME)

    try:
        snapshot_download(
            repo_id=MODEL_REPO_ID,
            local_dir=tmp_model_dir,
        )

        if not _is_model_ready(tmp_model_dir):
            raise RuntimeError("Downloaded model folder is missing required files.")

        shutil.move(tmp_model_dir, DEST_DIR)
        print(f"Model saved to: {DEST_DIR}")
        print("Done. You can now run app.py or build the EXE.")
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    download()
