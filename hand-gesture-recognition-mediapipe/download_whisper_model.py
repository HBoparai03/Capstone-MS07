from huggingface_hub import snapshot_download
import os
model_dir = os.path.join(os.environ["APPDATA"], "HandGestureApp", "models", "whisper-small")
os.makedirs(model_dir, exist_ok=True)
snapshot_download(repo_id="Systran/faster-whisper-small", local_dir=model_dir)