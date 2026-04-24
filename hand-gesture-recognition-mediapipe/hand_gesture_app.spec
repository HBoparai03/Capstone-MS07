# PyInstaller spec for Hand Gesture Recognition (one-dir build)
# Build: pyinstaller hand_gesture_app.spec

import os

import mediapipe
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

block_cipher = None

model_root = "model"
keypoint_dir = os.path.join(model_root, "keypoint_classifier")
point_history_dir = os.path.join(model_root, "point_history_classifier")

mp_root = os.path.dirname(mediapipe.__file__)
mp_binaries = collect_dynamic_libs("mediapipe")
faster_whisper_datas, faster_whisper_binaries, faster_whisper_hiddenimports = collect_all("faster_whisper", include_py_files=True)
ctranslate2_datas, ctranslate2_binaries, ctranslate2_hiddenimports = collect_all("ctranslate2", include_py_files=True)
av_datas, av_binaries, av_hiddenimports = collect_all("av", include_py_files=True)
sounddevice_datas = collect_data_files("_sounddevice_data")

fast_whisper_model_path = "faster-whisper-small"

datas = [
    (os.path.join(keypoint_dir, "keypoint_classifier.tflite"), keypoint_dir),
    (os.path.join(keypoint_dir, "keypoint_classifier_label.csv"), keypoint_dir),
    (os.path.join(point_history_dir, "point_history_classifier.tflite"), point_history_dir),
    (os.path.join(point_history_dir, "point_history_classifier_label.csv"), point_history_dir),
    (os.path.join(mp_root, "modules"), "mediapipe/modules"),
    ("icon.ico", "."),
    ("icons/Ok.png", "icons"),
    ("icons/fourfu.png", "icons"),
    ("icons/pinch.png", "icons"),
    ("icons/tdown.png", "icons"),
    ("icons/threefu.png", "icons"),
    ("icons/tup.png", "icons"),
    ("icons/twofu.png", "icons"),
] + faster_whisper_datas + ctranslate2_datas + av_datas + sounddevice_datas

if os.path.isdir(fast_whisper_model_path):
    datas.append((fast_whisper_model_path, fast_whisper_model_path))

hiddenimports = [
    "numpy",
    "cv2",
    "mediapipe",
    "tensorflow",
    "tensorflow.lite",
    "PyQt5.QtCore",
    "PyQt5.QtGui",
    "PyQt5.QtWidgets",
    "pystray",
    "PIL",
    "PIL._tkinter_finder",
    "pyautogui",
    "comtypes",
    "pycaw",
    "mediapipe.python._framework_bindings",
    "sounddevice",
    "_sounddevice",
    "_sounddevice_data",
    "faster_whisper",
    "ctranslate2",
    "av",
] + faster_whisper_hiddenimports + ctranslate2_hiddenimports + av_hiddenimports

binaries = mp_binaries + faster_whisper_binaries + ctranslate2_binaries + av_binaries

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["pyinstaller_hooks/rthook_mediapipe_dlls.py"],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    [],
    name="HandGestureRecognition",
    icon="icon.ico",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    exclude_binaries=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="HandGestureRecognition",
)
