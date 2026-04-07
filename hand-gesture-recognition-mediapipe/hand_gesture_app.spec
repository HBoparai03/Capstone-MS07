# PyInstaller spec for Hand Gesture Recognition (one-dir build, trimmed asset set)
# Build: pyinstaller hand_gesture_app.spec

import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None

# Data files to bundle (models and labels)
model_root = 'model'
keypoint_dir = os.path.join(model_root, 'keypoint_classifier')
point_history_dir = os.path.join(model_root, 'point_history_classifier')

mediapipe_binaries = collect_dynamic_libs('mediapipe')
mediapipe_datas = collect_data_files('mediapipe')
vosk_hiddenimports = collect_submodules('vosk')
sounddevice_datas = collect_data_files('_sounddevice_data')

# Bundle the vosk model folder if it has been downloaded
_vosk_model_src = 'vosk-model-small-en-us'
_vosk_model_datas = [(_vosk_model_src, _vosk_model_src)] if os.path.isdir(_vosk_model_src) else []

datas = [
    (os.path.join(keypoint_dir, 'keypoint_classifier.tflite'), keypoint_dir),
    (os.path.join(keypoint_dir, 'keypoint_classifier_label.csv'), keypoint_dir),
    (os.path.join(point_history_dir, 'point_history_classifier.tflite'), point_history_dir),
    (os.path.join(point_history_dir, 'point_history_classifier_label.csv'), point_history_dir),
    ('icon.ico', '.'),
    ('icons/Ok.png', 'icons'),
    ('icons/fourfu.png', 'icons'),
    ('icons/pinch.png', 'icons'),
    ('icons/tdown.png', 'icons'),
    ('icons/threefu.png', 'icons'),
    ('icons/tup.png', 'icons'),
    ('icons/twofu.png', 'icons'),
] + mediapipe_datas + sounddevice_datas + _vosk_model_datas

# Hidden imports often needed by TensorFlow, OpenCV, PyQt5, pystray
hiddenimports = [
    'numpy', 'cv2', 'mediapipe', 'tensorflow', 'tensorflow.lite',
    'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
    'pystray', 'PIL', 'PIL._tkinter_finder',
    'pyautogui', 'comtypes', 'pycaw',
    'sounddevice', '_sounddevice', '_sounddevice_data',
    'vosk', 'cffi',
    'mediapipe.python._framework_bindings',
    'mediapipe.python._framework_bindings.calculator_graph',
    'mediapipe.python._framework_bindings.packet',
] + vosk_hiddenimports

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=mediapipe_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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
    name='HandGestureRecognition',
    icon='icon.ico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window for GUI app
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
    name='HandGestureRecognition',
)
