# PyInstaller spec for Hand Gesture Recognition (one-dir build, trimmed asset set)
# Build: pyinstaller hand_gesture_app.spec

import os
import mediapipe
from PyInstaller.utils.hooks import collect_dynamic_libs  # ✅ ADD

block_cipher = None

# Data files to bundle (models and labels)
model_root = 'model'
keypoint_dir = os.path.join(model_root, 'keypoint_classifier')
point_history_dir = os.path.join(model_root, 'point_history_classifier')

mp_root = os.path.dirname(mediapipe.__file__)

# ✅ ADD (MediaPipe DLL fix)
mp_binaries = collect_dynamic_libs('mediapipe')

# ✅ OPTIONAL (Vosk model)
vosk_model_path = 'vosk-model-small-en-us'

datas = [
    (os.path.join(keypoint_dir, 'keypoint_classifier.tflite'), keypoint_dir),
    (os.path.join(keypoint_dir, 'keypoint_classifier_label.csv'), keypoint_dir),
    (os.path.join(point_history_dir, 'point_history_classifier.tflite'), point_history_dir),
    (os.path.join(point_history_dir, 'point_history_classifier_label.csv'), point_history_dir),
    (os.path.join(mp_root, 'modules'), 'mediapipe/modules'),
    ('icon.ico', '.'),
    ('icons/Ok.png', 'icons'),
    ('icons/fourfu.png', 'icons'),
    ('icons/pinch.png', 'icons'),
    ('icons/tdown.png', 'icons'),
    ('icons/threefu.png', 'icons'),
    ('icons/tup.png', 'icons'),
    ('icons/twofu.png', 'icons'),
]

# ✅ Add Vosk model ONLY if exists
if os.path.isdir(vosk_model_path):
    datas.append((vosk_model_path, vosk_model_path))

# Hidden imports often needed by TensorFlow, OpenCV, PyQt5, pystray
hiddenimports = [
    'numpy', 'cv2', 'mediapipe', 'tensorflow', 'tensorflow.lite',
    'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
    'pystray', 'PIL', 'PIL._tkinter_finder',
    'pyautogui', 'comtypes', 'pycaw',

    # ✅ ONLY correct MediaPipe binding
    'mediapipe.python._framework_bindings',

    # ✅ Minimal Vosk
    'vosk',
    'sounddevice',
    '_sounddevice',
]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=mp_binaries,  # ✅ FIX HERE
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

    # 🔥 KEEP TRUE until stable
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
    name='HandGestureRecognition',
)