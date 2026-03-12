# PyInstaller spec for Hand Gesture Recognition (best/default settings: new UI, high perf, 8 threads, etc.)
# Build: pyinstaller hand_gesture_app.spec

import os
import mediapipe

block_cipher = None

# Data files to bundle (models and labels)
model_root = 'model'
keypoint_dir = os.path.join(model_root, 'keypoint_classifier')
point_history_dir = os.path.join(model_root, 'point_history_classifier')

mp_root = os.path.dirname(mediapipe.__file__)

datas = [
    (os.path.join(keypoint_dir, 'keypoint_classifier.tflite'), keypoint_dir),
    (os.path.join(keypoint_dir, 'keypoint_classifier_label.csv'), keypoint_dir),
    (os.path.join(point_history_dir, 'point_history_classifier.tflite'), point_history_dir),
    (os.path.join(point_history_dir, 'point_history_classifier_label.csv'), point_history_dir),
    (os.path.join(mp_root, 'modules'), 'mediapipe/modules'),
    ('icon.ico', '.'),
    ('icons', 'icons'),
]

# Hidden imports often needed by TensorFlow, OpenCV, PyQt5, pystray
hiddenimports = [
    'numpy', 'cv2', 'mediapipe', 'tensorflow', 'tensorflow.lite',
    'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
    'pystray', 'PIL', 'PIL._tkinter_finder',
    'pyautogui', 'comtypes', 'pycaw',
    'mediapipe.python._framework_bindings',
    'mediapipe.python._framework_bindings.calculator_graph',
    'mediapipe.python._framework_bindings.packet',
]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
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
    a.datas,
    [],
    name='HandGestureRecognition',
    icon='icon.ico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
