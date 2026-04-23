# Hand Gesture Recognition with MediaPipe and TensorFlow Lite

This project is a real-time hand gesture recognition app built around:

- MediaPipe Hands for hand tracking
- TensorFlow Lite classifiers for static hand signs and point-history motion
- OpenCV and PyQt5 for display
- pyautogui and pycaw for desktop control
- Vosk and sounddevice for optional offline speech dictation

The app is launched with:

```bash
python app.py
```

The default experience is the PyQt5 overlay UI (`--ui new`).

## Current Feature Set

- Real-time hand landmark detection with MediaPipe
- Static hand-sign classification from 21 landmarks
- Point-history classification for motion labels shown in the UI
- Two UI modes:
  - `new`: transparent PyQt5 overlay with system tray controls
  - `old`: classic OpenCV window
- Desktop control gestures for tabs, media, volume, mouse movement, and click
- Push-to-talk speech dictation using an offline Vosk model
- Runtime gesture-hand switching and mouse enable/disable from the tray menu
- Latest-frame camera capture to reduce latency under load

## Gesture Map

Static hand-sign labels in the current model:

- `Open`
- `Close`
- `Pointer`
- `OK`
- `Thumbs Up`
- `Thumbs Down`
- `Two Fingers Up`
- `Three Fingers Up`
- `Pinch`
- `Four Fingers Up`

Current gesture actions:

| Gesture | Hand | Action |
| --- | --- | --- |
| `Close` | Gesture hand | Hold for push-to-talk speech dictation |
| `OK` | Gesture hand | `Ctrl+T` |
| `Four Fingers Up` | Gesture hand | `Ctrl+W` |
| `Thumbs Up` | Gesture hand | Volume up |
| `Thumbs Down` | Gesture hand | Volume down |
| `Two Fingers Up` | Gesture hand | Play/Pause |
| `Three Fingers Up` | Gesture hand | `Alt+Left` |
| `Pointer` | Mouse hand | Air mouse cursor movement |
| `Pinch` | Gesture hand | Left click after hold delay |

Point-history labels currently displayed by the motion classifier:

- `Stop`
- `Clockwise`
- `Counter Clockwise`
- `Move`

## UI Modes

### New UI (`--ui new`, default)

- Transparent full-screen PyQt5 overlay
- Live camera preview in the top-left
- Gesture label at the bottom
- Speech status and transcript area
- System tray menu for:
  - show/hide overlay
  - instructions table
  - mouse on/off
  - swapping gesture hand and mouse hand
  - transcript visibility
  - exit

Notes for the new UI:

- The initial gesture hand comes from `--gesturehand`
- The mouse hand is always the opposite hand in the tray-driven overlay mode
- `--mousehand` is still accepted by the CLI for compatibility, but the tray UI derives the mouse hand from the gesture hand

### Old UI (`--ui old`)

- Standard OpenCV debug window
- Uses the same gesture models and gesture mappings
- Accepts both `--gesturehand` and `--mousehand`
- Shows landmarks, point history, and FPS directly in the OpenCV frame

## Runtime Architecture

The current codebase uses:

- A background camera reader that always keeps the newest frame
- A worker processing loop for gesture inference in the new overlay UI
- A separate Qt render loop for display updates
- Background automation threads for mouse and keyboard actions
- A background speech worker for offline dictation

This keeps camera capture, inference, UI updates, and OS automation from blocking each other as much as possible.

## Requirements

- Windows is the intended runtime environment for the full feature set
- Python 3.10 is recommended
- A webcam
- A microphone if you want speech dictation

## Installation

### 1. Create and activate a virtual environment

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install the core runtime dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` currently includes the core CV and speech runtime packages:

- `numpy`
- `protobuf`
- `tensorflow`
- `mediapipe`
- `opencv-python`
- `PyAutoGUI`
- `vosk`
- `sounddevice`

### 3. Install the UI and Windows control packages used by the default app experience

```bash
pip install PyQt5 pystray pillow pycaw comtypes
```

These packages are needed for:

- the default overlay UI
- the system tray icon
- volume control
- the best Windows desktop-control experience

### 4. Optional notebook and training dependencies

Only needed if you want to use the included notebooks:

```bash
pip install scikit-learn jupyter ipykernel pandas seaborn matplotlib
```

## Speech Dictation Setup

Speech dictation is optional. The app will still run if the speech model is missing.

To install the offline Vosk model:

```bash
python download_vosk_model.py
```

This downloads `vosk-model-small-en-us/` next to `app.py`.

If the model is missing, the app still launches, but speech shows as unavailable in the UI.

For more detail, see [SPEECH_SETUP.md](SPEECH_SETUP.md).

## Running the App

### Default run

```bash
python app.py
```

This starts:

- `--ui new`
- high-performance mode
- model complexity `1`
- draw quality `medium`
- gesture hand `right`

### Run the classic OpenCV UI

```bash
python app.py --ui old
```

### Example: balanced quality profile

```bash
python app.py --ui new --high_performance --num_threads 1 --draw_quality high --mouse_update_rate 60 --min_mouse_movement 2 --model_complexity 1
```

### Example: lighter profile for lower-end systems

```bash
python app.py --no_high_performance --draw_quality low --mouse_update_rate 30 --min_mouse_movement 5 --model_complexity 0
```

## Important CLI Options

You can always see the full list with:

```bash
python app.py --help
```

Current runtime options:

| Option | Default | Notes |
| --- | --- | --- |
| `--ui` | `new` | `new` = PyQt5 overlay, `old` = OpenCV window |
| `--device` | `0` | Camera index |
| `--width` | `960` | Capture width |
| `--height` | `540` | Capture height |
| `--use_static_image_mode` | off | MediaPipe static-image mode |
| `--min_detection_confidence` | `0.7` | MediaPipe detection threshold |
| `--min_tracking_confidence` | `0.5` | MediaPipe tracking threshold |
| `--high_performance` | on | Use `--no_high_performance` to disable |
| `--num_threads` | `1` effective default | Threads for the TFLite classifiers |
| `--model_complexity` | `1` | `0` is lighter/faster, `1` is fuller model |
| `--draw_quality` | `medium` | `high`, `medium`, or `low` |
| `--gesturehand` | `right` | Initial gesture hand |
| `--mousehand` | `left` | Used directly by `old` UI; `new` UI derives opposite hand |
| `--mouse_sensitivity` | `1.0` | Air mouse travel scaling |
| `--mouse_smoothing` | `0.85` | Cursor smoothing factor |
| `--mouse_update_rate` | `120` | Mouse update target in Hz |
| `--min_mouse_movement` | `2` | Minimum pixel delta before updating mouse |
| `--gesture_hold_time` | `2.0` | Hold time before non-pinch gesture actions fire |
| `--gesture_cooldown` | `1.5` | Cooldown between repeated gesture activations |
| `--pinch_click_delay` | `1.5` | Hold time before pinch triggers click |

Compatibility note:

- `--enable_air_mouse` is still present in the CLI, but the current implementation already drives cursor movement from the `Pointer` gesture when mouse control is enabled

## Project Layout

```text
hand-gesture-recognition-mediapipe/
|-- app.py
|-- Capture.py
|-- Overlay.py
|-- Tray.py
|-- download_vosk_model.py
|-- requirements.txt
|-- SPEECH_SETUP.md
|-- hand_gesture_app.spec
|-- model/
|   |-- keypoint_classifier/
|   `-- point_history_classifier/
|-- icons/
|-- utils/
`-- vosk-model-small-en-us/   # created after running download_vosk_model.py
```

## Model and Notebook Files

Runtime model files:

- `model/keypoint_classifier/keypoint_classifier.tflite`
- `model/point_history_classifier/point_history_classifier.tflite`

Included notebooks for training or experimentation:

- `keypoint_classification_EN.ipynb`
- `point_history_classification.ipynb`

## Building the EXE

Install PyInstaller:

```bash
pip install pyinstaller
```

If you want the packaged app to include speech dictation, download the Vosk model before building:

```bash
python download_vosk_model.py
```

Then build:

```bash
pyinstaller hand_gesture_app.spec
```

Output:

```text
dist/HandGestureRecognition/HandGestureRecognition.exe
```

The current spec bundles:

- both TFLite model files
- both label CSV files
- MediaPipe runtime assets
- the tray/app icon and overlay PNGs
- Vosk Python package assets
- the Vosk model folder if it exists locally at build time

## Troubleshooting

### PyQt5 import errors on startup

Install the UI dependencies:

```bash
pip install PyQt5 pystray pillow
```

### Speech shows as unavailable

Check all of the following:

- `vosk-model-small-en-us/` exists next to `app.py`
- `vosk` is installed
- `sounddevice` is installed
- a microphone is available
- `pyautogui` is installed so dictated text can be typed back into the active window

### Volume gestures do nothing

Install the Windows audio-control packages:

```bash
pip install pycaw comtypes
```

### Overlay works but mouse or hotkeys do not

Make sure:

- `pyautogui` is installed
- the app has permission to control the desktop
- the active application is the one you expect to receive hotkeys or typed text

## Notes

- The app is designed to preserve the current gesture mappings and user-facing behavior
- Speech dictation is offline once the Vosk model has been downloaded
- The default UI path is `python app.py`
