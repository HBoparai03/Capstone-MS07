# Core Libraries

mediapipe
numpy
tensorflow
pyautogui
opencv-python
pycaw (for volume control)
comtypes (required by pycaw)

# CD into hand-gesture-recognition-mediapipe folder

cd .\hand-gesture-recognition-mediapipe\

# Python Version

### Create a virtual environment with the correct Python version

```bash
py -3.10 -m venv .venv
```

### Activate virtual environment

.\.venv\Scripts\Activate.ps1

# Install Dependencies

```bash
pip install protobuf==3.20.3

pip install tensorflow==2.15.0

pip install mediapipe==0.10.9

pip install opencv-python==4.8.1.78

pip install pyautogui==0.9.54

pip install numpy==1.26.4

pip install pycaw

pip install comtypes

pip install scikit-learn

pip install jupyter ipykernel

pip install pandas seaborn matplotlib

pip install PyQt5

pip install pystray

pip install pillow

```

### Run Code (default = best settings: new UI, high performance, 8 threads, 120 Hz mouse)

```bash
python app.py
```

### Use classic OpenCV window UI instead

```bash
python app.py --ui old
```

### For balanced performance/quality

```bash
python app.py --high_performance --num_threads 4 --draw_quality high --mouse_update_rate 60 --min_mouse_movement 2 --ui new
```

### For low-end systems

```bash
python app.py --no_high_performance --draw_quality low --mouse_update_rate 30 --min_mouse_movement 5
```

### Defaults (same as “Run Code” above)

- `--ui new` (PyQt5 overlay)
- `--high_performance` (use `--no_high_performance` to disable)
- `--num_threads 8` (when high performance)
- `--draw_quality medium`
- `--mouse_update_rate 120`
- `--min_mouse_movement 2`

---

# Build EXE (PyInstaller)

Install PyInstaller in the same environment:

```bash
pip install pyinstaller
```

Build the one-dir executable (faster startup and simpler packaging):

```bash
pyinstaller hand_gesture_app.spec
```

The executable is created at `dist\HandGestureRecognition\HandGestureRecognition.exe`. Run it as-is; no need to pass arguments unless you want to override defaults (e.g. run from a terminal with `HandGestureRecognition.exe --ui old`).

This build only bundles the seven overlay PNGs used by the app instead of the entire icons folder.

### Requirements for building

- Use the same Python version and dependencies as for running the app (e.g. Python 3.10, TensorFlow 2.15, MediaPipe, PyQt5, etc.).
- First run and test with `python app.py` in that environment, then run the spec above.

### Run the new PyQt5 overlay UI

```bash
python app.py --ui new
```

# Hand Assignment & Mouse Toggle (System Tray)

Gesture hand and mouse control are managed at runtime via the system tray icon:

- **Toggle Left/Right** — swaps the gesture hand; the mouse hand is always the opposite.
- **Toggle Mouse** — enables or disables cursor control on the mouse hand.

The `--gesturehand` flag can still be passed to set the _initial_ gesture hand (defaults to `right`). The `--mousehand` flag is no longer needed since it is derived automatically.
