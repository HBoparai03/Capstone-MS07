# Core Libraries
mediapipe
numpy
tensorflow
pyautogui
opencv-python
pycaw (for volume control)
comtypes (required by pycaw)

# Python Version
### Create a virtual environment with the correct Python version
Run: py -3.10 -m venv .venv

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
```

### Run Code
```bash
python app.py
```

# Max FPS Configuration
```bash
python app.py --high_performance --num_threads 8 --draw_quality medium --mouse_update_rate 120 --min_mouse_movement 2
```

### For Balanced Performance/Quality
```bash
python app.py --high_performance --num_threads 4 --draw_quality high --mouse_update_rate 60 --min_mouse_movement 2
```

### For Low-End Systems
```bash
python app.py --draw_quality low --mouse_update_rate 30 --min_mouse_movement 5
```
