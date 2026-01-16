# Core Libraries
mediapipe
numpy
tensorflow
pyautogui
opencv-python

# Python Version
### Create a virtual environment with the correct Python version
Run: py -3.10 -m venv .venv

### Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install Dependencies
pip install protobuf==3.20.3

pip install tensorflow==2.15.0

pip install mediapipe==0.10.9

pip install opencv-python==4.8.1.78

pip install pyautogui==0.9.54

pip install numpy==1.26.4

### Run Code
python app.py
# Max FPS Configuration
python app.py --high_performance --num_threads 8 --draw_quality medium --mouse_update_rate 60 --min_mouse_movement 3