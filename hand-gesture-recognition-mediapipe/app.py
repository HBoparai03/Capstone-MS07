#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import sys

import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import threading
import queue

try:
    import pyautogui  # For sending Ctrl+T to active window and mouse control
except Exception:
    pyautogui = None

try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    VOLUME_CONTROL_AVAILABLE = True
except Exception:
    VOLUME_CONTROL_AVAILABLE = False

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ui", 
                        help='UI version to use: "old" (OpenCV window) or "new" (PyQt5 overlay)',
                        type=str,
                        default='old',
                        choices=['old', 'new'])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    parser.add_argument("--enable_air_mouse",
                        help='Enable air mouse control when Pointer gesture is detected',
                        action='store_true')
    parser.add_argument("--mouse_sensitivity",
                        help='Mouse movement sensitivity (1.0 = full screen travel; >1 = faster, <1 = slower but may not reach edges)',
                        type=float,
                        default=1.0)
    parser.add_argument("--mouse_smoothing",
                        help='How much to smooth cursor movement (0=no smoothing, 1=very slow; default: 0.85)',
                        type=float,
                        default=0.85)
    parser.add_argument("--high_performance",
                        help='Enable high-performance mode (uses more CPU/GPU resources)',
                        action='store_true')
    parser.add_argument("--num_threads",
                        help='Number of threads for TensorFlow Lite (0 = auto, default: 4 in high-performance mode)',
                        type=int,
                        default=None)
    parser.add_argument("--mouse_update_rate",
                        help='Mouse update rate in Hz (higher = smoother but more CPU, default: 60)',
                        type=int,
                        default=60)
    parser.add_argument("--min_mouse_movement",
                        help='Minimum pixel movement before updating mouse (reduces overhead, default: 2)',
                        type=int,
                        default=2)
    parser.add_argument("--draw_quality",
                        help='Drawing quality: "high", "medium", "low" (default: high)',
                        type=str,
                        default='high',
                        choices=['high', 'medium', 'low'])

    args = parser.parse_args()

    return args


def main_new_ui(args):
    """Run the new PyQt5 overlay UI with gesture recognition."""
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    from Overlay import OverlayWindow
    from Tray import TrayIcon
    
    # Initialize MediaPipe and classifiers
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    
    # Determine number of threads for TensorFlow Lite
    if args.high_performance:
        num_threads = args.num_threads if args.num_threads is not None else 4
    else:
        num_threads = args.num_threads if args.num_threads is not None else 1
    
    keypoint_classifier = KeyPointClassifier(num_threads=num_threads)
    point_history_classifier = PointHistoryClassifier(num_threads=num_threads)
    
    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    
    with open('model/point_history_classifier/point_history_classifier_label.csv',
              encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]
    
    # Resolve label indices for hand signs
    def get_label_index(labels, name):
        try:
            return labels.index(name)
        except ValueError:
            return None
    
    open_label_index = get_label_index(keypoint_classifier_labels, 'Open')
    close_label_index = get_label_index(keypoint_classifier_labels, 'Close')
    thumbs_up_label_index = get_label_index(keypoint_classifier_labels, 'Thumbs Up')
    thumbs_down_label_index = get_label_index(keypoint_classifier_labels, 'Thumbs Down')
    two_fingers_up_label_index = get_label_index(keypoint_classifier_labels, 'Two Fingers Up')
    three_fingers_up_label_index = get_label_index(keypoint_classifier_labels, 'Three Fingers Up')
    pointer_label_index = get_label_index(keypoint_classifier_labels, 'Pointer')
    
    # Initialize volume control if available
    volume_interface = None
    if VOLUME_CONTROL_AVAILABLE:
        try:
            devices = AudioUtilities.GetSpeakers()
            volume_interface = devices.EndpointVolume.QueryInterface(IAudioEndpointVolume)
            print("Volume interface initialized OK")
        except Exception as e:
            print("Volume init error:", repr(e))
            volume_interface = None
    
    # Coordinate history
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    
    # Air mouse control variables
    mouse_sensitivity = args.mouse_sensitivity
    mouse_smoothing = max(0.0, min(0.99, args.mouse_smoothing))
    prev_mouse_pos = [None]  # Use list to allow mutation in nested function
    
    # Cache screen size
    screen_width, screen_height = None, None
    if pyautogui is not None:
        screen_width, screen_height = pyautogui.size()
    
    # Mouse movement throttling
    mouse_update_interval = 1.0 / args.mouse_update_rate
    min_mouse_movement = args.min_mouse_movement
    last_mouse_update_time = [0.0]
    
    # Debounce for hotkey actions
    last_hotkey_time = [0.0]
    hotkey_cooldown_sec = 1.5
    
    # Create the Qt application
    app = QApplication(sys.argv)
    
    # Create the overlay window
    overlay = OverlayWindow()
    
    # Store gesture state for the overlay
    current_gesture = ["Gesture: (awaiting detection...)"]
    
    def process_frame():
        """Process a frame and perform gesture recognition."""
        frame = overlay.detector.get_frame()
        if frame is None:
            return
        
        # Convert to RGB for MediaPipe
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Landmark calculation
                landmark_list = calc_landmark_list(frame, hand_landmarks)
                
                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    frame, point_history)
                
                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                hand_sign_label = keypoint_classifier_labels[hand_sign_id]
                
                is_pointer_gesture = (pointer_label_index is not None and 
                                     hand_sign_id == pointer_label_index)
                
                if is_pointer_gesture:
                    point_history.append(landmark_list[8])
                    
                    # Air mouse control
                    if pyautogui is not None:
                        cap_width = frame.shape[1]
                        cap_height = frame.shape[0]
                        index_finger_tip = landmark_list[8]
                        
                        normalized_x = index_finger_tip[0] / cap_width
                        normalized_y = index_finger_tip[1] / cap_height
                        
                        effective_x = (normalized_x - 0.5) * mouse_sensitivity + 0.5
                        effective_y = (normalized_y - 0.5) * mouse_sensitivity + 0.5
                        effective_x = max(0.0, min(1.0, effective_x))
                        effective_y = max(0.0, min(1.0, effective_y))
                        
                        screen_x = int(effective_x * screen_width)
                        screen_y = int(effective_y * screen_height)
                        
                        if prev_mouse_pos[0] is not None:
                            screen_x = int(screen_x * (1 - mouse_smoothing) + prev_mouse_pos[0][0] * mouse_smoothing)
                            screen_y = int(screen_y * (1 - mouse_smoothing) + prev_mouse_pos[0][1] * mouse_smoothing)
                        
                        screen_x = max(0, min(screen_width - 1, screen_x))
                        screen_y = max(0, min(screen_height - 1, screen_y))
                        
                        now = time.time()
                        should_update = False
                        
                        if prev_mouse_pos[0] is None:
                            should_update = True
                        else:
                            time_since_update = now - last_mouse_update_time[0]
                            if time_since_update >= mouse_update_interval:
                                dx = abs(screen_x - prev_mouse_pos[0][0])
                                dy = abs(screen_y - prev_mouse_pos[0][1])
                                if dx >= min_mouse_movement or dy >= min_mouse_movement:
                                    should_update = True
                        
                        if should_update:
                            try:
                                pyautogui.moveTo(screen_x, screen_y, duration=0.0)
                                prev_mouse_pos[0] = (screen_x, screen_y)
                                last_mouse_update_time[0] = now
                            except Exception:
                                pass
                else:
                    point_history.append([0, 0])
                    prev_mouse_pos[0] = None
                
                # Hotkey actions
                if pyautogui is not None:
                    now = time.time()
                    if now - last_hotkey_time[0] > hotkey_cooldown_sec:
                        if open_label_index is not None and hand_sign_id == open_label_index:
                            try:
                                pyautogui.hotkey('ctrl', 't')
                                last_hotkey_time[0] = now
                            except Exception:
                                pass
                        elif close_label_index is not None and hand_sign_id == close_label_index:
                            try:
                                pyautogui.hotkey('ctrl', 'w')
                                last_hotkey_time[0] = now
                            except Exception:
                                pass
                
                # Volume control
                if volume_interface is not None:
                    now = time.time()
                    if now - last_hotkey_time[0] > hotkey_cooldown_sec:
                        if thumbs_up_label_index is not None and hand_sign_id == thumbs_up_label_index:
                            try:
                                current_volume = volume_interface.GetMasterVolumeLevelScalar()
                                new_volume = min(1.0, current_volume + 0.05)
                                volume_interface.SetMasterVolumeLevelScalar(new_volume, None)
                                last_hotkey_time[0] = now
                            except Exception:
                                pass
                        elif thumbs_down_label_index is not None and hand_sign_id == thumbs_down_label_index:
                            try:
                                current_volume = volume_interface.GetMasterVolumeLevelScalar()
                                new_volume = max(0.0, current_volume - 0.05)
                                volume_interface.SetMasterVolumeLevelScalar(new_volume, None)
                                last_hotkey_time[0] = now
                            except Exception:
                                pass
                
                # Play/Pause and Go back
                if pyautogui is not None:
                    now = time.time()
                    if now - last_hotkey_time[0] > hotkey_cooldown_sec:
                        if two_fingers_up_label_index is not None and hand_sign_id == two_fingers_up_label_index:
                            try:
                                pyautogui.press('playpause')
                                last_hotkey_time[0] = now
                            except Exception:
                                pass
                        elif three_fingers_up_label_index is not None and hand_sign_id == three_fingers_up_label_index:
                            try:
                                pyautogui.hotkey('alt', 'left')
                                last_hotkey_time[0] = now
                            except Exception:
                                pass
                
                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()
                
                # Update gesture label
                handedness_label = handedness.classification[0].label
                finger_gesture_label = point_history_classifier_labels[most_common_fg_id[0][0]]
                current_gesture[0] = f"{handedness_label}: {hand_sign_label}"
                if finger_gesture_label:
                    current_gesture[0] += f" | {finger_gesture_label}"
        else:
            point_history.append([0, 0])
            current_gesture[0] = "Gesture: (no hand detected)"
    
    # Override the detector's get_gesture_label to return actual gesture
    original_get_gesture_label = overlay.detector.get_gesture_label
    def get_gesture_label_override():
        return current_gesture[0]
    overlay.detector.get_gesture_label = get_gesture_label_override
    
    # Create a timer to run gesture processing alongside the frame updates
    gesture_timer = QTimer()
    gesture_timer.timeout.connect(process_frame)
    gesture_timer.start(30)  # ~33 FPS
    
    # Show the overlay
    overlay.show()
    
    # Create and run the tray icon
    tray = TrayIcon(overlay)
    tray.run()
    
    # Run the Qt event loop
    sys.exit(app.exec_())


def main_old_ui(args):
    """Run the original OpenCV window UI with gesture recognition."""
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Determine number of threads for TensorFlow Lite
    if args.high_performance:
        num_threads = args.num_threads if args.num_threads is not None else 4
    else:
        num_threads = args.num_threads if args.num_threads is not None else 1
    
    keypoint_classifier = KeyPointClassifier(num_threads=num_threads)

    point_history_classifier = PointHistoryClassifier(num_threads=num_threads)

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # Resolve label indices for hand signs (if present)
    try:
        open_label_index = keypoint_classifier_labels.index('Open')
    except ValueError:
        open_label_index = None
    
    try:
        close_label_index = keypoint_classifier_labels.index('Close')
    except ValueError:
        close_label_index = None
    
    try:
        thumbs_up_label_index = keypoint_classifier_labels.index('Thumbs Up')
    except ValueError:
        thumbs_up_label_index = None
    
    try:
        thumbs_down_label_index = keypoint_classifier_labels.index('Thumbs Down')
    except ValueError:
        thumbs_down_label_index = None

    try:
        two_fingers_up_label_index = keypoint_classifier_labels.index('Two Fingers Up')
    except ValueError:
        two_fingers_up_label_index = None

    try:
        three_fingers_up_label_index = keypoint_classifier_labels.index('Three Fingers Up')
    except ValueError:
        three_fingers_up_label_index = None

    # Initialize volume control if available
    volume_interface = None
    if VOLUME_CONTROL_AVAILABLE:
        try:

            devices = AudioUtilities.GetSpeakers()
            volume_interface = devices.EndpointVolume.QueryInterface(IAudioEndpointVolume)

            print("Volume interface initialized OK")
        except Exception as e:
            print("Volume init error:", repr(e))
            volume_interface = None

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    # Debounce for hotkey actions
    last_hotkey_time = 0.0
    hotkey_cooldown_sec = 1.5

    # Air mouse control variables
    enable_air_mouse = True  # Always enabled
    mouse_sensitivity = args.mouse_sensitivity
    mouse_smoothing = max(0.0, min(0.99, args.mouse_smoothing))  # Clamp to avoid div/edge issues
    prev_mouse_pos = None
    pointer_label_index = None
    try:
        pointer_label_index = keypoint_classifier_labels.index('Pointer')
    except ValueError:
        pointer_label_index = None
    
    # Performance optimization: Cache screen size
    screen_width, screen_height = None, None
    if pyautogui is not None:
        screen_width, screen_height = pyautogui.size()
    
    # Mouse movement throttling
    mouse_update_rate = args.mouse_update_rate
    min_mouse_movement = args.min_mouse_movement
    mouse_update_interval = 1.0 / mouse_update_rate
    last_mouse_update_time = 0.0
    
    # Thread-safe mouse movement queue for high-performance mode
    mouse_queue = queue.Queue(maxsize=1) if args.high_performance and pyautogui else None
    mouse_thread_running = False
    
    def mouse_worker():
        """Background thread for mouse movement to avoid blocking main loop"""
        while mouse_thread_running:
            try:
                x, y = mouse_queue.get(timeout=0.1)
                pyautogui.moveTo(x, y, duration=0.0)
                mouse_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass
    
    # Start mouse worker thread if in high-performance mode
    if mouse_queue is not None:
        mouse_thread_running = True
        mouse_thread = threading.Thread(target=mouse_worker, daemon=True)
        mouse_thread.start()
    
    # Drawing quality settings
    draw_quality = args.draw_quality
    draw_landmarks_enabled = draw_quality != 'low'
    draw_point_history_enabled = draw_quality != 'low'
    draw_info_enabled = True  # Always show FPS

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        # Performance: Use .copy() instead of deepcopy (much faster for 2D arrays)
        debug_image = image.copy()

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                is_pointer_gesture = (pointer_label_index is not None and 
                                     hand_sign_id == pointer_label_index)
                
                if is_pointer_gesture:  # Pointer gesture
                    point_history.append(landmark_list[8])
                    
                    # Air mouse control (optimized for performance)
                    if enable_air_mouse and pyautogui is not None and mode == 0:
                        index_finger_tip = landmark_list[8]  # Index finger tip coordinates
                        
                        # Map camera coordinates to screen coordinates (full camera range -> full screen)
                        # Normalize to 0-1 range based on camera frame
                        normalized_x = index_finger_tip[0] / cap_width
                        normalized_y = index_finger_tip[1] / cap_height
                        
                        # Convert to screen coordinates with sensitivity as gain (not range shrink)
                        # Sensitivity: 1.0 = 1:1 mapping; >1.0 = more sensitive (small hand move = big cursor move)
                        # Full camera range (0-1) always maps to full screen; clamp keeps cursor on screen
                        effective_x = (normalized_x - 0.5) * mouse_sensitivity + 0.5
                        effective_y = (normalized_y - 0.5) * mouse_sensitivity + 0.5
                        effective_x = max(0.0, min(1.0, effective_x))
                        effective_y = max(0.0, min(1.0, effective_y))
                        
                        screen_x = int(effective_x * screen_width)
                        screen_y = int(effective_y * screen_height)
                        
                        # Smooth movement using previous position (slower = higher smoothing)
                        if prev_mouse_pos is not None:
                            screen_x = int(screen_x * (1 - mouse_smoothing) + prev_mouse_pos[0] * mouse_smoothing)
                            screen_y = int(screen_y * (1 - mouse_smoothing) + prev_mouse_pos[1] * mouse_smoothing)
                        
                        # Clamp to screen bounds
                        screen_x = max(0, min(screen_width - 1, screen_x))
                        screen_y = max(0, min(screen_height - 1, screen_y))
                        
                        # Performance optimization: Throttle mouse updates
                        now = time.time()
                        should_update = False
                        
                        if prev_mouse_pos is None:
                            # First update, always move
                            should_update = True
                        else:
                            # Check if enough time has passed (rate limiting)
                            time_since_update = now - last_mouse_update_time
                            if time_since_update >= mouse_update_interval:
                                # Check if movement is significant enough
                                dx = abs(screen_x - prev_mouse_pos[0])
                                dy = abs(screen_y - prev_mouse_pos[1])
                                if dx >= min_mouse_movement or dy >= min_mouse_movement:
                                    should_update = True
                        
                        if should_update:
                            try:
                                if mouse_queue is not None:
                                    # High-performance mode: non-blocking queue
                                    try:
                                        mouse_queue.put_nowait((screen_x, screen_y))
                                    except queue.Full:
                                        # Skip if queue is full (prevents lag buildup)
                                        pass
                                else:
                                    # Normal mode: direct call
                                    pyautogui.moveTo(screen_x, screen_y, duration=0.0)
                                prev_mouse_pos = (screen_x, screen_y)
                                last_mouse_update_time = now
                            except Exception:
                                pass
                else:
                    point_history.append([0, 0])
                    # Reset previous mouse position when not in pointer mode
                    prev_mouse_pos = None

                # Hotkey actions: Gesture -> Keyboard shortcut
                # - Only when not in logging modes (mode == 0)
                # - Debounced to avoid repeated triggers
                # - Requires pyautogui to be available
                if mode == 0 and pyautogui is not None:
                    now = time.time()
                    if now - last_hotkey_time > hotkey_cooldown_sec:
                        # Open hand -> Ctrl+T (new tab)
                        if (open_label_index is not None and
                                hand_sign_id == open_label_index):
                            try:
                                pyautogui.hotkey('ctrl', 't')
                                last_hotkey_time = now
                            except Exception:
                                pass
                        # Close fist -> Ctrl+W (close tab)
                        elif (close_label_index is not None and
                              hand_sign_id == close_label_index):
                            try:
                                pyautogui.hotkey('ctrl', 'w')
                                last_hotkey_time = now
                            except Exception:
                                pass
                
                # Volume control: Gesture -> Volume adjustment
                # - Only when not in logging modes (mode == 0)
                # - Debounced to avoid repeated triggers
                # - Requires volume control to be available
                if mode == 0 and volume_interface is not None:
                    now = time.time()
                    if now - last_hotkey_time > hotkey_cooldown_sec:
                        # Thumbs Up -> Increase volume
                        if (thumbs_up_label_index is not None and
                                hand_sign_id == thumbs_up_label_index):
                            try:
                                current_volume = volume_interface.GetMasterVolumeLevelScalar()
                                # Increase volume by 5% (0.05)
                                new_volume = min(1.0, current_volume + 0.05)
                                volume_interface.SetMasterVolumeLevelScalar(new_volume, None)
                                last_hotkey_time = now
                            except Exception:
                                pass
                        # Thumbs Down -> Decrease volume
                        elif (thumbs_down_label_index is not None and
                              hand_sign_id == thumbs_down_label_index):
                            try:
                                current_volume = volume_interface.GetMasterVolumeLevelScalar()
                                # Decrease volume by 5% (0.05)
                                new_volume = max(0.0, current_volume - 0.05)
                                volume_interface.SetMasterVolumeLevelScalar(new_volume, None)
                                last_hotkey_time = now
                            except Exception:
                                pass
                
                # Two Fingers Up -> Play/Pause toggle
                # Three Fingers Up -> Go back (Alt+Left)
                if mode == 0 and pyautogui is not None:
                    now = time.time()
                    if now - last_hotkey_time > hotkey_cooldown_sec:
                        if two_fingers_up_label_index is not None and hand_sign_id == two_fingers_up_label_index:
                            try:
                                pyautogui.press('playpause')
                                last_hotkey_time = now
                            except Exception:
                                pass
                        elif three_fingers_up_label_index is not None and hand_sign_id == three_fingers_up_label_index:
                            try:
                                pyautogui.hotkey('alt', 'left')
                                last_hotkey_time = now
                            except Exception:
                                pass

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part (optimized based on quality setting)
                if use_brect:
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                if draw_landmarks_enabled:
                    debug_image = draw_landmarks(debug_image, landmark_list, draw_quality)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        if draw_point_history_enabled:
            debug_image = draw_point_history(debug_image, point_history)
        if draw_info_enabled:
            debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    # Cleanup
    if mouse_queue is not None:
        mouse_thread_running = False
        mouse_thread.join(timeout=1.0)
    
    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    # Performance: Pre-allocate array instead of appending
    landmark_array = np.zeros((21, 2), dtype=np.int32)

    for idx, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array[idx] = [landmark_x, landmark_y]

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    # Performance: Use NumPy array instead of list operations
    temp_landmark_list = np.array(landmark_list, dtype=np.float32)

    # Convert to relative coordinates
    base_x, base_y = temp_landmark_list[0, 0], temp_landmark_list[0, 1]
    temp_landmark_list[:, 0] -= base_x
    temp_landmark_list[:, 1] -= base_y

    # Convert to a one-dimensional list
    temp_landmark_list = temp_landmark_list.flatten()

    # Normalization
    max_value = np.max(np.abs(temp_landmark_list))
    if max_value > 0:
        temp_landmark_list = temp_landmark_list / max_value
    else:
        temp_landmark_list = temp_landmark_list * 0.0

    return temp_landmark_list.tolist()


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    # Performance: Use NumPy array instead of list operations
    if len(point_history) == 0:
        return []
    
    temp_point_history = np.array(point_history, dtype=np.float32)

    # Convert to relative coordinates
    base_x, base_y = temp_point_history[0, 0], temp_point_history[0, 1]
    temp_point_history[:, 0] = (temp_point_history[:, 0] - base_x) / image_width
    temp_point_history[:, 1] = (temp_point_history[:, 1] - base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = temp_point_history.flatten()

    return temp_point_history.tolist()


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 20): #To change number of classes for keypoint classifier
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point, quality='high'):
    """Draw hand landmarks with optional quality reduction for performance"""
    if len(landmark_point) == 0:
        return image
    
    # Performance: Skip some drawing operations in lower quality modes
    draw_lines = quality != 'low'
    draw_all_points = quality == 'high'
    
    if draw_lines:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points (optimized: only draw important points in medium/low quality)
    if draw_all_points:
        # High quality: Draw all 21 points
        important_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        sizes = [5, 5, 5, 5, 8, 5, 5, 5, 8, 5, 5, 5, 8, 5, 5, 5, 8, 5, 5, 5, 8]
    else:
        # Medium/Low quality: Only draw fingertips and wrist
        important_indices = [0, 4, 8, 12, 16, 20]  # Wrist + all fingertips
        sizes = [5, 8, 8, 8, 8, 8]
    
    for idx, landmark_idx in enumerate(important_indices):
        if landmark_idx < len(landmark_point):
            landmark = landmark_point[landmark_idx]
            radius = sizes[idx]
            cv.circle(image, (landmark[0], landmark[1]), radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


def main():
    """Main entry point - dispatches to old or new UI based on --ui argument."""
    args = get_args()
    
    if args.ui == 'new':
        print("Starting new PyQt5 overlay UI...")
        main_new_ui(args)
    else:
        print("Starting classic OpenCV window UI...")
        main_old_ui(args)


if __name__ == '__main__':
    main()
