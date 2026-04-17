#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import json
import os
import sys
import ctypes
from collections import Counter
from collections import deque

if hasattr(sys, "_MEIPASS"):
    os.environ["MEDIAPIPE_RESOURCE_PATH"] = sys._MEIPASS


def _candidate_base_paths():
    candidates = []

    if hasattr(sys, "_MEIPASS"):
        candidates.append(sys._MEIPASS)

    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        candidates.append(exe_dir)
        candidates.append(os.path.join(exe_dir, "_internal"))

    candidates.append(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.abspath("."))

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(candidate)

    return unique_candidates


def resource_path(relative_path):
    for base_path in _candidate_base_paths():
        candidate = os.path.join(base_path, relative_path)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(_candidate_base_paths()[0], relative_path)

def get_base_path():
    """Return base path for resources (works when run as script or as PyInstaller exe)."""
    return os.path.dirname(resource_path("app.py")) if os.path.exists(resource_path("app.py")) else _candidate_base_paths()[0]

import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import threading
import queue

try:
    import pyautogui  # For sending Ctrl+T to active window and mouse control
    PYAUTOGUI_IMPORT_ERROR = None
except Exception as exc:
    pyautogui = None
    PYAUTOGUI_IMPORT_ERROR = exc

try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    VOLUME_CONTROL_AVAILABLE = True
except Exception:
    VOLUME_CONTROL_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_IMPORT_ERROR = None
except Exception as exc:
    sd = None
    SOUNDDEVICE_IMPORT_ERROR = exc

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_IMPORT_ERROR = None
except Exception as exc:
    VoskModel = None
    KaldiRecognizer = None
    VOSK_IMPORT_ERROR = exc

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
from Capture import GestureDetector, open_camera


def _configure_pyautogui():
    if pyautogui is None:
        return
    # Remove the library's built-in per-call sleeps so cursor updates can keep up.
    pyautogui.PAUSE = 0
    if hasattr(pyautogui, "MINIMUM_DURATION"):
        pyautogui.MINIMUM_DURATION = 0
    if hasattr(pyautogui, "MINIMUM_SLEEP"):
        pyautogui.MINIMUM_SLEEP = 0


def _default_classifier_threads():
    # These gesture classifiers are tiny; extra interpreter threads add contention.
    return 1


AIR_MOUSE_HORIZONTAL_MARGIN = 0.08
AIR_MOUSE_TOP_MARGIN = 0.08
AIR_MOUSE_BOTTOM_MARGIN = 0.22
AIR_MOUSE_MANUAL_OVERRIDE_DISTANCE_PX = 28
AIR_MOUSE_REENGAGE_DISTANCE_PX = 18


class _CursorPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _normalize_air_mouse_coordinate(value, lower_margin, upper_margin):
    usable_span = 1.0 - lower_margin - upper_margin
    if usable_span <= 0:
        return _clamp(value, 0.0, 1.0)
    return _clamp((value - lower_margin) / usable_span, 0.0, 1.0)


def _map_air_mouse_target(index_finger_tip, frame_width, frame_height, screen_width, screen_height, sensitivity):
    raw_x = index_finger_tip[0] / float(frame_width)
    raw_y = index_finger_tip[1] / float(frame_height)

    # Keep cursor reach away from the weakest camera edges so lower-frame tracking is more reliable.
    normalized_x = _normalize_air_mouse_coordinate(
        raw_x,
        AIR_MOUSE_HORIZONTAL_MARGIN,
        AIR_MOUSE_HORIZONTAL_MARGIN,
    )
    normalized_y = _normalize_air_mouse_coordinate(
        raw_y,
        AIR_MOUSE_TOP_MARGIN,
        AIR_MOUSE_BOTTOM_MARGIN,
    )

    effective_x = _clamp((normalized_x - 0.5) * sensitivity + 0.5, 0.0, 1.0)
    effective_y = _clamp((normalized_y - 0.5) * sensitivity + 0.5, 0.0, 1.0)

    screen_x = int(round(effective_x * max(0, screen_width - 1)))
    screen_y = int(round(effective_y * max(0, screen_height - 1)))
    return screen_x, screen_y, normalized_x, normalized_y


def _reset_air_mouse_state(state, controller=None):
    state["prev_pos"] = None
    state["last_update_time"] = 0.0
    state["manual_override"] = False
    state["override_anchor"] = None
    if controller is not None:
        controller.clear_cursor_target()


def _cursor_was_moved_manually(controller, threshold_px):
    current_cursor = controller.get_cursor_pos()
    last_programmatic_cursor = controller.get_last_programmatic_cursor_pos()
    if current_cursor is None or last_programmatic_cursor is None:
        return False, current_cursor

    dx = abs(current_cursor[0] - last_programmatic_cursor[0])
    dy = abs(current_cursor[1] - last_programmatic_cursor[1])
    return dx >= threshold_px or dy >= threshold_px, current_cursor


def _update_air_mouse_target(controller, landmark_list, frame_width, frame_height,
                             screen_width, screen_height, sensitivity, smoothing,
                             update_interval, min_movement, state):
    screen_x, screen_y, _, _ = _map_air_mouse_target(
        landmark_list[8],
        frame_width,
        frame_height,
        screen_width,
        screen_height,
        sensitivity,
    )

    if state["manual_override"]:
        override_anchor = state["override_anchor"]
        if override_anchor is None:
            state["override_anchor"] = (screen_x, screen_y)
            state["prev_pos"] = controller.get_cursor_pos()
            return

        finger_delta = max(
            abs(screen_x - override_anchor[0]),
            abs(screen_y - override_anchor[1]),
        )
        if finger_delta < AIR_MOUSE_REENGAGE_DISTANCE_PX:
            state["prev_pos"] = controller.get_cursor_pos()
            return

        state["manual_override"] = False
        state["override_anchor"] = None
        current_cursor = controller.get_cursor_pos()
        state["prev_pos"] = current_cursor
        controller.adopt_cursor_reference(current_cursor)

    manual_override, current_cursor = _cursor_was_moved_manually(
        controller,
        AIR_MOUSE_MANUAL_OVERRIDE_DISTANCE_PX,
    )
    if manual_override:
        state["manual_override"] = True
        state["override_anchor"] = (screen_x, screen_y)
        state["prev_pos"] = current_cursor
        controller.clear_cursor_target()
        return

    prev_pos = state["prev_pos"]
    if prev_pos is not None:
        screen_x = int(screen_x * (1 - smoothing) + prev_pos[0] * smoothing)
        screen_y = int(screen_y * (1 - smoothing) + prev_pos[1] * smoothing)

    screen_x = max(0, min(screen_width - 1, screen_x))
    screen_y = max(0, min(screen_height - 1, screen_y))

    now = time.perf_counter()
    should_update = False

    if prev_pos is None:
        should_update = True
    else:
        time_since_update = now - state["last_update_time"]
        if time_since_update >= update_interval:
            dx = abs(screen_x - prev_pos[0])
            dy = abs(screen_y - prev_pos[1])
            if dx >= min_movement or dy >= min_movement:
                should_update = True

    if should_update and controller.move_cursor(screen_x, screen_y):
        state["prev_pos"] = (screen_x, screen_y)
        state["last_update_time"] = now


_configure_pyautogui()


class AutomationController:
    """Dispatch OS input without blocking gesture or speech processing threads."""

    def __init__(self):
        self._available = pyautogui is not None
        self._stop_event = threading.Event()
        self._action_queue = queue.Queue()
        self._action_thread = None
        self._mouse_event = threading.Event()
        self._mouse_thread = None
        self._mouse_lock = threading.Lock()
        self._latest_mouse_pos = None
        self._cursor_lock = threading.Lock()
        self._last_programmatic_cursor_pos = None
        self._user32 = ctypes.windll.user32 if os.name == "nt" else None
        self._screen_size = self._resolve_screen_size()

    def start(self):
        if self._action_thread is not None and self._action_thread.is_alive():
            return
        self._action_thread = threading.Thread(target=self._action_worker, daemon=True)
        self._action_thread.start()
        self._mouse_thread = threading.Thread(target=self._mouse_worker, daemon=True)
        self._mouse_thread.start()

    def stop(self):
        self._stop_event.set()
        self._mouse_event.set()
        try:
            self._action_queue.put_nowait(None)
        except Exception:
            pass
        if self._action_thread is not None and self._action_thread.is_alive():
            self._action_thread.join(timeout=1.5)
        if self._mouse_thread is not None and self._mouse_thread.is_alive():
            self._mouse_thread.join(timeout=1.5)

    def is_available(self):
        return self._available

    def can_type_text(self):
        return self._available

    def get_screen_size(self):
        return self._screen_size

    def get_cursor_pos(self):
        if self._user32 is not None:
            try:
                point = _CursorPoint()
                if self._user32.GetCursorPos(ctypes.byref(point)):
                    return int(point.x), int(point.y)
            except Exception:
                pass
        if pyautogui is None:
            return None
        try:
            pos = pyautogui.position()
            return int(pos.x), int(pos.y)
        except Exception:
            return None

    def get_last_programmatic_cursor_pos(self):
        with self._cursor_lock:
            return self._last_programmatic_cursor_pos

    def move_cursor(self, x, y):
        if self._screen_size is None:
            return False
        with self._mouse_lock:
            self._latest_mouse_pos = (int(x), int(y))
        self._mouse_event.set()
        return True

    def clear_cursor_target(self):
        with self._mouse_lock:
            self._latest_mouse_pos = None

    def adopt_cursor_reference(self, pos=None):
        if pos is None:
            pos = self.get_cursor_pos()
        if pos is None:
            return
        with self._cursor_lock:
            self._last_programmatic_cursor_pos = (int(pos[0]), int(pos[1]))

    def click(self):
        return self._enqueue(("click",))

    def hotkey(self, *keys):
        return self._enqueue(("hotkey", keys))

    def press(self, key):
        return self._enqueue(("press", key))

    def write_text(self, text):
        if not text:
            return False
        return self._enqueue(("write_text", text))

    def _enqueue(self, action):
        if not self._available:
            return False
        try:
            self._action_queue.put_nowait(action)
            return True
        except Exception:
            return False

    def _resolve_screen_size(self):
        if self._user32 is not None:
            try:
                width = int(self._user32.GetSystemMetrics(0))
                height = int(self._user32.GetSystemMetrics(1))
                if width > 0 and height > 0:
                    return width, height
            except Exception:
                pass
        if pyautogui is None:
            return None
        try:
            return pyautogui.size()
        except Exception:
            return None

    def _move_cursor_now(self, x, y):
        if self._user32 is not None:
            try:
                self._user32.SetCursorPos(int(x), int(y))
                with self._cursor_lock:
                    self._last_programmatic_cursor_pos = (int(x), int(y))
                return
            except Exception:
                pass
        if pyautogui is not None:
            pyautogui.moveTo(int(x), int(y), duration=0.0)
            with self._cursor_lock:
                self._last_programmatic_cursor_pos = (int(x), int(y))

    def _mouse_worker(self):
        while not self._stop_event.is_set():
            self._mouse_event.wait(0.1)
            self._mouse_event.clear()
            if self._stop_event.is_set():
                break
            with self._mouse_lock:
                mouse_pos = self._latest_mouse_pos
            if mouse_pos is None:
                continue
            try:
                self._move_cursor_now(*mouse_pos)
            except Exception:
                pass

    def _action_worker(self):
        while not self._stop_event.is_set():
            try:
                action = self._action_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if action is None:
                break
            try:
                kind = action[0]
                if kind == "click":
                    pyautogui.click()
                elif kind == "hotkey":
                    pyautogui.hotkey(*action[1])
                elif kind == "press":
                    pyautogui.press(action[1])
                elif kind == "write_text":
                    pyautogui.write(action[1], interval=0.0)
            except Exception:
                pass
            finally:
                self._action_queue.task_done()


class SpeechDictationController:
    """Persistent background speech-to-text worker with push-to-talk control."""

    def __init__(self, input_controller=None, sample_rate=16000, language="en"):
        self.sample_rate = sample_rate
        self.language = language
        self._input_controller = input_controller
        self._model_path = resource_path("vosk-model-small-en-us")
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread = None
        self._model = None
        self._model_loading = False
        self._speech_enabled = False
        self._show_transcript = True
        self._last_transcript = ""
        self._last_typed_text = ""
        self._last_error = ""
        self._input_device_index = None
        self._input_device_name = "Unavailable"
        self._available = (
            sd is not None and
            VoskModel is not None and
            input_controller is not None and
            input_controller.can_type_text()
        )
        if not self._available:
            missing_parts = []
            if sd is None:
                message = "sounddevice import failed"
                if SOUNDDEVICE_IMPORT_ERROR is not None:
                    message = f"{message}: {SOUNDDEVICE_IMPORT_ERROR}"
                missing_parts.append(message)
            if VoskModel is None:
                message = "vosk import failed"
                if VOSK_IMPORT_ERROR is not None:
                    message = f"{message}: {VOSK_IMPORT_ERROR}"
                missing_parts.append(message)
            if input_controller is None or not input_controller.can_type_text():
                message = "input automation unavailable"
                if PYAUTOGUI_IMPORT_ERROR is not None:
                    message = f"{message}: {PYAUTOGUI_IMPORT_ERROR}"
                missing_parts.append(message)
            self._last_error = "; ".join(missing_parts)
        elif not os.path.isdir(self._model_path):
            self._available = False
            self._last_error = f"Speech model not found at {self._model_path}"
        else:
            self._input_device_index, self._input_device_name = self._resolve_input_device()
            if self._input_device_index is None:
                self._available = False
                self._last_error = "No input microphone found"

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._thread.start()
        self._wake_event.set()

    def stop(self):
        self._stop_event.set()
        self._wake_event.set()
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def is_available(self):
        with self._lock:
            return self._available

    def set_enabled(self, enabled):
        enabled = bool(enabled)
        with self._lock:
            if not self._available:
                return False
            changed = self._speech_enabled != enabled
            if not changed:
                return False
            self._speech_enabled = enabled
            if enabled:
                self._last_typed_text = ""
        if not enabled and sd is not None:
            try:
                sd.stop()
            except Exception:
                pass
        self._wake_event.set()
        return changed

    def toggle_enabled(self):
        with self._lock:
            enabled = not self._speech_enabled
        return self.set_enabled(enabled)

    def set_show_transcript(self, show_transcript):
        with self._lock:
            self._show_transcript = bool(show_transcript)

    def toggle_transcript(self):
        with self._lock:
            self._show_transcript = not self._show_transcript
            return self._show_transcript

    def get_snapshot(self):
        with self._lock:
            available = self._available
            enabled = self._speech_enabled
            show_transcript = self._show_transcript
            transcript = self._last_transcript
            last_error = self._last_error
            model_loading = self._model_loading
            model_loaded = self._model is not None
            input_device_name = self._input_device_name
        if not available:
            status = "Speech: Unavailable"
        elif model_loading:
            status = "Speech: Loading..."
        elif enabled:
            status = "Speech: Listening"
        elif model_loaded:
            status = "Speech: Ready"
        else:
            status = "Speech: Starting..."
        if last_error:
            status = f"{status} ({last_error})"
        return {
            "available": available,
            "enabled": enabled,
            "show_transcript": show_transcript,
            "transcript": transcript,
            "status": status,
            "input_device_name": input_device_name,
        }

    def _set_runtime_error(self, exc):
        message = exc.__class__.__name__
        details = str(exc).strip()
        if details:
            message = f"{message}: {details}"
        with self._lock:
            self._last_error = message
        print(f"[speech] {message}", file=sys.stderr)

    def _resolve_input_device(self):
        try:
            default_device = sd.default.device
            input_index = None
            if isinstance(default_device, (list, tuple)) and len(default_device) >= 1:
                input_index = default_device[0]
            elif isinstance(default_device, int):
                input_index = default_device

            if input_index is not None and input_index >= 0:
                device_info = sd.query_devices(input_index, "input")
                return input_index, device_info["name"]

            for index, device in enumerate(sd.query_devices()):
                if device.get("max_input_channels", 0) > 0:
                    return index, device["name"]
        except Exception as exc:
            self._set_runtime_error(exc)
        return None, "Unavailable"

    def _load_model(self):
        with self._lock:
            if self._model is not None:
                return self._model
            self._model_loading = True
            self._last_error = ""
        try:
            model = VoskModel(self._model_path)
        except Exception as exc:
            with self._lock:
                self._available = False
                self._model_loading = False
                self._speech_enabled = False
                self._last_error = f"{exc.__class__.__name__}: {exc}"
            print(f"[speech] {exc.__class__.__name__}: {exc}", file=sys.stderr)
            return None
        with self._lock:
            self._model = model
            self._model_loading = False
        return model

    def _speech_worker(self):
        while not self._stop_event.is_set():
            with self._lock:
                available = self._available
                enabled = self._speech_enabled
                model = self._model

            if not available:
                self._wake_event.wait(0.25)
                self._wake_event.clear()
                continue

            if model is None:
                model = self._load_model()
                if model is None:
                    self._wake_event.wait(0.25)
                    self._wake_event.clear()
                    continue

            if not enabled:
                self._wake_event.wait(0.1)
                self._wake_event.clear()
                continue

            try:
                rec = KaldiRecognizer(model, self.sample_rate)
                with sd.RawInputStream(
                    samplerate=self.sample_rate,
                    blocksize=4000,
                    dtype="int16",
                    channels=1,
                    device=self._input_device_index,
                ) as stream:
                    while not self._stop_event.is_set():
                        with self._lock:
                            enabled = self._speech_enabled
                        if not enabled:
                            break
                        data, _ = stream.read(4000)
                        if rec.AcceptWaveform(bytes(data)):
                            result = json.loads(rec.Result())
                            text = result.get("text", "").strip()
                            if not text:
                                continue
                            normalized = " ".join(text.split())
                            with self._lock:
                                self._last_transcript = normalized
                                already_typed = normalized == self._last_typed_text
                            if already_typed:
                                continue
                            if not self._input_controller.write_text(normalized + " "):
                                self._set_runtime_error(RuntimeError("Unable to queue dictated text"))
                                continue
                            with self._lock:
                                self._last_typed_text = normalized
            except Exception as exc:
                self._set_runtime_error(exc)
                self._wake_event.wait(0.25)
                self._wake_event.clear()


def update_push_to_talk(close_hand_detected, speech_controller):
    """Enable speech only while the close-hand push-to-talk gesture is actively held."""
    if speech_controller is None or not speech_controller.is_available():
        return
    speech_controller.set_enabled(close_hand_detected)


def _get_label_index(labels, name):
    try:
        return labels.index(name)
    except ValueError:
        return None


def _load_classifier_labels(base_path):
    with open(os.path.join(base_path, 'model', 'keypoint_classifier', 'keypoint_classifier_label.csv'),
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    with open(os.path.join(base_path, 'model', 'point_history_classifier', 'point_history_classifier_label.csv'),
              encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    return keypoint_classifier_labels, point_history_classifier_labels


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ui",
                        help='UI version to use: "old" (OpenCV window) or "new" (PyQt5 overlay)',
                        type=str,
                        default='new',
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
                        type=float,
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
                        help='Enable high-performance mode (higher capture/display rate and lower camera latency)',
                        action='store_true',
                        default=True)
    parser.add_argument("--no_high_performance",
                        help='Disable high-performance mode',
                        action='store_false',
                        dest='high_performance')
    parser.add_argument("--num_threads",
                        help='Number of threads for the TFLite gesture classifiers (default: 1)',
                        type=int,
                        default=None)
    parser.add_argument("--model_complexity",
                        help='MediaPipe Hands model complexity: 0=lite/faster, 1=full (default: 1)',
                        type=int,
                        default=1,
                        choices=[0, 1])
    parser.add_argument("--mouse_update_rate",
                        help='Mouse update rate in Hz (higher = smoother but more CPU, default: 120)',
                        type=int,
                        default=120)
    parser.add_argument("--min_mouse_movement",
                        help='Minimum pixel movement before updating mouse (reduces overhead, default: 2)',
                        type=int,
                        default=2)
    parser.add_argument("--draw_quality",
                        help='Drawing quality: "high", "medium", "low" (default: medium)',
                        type=str,
                        default='medium',
                        choices=['high', 'medium', 'low'])
    parser.add_argument("--gesturehand",
                        help='Hand that triggers gesture actions (Open, Close, OK, Thumbs Up/Down, Pinch=left click, etc.).',
                        type=str,
                        default='right',
                        choices=['left', 'right'])
    parser.add_argument("--mousehand",
                        help='Hand that moves the cursor. Pointer gesture on this hand controls cursor position only (no click).',
                        type=str,
                        default='left',
                        choices=['left', 'right'])
    parser.add_argument("--gesture_hold_time",
                        help='Seconds to hold a gesture before it activates',
                        type=float,
                        default=2)
    parser.add_argument("--gesture_cooldown",
                        help='Cooldown in seconds between re-triggering the same gesture',
                        type=float,
                        default=1.5)
    parser.add_argument("--pinch_click_delay",
                        help='Seconds to hold a pinch before triggering a click',
                        type=float,
                        default=1.5)

    args = parser.parse_args()

    return args


def main_new_ui(args):
    """Run the new PyQt5 overlay UI with gesture recognition."""
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer, Qt
    from Overlay import OverlayWindow
    from Tray import TrayIcon

    automation_controller = AutomationController()
    automation_controller.start()
    speech_controller = SpeechDictationController(input_controller=automation_controller)
    speech_controller.start()
    processing_target_fps = 60 if args.high_performance else 30
    
    # Initialize MediaPipe and classifiers
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    
    # Determine number of threads for TensorFlow Lite
    num_threads = args.num_threads if args.num_threads is not None else _default_classifier_threads()

    base_path = get_base_path()
    keypoint_model = os.path.join(base_path, 'model', 'keypoint_classifier', 'keypoint_classifier.tflite')
    point_history_model = os.path.join(base_path, 'model', 'point_history_classifier', 'point_history_classifier.tflite')
    keypoint_classifier = KeyPointClassifier(model_path=keypoint_model, num_threads=num_threads)
    point_history_classifier = PointHistoryClassifier(model_path=point_history_model, num_threads=num_threads)

    # Read labels
    keypoint_classifier_labels, point_history_classifier_labels = _load_classifier_labels(base_path)

    # Resolve label indices for hand signs
    open_label_index = _get_label_index(keypoint_classifier_labels, 'Open')
    close_label_index = _get_label_index(keypoint_classifier_labels, 'Close')
    ok_label_index = _get_label_index(keypoint_classifier_labels, 'OK')
    thumbs_up_label_index = _get_label_index(keypoint_classifier_labels, 'Thumbs Up')
    thumbs_down_label_index = _get_label_index(keypoint_classifier_labels, 'Thumbs Down')
    two_fingers_up_label_index = _get_label_index(keypoint_classifier_labels, 'Two Fingers Up')
    three_fingers_up_label_index = _get_label_index(keypoint_classifier_labels, 'Three Fingers Up')
    four_fingers_up_label_index = _get_label_index(keypoint_classifier_labels, 'Four Fingers Up')
    pointer_label_index = _get_label_index(keypoint_classifier_labels, 'Pointer')
    pinch_label_index = _get_label_index(keypoint_classifier_labels, 'Pinch')
    
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
    air_mouse_state = {
        "prev_pos": None,
        "last_update_time": 0.0,
        "manual_override": False,
        "override_anchor": None,
    }
    
    # Cache screen size
    screen_size = automation_controller.get_screen_size()
    if screen_size is None:
        screen_width, screen_height = None, None
    else:
        screen_width, screen_height = screen_size
    
    # Mouse movement throttling
    mouse_update_interval = 1.0 / args.mouse_update_rate
    min_mouse_movement = args.min_mouse_movement
    
    # Debounce for pinch clicks
    last_pinch_click_time = [0.0]
    pinch_click_cooldown_sec = 1.0
    pinch_click_delay_sec = args.pinch_click_delay
    # Gesture hold time tracking
    first_detected_time = {}  # When each gesture was first detected
    last_activated_time = {}  # When each gesture was last activated
    gesture_hold_time_sec = args.gesture_hold_time
    gesture_cooldown_sec = args.gesture_cooldown

    def can_activate_gesture(label, now, hold_time_sec=None, cooldown_sec=None):
        hold_time_sec = gesture_hold_time_sec if hold_time_sec is None else hold_time_sec
        cooldown_sec = gesture_cooldown_sec if cooldown_sec is None else cooldown_sec
        # Check if gesture has been held long enough
        if label not in first_detected_time:
            first_detected_time[label] = now
            return False
        
        hold_duration = now - first_detected_time[label]
        if hold_duration < hold_time_sec:
            return False
        
        # Check cooldown after activation
        last_activated = last_activated_time.get(label, 0.0)
        if now - last_activated < cooldown_sec:
            return False
        
        last_activated_time[label] = now
        return True
    
    def reset_gesture_hold(label):
        # Call this when a gesture is no longer detected
        if label in first_detected_time:
            del first_detected_time[label]
    
    # Create the Qt application
    app = QApplication(sys.argv)
    
    # Create the overlay window
    detector = GestureDetector(
        camera_index=args.device,
        frame_width=args.width,
        frame_height=args.height,
        target_fps=processing_target_fps,
    )
    overlay = OverlayWindow(detector=detector)
    overlay.set_speech_controller(speech_controller)
    
    # Create the tray icon (owns gesture_hand / mouse_enabled state)
    tray = TrayIcon(
        overlay,
        gesture_hand=args.gesturehand.capitalize(),
        mouse_enabled=True,
        speech_controller=speech_controller,
    )
    
    # Store gesture state for the overlay
    current_gesture = ["Gesture: (awaiting detection...)"]
    last_gesture_label = {}  # Track previous gesture per hand to detect changes
    processing_stop_event = threading.Event()
    automation_available = automation_controller.is_available()
    frame_sequence = [None]
    

    # Single camera read per tick: worker drives frame + gesture; overlay only displays.
    overlay.set_app_driven(True)

    def process_frame_loop():
        """Process frames on a worker thread and publish the latest frame to the overlay."""
        target_interval = 1.0 / processing_target_fps

        while not processing_stop_event.is_set():
            loop_started = time.perf_counter()
            frame_sequence[0], frame = overlay.detector.get_latest_frame(
                previous_sequence=frame_sequence[0],
                timeout=max(0.05, target_interval * 2.0),
            )
            if frame is None:
                processing_stop_event.wait(0.01)
                continue

            speech_push_to_talk_active = False
            frame_height, frame_width = frame.shape[:2]

            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True
            mouse_tracking_active_this_frame = False

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    landmark_list = calc_landmark_list(frame, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(frame, point_history)

                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    hand_sign_label = keypoint_classifier_labels[hand_sign_id]
                    hand_label = handedness.classification[0].label

                    hand_id = hand_label
                    if hand_id in last_gesture_label and last_gesture_label[hand_id] != hand_sign_label:
                        reset_gesture_hold(last_gesture_label[hand_id])
                    last_gesture_label[hand_id] = hand_sign_label

                    is_gesture_hand = (hand_label == tray.gesture_hand)
                    is_mouse_hand = (hand_label == tray.mouse_hand) and tray.mouse_enabled
                    is_pointer_gesture = (pointer_label_index is not None and hand_sign_id == pointer_label_index)
                    is_pinch_gesture = (pinch_label_index is not None and hand_sign_id == pinch_label_index)
                    is_close_gesture = (close_label_index is not None and hand_sign_id == close_label_index)
                    is_mouse_move_gesture = is_pointer_gesture and is_mouse_hand
                    if is_gesture_hand and is_close_gesture:
                        speech_push_to_talk_active = True
                    if is_pointer_gesture or is_pinch_gesture:
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    if is_mouse_move_gesture:
                        if screen_width is not None and screen_height is not None:
                            _update_air_mouse_target(
                                automation_controller,
                                landmark_list,
                                frame_width,
                                frame_height,
                                screen_width,
                                screen_height,
                                mouse_sensitivity,
                                mouse_smoothing,
                                mouse_update_interval,
                                min_mouse_movement,
                                air_mouse_state,
                            )
                            mouse_tracking_active_this_frame = True

                    now = time.perf_counter()

                    if automation_available and is_gesture_hand and is_pinch_gesture:
                        if can_activate_gesture(
                            hand_sign_label,
                            now,
                            hold_time_sec=pinch_click_delay_sec,
                            cooldown_sec=pinch_click_cooldown_sec,
                        ) and now - last_pinch_click_time[0] >= pinch_click_cooldown_sec:
                            try:
                                automation_controller.click()
                                last_pinch_click_time[0] = now
                            except Exception:
                                pass

                    if automation_available and is_gesture_hand:
                        if ok_label_index is not None and hand_sign_id == ok_label_index:
                            if can_activate_gesture(hand_sign_label, now):
                                try:
                                    automation_controller.hotkey('ctrl', 't')
                                except Exception:
                                    pass
                        elif four_fingers_up_label_index is not None and hand_sign_id == four_fingers_up_label_index:
                            if can_activate_gesture(hand_sign_label, now):
                                try:
                                    automation_controller.hotkey('ctrl', 'w')
                                except Exception:
                                    pass

                    if volume_interface is not None and is_gesture_hand:
                        if thumbs_up_label_index is not None and hand_sign_id == thumbs_up_label_index:
                            if can_activate_gesture(hand_sign_label, now):
                                try:
                                    current_volume = volume_interface.GetMasterVolumeLevelScalar()
                                    new_volume = min(1.0, current_volume + 0.05)
                                    volume_interface.SetMasterVolumeLevelScalar(new_volume, None)
                                except Exception:
                                    pass
                        elif thumbs_down_label_index is not None and hand_sign_id == thumbs_down_label_index:
                            if can_activate_gesture(hand_sign_label, now):
                                try:
                                    current_volume = volume_interface.GetMasterVolumeLevelScalar()
                                    new_volume = max(0.0, current_volume - 0.05)
                                    volume_interface.SetMasterVolumeLevelScalar(new_volume, None)
                                except Exception:
                                    pass

                    if automation_available and is_gesture_hand:
                        if two_fingers_up_label_index is not None and hand_sign_id == two_fingers_up_label_index:
                            if can_activate_gesture(hand_sign_label, now):
                                try:
                                    automation_controller.press('playpause')
                                except Exception:
                                    pass
                        elif three_fingers_up_label_index is not None and hand_sign_id == three_fingers_up_label_index:
                            if can_activate_gesture(hand_sign_label, now):
                                try:
                                    automation_controller.hotkey('alt', 'left')
                                except Exception:
                                    pass

                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_history).most_common(1)

                    handedness_label = handedness.classification[0].label
                    finger_gesture_label = point_history_classifier_labels[most_common_fg_id[0][0]] if most_common_fg_id else ""
                    current_gesture[0] = f"{handedness_label}: {hand_sign_label}"
                    if finger_gesture_label:
                        current_gesture[0] += f" | {finger_gesture_label}"
            else:
                point_history.append([0, 0])
                current_gesture[0] = "Gesture: (no hand detected)"

            if not mouse_tracking_active_this_frame:
                _reset_air_mouse_state(air_mouse_state, automation_controller)

            update_push_to_talk(speech_push_to_talk_active, speech_controller)
            overlay.set_camera_frame(frame)
            overlay.set_gesture_label(current_gesture[0])

            elapsed = time.perf_counter() - loop_started
            remaining = target_interval - elapsed
            if remaining > 0:
                processing_stop_event.wait(remaining)

    processing_thread = threading.Thread(target=process_frame_loop, daemon=True)
    processing_thread.start()

    display_timer = QTimer()
    display_timer.setTimerType(Qt.PreciseTimer)
    display_timer.timeout.connect(overlay.update_frame)
    display_timer.start(16 if args.high_performance else 33)

    def shutdown():
        processing_stop_event.set()
        display_timer.stop()
        try:
            overlay.detector.release()
        except Exception:
            pass
        try:
            if processing_thread.is_alive():
                processing_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            speech_controller.stop()
        except Exception:
            pass
        try:
            automation_controller.stop()
        except Exception:
            pass
        close_method = getattr(hands, 'close', None)
        if callable(close_method):
            try:
                close_method()
            except Exception:
                pass

    app.aboutToQuit.connect(shutdown)

    # Show the overlay
    overlay.show()
    
    # Start the tray icon
    tray.run()
    
    # Run the Qt event loop
    sys.exit(app.exec_())


def main_old_ui(args):
    """Run the original OpenCV window UI with gesture recognition."""
    automation_controller = AutomationController()
    automation_controller.start()
    speech_controller = SpeechDictationController(input_controller=automation_controller)
    speech_controller.start()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    detector = GestureDetector(
        camera_index=cap_device,
        frame_width=cap_width,
        frame_height=cap_height,
        target_fps=60 if args.high_performance else 30,
    )

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        model_complexity=args.model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Determine number of threads for TensorFlow Lite
    num_threads = args.num_threads if args.num_threads is not None else _default_classifier_threads()

    base_path = get_base_path()
    keypoint_model = os.path.join(base_path, 'model', 'keypoint_classifier', 'keypoint_classifier.tflite')
    point_history_model = os.path.join(base_path, 'model', 'point_history_classifier', 'point_history_classifier.tflite')
    keypoint_classifier = KeyPointClassifier(model_path=keypoint_model, num_threads=num_threads)
    point_history_classifier = PointHistoryClassifier(model_path=point_history_model, num_threads=num_threads)

    # Read labels ###########################################################
    keypoint_classifier_labels, point_history_classifier_labels = _load_classifier_labels(base_path)

    # Resolve label indices for hand signs (if present)
    open_label_index = _get_label_index(keypoint_classifier_labels, 'Open')
    close_label_index = _get_label_index(keypoint_classifier_labels, 'Close')
    ok_label_index = _get_label_index(keypoint_classifier_labels, 'OK')
    thumbs_up_label_index = _get_label_index(keypoint_classifier_labels, 'Thumbs Up')
    thumbs_down_label_index = _get_label_index(keypoint_classifier_labels, 'Thumbs Down')
    two_fingers_up_label_index = _get_label_index(keypoint_classifier_labels, 'Two Fingers Up')
    three_fingers_up_label_index = _get_label_index(keypoint_classifier_labels, 'Three Fingers Up')
    four_fingers_up_label_index = _get_label_index(keypoint_classifier_labels, 'Four Fingers Up')
    pinch_label_index = _get_label_index(keypoint_classifier_labels, 'Pinch')

    # Hand assignment: mouse hand = Pointer (cursor movement); gesture hand = all other gestures including Pinch (left click)
    gesture_hand_label = args.gesturehand.capitalize()   # "Left" or "Right"
    mouse_hand_label = args.mousehand.capitalize()

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

    # Debounce for pinch clicks
    last_pinch_click_time = 0.0
    pinch_click_cooldown_sec = 1.0
    pinch_click_delay_sec = args.pinch_click_delay

    # Gesture hold time tracking
    first_detected_time = {}  # When each gesture was first detected
    last_activated_time = {}  # When each gesture was last activated
    gesture_hold_time_sec = args.gesture_hold_time
    gesture_cooldown_sec = args.gesture_cooldown

    def can_activate_gesture(label, now, hold_time_sec=None, cooldown_sec=None):
        hold_time_sec = gesture_hold_time_sec if hold_time_sec is None else hold_time_sec
        cooldown_sec = gesture_cooldown_sec if cooldown_sec is None else cooldown_sec
        # Check if gesture has been held long enough
        if label not in first_detected_time:
            first_detected_time[label] = now
            return False
        
        hold_duration = now - first_detected_time[label]
        if hold_duration < hold_time_sec:
            return False
        
        # Check cooldown after activation
        last_activated = last_activated_time.get(label, 0.0)
        if now - last_activated < cooldown_sec:
            return False
        
        last_activated_time[label] = now
        return True
    
    def reset_gesture_hold(label):
        # Call this when a gesture is no longer detected
        if label in first_detected_time:
            del first_detected_time[label]

    # Air mouse control variables
    enable_air_mouse = True  # Always enabled
    mouse_sensitivity = args.mouse_sensitivity
    mouse_smoothing = max(0.0, min(0.99, args.mouse_smoothing))  # Clamp to avoid div/edge issues
    air_mouse_state = {
        "prev_pos": None,
        "last_update_time": 0.0,
        "manual_override": False,
        "override_anchor": None,
    }
    pointer_label_index = _get_label_index(keypoint_classifier_labels, 'Pointer')
    
    # Performance optimization: Cache screen size
    screen_size = automation_controller.get_screen_size()
    if screen_size is None:
        screen_width, screen_height = None, None
    else:
        screen_width, screen_height = screen_size
    
    # Mouse movement throttling
    mouse_update_rate = args.mouse_update_rate
    min_mouse_movement = args.min_mouse_movement
    mouse_update_interval = 1.0 / mouse_update_rate
    
    # Drawing quality settings
    draw_quality = args.draw_quality
    draw_landmarks_enabled = draw_quality != 'low'
    draw_point_history_enabled = draw_quality != 'low'
    draw_info_enabled = True  # Always show FPS
    
    # Track previous gesture per hand to detect changes
    last_gesture_label = {}
    automation_available = automation_controller.is_available()
    frame_sequence = None

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        frame_sequence, image = detector.get_latest_frame(previous_sequence=frame_sequence, timeout=0.1)
        if image is None:
            continue
        # Performance: Use .copy() instead of deepcopy (much faster for 2D arrays)
        debug_image = image.copy()

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        speech_push_to_talk_active = False
        mouse_tracking_active_this_frame = False

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                brect = calc_bounding_rect(debug_image, landmark_list)

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
                hand_sign_label = keypoint_classifier_labels[hand_sign_id]
                hand_label = handedness.classification[0].label  # "Left" or "Right"
                
                # Reset gesture hold if gesture changed
                hand_id = hand_label
                if hand_id in last_gesture_label and last_gesture_label[hand_id] != hand_sign_label:
                    reset_gesture_hold(last_gesture_label[hand_id])
                last_gesture_label[hand_id] = hand_sign_label
                
                is_gesture_hand = (hand_label == gesture_hand_label)
                is_mouse_hand = (hand_label == mouse_hand_label)
                is_pointer_gesture = (pointer_label_index is not None and 
                                     hand_sign_id == pointer_label_index)
                is_pinch_gesture = (pinch_label_index is not None and 
                                    hand_sign_id == pinch_label_index)
                is_close_gesture = (close_label_index is not None and
                                    hand_sign_id == close_label_index)
                # Cursor movement: only Pointer gesture on the mouse hand
                is_mouse_move_gesture = is_pointer_gesture and is_mouse_hand
                if is_gesture_hand and is_close_gesture and mode == 0:
                    speech_push_to_talk_active = True
                if is_pointer_gesture or is_pinch_gesture:
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])
                
                if is_mouse_move_gesture:  # Pointer on mouse hand = move cursor
                    # Air mouse control (optimized for performance)
                    if enable_air_mouse and screen_width is not None and screen_height is not None and mode == 0:
                        _update_air_mouse_target(
                            automation_controller,
                            landmark_list,
                            cap_width,
                            cap_height,
                            screen_width,
                            screen_height,
                            mouse_sensitivity,
                            mouse_smoothing,
                            mouse_update_interval,
                            min_mouse_movement,
                            air_mouse_state,
                        )
                        mouse_tracking_active_this_frame = True

                now = time.perf_counter()

                # Pinch on gesture hand = left click after a short hold
                if mode == 0 and automation_available and is_gesture_hand and is_pinch_gesture:
                    if can_activate_gesture(
                        hand_sign_label,
                        now,
                        hold_time_sec=pinch_click_delay_sec,
                        cooldown_sec=pinch_click_cooldown_sec,
                    ) and now - last_pinch_click_time >= pinch_click_cooldown_sec:
                        try:
                            automation_controller.click()
                            last_pinch_click_time = now
                        except Exception:
                            pass

                # Hotkey actions: Gesture -> Keyboard shortcut (gesture hand only)
                # - Only when not in logging modes (mode == 0)
                # - Per-gesture cooldown to avoid repeated triggers
                # - Requires pyautogui to be available
                if mode == 0 and automation_available and is_gesture_hand:
                    # OK sign -> Ctrl+T (new tab)
                    if (ok_label_index is not None and
                            hand_sign_id == ok_label_index):
                        if can_activate_gesture(hand_sign_label, now):
                            try:
                                automation_controller.hotkey('ctrl', 't')
                            except Exception:
                                pass
                    # Four Fingers Up -> Ctrl+W (close tab)
                    elif (four_fingers_up_label_index is not None and
                          hand_sign_id == four_fingers_up_label_index):
                        if can_activate_gesture(hand_sign_label, now):
                            try:
                                automation_controller.hotkey('ctrl', 'w')
                            except Exception:
                                pass
                
                # Volume control: Gesture -> Volume adjustment (gesture hand only)
                # - Only when not in logging modes (mode == 0)
                # - Per-gesture cooldown to avoid repeated triggers
                # - Requires volume control to be available
                if mode == 0 and volume_interface is not None and is_gesture_hand:
                    # Thumbs Up -> Increase volume
                    if (thumbs_up_label_index is not None and
                            hand_sign_id == thumbs_up_label_index):
                        if can_activate_gesture(hand_sign_label, now):
                            try:
                                current_volume = volume_interface.GetMasterVolumeLevelScalar()
                                # Increase volume by 5% (0.05)
                                new_volume = min(1.0, current_volume + 0.05)
                                volume_interface.SetMasterVolumeLevelScalar(new_volume, None)
                            except Exception:
                                pass
                    # Thumbs Down -> Decrease volume
                    elif (thumbs_down_label_index is not None and
                          hand_sign_id == thumbs_down_label_index):
                        if can_activate_gesture(hand_sign_label, now):
                            try:
                                current_volume = volume_interface.GetMasterVolumeLevelScalar()
                                # Decrease volume by 5% (0.05)
                                new_volume = max(0.0, current_volume - 0.05)
                                volume_interface.SetMasterVolumeLevelScalar(new_volume, None)
                            except Exception:
                                pass
                
                # Two Fingers Up -> Play/Pause toggle
                # Three Fingers Up -> Go back (Alt+Left) (gesture hand only)
                if mode == 0 and automation_available and is_gesture_hand:
                    if two_fingers_up_label_index is not None and hand_sign_id == two_fingers_up_label_index:
                        if can_activate_gesture(hand_sign_label, now):
                            try:
                                automation_controller.press('playpause')
                            except Exception:
                                pass
                    elif three_fingers_up_label_index is not None and hand_sign_id == three_fingers_up_label_index:
                        if can_activate_gesture(hand_sign_label, now):
                            try:
                                automation_controller.hotkey('alt', 'left')
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
                most_common_fg_id = Counter(finger_gesture_history).most_common(1)

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
                    point_history_classifier_labels[most_common_fg_id[0][0]] if most_common_fg_id else "",
                )
        else:
            point_history.append([0, 0])

        if not mouse_tracking_active_this_frame:
            _reset_air_mouse_state(air_mouse_state, automation_controller)

        update_push_to_talk(speech_push_to_talk_active, speech_controller)
        if draw_point_history_enabled:
            debug_image = draw_point_history(debug_image, point_history)
        if draw_info_enabled:
            debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    speech_controller.stop()
    automation_controller.stop()

    detector.release()
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


def calc_landmark_array(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((len(landmarks.landmark), 2), dtype=np.int32)

    for idx, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array[idx, 0] = landmark_x
        landmark_array[idx, 1] = landmark_y

    return landmark_array


def calc_bounding_rect(image, landmarks):
    landmark_array = landmarks if isinstance(landmarks, np.ndarray) else calc_landmark_array(image, landmarks)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    if isinstance(landmarks, np.ndarray):
        return landmarks
    return calc_landmark_array(image, landmarks)


def pre_process_landmark(landmark_list):
    temp_landmark_list = np.asarray(landmark_list, dtype=np.float32).copy()

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
        temp_landmark_list = np.zeros_like(temp_landmark_list)

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    if len(point_history) == 0:
        return np.empty((0,), dtype=np.float32)
    
    temp_point_history = np.asarray(point_history, dtype=np.float32).copy()

    # Convert to relative coordinates
    base_x, base_y = temp_point_history[0, 0], temp_point_history[0, 1]
    temp_point_history[:, 0] = (temp_point_history[:, 0] - base_x) / image_width
    temp_point_history[:, 1] = (temp_point_history[:, 1] - base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = temp_point_history.flatten()

    return temp_point_history


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
