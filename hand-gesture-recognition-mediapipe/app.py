#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import os
import sys
import atexit
import faulthandler
import logging
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
    return _candidate_base_paths()[0]


class _LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if not message:
            return
        for line in message.rstrip().splitlines():
            line = line.strip()
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        for handler in self.logger.handlers:
            try:
                handler.flush()
            except Exception:
                pass


_FAULT_LOG_HANDLE = None
APP_LOG_PATH = None


def _get_log_path():
    if getattr(sys, "frozen", False):
        log_root = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "HandGestureApp", "logs")
    else:
        log_root = os.path.join(get_base_path(), "logs")
    os.makedirs(log_root, exist_ok=True)
    return os.path.join(log_root, "hand_gesture_app.log")


def setup_runtime_logging():
    global _FAULT_LOG_HANDLE, APP_LOG_PATH

    APP_LOG_PATH = _get_log_path()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(APP_LOG_PATH, encoding="utf-8")],
        force=True,
    )

    app_logger = logging.getLogger("handgesture")
    app_logger.info("=== Application start ===")
    app_logger.info("Frozen=%s", getattr(sys, "frozen", False))
    app_logger.info("Executable=%s", sys.executable)
    app_logger.info("Base path=%s", get_base_path())
    app_logger.info("Log path=%s", APP_LOG_PATH)

    def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
        logging.getLogger("handgesture.crash").critical(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = _log_uncaught_exception

    sys.stdout = _LoggerWriter(logging.getLogger("stdout"), logging.INFO)
    sys.stderr = _LoggerWriter(logging.getLogger("stderr"), logging.ERROR)

    try:
        _FAULT_LOG_HANDLE = open(APP_LOG_PATH, "a", encoding="utf-8")
        faulthandler.enable(_FAULT_LOG_HANDLE, all_threads=True)
    except Exception:
        app_logger.exception("Failed to enable faulthandler")

    def _shutdown_logging():
        logging.getLogger("handgesture").info("=== Application shutdown ===")
        if _FAULT_LOG_HANDLE is not None:
            try:
                _FAULT_LOG_HANDLE.flush()
                _FAULT_LOG_HANDLE.close()
            except Exception:
                pass

    atexit.register(_shutdown_logging)


setup_runtime_logging()

import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import threading
import queue

if hasattr(threading, "excepthook"):
    def _threading_excepthook(args):
        logging.getLogger("handgesture.thread").critical(
            "Unhandled thread exception in %s",
            args.thread.name if args.thread is not None else "unknown",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
    threading.excepthook = _threading_excepthook

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

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier


class SpeechDictationController:
    """Persistent background speech-to-text worker controlled by gestures or tray actions."""

    def __init__(self, model_name="small", sample_rate=16000, chunk_seconds=3.5, language="en"):
        self._logger = logging.getLogger("handgesture.speech")
        self.model_name = model_name
        self._model_dir = self._resolve_model_dir(model_name)
        self.sample_rate = sample_rate
        self.chunk_frames = int(sample_rate * chunk_seconds)
        self.language = language
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
        self._available = (sd is not None and WhisperModel is not None and pyautogui is not None)
        self._logger.info(
            "Speech controller initialized: available=%s model_dir=%s sample_rate=%s chunk_frames=%s",
            self._available,
            self._model_dir,
            self.sample_rate,
            self.chunk_frames,
        )
        if self._available:
            self._input_device_index, self._input_device_name = self._resolve_input_device()
            if self._input_device_index is None:
                self._available = False
                self._last_error = "No input microphone found"
                self._logger.error(self._last_error)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._speech_worker, daemon=True, name="SpeechWorker")
        self._logger.info("Starting speech worker thread")
        self._thread.start()

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
        self._logger.info("Speech toggle requested: enabled=%s", enabled)
        with self._lock:
            if not self._available:
                self._logger.warning("Speech toggle ignored because speech is unavailable: %s", self._last_error)
                return False
            changed = self._speech_enabled != enabled
            self._speech_enabled = enabled
            if enabled:
                self._last_typed_text = ""
            else:
                self._model_loading = False
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
            input_device_name = self._input_device_name
        if not available:
            status = "Speech: Unavailable"
        elif model_loading:
            status = "Speech: Loading..."
        else:
            status = f"Speech: {'ON' if enabled else 'OFF'}"
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
        self._logger.exception("Speech runtime error", exc_info=(type(exc), exc, exc.__traceback__))

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

    @staticmethod
    def _resolve_model_dir(model_name):
        bundled_model_dir = resource_path(os.path.join("speech_models", f"whisper-{model_name}"))
        if os.path.isdir(bundled_model_dir):
            return bundled_model_dir

        appdata = os.environ.get("APPDATA", os.path.expanduser("~"))
        appdata_model_dir = os.path.join(appdata, "HandGestureApp", "models", f"whisper-{model_name}")

        if os.path.isdir(appdata_model_dir):
            return appdata_model_dir

        if not getattr(sys, 'frozen', False):
            return None  # script mode: faster-whisper resolves via HuggingFace cache

        return appdata_model_dir

    def _is_model_ready(self):
        """Return True if the model is available locally (or we are in script mode)."""
        if self._model_dir is None:
            return True  # script mode — let faster-whisper handle it
        required = ["model.bin", "config.json"]
        return os.path.isdir(self._model_dir) and all(
            os.path.isfile(os.path.join(self._model_dir, f)) for f in required
        )

    def download_model(self):
        """Download the Whisper model from HuggingFace into the local app data folder.

        Call this once before first use in EXE mode (e.g. from an installer or
        a one-time setup dialog). Not called automatically.
        """
        if self._model_dir is None:
            return  # nothing to do in script mode
        try:
            from huggingface_hub import snapshot_download
            os.makedirs(self._model_dir, exist_ok=True)
            snapshot_download(
                repo_id=f"Systran/faster-whisper-{self.model_name}",
                local_dir=self._model_dir,
            )
        except Exception as exc:
            self._set_runtime_error(exc)

    def _load_model(self):
        with self._lock:
            if self._model is not None:
                return self._model
            self._model_loading = True
            self._last_error = ""

        self._logger.info(
            "Loading Faster-Whisper model from %s",
            self._model_dir if self._model_dir is not None else self.model_name,
        )

        if not self._is_model_ready():
            with self._lock:
                self._available = False
                self._model_loading = False
                self._speech_enabled = False
                self._last_error = f"Speech model not found at {self._model_dir}"
            self._logger.error("Speech model not found at %s", self._model_dir)
            return None

        # In EXE mode use the local folder path; in script mode use the model name
        # so faster-whisper resolves it via the HuggingFace cache as normal.
        model_source = self._model_dir if self._model_dir is not None else self.model_name

        model = None
        load_error = None
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("CT2_USE_EXPERIMENTAL_PACKED_GEMM", "0")
        load_attempts = (
            {"device": "cpu", "compute_type": "float32", "cpu_threads": 1, "num_workers": 1},
        )
        for attempt in load_attempts:
            try:
                self._logger.info(
                    "Whisper load attempt: device=%s compute_type=%s cpu_threads=%s num_workers=%s source=%s",
                    attempt["device"],
                    attempt["compute_type"],
                    attempt["cpu_threads"],
                    attempt["num_workers"],
                    model_source,
                )
                model = WhisperModel(
                    model_source,
                    device=attempt["device"],
                    compute_type=attempt["compute_type"],
                    cpu_threads=attempt["cpu_threads"],
                    num_workers=attempt["num_workers"],
                )
                break
            except Exception as exc:
                load_error = exc
        if model is None:
            with self._lock:
                self._available = False
                self._model_loading = False
                self._speech_enabled = False
                self._last_error = f"{load_error.__class__.__name__}: {load_error}"
            self._logger.exception(
                "Whisper model load failed",
                exc_info=(type(load_error), load_error, load_error.__traceback__),
            )
            return None
        with self._lock:
            self._model = model
            self._model_loading = False
        self._logger.info("Whisper model loaded successfully")
        return model

    def _speech_worker(self):
        silence_threshold = 0.01
        self._logger.info("Speech worker loop started")

        while not self._stop_event.is_set():
            with self._lock:
                available = self._available
                enabled = self._speech_enabled

            if not available:
                self._wake_event.wait(0.25)
                self._wake_event.clear()
                continue

            if not enabled:
                self._wake_event.wait(0.1)
                self._wake_event.clear()
                continue

            model = self._model if self._model is not None else self._load_model()
            if model is None:
                self._wake_event.wait(0.25)
                self._wake_event.clear()
                continue

            try:
                audio = sd.rec(
                    self.chunk_frames,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    device=self._input_device_index,
                )
                sd.wait()
            except Exception as exc:
                self._set_runtime_error(exc)
                self._wake_event.wait(0.25)
                self._wake_event.clear()
                continue

            if self._stop_event.is_set():
                break

            with self._lock:
                enabled = self._speech_enabled
                self._last_error = ""

            if not enabled:
                continue

            audio = np.squeeze(audio)
            if audio.size == 0 or float(np.max(np.abs(audio))) < silence_threshold:
                continue

            try:
                segments, _ = model.transcribe(
                    audio,
                    language=self.language,
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
            except Exception as exc:
                self._set_runtime_error(exc)
                continue

            normalized_text = " ".join(text.split())
            if not normalized_text:
                continue

            self._logger.info("Speech transcript produced: %s", normalized_text)

            with self._lock:
                self._last_transcript = normalized_text
                if normalized_text == self._last_typed_text:
                    continue

            try:
                pyautogui.write(normalized_text + " ", interval=0.0)
            except Exception as exc:
                self._set_runtime_error(exc)
                continue

            with self._lock:
                self._last_typed_text = normalized_text


def handle_speech_gesture(hand_sign_id, hand_sign_label, now, open_label_index, close_label_index,
                          speech_controller, can_activate_gesture):
    """Use Open to enable dictation and Close to disable it, reusing gesture hold/cooldown logic."""
    if speech_controller is None or not speech_controller.is_available():
        return
    if open_label_index is not None and hand_sign_id == open_label_index:
        if can_activate_gesture(hand_sign_label, now):
            speech_controller.set_enabled(True)
    elif close_label_index is not None and hand_sign_id == close_label_index:
        if can_activate_gesture(hand_sign_label, now):
            speech_controller.set_enabled(False)


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
                        action='store_true',
                        default=True)
    parser.add_argument("--no_high_performance",
                        help='Disable high-performance mode',
                        action='store_false',
                        dest='high_performance')
    parser.add_argument("--num_threads",
                        help='Number of threads for TensorFlow Lite (default: 8 in high-performance, 1 otherwise)',
                        type=int,
                        default=None)
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

    speech_controller = SpeechDictationController()
    speech_controller.start()
    
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
        num_threads = args.num_threads if args.num_threads is not None else 8
    else:
        num_threads = args.num_threads if args.num_threads is not None else 1

    base_path = get_base_path()
    keypoint_model = os.path.join(base_path, 'model', 'keypoint_classifier', 'keypoint_classifier.tflite')
    point_history_model = os.path.join(base_path, 'model', 'point_history_classifier', 'point_history_classifier.tflite')
    keypoint_classifier = KeyPointClassifier(model_path=keypoint_model, num_threads=num_threads)
    point_history_classifier = PointHistoryClassifier(model_path=point_history_model, num_threads=num_threads)

    # Read labels
    with open(os.path.join(base_path, 'model', 'keypoint_classifier', 'keypoint_classifier_label.csv'),
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    with open(os.path.join(base_path, 'model', 'point_history_classifier', 'point_history_classifier_label.csv'),
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
    ok_label_index = get_label_index(keypoint_classifier_labels, 'OK')
    thumbs_up_label_index = get_label_index(keypoint_classifier_labels, 'Thumbs Up')
    thumbs_down_label_index = get_label_index(keypoint_classifier_labels, 'Thumbs Down')
    two_fingers_up_label_index = get_label_index(keypoint_classifier_labels, 'Two Fingers Up')
    three_fingers_up_label_index = get_label_index(keypoint_classifier_labels, 'Three Fingers Up')
    four_fingers_up_label_index = get_label_index(keypoint_classifier_labels, 'Four Fingers Up')
    pointer_label_index = get_label_index(keypoint_classifier_labels, 'Pointer')
    pinch_label_index = get_label_index(keypoint_classifier_labels, 'Pinch')
    
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
    overlay = OverlayWindow()
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
    

    # Single camera read per tick: worker drives frame + gesture; overlay only displays.
    overlay.set_app_driven(True)

    def process_frame_loop():
        """Process frames on a worker thread and publish the latest frame to the overlay."""
        target_interval = 1.0 / 30.0

        while not processing_stop_event.is_set():
            loop_started = time.perf_counter()
            frame = overlay.detector.get_frame()
            if frame is None:
                processing_stop_event.wait(0.01)
                continue

            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

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
                    is_mouse_move_gesture = is_pointer_gesture and is_mouse_hand
                    if is_pointer_gesture or is_pinch_gesture:
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    if is_mouse_move_gesture:
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
                        prev_mouse_pos[0] = None

                    if pyautogui is not None and is_gesture_hand and is_pinch_gesture:
                        now = time.time()
                        if can_activate_gesture(
                            hand_sign_label,
                            now,
                            hold_time_sec=pinch_click_delay_sec,
                            cooldown_sec=pinch_click_cooldown_sec,
                        ) and now - last_pinch_click_time[0] >= pinch_click_cooldown_sec:
                            try:
                                pyautogui.click()
                                last_pinch_click_time[0] = now
                            except Exception:
                                pass

                    if pyautogui is not None and is_gesture_hand:
                        now = time.time()
                        if open_label_index is not None and hand_sign_id == open_label_index:
                            handle_speech_gesture(
                                hand_sign_id,
                                hand_sign_label,
                                now,
                                open_label_index,
                                close_label_index,
                                speech_controller,
                                can_activate_gesture,
                            )
                        elif close_label_index is not None and hand_sign_id == close_label_index:
                            handle_speech_gesture(
                                hand_sign_id,
                                hand_sign_label,
                                now,
                                open_label_index,
                                close_label_index,
                                speech_controller,
                                can_activate_gesture,
                            )
                        elif ok_label_index is not None and hand_sign_id == ok_label_index:
                            if can_activate_gesture(hand_sign_label, now):
                                try:
                                    pyautogui.hotkey('ctrl', 't')
                                except Exception:
                                    pass
                        elif four_fingers_up_label_index is not None and hand_sign_id == four_fingers_up_label_index:
                            if can_activate_gesture(hand_sign_label, now):
                                try:
                                    pyautogui.hotkey('ctrl', 'w')
                                except Exception:
                                    pass

                    if volume_interface is not None and is_gesture_hand:
                        now = time.time()
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

                    if pyautogui is not None and is_gesture_hand:
                        now = time.time()
                        if two_fingers_up_label_index is not None and hand_sign_id == two_fingers_up_label_index:
                            if can_activate_gesture(hand_sign_label, now):
                                try:
                                    pyautogui.press('playpause')
                                except Exception:
                                    pass
                        elif three_fingers_up_label_index is not None and hand_sign_id == three_fingers_up_label_index:
                            if can_activate_gesture(hand_sign_label, now):
                                try:
                                    pyautogui.hotkey('alt', 'left')
                                except Exception:
                                    pass

                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_history).most_common()

                    handedness_label = handedness.classification[0].label
                    finger_gesture_label = point_history_classifier_labels[most_common_fg_id[0][0]]
                    current_gesture[0] = f"{handedness_label}: {hand_sign_label}"
                    if finger_gesture_label:
                        current_gesture[0] += f" | {finger_gesture_label}"
            else:
                point_history.append([0, 0])
                current_gesture[0] = "Gesture: (no hand detected)"

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
    display_timer.start(33)

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
    speech_controller = SpeechDictationController()
    speech_controller.start()

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
        num_threads = args.num_threads if args.num_threads is not None else 8
    else:
        num_threads = args.num_threads if args.num_threads is not None else 1

    base_path = get_base_path()
    keypoint_model = os.path.join(base_path, 'model', 'keypoint_classifier', 'keypoint_classifier.tflite')
    point_history_model = os.path.join(base_path, 'model', 'point_history_classifier', 'point_history_classifier.tflite')
    keypoint_classifier = KeyPointClassifier(model_path=keypoint_model, num_threads=num_threads)
    point_history_classifier = PointHistoryClassifier(model_path=point_history_model, num_threads=num_threads)

    # Read labels ###########################################################
    with open(os.path.join(base_path, 'model', 'keypoint_classifier', 'keypoint_classifier_label.csv'),
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(os.path.join(base_path, 'model', 'point_history_classifier', 'point_history_classifier_label.csv'),
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
        ok_label_index = keypoint_classifier_labels.index('OK')
    except ValueError:
        ok_label_index = None

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

    try:
        four_fingers_up_label_index = keypoint_classifier_labels.index('Four Fingers Up')
    except ValueError:
        four_fingers_up_label_index = None

    try:
        pinch_label_index = keypoint_classifier_labels.index('Pinch')
    except ValueError:
        pinch_label_index = None

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

    # Debounce for hotkey actions
    last_hotkey_time = 0.0
    hotkey_cooldown_sec = 1.5
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
    
    # Thread-safe mouse movement queue for high-performance mode (Event + sentinel for clean shutdown)
    mouse_queue = queue.Queue(maxsize=1) if args.high_performance and pyautogui else None
    mouse_stop_event = threading.Event() if (args.high_performance and pyautogui) else None
    _MOUSE_SENTINEL = object()  # unique sentinel to signal worker to exit

    def mouse_worker():
        """Background thread for mouse movement; exits when sentinel is received or stop event is set."""
        while not mouse_stop_event.is_set():
            try:
                item = mouse_queue.get(timeout=0.1)
                if item is _MOUSE_SENTINEL:
                    mouse_queue.task_done()
                    break
                x, y = item
                pyautogui.moveTo(x, y, duration=0.0)
                mouse_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    # Start mouse worker thread if in high-performance mode
    if mouse_queue is not None and mouse_stop_event is not None:
        mouse_thread = threading.Thread(target=mouse_worker, daemon=True)
        mouse_thread.start()
    
    # Drawing quality settings
    draw_quality = args.draw_quality
    draw_landmarks_enabled = draw_quality != 'low'
    draw_point_history_enabled = draw_quality != 'low'
    draw_info_enabled = True  # Always show FPS
    
    # Track previous gesture per hand to detect changes
    last_gesture_label = {}

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
                # Cursor movement: only Pointer gesture on the mouse hand
                is_mouse_move_gesture = is_pointer_gesture and is_mouse_hand
                if is_pointer_gesture or is_pinch_gesture:
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])
                
                if is_mouse_move_gesture:  # Pointer on mouse hand = move cursor
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
                    # Reset previous mouse position when not in pointer (move) mode
                    prev_mouse_pos = None

                # Pinch on gesture hand = left click after a short hold
                if mode == 0 and pyautogui is not None and is_gesture_hand and is_pinch_gesture:
                    now = time.time()
                    if can_activate_gesture(
                        hand_sign_label,
                        now,
                        hold_time_sec=pinch_click_delay_sec,
                        cooldown_sec=pinch_click_cooldown_sec,
                    ) and now - last_pinch_click_time >= pinch_click_cooldown_sec:
                        try:
                            pyautogui.click()
                            last_pinch_click_time = now
                        except Exception:
                            pass

                # Hotkey actions: Gesture -> Keyboard shortcut (gesture hand only)
                # - Only when not in logging modes (mode == 0)
                # - Per-gesture cooldown to avoid repeated triggers
                # - Requires pyautogui to be available
                if mode == 0 and pyautogui is not None and is_gesture_hand:
                    now = time.time()
                    # Open hand -> enable speech dictation
                    if (open_label_index is not None and
                            hand_sign_id == open_label_index):
                        handle_speech_gesture(
                            hand_sign_id,
                            hand_sign_label,
                            now,
                            open_label_index,
                            close_label_index,
                            speech_controller,
                            can_activate_gesture,
                        )
                    # Close hand -> disable speech dictation
                    elif (close_label_index is not None and
                            hand_sign_id == close_label_index):
                        handle_speech_gesture(
                            hand_sign_id,
                            hand_sign_label,
                            now,
                            open_label_index,
                            close_label_index,
                            speech_controller,
                            can_activate_gesture,
                        )
                    # OK sign -> Ctrl+T (new tab)
                    elif (ok_label_index is not None and
                            hand_sign_id == ok_label_index):
                        if can_activate_gesture(hand_sign_label, now):
                            try:
                                pyautogui.hotkey('ctrl', 't')
                            except Exception:
                                pass
                    # Four Fingers Up -> Ctrl+W (close tab)
                    elif (four_fingers_up_label_index is not None and
                          hand_sign_id == four_fingers_up_label_index):
                        if can_activate_gesture(hand_sign_label, now):
                            try:
                                pyautogui.hotkey('ctrl', 'w')
                            except Exception:
                                pass
                
                # Volume control: Gesture -> Volume adjustment (gesture hand only)
                # - Only when not in logging modes (mode == 0)
                # - Per-gesture cooldown to avoid repeated triggers
                # - Requires volume control to be available
                if mode == 0 and volume_interface is not None and is_gesture_hand:
                    now = time.time()
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
                if mode == 0 and pyautogui is not None and is_gesture_hand:
                    now = time.time()
                    if two_fingers_up_label_index is not None and hand_sign_id == two_fingers_up_label_index:
                        if can_activate_gesture(hand_sign_label, now):
                            try:
                                pyautogui.press('playpause')
                            except Exception:
                                pass
                    elif three_fingers_up_label_index is not None and hand_sign_id == three_fingers_up_label_index:
                        if can_activate_gesture(hand_sign_label, now):
                            try:
                                pyautogui.hotkey('alt', 'left')
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

    # Cleanup: signal mouse worker to exit and wait for it
    if mouse_queue is not None and mouse_stop_event is not None:
        mouse_stop_event.set()
        try:
            mouse_queue.put_nowait(_MOUSE_SENTINEL)
        except queue.Full:
            pass
        mouse_thread.join(timeout=1.5)

    speech_controller.stop()

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
