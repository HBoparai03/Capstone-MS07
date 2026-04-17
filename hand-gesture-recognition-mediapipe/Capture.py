import os
import threading
import time

import cv2


def open_camera(camera_index=0, frame_width=640, frame_height=480, target_fps=30):
    """Open a low-latency camera stream with a Windows-friendly backend when possible."""
    preferred_backends = []
    if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
        preferred_backends.append(cv2.CAP_DSHOW)
    preferred_backends.append(cv2.CAP_ANY)

    cap = None
    for backend in preferred_backends:
        if backend == cv2.CAP_ANY:
            candidate = cv2.VideoCapture(camera_index)
        else:
            candidate = cv2.VideoCapture(camera_index, backend)

        if candidate is not None and candidate.isOpened():
            cap = candidate
            break

        if candidate is not None:
            candidate.release()

    if cap is None:
        cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

class GestureDetector:
    """Owns a low-latency camera reader that always exposes the newest frame."""

    def __init__(self, camera_index=0, frame_width=640, frame_height=480, target_fps=30):
        self.cap = open_camera(
            camera_index=camera_index,
            frame_width=frame_width,
            frame_height=frame_height,
            target_fps=target_fps,
        )
        self._frame_condition = threading.Condition()
        self._stop_event = threading.Event()
        self._latest_frame = None
        self._frame_sequence = -1
        self._reader_thread = None
        if self.cap is not None and self.cap.isOpened():
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()

    def _reader_loop(self):
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            with self._frame_condition:
                self._latest_frame = frame
                self._frame_sequence += 1
                self._frame_condition.notify_all()

    def get_latest_frame(self, previous_sequence=None, timeout=0.05):
        """Return the newest frame once it advances beyond previous_sequence."""
        deadline = None if timeout is None else time.perf_counter() + timeout

        with self._frame_condition:
            while not self._stop_event.is_set():
                if self._latest_frame is not None:
                    current_sequence = self._frame_sequence
                    if previous_sequence is None or current_sequence != previous_sequence:
                        return current_sequence, self._latest_frame

                if timeout is None:
                    self._frame_condition.wait()
                    continue

                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._frame_condition.wait(remaining)

        return previous_sequence, None

    def get_frame(self):
        """Capture a frame from the camera."""
        _, frame = self.get_latest_frame(timeout=0.05)
        return frame

    def get_gesture_label(self):
        """Placeholder gesture label (will be replaced by MediaPipe output)."""
        return "Gesture: (awaiting detection...)"

    def release(self):
        self._stop_event.set()
        with self._frame_condition:
            self._frame_condition.notify_all()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
