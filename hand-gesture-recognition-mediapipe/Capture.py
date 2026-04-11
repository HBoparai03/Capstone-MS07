# gesture_detector.py
import os

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
    """Handles camera capture and gesture detection placeholder."""
    def __init__(self, camera_index=0, frame_width=640, frame_height=480, target_fps=30):
        self.cap = open_camera(
            camera_index=camera_index,
            frame_width=frame_width,
            frame_height=frame_height,
            target_fps=target_fps,
        )

    def get_frame(self):
        """Capture a frame from the camera."""
        if not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Flip horizontally for a mirrored view
        frame = cv2.flip(frame, 1)
        return frame

    def get_gesture_label(self):
        """Placeholder gesture label (will be replaced by MediaPipe output)."""
        return "Gesture: (awaiting detection...)"

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
