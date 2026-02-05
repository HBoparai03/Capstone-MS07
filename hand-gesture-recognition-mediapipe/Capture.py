# gesture_detector.py
import cv2

class GestureDetector:
    """Handles camera capture and gesture detection placeholder."""
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
