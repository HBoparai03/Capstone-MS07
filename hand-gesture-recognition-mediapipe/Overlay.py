from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QApplication,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter
import ctypes
import cv2
import os
import sys
from Capture import GestureDetector

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.detector = GestureDetector()
        self.overlay_visible = True
        self.table_visible = False
        # When True, frame/label are set by app (single camera read per tick); no get_frame in update_frame
        self._app_driven = False
        self._last_frame = None
        self._last_gesture_label = None
        self._speech_controller = None

        # --- Window Setup ---
        self.setWindowTitle("Gesture Overlay")
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool |
            Qt.BypassWindowManagerHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        screen = QApplication.primaryScreen()
        self.setGeometry(screen.availableGeometry())

        # --- Layout ---
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(10)

        # --- Camera feed ---
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(320, 240)
        self.camera_label.setStyleSheet(
            "border: 2px solid rgba(255,255,255,0.3); border-radius: 8px;"
        )
        self.layout.addWidget(self.camera_label, alignment=Qt.AlignTop | Qt.AlignLeft)

        # --- Gesture instruction table ---
        self.gesture_table = QTableWidget(self)
        self.gesture_table.setColumnCount(3)
        self.gesture_table.setRowCount(9)
        self.gesture_table.setHorizontalHeaderLabels(["Gesture", "Action", "Description"])
        self.gesture_table.verticalHeader().setVisible(False)
        self.gesture_table.horizontalHeader().setStretchLastSection(True)
        self.gesture_table.setWordWrap(True)

        # Allow table and rows to grow vertically
        self.gesture_table.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.gesture_table.setMinimumWidth(600)
        self.gesture_table.setMinimumHeight(620)

        # Style
        self.gesture_table.setStyleSheet("""
            QTableWidget {
                background-color: rgba(0, 0, 0, 140);
                color: white;
                gridline-color: rgba(255,255,255,40);
                border-radius: 10px;
            }
            QHeaderView::section {
                background-color: rgba(40,40,40,180);
                color: white;
                padding: 6px;
                border: none;
            }
            QScrollBar:vertical {
                background: rgba(0,0,0,60);
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255,255,255,120);
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # Placeholder rows
        placeholders = [
            ("Open Hand", "Enable speech dictation", "Starts microphone listening and Whisper dictation"),
            ("Closed Fist", "Disable speech dictation", "Stops microphone listening"),
            ("OK Sign", "Open a new browser tab (Ctrl + T)", ""),
            ("Four Fingers Up", "Close the current tab. (Ctrl + W)", ""),
            ("Thumbs Up", "Volume Up (system control)", ""),
            ("Thumbs Down", "Volume Down (system control)", ""),
            ("Two Fingers Up", "Play media (Spacebar)", ""),
            ("Three Fingers Up","Will go back one tab (alt + left)",""),
            ("Pinch","Starting from Index and Thumb out, moving to Pinch will click the mouse","")
        ]

        for row, (gesture, action, desc) in enumerate(placeholders):
            for col, text in enumerate((gesture, action, desc)):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.gesture_table.setItem(row, col, item)

        self.add_image_to_cell(2, 2, resource_path("icons/Ok.png"))
        self.add_image_to_cell(3, 2, resource_path("icons/fourfu.png"))
        self.add_image_to_cell(4, 2, resource_path("icons/tup.png"))
        self.add_image_to_cell(5, 2, resource_path("icons/tdown.png"))
        self.add_image_to_cell(6, 2, resource_path("icons/twofu.png"))
        self.add_image_to_cell(7, 2, resource_path("icons/threefu.png"))
        self.add_image_to_cell(8, 2, resource_path("icons/pinch.png"))

        # Resize rows/columns to fit content
        self.gesture_table.resizeColumnsToContents()
        self.gesture_table.resizeRowsToContents()
        self.gesture_table.setColumnWidth(1, 200)

        self.gesture_table.hide()
        self.layout.addWidget(self.gesture_table, alignment=Qt.AlignLeft)

        # --- Gesture label ---
        self.gesture_label = QLabel("Gesture: (awaiting detection...)")
        self.gesture_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
                background-color: rgba(0, 0, 0, 128);
                border-radius: 10px;
                padding: 8px 16px;
            }
        """)
        self.layout.addStretch()

        self.speech_status_label = QLabel("Speech: OFF")
        self.speech_status_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18px;
                background-color: rgba(0, 0, 0, 128);
                border-radius: 10px;
                padding: 6px 14px;
            }
        """)
        self.layout.addWidget(self.speech_status_label, alignment=Qt.AlignBottom | Qt.AlignHCenter)

        self.transcript_label = QLabel("Transcript: ")
        self.transcript_label.setWordWrap(True)
        self.transcript_label.setMaximumWidth(700)
        self.transcript_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 16px;
                background-color: rgba(0, 0, 0, 128);
                border-radius: 10px;
                padding: 6px 14px;
            }
        """)
        self.layout.addWidget(self.transcript_label, alignment=Qt.AlignBottom | Qt.AlignHCenter)
        self.layout.addWidget(self.gesture_label, alignment=Qt.AlignBottom | Qt.AlignHCenter)

        # --- Timer for frames ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # --- Show window and make click-through ---
        self.show()
        self.set_clickthrough_windows()
    

   

    def add_image_to_cell(self, row, col, image_path, size=(120, 80)):
        label = QLabel()
        pixmap = QPixmap(image_path)

        if not pixmap.isNull():
            label.setPixmap(
                pixmap.scaled(
                    size[0],
                    size[1],
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("background: transparent;")
        self.gesture_table.setCellWidget(row, col, label)
    # ----------------------------
    # Windows click-through / keyboard-through
    # ----------------------------
    def set_clickthrough_windows(self):
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        hwnd = self.winId().__int__()
        user32 = ctypes.windll.user32
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT)

    # ----------------------------
    # App-driven mode (single camera read in app.py; overlay just displays)
    # ----------------------------
    def set_app_driven(self, on):
        self._app_driven = bool(on)
        if self._app_driven:
            self.timer.stop()

    def set_camera_frame(self, frame):
        """Set the frame to display (used when app_driven)."""
        self._last_frame = frame

    def set_gesture_label(self, text):
        """Set the gesture label (used when app_driven)."""
        self._last_gesture_label = text

    def set_speech_controller(self, controller):
        self._speech_controller = controller

    # ----------------------------
    # Toggle overlay/table
    # ----------------------------
    def toggle_overlay(self):
        self.overlay_visible = not self.overlay_visible
        self.setVisible(self.overlay_visible)
        if not self.overlay_visible:
            self.gesture_table.hide()

    def toggle_gesture_table(self):
        self.table_visible = not self.table_visible
        if self.overlay_visible and self.table_visible:
            self.gesture_table.show()
            self.gesture_table.resizeRowsToContents()
        else:
            self.gesture_table.hide()

    # ----------------------------
    # Update camera frame
    # ----------------------------
    def update_frame(self):
        if self._app_driven:
            # Frame and label are set by app; just refresh display from last set values
            frame = self._last_frame
            text = self._last_gesture_label if self._last_gesture_label is not None else "Gesture: (awaiting...)"
        else:
            frame = self.detector.get_frame()
            text = self.detector.get_gesture_label()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(
                QPixmap.fromImage(image).scaled(
                    self.camera_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
        if text is not None:
            self.gesture_label.setText(text)
        if self._speech_controller is not None:
            snapshot = self._speech_controller.get_snapshot()
            self.speech_status_label.setText(snapshot["status"])
            transcript = snapshot["transcript"] if snapshot["transcript"] else "(awaiting speech)"
            self.transcript_label.setText(f"Transcript: {transcript}")
            self.transcript_label.setVisible(snapshot["show_transcript"])

    # ----------------------------
    # Paint translucent overlay
    # ----------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 10, 10, 80))

    # ----------------------------
    # Cleanup
    # ----------------------------
    def closeEvent(self, event):
        if hasattr(self, "timer"):
            self.timer.stop()
        self.detector.release()
        event.accept()
