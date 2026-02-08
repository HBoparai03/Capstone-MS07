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
from Capture import GestureDetector


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
        self.gesture_table.setRowCount(5)
        self.gesture_table.setHorizontalHeaderLabels(["Gesture", "Action", "Description"])
        self.gesture_table.verticalHeader().setVisible(False)
        self.gesture_table.horizontalHeader().setStretchLastSection(True)
        self.gesture_table.setWordWrap(True)

        # Allow table and rows to grow vertically
        self.gesture_table.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.gesture_table.setMinimumWidth(600)
        self.gesture_table.setMinimumHeight(400)

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
            ("Open Palm", "Open a new browser tab (Ctrl + T)", "The palm of the user is facing the camera with all fingers extended and clearly visible."),
            ("Fist", "Close the current tab. (Ctrl + W)", "The user's hand will be closed."),
            ("Thumbs Up", "Volume Up (system control)", "The thumb of the user will be upwards, other fingers folded."),
            ("Thumbs Down", "Volume Down (system control)", "The thumb of the user will be downwards, other fingers folded."),
            ("Index Finger Up", "Play media (Spacebar)", "The index finger of the user will be raised, others folded."),
        ]

        for row, (gesture, action, desc) in enumerate(placeholders):
            for col, text in enumerate((gesture, action, desc)):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.gesture_table.setItem(row, col, item)

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
        self.layout.addWidget(self.gesture_label, alignment=Qt.AlignBottom | Qt.AlignHCenter)

        # --- Timer for frames ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # --- Show window and make click-through ---
        self.show()
        self.set_clickthrough_windows()

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
        self.detector.release()
        event.accept()
