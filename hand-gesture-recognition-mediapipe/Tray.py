# tray_icon.py
import threading
import pystray
import os
import sys
from PIL import Image
#ushergil change

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


try:
    from PyQt5.QtCore import QObject, pyqtSignal
    _QT_AVAILABLE = True
except Exception:
    _QT_AVAILABLE = False
    QObject = object


class TrayActionBridge(QObject):
    """Lives in the Qt main thread. Tray thread emits actionRequested(str); slot runs on main thread."""
    if _QT_AVAILABLE:
        actionRequested = pyqtSignal(str)


class TrayIcon:
    def __init__(self, window, gesture_hand="Right", mouse_enabled=True):
        self.window = window
        self.gesture_hand = gesture_hand
        self.mouse_enabled = mouse_enabled
        self._bridge = None

        self.icon = pystray.Icon("Gesture Overlay", self._create_icon(), "Gesture Overlay", menu=pystray.Menu(
            pystray.MenuItem("Show/Hide Overlay", self.toggle_visibility),
            pystray.MenuItem("Instructions", self.show_instructions),
            pystray.MenuItem(
                lambda item: f"Mouse: {'ON' if self.mouse_enabled else 'OFF'}",
                self.toggle_mouse,
                checked=lambda item: self.mouse_enabled,
            ),
            pystray.MenuItem(
                lambda item: f"Gesture Hand: {self.gesture_hand}  |  Mouse Hand: {self.mouse_hand}",
                self.toggle_left_right,
            ),
            pystray.MenuItem("Exit", self.exit_app)
        ))

    @property
    def mouse_hand(self):
        return "Left" if self.gesture_hand == "Right" else "Right"

    def _create_icon(self):
        try:
            return Image.open(resource_path("icon.ico"))
        except Exception as e:
            # Fallback so tray still appears if icon fails
            img = Image.new("RGB", (64, 64), color="black")
            return img


    def _on_tray_action(self, action):
        """Runs on Qt main thread (connected to bridge signal)."""
        if action == "showhide":
            if self.window.isVisible():
                self.window.hide()
            else:
                self.window.show()
        elif action == "instructions":
            if self.window.isVisible():
                self.window.toggle_gesture_table()
        elif action == "exit":
            self._force_exit()

    def toggle_visibility(self, icon, item):
        if self._bridge is not None:
            self._bridge.actionRequested.emit("showhide")
        else:
            if self.window.isVisible():
                self.window.hide()
            else:
                self.window.show()

    def show_instructions(self, icon, item):
        if not self.window.isVisible():
             return
        self.window.toggle_gesture_table()

    def exit_app(self, icon, item):
        if self._bridge is not None:
            self._bridge.actionRequested.emit("exit")
        else:
            self._force_exit()

    def _force_exit(self):
        """Clean up and forcefully terminate the entire process."""
        try:
            self.icon.stop()
        except Exception:
            pass
        try:
            self.window.close()
        except Exception:
            pass
        try:
            from PyQt5.QtWidgets import QApplication
            q = QApplication.instance()
            if q is not None:
                q.quit()
        except Exception:
            pass
        os._exit(0)

    def toggle_left_right(self, icon, item):
        if self.gesture_hand == "Right":
            self.gesture_hand = "Left"
        else:
            self.gesture_hand = "Right"
        icon.update_menu()

    def toggle_mouse(self, icon, item):
        self.mouse_enabled = not self.mouse_enabled
        icon.update_menu()

    def run(self):
        threading.Thread(target=self.icon.run, daemon=True).start()
