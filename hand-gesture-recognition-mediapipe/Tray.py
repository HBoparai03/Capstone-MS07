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
    def __init__(self, window):
        self.window = window
        # Bridge must be created in main thread so signal delivery runs there
        self._bridge = TrayActionBridge() if _QT_AVAILABLE else None
        if self._bridge is not None:
            self._bridge.actionRequested.connect(self._on_tray_action)

        self.icon = pystray.Icon(
            "Gesture Overlay",
            self._create_icon(),
            "Gesture Overlay",
            menu=pystray.Menu(
                pystray.MenuItem("Show/Hide Overlay", self.toggle_visibility),
                pystray.MenuItem("Instructions", self.show_instructions),
                pystray.MenuItem("Exit", self.exit_app),
            ),
        )

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
            self.window.close()
            self.icon.stop()

    def toggle_visibility(self, icon, item):
        if self._bridge is not None:
            self._bridge.actionRequested.emit("showhide")
        else:
            if self.window.isVisible():
                self.window.hide()
            else:
                self.window.show()

    def show_instructions(self, icon, item):
        if self._bridge is not None:
            self._bridge.actionRequested.emit("instructions")
        else:
            if self.window.isVisible():
                self.window.toggle_gesture_table()

    def exit_app(self, icon, item):
        if self._bridge is not None:
            self._bridge.actionRequested.emit("exit")
        else:
            self.window.close()
            icon.stop()

    def run(self):
        threading.Thread(target=self.icon.run, daemon=True).start()
