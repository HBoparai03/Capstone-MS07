# tray_icon.py
import threading
import pystray
from PIL import Image, ImageDraw

class TrayIcon:
    def __init__(self, window, gesture_hand="Right", mouse_enabled=True):
        self.window = window
        self.gesture_hand = gesture_hand
        self.mouse_enabled = mouse_enabled

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
        image = Image.new('RGB', (64, 64), color='black')
        draw = ImageDraw.Draw(image)
        draw.ellipse((10, 10, 54, 54), fill='green')
        return image

    def toggle_visibility(self, icon, item):
        if self.window.isVisible():
            self.window.hide()
        else:
            self.window.show()

    def show_instructions(self, icon, item):
        if not self.window.isVisible():
             return
        self.window.toggle_gesture_table()

    def exit_app(self, icon, item):
        self.window.close()
        icon.stop()

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
