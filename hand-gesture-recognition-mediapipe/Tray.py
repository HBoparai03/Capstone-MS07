# tray_icon.py
import threading
import pystray
from PIL import Image, ImageDraw

class TrayIcon:
    def __init__(self, window):
        self.window = window
        self.icon = pystray.Icon("Gesture Overlay", self._create_icon(), "Gesture Overlay", menu=pystray.Menu(
            pystray.MenuItem("Show/Hide Overlay", self.toggle_visibility),
            pystray.MenuItem("Instructions", self.show_instructions ),
            pystray.MenuItem("Toggle Mouse", self.toggle_mouse),
            pystray.MenuItem("Toggle Left/Right Mouse Control", self.toggle_left_right),
            pystray.MenuItem("Exit", self.exit_app)
        ))

    def _create_icon(self):
        # Generate a simple circular icon
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
        # If overlay is not visible, do nothing (or optionally show overlay)
        if not self.window.isVisible():
             return

        # Toggle the instruction table
        self.window.toggle_gesture_table()

    def exit_app(self, icon, item):
        self.window.close()
        icon.stop()

    def toggle_left_right(self, icon, item):
        return

    def toggle_mouse(self, icon, item):
        return

    def run(self):
        threading.Thread(target=self.icon.run, daemon=True).start()
