# main.py
import sys
from PyQt5.QtWidgets import QApplication
from Overlay import OverlayWindow
from Tray import TrayIcon

def main():
    app = QApplication(sys.argv)

    overlay = OverlayWindow()
    overlay.show()

    tray = TrayIcon(overlay)
    tray.run()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
