import os
import sys


def _register_dll_directories():
    if sys.platform != "win32":
        return

    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return

    candidates = [
        meipass,
        os.path.join(meipass, "mediapipe", "python"),
        os.path.join(meipass, "ctranslate2"),
        os.path.join(meipass, "av"),
        os.path.join(meipass, "cv2"),
        os.path.join(meipass, "numpy.libs"),
        os.path.join(meipass, "scipy.libs"),
        os.path.join(meipass, "PyQt5", "Qt5", "bin"),
    ]

    handles = getattr(sys, "_dll_directory_handles", [])
    path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []

    for dll_dir in candidates:
        if not os.path.isdir(dll_dir):
            continue

        if dll_dir not in path_entries:
            path_entries.insert(0, dll_dir)

        try:
            handles.append(os.add_dll_directory(dll_dir))
        except (AttributeError, FileNotFoundError, OSError):
            pass

    os.environ["PATH"] = os.pathsep.join(path_entries)
    sys._dll_directory_handles = handles


_register_dll_directories()
