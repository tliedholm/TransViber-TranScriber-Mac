"""TransViber TransScriber — entry point (macOS)."""

import multiprocessing
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QObject

from models import audio_paths


class FileOpenFilter(QObject):
    """Feed macOS 'Open With' / drag-onto-Dock files to the window.

    Finder does not pass a running app files on the command line: it sends
    QFileOpenEvent to the application object. Without this handler the bundle
    advertised audio document types in Info.plist and then ignored every file
    opened through them.
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.FileOpen:
            # add_files skips anything that isn't a file.
            paths = audio_paths([event.file()]) or audio_paths([event.url().toLocalFile()])
            if paths:
                self.window.add_files(paths)
                self.window.raise_()
                self.window.activateWindow()
                return True
        return super().eventFilter(obj, event)


def main() -> int:
    # Must run before anything can spawn a worker process. In the frozen .app
    # a multiprocessing child re-executes this same binary, so without this
    # the child starts a second copy of the whole GUI instead of the worker.
    multiprocessing.freeze_support()
    # The rest of the app only after it: a child that re-executes this
    # binary stops at freeze_support(), and has no use for the GUI's imports.
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("TransViber TransScriber")
    app.setOrganizationName("TransViber")
    icon_path = Path(__file__).parent / "assets" / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))  # shown in the Dock on macOS
    window = MainWindow()
    # Files opened before the window existed: argv_emulation in the spec turns
    # a Finder "Open With" at launch into ordinary arguments.
    startup_files = [path for path in audio_paths(app.arguments()[1:]) if path.is_file()]
    file_filter = FileOpenFilter(window)
    app.installEventFilter(file_filter)
    window.show()
    if startup_files:
        window.add_files(startup_files)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
