"""Background poller for the system-utilization readout in the status bar."""

from __future__ import annotations

import re
import subprocess
import sys

import psutil
from PySide6.QtCore import QThread, Signal

POLL_MS = 2000
# The poll interval is slept in slices so stop() is noticed within one slice
# instead of up to a full interval later — shutdown waits on this thread.
SLEEP_SLICE_MS = 100
# ioreg is a local registry read; if it has not answered in a second it is not
# going to, and a longer timeout only delays quitting.
IOREG_TIMEOUT_S = 1.0

_UTIL_RE = re.compile(rb'"Device Utilization %"=(\d+)')


def gpu_utilization() -> int | None:
    """Apple-GPU busy percentage from the IORegistry (no privileges needed).

    Reads the IOAccelerator PerformanceStatistics that Activity Monitor's
    GPU history also uses. Whole-system figure, not per-process.
    """
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            capture_output=True, timeout=IOREG_TIMEOUT_S,
        ).stdout
    except Exception:
        return None
    m = _UTIL_RE.search(out)
    return int(m.group(1)) if m else None


class StatsPoller(QThread):
    """Emits a 'GPU 94% · CPU 21% · RAM 51%' string every couple of seconds.

    Runs in its own thread because the ioreg subprocess takes tens of
    milliseconds — long enough to stutter the UI if polled from a QTimer
    on the main thread.
    """

    stats = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _sleep_interruptibly(self, total_ms: int) -> None:
        slept = 0
        while slept < total_ms and not self._stop:
            self.msleep(min(SLEEP_SLICE_MS, total_ms - slept))
            slept += SLEEP_SLICE_MS

    def run(self) -> None:
        psutil.cpu_percent(interval=None)  # prime the since-last-call counter
        while not self._stop:
            parts = []
            gpu = gpu_utilization()
            if gpu is not None:
                parts.append(f"GPU {gpu}%")
            parts.append(f"CPU {psutil.cpu_percent(interval=None):.0f}%")
            parts.append(f"RAM {psutil.virtual_memory().percent:.0f}%")
            self.stats.emit(" · ".join(parts))
            self._sleep_interruptibly(POLL_MS)
