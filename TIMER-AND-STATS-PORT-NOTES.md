# Porting the elapsed timer + system-utilization readout to the Windows version

Two small UI features from the Mac app (`app/main_window.py`, `app/stats.py`),
written for the Windows/CUDA sibling codebase. Both are pure UI additions —
no changes to TranscribeWorker or the engines.

---

## Feature 1: Elapsed-time display next to the Transcribe button

**Behavior**: a label to the left of the Transcribe button starts at `0:00`
when a run starts, ticks every second while transcribing (`m:ss`, switching
to `h:mm:ss` past an hour), and freezes at `took 4:12` when the run ends
(finish, error, or cancel). The final status-bar message also becomes
"Transcription complete in 4:12". The frozen value persists until the next
run. Timing covers the whole run including model load/download.

**Implementation** — all in `main_window.py`, fully portable Qt (no
platform-specific code):

Toolbar construction (after `bar.addStretch()`, before the Transcribe button):

```python
# Elapsed time of the running (or last finished) transcription.
self.elapsed_label = QLabel("")
self.elapsed_label.setToolTip("How long the current or last transcription run took.")
bar.addWidget(self.elapsed_label)
self._run_started: float | None = None
self._elapsed_timer = QTimer(self)
self._elapsed_timer.setInterval(1000)
self._elapsed_timer.timeout.connect(self._tick_elapsed)
```

Where the worker starts (`toggle_transcription`, just before `w.start()`):

```python
self._run_started = time.monotonic()
self.elapsed_label.setText("0:00")
self._elapsed_timer.start()
```

Helpers (note `time.monotonic()`, not `time.time()` — immune to clock
changes/sleep weirdness):

```python
@staticmethod
def _format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def _tick_elapsed(self) -> None:
    if self._run_started is not None:
        self.elapsed_label.setText(self._format_elapsed(time.monotonic() - self._run_started))
```

At the top of `_on_finished` (before the status message is chosen):

```python
self._elapsed_timer.stop()
took = ""
if self._run_started is not None:
    elapsed = self._format_elapsed(time.monotonic() - self._run_started)
    self.elapsed_label.setText(f"took {elapsed}")
    self._run_started = None
    took = f" in {elapsed}"
...
else:
    msg = f"Transcription complete{took}"
```

(`import time` at the top of the file.)

---

## Feature 2: Live system-utilization readout in the status bar

**Behavior**: a permanent status-bar label next to the device label reading
e.g. `GPU 94% · CPU 21% · RAM 51%`, refreshed every 2 s, always on. Polling
runs in a background QThread because the GPU query takes tens of
milliseconds — long enough to stutter the UI from a main-thread QTimer.

**Structure**: `app/stats.py` holds a `StatsPoller(QThread)` with a
`stats = Signal(str)`; `main_window.py` connects it to a QLabel and
stops it in `closeEvent`.

`stats.py` (Mac version, verbatim):

```python
"""Background poller for the system-utilization readout in the status bar."""

from __future__ import annotations

import re
import subprocess
import sys

import psutil
from PySide6.QtCore import QThread, Signal

POLL_MS = 2000

_UTIL_RE = re.compile(rb'"Device Utilization %"=(\d+)')


def gpu_utilization() -> int | None:
    """Apple-GPU busy percentage from the IORegistry (no privileges needed)."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            capture_output=True, timeout=5,
        ).stdout
    except Exception:
        return None
    m = _UTIL_RE.search(out)
    return int(m.group(1)) if m else None


class StatsPoller(QThread):
    stats = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

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
            self.msleep(POLL_MS)
```

`main_window.py` wiring (status-bar construction):

```python
self.stats_label = QLabel("")
self.stats_label.setToolTip(...)
self.statusBar().addPermanentWidget(self.stats_label)   # before device_label
...
self.stats_poller = StatsPoller(self)
self.stats_poller.stats.connect(self.stats_label.setText)
self.stats_poller.start()
```

And in `closeEvent`, after stopping the worker:

```python
self.stats_poller.stop()
self.stats_poller.wait(3000)
```

**New dependency**: `psutil` (added to the setup script's pip install line).
CPU note: `psutil.cpu_percent(interval=None)` measures since the previous
call — call it once at thread start to prime it, or the first reading is 0.

### What Windows must replace: `gpu_utilization()`

The `ioreg` call is macOS-only (reads the IORegistry's IOAccelerator
"Device Utilization %" — whole-system GPU busy, no admin rights). For an
NVIDIA/CUDA machine, replace its body with NVML — everything else is
portable:

- **Preferred**: `pynvml` (pip package, ships with recent
  `nvidia-ml-py`): `nvmlDeviceGetUtilizationRates(handle).gpu` → integer
  percent. Init once (`nvmlInit`), not per poll.
- **Zero-dependency fallback**: parse
  `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits`
  (subprocess, like the Mac version does with ioreg — but slower, ~100 ms).
- Keep the `return None` on any failure — the label then simply shows
  CPU/RAM only, which is the intended degraded mode.
- On Windows, also consider adding VRAM instead of/next to RAM
  (`nvmlDeviceGetMemoryInfo`) — on a discrete-GPU machine the model lives
  in VRAM, so that's the "can I fit a bigger model" number there (on the
  Mac, unified memory means plain RAM already covers it).

### UX notes learned on the Mac side (apply as caveats, not code)

- The GPU number is **whole-system**, not per-process — tell the user, or
  they'll wonder why it's nonzero at idle.
- Utilization only reflects the device the engine actually uses: on the Mac,
  Parakeet/Canary run on CoreML (Neural Engine/CPU) and the GPU number stays
  low, which confused the user. On Windows/CUDA everything runs on the GPU,
  so this mostly disappears — but CPU-device runs will show the same effect.
- RAM percent reads high on any modern OS (caches count as occupied);
  expect "why is RAM at 63%?" questions. It is live, not a high-water mark.

### Unrelated Mac finding that may matter to the PC app

macOS's native Qt style never renders QProgressBar text (`setFormat(...)` is
silently dropped), which broke a "percentage inside the bar" design. The
Windows Qt styles DO render bar text, so this doesn't block the PC app —
noted only in case a shared design assumes text-in-bar works everywhere.
