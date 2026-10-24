"""The track table's Status cell for a track whose decoding ran into something.

A file whose timestamps jump back, or that had damaged packets skipped, still
transcribes in full, but times after the problem may be off. That shows as
"Done ⚠" with the reason in the tooltip, and it belongs to the transcript it
came with: a fresh run starts without it, and a failed rerun brings it back
with the transcript it restores.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtWidgets import QApplication

import main_window
from models import RunConfig, Segment, Track

_app = QApplication.instance() or QApplication([])

NOTE = "Its timestamps jump back at 0:05."
CONFIG = RunConfig(model="large-v3", language="en", backend="faster-whisper on CPU")


@pytest.fixture
def window(tmp_path, monkeypatch):
    """A real MainWindow with its settings in a scratch file and no stats thread."""
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(main_window, "QSettings", lambda: settings)
    monkeypatch.setattr(main_window.StatsPoller, "start", lambda self: None)
    w = main_window.MainWindow()
    track = Track(tmp_path / "audio.wav", "Alice")
    w.tracks.append(track)
    w._append_row(track)
    yield w
    # Deleted now rather than left to the garbage collector, which has
    # crashed PySide6 6.11 when it collects a closed MainWindow.
    w.close()
    w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def status(w):
    item = w.table.item(0, main_window.COL_STATUS)
    return item.text(), item.toolTip()


def finish(w, note=""):
    w._on_track_started(0)
    if note:
        w._on_track_note(0, note)
    w._on_track_done(0, "en", CONFIG)


def test_a_timing_note_shows_on_the_finished_track(window):
    finish(window, NOTE)
    assert status(window) == ("Done (en) ⚠", f"{CONFIG.summary()}\n\n{NOTE}")
    # A retry (the CPU fallback) starts the track again, note and all.
    finish(window)
    assert status(window) == ("Done (en)", CONFIG.summary())


def test_a_failed_rerun_brings_back_the_note_with_its_transcript(window):
    finish(window, NOTE)
    track = window.tracks[0]
    track.segments.append(Segment("Alice", 0, 1, "hello"))
    track.stash_result()                # what Transcribe does before a rerun
    window._on_track_started(0)
    assert track.note == ""
    window._on_track_failed(0, "Out of memory")
    text, tip = status(window)
    assert text == "Done (en) ⚠"
    assert track.note == NOTE
    assert "the previous transcript was kept" in tip
    assert "Out of memory" in tip and NOTE in tip
