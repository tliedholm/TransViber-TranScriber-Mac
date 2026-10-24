"""The window around export and closing, driven offscreen.

Export warns about every track that isn't fully transcribed and reports a
failed write as far as it got; a close never loses notes it could not
write, waits for each job's finish handler rather than its thread, and asks
once when the final save fails.
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt, QThread, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import llm
import main_window
import transcriber
from models import Segment, Track, TrackStatus

_app = QApplication.instance() or QApplication([])
_app.setQuitOnLastWindowClosed(False)
QCoreApplication.setOrganizationName("TransViber-tests")
QCoreApplication.setApplicationName("TransViber TransScriber tests")
QSettings.setDefaultFormat(QSettings.Format.IniFormat)


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        QTest.qWait(10)


def saved_notes():
    return QSettings().value("notes", "", str)


@pytest.fixture
def window(tmp_path):
    """A shown window whose settings live in a temp folder."""
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(tmp_path / "settings"))
    w = main_window.MainWindow()
    w.show()
    yield w
    # Let real jobs finish and their handlers run, drop stand-ins, and let
    # notes a test left unsaved go rather than hold on a prompt nobody will
    # answer. Deleted now: collecting a closed MainWindow later can crash
    # PySide6 partway through another test.
    for job in (w.worker, w.notes_worker):
        if isinstance(job, QThread):
            job.stop()
            job.wait(5000)
    QTest.qWait(50)
    w.worker = w.notes_worker = None
    w._ask_about_unsaved_notes = lambda: "discard"
    w.close()
    w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def done_track(name="Alice", text="Hello there."):
    return Track(Path(f"{name.lower()}.wav"), name, status=TrackStatus.DONE,
                 segments=[Segment(name, 1.0, 2.0, text)])


# ---------- Export ----------

@pytest.fixture
def export_ui(window, monkeypatch):
    """The window with one finished track, and the export dialogs stubbed."""
    ui = Mock()
    ui.warning = Mock(return_value=main_window.QMessageBox.StandardButton.Cancel)
    ui.save_dialog = Mock(return_value=("", ""))
    ui.critical = Mock()
    monkeypatch.setattr(main_window.QMessageBox, "warning", ui.warning)
    monkeypatch.setattr(main_window.QMessageBox, "critical", ui.critical)
    monkeypatch.setattr(main_window.QFileDialog, "getSaveFileName", ui.save_dialog)
    window.tracks.append(done_track())
    return ui


def warned_about(ui):
    """The warning's text, or None if export went straight to the dialog."""
    return ui.warning.call_args.args[2] if ui.warning.called else None


@pytest.mark.parametrize("status, described", [
    (TrackStatus.ERROR, "failed — nothing transcribed"),
    (TrackStatus.CANCELLED, "cancelled — nothing transcribed"),
    (TrackStatus.PENDING, "not transcribed"),
])
def test_a_track_with_no_transcript_is_warned_about(window, export_ui, status, described):
    window.tracks.append(Track(Path("bob.wav"), "Bob", status=status))
    window.export_dialog()
    assert f"bob.wav — {described}" in warned_about(export_ui)
    assert "alice.wav" not in warned_about(export_ui)
    export_ui.save_dialog.assert_not_called()  # Cancel is the default


@pytest.mark.parametrize("status, described", [
    (TrackStatus.ERROR, "failed partway — its partial transcript is left out"),
    (TrackStatus.CANCELLED, "cancelled partway — partial transcript"),
])
def test_a_partial_track_is_still_warned_about(window, export_ui, status, described):
    window.tracks.append(Track(Path("bob.wav"), "Bob", status=status,
                               segments=[Segment("Bob", 3.0, 4.0, "And then—")]))
    window.export_dialog()
    assert f"bob.wav — {described}" in warned_about(export_ui)


def test_a_finished_silent_track_is_not_a_gap(window, export_ui):
    window.tracks.append(Track(Path("bob.wav"), "Bob", status=TrackStatus.DONE))
    window.export_dialog()
    assert warned_about(export_ui) is None
    export_ui.save_dialog.assert_called_once()


def test_a_long_list_is_counted_in_full(window, export_ui):
    window.tracks.extend(Track(Path(f"guest{n}.wav"), f"Guest {n}") for n in range(7))
    window.export_dialog()
    text = warned_about(export_ui)
    assert "7 track(s) not fully transcribed" in text
    assert "guest4.wav" in text
    assert "guest5.wav" not in text
    assert "…and 2 more" in text


def test_continuing_exports_what_there_is_and_every_tracks_status(
        window, export_ui, tmp_path):
    export_ui.warning.return_value = main_window.QMessageBox.StandardButton.Yes
    window.tracks.append(Track(Path("bob.wav"), "Bob", status=TrackStatus.ERROR,
                               error="Model load failed"))
    target = tmp_path / "out.json"
    export_ui.save_dialog.return_value = (str(target), "JSON (*.json)")
    window.export_dialog()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert [(t["speaker"], t["status"]) for t in data["tracks"]] == [("Alice", "Done")]
    assert [(t["speaker"], t["status"]) for t in data["excluded_tracks"]] == [
        ("Bob", "Error")]
    assert [s["text"] for s in data["segments"]] == ["Hello there."]


def replace_failing_on(monkeypatch, call):
    real = os.replace
    calls = []

    def replace(source, target):
        calls.append(target)
        if len(calls) == call:
            raise PermissionError(13, "Permission denied")
        return real(source, target)
    monkeypatch.setattr(main_window.exporters.os, "replace", replace)


def test_a_partial_all_formats_export_names_what_was_written(
        window, export_ui, monkeypatch, tmp_path):
    all_formats = main_window.EXPORT_FILTERS.split(";;")[-1]
    export_ui.save_dialog.return_value = (str(tmp_path / "out.md"), all_formats)
    replace_failing_on(monkeypatch, 2)
    window.export_dialog()
    title, message = export_ui.critical.call_args.args[1:3]
    assert title == "Export incomplete"
    assert "Exported out.md, but could not write out.txt" in message
    assert window.statusBar().currentMessage() == "Partly exported: out.md"


def test_a_failed_single_export_says_nothing_was_written(
        window, export_ui, monkeypatch, tmp_path):
    export_ui.save_dialog.return_value = (str(tmp_path / "out.srt"),
                                          "SubRip subtitles (*.srt)")
    replace_failing_on(monkeypatch, 1)
    window.export_dialog()
    title, message = export_ui.critical.call_args.args[1:3]
    assert title == "Export failed"
    assert "Nothing was exported" in message
    assert list(tmp_path.glob("out*")) == []


def test_a_failed_notes_export_keeps_the_earlier_file(
        window, export_ui, monkeypatch, tmp_path):
    target = tmp_path / "notes-export.md"
    target.write_text("Last week's notes", encoding="utf-8")
    window.notes_edit.setPlainText("This week's notes")
    export_ui.save_dialog.return_value = (str(target), "Markdown (*.md)")
    replace_failing_on(monkeypatch, 1)
    window.export_notes_dialog()
    assert export_ui.critical.call_args.args[1] == "Export failed"
    assert target.read_text(encoding="utf-8") == "Last week's notes"


# ---------- Notes on close ----------

class _DiskFull(QSettings):
    """Settings whose notes write is refused, as a full disk would refuse it."""

    def setValue(self, key, value):
        if key != "notes":
            super().setValue(key, value)

    def status(self):
        return QSettings.Status.AccessError


class _StreamingNotes(llm.NotesWorker):
    """Streams a draft, then, once stopped, the cancelled marker: a real run's
    last chunks, sent from its own thread."""

    def run(self):
        self.chunk.emit("Draft notes.")
        deadline = time.monotonic() + 5.0
        while not self._stop and time.monotonic() < deadline:
            time.sleep(0.01)
        self.chunk.emit("\n\n*(cancelled)*")
        self.chunk.emit("\n")
        self.completed.emit()


class _DyingNotes(llm.NotesWorker):
    """A run that stops without ever sending `completed` or `failed`."""

    def run(self):
        self.chunk.emit("Half a draft.")
        deadline = time.monotonic() + 5.0
        while not self._stop and time.monotonic() < deadline:
            time.sleep(0.01)


@pytest.fixture
def notes(window, monkeypatch):
    """Saved notes to start from, and the unsaved-notes prompt scripted."""
    settings = QSettings()
    settings.setValue("notes", "Old notes")
    settings.sync()
    window._load_settings()
    helpers = Mock()
    helpers.prompt = Mock(return_value="return")
    monkeypatch.setattr(main_window.MainWindow, "_ask_about_unsaved_notes", helpers.prompt)

    def disk_full():
        monkeypatch.setattr(main_window, "QSettings", _DiskFull)

    def disk_ok():
        monkeypatch.setattr(main_window, "QSettings", QSettings)

    def start(worker_class=_StreamingNotes):
        """AI Notes running on a transcript, streaming from its own thread."""
        window.tracks[:] = [done_track("Bob")]
        monkeypatch.setattr(main_window, "NotesWorker", worker_class)
        window.toggle_ai_notes()
        wait_until(lambda: "raft" in window.notes_edit.toPlainText())

    def close():
        event = QCloseEvent()
        window.closeEvent(event)
        return event.isAccepted()

    helpers.disk_full = disk_full
    helpers.disk_ok = disk_ok
    helpers.start = start
    helpers.close = close
    return helpers


def test_loading_saved_notes_does_not_schedule_a_rewrite(window, notes):
    assert window.notes_edit.toPlainText() == "Old notes"
    assert not window._notes_dirty
    assert not window._notes_save_timer.isActive()


def test_a_failed_save_is_visible_and_stays_unsaved(window, notes):
    window.notes_edit.setPlainText("Notes")
    notes.disk_full()
    assert window._save_notes() is False
    assert "Notes could not be saved" in window.statusBar().currentMessage()
    assert window._notes_dirty


def test_a_failed_final_save_keeps_the_window_and_the_notes(window, notes):
    window.notes_edit.setPlainText("Unsaved notes")
    notes.disk_full()
    assert not notes.close()
    notes.prompt.assert_called_once()
    assert "could not be written" in window._notes_save_error
    assert window.notes_edit.toPlainText() == "Unsaved notes"
    assert window._notes_dirty
    notes.disk_ok()
    assert saved_notes() == "Old notes"


def test_retry_writes_the_notes_then_closes(window, notes):
    window.notes_edit.setPlainText("Unsaved notes")
    notes.disk_full()

    def prompt():
        if notes.prompt.call_count >= 2:   # space is freed before the second retry
            notes.disk_ok()
        return "retry"
    notes.prompt.side_effect = prompt
    assert notes.close()
    assert notes.prompt.call_count == 2    # a retry that fails asks again
    assert saved_notes() == "Unsaved notes"
    assert not window._notes_dirty


def test_closing_without_saving_is_the_users_choice(window, notes):
    window.notes_edit.setPlainText("Unsaved notes")
    notes.prompt.return_value = "discard"
    notes.disk_full()
    assert notes.close()
    notes.disk_ok()
    assert saved_notes() == "Old notes"


def test_saving_elsewhere_asks_again_until_a_file_is_written(
        window, notes, monkeypatch, tmp_path):
    window.notes_edit.appendPlainText("Typed since.")
    target = tmp_path / "rescued.md"
    notes.prompt.return_value = "elsewhere"
    answers = iter([("", ""), (str(target), "Markdown (*.md)")])   # cancelled, then a file
    monkeypatch.setattr(main_window.QFileDialog, "getSaveFileName",
                        lambda *args: next(answers))
    notes.disk_full()
    assert notes.close()
    assert notes.prompt.call_count == 2
    assert "Typed since." in target.read_text(encoding="utf-8")


def test_untouched_notes_close_without_a_write(window, notes, monkeypatch):
    written = []

    class Spy(QSettings):
        def setValue(self, key, value):
            written.append(key)
            super().setValue(key, value)
    monkeypatch.setattr(main_window, "QSettings", Spy)
    assert notes.close()
    assert "notes" not in written
    notes.prompt.assert_not_called()


def test_the_last_notes_queued_behind_a_stopped_run_are_saved(window, notes):
    notes.start()
    assert not notes.close()
    # The thread is over; its last chunks and `completed` are still queued.
    assert window.notes_worker is not None and window.notes_worker.isFinished()
    assert not notes.close()
    assert "*(cancelled)*" not in window.notes_edit.toPlainText()
    wait_until(lambda: not window.isVisible())   # delivered, then the deferred close
    assert not window._notes_save_timer.isActive()   # the close saved them itself
    saved = saved_notes()
    assert "Draft notes." in saved
    assert saved.rstrip().endswith("*(cancelled)*")
    notes.prompt.assert_not_called()


def test_return_to_app_puts_everything_back(window, notes):
    notes.start()
    notes.disk_full()
    window.close()
    wait_until(lambda: notes.prompt.called and not window._quit_requested)
    assert window.isVisible()
    assert window.stats_poller.isRunning()   # still polling
    assert window.notes_worker is None
    assert window.btn_ai_notes.text() == "AI Notes"
    assert window.btn_ai_notes.isEnabled()
    assert window.btn_transcribe.isEnabled()
    assert not window._notes_elapsed_timer.isActive()
    assert "*(cancelled)*" in window.notes_edit.toPlainText()
    assert "not saved" in window.statusBar().currentMessage()
    window.notes_edit.appendPlainText("More.")
    assert window._notes_save_timer.isActive()   # autosave is back
    # And the next close, with room on the disk again, saves and goes.
    notes.disk_ok()
    window.close()
    assert not window.isVisible()
    assert not window.stats_poller.isRunning()
    assert saved_notes().endswith("More.")
    notes.prompt.assert_called_once()


def test_one_prompt_however_often_close_is_asked_again(window, notes):
    notes.start()

    def prompt():
        assert not notes.close()   # another Quit, under the prompt
        QTest.qWait(100)
        return "return"
    notes.prompt.side_effect = prompt
    notes.disk_full()
    window.close()
    wait_until(lambda: notes.prompt.called and not window._quit_requested)
    QTest.qWait(100)
    notes.prompt.assert_called_once()


def test_a_run_that_dies_without_finishing_still_lets_the_quit_complete(window, notes):
    notes.start(_DyingNotes)
    window.close()
    wait_until(lambda: not window.isVisible())
    assert window.notes_worker is None
    assert window.btn_ai_notes.text() == "AI Notes"
    assert "Half a draft." in saved_notes()


def test_a_transcription_that_dies_without_finishing_resets_the_controls(
        window, monkeypatch):
    class DyingTranscription(transcriber.TranscribeWorker):
        def run(self):
            pass
    monkeypatch.setattr(main_window, "TranscribeWorker", DyingTranscription)
    monkeypatch.setattr(main_window.llm, "free_llm", lambda: None)
    track = Track(Path("bob.wav"), "Bob")
    window.tracks.append(track)
    window._append_row(track)
    window.toggle_transcription()
    wait_until(lambda: window.worker is None)
    assert window.btn_transcribe.text() == "Transcribe"
    assert window.btn_add.isEnabled()


# ---------- A job that will not stop ----------

def press_on_exec(monkeypatch, label):
    """Answer the next QMessageBox.exec() by clicking the button `label`."""
    def exec_(box):
        next(b for b in box.buttons() if b.text() == label).click()
        return 0
    monkeypatch.setattr(main_window.QMessageBox, "exec", exec_)


def stuck_worker():
    worker = Mock()
    worker.wait.return_value = False   # still inside an uninterruptible step
    return worker


def test_waiting_for_a_stuck_step_closes_once_its_handler_runs(window, monkeypatch):
    window.worker = stuck_worker()
    press_on_exec(monkeypatch, "Wait for it")
    assert not window.close()
    window.worker.stop.assert_called_once()
    assert window.isVisible()
    window._on_finished(True)    # the step ends; its handler runs
    wait_until(lambda: not window.isVisible())


def test_quit_now_asks_about_notes_it_could_not_save(window, notes, monkeypatch):
    exit_ = Mock()
    monkeypatch.setattr(main_window.os, "_exit", exit_)
    window.worker = stuck_worker()
    window.notes_edit.setPlainText("Unsaved notes")
    notes.disk_full()
    press_on_exec(monkeypatch, "Quit now")
    assert not notes.close()
    exit_.assert_not_called()          # Return to App: the quit is called off
    assert not window._quit_requested
    notes.prompt.return_value = "discard"
    notes.close()
    exit_.assert_called_once_with(0)
    window.worker = None


# ---------- The prompt itself ----------

def answer(window, button):
    """Open the real prompt and press `button` (a label, or Escape).
    Returns the choice and the buttons it offered."""
    offered = []

    def press():
        box = QApplication.activeModalWidget()
        offered.extend(b.text() for b in box.buttons())
        if button == "Escape":
            QTest.keyClick(box, Qt.Key.Key_Escape)
        else:
            next(b for b in box.buttons() if b.text() == button).click()
    QTimer.singleShot(0, press)
    return main_window.MainWindow._ask_about_unsaved_notes(window), offered


@pytest.mark.parametrize("button, choice", [
    ("Retry", "retry"), ("Save Elsewhere…", "elsewhere"),
    ("Return to App", "return"), ("Close Without Saving", "discard"),
    ("Escape", "return"),
])
def test_the_prompt_buttons(window, button, choice):
    window._notes_save_error = "The settings file could not be written: x.plist"
    answered, offered = answer(window, button)
    assert answered == choice
    assert offered[0] == "Retry"
