"""Notes models too big for this Mac's GPU, in the window, driven offscreen.

They are greyed out in the menu, and AI Notes refuses one that was saved or
typed before any worker starts. The GPU budget is faked, so these run the
same on any machine.
"""

import os
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtWidgets import QApplication

import llm
import main_window
from models import Segment, Track, TrackStatus

_app = QApplication.instance() or QApplication([])
_app.setQuitOnLastWindowClosed(False)
QCoreApplication.setOrganizationName("TransViber-tests")
QCoreApplication.setApplicationName("TransViber TransScriber tests")
QSettings.setDefaultFormat(QSettings.Format.IniFormat)

M1_PRO_16GB = 12_713_115_648


def build_window(tmp_path, monkeypatch, budget):
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(tmp_path / "settings"))
    monkeypatch.setattr(llm, "gpu_budget", lambda: budget)
    return main_window.MainWindow()


@pytest.fixture
def windows():
    """Deletes every window a test builds: collecting a closed MainWindow
    later can crash PySide6 partway through another test."""
    built = []
    yield built
    for w in built:
        w.close()
        w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def menu_rows(window):
    combo = window.combo_notes_model
    items = combo.model()
    return {combo.itemText(row): items.item(row) for row in range(combo.count())
            if combo.itemText(row) in llm.LLM_REPOS}


def test_gpt_oss_is_greyed_out_on_a_16_gb_mac(tmp_path, monkeypatch, windows):
    windows.append(w := build_window(tmp_path, monkeypatch, M1_PRO_16GB))
    rows = menu_rows(w)
    assert not rows["gpt-oss-20b"].isEnabled()
    assert "24 GB" in rows["gpt-oss-20b"].toolTip()
    assert all(item.isEnabled() and not item.toolTip()
               for name, item in rows.items() if name != "gpt-oss-20b")


@pytest.mark.parametrize("budget", [None, 17_179_869_184])
def test_every_model_is_offered_where_it_fits_or_mlx_cannot_say(
        tmp_path, monkeypatch, windows, budget):
    windows.append(w := build_window(tmp_path, monkeypatch, budget))
    assert all(item.isEnabled() for item in menu_rows(w).values())


def test_a_saved_choice_that_cannot_fit_is_kept_but_refused(tmp_path, monkeypatch, windows):
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(tmp_path / "settings"))
    QSettings().setValue("notes_model", "gpt-oss-20b")
    windows.append(w := build_window(tmp_path, monkeypatch, M1_PRO_16GB))
    assert w.combo_notes_model.currentText() == "gpt-oss-20b"  # not swapped

    w.tracks.append(Track(Path("alice.wav"), "Alice", status=TrackStatus.DONE,
                          segments=[Segment("Alice", 1.0, 2.0, "Hello there.")]))
    warning = Mock()
    monkeypatch.setattr(main_window.QMessageBox, "warning", warning)
    monkeypatch.setattr(main_window, "NotesWorker",
                        Mock(side_effect=AssertionError("a worker was started")))
    w.toggle_ai_notes()
    warning.assert_called_once()
    assert warning.call_args.args[2] == llm.gpu_shortfall("gpt-oss-20b", M1_PRO_16GB)
    assert w.notes_worker is None
    assert w.btn_transcribe.isEnabled()
    assert w.notes_edit.toPlainText() == ""
