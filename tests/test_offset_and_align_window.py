"""The Offset column and Align, driven through a real MainWindow (offscreen).

Typed offsets commit through `_OffsetDelegate`, not `itemChanged`: Qt drops a
write of unchanged text, and retyping the shown `0:00` is a deliberate
"starts at zero" that must still count as typed, so a few tests type into the
cell's real editor. Align runs the real AlignWorker thread with its run()
scripted, so its signals cross threads as they do in the app, and plays the
review window by replacing only its exec().
"""

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit

import align
import main_window
from align import AUTOMATIC_REFERENCE, SESSION_REFERENCE, AlignmentResult, Suggestion
from models import OffsetSource, Segment, Track, TrackStatus

_app = QApplication.instance() or QApplication([])


@pytest.fixture
def window(tmp_path, monkeypatch):
    """A real MainWindow with its settings in a scratch file, no stats thread
    and no log writes."""
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(main_window, "QSettings", lambda: settings)
    monkeypatch.setattr(main_window.StatsPoller, "start", lambda self: None)
    monkeypatch.setattr(main_window.diagnostics, "log_event", lambda *args, **fields: None)
    w = main_window.MainWindow()
    yield w
    # Deleted now rather than left to the garbage collector, which has
    # crashed PySide6 6.11 when it collects a closed MainWindow.
    w.close()
    w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def add(w, *tracks):
    """Put tracks straight into the list, as add_files would."""
    for track in tracks:
        w.tracks.append(track)
        w._append_row(track)


def offset_cell(w, row):
    return w.table.item(row, main_window.COL_OFFSET)


def type_into_offset_cell(w, row, keys, enter=True):
    """Open the cell's real editor, type `keys` (None: type nothing), and press
    Enter unless `enter` is False, which leaves the edit open."""
    w.show()
    w.table.editItem(offset_cell(w, row))
    editor = w.table.viewport().findChild(QLineEdit)
    assert editor is not None
    if keys is not None:
        editor.selectAll()
        QTest.keyClicks(editor, keys)
    if enter:
        QTest.keyClick(editor, Qt.Key.Key_Return)
        _app.processEvents()


# ---------- The Offset column ----------

def test_typing_an_offset_reorders_the_transcript(window):
    alice = Track(Path("alice.wav"), "Alice",
                  segments=[Segment("Alice", 180, 181, "Three minutes in.")])
    bob = Track(Path("bob.wav"), "Bob", segments=[Segment("Bob", 60, 61, "Sorry I'm late.")])
    add(window, alice, bob)
    window._render_transcript()
    text = window.transcript_view.toPlainText()
    assert text.index("Sorry") < text.index("Three")

    window._on_offset_edited(1, "5:00")
    assert (bob.offset, bob.offset_source) == (300.0, OffsetSource.MANUAL)
    window._render_transcript()
    text = window.transcript_view.toPlainText()
    assert text.index("Three") < text.index("Sorry")
    assert "[6:00]" in text
    assert bob.segments[0].start == 60  # raw times stay raw


def test_a_bad_offset_is_rejected_and_the_old_value_stays(window):
    bob = Track(Path("bob.wav"), "Bob")
    add(window, bob)
    window._on_offset_edited(0, "2:30")
    window._on_offset_edited(0, "1:75")
    assert bob.offset == 150.0
    assert offset_cell(window, 0).text() == "2:30"
    assert "under 60" in window.statusBar().currentMessage()


def test_the_cell_is_rewritten_in_standard_form(window):
    add(window, Track(Path("bob.wav"), "Bob"))
    assert offset_cell(window, 0).text() == "0:00"
    assert "Not set" in offset_cell(window, 0).toolTip()
    window._on_offset_edited(0, "150,5")
    assert offset_cell(window, 0).text() == "2:30.5"
    assert "Typed by you" in offset_cell(window, 0).toolTip()


def test_a_typed_zero_counts_as_set_by_hand(window):
    bob = Track(Path("bob.wav"), "Bob")
    add(window, bob)
    window._on_offset_edited(0, "0")
    assert (bob.offset, bob.offset_source) == (0.0, OffsetSource.MANUAL)


def test_typing_in_the_real_editor_sets_the_offset(window):
    bob = Track(Path("bob.wav"), "Bob")
    add(window, bob)
    type_into_offset_cell(window, 0, "2:30,5")
    assert bob.offset == 150.5
    assert offset_cell(window, 0).text() == "2:30.5"


def test_retyping_the_shown_zero_counts_as_set_by_hand(window):
    bob = Track(Path("bob.wav"), "Bob")
    add(window, bob)
    type_into_offset_cell(window, 0, "0:00")
    assert bob.offset_source == OffsetSource.MANUAL
    assert "Typed by you" in offset_cell(window, 0).toolTip()


def test_opening_and_closing_a_cell_without_typing_changes_nothing(window):
    bob = Track(Path("bob.wav"), "Bob")
    add(window, bob)
    type_into_offset_cell(window, 0, None)
    assert bob.offset_source == OffsetSource.DEFAULT


def test_an_absurd_offset_is_rejected_and_the_table_keeps_working(window):
    bob = Track(Path("bob.wav"), "Bob")
    add(window, bob)
    window._on_offset_edited(0, "9" * 400)
    assert bob.offset == 0.0
    assert offset_cell(window, 0).text() == "0:00"
    window._on_offset_edited(0, "1:00")
    assert bob.offset == 60.0


def test_renaming_a_speaker_keeps_the_offset(window):
    bob = Track(Path("bob.wav"), "Bob", segments=[Segment("Bob", 1, 2, "Hi.")])
    add(window, bob)
    window._on_offset_edited(0, "1:00")
    window.table.item(0, main_window.COL_SPEAKER).setText("Robert")
    assert bob.offset == 60.0
    window._render_transcript()
    assert window.transcript_view.toPlainText().strip() == "Robert [1:01]: Hi."


def test_splitting_a_track_keeps_its_offset(window):
    add(window, Track(Path("room.wav"), "Room", offset=300.0,
                      offset_source=OffsetSource.MANUAL))
    with patch("main_window.probe_streams", return_value=[{"channels": 2, "duration": 10.0}]):
        window._split_track_channels(0)
    assert len(window.tracks) == 2
    for row, new in enumerate(window.tracks):
        assert (new.offset, new.offset_source) == (300.0, OffsetSource.MANUAL)
        assert offset_cell(window, row).text() == "5:00"


# ---------- Align ----------

def _wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the window")
        QTest.qWait(10)


class Session:
    """A window with stand-in files: Align from it, and what the user is told."""

    def __init__(self, w, folder: Path) -> None:
        self.w, self.folder = w, folder

    def file(self, name):
        path = self.folder / name
        path.touch()
        return path

    def stereo(self, name):
        """A file loaded as two rows, the way a split stereo track is."""
        path = self.file(name)
        add(self.w, *(Track(path, path.stem.title() + suffix, duration=600.0, channel=ch)
                      for ch, suffix in enumerate((" L", " R"))))
        return path

    def align(self, rows, review=None, *, mode=AUTOMATIC_REFERENCE, reference=None,
              note="", result=None, run=None):
        """Press Align with the worker scripted to return `result` (by default
        one made of `rows`), or to `run(worker)`, then play the review window
        with `review(dialog) -> accepted`. Returns what was seen."""
        seen = {"constructed": None, "dialogs": 0}
        rows = list(rows)
        w = self.w

        class Scripted(align.AlignWorker):
            def __init__(inner, *args, **kwargs):
                super().__init__(*args, **kwargs)
                seen["constructed"] = (inner.units, inner.session_reference)
                seen["known"] = inner.known

            def run(inner):
                if run is not None:
                    run(inner)
                    return
                inner.result = result or AlignmentResult(
                    rows, mode, reference or rows[0].path, note)

        def play(dialog):
            # The window deletes the dialog once it closes: keep what it showed.
            seen["dialogs"] += 1
            seen["basis"] = dialog.basis_label.text()
            seen["note"] = dialog.note_label.text()
            seen["cells"] = [[dialog.table.item(row, col).text() for col in range(5)]
                             for row in range(dialog.table.rowCount())]
            seen["locked_in_review"] = [not b.isEnabled() for b in (
                w.btn_add, w.btn_remove, w.btn_clear, w.btn_reference,
                w.btn_remove_reference, w.btn_align)]
            seen["ticks"] = {s.path.name + f"#{i}": dialog.is_ticked(i)
                             for i, s in enumerate(dialog.suggestions)}
            return review(dialog) if review else False

        with patch("main_window.AlignWorker", Scripted), \
                patch.object(main_window.AlignReviewDialog, "exec", play):
            w.toggle_alignment()
            seen["while_running"] = {
                "align_text": w.btn_align.text(),
                "locked": [not b.isEnabled() for b in (
                    w.btn_add, w.btn_remove, w.btn_clear, w.btn_reference,
                    w.btn_remove_reference)],
                "transcribe": w.btn_transcribe.isEnabled(),
                "offset_editable": bool(offset_cell(w, 0).flags() & Qt.ItemFlag.ItemIsEditable)
                if w.tracks else None,
            }
            _wait_until(lambda: w.align_worker is None and not w._align_reviewing)
        return seen

    def automatic(self, measured, anchor=0.0):
        """The result an automatic-mode run hands over, anchored at `anchor`."""
        return align.rebase(AlignmentResult([], AUTOMATIC_REFERENCE, measured[0].path,
                                            measured=measured), anchor)

    @staticmethod
    def run_while(result, during):
        """A worker run that lets `during` happen on the main thread, then finishes."""
        release = threading.Event()

        def slow(worker):
            release.wait(5)
            worker.result = result

        def act():
            during()
            release.set()
        QTimer.singleShot(0, act)
        return slow


@pytest.fixture
def s(window, tmp_path, monkeypatch):
    """Stand-in files: each loads as one 10-minute speaker, and anything picked
    as a session reference probes as an hour of mono. Message boxes are mocks."""
    monkeypatch.setattr(main_window, "tracks_for_file",
                        lambda path: [Track(path, path.stem.title(), duration=600.0)])
    monkeypatch.setattr(main_window, "probe_streams",
                        lambda path: [{"channels": 1, "duration": 3600.0}])
    session = Session(window, tmp_path)
    session.info = Mock()
    session.warning = Mock()
    monkeypatch.setattr(main_window.QMessageBox, "information", session.info)
    monkeypatch.setattr(main_window.QMessageBox, "warning", session.warning)
    return session


def test_a_reference_alone_has_nothing_to_align_or_transcribe(s):
    w = s.w
    assert w.set_session_reference(s.file("session.wav"))
    w.toggle_alignment()
    assert w.align_worker is None
    assert "Nothing to align" in s.info.call_args.args[1]
    w.toggle_transcription()
    assert w.worker is None
    assert s.info.call_args.args[1] == "No tracks"
    assert not w.btn_export.isEnabled()
    assert w.tracks == []


def test_one_speaker_file_and_a_reference_can_align(s):
    w = s.w
    session, bob = s.file("session.wav"), s.file("bob.wav")
    w.set_session_reference(session)
    w.add_files([bob])
    seen = s.align([Suggestion(bob, 900.0, "matched", "Matched sound with session.wav "
                               "(strong match)", "strong")],
                   lambda dialog: True, mode=SESSION_REFERENCE, reference=session)
    assert seen["constructed"] == ([(bob, [None])], (session, [None]))
    assert "Reference: session.wav; session starts at 0:00" in seen["basis"]
    assert (w.tracks[0].offset, w.tracks[0].offset_source) == (900.0, OffsetSource.MATCHED)
    assert offset_cell(w, 0).text() == "15:00"


def test_tracks_from_one_file_have_nothing_to_align(s):
    s.stereo("room.wav")
    s.w.toggle_alignment()
    assert s.w.align_worker is None
    assert "already in step" in s.info.call_args.args[2]


def test_the_reference_cannot_also_be_a_speaker_file(s):
    w = s.w
    session = s.file("session.wav")
    (s.folder / "sub").mkdir()
    (s.folder / "link.wav").symlink_to(session)
    w.set_session_reference(session)
    for other_name in (s.folder / "sub" / ".." / "session.wav", s.folder / "link.wav"):
        w.add_files([other_name])
        assert w.tracks == []
        assert "is the session reference" in s.info.call_args.args[2]


def test_a_speaker_file_cannot_also_be_the_reference(s):
    w = s.w
    session, alice = s.file("session.wav"), s.file("alice.wav")
    w.set_session_reference(session)
    w.add_files([alice])
    (s.folder / "sub").mkdir()
    assert not w.set_session_reference(s.folder / "sub" / ".." / "alice.wav")
    assert w.session_reference.path == session
    assert "remove it from the track list first" in s.info.call_args.args[2].lower()


def test_an_unreadable_reference_keeps_the_previous_one(s, monkeypatch):
    w = s.w
    session = s.file("session.wav")
    w.set_session_reference(session)
    monkeypatch.setattr(main_window, "probe_streams", Mock(side_effect=OSError("not audio")))
    assert not w.set_session_reference(s.file("notes.wav"))
    assert w.session_reference.path == session
    assert "not audio" in s.warning.call_args.args[2]


def test_reference_controls_show_the_file_and_clear_removes_it(s):
    w = s.w
    w.show()
    assert not w.btn_remove_reference.isVisible()
    w.set_session_reference(s.file("session.wav"))
    assert w.btn_reference.text() == "Replace…"
    assert w.btn_remove_reference.isVisible()
    for shown in ("session.wav", "1:00:00", "Timing reference only"):
        assert shown in w.reference_label.text()
    w.clear_tracks()
    assert w.session_reference is None
    assert w.btn_reference.text() == "Add session reference…"
    assert not w.btn_remove_reference.isVisible()


def test_replacing_or_removing_the_reference_keeps_offsets(s):
    w = s.w
    w.add_files([s.file("bob.wav")])
    w._on_offset_edited(0, "5:00")
    w.set_session_reference(s.file("session.wav"))
    w.set_session_reference(s.file("other.wav"))
    assert w.session_reference.path.name == "other.wav"
    w.remove_session_reference()
    assert w.session_reference is None
    assert (w.tracks[0].offset, w.tracks[0].offset_source) == (300.0, OffsetSource.MANUAL)


def test_the_reference_never_reaches_transcription(s):
    w = s.w
    w.set_session_reference(s.file("session.wav"))
    w.add_files([s.file("bob.wav")])
    with patch("main_window.TranscribeWorker") as worker_class, \
            patch("main_window.llm.free_llm"):
        w.toggle_transcription()
    tracks, indices = worker_class.call_args.args
    assert [t.path.name for t in tracks] == ["bob.wav"]
    assert indices == [0]
    w.worker = None
    w._set_running(False)


def test_controls_stay_locked_through_alignment_and_review(s):
    w = s.w
    alice, bob = s.file("alice.wav"), s.file("bob.wav")
    w.set_session_reference(s.file("session.wav"))
    w.add_files([alice, bob])
    seen = s.align([Suggestion(alice, 0.0, "matched", "m", "strong"),
                    Suggestion(bob, 4.0, "matched", "m", "strong")],
                   lambda dialog: False, mode=SESSION_REFERENCE)
    running = seen["while_running"]
    assert running["align_text"] == "Cancel align"
    assert running["locked"] == [True] * 5
    assert running["transcribe"]          # no GPU or model involved
    assert running["offset_editable"]     # typing an offset stays allowed
    assert seen["locked_in_review"] == [True] * 6
    for button in (w.btn_add, w.btn_remove, w.btn_clear, w.btn_reference,
                   w.btn_remove_reference, w.btn_align):
        assert button.isEnabled()
    assert w.btn_align.text() == "Align…"


def test_shared_controls_wait_for_both_jobs(s):
    w = s.w
    w.add_files([s.file("alice.wav"), s.file("bob.wav")])
    w.worker = Mock()
    w._set_running(True)
    w.align_worker = Mock()
    w._update_busy()
    w.worker = None
    w._set_running(False)                 # transcription done, Align still running
    assert not w.btn_add.isEnabled()
    w.add_files([s.file("carol.wav")])    # as a file opened from Finder would be
    assert len(w.tracks) == 2
    assert "Files not added" in w.statusBar().currentMessage()
    w.align_worker = None
    w._update_busy()
    assert w.btn_add.isEnabled()


def test_strong_matches_start_ticked_and_apply_writes_every_row(s):
    w = s.w
    alice = s.file("alice.wav")
    w.add_files([alice])
    room = s.stereo("room.wav")
    carol, dave = s.file("carol.wav"), s.file("dave.wav")
    w.add_files([carol, dave])
    rows = [Suggestion(alice, 0.0, "reference", "Reference (longest recording)"),
            Suggestion(room, 12.5, "matched", "strong", "strong"),
            Suggestion(carol, 3.0, "matched", "weak", "weak"),
            Suggestion(dave, 40.0, "guess", "Guess")]
    seen = s.align(rows, lambda dialog: True)
    assert list(seen["ticks"].values()) == [True, True, False, False]
    assert [(t.path.name, t.offset, t.offset_source) for t in w.tracks] == [
        ("alice.wav", 0.0, OffsetSource.MATCHED),   # the anchor: room hangs from it
        ("room.wav", 12.5, OffsetSource.MATCHED),
        ("room.wav", 12.5, OffsetSource.MATCHED),
        ("carol.wav", 0.0, OffsetSource.DEFAULT),
        ("dave.wav", 0.0, OffsetSource.DEFAULT),
    ]
    assert offset_cell(w, 2).text() == "0:12.5"
    assert "Applied 1 offset" in w.statusBar().currentMessage()


def test_a_ticked_guess_is_recorded_as_a_guess(s):
    w = s.w
    alice, bob = s.file("alice.wav"), s.file("bob.wav")
    w.add_files([alice, bob])

    def tick_all(dialog):
        for row in range(len(dialog.suggestions)):
            dialog.set_ticked(row, True)
        return True
    s.align([Suggestion(alice, 0.0, "reference", "r"),
             Suggestion(bob, 40.0, "guess", "Guess")], tick_all)
    assert (w.tracks[1].offset, w.tracks[1].offset_source) == (40.0, OffsetSource.GUESS)


def test_a_manual_value_on_any_row_protects_the_whole_file(s):
    w = s.w
    alice = s.file("alice.wav")
    w.add_files([alice])
    room = s.stereo("room.wav")
    w._on_offset_edited(2, "0")            # a typed zero on the right channel
    rows = [Suggestion(alice, 0.0, "reference", "r"),
            Suggestion(room, 12.5, "matched", "Matched sound (strong match)", "strong")]
    seen = s.align(rows, lambda dialog: True)
    assert list(seen["ticks"].values()) == [True, False]
    assert "You set the current value" in seen["cells"][1][4]
    assert [t.offset for t in w.tracks[1:]] == [0.0, 0.0]

    def accept_room(dialog):
        dialog.set_ticked(1, True)
        return True
    s.align(rows, accept_room)
    assert [(t.offset, t.offset_source) for t in w.tracks[1:]] == [
        (12.5, OffsetSource.MATCHED)] * 2


def test_an_offset_typed_during_alignment_shows_as_current_and_manual(s):
    w = s.w
    alice, bob = s.file("alice.wav"), s.file("bob.wav")
    w.add_files([alice, bob])
    result = AlignmentResult([Suggestion(alice, 0.0, "reference", "r"),
                              Suggestion(bob, 4.0, "matched", "m", "strong")],
                             AUTOMATIC_REFERENCE, alice)
    seen = s.align([], lambda dialog: True,
                   run=s.run_while(result, lambda: w._on_offset_edited(1, "1:00")))
    assert seen["cells"][1][2] == "1:00"
    assert not seen["ticks"]["bob.wav#1"]
    assert (w.tracks[1].offset, w.tracks[1].offset_source) == (60.0, OffsetSource.MANUAL)


def test_cancel_in_the_review_window_changes_nothing(s):
    w = s.w
    alice, bob = s.file("alice.wav"), s.file("bob.wav")
    w.add_files([alice, bob])
    s.align([Suggestion(alice, 0.0, "reference", "r"),
             Suggestion(bob, 4.0, "matched", "m", "strong")], lambda dialog: False)
    assert [t.offset for t in w.tracks] == [0.0, 0.0]
    assert "No offsets were changed" in w.statusBar().currentMessage()


def test_cancelling_alignment_changes_nothing(s):
    w = s.w
    w.add_files([s.file("alice.wav"), s.file("bob.wav")])

    def until_stopped(worker):
        while not worker._stop:
            time.sleep(0.01)
        worker.cancelled = True
    QTimer.singleShot(0, w.toggle_alignment)   # press Cancel align
    seen = s.align([], run=until_stopped)
    assert seen["dialogs"] == 0
    assert [t.offset for t in w.tracks] == [0.0, 0.0]
    assert "cancelled" in w.statusBar().currentMessage()
    assert w.btn_align.text() == "Align…"


def test_a_failed_run_explains_and_changes_nothing(s):
    w = s.w
    w.set_session_reference(s.file("session.wav"))
    w.add_files([s.file("alice.wav")])
    message = "Couldn't use the session reference session.wav: no audio."

    def fail(worker):
        worker.error = message
    seen = s.align([], run=fail)
    assert seen["dialogs"] == 0
    assert s.warning.call_args.args[2] == message
    assert w.tracks[0].offset == 0.0
    assert w.btn_reference.isEnabled()


def test_rows_that_cannot_apply_have_no_checkbox(s):
    w = s.w
    session, alice, bob = (s.file(n) for n in ("session.wav", "alice.wav", "bob.wav"))
    w.set_session_reference(session)
    w.add_files([alice, bob])
    rows = [Suggestion(alice, -5.0, "matched", "Starts before the session reference by "
                       "0:05 (strong match with session.wav)", "strong"),
            Suggestion(bob, None, "error", "Couldn't read it: bad header")]

    def check(dialog):
        assert not dialog.is_checkable(0)
        assert not dialog.is_checkable(1)
        assert dialog.table.item(0, dialog.COL_SUGGESTED).text() == "-0:05"
        assert dialog.table.item(1, dialog.COL_SUGGESTED).text() == "—"
        for row in range(2):
            dialog.set_ticked(row, True)
        return True
    s.align(rows, check, mode=SESSION_REFERENCE, reference=session)
    assert [t.offset for t in w.tracks] == [0.0, 0.0]


def test_an_offset_still_being_typed_counts_when_the_review_opens(s):
    w = s.w
    w.show()
    alice, bob = s.file("alice.wav"), s.file("bob.wav")
    w.add_files([alice, bob])
    type_into_offset_cell(w, 1, "1:00", enter=False)   # Enter not pressed yet
    seen = s.align([Suggestion(alice, 0.0, "reference", "r"),
                    Suggestion(bob, 4.0, "matched", "m", "strong")], lambda dialog: True)
    assert seen["cells"][1][2] == "1:00"
    assert not seen["ticks"]["bob.wav#1"]
    assert (w.tracks[1].offset, w.tracks[1].offset_source) == (60.0, OffsetSource.MANUAL)
    assert w._offset_delegate.open_editor is None


def test_what_align_read_is_kept_for_the_files_still_loaded(s):
    w = s.w
    alice, bob, carol = s.file("alice.wav"), s.file("bob.wav"), s.file("carol.wav")
    w.add_files([alice, bob])
    kept_alice = ((1, 10), "alice's fingerprint")
    read = {(alice, (None,)): kept_alice, (bob, (None,)): ((2, 20), "bob's")}

    def cancelled(worker):
        worker.fingerprints = dict(read)
        worker.cancelled = True
    seen = s.align([], run=cancelled)
    assert seen["known"] == {}
    assert w._align_fingerprints == read     # a cancelled run's too

    w.table.selectRow(1)
    w.remove_selected()
    w.add_files([carol])
    result = AlignmentResult([Suggestion(alice, 0.0, "reference", "r"),
                              Suggestion(carol, 4.0, "matched", "m", "strong")],
                             AUTOMATIC_REFERENCE, alice)
    seen = s.align([], result=result)
    assert seen["known"] == {(alice, (None,)): kept_alice}
    assert w._align_fingerprints == {(alice, (None,)): kept_alice}


def test_suggestions_follow_the_reference_offset_typed_during_alignment(s):
    w = s.w
    alice, bob = s.file("alice.wav"), s.file("bob.wav")
    w.add_files([alice, bob])
    result = s.automatic([Suggestion(alice, 0.0, "reference", "Reference"),
                          Suggestion(bob, 4.0, "matched", "m", "strong")])
    seen = s.align([], lambda dialog: True,
                   run=s.run_while(result, lambda: w._on_offset_edited(0, "1:00")))
    assert [row[3] for row in seen["cells"]] == ["1:00", "1:04"]
    assert [t.offset for t in w.tracks] == [60.0, 64.0]
    assert w.tracks[0].offset_source == OffsetSource.MANUAL


def test_a_shift_past_a_typed_offset_leaves_everything_unticked(s):
    w = s.w
    alice, bob = s.file("alice.wav"), s.file("bob.wav")
    w.add_files([alice, bob])
    w._on_offset_edited(0, "0:20")
    # bob started 10 s before alice; then alice is retyped to 0:05, which
    # would put bob before 0:00, so every suggestion moves 5 s later.
    result = s.automatic([Suggestion(alice, 0.0, "reference", "Reference"),
                          Suggestion(bob, -10.0, "matched", "m", "strong")], anchor=20.0)
    seen = s.align([], lambda dialog: True,
                   run=s.run_while(result, lambda: w._on_offset_edited(0, "0:05")))
    assert [row[3] for row in seen["cells"]] == ["0:10", "0:00"]
    assert list(seen["ticks"].values()) == [False, False]
    assert "alice.wav's offset changed while Align ran" in seen["note"]
    assert "moved 0:05 later" in seen["note"]
    assert "Nothing is ticked yet: alice.wav already has an offset" in seen["note"]
    assert [(t.offset, t.offset_source) for t in w.tracks] == [
        (5.0, OffsetSource.MANUAL), (0.0, OffsetSource.DEFAULT)]


def test_a_shift_past_an_earlier_guess_leaves_everything_unticked(s):
    w = s.w
    alice, bob, carol = (s.file(n) for n in ("alice.wav", "bob.wav", "carol.wav"))
    w.add_files([alice, bob, carol])
    w.tracks[2].offset, w.tracks[2].offset_source = 40.0, OffsetSource.GUESS
    result = s.automatic([Suggestion(alice, 0.0, "reference", "Reference"),
                          Suggestion(bob, -30.0, "matched", "m", "strong"),
                          Suggestion(carol, 70.0, "guess", "Guess")])
    seen = s.align([], lambda dialog: True, result=result)
    assert list(seen["ticks"].values()) == [False, False, False]
    assert "carol.wav already has an offset" in seen["note"]
    assert [t.offset for t in w.tracks] == [0.0, 0.0, 40.0]


def test_a_new_reference_keeps_an_earlier_run_together(s):
    w = s.w
    alice, bob = s.file("alice.wav"), s.file("bob.wav")
    w.add_files([alice, bob])
    s.align([Suggestion(alice, 0.0, "reference", "Reference"),
             Suggestion(bob, 10.0, "matched", "m", "strong")], lambda dialog: True)
    assert [(t.offset, t.offset_source) for t in w.tracks] == [
        (0.0, OffsetSource.MATCHED), (10.0, OffsetSource.MATCHED)]
    # A longer recording comes in and becomes the reference; it places bob
    # clearly but alice only weakly. Ticking bob alone would put the two 30 s
    # apart where the first run measured 10.
    room = s.file("room.wav")
    w.add_files([room])
    seen = s.align([Suggestion(room, 0.0, "reference", "Reference"),
                    Suggestion(alice, 20.0, "matched", "m", "weak"),
                    Suggestion(bob, 30.0, "matched", "m", "strong")], lambda dialog: True)
    assert list(seen["ticks"].values()) == [False, False, False]
    assert "alice.wav already has an offset" in seen["note"]
    assert [t.offset for t in w.tracks] == [0.0, 10.0, 0.0]


def test_a_placed_file_that_starts_ticked_is_not_left_behind(s):
    w = s.w
    alice, bob, carol = (s.file(n) for n in ("alice.wav", "bob.wav", "carol.wav"))
    w.add_files([alice, bob, carol])
    w.tracks[1].offset, w.tracks[1].offset_source = 4.0, OffsetSource.MATCHED
    result = s.automatic([Suggestion(alice, 0.0, "reference", "Reference"),
                          Suggestion(bob, -10.0, "matched", "m", "strong"),
                          Suggestion(carol, 3.0, "matched", "m", "weak")])
    seen = s.align([], lambda dialog: True, result=result)
    assert list(seen["ticks"].values()) == [True, True, False]
    assert "Rows left unticked keep their current offsets" in seen["note"]
    assert [t.offset for t in w.tracks] == [10.0, 0.0, 0.0]


def test_a_shift_with_nothing_typed_keeps_the_usual_ticks(s):
    w = s.w
    alice, bob, carol = (s.file(n) for n in ("alice.wav", "bob.wav", "carol.wav"))
    w.add_files([alice, bob, carol])
    result = s.automatic([Suggestion(alice, 0.0, "reference", "Reference"),
                          Suggestion(bob, -10.0, "matched", "m", "strong"),
                          Suggestion(carol, 3.0, "matched", "m", "weak")])
    seen = s.align([], lambda dialog: True, result=result)
    assert list(seen["ticks"].values()) == [True, True, False]
    assert "Rows left unticked keep their current offsets" in seen["note"]
    assert [t.offset for t in w.tracks] == [10.0, 0.0, 0.0]


def test_the_review_waits_for_another_open_prompt(s):
    w = s.w
    w.add_files([s.file("alice.wav"), s.file("bob.wav")])
    prompt = QDialog(w)
    prompt.setModal(True)
    prompt.show()
    QTimer.singleShot(400, prompt.close)
    opened_over = []
    s.align([Suggestion(w.tracks[0].path, 0.0, "reference", "r")],
            lambda dialog: opened_over.append(prompt.isVisible()) or False)
    assert opened_over == [False]


def test_quitting_while_the_review_waits_ends_it(s):
    w = s.w
    w.add_files([s.file("alice.wav"), s.file("bob.wav")])
    prompt = QDialog(w)
    prompt.setModal(True)
    prompt.show()
    QTimer.singleShot(100, lambda: (setattr(w, "_quit_requested", True), prompt.close()))
    seen = s.align([Suggestion(w.tracks[0].path, 0.0, "reference", "r")])
    assert seen["dialogs"] == 0
    assert not w._align_reviewing


def test_transcription_started_during_alignment_keeps_the_list_locked(s, monkeypatch):
    w = s.w
    w.set_session_reference(s.file("session.wav"))
    alice, bob = s.file("alice.wav"), s.file("bob.wav")
    w.add_files([alice, bob])
    worker_class = Mock()
    monkeypatch.setattr(main_window, "TranscribeWorker", worker_class)
    monkeypatch.setattr(main_window.llm, "free_llm", lambda: None)
    result = AlignmentResult([Suggestion(alice, 0.0, "matched", "m", "strong"),
                              Suggestion(bob, 4.0, "matched", "m", "strong")],
                             SESSION_REFERENCE, w.session_reference.path)
    s.align([], lambda dialog: True, mode=SESSION_REFERENCE,
            run=s.run_while(result, w.toggle_transcription))
    tracks, indices = worker_class.call_args.args
    assert [t.path.name for t in tracks] == ["alice.wav", "bob.wav"]
    assert indices == [0, 1]
    assert w.tracks[1].offset == 4.0
    for button in (w.btn_add, w.btn_remove, w.btn_clear):
        assert not button.isEnabled()   # transcription is still running
    assert w.btn_reference.isEnabled()
    w.worker = None
    w._set_running(False)
    assert w.btn_add.isEnabled()


def _transcribed(s):
    """A session reference plus one transcribed speaker, Bob."""
    w = s.w
    w.set_session_reference(s.file("session.wav"))
    w.add_files([s.file("bob.wav")])
    bob = w.tracks[0]
    bob.segments.append(Segment("Bob", 1.0, 2.0, "Hello there."))
    bob.status = TrackStatus.DONE
    return bob


def test_the_reference_never_reaches_ai_notes(s):
    w = s.w
    _transcribed(s)
    with patch("main_window.NotesWorker") as notes_class:
        w.toggle_ai_notes()
    assert notes_class.call_args.args[0].strip() == "[0:01] Bob: Hello there."
    w.notes_worker = None


def test_the_reference_never_reaches_exports_and_offsets_are_read_at_write(s):
    w = s.w
    bob = _transcribed(s)
    target = s.folder / "out.json"

    def save_dialog(*_args):
        bob.offset = 5.0     # an offset that changed while the dialog was open
        return str(target), "JSON (*.json)"
    with patch("main_window.QFileDialog.getSaveFileName", side_effect=save_dialog):
        w.export_dialog()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert [t["file"] for t in data["tracks"]] == [str(bob.path)]
    assert data["segments"][0]["start"] == 6.0


def test_quitting_stops_a_running_alignment_and_waits_for_it(s):
    w = s.w
    w.add_files([s.file("alice.wav"), s.file("bob.wav")])
    started = threading.Event()

    class Reading(align.AlignWorker):
        def run(inner):
            started.set()
            while not inner._stop:
                time.sleep(0.01)
            inner.cancelled = True

    with patch("main_window.AlignWorker", Reading):
        w.toggle_alignment()
    worker = w.align_worker
    assert started.wait(5)
    event = QCloseEvent()
    w.closeEvent(event)
    assert worker._stop
    assert event.isAccepted()
    assert not worker.isRunning()     # never destroyed while its thread is alive
