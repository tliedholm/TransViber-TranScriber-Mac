"""Main application window."""

from __future__ import annotations

import html
import os
import sys
import time
from functools import partial
from pathlib import Path

from PySide6.QtCore import QSettings, QSignalBlocker, Qt, QTimer, Slot
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QGuiApplication, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemDelegate,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

import diagnostics
import exporters
from align import (
    CONSISTENCY_SECONDS,
    SESSION_REFERENCE,
    AlignmentResult,
    AlignWorker,
    format_offset,
    parse_offset,
    rebase,
)
from audio import channel_tracks, probe_streams, tracks_for_file
from llm import DEFAULT_NOTES_MODEL, NOTES_MODEL_GROUPS, NotesWorker
from merger import coalesce, discarded_segment_count, merge_tracks
from models import (
    AUDIO_EXTENSIONS,
    OffsetSource,
    RunConfig,
    SessionReference,
    Track,
    TrackStatus,
    audio_paths,
)
from stats import StatsPoller
from transcriber import (
    TranscribeWorker,
    cuda_available,
    engine_for,
    language_supported,
    mlx_installed,
)

IS_MAC = sys.platform == "darwin"

# Tiered to match the PC edition, plus a Drafts tier for the sub-medium
# Whisper sizes the user asked to keep on the Mac.
MODEL_GROUPS = [
    ("Max Detail (Large Tier)", ["qwen3-asr-1.7b", "large-v3", "large-v2"]),
    ("Standard Speed (Medium Tier)",
     ["canary-1b-v2", "large-v3-turbo", "medium", "medium.en"]),
    ("Quick Review / Rapid Capture (Small Tier)",
     ["distil-large-v3.5-ct2", "distil-large-v3", "qwen3-asr-0.6b",
      "parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v2"]),
    ("Drafts (Sub-medium Whisper)",
     ["small", "small.en", "base", "base.en", "tiny", "tiny.en"]),
]
MODELS = [name for _, names in MODEL_GROUPS for name in names]


def _add_grouped_models(combo: QComboBox, groups: list[tuple[str, list[str]]]) -> None:
    """Populate a combo with bold, non-selectable tier headings between groups."""
    first_model_index = -1
    for group_number, (label, names) in enumerate(groups):
        if group_number:
            combo.insertSeparator(combo.count())
        combo.addItem(label)
        header = combo.model().item(combo.count() - 1)
        if header is not None:
            header.setEnabled(False)
            header.setSelectable(False)
            font = header.font()
            font.setBold(True)
            header.setFont(font)
        for name in names:
            combo.addItem(name)
            if first_model_index < 0:
                first_model_index = combo.count() - 1
    if first_model_index >= 0:
        combo.setCurrentIndex(first_model_index)

LANGUAGES = [
    ("Auto-detect", None), ("English", "en"), ("Swedish", "sv"), ("Spanish", "es"),
    ("French", "fr"), ("German", "de"), ("Italian", "it"), ("Portuguese", "pt"),
    ("Dutch", "nl"), ("Danish", "da"), ("Norwegian", "no"), ("Finnish", "fi"),
    ("Polish", "pl"), ("Russian", "ru"), ("Ukrainian", "uk"), ("Japanese", "ja"),
    ("Korean", "ko"), ("Chinese", "zh"), ("Arabic", "ar"), ("Hindi", "hi"),
    ("Turkish", "tr"),
]

COL_SPEAKER, COL_FILE, COL_DURATION, COL_OFFSET, COL_STATUS, COL_PROGRESS = range(6)

OFFSET_HEADER_TIP = (
    "When this recording started, measured from the start of the session.\n"
    "Set it for someone who joined late: 2:30 puts all of their lines\n"
    "2 minutes 30 seconds later. Leave it at 0:00 if the file already\n"
    "includes the waiting time at the start.\n"
    "\n"
    "Type seconds (150), minutes:seconds (2:30) or hours:minutes:seconds\n"
    "(1:02:30). The transcript re-orders straight away; nothing is\n"
    "transcribed again. Type 0 to reset it."
)

OFFSET_SOURCE_TIPS = {
    OffsetSource.DEFAULT: "Not set: assumed to start with the session.",
    OffsetSource.MANUAL: "Typed by you.",
    OffsetSource.MATCHED: "Found by Align from sound shared with another recording.",
    OffsetSource.GUESS: "Align's guess: assumes this recorder stopped when the reference did.",
}

ALIGN_TIP = (
    "Work out when each speaker file started, from sound the recordings\n"
    "share: a session reference, the same room, or one voice picked up by\n"
    "another mic. Nothing changes until you review the suggestions and\n"
    "press Apply; you can still edit any offset afterwards."
)

REFERENCE_TIP = (
    "Optional: one recording of the whole session, such as a call or room\n"
    "mix, for Align to time every speaker file against. Its start is\n"
    "session 0:00. It is used for timing only: never transcribed, exported\n"
    "or sent to AI Notes. The speaker files are still what gets transcribed."
)

FILE_FILTER = "Audio files (*" + " *".join(sorted(AUDIO_EXTENSIONS)) + ");;All files (*.*)"

EXPORT_FILTERS = (
    "Markdown (*.md);;Plain text (*.txt);;SubRip subtitles (*.srt);;"
    "WebVTT subtitles (*.vtt);;JSON (*.json);;All formats (*.md *.txt *.srt *.vtt *.json)"
)


def _token_readout(tokens: int, tps: float) -> str:
    """'1,234 tok · 18 tok/s', the speed left out until there is one."""
    return f"{tokens:,} tok" + (f" · {tps:.0f} tok/s" if tps > 0 else "")


class _RunClock:
    """A run's elapsed time in a label: ticking while it runs, then "took …"."""

    def __init__(self, label: QLabel, parent) -> None:
        self.label = label
        self.started: float | None = None
        self.timer = QTimer(parent)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)

    def start(self) -> None:
        self.started = time.monotonic()
        self.label.setText("0:00")
        self.timer.start()

    def stop(self) -> str:
        """Show how long the run took, and return it ("" if none was running)."""
        self.timer.stop()
        if self.started is None:
            return ""
        elapsed = exporters.hms(time.monotonic() - self.started)
        self.label.setText(f"took {elapsed}")
        self.started = None
        return elapsed

    def _tick(self) -> None:
        if self.started is not None:
            self.label.setText(exporters.hms(time.monotonic() - self.started))


def _canonical(path: Path):
    """One key per file, so the same recording can't be loaded in two roles
    under different names: letter case (a Mac disk ignores it), a relative
    path, a link. The file's identity on disk where it can be looked at,
    and its resolved path where it can't."""
    try:
        st = path.stat()
        return st.st_dev, st.st_ino
    except OSError:
        pass
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


class _OffsetDelegate(QStyledItemDelegate):
    """Hands a committed Offset edit straight to the window.

    Going through the model would drop one edit that matters: retyping the
    value a cell already shows — "0:00" over a default 0:00 — is a deliberate
    "this one really starts at zero", but QTableWidgetItem ignores a write of
    the text it already holds, so itemChanged never fires. A cell opened and
    closed without typing is not an edit and is left alone.
    """

    def __init__(self, on_commit, parent=None) -> None:
        super().__init__(parent)
        self._on_commit = on_commit
        self.open_editor = None

    def createEditor(self, parent, option, index):
        self.open_editor = super().createEditor(parent, option, index)
        return self.open_editor

    def destroyEditor(self, editor, index) -> None:
        if editor is self.open_editor:
            self.open_editor = None
        super().destroyEditor(editor, index)

    def finish_open_edit(self) -> None:
        """Commit and close an editor still open, as clicking elsewhere would."""
        editor = self.open_editor
        if editor is not None:
            self.commitData.emit(editor)
            self.closeEditor.emit(editor, QAbstractItemDelegate.EndEditHint.NoHint)

    def setModelData(self, editor, model, index) -> None:
        if editor.isModified():
            self._on_commit(index.row(), editor.text())


class AlignReviewDialog(QDialog):
    """Align's suggestions, one row per speaker file. Applies nothing itself.

    Strong matches start ticked; weak matches, guesses and anything you typed
    yourself start unticked, and a suggestion that can't be applied (an
    unreadable file, a start before the session reference) has no checkbox.
    If the clock moves (every suggestion shifted later, or a file placed
    before would move) and a file that already has an offset would start
    unticked, nothing starts ticked. "Now" is read when the window opens, so
    an offset typed while Align ran shows as current, and, being typed,
    protects its row.
    """

    COL_APPLY, COL_FILE, COL_NOW, COL_SUGGESTED, COL_HOW = range(5)

    def __init__(self, result: AlignmentResult, tracks: list[Track], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Align: review suggested offsets")
        self.suggestions = result.suggestions
        layout = QVBoxLayout(self)
        if result.mode == SESSION_REFERENCE:
            basis = (f"Reference: {result.reference.name}; session starts at 0:00; "
                     f"used for timing only.")
        else:
            basis = (f"No session reference: matched against {result.reference.name}, "
                     f"the longest recording.")
        self.basis_label = QLabel(basis)
        self.basis_label.setWordWrap(True)
        layout.addWidget(self.basis_label)
        # When the clock moves — every suggestion shifted later, or a file an
        # earlier run placed would move — a file that already has an offset
        # (typed, or from that earlier run) and isn't ticked would stay where
        # it is while the rest move, so then nothing starts ticked.
        rows_of = {s.path: [t for t in tracks if t.path == s.path] for s in self.suggestions}
        ticked = [s for s in self.suggestions if self._starts_ticked(s, rows_of[s.path])]

        def placed(s) -> bool:
            return any(t.offset_source is not OffsetSource.DEFAULT for t in rows_of[s.path])
        left_behind = [s.path.name for s in self.suggestions if s not in ticked and placed(s)]
        clock_moved = bool(result.shift) or any(
            placed(s) and any(abs(t.offset - s.offset) > CONSISTENCY_SECONDS
                              for t in rows_of[s.path]) for s in ticked)
        default_ticks = not (clock_moved and left_behind)
        note = result.note
        if not default_ticks:
            names = ", ".join(left_behind)
            note += (f" Nothing is ticked yet: {names} already "
                     f"{'have offsets' if len(left_behind) > 1 else 'has an offset'}, and "
                     f"applying only the others would leave "
                     f"{'them' if len(left_behind) > 1 else 'it'} out of step.")
        elif result.shift:
            note += " Rows left unticked keep their current offsets."
        self.note_label = QLabel(note.strip())
        self.note_label.setWordWrap(True)
        self.note_label.setVisible(bool(note.strip()))
        layout.addWidget(self.note_label)

        self.table = QTableWidget(len(self.suggestions), 5)
        self.table.setHorizontalHeaderLabels(
            ["Apply", "File", "Now", "Suggested", "How it was found"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        header = self.table.horizontalHeader()
        for col in range(4):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_HOW, QHeaderView.ResizeMode.Stretch)
        for row, suggestion in enumerate(self.suggestions):
            self._fill_row(row, suggestion, rows_of[suggestion.path],
                           default_ticks and suggestion in ticked)
        layout.addWidget(self.table)

        hint = QLabel("Ticked rows go into the Offset column, where you can still change "
                      "them. Cancel changes nothing.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(860, 160 + 30 * len(self.suggestions))

    @staticmethod
    def _starts_ticked(suggestion, rows: list[Track]) -> bool:
        """Strong matches and the reference, unless you typed the file's offset."""
        return (suggestion.applicable
                and not any(t.offset_source is OffsetSource.MANUAL for t in rows)
                and (suggestion.method == "reference" or suggestion.strength == "strong"))

    def _fill_row(self, row: int, suggestion, rows: list[Track], ticked: bool) -> None:
        # The track list is locked from Align until this window closes, so
        # every suggestion still has its rows.
        now = format_offset(rows[0].offset) if len({t.offset for t in rows}) == 1 else "varies"
        reason = suggestion.reason
        if any(t.offset_source is OffsetSource.MANUAL for t in rows):
            reason += ". You set the current value"

        apply_item = QTableWidgetItem()
        apply_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        if suggestion.applicable:
            apply_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            apply_item.setCheckState(Qt.CheckState.Checked if ticked else Qt.CheckState.Unchecked)
        self.table.setItem(row, self.COL_APPLY, apply_item)
        suggested = format_offset(suggestion.offset) if suggestion.offset is not None else "—"
        for col, text in ((self.COL_FILE, suggestion.path.name), (self.COL_NOW, now),
                          (self.COL_SUGGESTED, suggested), (self.COL_HOW, reason)):
            item = QTableWidgetItem(text)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            if col == self.COL_FILE:
                item.setToolTip(str(suggestion.path))
            elif col == self.COL_HOW:
                item.setToolTip(reason)
            self.table.setItem(row, col, item)

    def is_checkable(self, row: int) -> bool:
        return bool(self.table.item(row, self.COL_APPLY).flags()
                    & Qt.ItemFlag.ItemIsUserCheckable)

    def is_ticked(self, row: int) -> bool:
        return (self.is_checkable(row) and
                self.table.item(row, self.COL_APPLY).checkState() == Qt.CheckState.Checked)

    def set_ticked(self, row: int, ticked: bool) -> None:
        if self.is_checkable(row):
            self.table.item(row, self.COL_APPLY).setCheckState(
                Qt.CheckState.Checked if ticked else Qt.CheckState.Unchecked)

    def chosen(self) -> list:
        return [s for row, s in enumerate(self.suggestions) if self.is_ticked(row)]


def _missing_from_export(track: Track) -> str | None:
    """How much of a track's speech an export would leave out, or None.

    A finished track with no segments is silent audio, not a gap, so only
    the status decides. A failed track is left out whole, partial segments
    and all (see merge_tracks); a cancelled one keeps what it finished; one
    that never started is missing as surely as either.
    """
    if track.status is TrackStatus.DONE:
        return None
    if track.status is TrackStatus.PENDING:
        return "not transcribed"
    if track.status is TrackStatus.TRANSCRIBING:
        return "still transcribing"
    if track.status is TrackStatus.ERROR:
        return ("failed partway — its partial transcript is left out" if track.segments
                else "failed — nothing transcribed")
    return ("cancelled partway — partial transcript" if track.segments
            else "cancelled — nothing transcribed")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TransViber TransScriber")
        self.resize(1100, 750)
        self.setAcceptDrops(True)

        self.tracks: list[Track] = []
        self.worker: TranscribeWorker | None = None
        # Align's timing-only file: never a Track, so never transcribed.
        self.session_reference: SessionReference | None = None
        self.align_worker: AlignWorker | None = None
        self._align_reviewing = False
        # Fingerprints from the last Align, by (path, streams), for the files
        # it read: one unchanged since isn't decoded again (see AlignWorker).
        self._align_fingerprints: dict = {}
        # The merged transcript as last drawn, and the views not yet showing it.
        self._transcript_html = ""
        self._stale_views: set = set()
        # Set once the user asks to quit while a worker is still running; the
        # close is completed by _quit_if_idle() when the worker really stops.
        self._quit_requested = False
        # Every change to the notes sets it; only a write that succeeded
        # clears it. A close never drops notes it could not write unasked.
        self._notes_dirty = False
        self._notes_save_error = ""
        self._asking_about_notes = False

        self._build_ui()
        self._load_settings()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(400)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._render_transcript)

    # ---------- UI construction ----------

    def _build_ui(self) -> None:
        has_cuda = not IS_MAC and cuda_available()
        has_mlx = mlx_installed()
        central = QWidget()
        root = QVBoxLayout(central)

        # Top control bar
        bar = QHBoxLayout()
        self.btn_add = QPushButton("Add Files…")
        self.btn_add.clicked.connect(self.add_files_dialog)
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_tracks)
        self.btn_align = QPushButton("Align…")
        self.btn_align.setToolTip(ALIGN_TIP)
        self.btn_align.clicked.connect(self.toggle_alignment)
        bar.addWidget(self.btn_add)
        bar.addWidget(self.btn_remove)
        bar.addWidget(self.btn_clear)
        bar.addWidget(self.btn_align)
        bar.addSpacing(20)

        bar.addWidget(QLabel("Model:"))
        self.combo_model = QComboBox()
        self.combo_model.setEditable(True)
        _add_grouped_models(self.combo_model, MODEL_GROUPS)
        self.combo_model.setToolTip(
            "Whisper models: pick a size, or type any CTranslate2 repo from\n"
            "Hugging Face (e.g. distil-whisper/distil-large-v3.5-ct2).\n"
            "'.en' models are English-only and slightly more accurate for English.\n"
            "\n"
            "Qwen models (via mlx-qwen3-asr, Apple GPU only): qwen3-asr-1.7b —\n"
            "state-of-the-art accuracy, 30 languages (no Norwegian or\n"
            "Ukrainian), supports vocabulary hints. -0.6b is the faster,\n"
            "lighter sibling. The Device setting is ignored (always Metal GPU).\n"
            "\n"
            "NVIDIA models (via onnx-asr): parakeet-tdt-0.6b-v2 (English) and\n"
            "-v3 (25 European languages) — faster than Whisper, top English\n"
            "accuracy. canary-1b-v2 — very accurate, CC-BY-4.0 (free, incl.\n"
            "commercial use). The Language setting is ignored (auto)."
        )
        bar.addWidget(self.combo_model)
        self.star_model_btn = self._make_star_button(self.combo_model, "starred_model")
        bar.addWidget(self.star_model_btn)

        bar.addWidget(QLabel("Language:"))
        self.combo_lang = QComboBox()
        for name, code in LANGUAGES:
            self.combo_lang.addItem(name, code)
        bar.addWidget(self.combo_lang)
        # Shown while the chosen language is one the model can't take.
        self.label_lang_conflict = QLabel("⚠ Not supported")
        self.label_lang_conflict.hide()
        bar.addWidget(self.label_lang_conflict)
        self.combo_model.currentTextChanged.connect(self._update_language_choices)
        self.combo_lang.currentIndexChanged.connect(self._update_language_choices)

        bar.addWidget(QLabel("Device:"))
        self.combo_device = QComboBox()
        if IS_MAC:
            self.combo_device.addItem("GPU (Metal)", "mlx")
            self.combo_device.addItem("CPU", "cpu")
            if not has_mlx:
                self.combo_device.setToolTip(
                    "Whisper models need mlx-whisper for the GPU (run setup_mac.sh);\n"
                    "without it they run on CPU. Parakeet/Canary always run on CPU\n"
                    "(fastest path for them on Apple silicon)."
                )
        else:
            self.combo_device.addItem(
                "GPU (CUDA)" if has_cuda else "GPU (unavailable)", "cuda")
            self.combo_device.addItem("CPU", "cpu")
            if not has_cuda:
                self.combo_device.model().item(0).setEnabled(False)
                self.combo_device.setCurrentIndex(1)
        bar.addWidget(self.combo_device)

        bar.addStretch()
        # Elapsed time of the running (or last finished) transcription.
        self.elapsed_label = QLabel("")
        self.elapsed_label.setToolTip("How long the current or last transcription run took.")
        bar.addWidget(self.elapsed_label)
        self._run_clock = _RunClock(self.elapsed_label, self)
        self.btn_transcribe = QPushButton("Transcribe")
        self.btn_transcribe.setDefault(True)
        self.btn_transcribe.clicked.connect(self.toggle_transcription)
        bar.addWidget(self.btn_transcribe)
        self.btn_export = QPushButton("Export…")
        self.btn_export.clicked.connect(self.export_dialog)
        self.btn_export.setEnabled(False)
        bar.addWidget(self.btn_export)
        root.addLayout(bar)

        # Vocabulary hints (Whisper and Qwen): bias recognition toward unusual
        # names and terms — campaign characters, places, jargon. Hidden when
        # a model without hint support (Parakeet/Canary) is selected. It lives
        # in a tall box down the left side so a long D&D name list stays
        # readable and easy to scan while transcribing.
        self.hints_widget = QWidget()
        hints_col = QVBoxLayout(self.hints_widget)
        hints_col.setContentsMargins(0, 0, 0, 0)
        hints_col.addWidget(QLabel("Vocabulary hints:"))
        self.edit_hints = QPlainTextEdit()
        self.edit_hints.setPlaceholderText(
            "Names and terms to spell correctly, one per line or "
            "comma-separated, e.g.\nZephyrine\nThaldrin\nWaterdeep\nvecna\n"
            "owlbear"
        )
        self.edit_hints.setToolTip(
            "Words or phrases the audio is likely to contain — put one per\n"
            "line or separate them with commas. Whisper is nudged toward\n"
            "these spellings — great for D&D character names, places, and\n"
            "other words it would otherwise mangle. Remembered between sessions."
        )
        hints_col.addWidget(self.edit_hints)
        self.combo_model.currentTextChanged.connect(self._update_hints_visibility)

        # Track table + transcript view
        splitter = QSplitter(Qt.Orientation.Vertical)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Speaker", "File", "Duration", "Offset", "Status", "Progress"])
        self.table.horizontalHeaderItem(COL_OFFSET).setToolTip(OFFSET_HEADER_TIP)
        self._offset_delegate = _OffsetDelegate(self._on_offset_edited, self.table)
        self.table.setItemDelegateForColumn(COL_OFFSET, self._offset_delegate)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_SPEAKER, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_FILE, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_DURATION, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_OFFSET, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_PROGRESS, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(COL_SPEAKER, 180)
        self.table.setColumnWidth(COL_PROGRESS, 140)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_table_menu)

        # Optional session reference: a timing-only file kept out of the table.
        reference_row = QHBoxLayout()
        self.btn_reference = QPushButton("Add session reference…")
        self.btn_reference.setToolTip(REFERENCE_TIP)
        self.btn_reference.clicked.connect(self.choose_session_reference)
        self.reference_label = QLabel()
        self.reference_label.setToolTip(REFERENCE_TIP)
        self.btn_remove_reference = QPushButton("Remove reference")
        self.btn_remove_reference.clicked.connect(self.remove_session_reference)
        reference_row.addWidget(self.btn_reference)
        reference_row.addWidget(self.reference_label, 1)
        reference_row.addWidget(self.btn_remove_reference)
        tracks_box = QWidget()
        tracks_col = QVBoxLayout(tracks_box)
        tracks_col.setContentsMargins(0, 0, 0, 0)
        tracks_col.addLayout(reference_row)
        tracks_col.addWidget(self.table)
        splitter.addWidget(tracks_box)
        self._show_session_reference()

        self.transcript_view = QTextBrowser()
        self.transcript_view.setOpenExternalLinks(False)
        self.transcript_view.setPlaceholderText(
            "Drop audio tracks above (one file per speaker), set speaker names, "
            "then click Transcribe. The merged conversation appears here."
        )
        splitter.addWidget(self.transcript_view)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        # Hints box on the left, the table/transcript stack on the right.
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(self.hints_widget)
        main_splitter.addWidget(splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([280, 720])
        root.addWidget(main_splitter)

        self.tabs = QTabWidget()
        self.tabs.addTab(central, "Transcribe")
        self.tabs.addTab(self._build_notes_page(), "Notes")
        self.setCentralWidget(self.tabs)
        self.tabs.currentChanged.connect(lambda _index: self._sync_transcript_views())
        self.check_show_transcript.toggled.connect(
            lambda _shown: self._sync_transcript_views())

        if IS_MAC:
            device_text = ("GPU: Apple Silicon (Metal)" if has_mlx
                           else "GPU: unavailable (mlx-whisper not installed)")
        elif has_cuda:
            device_text = "GPU: CUDA available"
        else:
            device_text = "GPU not available — using CPU"
        # Model-download progress, shown in the status bar while weights
        # download on first use of a model. A text label, not a QProgressBar:
        # macOS's Qt style never renders progress-bar text, and the user
        # wants the percentage, not a bar.
        self.download_label = QLabel("")
        self.statusBar().addPermanentWidget(self.download_label)
        self.dark_checkbox = QCheckBox("Dark mode")
        self.dark_checkbox.setToolTip(
            "Override the Mac's appearance for this app. Until changed, the app follows macOS.")
        self.dark_checkbox.toggled.connect(self._set_dark_mode)
        self.statusBar().addPermanentWidget(self.dark_checkbox)
        # Live system utilization next to the device label, refreshed by a
        # background poller thread.
        self.stats_label = QLabel("")
        self.stats_label.setToolTip(
            "Whole-system utilization: Apple GPU, CPU, and unified memory\n"
            "(the RAM the GPU shares). Same source as Activity Monitor."
        )
        self.statusBar().addPermanentWidget(self.stats_label)
        self.device_label = QLabel(device_text)
        self.statusBar().addPermanentWidget(self.device_label)
        self.stats_poller = StatsPoller(self)
        self.stats_poller.stats.connect(self.stats_label.setText)
        self.stats_poller.start()
        self.statusBar().showMessage("Ready")
        # Summary of the last finished transcription/notes run ("took x:xx…").
        # It lives in the status bar's message slot (bottom left) and is
        # re-shown whenever a temporary message expires, so it persists until
        # the next run starts and clears it.
        self._last_summary: str = ""
        self.statusBar().messageChanged.connect(self._on_status_message_changed)

    def _on_status_message_changed(self, text: str) -> None:
        if not text and self._last_summary:
            self.statusBar().showMessage(self._last_summary)

    def _set_summary(self, msg: str) -> None:
        """Show a finished run's summary, kept in the status bar until the next run."""
        self._last_summary = msg
        self.statusBar().showMessage(msg)

    def _on_download_progress(self, model: str, percent: int, total_gb: float) -> None:
        """The shared download readout; `model` is the downloading job's own,
        bound when it started, not whatever its combo box says now."""
        if percent < 0:
            self.download_label.setText("")
            return
        size = f" of {total_gb:.1f} GB" if total_gb >= 0.05 else ""
        self.download_label.setText(f"Downloading model: {percent}%{size}")
        self.statusBar().showMessage(f"Downloading model '{model}' — {percent}%{size}…")

    @staticmethod
    def _apply_color_scheme(dark: bool) -> None:
        QGuiApplication.styleHints().setColorScheme(
            Qt.ColorScheme.Dark if dark else Qt.ColorScheme.Light)

    def _set_dark_mode(self, dark: bool) -> None:
        """Apply and remember an explicit appearance choice."""
        self._apply_color_scheme(dark)
        QSettings().setValue("dark", dark)

    def _update_hints_visibility(self) -> None:
        model = self.combo_model.currentText().strip()
        self.hints_widget.setVisible(engine_for(model) in ("whisper", "qwen"))

    def _update_language_choices(self) -> None:
        """Grey out the languages the chosen model can't take.

        One that is already selected stays selected and is flagged, rather
        than quietly becoming Auto-detect, which would change what was asked
        for; Transcribe refuses it until the language or model changes.
        """
        model = self.combo_model.currentText().strip()
        items = self.combo_lang.model()
        for row in range(self.combo_lang.count()):
            supported = language_supported(model, self.combo_lang.itemData(row))
            items.item(row).setEnabled(supported)
            items.item(row).setToolTip("" if supported else f"Not supported by {model}.")
        conflict = self._language_conflict()
        self.label_lang_conflict.setToolTip(conflict or "")
        self.label_lang_conflict.setVisible(conflict is not None)

    def _language_conflict(self) -> str | None:
        """Why the chosen model can't take the chosen language, or None."""
        model = self.combo_model.currentText().strip()
        if language_supported(model, self.combo_lang.currentData()):
            return None
        language = self.combo_lang.currentText()
        return (f"{model} can't transcribe {language}. Choose another language "
                f"or Auto-detect, or a Whisper model for {language}.")

    def _make_star_button(self, combo: QComboBox, settings_key: str) -> QPushButton:
        """Flat ☆/★ toggle pinning the combo's current model as startup default."""
        btn = QPushButton("☆")
        btn.setFlat(True)
        btn.setFixedWidth(28)
        btn.setToolTip("Star the selected model to make it the default at startup.")

        def refresh() -> None:
            starred = QSettings().value(settings_key, "", str)
            btn.setText("★" if starred and starred == combo.currentText().strip() else "☆")

        def toggle() -> None:
            settings = QSettings()
            current = combo.currentText().strip()
            if settings.value(settings_key, "", str) == current:
                settings.remove(settings_key)
            else:
                settings.setValue(settings_key, current)
            refresh()

        btn.clicked.connect(toggle)
        combo.currentTextChanged.connect(lambda _text: refresh())
        refresh()
        return btn

    # ---------- Notes tab ----------

    def _build_notes_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        toggles = QHBoxLayout()
        toggles.setContentsMargins(6, 6, 6, 0)
        self.check_show_instructions = QCheckBox("Show instructions")
        self.check_show_transcript = QCheckBox("Show transcript")
        toggles.addWidget(self.check_show_instructions)
        toggles.addWidget(self.check_show_transcript)
        toggles.addStretch(1)
        page_layout.addLayout(toggles)

        split = QSplitter(Qt.Orientation.Vertical)
        top = QSplitter(Qt.Orientation.Horizontal)

        # Left: AI notes controls (model, extra instructions, actions).
        left = QWidget()
        controls = QVBoxLayout(left)
        controls.setContentsMargins(0, 0, 0, 0)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Notes model:"))
        self.combo_notes_model = QComboBox()
        self.combo_notes_model.setEditable(True)
        _add_grouped_models(self.combo_notes_model, NOTES_MODEL_GROUPS)
        self.combo_notes_model.setCurrentText(DEFAULT_NOTES_MODEL)
        self.combo_notes_model.setToolTip(
            "Local LLM (4-bit MLX, Apple GPU) that writes notes on the\n"
            "merged transcript. Bigger tiers give richer notes, slower.\n"
            "Experimental models may be slow or fail on a 16 GB Mac.\n"
            "Type any MLX repo id (e.g. from mlx-community) for other models."
        )
        model_row.addWidget(self.combo_notes_model, 1)
        self.star_notes_btn = self._make_star_button(
            self.combo_notes_model, "starred_notes_model")
        model_row.addWidget(self.star_notes_btn)
        controls.addLayout(model_row)

        self.edit_notes_instructions = QPlainTextEdit()
        self.edit_notes_instructions.setPlaceholderText(
            "Optional extra instructions, e.g. 'focus on loot and quest leads'"
        )
        self.edit_notes_instructions.setToolTip(
            "Appended to the default notes prompt — it never replaces it."
        )
        controls.addWidget(self.edit_notes_instructions, 1)

        action_row = QHBoxLayout()
        self.btn_ai_notes = QPushButton("AI Notes")
        self.btn_ai_notes.clicked.connect(self.toggle_ai_notes)
        action_row.addWidget(self.btn_ai_notes)
        self.btn_export_notes = QPushButton("Export Notes…")
        self.btn_export_notes.clicked.connect(self.export_notes_dialog)
        action_row.addWidget(self.btn_export_notes)
        action_row.addStretch(1)
        controls.addLayout(action_row)

        status_row = QHBoxLayout()
        # A text label, not a QProgressBar: macOS's Qt style never renders
        # progress-bar text, which would hide the percent and token counter.
        self.notes_progress = QLabel("")
        status_row.addWidget(self.notes_progress)
        self.notes_elapsed_label = QLabel("")
        status_row.addWidget(self.notes_elapsed_label)
        status_row.addStretch(1)
        controls.addLayout(status_row)

        self.notes_controls_panel = left
        top.addWidget(left)

        # Right: the notes editor.
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Session notes… (autosaved)")
        top.addWidget(self.notes_edit)
        top.setSizes([350, 750])
        split.addWidget(top)

        # Bottom: read-only mirror of the merged transcript.
        self.notes_transcript_view = QTextBrowser()
        self.notes_transcript_view.setPlaceholderText(
            "The merged transcript appears here once tracks are transcribed.")
        split.addWidget(self.notes_transcript_view)
        split.setSizes([420, 240])
        page_layout.addWidget(split, 1)

        # Hiding a pane collapses its splitter slot; the notes editor keeps
        # the freed space. Both default to shown, remembered across launches.
        settings = QSettings()
        self.check_show_instructions.setChecked(
            settings.value("notes_show_instructions", True, bool))
        self.check_show_transcript.setChecked(
            settings.value("notes_show_transcript", True, bool))

        def _apply_notes_toggles() -> None:
            show_instructions = self.check_show_instructions.isChecked()
            show_transcript = self.check_show_transcript.isChecked()
            self.notes_controls_panel.setVisible(show_instructions)
            self.notes_transcript_view.setVisible(show_transcript)
            s = QSettings()
            s.setValue("notes_show_instructions", show_instructions)
            s.setValue("notes_show_transcript", show_transcript)

        self.check_show_instructions.toggled.connect(_apply_notes_toggles)
        self.check_show_transcript.toggled.connect(_apply_notes_toggles)
        _apply_notes_toggles()

        # Autosave 3 s after the last keystroke (plus on close).
        self._notes_save_timer = QTimer(self)
        self._notes_save_timer.setInterval(3000)
        self._notes_save_timer.setSingleShot(True)
        self._notes_save_timer.timeout.connect(self._save_notes)
        self.notes_edit.textChanged.connect(self._on_notes_changed)

        self.notes_worker: NotesWorker | None = None
        self._notes_last_progress: tuple[int, float] | None = None  # (tokens, tok/s)
        self._notes_clock = _RunClock(self.notes_elapsed_label, self)
        return page

    def _on_notes_changed(self) -> None:
        """Typing and streamed AI notes alike: unsaved until a write succeeds."""
        self._notes_dirty = True
        self._notes_save_timer.start()

    def _save_notes(self) -> bool:
        """Store the Notes tab; True once the settings file holds what the
        editor does. Nothing changed since the last save, nothing is written.

        Synced and checked rather than left to QSettings' lazy write, which
        never reports a failure. Failures appear in the status bar: a modal
        error after every edit would be unusable. The text stays in the
        editor, still unsaved, and is retried on the next edit and on close,
        which will not go ahead without it (see _settle_unsaved_notes).
        """
        if not self._notes_dirty:
            return True
        settings = QSettings()
        settings.setValue("notes", self.notes_edit.toPlainText())
        settings.sync()
        if settings.status() != settings.Status.NoError:
            self._notes_save_error = (
                f"The settings file could not be written: {settings.fileName()}")
            self.statusBar().showMessage(
                f"Notes could not be saved: {self._notes_save_error}")
            return False
        self._notes_dirty = False
        return True

    def export_notes_dialog(self) -> None:
        if not self.notes_edit.toPlainText().strip():
            return
        self._write_notes_copy("Export notes")

    def _write_notes_copy(self, title: str) -> bool:
        """Ask for a file and write the notes to it; True once written."""
        path_str, _ = QFileDialog.getSaveFileName(
            self, title, str(Path.home() / "notes.md"),
            "Markdown (*.md);;Plain text (*.txt)")
        if not path_str:
            return False
        try:
            exporters.write_file(Path(path_str), self.notes_edit.toPlainText())
        except OSError as exc:
            diagnostics.log_exception("notes export", exc, path=path_str)
            QMessageBox.critical(self, "Export failed", str(exc))
            return False
        self.statusBar().showMessage(f"Exported {path_str}", 5000)
        return True

    # ---------- AI notes ----------

    def toggle_ai_notes(self) -> None:
        if self.notes_worker is not None:
            self.btn_ai_notes.setEnabled(False)
            self.btn_ai_notes.setText("Cancelling…")
            # Generation stops between tokens, but loading or downloading the
            # model has to run to completion first.
            self.statusBar().showMessage(
                "Cancelling notes — a model load or download in progress has "
                "to finish first…")
            self.notes_worker.stop()
            return
        segments = merge_tracks(self.tracks)
        if not segments:
            QMessageBox.information(
                self, "No transcript",
                "Transcribe something first — AI notes are written on the merged transcript.")
            return
        transcript = exporters.to_text(segments)
        if self.notes_edit.toPlainText().strip():
            self._append_notes_chunk("\n\n")
        self.notes_worker = NotesWorker(
            transcript,
            self.edit_notes_instructions.toPlainText(),
            self.combo_notes_model.currentText(),
        )
        w = self.notes_worker
        w.status.connect(self.statusBar().showMessage)
        w.download_progress.connect(partial(self._on_download_progress, w.model_name))
        w.chunk.connect(self._append_notes_chunk)
        w.progress.connect(self._on_notes_progress)
        w.completed.connect(self._on_notes_done)
        w.failed.connect(self._on_notes_failed)
        w.finished.connect(self._on_thread_finished)
        # ASR model and notes LLM must never be loaded together: the worker
        # frees the ASR cache before loading, and Transcribe stays disabled
        # until notes finish (the reverse holds in toggle_transcription).
        self.btn_transcribe.setEnabled(False)
        self.btn_ai_notes.setText("Cancel notes")
        self.notes_progress.setText(f"0% · {_token_readout(0, 0.0)}")
        self._last_summary = ""
        self._notes_last_progress = None
        self._notes_clock.start()
        w.start()

    def _append_notes_chunk(self, text: str) -> None:
        cursor = self.notes_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.notes_edit.setTextCursor(cursor)
        self.notes_edit.ensureCursorVisible()

    def _on_notes_progress(self, percent: int, tokens: int, tps: float) -> None:
        self._notes_last_progress = (tokens, tps)
        self.notes_progress.setText(f"{percent}% · {_token_readout(tokens, tps)}")

    def _finish_notes_run(self) -> str:
        if self.notes_worker is not None:
            self.notes_worker.wait()
            self.notes_worker = None
        elapsed = self._notes_clock.stop()
        self.btn_ai_notes.setText("AI Notes")
        self.btn_ai_notes.setEnabled(True)
        self.btn_transcribe.setEnabled(self.worker is None)
        self._save_notes()
        self._quit_if_idle()
        return elapsed

    def _on_notes_done(self) -> None:
        elapsed = self._finish_notes_run()
        msg = "AI notes done"
        if elapsed:
            msg += f" — took {elapsed}"
        if self._notes_last_progress is not None:
            msg += " · " + _token_readout(*self._notes_last_progress)
        self._set_summary(msg)

    def _on_notes_failed(self, error: str) -> None:
        elapsed = self._finish_notes_run()
        self.notes_progress.setText("")
        self._set_summary(f"AI notes failed — took {elapsed}" if elapsed
                          else "AI notes failed")
        if not self._quit_requested:
            QMessageBox.warning(self, "AI notes failed", error)

    # ---------- Drag & drop ----------

    @staticmethod
    def _dropped_audio(event) -> list[Path]:
        return audio_paths(url.toLocalFile() for url in event.mimeData().urls())

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if not self._tracks_locked() and self._dropped_audio(event):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        self.add_files(self._dropped_audio(event))

    # ---------- Track management ----------

    def add_files_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add audio tracks", "", FILE_FILTER)
        self.add_files([Path(f) for f in files])

    def add_files(self, paths: list[Path]) -> None:
        if self._tracks_locked():
            # Files opened from Finder arrive here whatever is running.
            if paths:
                self.statusBar().showMessage(
                    "Files not added: the track list can't change while transcription "
                    "or Align is running. Add them again once it has finished.", 8000)
            return
        existing = self._speaker_file_keys()
        reference = self.session_reference
        reference_key = _canonical(reference.path) if reference is not None else None
        refused = None
        unreadable: list[Track] = []
        for path in paths:
            key = _canonical(path)
            if key in existing or not path.is_file():
                continue
            if key == reference_key:
                refused = path
                continue
            for track in tracks_for_file(path):
                self.tracks.append(track)
                self._append_row(track)
                if track.probe_error:
                    unreadable.append(track)
            existing.add(key)
        self.statusBar().showMessage(f"{len(self.tracks)} track(s) loaded", 5000)
        if refused is not None:
            QMessageBox.information(
                self, "That's the session reference",
                f"{refused.name} is the session reference, which is used for timing "
                f"only and never transcribed, so it can't also be a speaker file.\n\n"
                f"To transcribe it, click Remove reference above the track list, "
                f"then add it again.")
        if unreadable:
            detail = "\n\n".join(f"{t.path.name}:\n{t.probe_error}"
                                 for t in unreadable[:5])
            QMessageBox.warning(
                self, "Could not read these files",
                "They were added anyway as a single track each, but they will "
                "probably fail to transcribe:\n\n" + detail)

    def _speaker_file_keys(self) -> set:
        """Every speaker file, `_canonical`, looked up once per file rather
        than once per row: that can be slow on a network share."""
        return {_canonical(path) for path in {t.path for t in self.tracks}}

    def _append_row(self, track: Track, row: int | None = None) -> None:
        if row is None:
            row = self.table.rowCount()
        with QSignalBlocker(self.table):
            self.table.insertRow(row)

            speaker_item = QTableWidgetItem(track.speaker)
            self.table.setItem(row, COL_SPEAKER, speaker_item)
            offset_item = QTableWidgetItem(format_offset(track.offset))
            offset_item.setToolTip(OFFSET_SOURCE_TIPS[track.offset_source])
            self.table.setItem(row, COL_OFFSET, offset_item)

            for col, text in [
                (COL_FILE, track.source_label),
                (COL_DURATION, exporters.hms(track.duration)),
                (COL_STATUS, track.status.value),
            ]:
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col == COL_FILE:
                    item.setToolTip(str(track.path))
                elif col == COL_STATUS and track.probe_error:
                    item.setToolTip("This file could not be probed:\n" + track.probe_error)
                self.table.setItem(row, col, item)

            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(0)
            self.table.setCellWidget(row, COL_PROGRESS, progress)

    def _show_offset(self, row: int) -> None:
        """Write a track's offset into its cell, in standard form."""
        track = self.tracks[row]
        item = self.table.item(row, COL_OFFSET)
        item.setText(format_offset(track.offset))
        item.setToolTip(OFFSET_SOURCE_TIPS[track.offset_source])

    def _set_offset(self, row: int, offset: float, source: OffsetSource) -> None:
        track = self.tracks[row]
        track.offset, track.offset_source = offset, source
        self._show_offset(row)

    def _on_offset_edited(self, row: int, text: str) -> None:
        """Apply an Offset edit committed through _OffsetDelegate, the only way
        one arrives, or put the old value back."""
        if not 0 <= row < len(self.tracks):
            return
        try:
            offset = parse_offset(text)
        except ValueError as exc:
            self.statusBar().showMessage(f"Offset not changed: {exc}", 8000)
            self._show_offset(row)
            return
        self._set_offset(row, offset, OffsetSource.MANUAL)
        self._schedule_refresh()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != COL_SPEAKER:
            return
        row = item.row()
        if not 0 <= row < len(self.tracks):
            return
        track = self.tracks[row]
        new_name = item.text().strip()
        if not new_name:
            # A blank cell used to leave the track named and the table empty.
            with QSignalBlocker(self.table):
                item.setText(track.speaker)
            self.statusBar().showMessage("A speaker needs a name — kept "
                                         f"'{track.speaker}'.", 4000)
            return
        track.speaker = new_name  # merge_tracks names the segments after it
        self._schedule_refresh()

    def _show_table_menu(self, pos) -> None:
        row = self.table.rowAt(pos.y())
        if row < 0 or self._tracks_locked():
            return
        track = self.tracks[row]
        menu = QMenu(self)
        split = menu.addAction("Split into one speaker per channel")
        split.setEnabled(track.channel is None)
        if menu.exec(self.table.viewport().mapToGlobal(pos)) is split:
            self._split_track_channels(row)

    def _split_track_channels(self, row: int) -> None:
        if self._tracks_locked():
            return
        track = self.tracks[row]
        try:
            streams = probe_streams(track.path)
            channels = streams[track.stream_index or 0]["channels"]
        except Exception as exc:
            QMessageBox.warning(self, "Cannot split", str(exc))
            return
        if channels < 2:
            QMessageBox.information(self, "Mono track",
                                    "This track has only one channel — nothing to split.")
            return
        self.table.removeRow(row)
        del self.tracks[row]
        for ch, new in enumerate(channel_tracks(
                track.path, track.speaker, channels, duration=track.duration,
                stream_index=track.stream_index, offset=track.offset,
                offset_source=track.offset_source)):
            self.tracks.insert(row + ch, new)
            self._append_row(new, row + ch)
        self._schedule_refresh()

    def remove_selected(self) -> None:
        if self._tracks_locked():
            return
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
            del self.tracks[row]
        self._prune_align_fingerprints()
        self._schedule_refresh()

    def clear_tracks(self) -> None:
        if self._tracks_locked():
            return
        # Clearing is the one button that can destroy hours of transcription
        # in a single click, so it asks once there is something to lose.
        if any(t.segments for t in self.tracks) and not self._confirm(
                "Clear the transcript?",
                "This removes every track and the merged transcript.\n"
                "Export first if you want to keep it."):
            return
        self.table.setRowCount(0)
        self.tracks.clear()
        self.session_reference = None
        self._show_session_reference()
        self._prune_align_fingerprints()
        self._render_transcript()  # empties both views; the notes text itself stays
        self.btn_export.setEnabled(False)

    def _prune_align_fingerprints(self) -> None:
        """Let go of what Align read from files no longer loaded: a four-hour
        file's fingerprint is about 11 MB, kept otherwise until the next Align."""
        loaded = {t.path for t in self.tracks}
        if self.session_reference is not None:
            loaded.add(self.session_reference.path)
        self._align_fingerprints = {key: kept for key, kept in self._align_fingerprints.items()
                                    if key[0] in loaded}

    # ---------- Session reference ----------

    def choose_session_reference(self) -> None:
        if self._aligning():
            return
        file, _ = QFileDialog.getOpenFileName(
            self, "Choose a recording of the whole session", "", FILE_FILTER)
        if file:
            self.set_session_reference(Path(file))

    def set_session_reference(self, path: Path) -> bool:
        """Use `path` as Align's session reference, or say why not and keep the old one."""
        if self._aligning():
            return False
        key = _canonical(path)
        if key in self._speaker_file_keys():
            QMessageBox.information(
                self, "Already a speaker file",
                f"{path.name} is in the track list as a speaker file, so it can't "
                f"also be the session reference.\n\nTo use it as the reference, "
                f"remove it from the track list first.")
            return False
        try:
            streams = probe_streams(path)
            if not streams:
                raise ValueError("it has no audio in it")
        except Exception as exc:
            QMessageBox.warning(
                self, "Can't use that file",
                f"{path.name} can't be read as audio: {exc}\n\n"
                f"The session reference was not changed.")
            return False
        self.session_reference = SessionReference(
            path, max(s["duration"] for s in streams), len(streams))
        self._show_session_reference()
        self._prune_align_fingerprints()
        self.statusBar().showMessage(
            f"Session reference: {path.name}. Click Align… to time the speaker "
            f"files against it; offsets already set stay as they are.", 10000)
        return True

    def remove_session_reference(self) -> None:
        if self._aligning() or self.session_reference is None:
            return
        self.session_reference = None
        self._show_session_reference()
        self._prune_align_fingerprints()
        self.statusBar().showMessage(
            "Session reference removed. Offsets already set stay as they are.", 8000)

    def _show_session_reference(self) -> None:
        reference = self.session_reference
        if reference is None:
            self.reference_label.setText(
                "<span style='color:gray'>Optional: a recording of the whole session "
                "for Align to time the speaker files against.</span>")
        else:
            self.reference_label.setText(
                f"<b>{html.escape(reference.path.name)}</b> "
                f"({exporters.hms(reference.duration)}) — Timing reference only; "
                f"individual files are transcribed")
        self.btn_reference.setText("Replace…" if reference else "Add session reference…")
        self.btn_remove_reference.setVisible(reference is not None)
        self._update_busy()

    # ---------- Busy state ----------

    def _aligning(self) -> bool:
        """From Align until its review window closes: the reference stays put,
        so the review's "Reference:" line still names what it was matched to."""
        return self.align_worker is not None or self._align_reviewing

    def _tracks_locked(self) -> bool:
        """Whether the track list must stay as it is: transcription indexes into
        it, and Align's results are matched back to it by path."""
        return self.worker is not None or self._aligning()

    def _update_busy(self) -> None:
        """Enable the controls the running jobs share; either job can lock them."""
        locked = self._tracks_locked()
        for widget in (self.btn_add, self.btn_remove, self.btn_clear):
            widget.setEnabled(not locked)
        aligning = self._aligning()
        self.btn_reference.setEnabled(not aligning)
        self.btn_remove_reference.setEnabled(not aligning)
        if self.align_worker is None:
            self.btn_align.setText("Align…")
            self.btn_align.setEnabled(not self._align_reviewing)

    # ---------- Alignment ----------

    def _alignment_units(self) -> list[tuple[Path, list[int | None]]]:
        """One (path, streams) pair per speaker file: its rows are already in
        step with each other, so the file is matched once, as a whole."""
        units: dict[Path, list[int | None]] = {}
        for track in self.tracks:
            streams = units.setdefault(track.path, [])
            if track.stream_index not in streams:
                streams.append(track.stream_index)
        return list(units.items())

    def toggle_alignment(self) -> None:
        if self._quit_requested or self._align_reviewing:
            return
        if self.align_worker is not None:
            self.btn_align.setEnabled(False)
            self.btn_align.setText("Cancelling…")
            self.align_worker.stop()
            return
        units = self._alignment_units()
        reference = self.session_reference
        if not units:
            QMessageBox.information(
                self, "Nothing to align",
                "Add the speaker files first. Align works out when each one started, "
                "from sound they share with each other or with a session reference.")
            return
        if reference is None and len(units) < 2:
            QMessageBox.information(
                self, "Nothing to align",
                f"Every track comes from {units[0][0].name}, so they are already in "
                f"step with each other. To place it on another recording's clock, add "
                f"that recording as a session reference.")
            return
        offsets: dict[Path, float] = {}
        for track in self.tracks:
            offsets.setdefault(track.path, track.offset)
        reference_unit = ((reference.path, reference.stream_indices)
                          if reference is not None else None)
        # Only what this run can use is kept: a file since removed lets go of its own.
        jobs = units + ([reference_unit] if reference_unit is not None else [])
        wanted = {(path, tuple(streams)) for path, streams in jobs}
        self._align_fingerprints = {key: kept for key, kept in self._align_fingerprints.items()
                                    if key in wanted}
        self.align_worker = AlignWorker(units, offsets, reference_unit,
                                        known=self._align_fingerprints)
        w = self.align_worker
        w.progress.connect(lambda _percent, message: self.statusBar().showMessage(message))
        w.finished.connect(self._on_align_finished)
        self.btn_align.setText("Cancel align")
        self._update_busy()
        diagnostics.log_event("align started", files=len(units),
                              session_reference=reference is not None)
        w.start()

    @Slot()
    def _on_align_finished(self) -> None:
        worker = self.align_worker
        if worker is None:
            return
        worker.wait()
        self.align_worker = None
        self._align_fingerprints.update(worker.fingerprints)
        if self._quit_requested:
            self._update_busy()
            self._quit_if_idle()
            return
        result = worker.result
        if result is None:      # failed or cancelled
            self._update_busy()
            if worker.error:
                diagnostics.log_event("align failed", error=worker.error)
                self.statusBar().showMessage("Align stopped. No offsets were changed.", 8000)
                QMessageBox.warning(self, "Align", worker.error)
            else:
                self.statusBar().showMessage(
                    "Align cancelled. No offsets were changed.", 8000)
            return
        self._align_reviewing = True
        self._update_busy()
        self.statusBar().showMessage("Align finished: review the suggested offsets.")
        self._review_alignment(result)

    def _review_alignment(self, result: AlignmentResult) -> None:
        """Open the review window once nothing else modal is open, then apply."""
        if self._quit_requested:
            self._align_reviewing = False
            self._update_busy()
            return
        if QApplication.activeModalWidget() is not None:
            # Over another prompt (Export's overwrite question, say) Apply
            # would change offsets underneath it; wait for it to close.
            QTimer.singleShot(250, self, lambda: self._review_alignment(result))
            return
        # An Offset typed but not yet committed is the user's: make it count
        # before "Now" and the typed-value protection are read.
        self._offset_delegate.finish_open_edit()
        result = self._on_current_anchor(result)
        try:
            dialog = AlignReviewDialog(result, self.tracks, self)
            accepted = bool(dialog.exec())
            chosen = dialog.chosen()
            dialog.deleteLater()
        finally:
            self._align_reviewing = False
            self._update_busy()
        if self._quit_requested:
            return
        if not accepted:
            self.statusBar().showMessage("No offsets were changed.", 8000)
            return
        self._apply_alignment(chosen)

    def _on_current_anchor(self, result: AlignmentResult) -> AlignmentResult:
        """Without a session reference, hang the suggestions from the reference
        file's offset as it is now: it may have been typed while Align ran."""
        rows = [t for t in self.tracks if t.path == result.reference]
        if result.mode == SESSION_REFERENCE or not rows or rows[0].offset == result.anchor:
            return result
        moved = rebase(result, rows[0].offset)
        moved.note = (f"{result.reference.name}'s offset changed while Align ran, so these "
                      f"are measured from its new value, {format_offset(rows[0].offset)}. "
                      + moved.note).strip()
        return moved

    def _apply_alignment(self, chosen: list) -> None:
        """Write ticked suggestions to every row of their file."""
        applied = 0
        anchor_rows: list[int] = []
        for suggestion in chosen:
            rows = [i for i, t in enumerate(self.tracks) if t.path == suggestion.path]
            if suggestion.method == "reference" and all(
                    self.tracks[i].offset == suggestion.offset for i in rows):
                anchor_rows = rows    # didn't move: the value stays, and so does a typed one
                continue
            source = OffsetSource.GUESS if suggestion.method == "guess" else OffsetSource.MATCHED
            for i in rows:
                self._set_offset(i, suggestion.offset, source)
            applied += 1
        if applied:
            # The others now hang from the anchor's value: it is part of the
            # alignment, so a later run counts it as placed (see the review).
            for i in anchor_rows:
                if self.tracks[i].offset_source is OffsetSource.DEFAULT:
                    self._set_offset(i, self.tracks[i].offset, OffsetSource.MATCHED)
        self._schedule_refresh()
        self.statusBar().showMessage(
            f"Applied {applied} offset(s)." if applied else "No offsets were changed.", 8000)

    # ---------- Transcription ----------

    def _current_config(self) -> RunConfig:
        """The settings a run started right now would use (backend not known yet)."""
        return RunConfig(
            model=self.combo_model.currentText().strip(),
            language=self.combo_lang.currentData() or "",
            device=self.combo_device.currentData(),
            hints=self.edit_hints.toPlainText().strip(),
        )

    def _indices_to_run(self, wanted: RunConfig) -> list[int] | None:
        """Which tracks this run should cover, or None if the user backed out.

        Two things can go wrong silently here. Re-running when everything is
        already done throws away a finished transcript on one stray click, and
        transcribing only the leftovers after the model or language changed
        stitches together a transcript whose tracks came from different
        settings. Both now ask.
        """
        unfinished = [i for i, t in enumerate(self.tracks) if t.status != TrackStatus.DONE]
        finished = [i for i, t in enumerate(self.tracks) if t.status == TrackStatus.DONE]

        if not unfinished:
            if not self._confirm(
                    "Transcribe everything again?",
                    f"All {len(finished)} track(s) are already transcribed.\n\n"
                    "Running again replaces the existing transcript. The current "
                    "one is kept until the new run succeeds."):
                return None
            return list(range(len(self.tracks)))

        if not finished:
            return unfinished

        changed = sorted({
            name
            for i in finished
            if self.tracks[i].config is not None
            for name in wanted.differs_from(self.tracks[i].config)
        })
        if not changed:
            return unfinished

        choice = self._choose(
            "Settings changed since the finished tracks",
            "The " + ", ".join(changed) + " changed since "
            f"{len(finished)} track(s) were transcribed.",
            "Transcribing only the remaining tracks would mix results from "
            "different settings in one transcript.",
            [("Redo all tracks", QMessageBox.ButtonRole.AcceptRole, "all"),
             ("Only the remaining tracks", QMessageBox.ButtonRole.DestructiveRole,
              "remaining"),
             (QMessageBox.StandardButton.Cancel, None, None)],
            default="all")
        if choice == "all":
            return list(range(len(self.tracks)))
        if choice == "remaining":
            return unfinished
        return None

    def toggle_transcription(self) -> None:
        if self.worker is not None:
            self.btn_transcribe.setEnabled(False)
            self.btn_transcribe.setText("Cancelling…")
            # Cancellation is a flag the worker checks between units of work;
            # it cannot interrupt a blocking native call already under way.
            self.statusBar().showMessage(
                "Cancelling — waiting for the current step to finish "
                "(a track, a speech region, or a model download cannot be "
                "interrupted part-way)…")
            self.worker.stop()
            return

        if not self.tracks:
            QMessageBox.information(self, "No tracks", "Add at least one audio track first.")
            return
        # Before asking which tracks to run, so nothing is asked and then refused.
        conflict = self._language_conflict()
        if conflict is not None:
            QMessageBox.warning(self, "Language not supported", conflict)
            return

        wanted = self._current_config()
        indices = self._indices_to_run(wanted)
        if indices is None:
            return

        for i in indices:
            track = self.tracks[i]
            # Keep any finished result aside instead of deleting it now: a
            # cancelled, failed or aborted rerun restores it (models.Track).
            track.stash_result()
            track.status = TrackStatus.PENDING
            track.error = ""
            track.note = ""
            self._set_status(i, track.status.value)
            self._progress_bar(i).setValue(0)

        self.worker = TranscribeWorker(
            self.tracks, indices,
            model_name=wanted.model,
            device=wanted.device,
            language=wanted.language or None,
            hotwords=wanted.hints,
        )
        w = self.worker
        w.model_status.connect(self.statusBar().showMessage)
        w.download_progress.connect(partial(self._on_download_progress, w.model_name))
        w.track_started.connect(self._on_track_started)
        w.segment_ready.connect(self._on_segment)
        w.track_progress.connect(lambda i, p: self._progress_bar(i).setValue(p))
        w.track_done.connect(self._on_track_done)
        w.track_failed.connect(self._on_track_failed)
        w.track_note.connect(self._on_track_note)
        w.finished_all.connect(self._on_finished)
        w.finished.connect(self._on_thread_finished)
        diagnostics.log_event("transcription started", model=wanted.model,
                              device=wanted.device, language=wanted.language,
                              tracks=len(indices))
        self._set_running(True)
        self._last_summary = ""
        self._run_clock.start()
        w.start()

    def _set_running(self, running: bool) -> None:
        for widget in (self.combo_model, self.combo_lang, self.combo_device,
                       self.edit_hints):
            widget.setEnabled(not running)
        self._update_busy()
        self.btn_export.setEnabled(not running and bool(merge_tracks(self.tracks)))
        self.btn_ai_notes.setEnabled(not running and self.notes_worker is None)
        self.btn_transcribe.setText("Cancel" if running else "Transcribe")
        self.btn_transcribe.setEnabled(True)

    def _progress_bar(self, row: int) -> QProgressBar:
        return self.table.cellWidget(row, COL_PROGRESS)

    def _set_status(self, row: int, text: str) -> None:
        with QSignalBlocker(self.table):
            self.table.item(row, COL_STATUS).setText(text)

    @Slot(int)
    def _on_track_started(self, i: int) -> None:
        track = self.tracks[i]
        track.segments.clear()  # a CPU-fallback retry restarts the track from scratch
        track.note = ""
        track.status = TrackStatus.TRANSCRIBING
        self._set_status(i, TrackStatus.TRANSCRIBING.value)
        self._progress_bar(i).setValue(0)
        self.statusBar().showMessage(f"Transcribing {track.path.name}…")
        self._schedule_refresh()

    @Slot(int, object)
    def _on_segment(self, i: int, segment) -> None:
        self.tracks[i].segments.append(segment)
        self._schedule_refresh()

    def _show_done(self, i: int) -> None:
        track = self.tracks[i]
        label = TrackStatus.DONE.value + (f" ({track.language})" if track.language else "")
        self._set_status(i, label + (" ⚠" if track.note else ""))
        tip = [track.config.summary()] if track.config else []
        if track.note:
            tip.append(track.note)
        self.table.item(i, COL_STATUS).setToolTip("\n\n".join(tip))

    @Slot(int, str)
    def _on_track_note(self, i: int, note: str) -> None:
        self.tracks[i].note = note

    @Slot(int, str, object)
    def _on_track_done(self, i: int, language: str, config) -> None:
        track = self.tracks[i]
        track.status = TrackStatus.DONE
        track.language = language
        track.config = config
        track.commit_result()  # the replacement landed; drop the stashed one
        self._show_done(i)

    @Slot(int, str)
    def _on_track_failed(self, i: int, message: str) -> None:
        track = self.tracks[i]
        track.error = message
        partial = len(track.segments)
        if track.restore_result():
            # A rerun failed, but the earlier successful transcript survived.
            self._show_done(i)
            tip = f"The rerun failed; the previous transcript was kept.\n\n{message}"
            if track.note:
                tip += f"\n\n{track.note}"
            self.table.item(i, COL_STATUS).setToolTip(tip)
            self._schedule_refresh()
            return
        track.status = TrackStatus.ERROR
        self._set_status(i, TrackStatus.ERROR.value)
        tip = message
        if partial:
            tip += (f"\n\n{partial} partial segment(s) were produced before the "
                    "failure. They are left out of the transcript and exports "
                    "because the track is incomplete.")
        self.table.item(i, COL_STATUS).setToolTip(tip)
        self._schedule_refresh()

    @Slot(bool)
    def _on_finished(self, cancelled: bool) -> None:
        if self.worker is not None:
            self.worker.wait()
            self.worker = None
        if cancelled:
            for i, track in enumerate(self.tracks):
                if track.status == TrackStatus.TRANSCRIBING:
                    track.status = TrackStatus.CANCELLED
                    self._set_status(i, TrackStatus.CANCELLED.value)
        # Any track still holding a stashed result never got a replacement —
        # the run was cancelled or aborted before reaching it. Put it back.
        for i, track in enumerate(self.tracks):
            if track.previous is not None and track.status != TrackStatus.DONE:
                track.restore_result()
                self._show_done(i)
                self._progress_bar(i).setValue(100)
        errors = [t for t in self.tracks if t.status == TrackStatus.ERROR]
        discarded = discarded_segment_count(self.tracks)
        elapsed = self._run_clock.stop()
        took = f" — took {elapsed}" if elapsed else ""
        done = sum(1 for t in self.tracks if t.status == TrackStatus.DONE)
        if cancelled:
            msg = f"Transcription cancelled{took}"
        elif errors:
            msg = f"Transcription finished with {len(errors)} error(s){took}"
        else:
            msg = f"Transcription complete{took} · {done} track(s)"
        if discarded:
            msg += f" · {discarded} partial segment(s) excluded"
        self._set_summary(msg)
        diagnostics.log_event("transcription finished", cancelled=cancelled,
                              done=done, errors=len(errors), discarded=discarded)
        self._set_running(False)
        self._render_transcript()
        self._quit_if_idle()
        if errors and not cancelled and not self._quit_requested:
            detail = "\n\n".join(f"{t.source_label}:\n{t.error}" for t in errors[:5])
            if discarded:
                detail += (f"\n\nPartial output from failed tracks ({discarded} "
                           "segment(s)) is left out of the transcript and exports.")
            detail += f"\n\nFull details: {diagnostics.log_path()}"
            QMessageBox.warning(self, "Some tracks failed", detail)

    # ---------- Transcript view ----------

    def _schedule_refresh(self) -> None:
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def _render_transcript(self) -> None:
        self._refresh_timer.stop()  # this is the refresh a pending one would do
        parts = []
        for seg in coalesce(merge_tracks(self.tracks)):
            ts = exporters.hms(seg.start)
            parts.append(
                f"<p><b>{html.escape(seg.speaker)}</b> "
                f"<span style='color:gray'>[{ts}]</span>: "
                f"{html.escape(seg.text.strip())}</p>"
            )
        html_text = "\n".join(parts)
        if html_text == self._transcript_html:
            return
        self._transcript_html = html_text
        self._stale_views = {self.transcript_view, self.notes_transcript_view}
        self._sync_transcript_views()

    def _sync_transcript_views(self) -> None:
        """Draw the transcript into the view on screen, if it is behind.

        Only one of the two can be seen at a time, and drawing a long session
        takes a fifth of a second of the GUI thread, so the other waits until
        its tab is opened (or the Notes tab's transcript pane is shown).
        """
        if self.tabs.currentIndex() == 0:
            view = self.transcript_view
        elif self.check_show_transcript.isChecked():
            view = self.notes_transcript_view
        else:
            return
        if view not in self._stale_views:
            return
        self._stale_views.discard(view)
        if not self._transcript_html:
            view.clear()
            return
        scrollbar = view.verticalScrollBar()
        position = scrollbar.value()
        at_bottom = position >= scrollbar.maximum() - 10
        view.setHtml(self._transcript_html)
        if view is self.notes_transcript_view:
            # The notes-side mirror keeps the reader's place instead of
            # following the bottom.
            scrollbar.setValue(min(position, scrollbar.maximum()))
        elif at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    # ---------- Export ----------

    def _confirm_overwrites(self, targets: list[Path], asked_for: Path) -> bool:
        """Confirm files the save dialog did not already ask about.

        The dialog only warns about the exact name the user typed. "All
        formats" writes four more siblings, and a corrected extension writes a
        different name entirely — those overwrites were never confirmed.
        """
        surprises = [p for p in targets if p.exists() and p != asked_for]
        if not surprises:
            return True
        listed = "\n".join(f"  {p.name}" for p in surprises)
        return self._confirm(
            "Replace existing files?",
            f"These files already exist in {surprises[0].parent} and will be "
            f"replaced:\n\n{listed}")

    def _confirm_incomplete_export(self) -> bool:
        """Warn before exporting when any track isn't fully transcribed.

        A failed track is left out of the export entirely, and one that was
        cancelled, or never ran, is missing all or part of its speaker. Only
        JSON records track status; the other formats would leave that
        speaker out without a trace.
        """
        incomplete = [(t, gap) for t in self.tracks
                      if (gap := _missing_from_export(t)) is not None]
        if not incomplete:
            return True
        listed = "\n".join(f"  • {t.source_label} — {gap}"
                           for t, gap in incomplete[:5])
        if len(incomplete) > 5:
            listed += f"\n  • …and {len(incomplete) - 5} more"
        return self._confirm(
            "Incomplete transcript",
            f"{len(incomplete)} track(s) not fully transcribed:\n\n{listed}\n\n"
            "Whatever those speakers said that wasn't transcribed will be "
            "missing from the export. Export anyway?",
            warning=True)

    def export_dialog(self) -> None:
        if not merge_tracks(self.tracks):
            return
        if not self._confirm_incomplete_export():
            return
        default_name = str(Path.home() / "transcript.md")
        path_str, selected = QFileDialog.getSaveFileName(
            self, "Export transcript", default_name, EXPORT_FILTERS
        )
        if not path_str:
            return
        asked_for = Path(path_str)
        path = asked_for
        # The exporters merge the tracks as they write, after every prompt
        # here: Align's review can open over these prompts and move offsets.
        try:
            if selected.startswith("All formats"):
                targets = [path.with_suffix(ext) for ext in exporters.EXPORTERS]
                if not self._confirm_overwrites(targets, asked_for):
                    return
                written = exporters.export_all(path, self.tracks)
                self.statusBar().showMessage(
                    "Exported: " + ", ".join(p.name for p in written), 5000)
            else:
                if path.suffix.lower() not in exporters.EXPORTERS:
                    ext = selected.split("*")[1].rstrip(") ")
                    path = path.with_suffix(ext)
                if not self._confirm_overwrites([path], asked_for):
                    return
                exporters.export(path, self.tracks)
                self.statusBar().showMessage(f"Exported {path}", 5000)
        except exporters.ExportError as exc:
            written = ", ".join(p.name for p in exc.written)
            diagnostics.log_exception("export", exc, path=str(path), written=written)
            # Some formats can be new and the rest old: say so, not just "failed".
            if exc.written:
                self.statusBar().showMessage("Partly exported: " + written)
            QMessageBox.critical(
                self, "Export incomplete" if exc.written else "Export failed", str(exc))
        except OSError as exc:
            diagnostics.log_exception("export", exc, path=str(path))
            QMessageBox.critical(self, "Export failed", str(exc))

    # ---------- Settings persistence ----------

    def _load_settings(self) -> None:
        settings = QSettings()
        self.edit_hints.setPlainText(settings.value("hints", "", str))
        self._restore_model(settings, self.combo_model, "model")
        lang_index = self.combo_lang.findData(settings.value("language", None))
        if lang_index >= 0:
            self.combo_lang.setCurrentIndex(lang_index)
        self._update_hints_visibility()
        self._update_language_choices()
        # Loading is not a change: it must not mark the notes unsaved or
        # schedule a rewrite of what was just read.
        with QSignalBlocker(self.notes_edit):
            self.notes_edit.setPlainText(settings.value("notes", "", str))
        self._restore_model(settings, self.combo_notes_model, "notes_model")
        self.edit_notes_instructions.setPlainText(
            settings.value("notes_instructions", "", str))
        if settings.contains("dark"):
            dark = settings.value("dark", False, bool)
            with QSignalBlocker(self.dark_checkbox):
                self.dark_checkbox.setChecked(dark)
            self._apply_color_scheme(dark)

    @staticmethod
    def _restore_model(settings: QSettings, combo: QComboBox, key: str) -> None:
        """A starred model (see _make_star_button) beats the last-used one."""
        model = settings.value(f"starred_{key}", "", str) or settings.value(key, "", str)
        if model:
            combo.setCurrentText(model)

    def _save_settings(self) -> None:
        settings = QSettings()
        settings.setValue("hints", self.edit_hints.toPlainText())
        settings.setValue("model", self.combo_model.currentText().strip())
        settings.setValue("language", self.combo_lang.currentData())
        settings.setValue("notes_model", self.combo_notes_model.currentText().strip())
        settings.setValue(
            "notes_instructions", self.edit_notes_instructions.toPlainText())
        # The notes are saved on their own, so that a failure can stop the
        # close — see _save_notes.

    # ---------- Shutdown ----------

    def _unfinished_jobs(self) -> list:
        """Jobs whose finish handler has not run yet.

        Not the same as threads still running: a notes run's last chunks and
        its `completed` signal are still queued behind a thread that has
        already stopped, and closing then would lose them. Each finish
        handler drops the window's reference to its job.
        """
        return [w for w in (self.worker, self.notes_worker, self.align_worker)
                if w is not None]

    def _quit_if_idle(self) -> None:
        """Finish a deferred quit once the last job's finish handler has run.

        Deferred to the next event-loop turn so the handler that called this
        can finish before the window goes away.
        """
        if self._quit_requested and not self._unfinished_jobs():
            QTimer.singleShot(0, self.close)

    @Slot()
    def _on_thread_finished(self) -> None:
        """Backstop for a run that ended without sending its finish signal.

        QThread.finished is queued behind every signal the run emitted, so
        by now the job's own handler has normally run and dropped it. A run
        that died first would otherwise leave its controls stuck and a quit
        waiting forever (see _unfinished_jobs).
        """
        job = self.sender()
        if job is None:
            return
        if job is self.worker:
            diagnostics.log_event("transcription thread ended without finishing")
            self._on_finished(True)
        elif job is self.notes_worker:
            diagnostics.log_event("notes thread ended without finishing")
            self._on_notes_failed(
                f"AI notes stopped unexpectedly.\n\nDetails: {diagnostics.log_path()}")

    def closeEvent(self, event) -> None:
        """Close only once every job is over and the notes are saved.

        Waiting a fixed few seconds and accepting the close regardless used to
        leave live QThreads attached to a destroyed window — a crash or hang.
        Several steps genuinely outlast any timeout: mlx-whisper transcribes a
        whole track in one blocking call, a Qwen speech region runs up to
        240 s, and model downloads are unbounded. None of them can be
        interrupted, so the window waits instead, and the user is offered an
        explicit way out. A failed final notes save keeps the window open
        behind one prompt rather than losing what is only in the editor.
        """
        if self._asking_about_notes:
            event.ignore()  # another Quit, or the deferred close, under the prompt
            return
        self._save_settings()
        jobs = self._unfinished_jobs()
        if not jobs:
            self._close_with_notes_saved(event)
            return

        if not self._quit_requested:
            self._quit_requested = True
            self._save_notes()  # a first copy now, should stopping take long
            for job in jobs:
                job.stop()
            self.statusBar().showMessage(
                "Quitting — waiting for the current step to finish…")

        # A short wait covers the common case: between two tracks, or mid
        # generation, the stop flag is noticed almost immediately. The close
        # still waits for the finish handlers, which call _quit_if_idle():
        # the last notes chunks are queued behind the stopped thread.
        if all(job.wait(2000) for job in jobs):
            event.ignore()
            return

        choice = self._choose(
            "Still working", "A step that cannot be interrupted is still running.",
            "Transcribing a track, a Qwen speech region, and downloading a "
            "model all have to finish before the app can close cleanly.\n\n"
            "The window closes on its own as soon as that happens.",
            [("Wait for it", QMessageBox.ButtonRole.AcceptRole, "wait"),
             ("Quit now", QMessageBox.ButtonRole.DestructiveRole, "quit")],
            default="wait")
        if choice != "wait":
            # Nothing can unblock a native call already in flight, and tearing
            # down running QThreads crashes Qt. Leave the process instead —
            # settings are saved above, and the notes are saved now or the
            # user has said what to do with them.
            if not self._notes_ready_to_close():
                event.ignore()
                return
            diagnostics.log_event("force quit while workers were running")
            os._exit(0)
        # The finish handlers call _quit_if_idle(), so no polling is needed.
        event.ignore()

    def _close_with_notes_saved(self, event) -> None:
        """Nothing is running: the last notes save, then the close itself."""
        self._notes_save_timer.stop()
        if not self._notes_ready_to_close():
            event.ignore()
            return
        self.stats_poller.stop()
        self.stats_poller.wait(3000)
        event.accept()

    def _notes_ready_to_close(self) -> bool:
        """The final notes save, or the user's say on notes it could not
        write; False once the close has been called off."""
        if self._save_notes() or self._settle_unsaved_notes():
            return True
        self._call_off_close()
        return False

    def _settle_unsaved_notes(self) -> bool:
        """The final save failed: True to close once the notes are written
        after all, written elsewhere, or knowingly discarded."""
        self._asking_about_notes = True
        try:
            while True:
                choice = self._ask_about_unsaved_notes()
                if choice == "retry" and self._save_notes():
                    return True
                if choice == "elsewhere" and self._write_notes_copy("Save notes elsewhere"):
                    return True
                if choice == "discard":
                    return True
                if choice == "return":
                    return False
        finally:
            self._asking_about_notes = False

    def _ask_about_unsaved_notes(self) -> str:
        """One prompt: "retry", "elsewhere", "return" or "discard"."""
        return self._choose(
            "Notes not saved", "Your latest notes could not be saved.",
            f"{self._notes_save_error}\n\nClosing now would lose them.",
            [("Retry", QMessageBox.ButtonRole.AcceptRole, "retry"),
             ("Save Elsewhere…", QMessageBox.ButtonRole.ActionRole, "elsewhere"),
             ("Return to App", QMessageBox.ButtonRole.RejectRole, "return"),
             ("Close Without Saving", QMessageBox.ButtonRole.DestructiveRole, "discard")],
            default="retry", escape="return") or "return"

    # ---------- Prompts ----------

    def _confirm(self, title: str, text: str, warning: bool = False) -> bool:
        """Ask Yes or Cancel, Cancel the default; True for Yes."""
        ask = QMessageBox.warning if warning else QMessageBox.question
        buttons = QMessageBox.StandardButton
        return ask(self, title, text, buttons.Yes | buttons.Cancel,
                   buttons.Cancel) == buttons.Yes

    def _choose(self, title: str, text: str, informative: str, choices: list,
                default: str, escape: str | None = None) -> str | None:
        """A warning with its own buttons: the key of the one pressed, or None.

        `choices` holds (label, role, key), or (standard button, None, key).
        `default` and `escape` name buttons by key.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(text)
        box.setInformativeText(informative)
        buttons = [(box.addButton(label) if role is None else box.addButton(label, role), key)
                   for label, role, key in choices]
        keyed = {key: button for button, key in buttons}
        box.setDefaultButton(keyed[default])
        if escape is not None:
            box.setEscapeButton(keyed[escape])
        box.exec()
        clicked = box.clickedButton()
        box.deleteLater()
        return next((key for button, key in buttons if button is clicked), None)

    def _call_off_close(self) -> None:
        """Back to the app with the notes still in the editor. Any job told to
        stop still finishes, and its handler resets the controls; stats still
        poll and autosave still runs."""
        self._quit_requested = False
        self.statusBar().showMessage(
            "Not closed: your notes are not saved yet. Export Notes… keeps a copy.")
