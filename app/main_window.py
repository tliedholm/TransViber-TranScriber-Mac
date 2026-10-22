"""Main application window."""

from __future__ import annotations

import html
import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer, Slot
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QGuiApplication, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

import diagnostics
import exporters
import llm
from audio import _channel_suffixes, probe_streams, tracks_for_file
from llm import DEFAULT_NOTES_MODEL, NOTES_MODEL_GROUPS, NotesWorker
from merger import coalesce, discarded_segment_count, merge_tracks
from models import AUDIO_EXTENSIONS, RunConfig, Track, TrackStatus
from stats import StatsPoller
from transcriber import TranscribeWorker, cuda_available, engine_for, mlx_available

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

COL_SPEAKER, COL_FILE, COL_DURATION, COL_STATUS, COL_PROGRESS = range(5)

FILE_FILTER = "Audio files (*" + " *".join(sorted(AUDIO_EXTENSIONS)) + ");;All files (*.*)"

EXPORT_FILTERS = (
    "Markdown (*.md);;Plain text (*.txt);;SubRip subtitles (*.srt);;"
    "WebVTT subtitles (*.vtt);;JSON (*.json);;All formats (*.md *.txt *.srt *.vtt *.json)"
)


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TransViber TransScriber")
        self.resize(1100, 750)
        self.setAcceptDrops(True)

        self.tracks: list[Track] = []
        self.worker: TranscribeWorker | None = None
        self.has_cuda = not IS_MAC and cuda_available()
        self.has_mlx = mlx_available()
        # Set once the user asks to quit while a worker is still running; the
        # close is completed by _quit_if_idle() when the worker really stops.
        self._quit_requested = False

        self._build_ui()
        self._load_settings()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(400)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._render_transcript)

    # ---------- UI construction ----------

    def _build_ui(self) -> None:
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
        bar.addWidget(self.btn_add)
        bar.addWidget(self.btn_remove)
        bar.addWidget(self.btn_clear)
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
            "state-of-the-art accuracy, 52 languages, supports vocabulary\n"
            "hints. -0.6b is the faster, lighter sibling. The Device setting\n"
            "is ignored (always Metal GPU).\n"
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

        bar.addWidget(QLabel("Device:"))
        self.combo_device = QComboBox()
        if IS_MAC:
            self.combo_device.addItem("GPU (Metal)", "mlx")
            self.combo_device.addItem("CPU", "cpu")
            if not self.has_mlx:
                self.combo_device.setToolTip(
                    "Whisper models need mlx-whisper for the GPU (run setup_mac.sh);\n"
                    "without it they run on CPU. Parakeet/Canary always run on CPU\n"
                    "(fastest path for them on Apple silicon)."
                )
        else:
            self.combo_device.addItem(
                "GPU (CUDA)" if self.has_cuda else "GPU (unavailable)", "cuda")
            self.combo_device.addItem("CPU", "cpu")
            if not self.has_cuda:
                self.combo_device.model().item(0).setEnabled(False)
                self.combo_device.setCurrentIndex(1)
        bar.addWidget(self.combo_device)

        bar.addStretch()
        # Elapsed time of the running (or last finished) transcription.
        self.elapsed_label = QLabel("")
        self.elapsed_label.setToolTip("How long the current or last transcription run took.")
        bar.addWidget(self.elapsed_label)
        self._run_started: float | None = None
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
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

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Speaker", "File", "Duration", "Status", "Progress"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_SPEAKER, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_FILE, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_DURATION, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_PROGRESS, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(COL_SPEAKER, 180)
        self.table.setColumnWidth(COL_PROGRESS, 140)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_table_menu)
        splitter.addWidget(self.table)

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

        if IS_MAC:
            device_text = ("GPU: Apple Silicon (Metal)" if self.has_mlx
                           else "GPU: unavailable (mlx-whisper not installed)")
        elif self.has_cuda:
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

    def _on_download_progress(self, percent: int, total_gb: float) -> None:
        if percent < 0:
            self.download_label.setText("")
        elif total_gb >= 0.05:
            self.download_label.setText(f"Downloading model: {percent}% of {total_gb:.1f} GB")
        else:
            self.download_label.setText(f"Downloading model: {percent}%")
        if percent >= 0:
            model = (self.combo_notes_model.currentText().strip()
                     if self.notes_worker is not None
                     else self.combo_model.currentText().strip())
            size = f" of {total_gb:.1f} GB" if total_gb >= 0.05 else ""
            self.statusBar().showMessage(
                f"Downloading model '{model}' — {percent}%{size}…")

    def _set_dark_mode(self, dark: bool) -> None:
        """Apply and remember an explicit appearance choice."""
        QGuiApplication.styleHints().setColorScheme(
            Qt.ColorScheme.Dark if dark else Qt.ColorScheme.Light)
        QSettings().setValue("dark", dark)

    def _update_hints_visibility(self) -> None:
        model = self.combo_model.currentText().strip()
        self.hints_widget.setVisible(engine_for(model) in ("whisper", "qwen"))

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
            "Local LLM (mlx-community 4-bit, Apple GPU) that writes notes on\n"
            "the merged transcript. Bigger tiers give richer notes, slower.\n"
            "Type any mlx-community repo id for other models."
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
        self.notes_edit.textChanged.connect(self._notes_save_timer.start)

        self.notes_worker: NotesWorker | None = None
        self._notes_started: float | None = None
        self._notes_last_progress: tuple[int, float] | None = None  # (tokens, tok/s)
        self._notes_elapsed_timer = QTimer(self)
        self._notes_elapsed_timer.setInterval(1000)
        self._notes_elapsed_timer.timeout.connect(self._tick_notes_elapsed)
        return page

    def _save_notes(self) -> None:
        QSettings().setValue("notes", self.notes_edit.toPlainText())

    def _tick_notes_elapsed(self) -> None:
        if self._notes_started is not None:
            self.notes_elapsed_label.setText(
                self._format_elapsed(time.monotonic() - self._notes_started))

    def export_notes_dialog(self) -> None:
        text = self.notes_edit.toPlainText()
        if not text.strip():
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Export notes", str(Path.home() / "notes.md"),
            "Markdown (*.md);;Plain text (*.txt)")
        if not path_str:
            return
        try:
            Path(path_str).write_text(text, encoding="utf-8")
            self.statusBar().showMessage(f"Exported {path_str}", 5000)
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

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
        transcript = exporters.to_text(coalesce(segments))
        if self.notes_edit.toPlainText().strip():
            self._append_notes_chunk("\n\n")
        self.notes_worker = NotesWorker(
            transcript,
            self.edit_notes_instructions.toPlainText(),
            self.combo_notes_model.currentText(),
        )
        w = self.notes_worker
        w.status.connect(self.statusBar().showMessage)
        w.download_progress.connect(self._on_download_progress)
        w.chunk.connect(self._append_notes_chunk)
        w.progress.connect(self._on_notes_progress)
        w.completed.connect(self._on_notes_done)
        w.failed.connect(self._on_notes_failed)
        # ASR model and notes LLM must never be loaded together: the worker
        # frees the ASR cache before loading, and Transcribe stays disabled
        # until notes finish (the reverse holds in toggle_transcription).
        self.btn_transcribe.setEnabled(False)
        self.btn_ai_notes.setText("Cancel notes")
        self.notes_progress.setText("0% · 0 tok")
        self._last_summary = ""
        self._notes_last_progress = None
        self._notes_started = time.monotonic()
        self.notes_elapsed_label.setText("0:00")
        self._notes_elapsed_timer.start()
        w.start()

    def _append_notes_chunk(self, text: str) -> None:
        cursor = self.notes_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.notes_edit.setTextCursor(cursor)
        self.notes_edit.ensureCursorVisible()

    def _on_notes_progress(self, percent: int, tokens: int, tps: float) -> None:
        speed = f" · {tps:.0f} tok/s" if tps > 0 else ""
        self._notes_last_progress = (tokens, tps)
        self.notes_progress.setText(f"{percent}% · {tokens:,} tok{speed}")

    def _finish_notes_run(self) -> str:
        if self.notes_worker is not None:
            self.notes_worker.wait()
            self.notes_worker = None
        self._notes_elapsed_timer.stop()
        elapsed = ""
        if self._notes_started is not None:
            elapsed = self._format_elapsed(time.monotonic() - self._notes_started)
            self.notes_elapsed_label.setText(f"took {elapsed}")
            self._notes_started = None
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
            tokens, tps = self._notes_last_progress
            msg += f" · {tokens:,} tok"
            if tps > 0:
                msg += f" · {tps:.0f} tok/s"
        self._last_summary = msg
        self.statusBar().showMessage(msg)

    def _on_notes_failed(self, error: str) -> None:
        elapsed = self._finish_notes_run()
        self.notes_progress.setText("")
        msg = f"AI notes failed — took {elapsed}" if elapsed else "AI notes failed"
        self._last_summary = msg
        self.statusBar().showMessage(msg)
        if not self._quit_requested:
            QMessageBox.warning(self, "AI notes failed", error)

    # ---------- Drag & drop ----------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self.worker is None and any(
            Path(u.toLocalFile()).suffix.lower() in AUDIO_EXTENSIONS
            for u in event.mimeData().urls()
        ):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [
            Path(u.toLocalFile())
            for u in event.mimeData().urls()
            if Path(u.toLocalFile()).suffix.lower() in AUDIO_EXTENSIONS
        ]
        self.add_files(paths)

    # ---------- Track management ----------

    def add_files_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add audio tracks", "", FILE_FILTER)
        self.add_files([Path(f) for f in files])

    def add_files(self, paths: list[Path]) -> None:
        existing = {t.path for t in self.tracks}
        unreadable: list[Track] = []
        for path in paths:
            if path in existing or not path.is_file():
                continue
            for track in tracks_for_file(path):
                self.tracks.append(track)
                self._append_row(track)
                if track.probe_error:
                    unreadable.append(track)
            existing.add(path)
        self.statusBar().showMessage(f"{len(self.tracks)} track(s) loaded", 5000)
        if unreadable:
            detail = "\n\n".join(f"{t.path.name}:\n{t.probe_error}"
                                 for t in unreadable[:5])
            QMessageBox.warning(
                self, "Could not read these files",
                "They were added anyway as a single track each, but they will "
                "probably fail to transcribe:\n\n" + detail)

    def _append_row(self, track: Track, row: int | None = None) -> None:
        if row is None:
            row = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(row)

        speaker_item = QTableWidgetItem(track.speaker)
        self.table.setItem(row, COL_SPEAKER, speaker_item)

        for col, text in [
            (COL_FILE, track.source_label),
            (COL_DURATION, _fmt_duration(track.duration)),
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
        self.table.blockSignals(False)

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
            self.table.blockSignals(True)
            item.setText(track.speaker)
            self.table.blockSignals(False)
            self.statusBar().showMessage("A speaker needs a name — kept "
                                         f"'{track.speaker}'.", 4000)
            return
        track.speaker = new_name
        for seg in track.segments:
            seg.speaker = new_name
        if track.previous is not None:  # a rerun is in flight; rename the stash too
            for seg in track.previous.segments:
                seg.speaker = new_name
        self._schedule_refresh()

    def _show_table_menu(self, pos) -> None:
        row = self.table.rowAt(pos.y())
        if row < 0 or self.worker is not None:
            return
        track = self.tracks[row]
        menu = QMenu(self)
        split = menu.addAction("Split into one speaker per channel")
        split.setEnabled(track.channel is None)
        if menu.exec(self.table.viewport().mapToGlobal(pos)) is split:
            self._split_track_channels(row)

    def _split_track_channels(self, row: int) -> None:
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
        for ch, suffix in enumerate(_channel_suffixes(channels)):
            new = Track(path=track.path, speaker=track.speaker + suffix,
                        duration=track.duration, stream_index=track.stream_index,
                        channel=ch)
            self.tracks.insert(row + ch, new)
            self._append_row(new, row + ch)
        self._schedule_refresh()

    def remove_selected(self) -> None:
        if self.worker is not None:
            return
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
            del self.tracks[row]
        self._schedule_refresh()

    def clear_tracks(self) -> None:
        if self.worker is not None:
            return
        # Clearing is the one button that can destroy hours of transcription
        # in a single click, so it asks once there is something to lose.
        if merge_tracks(self.tracks, include_failed=True):
            answer = QMessageBox.question(
                self, "Clear the transcript?",
                "This removes every track and the merged transcript.\n"
                "Export first if you want to keep it.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.table.setRowCount(0)
        self.tracks.clear()
        self.transcript_view.clear()
        self.notes_transcript_view.clear()  # the notes text itself stays
        self.btn_export.setEnabled(False)

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
            answer = QMessageBox.question(
                self, "Transcribe everything again?",
                f"All {len(finished)} track(s) are already transcribed.\n\n"
                "Running again replaces the existing transcript. The current "
                "one is kept until the new run succeeds.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
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

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Settings changed since the finished tracks")
        box.setText("The " + ", ".join(changed) + " changed since "
                    f"{len(finished)} track(s) were transcribed.")
        box.setInformativeText(
            "Transcribing only the remaining tracks would mix results from "
            "different settings in one transcript.")
        rerun = box.addButton("Redo all tracks", QMessageBox.ButtonRole.AcceptRole)
        remaining = box.addButton("Only the remaining tracks",
                                  QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(rerun)
        box.exec()
        if box.clickedButton() is rerun:
            return list(range(len(self.tracks)))
        if box.clickedButton() is remaining:
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
            self._set_status(i, track.status.value)
            self._progress_bar(i).setValue(0)

        self.worker = TranscribeWorker(
            self.tracks, indices,
            model_name=wanted.model,
            device=wanted.device,
            language=self.combo_lang.currentData(),
            hotwords=self.edit_hints.toPlainText(),
        )
        w = self.worker
        w.model_status.connect(self.statusBar().showMessage)
        w.download_progress.connect(self._on_download_progress)
        w.track_started.connect(self._on_track_started)
        w.segment_ready.connect(self._on_segment)
        w.track_progress.connect(lambda i, p: self._progress_bar(i).setValue(p))
        w.track_done.connect(self._on_track_done)
        w.track_failed.connect(self._on_track_failed)
        w.finished_all.connect(self._on_finished)
        llm.free_llm()  # ASR and notes LLM never share memory
        diagnostics.log_event("transcription started", model=wanted.model,
                              device=wanted.device, language=wanted.language,
                              tracks=len(indices))
        self._set_running(True)
        self._last_summary = ""
        self._run_started = time.monotonic()
        self.elapsed_label.setText("0:00")
        self._elapsed_timer.start()
        w.start()

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def _tick_elapsed(self) -> None:
        if self._run_started is not None:
            self.elapsed_label.setText(self._format_elapsed(time.monotonic() - self._run_started))

    def _set_running(self, running: bool) -> None:
        for widget in (self.btn_add, self.btn_remove, self.btn_clear,
                       self.combo_model, self.combo_lang, self.combo_device,
                       self.edit_hints):
            widget.setEnabled(not running)
        self.btn_export.setEnabled(not running and bool(merge_tracks(self.tracks)))
        self.btn_ai_notes.setEnabled(not running and self.notes_worker is None)
        self.btn_transcribe.setText("Cancel" if running else "Transcribe")
        self.btn_transcribe.setEnabled(True)
        if not running and self.has_cuda:
            pass  # device combo re-enabled above; GPU entry stays valid

    def _progress_bar(self, row: int) -> QProgressBar:
        return self.table.cellWidget(row, COL_PROGRESS)

    def _set_status(self, row: int, text: str) -> None:
        self.table.blockSignals(True)
        self.table.item(row, COL_STATUS).setText(text)
        self.table.blockSignals(False)

    @Slot(int)
    def _on_track_started(self, i: int) -> None:
        track = self.tracks[i]
        track.segments.clear()  # a CPU-fallback retry restarts the track from scratch
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
        self._set_status(i, label)
        self.table.item(i, COL_STATUS).setToolTip(
            track.config.summary() if track.config else "")

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
            self.table.item(i, COL_STATUS).setToolTip(
                f"The rerun failed; the previous transcript was kept.\n\n{message}")
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
        self._elapsed_timer.stop()
        took = ""
        if self._run_started is not None:
            elapsed = self._format_elapsed(time.monotonic() - self._run_started)
            self.elapsed_label.setText(f"took {elapsed}")
            self._run_started = None
            took = f" — took {elapsed}"
        done = sum(1 for t in self.tracks if t.status == TrackStatus.DONE)
        if cancelled:
            msg = f"Transcription cancelled{took}"
        elif errors:
            msg = f"Transcription finished with {len(errors)} error(s){took}"
        else:
            msg = f"Transcription complete{took} · {done} track(s)"
        if discarded:
            msg += f" · {discarded} partial segment(s) excluded"
        self._last_summary = msg
        self.statusBar().showMessage(msg)
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
        segments = merge_tracks(self.tracks)
        if not segments:
            self.transcript_view.clear()
            self.notes_transcript_view.clear()
            return
        parts = []
        for seg in coalesce(segments):
            ts = _fmt_duration(seg.start)
            parts.append(
                f"<p><b>{html.escape(seg.speaker)}</b> "
                f"<span style='color:gray'>[{ts}]</span>: "
                f"{html.escape(seg.text.strip())}</p>"
            )
        html_text = "\n".join(parts)
        scrollbar = self.transcript_view.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 10
        self.transcript_view.setHtml(html_text)
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        # The notes-side mirror keeps the reader's place instead of
        # following the bottom.
        notes_sb = self.notes_transcript_view.verticalScrollBar()
        pos = notes_sb.value()
        self.notes_transcript_view.setHtml(html_text)
        notes_sb.setValue(min(pos, notes_sb.maximum()))

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
        answer = QMessageBox.question(
            self, "Replace existing files?",
            f"These files already exist in {surprises[0].parent} and will be "
            f"replaced:\n\n{listed}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Yes

    def export_dialog(self) -> None:
        segments = merge_tracks(self.tracks)
        if not segments:
            return
        default_name = str(Path.home() / "transcript.md")
        path_str, selected = QFileDialog.getSaveFileName(
            self, "Export transcript", default_name, EXPORT_FILTERS
        )
        if not path_str:
            return
        asked_for = Path(path_str)
        path = asked_for
        try:
            if selected.startswith("All formats"):
                targets = [path.with_suffix(ext) for ext in exporters.EXPORTERS]
                if not self._confirm_overwrites(targets, asked_for):
                    return
                written = exporters.export_all(path, segments, self.tracks)
                self.statusBar().showMessage(
                    "Exported: " + ", ".join(p.name for p in written), 5000)
            else:
                if path.suffix.lower() not in exporters.EXPORTERS:
                    ext = selected.split("*")[1].rstrip(") ")
                    path = path.with_suffix(ext)
                if not self._confirm_overwrites([path], asked_for):
                    return
                exporters.export(path, segments, self.tracks)
                self.statusBar().showMessage(f"Exported {path}", 5000)
        except OSError as exc:
            diagnostics.log_exception("export", exc, path=str(path))
            QMessageBox.critical(self, "Export failed", str(exc))

    # ---------- Settings persistence ----------

    def _load_settings(self) -> None:
        settings = QSettings()
        self.edit_hints.setPlainText(settings.value("hints", "", str))
        # A starred model beats the last-used one.
        model = (settings.value("starred_model", "", str)
                 or settings.value("model", "", str))
        if model:
            self.combo_model.setCurrentText(model)
        lang_index = self.combo_lang.findData(settings.value("language", None))
        if lang_index >= 0:
            self.combo_lang.setCurrentIndex(lang_index)
        self._update_hints_visibility()
        self.notes_edit.setPlainText(settings.value("notes", "", str))
        notes_model = (settings.value("starred_notes_model", "", str)
                       or settings.value("notes_model", "", str))
        if notes_model:
            self.combo_notes_model.setCurrentText(notes_model)
        self.edit_notes_instructions.setPlainText(
            settings.value("notes_instructions", "", str))
        if settings.contains("dark"):
            self.dark_checkbox.blockSignals(True)
            self.dark_checkbox.setChecked(settings.value("dark", False, bool))
            self.dark_checkbox.blockSignals(False)
            QGuiApplication.styleHints().setColorScheme(
                Qt.ColorScheme.Dark if self.dark_checkbox.isChecked()
                else Qt.ColorScheme.Light)

    def _save_settings(self) -> None:
        settings = QSettings()
        settings.setValue("hints", self.edit_hints.toPlainText())
        settings.setValue("model", self.combo_model.currentText().strip())
        settings.setValue("language", self.combo_lang.currentData())
        settings.setValue("notes_model", self.combo_notes_model.currentText().strip())
        settings.setValue(
            "notes_instructions", self.edit_notes_instructions.toPlainText())
        self._save_notes()

    # ---------- Shutdown ----------

    def _live_workers(self) -> list:
        return [w for w in (self.worker, self.notes_worker)
                if w is not None and w.isRunning()]

    def _quit_if_idle(self) -> None:
        """Finish a deferred quit once the last worker has really stopped.

        Deferred to the next event-loop turn so the handler that called this
        can finish before the window goes away.
        """
        if self._quit_requested and not self._live_workers():
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event) -> None:
        """Close only once no worker thread is still alive.

        Waiting a fixed few seconds and accepting the close regardless used to
        leave live QThreads attached to a destroyed window — a crash or hang.
        Several steps genuinely outlast any timeout: mlx-whisper transcribes a
        whole track in one blocking call, a Qwen speech region runs up to
        240 s, and model downloads are unbounded. None of them can be
        interrupted, so the window waits instead, and the user is offered an
        explicit way out.
        """
        self._save_settings()
        workers = self._live_workers()
        if not workers:
            self.stats_poller.stop()
            self.stats_poller.wait(3000)
            event.accept()
            return

        if not self._quit_requested:
            self._quit_requested = True
            for worker in workers:
                worker.stop()
            self.statusBar().showMessage(
                "Quitting — waiting for the current step to finish…")

        # A short wait covers the common case: between two tracks, or mid
        # generation, the stop flag is noticed almost immediately.
        if all(worker.wait(2000) for worker in workers) and not self._live_workers():
            self.stats_poller.stop()
            self.stats_poller.wait(3000)
            event.accept()
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Still working")
        box.setText("A step that cannot be interrupted is still running.")
        box.setInformativeText(
            "Transcribing a track, a Qwen speech region, and downloading a "
            "model all have to finish before the app can close cleanly.\n\n"
            "The window closes on its own as soon as that happens.")
        keep = box.addButton("Wait for it", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Quit now", QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(keep)
        box.exec()
        if box.clickedButton() is not keep:
            # Nothing can unblock a native call already in flight, and tearing
            # down running QThreads crashes Qt. Leave the process instead;
            # settings and notes are already saved above.
            diagnostics.log_event("force quit while workers were running")
            os._exit(0)
        # The finish handlers call _quit_if_idle(), so no polling is needed.
        event.ignore()
