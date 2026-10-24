"""Qwen's Language choices are a closed list; every other engine takes any.

mlx-qwen3-asr does not refuse a language Qwen3-ASR lacks (Norwegian,
Ukrainian): it warns on stderr, which the app never shows, and forces the name
into the prompt, so the model writes a language it was never trained on. The
app is the only thing that says no: greyed out in the window, refused by
Transcribe, and refused again by the worker before any model loads.
"""

import json
import os
from pathlib import Path

import pytest

import transcriber
from transcriber import QWEN_LANGUAGE_NAMES, QWEN_REPOS, language_supported

QWEN_MODELS = ("qwen3-asr-1.7b", "qwen3-asr-0.6b", "Qwen/Qwen3-ASR-0.6B")


@pytest.mark.parametrize("model", QWEN_MODELS)
@pytest.mark.parametrize("language", ["no", "uk"])
def test_qwen_refuses_what_the_model_does_not_support(model, language):
    assert not language_supported(model, language)


@pytest.mark.parametrize("language", [None, "en", "sv", "zh"])
def test_qwen_takes_auto_detect_and_its_own_languages(language):
    assert language_supported("qwen3-asr-1.7b", language)


@pytest.mark.parametrize("model", ["large-v3-turbo", "parakeet-tdt-0.6b-v3"])
@pytest.mark.parametrize("language", ["no", "uk", "en", None])
def test_whisper_and_nvidia_models_take_every_choice(model, language):
    assert language_supported(model, language)


def test_every_qwen_language_is_one_the_model_supports():
    # The model's own list is `support_languages` in its config.json, which
    # mlx-qwen3-asr reads but does not enforce. Read from the local download
    # only; no test may fetch a model.
    from huggingface_hub import try_to_load_from_cache

    checked = 0
    for repo in sorted(set(QWEN_REPOS.values())):
        path = try_to_load_from_cache(repo, "config.json")
        if not isinstance(path, str):
            continue
        supported = json.loads(Path(path).read_text(encoding="utf-8"))["support_languages"]
        assert set(QWEN_LANGUAGE_NAMES.values()) - set(supported) == set(), repo
        checked += 1
    if not checked:
        pytest.skip("no Qwen3-ASR model has been downloaded")


# ---- The worker ------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_worker_refuses_before_loading_a_model(qapp, monkeypatch):
    loads = []
    monkeypatch.setattr(transcriber, "_get_model", lambda *args: loads.append(args))
    worker = transcriber.TranscribeWorker([], [0, 1], "qwen3-asr-1.7b", "mlx", "no")
    failures, finished = [], []
    worker.track_failed.connect(lambda i, message: failures.append((i, message)))
    worker.finished_all.connect(finished.append)

    worker.run()

    assert loads == []
    assert [i for i, _ in failures] == [0, 1]
    assert "can't transcribe" in failures[0][1]
    assert finished == [False]


# ---- The window ------------------------------------------------------------

@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    """A real MainWindow whose settings live in tmp_path, never the user's."""
    from PySide6.QtCore import QCoreApplication, QEvent, QSettings

    import main_window

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(main_window, "QSettings", lambda: settings)
    monkeypatch.setattr(main_window.StatsPoller, "start", lambda self: None)
    w = main_window.MainWindow()
    w.settings = settings
    yield w
    # Deleted now rather than left to the garbage collector, which can take
    # a closed MainWindow down partway through a later test on PySide6 6.11.
    w.close()
    w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def row(window, code):
    return window.combo_lang.findData(code)


def item(window, code):
    return window.combo_lang.model().item(row(window, code))


def test_the_dropdown_offers_qwen_everything_but_norwegian_and_ukrainian(window):
    import main_window

    refused = {code for _, code in main_window.LANGUAGES
               if not language_supported("qwen3-asr-1.7b", code)}
    assert refused == {"no", "uk"}


def test_qwen_greys_out_what_it_lacks_and_whisper_gets_it_back(window):
    window.combo_model.setCurrentText("qwen3-asr-1.7b")
    assert not item(window, "no").isEnabled()
    assert not item(window, "uk").isEnabled()
    assert item(window, "en").isEnabled()
    assert item(window, None).isEnabled()
    window.combo_model.setCurrentText("large-v3")
    assert item(window, "no").isEnabled()
    assert item(window, "uk").isEnabled()


def test_a_typed_qwen_repo_gets_the_same_choices(window):
    window.combo_model.setCurrentText("Qwen/Qwen3-ASR-0.6B")
    assert not item(window, "no").isEnabled()


def test_switching_to_qwen_keeps_the_language_and_flags_it(window):
    window.combo_model.setCurrentText("large-v3")
    window.combo_lang.setCurrentIndex(row(window, "no"))
    assert window.label_lang_conflict.isHidden()
    window.combo_model.setCurrentText("qwen3-asr-1.7b")
    # Still Norwegian, not quietly turned into Auto-detect.
    assert window.combo_lang.currentData() == "no"
    assert not window.label_lang_conflict.isHidden()
    assert "Norwegian" in window.label_lang_conflict.toolTip()
    window.combo_lang.setCurrentIndex(row(window, "en"))
    assert window.label_lang_conflict.isHidden()


def test_transcribe_refuses_the_conflict_and_changes_nothing(window, monkeypatch):
    import main_window
    from models import Track

    track = Track(Path("alice.wav"), "Alice")
    window.tracks.append(track)
    window._append_row(track)
    window.combo_model.setCurrentText("qwen3-asr-1.7b")
    window.combo_lang.setCurrentIndex(row(window, "uk"))
    status = window.table.item(0, main_window.COL_STATUS).text()
    warnings, started, freed = [], [], []
    monkeypatch.setattr(main_window.QMessageBox, "warning",
                        lambda *args: warnings.append(args))
    monkeypatch.setattr(main_window, "TranscribeWorker",
                        lambda *args, **kwargs: started.append(args))
    monkeypatch.setattr(main_window.llm, "free_llm", lambda: freed.append(True))

    window.toggle_transcription()

    assert started == [] and freed == []
    assert "Ukrainian" in warnings[0][2]
    assert window.worker is None
    assert window.table.item(0, main_window.COL_STATUS).text() == status


def test_a_saved_conflict_is_flagged_at_startup(window):
    window.settings.setValue("model", "qwen3-asr-0.6b")
    window.settings.setValue("language", "no")
    window._load_settings()
    assert window.combo_lang.currentData() == "no"
    assert not window.label_lang_conflict.isHidden()
