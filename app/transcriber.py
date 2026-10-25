"""Background transcription worker using faster-whisper (CTranslate2)."""

from __future__ import annotations

import gc
import sys
import threading
from typing import NamedTuple

from PySide6.QtCore import QThread, Signal

import diagnostics
from audio import SAMPLE_RATE, AudioFileError, Timing, decode_track_audio
from models import RunConfig, Segment, Track

# A pause this long between words splits a Whisper segment in two. Whisper can
# lump utterances separated by silence into one segment, which would sort a
# speaker's later remark at the time of their earlier one and break the
# conversation interleaving.
SPLIT_GAP_SECONDS = 1.0

# NVIDIA models (and other non-Whisper models) run through onnx-asr instead
# of faster-whisper. UI names map to onnx-asr model ids.
ONNX_ALIASES = {
    "parakeet-tdt-0.6b-v3": "nemo-parakeet-tdt-0.6b-v3",
    "parakeet-tdt-0.6b-v2": "nemo-parakeet-tdt-0.6b-v2",
    "parakeet": "nemo-parakeet-tdt-0.6b-v3",
    "canary-1b-v2": "nemo-canary-1b-v2",
    "canary": "nemo-canary-1b-v2",
}


# Qwen3-ASR (30 languages plus 22 Chinese dialects) runs through
# mlx-qwen3-asr on the Apple GPU (macOS only). UI names map to Hugging Face
# repos.
QWEN_REPOS = {
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-asr": "Qwen/Qwen3-ASR-1.7B",
}

# Qwen3-ASR expects full language names, not Whisper's ISO codes. This is
# also the list of Language choices Qwen can take, so it holds only names in
# the model's own `support_languages` (its config.json), which has no
# Norwegian or Ukrainian. mlx-qwen3-asr 0.3.5 does not refuse a name outside
# that list: it only warns on stderr, which the app never shows, and forces
# the name into the prompt anyway, so the model is asked for a language it
# was never trained on. See language_supported.
QWEN_LANGUAGE_NAMES = {
    "en": "English", "sv": "Swedish", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese", "nl": "Dutch",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ar": "Arabic",
    "hi": "Hindi", "tr": "Turkish",
}


def engine_for(model_name: str) -> str:
    n = model_name.lower()
    # Route anything Parakeet/Canary-shaped to onnx-asr, even misspellings of
    # the exact id — its "unknown model" error is clearer than faster-whisper
    # failing to find a Hugging Face repo of that name.
    if n in ONNX_ALIASES or n.startswith(
        ("nemo-", "parakeet", "canary", "gigaam", "alphacep/", "t-tech/")
    ):
        return "onnx"
    if n in QWEN_REPOS or n.startswith("qwen"):
        return "qwen"
    return "whisper"


def language_supported(model_name: str, language: str | None) -> bool:
    """Whether `model_name` can be told to transcribe `language`, a Language
    dropdown code (None is auto-detect). Only Qwen has a closed list: Whisper
    knows every language the dropdown offers, and the NVIDIA models ignore
    the setting."""
    return (language is None or engine_for(model_name) != "qwen"
            or language in QWEN_LANGUAGE_NAMES)


# Friendly UI names for Whisper-family CTranslate2 repos, checked before the
# name is handed to faster-whisper.
WHISPER_ALIASES = {
    "distil-large-v3.5-ct2": "distil-whisper/distil-large-v3.5-ct2",
}

ENGINE_LABELS = {
    "whisper-mlx": "mlx-whisper",
    "whisper-ct2": "faster-whisper",
    "onnx": "onnx-asr",
    "qwen": "mlx-qwen3-asr",
}
DEVICE_LABELS = {"metal": "Metal GPU", "cuda": "CUDA GPU", "cpu": "CPU"}


class Backend(NamedTuple):
    """The runtime and compute device that actually ran a track.

    Deliberately distinct from the Device combo's value: the requested device
    is a wish, and several engines ignore it. Qwen is always MLX; onnx-asr is
    always the CPU EP on Apple silicon; Whisper drops to faster-whisper on the
    CPU when mlx-whisper is missing. Reporting the request as if it were the
    result made "falling back to CPU" messages fictitious.
    """

    engine: str  # whisper-mlx | whisper-ct2 | onnx | qwen
    device: str  # metal | cuda | cpu

    @property
    def label(self) -> str:
        return (f"{ENGINE_LABELS.get(self.engine, self.engine)} on "
                f"{DEVICE_LABELS.get(self.device, self.device)}")


def resolve_backend(model_name: str, device: str) -> Backend:
    """What a (model, requested device) pair will really load."""
    engine = engine_for(model_name)
    if engine == "onnx":
        # The CUDA EP only on an actual CUDA box; everywhere else, Apple
        # silicon included and on purpose, the CPU EP (see _get_model).
        return Backend("onnx", "cuda" if device == "cuda" else "cpu")
    if engine == "qwen":
        return Backend("qwen", "metal")  # MLX-only; the Device setting is ignored
    if device == "mlx" and mlx_available():
        return Backend("whisper-mlx", "metal")
    return Backend("whisper-ct2", "cuda" if device == "cuda" else "cpu")


def fallback_backend(backend: Backend) -> Backend | None:
    """A genuinely different backend to retry a failure on, or None.

    Reloading the same runtime cannot fix a runtime-specific failure, and
    announcing a fallback that did not happen hides the real cause. Only the
    transitions that cross a real boundary are offered.
    """
    if backend.device == "cpu":
        return None
    if backend.engine == "whisper-mlx":
        return Backend("whisper-ct2", "cpu")  # a different Whisper runtime entirely
    if backend.engine in ("whisper-ct2", "onnx") and backend.device == "cuda":
        return Backend(backend.engine, "cpu")
    # Qwen has no CPU path here, and onnx on Apple silicon is already the CPU EP.
    return None


# The model is loaded lazily and cached so repeated runs (and multiple
# tracks) reuse it. Only one model is kept to limit VRAM use.
_model_cache: dict = {"key": None, "model": None}


def release_unified_memory() -> None:
    """Hand back what dropping a model freed: Python's garbage, then MLX's
    buffer cache, which otherwise keeps the GPU memory for reuse."""
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except ImportError:
        pass


def free_asr_models() -> None:
    """Drop every loaded ASR model so the notes LLM can have the memory."""
    _model_cache["key"] = None
    _model_cache["model"] = None
    # Two engines keep models of their own, out of _model_cache's sight, and
    # each would stay resident under the notes model: mlx-whisper's weights,
    # loaded on its first transcribe, and the forced aligner that gives Qwen
    # its word timestamps.
    whisper = getattr(sys.modules.get("mlx_whisper.transcribe"), "ModelHolder", None)
    if whisper is not None:
        whisper.model = whisper.model_path = None
    qwen = getattr(sys.modules.get("mlx_qwen3_asr.load_models"), "_ModelHolder", None)
    if qwen is not None:
        qwen.clear()
    release_unified_memory()


# ---- Model-download progress -----------------------------------------------
# Every engine fetches weights through huggingface_hub, whose byte-level
# progress bars subclass tqdm.std.tqdm without overriding update(). Wrapping
# update()/close() once lets the UI show one aggregate download percentage.
# Only bars measuring bytes (unit == "B") count, which excludes
# snapshot_download's outer "Fetching N files" bar and unrelated tqdm use.
# hf_hub downloads files on a thread pool, so the callback fires from
# arbitrary threads — safe here because Qt queues cross-thread signal emits.

_download_lock = threading.Lock()
_download_bars: dict[int, tuple[int, int]] = {}
_download_cb = None  # set only while a model load is in flight
_hook_installed = False


def begin_download_reporting(callback) -> None:
    """Route Hugging Face byte progress to one active worker."""
    global _download_cb, _hook_installed
    with _download_lock:
        if not _hook_installed:
            # On the first run rather than at import: the hook imports
            # faster-whisper, which the window does not need to open.
            _install_download_hook()
            _hook_installed = True
        _download_bars.clear()
    _download_cb = callback


def reset_download_reporting(callback) -> None:
    """Forget finished download bars and hide the readout until a new one starts."""
    with _download_lock:
        _download_bars.clear()
    callback(-1, 0.0)


def end_download_reporting(callback) -> None:
    """Stop the active worker's download reporting and hide its readout."""
    global _download_cb
    # ASR and AI-notes workers are mutually exclusive, so there can only be
    # one owner. Do not compare PySide bound-signal wrappers: each attribute
    # access can produce a distinct Python wrapper for the same Qt signal.
    _download_cb = None
    reset_download_reporting(callback)


def _report_download(bar) -> None:
    cb = _download_cb
    if cb is None or getattr(bar, "unit", "") != "B" or not getattr(bar, "total", 0):
        return
    with _download_lock:
        _download_bars[id(bar)] = (bar.n, bar.total)
        done = sum(n for n, _ in _download_bars.values())
        total = sum(t for _, t in _download_bars.values())
    if total > 0:
        cb(min(100, int(done * 100 / total)), total / 1e9)


def _install_download_hook() -> None:
    import tqdm.std

    orig_update = tqdm.std.tqdm.update
    orig_close = tqdm.std.tqdm.close

    def update(self, n=1):
        displayed = orig_update(self, n)
        _report_download(self)
        return displayed

    def close(self):
        orig_close(self)
        _report_download(self)

    tqdm.std.tqdm.update = update
    tqdm.std.tqdm.close = close

    # huggingface_hub auto-disables its bars when stderr is not a TTY — the
    # normal case for a GUI app — and a disabled tqdm skips the byte
    # accounting the hook above reads. Force bars on while a worker run has
    # a download callback active.
    # huggingface_hub.utils re-exports the tqdm *class* under the same name
    # as the submodule, so attribute access yields the class — go through
    # sys.modules to reach the module itself.
    import huggingface_hub.utils.tqdm  # noqa: F401  (ensure the module is loaded)

    hub_tqdm_module = sys.modules["huggingface_hub.utils.tqdm"]
    orig_is_disabled = hub_tqdm_module.is_tqdm_disabled

    def is_tqdm_disabled(log_level):
        if _download_cb is not None:
            return False
        return orig_is_disabled(log_level)

    hub_tqdm_module.is_tqdm_disabled = is_tqdm_disabled

    # faster-whisper passes snapshot_download a hard-disabled tqdm class
    # that never counts bytes; swap in the hub's live class so its model
    # downloads report progress like the other engines.
    import faster_whisper.utils
    from huggingface_hub.utils import tqdm as hub_tqdm_class

    faster_whisper.utils.disabled_tqdm = hub_tqdm_class


def split_timed_words(words: list[tuple[float, float, str]], speaker: str, *,
                      sep: str = "", offset: float = 0.0,
                      max_gap: float = SPLIT_GAP_SECONDS) -> list[Segment]:
    """Group (start, end, text) words into Segments, splitting wherever the
    silence between two words is longer than `max_gap`. Every engine with
    word timestamps goes through here. `words` must not be empty; `offset`
    is added to every time."""
    groups = [[words[0]]]
    for word in words[1:]:
        if word[0] - groups[-1][-1][1] > max_gap:
            groups.append([word])
        else:
            groups[-1].append(word)
    return [Segment(speaker, offset + group[0][0], offset + group[-1][1],
                    sep.join(text for _start, _end, text in group))
            for group in groups]


def split_on_word_gaps(fw_segment, speaker: str, max_gap: float = SPLIT_GAP_SECONDS) -> list[Segment]:
    """Split a faster-whisper segment at silence gaps between its words."""
    words = fw_segment.words
    if not words:
        return [Segment(speaker, fw_segment.start, fw_segment.end, fw_segment.text)]
    return split_timed_words([(w.start, w.end, w.word) for w in words], speaker,
                             max_gap=max_gap)


def cuda_available() -> bool:
    import ctranslate2

    return ctranslate2.get_cuda_device_count() > 0


def mlx_installed() -> bool:
    """Whether mlx-whisper is installed, without importing it: importing it
    pulls in numba, scipy and mlx, most of a second at every launch, and
    the window only needs to know for a label."""
    import importlib.util

    return sys.platform == "darwin" and importlib.util.find_spec("mlx_whisper") is not None


def mlx_available() -> bool:
    """mlx-whisper runs Whisper on Apple-silicon GPUs (macOS only)."""
    if sys.platform != "darwin":
        return False
    try:
        import mlx_whisper  # noqa: F401

        return True
    except ImportError:
        return False


# Whisper model name -> MLX conversion on Hugging Face, used when the device
# is Apple GPU ("mlx"). Unlisted names are passed through as repo ids.
MLX_REPOS = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
    "medium": "mlx-community/whisper-medium-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "tiny": "mlx-community/whisper-tiny",
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
}


class _MlxModel:
    """Marker object standing in for a model handled by mlx_whisper.transcribe."""

    def __init__(self, repo: str):
        self.repo = repo


class _QwenModel:
    """Qwen3-ASR session plus the forced aligner that provides word timestamps.

    The aligner is loaded once here rather than per transcribe() call —
    mlx-qwen3-asr would otherwise re-resolve it for every track.
    """

    def __init__(self, repo: str):
        from mlx_qwen3_asr import ForcedAligner, Session

        self.session = Session(model=repo)
        self.aligner = ForcedAligner()


def _get_model(model_name: str, backend: Backend, status_cb) -> object:
    """Load (or reuse) the model for exactly this backend.

    Takes a resolved Backend rather than the requested device so the branch
    taken here and the backend reported to the user cannot drift apart.
    """
    key = (model_name, backend)
    if _model_cache["key"] != key:
        free_asr_models()  # the previous model goes before the new one loads
        status_cb(f"Loading model '{model_name}' ({backend.label}) — first use "
                  "downloads it, which can take several minutes for large models…")
        if backend.engine == "onnx":
            import onnx_asr

            # No CoreML EP on Apple silicon, on purpose. Two reasons, both
            # measured on the M1 Pro with parakeet-tdt-0.6b-v3:
            #  * onnxruntime <= 1.27 cannot even build the session — the fp32
            #    encoder keeps its 2.4 GB of weights in an external
            #    encoder-model.onnx.data file, and the CoreML partitioner
            #    loses the model path while fusing subgraphs ("model_path
            #    must not be empty"). Fixed in onnxruntime 1.28.
            #  * Where it does load (1.28), it is slower, not faster: CoreML
            #    claims only ~1000 of the encoder's 2115 nodes and splits
            #    them into ~291 partitions, so the graph ping-pongs between
            #    CoreML and CPU. 124 s of speech took 10.5 s vs 7.2 s on the
            #    plain CPU EP, plus ~5 s more to load. A single 60 s encoder
            #    call (no VAD chunking) took 76 s on CoreML vs 2.9 s on CPU.
            # The CPU EP is the fast path for these models here; ask for it
            # directly rather than failing over to it.
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if backend.device == "cuda" else ["CPUExecutionProvider"])
            onnx_name = ONNX_ALIASES.get(model_name.lower(), model_name)
            model = onnx_asr.load_model(onnx_name, providers=providers)
            _model_cache["model"] = model.with_vad(onnx_asr.load_vad("silero"))
        elif backend.engine == "qwen":
            if sys.platform != "darwin":
                raise RuntimeError("Qwen3-ASR runs via MLX and needs an Apple-silicon Mac.")
            repo = QWEN_REPOS.get(model_name.lower(), model_name)
            _model_cache["model"] = _QwenModel(repo)
        elif backend.engine == "whisper-mlx":
            _model_cache["model"] = _MlxModel(MLX_REPOS.get(model_name, model_name))
        else:
            from faster_whisper import WhisperModel

            compute_type = "float16" if backend.device == "cuda" else "int8"
            fw_name = WHISPER_ALIASES.get(model_name.lower(), model_name)
            _model_cache["model"] = WhisperModel(fw_name, device=backend.device,
                                                 compute_type=compute_type)
        _model_cache["key"] = key
    return _model_cache["model"]


class TranscribeWorker(QThread):
    model_status = Signal(str)
    download_progress = Signal(int, float)  # (percent 0-100 or -1 = hide, total GB)
    track_started = Signal(int)
    segment_ready = Signal(int, object)  # (track index, Segment)
    track_progress = Signal(int, int)  # (track index, percent)
    track_done = Signal(int, str, object)  # (track index, language, RunConfig)
    track_failed = Signal(int, str)  # (track index, error message)
    track_note = Signal(int, str)  # (track index, what its timestamps ran into)
    finished_all = Signal(bool)  # cancelled?

    def __init__(self, tracks: list[Track], indices: list[int], model_name: str,
                 device: str, language: str | None, hotwords: str = "", parent=None):
        super().__init__(parent)
        self.tracks = tracks
        self.indices = indices
        self.model_name = model_name
        self.device = device  # what the Device combo asked for
        self.backend: Backend | None = None  # what actually ran; set by _load_model
        self.language = language
        self.hotwords = hotwords.strip()
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def config(self) -> RunConfig:
        """The settings behind the segments produced right now."""
        return RunConfig(
            model=self.model_name,
            language=self.language or "",
            device=self.device,
            backend=self.backend.label if self.backend else "",
            hints=self.hotwords,
        )

    def _load_model(self, backend: Backend) -> object:
        """Load a model for `backend`, retrying only on a real alternative."""
        try:
            model = _get_model(self.model_name, backend, self.model_status.emit)
        except Exception as exc:
            diagnostics.log_exception(
                "model load", exc,
                model=self.model_name, requested_device=self.device,
                backend=backend.label)
            alternative = fallback_backend(backend)
            if alternative is None:
                raise
            self.model_status.emit(
                f"{backend.label}: model load failed ({exc}) — "
                f"retrying on {alternative.label}…")
            backend = alternative
            model = _get_model(self.model_name, backend, self.model_status.emit)
        self.backend = backend
        return model

    def _decode(self, i: int):
        """Track i's audio, 16 kHz mono on its file's own clock.

        Every engine decodes here, never from a path. faster-whisper's own
        decoder ignores timestamps (see audio.py), so a gap or a late start
        in the file would move every word after it; mlx-whisper's shells out
        to the ffmpeg CLI, which a stock Mac does not have, and the failure
        would be retried on the CPU engine, silently losing the GPU.
        """
        track = self.tracks[i]
        timing = Timing()
        audio = decode_track_audio(track.path, track.stream_index, track.channel, timing)
        note = timing.summary()
        if note:
            diagnostics.log_event("decode note", track=str(track.path), note=note)
            self.track_note.emit(i, note)
        return audio

    def _transcribe_track(self, model, i: int) -> bool:
        """Transcribe one track. Returns True if cancelled mid-track.

        The start, the decode, the stop check and the finish are the same for
        every engine; each engine method only turns the audio into segments
        and returns the language it heard, or None if it was cancelled.
        """
        engines = {
            "whisper-ct2": self._transcribe_ct2,
            "whisper-mlx": self._transcribe_mlx,
            "qwen": self._transcribe_qwen,
            "onnx": self._transcribe_onnx,
        }
        transcribe = engines[self.backend.engine]
        self.track_started.emit(i)
        audio = self._decode(i)
        if self._stop:
            return True
        language = transcribe(model, i, audio)
        if language is None:
            return True
        self.track_progress.emit(i, 100)
        self.track_done.emit(i, language, self.config())
        return False

    def _progress(self, i: int, reached: float, total: float) -> None:
        if total > 0:
            self.track_progress.emit(i, min(100, int(reached / total * 100)))

    def _transcribe_ct2(self, model, i: int, audio) -> str | None:
        """faster-whisper (CTranslate2), segment by segment."""
        track = self.tracks[i]
        segments, info = model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=False,
            hotwords=self.hotwords or None,
        )
        duration = info.duration or track.duration or 0.0
        for seg in segments:
            if self._stop:
                return None
            for piece in split_on_word_gaps(seg, track.speaker):
                self.segment_ready.emit(i, piece)
            self._progress(i, seg.end, duration)
        return info.language or ""

    def _transcribe_mlx(self, model: _MlxModel, i: int, audio) -> str | None:
        """mlx-whisper (Apple-silicon GPU).

        mlx_whisper.transcribe is blocking, so segments arrive when the whole
        track is done rather than live. Whisper's silence-spanning segments
        are split at word gaps just like the faster-whisper path.
        """
        import mlx_whisper

        speaker = self.tracks[i].speaker
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model.repo,
            language=self.language,
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=self.hotwords or None,
            verbose=None,
        )
        if self._stop:
            return None
        for seg in result.get("segments", []):
            # mlx-whisper has no VAD filter; drop likely-hallucinated silence.
            if seg.get("no_speech_prob", 0.0) > 0.6:
                continue
            words = seg.get("words") or []
            if not words:
                self.segment_ready.emit(
                    i, Segment(speaker, seg["start"], seg["end"], seg["text"]))
                continue
            for piece in split_timed_words(
                    [(w["start"], w["end"], w["word"]) for w in words], speaker):
                self.segment_ready.emit(i, piece)
        return result.get("language") or ""

    def _transcribe_qwen(self, model: _QwenModel, i: int, audio) -> str | None:
        """Qwen3-ASR (mlx-qwen3-asr, Apple-silicon GPU).

        Qwen must never see silence: on speech+silence mixes (i.e. sparse
        speaker tracks) it hallucinates repeated text, and the forced aligner
        then smears the phantom words across the silent stretches, misplacing
        segments. Silero VAD (via faster-whisper) finds the speech regions;
        each region is transcribed alone and its word timestamps are shifted
        back by the region offset. Silences under ~2 s stay inside a region,
        so normal pauses still reach the 1 s gap-split grouping; regions are
        capped at 240 s, under the aligner's 5-minute window. Vocabulary
        hints go in as decoder context.
        """
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        speaker = self.tracks[i].speaker
        regions = get_speech_timestamps(
            audio,
            vad_options=VadOptions(max_speech_duration_s=240, speech_pad_ms=400),
        )
        # run() has already refused a language Qwen can't take, so a missing
        # entry here is a bug, not a cue to auto-detect.
        forced = QWEN_LANGUAGE_NAMES[self.language] if self.language else None
        language = ""
        for region in regions:
            if self._stop:
                return None
            offset = region["start"] / SAMPLE_RATE
            result = model.session.transcribe(
                (audio[region["start"]:region["end"]], SAMPLE_RATE),
                context=self.hotwords,
                language=forced,
                return_timestamps=True,
                forced_aligner=model.aligner,
            )
            language = result.language or language
            words = result.segments or []
            if words:
                for piece in split_timed_words(
                        [(w["start"], w["end"], w["text"]) for w in words], speaker,
                        sep=" ", offset=offset):
                    self.segment_ready.emit(i, piece)
            elif result.text.strip():
                # Aligner produced nothing but there is text: keep it,
                # spanning the region rather than dropped.
                self.segment_ready.emit(i, Segment(
                    speaker, offset, region["end"] / SAMPLE_RATE, result.text))
            self._progress(i, region["end"], len(audio))
        return language

    def _transcribe_onnx(self, model, i: int, audio) -> str | None:
        """onnx-asr (Parakeet, Canary), which reports no language."""
        speaker = self.tracks[i].speaker
        duration = len(audio) / SAMPLE_RATE
        for res in model.recognize(audio, sample_rate=SAMPLE_RATE):
            if self._stop:
                return None
            if not res.text.strip():
                continue
            self.segment_ready.emit(i, Segment(speaker, res.start, res.end, res.text))
            self._progress(i, res.end, duration)
        return ""

    def run(self) -> None:
        # Before any model loads or downloads: no backend can transcribe a
        # language the model doesn't have, and mlx-qwen3-asr would not say so.
        if not language_supported(self.model_name, self.language):
            for i in self.indices:
                self.track_failed.emit(
                    i, f"{self.model_name} can't transcribe language "
                       f"'{self.language}'. Choose another language or "
                       f"Auto-detect, or a Whisper model.")
            self.finished_all.emit(False)
            return
        # The notes LLM and an ASR model never share unified memory.
        import llm

        llm.free_llm()
        # Model weights can download at load time or — on the mlx-whisper
        # path — lazily during the first transcribe call, so the download
        # callback stays active for the whole run. Resets between phases stop
        # a finished download from lingering at 100% or skewing the next one.
        begin_download_reporting(self.download_progress.emit)
        try:
            requested = resolve_backend(self.model_name, self.device)
            if self.device == "mlx" and requested.engine == "whisper-ct2":
                # Silently running Whisper on the CPU while the UI said Metal
                # was the single most misleading thing this worker did.
                self.model_status.emit(
                    "mlx-whisper is not installed — running Whisper on the CPU "
                    "instead of the Metal GPU (see setup_mac.command).")
            try:
                model = self._load_model(requested)
            except Exception as exc:  # load failed on every backend available
                diagnostics.log_exception(
                    "model load (final)", exc,
                    model=self.model_name, requested_device=self.device)
                for i in self.indices:
                    self.track_failed.emit(i, f"Model load failed: {exc}")
                self.finished_all.emit(False)
                return
            self.model_status.emit(f"Running on {self.backend.label}.")
            reset_download_reporting(self.download_progress.emit)

            cancelled = False
            for i in self.indices:
                if self._stop:
                    cancelled = True
                    break
                alternative = None
                try:
                    cancelled = self._transcribe_track(model, i)
                except AudioFileError as exc:
                    # The file itself: another backend would read it no
                    # better, and moving to it would leave every later
                    # track off the GPU too.
                    diagnostics.log_exception(
                        "decode", exc, track=str(self.tracks[i].path),
                        model=self.model_name, backend=self.backend.label)
                    self.track_failed.emit(i, str(exc))
                except Exception as exc:
                    diagnostics.log_exception(
                        "transcribe", exc, track=str(self.tracks[i].path),
                        model=self.model_name, backend=self.backend.label)
                    alternative = fallback_backend(self.backend)
                    if alternative is None:
                        # No other backend exists for this engine — retrying
                        # the same one would only repeat the failure.
                        self.track_failed.emit(i, str(exc))
                    else:
                        # Backend trouble (e.g. out of unified memory) — move
                        # this and every later track to the alternative.
                        self.model_status.emit(
                            f"{self.backend.label}: error on "
                            f"{self.tracks[i].path.name} ({exc}) — "
                            f"retrying on {alternative.label}…")
                # The retry runs after the except block, not in it: the
                # failure's traceback holds the failed attempt's frames, and
                # with them its model and the track's audio, gigabytes for
                # a long track, which would stay pinned for the whole retry.
                if alternative is not None:
                    try:
                        model = _get_model(self.model_name, alternative,
                                           self.model_status.emit)
                        self.backend = alternative
                        cancelled = self._transcribe_track(model, i)
                    except Exception as retry_exc:
                        diagnostics.log_exception(
                            "transcribe (retry)", retry_exc,
                            track=str(self.tracks[i].path),
                            model=self.model_name, backend=alternative.label)
                        self.track_failed.emit(i, str(retry_exc))
                reset_download_reporting(self.download_progress.emit)

            self.finished_all.emit(cancelled)
        finally:
            end_download_reporting(self.download_progress.emit)
