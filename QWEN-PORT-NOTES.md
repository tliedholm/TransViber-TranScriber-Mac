# Porting the Qwen3-ASR engine to the Windows version

Summary of the Qwen3-ASR integration in the Mac app (`app/transcriber.py`),
written for the Windows/CUDA sibling codebase. The Mac runtime itself is
**Apple-silicon-only**, so item 6 (what to replace) matters most.

## 1. Models

| Purpose | Hugging Face repo | Size |
|---|---|---|
| ASR (accuracy pick) | `Qwen/Qwen3-ASR-1.7B` | 1.7B, ~3.4 GB |
| ASR (fast pick) | `Qwen/Qwen3-ASR-0.6B` | 0.6B, ~1.2 GB |
| Word timestamps | `Qwen/Qwen3-ForcedAligner-0.6B` | 0.6B |

UI names `qwen3-asr-1.7b` / `qwen3-asr-0.6b` map to the repos via a
`QWEN_REPOS` dict. Qwen3-ASR is Apache-2.0, 52 languages.

## 2. Runtime / packages (Mac)

- **`mlx-qwen3-asr==0.3.5`**, installed as `pip install "mlx-qwen3-asr[aligner]"`
  (the `[aligner]` extra pulls what the forced aligner needs).
- That is the only direct dependency added; it brings its own transitive deps
  (mlx, etc.). **MLX is Apple-silicon-only — this package cannot run on
  Windows.** See §6 for CUDA alternatives.

## 3. Audio input

- Decoded by the app itself (PyAV) to **mono float32 numpy at 16 kHz** —
  same `decode_track_audio()` output every other engine consumes.
- Passed to the runtime as a `(numpy_array, 16000)` tuple. The package also
  accepts file paths, but the array path is used so multitrack sub-tracks
  (stream/channel slices) work.

## 4. Timestamps

- **No VAD, no chunking on our side.** Qwen3-ASR itself returns plain text;
  word-level timestamps come from the companion **forced aligner model**
  (`Qwen/Qwen3-ForcedAligner-0.6B`), which the runtime applies when
  `return_timestamps=True` — it aligns the decoded text against the audio
  and returns `result.segments` as word dicts: `[{text, start, end}, ...]`
  (floats, seconds).
- We then regroup words into utterance segments, splitting whenever the gap
  between consecutive words exceeds `SPLIT_GAP_SECONDS = 1.0` — identical
  rule to the Whisper paths, and required for chronological merging.
- Hallucination/silence guard: Qwen3-ASR emits **empty output for
  non-speech audio** (verified on 6 s of pure silence → `text=''`,
  `segments=[]`), so no `no_speech_prob` filtering or silero VAD is needed.
  Verify this holds in whatever CUDA runtime you choose before trusting
  sparse speaker tracks to it.
- Vocabulary hints are passed as `context="..."` (a space/comma-separated
  term list injected into the system prompt) — Qwen's equivalent of
  faster-whisper `hotwords=`. Works well; hint words came out correctly
  spelled in testing.
- Language: Qwen expects full names ("English", "Swedish"), not ISO codes —
  mapped via a `QWEN_LANGUAGE_NAMES` dict; `None` = auto-detect.
- Long audio: the runtime chunks internally above 1200 s; nothing needed on
  our side. `on_progress` callback events carry a `progress` float used for
  the per-track progress bar.

## 5. The Mac code (app/transcriber.py)

```python
QWEN_REPOS = {
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-asr": "Qwen/Qwen3-ASR-1.7B",
}

QWEN_LANGUAGE_NAMES = {
    "en": "English", "sv": "Swedish", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese", "nl": "Dutch",
    "da": "Danish", "no": "Norwegian", "fi": "Finnish", "pl": "Polish",
    "ru": "Russian", "uk": "Ukrainian", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "ar": "Arabic", "hi": "Hindi", "tr": "Turkish",
}

# in engine_for():
    if n in QWEN_REPOS or n.startswith("qwen"):
        return "qwen"


class _QwenModel:
    """Qwen3-ASR session plus the forced aligner that provides word timestamps.

    The aligner is loaded once here rather than per transcribe() call —
    mlx-qwen3-asr would otherwise re-resolve it for every track.
    """

    def __init__(self, repo: str):
        from mlx_qwen3_asr import ForcedAligner, Session

        self.session = Session(model=repo)
        self.aligner = ForcedAligner()


# in _get_model():
        elif engine == "qwen":
            if sys.platform != "darwin":
                raise RuntimeError("Qwen3-ASR runs via MLX and needs an Apple-silicon Mac.")
            repo = QWEN_REPOS.get(model_name.lower(), model_name)
            _model_cache["model"] = _QwenModel(repo)


# in TranscribeWorker:
    def _transcribe_track_qwen(self, model: _QwenModel, i: int) -> bool:
        """Transcribe one track with Qwen3-ASR (mlx-qwen3-asr, Apple-silicon GPU).

        The transcribe call is blocking, so segments arrive when the whole
        track is done. The forced aligner returns word-level timestamps
        ([{text, start, end}, ...]); words are regrouped into segments at
        silence gaps, same rule as the Whisper paths. Vocabulary hints go in
        as decoder context. Qwen3-ASR outputs nothing for non-speech audio,
        which stands in for a VAD/hallucination guard on sparse tracks.
        """
        from audio import decode_track_audio

        track = self.tracks[i]
        self.track_started.emit(i)
        audio = decode_track_audio(track.path, track.stream_index, track.channel)
        duration = len(audio) / 16000

        def on_progress(event: dict) -> None:
            p = event.get("progress")
            if isinstance(p, (int, float)):
                self.track_progress.emit(i, min(100, int(p * 100 if p <= 1 else p)))

        result = model.session.transcribe(
            (audio, 16000),
            context=self.hotwords,
            language=QWEN_LANGUAGE_NAMES.get(self.language) if self.language else None,
            return_timestamps=True,
            forced_aligner=model.aligner,
            on_progress=on_progress,
        )
        if self._stop:
            return True
        words = result.segments or []
        if not words and result.text.strip():
            # Aligner produced nothing but there is text: keep it, unplaced
            # within the track rather than dropped.
            self.segment_ready.emit(i, Segment(track.speaker, 0.0, duration, result.text))
        elif words:
            current = [words[0]]
            for word in words[1:]:
                if word["start"] - current[-1]["end"] > SPLIT_GAP_SECONDS:
                    self.segment_ready.emit(i, Segment(
                        track.speaker, current[0]["start"], current[-1]["end"],
                        " ".join(w["text"] for w in current)))
                    current = [word]
                else:
                    current.append(word)
            self.segment_ready.emit(i, Segment(
                track.speaker, current[0]["start"], current[-1]["end"],
                " ".join(w["text"] for w in current)))
        self.track_progress.emit(i, 100)
        self.track_done.emit(i, result.language or "")
        return False
```

UI side (`main_window.py`): the two model names added to `MODELS`, and the
vocabulary-hints field is shown when `engine_for(model) in ("whisper", "qwen")`.

## 6. Mac-specific parts a Windows/CUDA port must replace

**The entire runtime.** `mlx-qwen3-asr` is built on Apple's MLX and only runs
on Apple-silicon Macs — there is no CUDA build. What carries over unchanged
is the *shape* of the integration: model repos, 16 kHz mono float32 input,
context-string vocab biasing, forced-aligner word timestamps, and the
1.0 s word-gap regrouping.

CUDA options to evaluate (in rough order of likely fit):

1. **`qwen-asr` (PyPI)** — reference implementation associated with the
   official Qwen release; check its CUDA support and whether it exposes the
   forced aligner + context biasing.
2. **Hugging Face `transformers`** — Qwen3-ASR ships weights on HF; check
   whether current transformers has a pipeline for it and how to run
   `Qwen3-ForcedAligner-0.6B` alongside (the aligner is what makes the
   timestamps possible — without it you only get plain text, which is
   useless for this app's chronological merging).
3. **vLLM** — Qwen's server-grade path; heavier than a desktop app wants.

Non-negotiables for whichever runtime is chosen:

- Word- (or at least sub-segment-) level timestamps, so the 1 s gap-split
  can preserve conversation interleaving.
- Empty output (or a usable confidence signal) on silence — sparse speaker
  tracks are mostly silence.
- Context/vocabulary biasing if hints should keep working (else hide the
  hints field for qwen models like the onnx ones do).
