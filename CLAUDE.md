# TransViber TransScriber — Mac edition

PySide6 desktop app that transcribes multi-track recordings (one audio
file/stream/channel per speaker) locally and merges the segments into one
conversation ordered by who speaks when. This is the macOS fork of a Windows
app; the user's PC copy is maintained separately — changes here do NOT need
to stay compatible with Windows, but the user may ask to port fixes across.

## Run / setup

- One-time: `bash setup_mac.command` (creates `.venv`, installs the pinned
  deps from `requirements-mac.txt`, then runs `packaging/preflight.py`).
  Launch: `./TransViber TransScriber.command` or
  `.venv/bin/python app/main.py`.
- Tests: `.venv/bin/python -m pytest tests` (fast; no models or downloads).
  Environment check: `.venv/bin/python packaging/preflight.py`.
- Dependency versions are pinned in `requirements-mac.txt`; bump there and
  re-run tests + preflight + the frozen self-test before shipping.
- The user normally launches via `~/Applications/TransViber TransScriber.app`
  — a thin bundle whose `Contents/MacOS/TransViberTransScriber` script execs
  this folder's `.venv/bin/python app/main.py` (hardcoded `APP_DIR`; update it
  if this folder moves). Icon lives in the bundle's Resources as `icon.icns`,
  regenerated from `app/assets/icon.png` via sips/iconutil.
- Models download to `~/.cache/huggingface` on first use.
- `bash packaging/build_app.sh` builds the self-contained, ad-hoc-signed
  `.app` for sharing (PyInstaller; `packaging/TransViber.spec`) and zips it
  with `READ ME FIRST.txt`. Version lives in the spec's `BUNDLE`. It gates
  the build on, in order: `pytest tests`, `packaging/preflight.py`, the
  spec's required-package check, and the frozen self-test — any failure
  stops the build before anything is signed.
  Both specs get their collected packages from
  `packaging/analysis_inputs.py`; a *required* package that cannot be
  collected raises `MissingPackage` instead of being skipped, so an engine
  can no longer go missing from a bundle that still builds and signs.
  `packaging/selftest.py` + `selftest.spec` build a console binary from that
  same analysis. `selftest --imports-only` checks every engine module and
  resource (incl. mlx.metallib) and needs no audio; `selftest <audio.wav>`
  additionally runs every engine and asserts the backend that ran matches
  `resolve_backend()`. Set `SELFTEST_AUDIO` when building to run the latter.

## Architecture (app/)

- `main.py` — entry point. Also handles files opened from Finder: argv at
  launch (argv_emulation) and `QFileOpenEvent` for an already-running app.
- `main_window.py` — all UI. Transcribe tab, live merged transcript, and
  Notes tab with autosaved manual/local-AI notes.
  `IS_MAC` switches the Device combo to "GPU (Metal)" (`"mlx"`) / CPU.
  Settings (model, language, vocab hints) persist via QSettings.
- `diagnostics.py` — rotating log at
  `~/Library/Logs/TransViber TransScriber/transviber.log`. Every engine
  failure lands there with traceback, model, requested device, backend
  actually used and package versions; dialogs stay short and point at it.
- `transcriber.py` — `TranscribeWorker` (QThread) with four engine paths,
  dispatched per model name / device:
  1. faster-whisper (CTranslate2, CPU on Mac) — Whisper names like
     `large-v3-turbo`.
  2. mlx-whisper (`_MlxModel`, Apple GPU) — used when device is `"mlx"` and
     mlx_whisper imports; `MLX_REPOS` maps Whisper names to mlx-community
     HF repos.
  3. onnx-asr (`engine_for()` routes parakeet/canary/nemo-*) — NVIDIA
     Parakeet/Canary with silero VAD; always the CPU EP on Mac (see the
     CoreML note under the invariants).
  4. mlx-qwen3-asr (`_QwenModel`, Apple GPU; `engine_for()` routes qwen*) —
     Qwen3-ASR via `QWEN_REPOS`. Always MLX regardless of Device. Word
     timestamps come from `ForcedAligner` (Qwen3-ForcedAligner-0.6B), loaded
     once in `_QwenModel` and passed per call. Vocab hints -> `context=`;
     language codes map to full names via `QWEN_LANGUAGE_NAMES`.
  `resolve_backend(model, device)` decides which of the four actually runs
  and returns a `Backend(engine, device)`; `_get_model` branches on that
  same value, so the path taken and the path reported cannot disagree.
  `fallback_backend()` offers a retry only where a real alternative exists
  (mlx-whisper -> faster-whisper CPU; CUDA -> CPU). Qwen (MLX-only) and
  onnx on a Mac (already the CPU EP) get none — the worker reports the
  failure rather than "retrying on CPU" onto the same runtime.
  The backend that ran is recorded per track in `Track.config` (a
  `RunConfig`) and exported in the JSON, because a transcript assembled over
  several runs can legitimately mix models.
  Model-download progress (plain percentage label in the status bar) comes
  from a tqdm hook in transcriber.py: it wraps tqdm.std.tqdm.update/close
  and aggregates byte-unit bars while a worker run is active. It also
  forces huggingface_hub bars on (they self-disable without a TTY, losing
  the byte counts) and swaps faster-whisper's disabled_tqdm for a live one.
- `llm.py` — `NotesWorker` streams local Markdown notes through mlx-lm.
  Long transcripts are summarized in parts and combined; ASR and notes
  model caches are mutually exclusive to protect unified memory.
  Every pass is budgeted against the model's real context window
  (`context_window()` reads `model.args.max_position_embeddings` — 40,960
  for Qwen3 — and never trusts `tokenizer.model_max_length` alone), minus
  the prompt and `MAX_NOTES_TOKENS` of output. `split_text()` splits at line
  boundaries *and* at token boundaries within an over-long line, because
  `coalesce()` turns a monologue into one enormous line. The combine step is
  hierarchical: `batch_by_tokens()` groups part summaries into passes that
  fit, merged a level at a time with `MERGE_PROMPT` until one batch remains.
  Notes end with 'Per-person summary', then 'Where we left off' and
  'Unresolved threads' (shared wording in `TAIL_SECTIONS`) so the next
  session knows where to pick up. Part passes use `PART_PROMPT`, which
  omits those tail sections — a middle part's ending is not the session's
  — and only the final/combine pass emits them.
- `stats.py` — `StatsPoller` QThread feeding the status-bar utilization
  label (GPU/CPU/RAM every 2 s). GPU % is parsed from
  `ioreg -c IOAccelerator` ("Device Utilization %", whole-system, no sudo);
  CPU/RAM via psutil. Polls off the main thread because ioreg takes ~50 ms.
- `audio.py` — `tracks_for_file()` expands multitrack files (multiple audio
  streams -> track per stream; >2 channels -> track per channel; stereo split
  is a right-click action). `decode_track_audio()` -> mono float32 16 kHz.
- `merger.py` / `exporters.py` — chronological merge, same-speaker
  coalescing, and md/txt/srt/vtt/json exporters. `merge_tracks()` skips
  `ERROR` tracks: an engine can emit segments and *then* raise, and those
  partial segments would make an incomplete track read as finished. They
  stay on the Track (the UI reports the count) but never reach the
  transcript or an export. Speaker names are escaped per format.

## Critical invariants — do not break

- **Word-gap splitting**: Whisper merges utterances separated by silence
  into one segment, which destroys the chronological interleaving. Both
  Whisper paths must split segments at word gaps > `SPLIT_GAP_SECONDS` (1 s)
  using word timestamps (`split_on_word_gaps` for faster-whisper; inline
  dict-based splitting in `_transcribe_track_mlx`).
- **Hallucination guards**: faster-whisper uses `vad_filter=True`; the MLX
  path has no VAD so it drops segments with `no_speech_prob > 0.6`. Sparse
  speaker tracks are mostly silence — removing these guards produces
  garbage text at silent stretches. Qwen additionally runs Silero VAD first
  and transcribes only speech regions, then restores each region's original
  timeline offset. This is required because speech mixed with long silence
  can make Qwen repeat text and corrupt forced-aligned timestamps.
- All tracks are assumed to start at the same instant (multitrack recorder
  convention); merging is a sort by segment start time.
- **Never report the requested device as the one that ran.** Qwen ignores
  Device, onnx on a Mac ignores it, and Whisper ignores it without
  mlx-whisper. Everything user-facing (status bar, JSON export, self-test)
  goes through `Backend`, never `worker.device`.
- **A finished transcript is only replaced by a finished one.** A rerun
  stashes the old result (`Track.stash_result()`); a failure, cancel or
  abort restores it, and only `track_done` commits the replacement. Clearing
  and full reruns ask first.
- **Never destroy the window while a worker thread is alive.** mlx-whisper
  transcribes a whole track in one blocking call, a Qwen region runs up to
  240 s, and downloads are unbounded — none can be interrupted. `closeEvent`
  defers the close and `_quit_if_idle()` completes it; "Quit now" uses
  `os._exit(0)` rather than tearing down a live `QThread`.
- **No ffmpeg CLI**: all decoding goes through PyAV (`decode_track_audio`).
  Never hand mlx-whisper a file path — its loader shells out to an `ffmpeg`
  binary that is not installed here and is absent from the shipped `.app`;
  the failure is masked by the worker's silent CPU fallback.
- `main.py` must call `multiprocessing.freeze_support()` first. In the frozen
  `.app` a multiprocessing child re-executes the same binary, so without it
  a child launches a second copy of the GUI.
- Vocabulary hints: faster-whisper uses `hotwords=`, MLX uses
  `initial_prompt=`, Qwen uses `context=`; the field is hidden for onnx
  models (no support).
- The Qwen sparse-track VAD path was verified on this Mac with utterances at
  5/25/50 s: exactly three segments, starts at 5.02/25.06/50.03 s.
- **No CoreML EP for onnx-asr**: don't "restore" it in `_get_model`. It
  cannot load at all on onnxruntime <= 1.27 (the parakeet encoder's weights
  live in an external `.onnx.data` file and the CoreML partitioner loses the
  model path), and on 1.28, where it does load, it is *slower* than the CPU
  EP — CoreML takes ~1000 of the encoder's 2115 nodes as ~291 partitions, so
  the graph bounces between backends. Measured on the M1 Pro: 124 s of
  speech in 10.5 s (CoreML) vs 7.2 s (CPU); a single un-chunked 60 s encoder
  call, 76 s vs 2.9 s. The comment in `_get_model` carries these numbers.

## User context

- Primary use: transcribing D&D sessions (Discord/Craig multitrack
  recordings); vocab hints hold campaign names.
- Everything must stay free/local/unlimited — no cloud transcription APIs.
