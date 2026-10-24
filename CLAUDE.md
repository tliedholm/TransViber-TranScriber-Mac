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
  `tests/test_track_status.py` and `tests/test_window_lifecycle.py` build a
  real `MainWindow` offscreen, with settings in a scratch file; a new window
  test should delete its window explicitly, since PySide6 6.11 can crash
  collecting a closed one. `test_window_lifecycle` covers the export
  warning and failure reports, and closing with unsaved notes — including
  a real notes thread whose last chunks are still queued when it stops.
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
  Notes tab with autosaved manual/local-AI notes. The notes live in
  QSettings (`"notes"`); `_save_notes` syncs and checks `status()` so a
  failed write is reported, never assumed. `_notes_dirty` is set by every
  change, typed or streamed, and cleared only by a save that succeeded, so
  untouched notes are never rewritten and a failed save is never forgotten.
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
     Its languages are a closed list: `QWEN_LANGUAGE_NAMES` holds only names
     in the model's own `support_languages` (no Norwegian or Ukrainian).
     mlx-qwen3-asr does not enforce that list (it warns on stderr and forces
     the name into the prompt), so `language_supported` is checked twice:
     the window greys out what the chosen model lacks and refuses a
     selection it already had, never turning it into auto-detect, and
     `TranscribeWorker.run` refuses it before any model loads.
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
  the whole request and `MAX_NOTES_TOKENS` of output. What a pass carries
  besides its content (system text, user instructions, chat template,
  Qwen3's thinking switch, Gemma's folded system turn) is measured once a
  run by building that pass with empty content through `_request()`, the
  same builder generation uses. `stream_generate` is handed `_request()`'s
  ids, so what was counted is what runs; the direct prompt is counted whole
  before it is sent. (The PC's `/tokenize` and character fallbacks have no
  counterpart here: the tokenizer is in-process and is generation's own.)
  Instructions that leave less than `MIN_PROMPT_TOKENS` of room fail the
  run in `_budgets()` with a "shorten them" message, before the notes
  header is written. The minimum is a check, never a floor: the old
  1,000-token floor sent prompts bigger than the window. mlx-lm neither
  truncates nor refuses an oversized prompt — it prefills all of it and
  the model runs past its trained window, so the notes degrade with no
  error — which makes this sizing the only guard. `split_text()` splits at line
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
  is a right-click action). `decode_track_audio()` -> mono float32 16 kHz,
  on the file's own clock (see the invariants), reporting what it ran into
  in a `Timing`.
- `merger.py` / `exporters.py` — chronological merge, same-speaker
  coalescing, and md/txt/srt/vtt/json exporters. `merge_tracks()` skips
  `ERROR` tracks: an engine can emit segments and *then* raise, and those
  partial segments would make an incomplete track read as finished. They
  stay on the Track (the UI reports the count) but never reach the
  transcript or an export. Speaker names are escaped per format.
  Export first warns about every track that isn't Done
  (`_missing_from_export`): failed (left out whole), cancelled (partial or
  empty) and never transcribed alike; a Done track with no segments is
  silent audio and doesn't count. Every export, Export Notes included,
  stages each file beside its target (unique name, fsynced, the replaced
  file's permissions or the umask's — not mkstemp's 0o600) before renaming
  any into place, so a failure never half-writes a file. A rename that
  fails raises `ExportError` (an `OSError`) naming the files already new
  and the ones still old. Five renames are not one transaction; don't let
  a docstring claim they are.

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
- **Decode on the file's own clock.** Every engine, faster-whisper included,
  is handed `decode_track_audio`'s array, never a path: faster-whisper's own
  decoder lays audio end to end, so a stream that began late in its file
  lands at 0:00 and everything after a gap in the timestamps moves earlier,
  which breaks the same-instant rule above. `audio.py` measures every stream
  in a file from one 0:00 (`container.start_time`, or the earliest packet
  timestamp when the file declares none) and places each run of frames by
  its timestamp: within `TIMING_TOLERANCE` (20 ms) it follows on; later is a
  gap filled with silence (the resampler is restarted so it never blends
  across one); earlier plays on from where the audio had got to and is
  noted as a jump back, so no audio is dropped. A packet that won't decode
  is skipped and counted. Silence is made up only to `MAX_TIMELINE_SECONDS`
  (24 h). Ordinary files decode sample for sample as before (checked on
  flac, m4a, mp3, ogg, opus, wav, mka, webm).
  What decoding ran into reaches the track as `Track.note` via the worker's
  `track_note` signal: "Done ⚠" with the note in the Status tooltip,
  `decode_note` in the JSON export, `note:` lines in the self-test. The note
  is stashed and restored with the transcript it came with. An
  `AudioFileError` is the file's own problem, so the worker fails that track
  without the fallback retry, which would read it no better and leave every
  later track off the GPU.
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
  `os._exit(0)` rather than tearing down a live `QThread`. A job counts as
  over once its finish handler has run (it drops the window's reference),
  not when `isRunning()` turns false: the last notes chunks are still
  queued behind a finished thread. `QThread.finished` is the backstop for a
  run that dies without its finish signal (`_on_thread_finished`).
- **A close never loses notes it could not write.** Only once every job is
  over are the notes saved for the last time; if that save fails (or the
  notes can't be saved before "Quit now"), one prompt offers Retry / Save
  Elsewhere / Return to App / Close Without Saving. Return to App calls the
  quit off with the stats poller and autosave still running, and the
  finish handlers reset their controls even while quitting.
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
