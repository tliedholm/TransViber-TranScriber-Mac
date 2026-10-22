# TransViber TransScriber Code Audit

**Audit date:** August 6, 2026  
**Scope:** Read-only review of the macOS transcription app, including audio handling, transcription engines, AI Notes, UI state, exports, setup, and packaging.  
**Changes made during audit:** None.

## Executive summary

The code is generally well structured, and the audit found no basic syntax, dependency, audio-decoding, or bundle-signature failure. However, several correctness and reliability problems should be addressed.

The recommended implementation order is:

1. Backend and fallback correctness.
2. Failed-result isolation.
3. Per-run transcription configuration tracking.
4. Long-session AI Notes chunking.
5. Cancellation and shutdown behavior.
6. Packaging validation and automated tests.
7. Medium-priority data-loss and UX issues.

## High-priority findings

### 1. Backend fallback is sometimes fictitious or misleading

The retry logic assumes every non-CPU device can fall back to CPU. That assumption is not correct for all engines:

- Qwen always uses MLX regardless of the selected device. A Qwen failure may therefore reload and retry the same MLX backend while claiming it is falling back to CPU.
- ONNX uses the CPU on macOS even when the worker's requested device is `mlx`. An ONNX CPU failure may consequently be reported as a GPU failure and retried on the CPU again.
- If `mlx-whisper` is unavailable, Whisper silently runs through faster-whisper on the CPU while the worker and UI continue to say Metal.
- The packaging self-test checks `worker.device`, so it can incorrectly count silent CPU execution as a successful GPU test.

Relevant code:

- `app/transcriber.py:260`
- `app/transcriber.py:335`
- `app/transcriber.py:550`
- `packaging/selftest.py:55`

Recommended change: track the backend that was actually loaded separately from the requested device. Only offer a fallback when a genuine alternative backend exists, and report the backend ultimately used.

### 2. Segments from failed tracks can remain in the transcript and be exported

If an engine emits some segments and then raises an exception, the track becomes `ERROR`, but the segments already emitted remain attached to it. `merge_tracks()` includes segments from every track regardless of its status, so the UI and exports can present an incomplete track as part of the finished conversation.

Relevant code:

- `app/transcriber.py:549`
- `app/main_window.py:825`
- `app/merger.py:8`

Recommended change: buffer the segments from each attempt until that attempt succeeds, or explicitly discard and clearly label partial results when a track fails.

### 3. A transcript can silently contain results from different models or settings

When some tracks are already `DONE`, another run processes only tracks that are not done. The model, language, device, and vocabulary hints can all be changed between runs. This can create a mixed transcript in which different tracks were generated with different configurations.

JSON export then records only the currently selected model, making it appear that one model produced the entire transcript.

Relevant code:

- `app/main_window.py:745`
- `app/main_window.py:760`
- `app/exporters.py:67`

Recommended change: store the transcription configuration per track or per run. When material settings change, require or offer a full rerun. Export the actual configuration used for each track.

### 4. Long AI Notes jobs can exceed the model's context window

Transcript splitting only occurs at line boundaries. An individual oversized line is never subdivided. A read-only probe confirmed that a 70,000-token single line remains one 70,000-token part.

This can happen naturally because consecutive same-speaker segments are coalesced into one line. A single-speaker recording or long monologue may therefore bypass the intended token limit.

The combine pass also sends every part summary into one final prompt. With up to 6,000 generated tokens per part, a sufficiently long D&D session can exceed the default Qwen3 model's verified 40,960-token context window.

Relevant code:

- `app/llm.py:84`
- `app/llm.py:135`
- `app/llm.py:229`
- `app/merger.py:15`

Recommended change:

- Split oversized lines using token boundaries.
- Include prompt and expected output size in the token budget.
- Combine summaries hierarchically in bounded batches instead of one unlimited final pass.

### 5. Closing during transcription or model loading can leave live threads behind

Shutdown waits five seconds for each main worker and three seconds for the statistics worker, then accepts the close event whether or not those threads actually stopped.

Some operations can block much longer:

- MLX Whisper performs one blocking call for an entire track.
- Qwen speech regions can be as long as 240 seconds.
- Model loading and Hugging Face downloads are blocking.
- The statistics thread's `ioreg` call can itself wait up to five seconds before the thread sleeps.

Destroying a running `QThread` can crash or hang the application. The Cancel button has a related limitation: it changes a Boolean flag, but cannot interrupt model loading or a currently blocking MLX call.

Relevant code:

- `app/main_window.py:981`
- `app/transcriber.py:383`
- `app/transcriber.py:437`
- `app/stats.py:53`

Recommended change: do not destroy the window until workers have genuinely finished. Clearly communicate when cancellation must wait for the current operation, and use smaller interruptible chunks where supported.

### 6. The distributable build can succeed with missing engine components

The PyInstaller specification catches and ignores every `collect_all()` failure. The build script does not perform dependency preflight checks and does not build or execute the supplied frozen self-test. There are also no standard unit tests in the repository.

A bundle can therefore be produced and signed even when one of its advertised engines is incomplete.

Relevant code:

- `packaging/TransViber.spec:20`
- `packaging/build_app.sh:37`
- `packaging/selftest.py:1`

Recommended change:

- Fail the build when required packages cannot be collected.
- Run explicit import and resource checks before packaging.
- Build and run the frozen engine self-test as a release gate.
- Add fast unit tests for routing, splitting, merging, exports, state transitions, and fallback behavior.

## Medium-priority findings

### 7. Finder "Open With" and dragging audio onto the app icon are not implemented

The bundle registers audio document types and enables `argv_emulation`, but `main.py` never forwards file arguments to `MainWindow.add_files()`. There is also no macOS file-open event handler for files sent to an already-running app.

Dragging files into the visible window works. Opening them through Finder or the Dock likely does not.

Relevant code:

- `packaging/TransViber.spec:150`
- `app/main.py:13`

Recommended change: handle initial file arguments and macOS `QFileOpenEvent` events, validate the extensions, and pass accepted paths to `add_files()`.

### 8. Rerunning completed tracks destroys the existing transcript immediately

If every track is done, pressing Transcribe selects every track and clears all existing segments before the replacement run succeeds. An accidental click, cancellation, model failure, or download failure can erase hours of successful transcription.

The Clear action also removes the loaded transcript without confirmation.

Relevant code:

- `app/main_window.py:745`
- `app/main_window.py:752`
- `app/main_window.py:727`

Recommended change: keep the previous successful transcript until the replacement succeeds. Add confirmation for destructive reruns and clearing nonempty results.

### 9. "All formats" can overwrite additional files without confirmation

The save dialog confirms only the selected pathname. `export_all()` then writes `.md`, `.txt`, `.srt`, `.vtt`, and `.json` siblings directly. Four files the user did not explicitly select can therefore be overwritten without individual warning.

Relevant code:

- `app/main_window.py:914`
- `app/exporters.py:115`

Recommended change: preview all target paths and request one confirmation if any of them already exist.

### 10. Dependencies are completely unpinned

`setup_mac.command` installs the latest available versions of all major ML and UI packages. There is no lockfile or tested version range.

Consequences include:

- A clean installation may behave differently from the current working environment.
- An existing virtual environment may retain unexpectedly old versions.
- A future upstream API change can break model loading, timestamps, or packaging without any source-code change in this project.

Relevant code:

- `setup_mac.command:36`

Recommended change: maintain a tested dependency lockfile or explicit compatible version ranges and record the environment used for each released bundle.

### 11. Model failures do not retain enough diagnostic information

Most exceptions are reduced to `str(exc)`. No traceback or persistent diagnostic log is kept. This makes failures inside a frozen, console-free app difficult to investigate.

Relevant code:

- `app/transcriber.py:536`
- `app/llm.py:253`

Recommended change: write a local rotating diagnostic log containing timestamps, selected engine, requested and actual backend, track path, dependency versions, exception type, and traceback. Keep user-facing dialogs concise.

### 12. Audio probing suppresses useful errors

A corrupt or unsupported file is added as a normal zero-duration track because probing catches every exception and treats it like an unknown single-stream file. The user only sees a failure later during transcription.

Relevant code:

- `app/audio.py:52`

Recommended change: distinguish a genuine single-stream fallback from a probe failure. Surface the original codec or container error when adding the file.

## Smaller findings

### 13. Blank speaker names create a UI/data mismatch

Entering a blank speaker name leaves the table cell blank while the internal track retains the previous speaker name.

Relevant code:

- `app/main_window.py:673`

Recommended change: restore the previous text in the cell or reject the edit visibly.

### 14. Markdown and WebVTT speaker names are not escaped

Speaker names containing Markdown markers, `<`, `>`, or `&` can produce malformed Markdown or WebVTT output.

Relevant code:

- `app/exporters.py:32`
- `app/exporters.py:58`

Recommended change: escape speaker names according to each output format and add exporter tests using punctuation and markup characters.

### 15. The packaged documentation mislabels Parakeet v3

The shareable readme labels `parakeet-tdt-0.6b-v3` as English-only, while the UI and main readme describe v3 as multilingual.

Relevant documentation:

- `packaging/READ ME FIRST.txt:49`

Recommended change: make the model descriptions consistent across the UI and both readmes.

## Verification performed

The following read-only checks passed:

- All 11 project Python files parsed successfully with Python's AST parser.
- `pip check` reported no broken installed requirements.
- Synthetic stereo decoding preserved distinct left and right channel values.
- The synthetic 48 kHz input was correctly resampled to 16 kHz.
- Chronological merge behavior passed a smoke check.
- Markdown, text, SRT, VTT, and JSON exporters passed smoke checks.
- The current `.app` passed deep, strict code-signature verification.
- The important Qwen VAD and timestamp code still follows the previously verified sparse-track design.

## Verification limitations

Full Metal/MLX inference could not safely be rerun in the audit environment. Importing the MLX runtime terminated because no Metal device is available in the sandbox. Native validation should therefore be rerun outside the restricted environment before distributing a new build.

No source code or project configuration was changed during the audit.
