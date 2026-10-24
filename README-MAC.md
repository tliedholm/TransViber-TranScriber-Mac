# TransViber TransScriber — Mac edition

Multi-track speaker-aligned transcription, running fully on your Mac.
Built for Apple Silicon (M1 Pro and newer).

## Install (one time)

1. Copy this whole folder somewhere permanent (e.g. Applications or your
   home folder) — not straight from the zip preview.
2. Open **Terminal**, type `bash ` (with the trailing space), drag
   `setup_mac.command` from Finder into the Terminal window, press Enter.
3. Wait for "Setup complete" (downloads a few hundred MB of libraries once).
   If it says no Python 3.10+ was found, install one with
   [Homebrew](https://brew.sh): `brew install python@3.12`, then re-run.

## Run

Launch **TransViber TransScriber** from Spotlight (Cmd+Space) or from
`~/Applications` — it runs as a normal Mac app with its own Dock icon.

The app bundle in `~/Applications` is a thin launcher that points at this
folder's `.venv` and `app/` code. If you move this folder, edit `APP_DIR`
in the bundle's `Contents/MacOS/TransViberTransScriber` script.

Fallback: double-clicking **`TransViber TransScriber.command`** still works
(opens a Terminal window alongside the app).

## Usage

Identical to the Windows version: drop one audio file per speaker (or a
single multitrack/multichannel file), name the speakers, pick a model,
Transcribe, Export. Vocabulary hints work with Whisper models for D&D names.

### Someone joined late: Offset and Align

Multitrack recorders (Craig, OBS, REAPER) start every track at the same
moment, so there is nothing to set. If someone recorded separately and
joined late, set their track's **Offset**: when their file started,
measured from the start of the session. Double-click the cell and type
seconds (`150`), minutes:seconds (`2:30`) or hours:minutes:seconds
(`1:02:30`); a comma works as the decimal point. Their lines move to the
right place straight away, in the transcript, AI Notes and every export,
without transcribing again. Leave it at `0:00` if the file already contains
the waiting time as silence at the start, and type `0` to reset it.

**Align…** can work the offsets out for you from sound the recordings
share: people in the same room, one voice picked up by another mic, or a
recording of the whole session. Separately recorded headsets usually share
no sound with each other, so for those, add a full-session recording (a
call or room mix) with **Add session reference…** above the track list.
It is used for timing only: it is never transcribed, its start counts as
0:00, and the individual speaker files are still what gets transcribed.
Align only reads the files (no AI model, no GPU) and shows a review window:
strong matches are ticked, weak matches and guesses are not, and nothing
changes until you press **Apply**. When there is no shared sound to go on,
the only suggestion is a labelled guess that every recorder stopped at the
same moment; check it before ticking it.

The **Notes** tab mirrors the merged transcript and keeps manually entered
notes autosaved on this Mac. **AI Notes** uses a downloadable local MLX model
to write Markdown summaries; its optional instructions are added to the
built-in summary prompt. ASR and notes models are unloaded before the other
kind starts so they do not compete for unified memory.

AI Notes are laid out as: summary, key points and decisions, memorable
moments, open questions, then **Per-person summary** (each speaker and any
character they play), then two sections aimed at the *next* session:

- **Where we left off** — the situation in progress when the session ended,
  anything interrupted or unfinished, and the immediate next steps.
- **Unresolved threads** — one bullet per still-open thread, with what it
  is, who it involves, its last known status, and the suggested next action.

These are written from the session you just transcribed; nothing from an
earlier session is carried in.

## Models & speed on the M1 Pro

- **Device: GPU (Metal)** is the default.
  - Whisper models run through **mlx-whisper** on the Metal GPU —
    `large-v3-turbo` is roughly 5–10× faster than real time.
  - `qwen3-asr-1.7b` / `-0.6b` run through **mlx-qwen3-asr** on the Metal
    GPU (always, whatever Device says) — state-of-the-art accuracy,
    30 languages, and vocabulary hints work. Qwen3-ASR has no Norwegian or
    Ukrainian, so those are greyed out while a Qwen model is selected, and
    Transcribe asks for another choice if one of them was already picked.
    First use downloads the model plus a small aligner model for word
    timestamps.
  - `parakeet-tdt-0.6b-v2/v3` run through **onnx-asr on the CPU** whatever
    Device says — still very fast (about 15–20× real time), great English
    accuracy, no vocabulary-hint support. CoreML is deliberately not used:
    it splits the encoder into hundreds of fragments and comes out slower
    than the CPU.
- **Device: CPU** uses faster-whisper (int8). Fine for smaller models.
- Live segment-by-segment transcript updates only happen on the CPU engine;
  on the Metal GPU each track's text appears when that track finishes.
- Models download on first use to `~/.cache/huggingface` and are then local
  and offline. Everything is free and unlimited.
- The status bar reports the engine and device that **actually** ran, which
  is not always what Device asked for — Qwen is always Metal, Parakeet and
  Canary are always the CPU, and Whisper says so if it had to use the CPU.

## What happens when something goes wrong

- **A track fails.** Any partial text it produced is left out of the merged
  transcript and every export — an incomplete track would otherwise read as
  finished. The count is shown in the status bar, and the Status cell's
  tooltip has the error.
- **A rerun fails or is cancelled.** The previous successful transcript for
  that track is restored, so a stray click, a failed download or a cancel
  cannot destroy hours of finished work.
- **Settings changed since some tracks finished.** Transcribing the rest
  would mix models or languages in one transcript, so the app asks whether to
  redo everything instead.
- **Quitting mid-run.** The window stays open until the worker really stops.
  Transcribing a track, a Qwen speech region and a model download cannot be
  interrupted part-way, so cancelling waits for the current step; there is a
  "Quit now" button if you would rather not wait.
- **Details.** Every failure is written with its traceback, the engine and
  backend used, and package versions to
  `~/Library/Logs/TransViber TransScriber/transviber.log` (rotating).

## Development

- Fast unit tests (no models, no downloads): `.venv/bin/python -m pytest tests`
- Environment check: `.venv/bin/python packaging/preflight.py`
- Dependencies are pinned in `requirements-mac.txt`. Bump them there, re-run
  the tests, the preflight and the frozen self-test, then note the versions
  in that file's header.
- `bash packaging/build_app.sh` runs the tests, the preflight, the spec's
  required-package check and the frozen bundle self-test before it will
  produce a signed `.app`. Set `SELFTEST_AUDIO=/path/to/sample.wav` to also
  run every engine inside the frozen bundle — do that before a release, as
  it is the only thing that proves Metal inference survived freezing.

## Notes

- All processing is local; nothing is uploaded anywhere.
- A 3–4 hour D&D session across 4–5 speaker tracks is a normal workload —
  expect roughly 20–40 minutes with `large-v3-turbo` on the M1 Pro, less
  with Parakeet. Plug the MacBook in; it will run the fans.
- Audio files can be opened from Finder ("Open With") or dropped on the Dock
  icon as well as dragged into the window.
