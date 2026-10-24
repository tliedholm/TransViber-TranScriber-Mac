# TransViber TransScriber — Mac edition

Local, multi-track transcription for Apple Silicon Macs. Give it one audio
track per speaker (or a single multitrack recording) and it transcribes each
one on your Mac, then merges the results into a single conversation in the
order people actually spoke.

It was built for tabletop sessions recorded with Discord's Craig bot, where
every player ends up on their own track, but it works for any recording made
that way: podcasts, interviews, meetings.

- **Fully local.** Audio never leaves the Mac. No accounts, no API keys, no
  usage limits. Models download once and then work offline.
- **Four speech engines** on the Apple GPU or CPU: Qwen3-ASR, Whisper
  (mlx-whisper or faster-whisper) and NVIDIA Parakeet/Canary.
- **Speaker-aligned merge.** Segments are split at pauses in speech and
  sorted by time, so cross-talk interleaves correctly instead of each speaker
  arriving as one block.
- **Offset and Align** for someone who joined late or recorded on another
  device: type when their file began, or let Align suggest it from sound
  the recordings share.
- **Vocabulary hints** for names the model wouldn't know (characters, places,
  jargon).
- **AI Notes**: a local LLM (via mlx-lm) writes a Markdown session summary,
  ending with where things were left off and the threads still open.
- **Exports** to Markdown, plain text, SRT, WebVTT and JSON.

This is the macOS fork of the Windows app of the same name. The two are
maintained separately.

## Requirements

- A Mac with Apple Silicon (M1 or newer). Developed and tuned on an M1 Pro.
- macOS 13.5 or later (MLX's minimum). Last verified on macOS 26.
- Python 3.10 or newer to run from source (3.12 is what the pinned versions
  were verified with). If you don't have one: `brew install python@3.12`.
- Disk space for models: a few hundred MB of libraries, plus anywhere from a
  few hundred MB to a few GB per model you use.

## Install and run (from source)

```bash
git clone https://github.com/tliedholm/TransViber-TranScriber-Mac.git
cd TransViber-TranScriber-Mac
bash setup_mac.command
```

`setup_mac.command` finds a Python 3.10+, creates `.venv`, installs the
pinned dependencies from `requirements-mac.txt`, and runs
`packaging/preflight.py` to check that every engine imports. It only needs
to run once (or again after the requirements change).

Then launch the app with either of:

```bash
./TransViber\ TransScriber.command      # or double-click it in Finder
.venv/bin/python app/main.py
```

If macOS blocks the `.command` file the first time, right-click it, choose
**Open**, then **Open** again.

To share the app with someone who won't set up Python, build the
self-contained `.app` described under [Building the app](#building-the-app).
That zip includes `READ ME FIRST.txt`, which walks a non-technical user
through installing it.

## Using it

1. **Add the audio.** Drag files into the window, use **Add Files…**, or open
   them from Finder ("Open With", or drop them on the Dock icon). Each file
   becomes a track named after the file, e.g. `alice_track-01.flac` becomes
   "Alice Track 01".
   - A file with several audio streams (Craig, OBS) becomes one track per
     stream, and a file with more than two channels one track per channel.
   - For a stereo file with a different person on each side, right-click it
     and choose **Split into one speaker per channel**.
   - Supported: `.flac .m4a .mp4 .aac .mp3 .wav .ogg .opus .wma .webm .mka
     .mkv .mov .aif .aiff`. Decoding is done in-process by PyAV, so no
     `ffmpeg` install is needed.
2. **Name the speakers** by editing the Speaker column.
3. **Pick a model, language and device.** See [Engines and
   models](#engines-and-models). The ☆ next to the model pins it as the
   default at startup.
4. **Add vocabulary hints** (left-hand box): one name or term per line, or
   comma-separated. Remembered between sessions. Hidden for Parakeet and
   Canary, which don't support them.
5. **Transcribe.** Each track's status and progress show in the table, the
   merged transcript fills in below, and the status bar shows the engine and
   device that actually ran, model-download progress, and GPU/CPU/RAM load.
6. **Export…** as Markdown, plain text, SRT, WebVTT, JSON, or all five at
   once. Markdown and text merge consecutive lines from the same speaker;
   JSON also records, per track, the model and backend that produced it and
   its offset.

Multitrack recorders such as Craig start every track at the same moment, so
every **Offset** stays at `0:00`. When a file began later (someone joined
late, or recorded on their own device), type when it began in its Offset
cell (`150`, `2:30` or `1:02:30`). The transcript, AI Notes and exports
follow straight away, with no re-transcription. **Align…** can suggest the
offsets from sound the files share, or from a full-session mix added as the
session reference, which is used for timing only and never transcribed.
Nothing changes until you tick its suggestions and press Apply.
[`README-MAC.md`](README-MAC.md) has the details.

### Notes tab

The **Notes** tab has a notes editor that autosaves as you type, next to a
read-only copy of the merged transcript. **Show instructions** and **Show
transcript** hide the AI Notes controls or the transcript copy when you want
more room to write.

**AI Notes** sends the merged transcript to a local LLM and streams Markdown
notes into the editor: a summary, key points and decisions, memorable
moments, open questions, a **Per-person summary** (each speaker and any
character they play), then two sections for picking up next time:

- **Where we left off**: the situation in progress when the session ended,
  anything unfinished, and the immediate next steps.
- **Unresolved threads**: one bullet per open thread, with who it involves,
  its last known status and a suggested next action.

Notes models come from mlx-community (4-bit, Apple GPU), in three tiers:

| Tier | Models |
|---|---|
| Max detail | `qwen3-14b`, `gemma-3-12b`, `mistral-nemo-12b` |
| Standard | `qwen3-8b`, `llama-3.1-8b`, `ministral-8b` |
| Quick | `qwen3-4b` (default), `llama-3.2-3b`, `phi-4-mini`, `granite-3.3-2b` |

You can also type any mlx-community repo id. Optional instructions (e.g.
"focus on loot and quest leads") are added to the built-in prompt, never in
place of it. Long sessions are summarized in parts and then combined, with
every pass sized to the model's real context window. Instructions too long
to leave room for the transcript are refused up front. Speech and notes
models are never loaded at the same time, so they don't compete for unified
memory. **Export Notes…** saves the notes as Markdown or plain text.

## Engines and models

The Model menu is grouped by tier, from most detailed to fastest.

| Model | Engine | Runs on | Notes |
|---|---|---|---|
| `qwen3-asr-1.7b`, `qwen3-asr-0.6b` | mlx-qwen3-asr, with Qwen3-ForcedAligner for word timings | Metal GPU, always | Highest accuracy. Vocabulary hints supported. All of the app's languages except Norwegian and Ukrainian, which are greyed out while Qwen is selected. |
| `large-v3`, `large-v2`, `large-v3-turbo`, `medium`, `medium.en`, `distil-large-v3`, `small`, `base`, `tiny` (and `.en` variants) | mlx-whisper on GPU, faster-whisper (int8) on CPU | Follows the Device setting | `large-v3-turbo` is the all-rounder: roughly 5–10× real time on an M1 Pro. `.en` models are English-only. |
| `distil-large-v3.5-ct2`, or any CTranslate2 repo id you type | faster-whisper | CPU | Use with Device set to CPU. |
| `parakeet-tdt-0.6b-v3`, `parakeet-tdt-0.6b-v2` | onnx-asr with Silero VAD | CPU, always | Fastest: about 15–20× real time. v3 covers 25 European languages, v2 is English-only. No vocabulary hints; Language is ignored. |
| `canary-1b-v2` | onnx-asr with Silero VAD | CPU, always | Very accurate. No vocabulary hints; Language is ignored. |

A few things worth knowing:

- **Device** is a request, not a guarantee. Qwen always runs on the GPU,
  Parakeet and Canary always on the CPU, and Whisper drops to the CPU if
  mlx-whisper isn't available. The status bar and the JSON export always
  report what actually ran.
- Parakeet and Canary stay on the CPU on purpose. CoreML splits their
  encoder into hundreds of fragments and comes out slower than the plain CPU.
- Whisper splits its segments wherever there is more than a second of
  silence between words, so one speaker's separate remarks land at their
  real times. Both Whisper paths and Qwen also skip silence (VAD or a
  no-speech filter), which keeps mostly-silent speaker tracks from producing
  invented text.
- faster-whisper, Parakeet/Canary and Qwen show text as they go (Qwen one
  stretch of speech at a time). mlx-whisper transcribes a whole track in one
  call, so its text appears when that track finishes.
- A 3–4 hour session across 4–5 speaker tracks takes roughly 20–40 minutes
  with `large-v3-turbo` on an M1 Pro, less with Parakeet. Plug the laptop in.
- Models download on first use to `~/.cache/huggingface`. Delete that folder
  to reclaim the space.

## When something goes wrong

- **A track fails.** Any partial text it produced is kept out of the merged
  transcript and every export, so an incomplete track can't pass as a
  finished one. The Status cell's tooltip has the error.
- **A rerun fails or is cancelled.** The track's previous finished transcript
  is restored. Clearing tracks or redoing everything asks first.
- **Settings changed since some tracks finished.** The app asks whether to
  redo everything rather than mix models or languages in one transcript.
- **Odd timestamps in a file.** Gaps and jumps in a recording's timing are
  handled on decode, and anything unusual is shown as "Done ⚠" with the
  details in the Status tooltip (and `decode_note` in the JSON).
- **Quitting mid-run.** Some steps (a model download, an mlx-whisper track, a
  Qwen speech region) can't be interrupted, so the window waits for the
  current one to finish. **Quit now** is there if you'd rather not wait.
- **Logs.** Every failure is written with its traceback, model, requested
  device, backend actually used and package versions to
  `~/Library/Logs/TransViber TransScriber/transviber.log`.

## Development

```bash
.venv/bin/python -m pytest tests          # fast: no models, no downloads
.venv/bin/python packaging/preflight.py   # every engine and resource imports
```

- The window tests build a real `MainWindow` offscreen, with settings in a
  scratch file.
- Direct dependencies are pinned in `requirements-mac.txt`. To upgrade, bump
  them there, then re-run the tests, the preflight and the frozen self-test
  below, and note the versions in that file's header.
- `CLAUDE.md` describes the architecture and the invariants that must not be
  broken (word-gap splitting, hallucination guards, decoding on the file's
  own clock, offsets living in one place, reporting the backend that
  actually ran, and more). Read it before changing the engines, audio
  decoding, the merge or Align.

### Building the app

```bash
bash packaging/build_app.sh
SELFTEST_AUDIO=/path/to/sample.wav bash packaging/build_app.sh   # before a release
```

This builds a self-contained, ad-hoc-signed `TransViber TransScriber.app`
with PyInstaller (`packaging/TransViber.spec`) and zips it with
`READ ME FIRST.txt`. The version lives in the spec's `BUNDLE`. The build stops
before signing if any of these fail, in order: the unit tests, the preflight,
the spec's required-package check (a missing engine raises instead of being
skipped), and the frozen self-test (`selftest --imports-only`). With
`SELFTEST_AUDIO` set, the self-test also transcribes the sample with every
engine inside the frozen bundle and checks each ran on the expected backend.
That is the only check that proves Metal inference survived freezing.

The resulting app isn't notarized, so on first launch the recipient has to
allow it once in System Settings > Privacy & Security ("Open Anyway").
`READ ME FIRST.txt` explains this.

### Project layout

```
app/
  main.py          entry point; handles files opened from Finder
  main_window.py   the whole UI: Transcribe and Notes tabs, settings
  transcriber.py   TranscribeWorker and the four engine paths, backend resolution
  audio.py         PyAV decoding to 16 kHz mono on the file's own clock; multitrack expansion
  align.py         the Offset column's parsing, and Align: offsets from shared sound
  merger.py        chronological merge (with each track's offset) and same-speaker coalescing
  exporters.py     md / txt / srt / vtt / json
  llm.py           AI Notes (mlx-lm), context-window budgeting, part/combine passes
  models.py        Track, Segment, RunConfig and SessionReference
  stats.py         GPU/CPU/RAM poller for the status bar
  diagnostics.py   rotating log file
packaging/         build_app.sh, PyInstaller specs, preflight and frozen self-test
tests/             pytest suite
setup_mac.command  one-time environment setup
```

Other docs in the repo: [`README-MAC.md`](README-MAC.md) (the end-user guide
for this folder), [`CLAUDE.md`](CLAUDE.md) (architecture and invariants),
[`CODE-AUDIT.md`](CODE-AUDIT.md), and the notes for porting features to the
Windows edition, [`QWEN-PORT-NOTES.md`](QWEN-PORT-NOTES.md) and
[`TIMER-AND-STATS-PORT-NOTES.md`](TIMER-AND-STATS-PORT-NOTES.md).
