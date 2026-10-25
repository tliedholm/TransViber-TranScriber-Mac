"""Export merged transcripts to Markdown, plain text, SRT, VTT, and JSON."""

from __future__ import annotations

import contextlib
import html
import json
import os
import re
import secrets
import stat
from datetime import datetime
from pathlib import Path

from merger import coalesce, merge_tracks
from models import Segment, Track

# Characters that would otherwise turn a speaker's name into Markdown markup:
# a name like *Nyx* or a nickname written as _Bones_ silently became emphasis,
# and one containing [] or ` produced broken links and code spans.
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>])")


def _md(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", text)


def _vtt(text: str) -> str:
    """Escape a name for a WebVTT <v ...> cue tag."""
    return html.escape(text, quote=False)


def hms(seconds: float) -> str:
    """Whole seconds as `m:ss`, or `h:mm:ss` from an hour: the clock the
    track table, the transcript and the text exports all show."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _ts_srt(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(seconds: float) -> str:
    return _ts_srt(seconds).replace(",", ".")


def to_markdown(segments: list[Segment]) -> str:
    lines = ["# Transcript", ""]
    for seg in coalesce(segments):
        lines.append(f"**{_md(seg.speaker)}** \\[{hms(seg.start)}\\]: {seg.text}")
        lines.append("")
    return "\n".join(lines)


def to_text(segments: list[Segment]) -> str:
    lines = []
    for seg in coalesce(segments):
        lines.append(f"[{hms(seg.start)}] {seg.speaker}: {seg.text}")
        lines.append("")
    return "\n".join(lines)


def to_srt(segments: list[Segment]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_ts_srt(seg.start)} --> {_ts_srt(seg.end)}")
        lines.append(f"{seg.speaker}: {seg.text.strip()}")
        lines.append("")
    return "\n".join(lines)


def to_vtt(segments: list[Segment]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_ts_vtt(seg.start)} --> {_ts_vtt(seg.end)}")
        lines.append(f"<v {_vtt(seg.speaker)}>{_vtt(seg.text.strip())}")
        lines.append("")
    return "\n".join(lines)


def _track_entry(track: Track) -> dict:
    entry = {
        "file": str(track.path),
        "speaker": track.speaker,
        "duration": round(track.duration, 3),
        "language": track.language,
        "stream": track.stream_index,
        "channel": track.channel,
        # Segment times are session times: this track's own file times plus
        # this offset.
        "offset": round(track.offset, 3),
        "offset_source": track.offset_source.value,
        "status": track.status.value,
    }
    # What decoding ran into (timestamps that jump back, damaged packets):
    # the words are all there, but times after it may be off.
    if track.note:
        entry["decode_note"] = track.note
    # The settings that made *this* track, not whatever the combo boxes hold
    # now: tracks can be transcribed across runs with different models.
    if track.config is not None:
        entry["transcription"] = {
            "model": track.config.model,
            "language_requested": track.config.language,
            "device_requested": track.config.device,
            "backend": track.config.backend,
            "vocabulary_hints": track.config.hints,
        }
    return entry


def _models_used(tracks: list[Track]) -> list[str]:
    seen = []
    for track in tracks:
        if track.config and track.config.model and track.config.model not in seen:
            seen.append(track.config.model)
    return seen


def to_json(segments: list[Segment], tracks: list[Track]) -> str:
    exported = [t for t in tracks if t.in_transcript]
    data = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        # A list, because a transcript assembled over several runs genuinely
        # has more than one. Recording only the current selection made a
        # mixed transcript look like one model's work.
        "models": _models_used(exported),
        "tracks": [_track_entry(t) for t in exported],
        "excluded_tracks": [_track_entry(t) for t in tracks if not t.in_transcript],
        "segments": [
            {
                "speaker": seg.speaker,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            }
            for seg in segments
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


# Every renderer takes (segments, tracks); only JSON records the tracks.
EXPORTERS = {
    ".md": lambda segments, _tracks: to_markdown(segments),
    ".txt": lambda segments, _tracks: to_text(segments),
    ".srt": lambda segments, _tracks: to_srt(segments),
    ".vtt": lambda segments, _tracks: to_vtt(segments),
    ".json": to_json,
}


class ExportError(OSError):
    """An export that stopped partway, and exactly how far it got.

    `written` were replaced in full before `failed` could not be; every
    path in `unchanged` (`failed` first) still holds whatever it held
    before, if it existed at all. An OSError, so a caller that only reports
    `str(exc)` still says all of that.
    """

    def __init__(self, written: list[Path], failed: Path,
                 unchanged: list[Path], cause: OSError):
        self.written = written
        self.failed = failed
        self.unchanged = unchanged
        self.cause = cause
        reason = cause.strerror or str(cause)
        if written:
            message = (
                f"Exported {_names(written)}, but could not write "
                f"{failed.name}: {reason}\n\nNot updated: {_names(unchanged)}. "
                "Any earlier copies of those were left as they were, so they "
                "may not match the files just exported.")
        else:
            message = (f"Could not write {failed.name}: {reason}\n\n"
                       "Nothing was exported, and no existing file was changed.")
        super().__init__(message)


def _names(paths: list[Path]) -> str:
    return ", ".join(p.name for p in paths)


def _stage(path: Path, text: str) -> Path:
    """Write `text` to a new temporary file beside `path`, flushed to disk.

    Beside it, so the rename that publishes it stays on one volume; a
    unique name, so two exports to one folder can't share a temp file.
    Created with the permissions a plain write would leave: the replaced
    file's own, or 0o666 less the umask for a new one. (mkstemp's 0o600
    would make every export readable by its owner alone once renamed.)
    """
    try:
        keep_mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        keep_mode = None
    while True:
        temp = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            break
        except FileExistsError:
            continue
    try:
        with open(fd, "w", encoding="utf-8") as handle:
            if keep_mode is not None:
                os.fchmod(handle.fileno(), keep_mode)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        _discard(temp)
        raise
    return temp


def _discard(temp: Path) -> None:
    """Remove a staged file, quietly: a temp file left behind must never
    replace the error that explains why the export failed."""
    with contextlib.suppress(OSError):
        temp.unlink()


def _publish(files: list[tuple[Path, str]]) -> list[Path]:
    """Replace each file with its text, whole or not at all.

    Every file is staged before any is replaced, so a full disk or a
    folder that can't be written leaves every existing file as it was. The
    replacements are separate renames, not one transaction: if one fails,
    the ones before it are already new and the rest still old, and the
    ExportError says which.
    """
    paths = [path for path, _text in files]
    staged: list[Path] = []
    written: list[Path] = []
    current = paths[0]   # the file being staged or replaced when one fails
    try:
        for current, text in files:
            staged.append(_stage(current, text))
        for current, temp in zip(paths, staged):
            os.replace(temp, current)
            written.append(current)
    except OSError as exc:
        unchanged = [current] + [p for p in paths if p not in written and p != current]
        raise ExportError(written, current, unchanged, exc) from exc
    finally:
        for temp in staged[len(written):]:
            _discard(temp)
    return written


def write_file(path: Path, text: str) -> None:
    """Replace `path` with `text` whole, or leave it as it was."""
    _publish([(path, text)])


def _render(path: Path, segments: list[Segment], tracks: list[Track]) -> str:
    render = EXPORTERS.get(path.suffix.lower())
    if render is None:
        raise ValueError(f"Unsupported export format: {path.suffix.lower()}")
    return render(segments, tracks)


def export(path: Path, tracks: list[Track]) -> None:
    """Write the tracks' transcript as `path`'s format, merged as they are now."""
    write_file(path, _render(path, merge_tracks(tracks), tracks))


def export_all(base_path: Path, tracks: list[Track]) -> list[Path]:
    """Write every format, each file replaced whole or not at all.

    The tracks are merged once, here, at write time, so every format and the
    JSON's track list come from one snapshot. All five are rendered and
    staged before any file is replaced, so an error in rendering, or a disk
    that fills while staging, changes nothing. The set as a whole is not
    atomic: if one replacement fails, the ExportError raised names the
    formats already written and the ones still holding their old contents.
    """
    segments = merge_tracks(tracks)
    return _publish([(target, _render(target, segments, tracks))
                     for target in (base_path.with_suffix(ext) for ext in EXPORTERS)])
