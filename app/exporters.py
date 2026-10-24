"""Export merged transcripts to Markdown, plain text, SRT, VTT, and JSON."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from merger import coalesce
from models import Segment, Track, TrackStatus

# Characters that would otherwise turn a speaker's name into Markdown markup:
# a name like *Nyx* or a nickname written as _Bones_ silently became emphasis,
# and one containing [] or ` produced broken links and code spans.
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>])")


def _md(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", text)


def _vtt(text: str) -> str:
    """Escape a name for a WebVTT <v ...> cue tag."""
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


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
    exported = [t for t in tracks if t.status != TrackStatus.ERROR]
    data = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        # A list, because a transcript assembled over several runs genuinely
        # has more than one. Recording only the current selection made a
        # mixed transcript look like one model's work.
        "models": _models_used(exported),
        "tracks": [_track_entry(t) for t in exported],
        "excluded_tracks": [_track_entry(t) for t in tracks
                            if t.status == TrackStatus.ERROR],
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


EXPORTERS = {
    ".md": to_markdown,
    ".txt": to_text,
    ".srt": to_srt,
    ".vtt": to_vtt,
    ".json": to_json,
}


def export(path: Path, segments: list[Segment], tracks: list[Track]) -> None:
    ext = path.suffix.lower()
    if ext == ".json":
        content = to_json(segments, tracks)
    elif ext in EXPORTERS:
        content = EXPORTERS[ext](segments)
    else:
        raise ValueError(f"Unsupported export format: {ext}")
    path.write_text(content, encoding="utf-8")


def export_all(base_path: Path, segments: list[Segment], tracks: list[Track]) -> list[Path]:
    written = []
    for ext in EXPORTERS:
        target = base_path.with_suffix(ext)
        export(target, segments, tracks)
        written.append(target)
    return written
