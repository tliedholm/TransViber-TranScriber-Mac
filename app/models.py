"""Data model for tracks and transcript segments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TrackStatus(Enum):
    PENDING = "Pending"
    TRANSCRIBING = "Transcribing"
    DONE = "Done"
    ERROR = "Error"
    CANCELLED = "Cancelled"


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class RunConfig:
    """What actually produced a track's segments.

    Tracks can be transcribed across several runs with different settings —
    a rerun of the failures alone, a different model, a language change. The
    settings are therefore recorded per track instead of read off the combo
    boxes at export time, where they only describe the *current* selection.
    `backend` is what really ran (see transcriber.Backend), which is not
    always what `device` asked for.
    """

    model: str = ""
    language: str = ""  # requested; "" means auto-detect
    device: str = ""  # requested
    backend: str = ""  # actually used
    hints: str = ""

    def differs_from(self, other: "RunConfig") -> list[str]:
        """Names of the settings that would change the transcript."""
        return [
            label
            for label, mine, theirs in (
                ("model", self.model, other.model),
                ("language", self.language, other.language),
                ("vocabulary hints", self.hints, other.hints),
            )
            if mine != theirs
        ]

    def summary(self) -> str:
        parts = [self.model or "?"]
        if self.backend:
            parts.append(self.backend)
        parts.append(self.language or "auto")
        return " · ".join(parts)


@dataclass
class Track:
    path: Path
    speaker: str
    duration: float = 0.0
    status: TrackStatus = TrackStatus.PENDING
    language: str = ""
    error: str = ""
    segments: list[Segment] = field(default_factory=list)
    # Set when this track is a slice of a multitrack file: an audio stream
    # within the container, and optionally a single channel of that stream.
    stream_index: int | None = None
    channel: int | None = None
    # How this track's segments were produced (None until it finishes once).
    config: RunConfig | None = None
    # Why probing the container failed, if it did. Such a track is a guess —
    # one whole-file stream — and usually fails again at transcription time.
    probe_error: str = ""
    # The last successful result, kept while a rerun is in flight so a failed
    # or cancelled replacement does not destroy hours of finished work.
    previous: "TrackResult | None" = None

    @property
    def is_subtrack(self) -> bool:
        return self.stream_index is not None or self.channel is not None

    def stash_result(self) -> None:
        """Set the current DONE result aside before a rerun overwrites it."""
        if self.status == TrackStatus.DONE and self.segments:
            self.previous = TrackResult(list(self.segments), self.language, self.config)

    def restore_result(self) -> bool:
        """Put a stashed result back after a failed or cancelled rerun."""
        if self.previous is None:
            return False
        self.segments = list(self.previous.segments)
        self.language = self.previous.language
        self.config = self.previous.config
        self.status = TrackStatus.DONE
        self.previous = None
        return True

    def commit_result(self) -> None:
        """Drop the stash: the replacement run succeeded."""
        self.previous = None

    @property
    def source_label(self) -> str:
        parts = []
        if self.stream_index is not None:
            parts.append(f"track {self.stream_index + 1}")
        if self.channel is not None:
            parts.append(f"ch {self.channel + 1}")
        return f"{self.path.name} [{', '.join(parts)}]" if parts else self.path.name


@dataclass
class TrackResult:
    """A finished transcription of one track, kept aside during a rerun."""

    segments: list[Segment]
    language: str
    config: RunConfig | None


AUDIO_EXTENSIONS = {
    ".flac", ".m4a", ".mp4", ".aac", ".mp3", ".wav", ".ogg",
    ".opus", ".wma", ".webm", ".mka", ".mkv", ".mov", ".aif", ".aiff",
}


def default_speaker_name(path: Path) -> str:
    """Derive a speaker name from a filename: 'alice_track-01.flac' -> 'Alice Track 01'."""
    name = path.stem.replace("_", " ").replace("-", " ").strip()
    return " ".join(w.capitalize() if w.islower() else w for w in name.split()) or path.stem
