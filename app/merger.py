"""Merge per-track segments into a single chronological conversation."""

from __future__ import annotations

from models import Segment, Track, TrackStatus


def merge_tracks(tracks: list[Track], include_failed: bool = False) -> list[Segment]:
    """All segments from all usable tracks, ordered by start time.

    An engine can emit segments and *then* raise: those partial segments are
    a fragment of one speaker's audio, and letting them into the merged
    conversation makes an incomplete transcript look finished. They stay on
    the Track (nothing is thrown away silently, and the UI reports how many)
    but are kept out of the transcript and every export unless asked for.
    """
    segments = [
        seg
        for track in tracks
        if include_failed or track.status != TrackStatus.ERROR
        for seg in track.segments
    ]
    segments.sort(key=lambda s: (s.start, s.end))
    return segments


def discarded_segment_count(tracks: list[Track]) -> int:
    """How many segments merge_tracks is holding back from failed tracks."""
    return sum(len(t.segments) for t in tracks if t.status == TrackStatus.ERROR)


def coalesce(segments: list[Segment]) -> list[Segment]:
    """Join consecutive segments from the same speaker into one block.

    Used for readable (Markdown/plain text) output. SRT/VTT keep the
    original fine-grained segments for accurate subtitle timing.
    """
    blocks: list[Segment] = []
    for seg in segments:
        if blocks and blocks[-1].speaker == seg.speaker:
            prev = blocks[-1]
            prev.text = (prev.text.rstrip() + " " + seg.text.strip()).strip()
            prev.end = max(prev.end, seg.end)
        else:
            blocks.append(Segment(seg.speaker, seg.start, seg.end, seg.text.strip()))
    return blocks
