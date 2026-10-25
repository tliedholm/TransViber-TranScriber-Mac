"""Merge per-track segments into a single chronological conversation."""

from __future__ import annotations

from models import Segment, Track


def merge_tracks(tracks: list[Track]) -> list[Segment]:
    """All segments from all usable tracks, on the session clock, ordered by start time.

    Segment times inside a track are relative to its own file; each track's
    offset is added here, to copies, so the tracks keep their raw times and
    the offset can never be applied twice — however often this runs or the
    offset changes. The speaker is the track's name now, stamped here the
    same way, so a rename never has to rewrite segments. What this returns is
    a view: editing it changes no track. Python's sort is stable, so exact
    ties keep track order. A failed track is left out (Track.in_transcript).
    """
    segments = [
        Segment(track.speaker, seg.start + track.offset, seg.end + track.offset, seg.text)
        for track in tracks
        if track.in_transcript
        for seg in track.segments
    ]
    segments.sort(key=lambda s: (s.start, s.end))
    return segments


def discarded_segment_count(tracks: list[Track]) -> int:
    """How many segments merge_tracks is holding back from failed tracks."""
    return sum(len(t.segments) for t in tracks if not t.in_transcript)


def coalesce(segments: list[Segment]) -> list[Segment]:
    """Join consecutive segments from the same speaker into one block.

    Used for readable (Markdown/plain text) output. SRT/VTT keep the
    original fine-grained segments for accurate subtitle timing.
    """
    blocks: list[Segment] = []
    texts: list[list[str]] = []
    for seg in segments:
        if blocks and blocks[-1].speaker == seg.speaker:
            blocks[-1].end = max(blocks[-1].end, seg.end)
            texts[-1].append(seg.text.strip())
        else:
            blocks.append(Segment(seg.speaker, seg.start, seg.end, ""))
            texts.append([seg.text.strip()])
    # Joined once per block: adding to a block's text segment by segment
    # copied the whole block each time, which the live view paid on every
    # refresh of a long monologue.
    for block, parts in zip(blocks, texts):
        block.text = " ".join(part for part in parts if part)
    return blocks
