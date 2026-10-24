"""Merging, coalescing and the five exporters."""

import json
from pathlib import Path

import pytest

import exporters
from merger import coalesce, discarded_segment_count, merge_tracks
from models import RunConfig, Segment, Track, TrackStatus


def track(speaker, times, status=TrackStatus.DONE, **kwargs):
    return Track(
        path=Path(f"/tmp/{speaker}.wav"),
        speaker=speaker,
        status=status,
        segments=[Segment(speaker, s, e, t) for s, e, t in times],
        **kwargs,
    )


def test_merge_is_chronological_across_tracks():
    a = track("Alice", [(0.0, 1.0, "one"), (4.0, 5.0, "three")])
    b = track("Bob", [(2.0, 3.0, "two")])
    assert [s.text for s in merge_tracks([a, b])] == ["one", "two", "three"]


def test_failed_tracks_are_left_out_of_the_merge():
    good = track("Alice", [(0.0, 1.0, "kept")])
    bad = track("Bob", [(0.5, 1.5, "partial")], status=TrackStatus.ERROR)
    assert [s.text for s in merge_tracks([good, bad])] == ["kept"]
    assert discarded_segment_count([good, bad]) == 1
    # The segments are still there when explicitly asked for.
    assert len(merge_tracks([good, bad], include_failed=True)) == 2


def test_cancelled_tracks_keep_their_partial_output():
    # A cancelled track is the user's own choice to stop; what finished stays.
    done = track("Alice", [(0.0, 1.0, "a")])
    stopped = track("Bob", [(2.0, 3.0, "b")], status=TrackStatus.CANCELLED)
    assert len(merge_tracks([done, stopped])) == 2


def test_coalesce_joins_consecutive_same_speaker():
    segments = [Segment("Alice", 0, 1, "one"), Segment("Alice", 1, 2, "two"),
                Segment("Bob", 2, 3, "three")]
    blocks = coalesce(segments)
    assert [(b.speaker, b.text) for b in blocks] == [("Alice", "one two"), ("Bob", "three")]
    assert blocks[0].end == 2


def test_coalesce_does_not_mutate_its_input():
    segments = [Segment("Alice", 0, 1, "one"), Segment("Alice", 1, 2, "two")]
    coalesce(segments)
    assert segments[0].text == "one"


@pytest.mark.parametrize("name", [
    "*Nyx*", "_Bones_", "Al[ice]", "a`b`c", "R&B <tag>", "He said: #1", "a|b",
])
def test_markdown_escapes_speaker_names(name):
    out = exporters.to_markdown([Segment(name, 0, 1, "hi")])
    body = out.splitlines()[2]
    # The only unescaped markup left is the bold wrapper the exporter adds.
    assert body.startswith("**") and body[2:].startswith(exporters._md(name))
    assert exporters._md(name) != name  # something was actually escaped


def test_vtt_escapes_the_cue_tag():
    out = exporters.to_vtt([Segment("R&B <Ana>", 0, 1, "a > b & c")])
    assert "<v R&amp;B &lt;Ana&gt;>" in out
    assert "a &gt; b &amp; c" in out
    # A raw '>' inside the name would have closed the tag early.
    assert "<v R&B <Ana>>" not in out


def test_srt_and_vtt_timestamps():
    segment = Segment("Alice", 3661.5, 3662.25, "late")
    assert "01:01:01,500 --> 01:01:02,250" in exporters.to_srt([segment])
    assert "01:01:01.500 --> 01:01:02.250" in exporters.to_vtt([segment])


def test_json_records_the_config_of_each_track():
    turbo = RunConfig(model="large-v3-turbo", language="en", device="mlx",
                      backend="mlx-whisper on Metal GPU")
    qwen = RunConfig(model="qwen3-asr-1.7b", language="", device="mlx",
                     backend="mlx-qwen3-asr on Metal GPU")
    a = track("Alice", [(0, 1, "a")], config=turbo)
    b = track("Bob", [(1, 2, "b")], config=qwen)
    data = json.loads(exporters.to_json(merge_tracks([a, b]), [a, b]))
    assert data["models"] == ["large-v3-turbo", "qwen3-asr-1.7b"]
    assert data["tracks"][0]["transcription"]["backend"] == "mlx-whisper on Metal GPU"
    assert data["tracks"][1]["transcription"]["model"] == "qwen3-asr-1.7b"


def test_json_separates_failed_tracks():
    good = track("Alice", [(0, 1, "a")])
    bad = track("Bob", [(1, 2, "partial")], status=TrackStatus.ERROR)
    data = json.loads(exporters.to_json(merge_tracks([good, bad]), [good, bad]))
    assert [t["speaker"] for t in data["tracks"]] == ["Alice"]
    assert [t["speaker"] for t in data["excluded_tracks"]] == ["Bob"]
    assert [s["text"] for s in data["segments"]] == ["a"]


def test_json_says_what_decoding_ran_into():
    noted = track("Alice", [(0, 1, "a")], note="Its timestamps jump back at 0:05.")
    plain = track("Bob", [(1, 2, "b")])
    data = json.loads(exporters.to_json(merge_tracks([noted, plain]), [noted, plain]))
    assert data["tracks"][0]["decode_note"] == "Its timestamps jump back at 0:05."
    assert "decode_note" not in data["tracks"][1]


def test_export_all_writes_every_format(tmp_path):
    a = track("Alice", [(0, 1, "hello")])
    written = exporters.export_all(tmp_path / "out.md", merge_tracks([a]), [a])
    assert sorted(p.suffix for p in written) == [".json", ".md", ".srt", ".txt", ".vtt"]
    for path in written:
        assert path.read_text(encoding="utf-8").strip()


def test_export_rejects_an_unknown_extension(tmp_path):
    with pytest.raises(ValueError):
        exporters.export(tmp_path / "out.doc", [], [])
