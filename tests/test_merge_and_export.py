"""Merging, coalescing and the five exporters."""

import json
from pathlib import Path

import pytest

import exporters
from merger import coalesce, discarded_segment_count, merge_tracks
from models import OffsetSource, RunConfig, Segment, Track, TrackStatus


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


# A track's offset is when its file began on the session clock.

def test_an_offset_moves_every_segment_of_its_track():
    bob = track("Bob", [(10.0, 12.0, "Hello.")], offset=900.0)
    [seg] = merge_tracks([bob])
    assert (seg.start, seg.end) == (910.0, 912.0)


def test_ordering_across_tracks_follows_session_time():
    # Bob joined five minutes late: his first line, a minute into his file,
    # comes after Alice's line at 3:00 in hers.
    alice = track("Alice", [(180.0, 181.0, "Three minutes in.")])
    bob = track("Bob", [(60.0, 61.0, "Sorry I'm late.")], offset=300.0)
    assert [s.speaker for s in merge_tracks([bob, alice])] == ["Alice", "Bob"]


def test_the_offset_is_applied_once_and_tracks_keep_raw_times():
    bob = track("Bob", [(1.0, 2.0, "Hi.")], offset=5.0)
    first, second = merge_tracks([bob]), merge_tracks([bob])
    assert [(s.start, s.end) for s in first] == [(s.start, s.end) for s in second]
    assert (bob.segments[0].start, bob.segments[0].end) == (1.0, 2.0)


def test_editing_the_merge_changes_no_track():
    alice = track("Alice", [(0.0, 1.0, "Hi.")])
    merge_tracks([alice])[0].text = "changed"
    assert alice.segments[0].text == "Hi."


def test_zero_offsets_keep_the_old_order_including_ties_and_overlaps():
    a = track("A", [(1.0, 5.0, "a-long"), (6.0, 7.0, "a-later")])
    b = track("B", [(1.0, 5.0, "b-tie"), (2.0, 3.0, "b-overlap")])
    merged = merge_tracks([a, b])
    assert [s.text for s in merged] == ["a-long", "b-tie", "b-overlap", "a-later"]
    assert [(s.speaker, s.start, s.end) for s in merged] == [
        ("A", 1.0, 5.0), ("B", 1.0, 5.0), ("B", 2.0, 3.0), ("A", 6.0, 7.0)]


def test_offset_milliseconds_survive():
    [seg] = merge_tracks([track("Bob", [(1.1, 2.4, "Hi.")], offset=150.25)])
    assert seg.start == pytest.approx(151.35)
    assert seg.end == pytest.approx(152.65)


def test_offsets_reach_subtitle_times_through_the_merge():
    bob = track("Bob", [(1.1, 2.4, "Hi.")], offset=150.25)
    assert "00:02:31,350 --> 00:02:32,650" in exporters.to_srt(merge_tracks([bob]))
    assert "00:02:31.350 --> 00:02:32.650" in exporters.to_vtt(merge_tracks([bob]))


def test_json_records_each_tracks_offset_and_how_it_was_set():
    alice = track("Alice", [(0, 1, "a")])
    bob = track("Bob", [], offset=312.4567, offset_source=OffsetSource.MANUAL)
    data = json.loads(exporters.to_json(merge_tracks([alice, bob]), [alice, bob]))
    assert (data["tracks"][0]["offset"], data["tracks"][0]["offset_source"]) == (0.0, "default")
    assert (data["tracks"][1]["offset"], data["tracks"][1]["offset_source"]) == (312.457,
                                                                                 "manual")


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


def test_readable_timestamps_drop_a_zero_hour():
    assert exporters.hms(75) == "1:15"
    assert exporters.hms(3675) == "1:01:15"


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
