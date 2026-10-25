"""Merging, coalescing and the five exporters."""

import json
import os
import stat
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
    # The segments are still on the track, never thrown away silently.
    assert [s.text for s in bad.segments] == ["partial"]


def test_the_merge_names_each_segment_after_its_track_as_it_is_now():
    # A rename only changes the track: the merge stamps the current name.
    alice = track("Alice", [(0.0, 1.0, "hi")])
    alice.speaker = "Zephyrine"
    assert [s.speaker for s in merge_tracks([alice])] == ["Zephyrine"]


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


def test_coalesce_drops_blank_text_and_edge_spaces():
    segments = [Segment("Alice", 0, 1, "  "), Segment("Alice", 1, 2, " one "),
                Segment("Alice", 2, 3, ""), Segment("Alice", 3, 4, "two  ")]
    assert [b.text for b in coalesce(segments)] == ["one two"]


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
    written = exporters.export_all(tmp_path / "out.md", [a])
    assert sorted(p.suffix for p in written) == [".json", ".md", ".srt", ".txt", ".vtt"]
    for path in written:
        assert path.read_text(encoding="utf-8").strip()


def test_export_rejects_an_unknown_extension(tmp_path):
    with pytest.raises(ValueError):
        exporters.export(tmp_path / "out.doc", [])


# ---------- Each file replaced whole, and a failure says how far it got ----------

SEGMENTS = [Segment("Alice", 0.0, 1.0, "hello")]
TRACKS = [track("Alice", [(0.0, 1.0, "hello")])]


def failing_on(call, real, error=None):
    """A stand-in for `real` that raises `error` on its `call`th use (from 1)
    and otherwise does what the real function does."""
    error = error or PermissionError(13, "Permission denied")
    count = {"n": 0}

    def stand_in(*args, **kwargs):
        count["n"] += 1
        if count["n"] == call:
            raise error
        return real(*args, **kwargs)
    return stand_in


@pytest.fixture
def old_exports(tmp_path):
    """A folder holding an earlier export of every format."""
    base = tmp_path / "session.md"
    targets = [base.with_suffix(ext) for ext in exporters.EXPORTERS]
    for path in targets:
        path.write_text("old " + path.suffix, encoding="utf-8")
    return base, targets


def contents(paths):
    return {p.name: p.read_text(encoding="utf-8") for p in paths}


def untouched(targets):
    return contents(targets) == {p.name: "old " + p.suffix for p in targets}


def no_staged_files_left(targets):
    return sorted(targets[0].parent.iterdir()) == sorted(targets)


def test_a_full_disk_while_staging_changes_nothing(old_exports, monkeypatch):
    base, targets = old_exports
    full = OSError(28, "No space left on device")
    monkeypatch.setattr(exporters.os, "fsync", failing_on(3, os.fsync, full))
    with pytest.raises(exporters.ExportError) as caught:
        exporters.export_all(base, TRACKS)
    assert untouched(targets)
    assert no_staged_files_left(targets)
    error = caught.value
    assert error.written == []
    assert error.failed == targets[2]
    assert set(error.unchanged) == set(targets)
    assert "No space left on device" in str(error)
    assert "Nothing was exported" in str(error)


def test_a_failed_first_replacement_reports_nothing_written(old_exports, monkeypatch):
    base, targets = old_exports
    monkeypatch.setattr(exporters.os, "replace", failing_on(1, os.replace))
    with pytest.raises(exporters.ExportError) as caught:
        exporters.export_all(base, TRACKS)
    assert untouched(targets)
    assert no_staged_files_left(targets)
    assert caught.value.written == []
    assert caught.value.unchanged == targets


def test_a_failed_second_replacement_names_what_is_new_and_what_is_old(
        old_exports, monkeypatch):
    base, targets = old_exports
    monkeypatch.setattr(exporters.os, "replace", failing_on(2, os.replace))
    with pytest.raises(exporters.ExportError) as caught:
        exporters.export_all(base, TRACKS)
    error = caught.value
    md, txt = targets[:2]
    assert error.written == [md]
    assert error.failed == txt
    assert error.unchanged == targets[1:]
    assert isinstance(error, OSError)  # the window's `except OSError` still catches it
    assert isinstance(error.__cause__, PermissionError)
    # The one that was replaced is whole; the rest are exactly as they were.
    now = contents(targets)
    assert now.pop(md.name) == exporters.to_markdown(SEGMENTS)
    assert now == {p.name: "old " + p.suffix for p in targets[1:]}
    assert no_staged_files_left(targets)
    message = str(error)
    assert ("Exported session.md, but could not write session.txt: Permission denied"
            in message)
    assert "Not updated: session.txt, session.srt, session.vtt, session.json" in message


def test_a_cleanup_failure_never_hides_the_real_error(old_exports, monkeypatch):
    base, targets = old_exports
    monkeypatch.setattr(exporters.os, "replace", failing_on(1, os.replace))

    def unlink(self, *args, **kwargs):
        raise OSError("busy")
    monkeypatch.setattr(exporters.Path, "unlink", unlink)
    with pytest.raises(exporters.ExportError) as caught:
        exporters.export_all(base, TRACKS)
    assert "Permission denied" in str(caught.value)
    assert "busy" not in str(caught.value)
    assert untouched(targets)


def test_a_failed_single_format_export_keeps_the_old_file(old_exports, monkeypatch):
    base, targets = old_exports
    srt = base.with_suffix(".srt")
    monkeypatch.setattr(exporters.os, "replace", failing_on(1, os.replace))
    with pytest.raises(exporters.ExportError) as caught:
        exporters.export(srt, TRACKS)
    assert srt.read_text(encoding="utf-8") == "old .srt"
    assert no_staged_files_left(targets)
    assert (str(caught.value).splitlines()[0]
            == "Could not write session.srt: Permission denied")


def test_a_successful_export_replaces_every_file_whole(old_exports):
    base, targets = old_exports
    assert exporters.export_all(base, TRACKS) == targets
    assert contents(targets)["session.vtt"] == exporters.to_vtt(SEGMENTS)
    assert no_staged_files_left(targets)


def test_a_replaced_file_keeps_its_permissions(tmp_path):
    # Staged through a temp file, but not with mkstemp's owner-only 0o600.
    target = tmp_path / "notes.md"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)
    exporters.write_file(target, "new")
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert target.read_text(encoding="utf-8") == "new"


def test_a_new_file_gets_the_permissions_a_plain_write_would(tmp_path):
    plain = tmp_path / "plain.txt"
    plain.write_text("x", encoding="utf-8")
    staged = tmp_path / "staged.txt"
    exporters.write_file(staged, "x")
    assert stat.S_IMODE(staged.stat().st_mode) == stat.S_IMODE(plain.stat().st_mode)
