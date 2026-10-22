"""Track state transitions around reruns.

The rule these protect: a finished transcript is only replaced by a finished
transcript. A cancelled, failed or aborted rerun leaves the old one intact.
"""

from pathlib import Path

from models import RunConfig, Segment, Track, TrackStatus


def finished(text="original"):
    return Track(
        path=Path("/tmp/a.wav"), speaker="Alice", status=TrackStatus.DONE,
        language="en", segments=[Segment("Alice", 0, 1, text)],
        config=RunConfig(model="large-v3", language="en"),
    )


def test_stash_then_restore_returns_the_original():
    track = finished()
    track.stash_result()
    track.status = TrackStatus.TRANSCRIBING
    track.segments.clear()  # what _on_track_started does
    track.segments.append(Segment("Alice", 0, 1, "half a rerun"))

    assert track.restore_result() is True
    assert [s.text for s in track.segments] == ["original"]
    assert track.status is TrackStatus.DONE
    assert track.language == "en"
    assert track.previous is None


def test_commit_drops_the_stash():
    track = finished()
    track.stash_result()
    assert track.previous is not None
    track.commit_result()
    assert track.previous is None
    assert track.restore_result() is False


def test_nothing_to_stash_when_the_track_never_finished():
    track = Track(path=Path("/tmp/a.wav"), speaker="Alice")
    track.stash_result()
    assert track.previous is None
    assert track.restore_result() is False


def test_stash_survives_clearing_the_live_segment_list():
    track = finished()
    track.stash_result()
    track.segments.clear()
    assert len(track.previous.segments) == 1


def test_config_differences_are_named():
    made_with = RunConfig(model="large-v3", language="en", hints="Zephyrine")
    same = RunConfig(model="large-v3", language="en", hints="Zephyrine",
                     device="cpu", backend="faster-whisper on CPU")
    # Device and backend do not change what the transcript says.
    assert same.differs_from(made_with) == []
    assert RunConfig(model="qwen3-asr-1.7b", language="en",
                     hints="Zephyrine").differs_from(made_with) == ["model"]
    changed = RunConfig(model="large-v3", language="sv", hints="")
    assert set(changed.differs_from(made_with)) == {"language", "vocabulary hints"}


def test_subtrack_labelling():
    track = Track(path=Path("/tmp/craig.flac"), speaker="Alice",
                  stream_index=2, channel=0)
    assert track.is_subtrack
    assert track.source_label == "craig.flac [track 3, ch 1]"
    assert not Track(path=Path("/tmp/a.wav"), speaker="A").is_subtrack
