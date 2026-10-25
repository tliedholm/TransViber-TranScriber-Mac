"""Word-gap splitting and multitrack expansion.

Word-gap splitting is a critical invariant: Whisper merges utterances
separated by silence into one segment, which would sort a speaker's later
remark at the time of their earlier one and destroy the interleaving.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import audio
import transcriber
from transcriber import SPLIT_GAP_SECONDS, split_on_word_gaps, split_timed_words


@dataclass
class Word:
    start: float
    end: float
    word: str


@dataclass
class FwSegment:
    start: float
    end: float
    text: str
    words: list


def test_no_words_yields_the_segment_unchanged():
    seg = FwSegment(1.0, 2.0, "hello", [])
    pieces = split_on_word_gaps(seg, "Alice")
    assert len(pieces) == 1
    assert (pieces[0].speaker, pieces[0].start, pieces[0].end) == ("Alice", 1.0, 2.0)


def test_segment_spanning_silence_is_split():
    seg = FwSegment(0.0, 40.0, " one two three", [
        Word(0.0, 0.5, " one"), Word(0.6, 1.0, " two"),
        Word(30.0, 30.5, " three"),  # 29 s of silence before this
    ])
    pieces = split_on_word_gaps(seg, "Alice")
    assert len(pieces) == 2
    assert pieces[0].text == " one two" and pieces[0].end == 1.0
    assert pieces[1].text == " three" and pieces[1].start == 30.0


def test_normal_pauses_do_not_split():
    gap = SPLIT_GAP_SECONDS / 2
    seg = FwSegment(0.0, 3.0, " a b", [
        Word(0.0, 0.5, " a"), Word(0.5 + gap, 1.0 + gap, " b"),
    ])
    assert len(split_on_word_gaps(seg, "Alice")) == 1


def test_every_word_survives_a_split():
    seg = FwSegment(0.0, 100.0, "", [Word(i * 10.0, i * 10.0 + 0.5, f" w{i}")
                                     for i in range(6)])
    pieces = split_on_word_gaps(seg, "Alice")
    assert len(pieces) == 6
    assert "".join(p.text for p in pieces) == "".join(w.word for w in seg.words)


def test_pieces_are_chronological():
    seg = FwSegment(0.0, 60.0, "", [Word(0, 1, "a"), Word(20, 21, "b"), Word(50, 51, "c")])
    pieces = split_on_word_gaps(seg, "Alice")
    assert [p.start for p in pieces] == sorted(p.start for p in pieces)


def test_qwen_words_split_the_same_way_back_on_the_region_offset():
    # Qwen's words are bare (no leading space) and timed from their region.
    words = [(0.0, 0.4, "one"), (0.5, 0.9, "two"), (5.0, 5.4, "three")]
    pieces = split_timed_words(words, "Alice", sep=" ", offset=100.0)
    assert [(p.start, p.end, p.text) for p in pieces] == [
        (100.0, 100.9, "one two"), (105.0, 105.4, "three")]


def test_free_asr_models_also_drops_the_engines_own_caches(monkeypatch):
    # mlx-whisper's model and Qwen's forced aligner are cached by the
    # libraries themselves, where the app's own cache never sees them.
    whisper = SimpleNamespace(model=object(), model_path="mlx-community/whisper-large-v3-mlx")
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe",
                        SimpleNamespace(ModelHolder=whisper))
    aligner = {"Qwen/Qwen3-ForcedAligner-0.6B": object()}
    qwen = SimpleNamespace(clear=aligner.clear)
    monkeypatch.setitem(sys.modules, "mlx_qwen3_asr.load_models",
                        SimpleNamespace(_ModelHolder=qwen))
    transcriber.free_asr_models()
    assert (whisper.model, whisper.model_path) == (None, None)
    assert aligner == {}


# ---- multitrack expansion --------------------------------------------------

def test_one_stereo_stream_stays_one_speaker(monkeypatch):
    monkeypatch.setattr(audio, "probe_streams",
                        lambda p: [{"channels": 2, "duration": 12.0}])
    tracks = audio.tracks_for_file(Path("/tmp/alice_track-01.flac"))
    assert len(tracks) == 1
    assert tracks[0].speaker == "Alice Track 01"
    assert tracks[0].duration == 12.0
    assert tracks[0].channel is None


def test_split_stereo_makes_two_speakers(monkeypatch):
    monkeypatch.setattr(audio, "probe_streams",
                        lambda p: [{"channels": 2, "duration": 12.0}])
    tracks = audio.tracks_for_file(Path("/tmp/session.wav"), split_stereo=True)
    assert [t.speaker for t in tracks] == ["Session L", "Session R"]
    assert [t.channel for t in tracks] == [0, 1]


def test_multiple_streams_become_one_track_each(monkeypatch):
    monkeypatch.setattr(audio, "probe_streams", lambda p: [
        {"channels": 1, "duration": 60.0}, {"channels": 1, "duration": 60.0}])
    tracks = audio.tracks_for_file(Path("/tmp/craig.mka"))
    assert [t.speaker for t in tracks] == ["Craig T1", "Craig T2"]
    assert [t.stream_index for t in tracks] == [0, 1]


def test_more_than_two_channels_split_automatically(monkeypatch):
    monkeypatch.setattr(audio, "probe_streams",
                        lambda p: [{"channels": 4, "duration": 5.0}])
    tracks = audio.tracks_for_file(Path("/tmp/quad.wav"))
    assert [t.channel for t in tracks] == [0, 1, 2, 3]
    assert tracks[0].speaker.endswith("Ch1")


def test_a_probe_failure_is_recorded_not_swallowed(monkeypatch):
    def boom(path):
        raise ValueError("Invalid data found when processing input")

    monkeypatch.setattr(audio, "probe_streams", boom)
    tracks = audio.tracks_for_file(Path("/tmp/broken.m4a"))
    # Still added — the user asked for it — but the reason travels with it,
    # instead of surfacing much later as a transcription failure.
    assert len(tracks) == 1
    assert "Invalid data" in tracks[0].probe_error


def test_a_file_with_no_audio_stream_is_flagged(monkeypatch):
    monkeypatch.setattr(audio, "probe_streams", lambda p: [])
    tracks = audio.tracks_for_file(Path("/tmp/video-only.mkv"))
    assert tracks[0].probe_error


def test_a_readable_file_has_no_probe_error(monkeypatch):
    monkeypatch.setattr(audio, "probe_streams",
                        lambda p: [{"channels": 1, "duration": 3.0}])
    assert audio.tracks_for_file(Path("/tmp/ok.wav"))[0].probe_error == ""


def test_vocabulary_hints_reach_every_engine_that_supports_them():
    # faster-whisper uses hotwords=, MLX initial_prompt=, Qwen context=;
    # onnx models have no support and the field is hidden for them.
    source = Path(transcriber.__file__).read_text(encoding="utf-8")
    assert "hotwords=self.hotwords or None" in source
    assert "initial_prompt=self.hotwords or None" in source
    assert "context=self.hotwords" in source
