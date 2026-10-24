"""Decoding on the file's own clock: where each stream's audio goes.

The files are written here with PyAV, the way recorders write them: Matroska
keeps timestamps to the millisecond, a stream can begin after the others, a
paused recorder leaves a gap in the timestamps, and two recordings joined end
to end make them jump back. What used to happen to all of them is the same:
the audio was laid end to end from 0:00, and everything after a gap or in a
late stream moved earlier. Merging places each file on the session clock
with one offset per track, and no single offset can put that right. Align
reads the same way (as power, through iter_stream_power).
"""

import sys
import wave
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import audio
import transcriber
from audio import (
    SAMPLE_RATE,
    TIMING_TOLERANCE,
    AudioFileError,
    Timing,
    decode_track_audio,
    iter_stream_power,
)
from models import Track
from transcriber import Backend, TranscribeWorker

RATE = 48000


def noise(seconds: float, seed: int, channels: int = 1, rate: int = RATE):
    """Steady noise, about -16 dBFS: loud enough that silence beside it is plain."""
    shape = (int(seconds * rate), channels)
    return (0.15 * np.random.default_rng(seed).standard_normal(shape)).astype(np.float32)


def write_mkv(path: Path, streams, rate: int = RATE, frame: int = 1024,
              fmt: str = "matroska", codec: str = "pcm_s16le") -> Path:
    """One stream per entry of `streams`, each a list of (start s, samples) blocks.

    `samples` has a row per sample and a column per channel. Every block is
    cut into `frame`-sample frames stamped from where the block starts, so a
    gap between blocks is a gap in the timestamps, and a block that starts
    late is a stream that began late.
    """
    import av

    frames = []
    with av.open(str(path), "w", format=fmt) as out:
        for blocks in streams:
            channels = blocks[0][1].shape[1]
            stream = out.add_stream(codec, rate=rate,
                                    layout="mono" if channels == 1 else "stereo")
            for start, samples in blocks:
                for k in range(0, len(samples), frame):
                    chunk = np.ascontiguousarray(samples[k:k + frame], np.float32).reshape(1, -1)
                    f = av.AudioFrame.from_ndarray(chunk, format="flt",
                                                   layout=stream.layout.name)
                    f.sample_rate = rate
                    f.time_base = Fraction(1, rate)
                    f.pts = round(start * rate) + k
                    frames.append((f.pts, stream.index, stream, f))
        frames.sort(key=lambda item: item[:2])
        for _pts, _index, stream, f in frames:
            for packet in stream.encode(f):
                out.mux(packet)
        for stream in out.streams.audio:
            for packet in stream.encode(None):
                out.mux(packet)
    return path


def joined_ts(folder: Path, first_seconds: float, second_seconds: float) -> Path:
    """Two MPEG-TS recordings joined end to end: the second's clock starts at 0 again."""
    parts = []
    for name, seconds, seed in (("one.ts", first_seconds, 11), ("two.ts", second_seconds, 12)):
        parts.append(write_mkv(folder / name, [[(0, noise(seconds, seed))]],
                               frame=1152, fmt="mpegts", codec="mp2"))
    joined = folder / "joined.ts"
    joined.write_bytes(b"".join(p.read_bytes() for p in parts))
    return joined


def loud_seconds(samples, rate: int = SAMPLE_RATE) -> list[int]:
    """The whole seconds of `samples` that hold sound."""
    seconds = len(samples) // rate
    power = np.square(np.asarray(samples[:seconds * rate], dtype=np.float64))
    return [int(s) for s in np.flatnonzero(power.reshape(seconds, rate).mean(axis=1) > 1e-4)]


def power_of(path: Path, stream_index=None, rate: int = 8000, timing=None):
    """What Align hears: the stream's power, summed over its channels."""
    return np.concatenate(list(iter_stream_power(path, stream_index, rate, timing)))


# ---- following the timestamps -----------------------------------------------

def test_a_gap_in_the_timestamps_stays_a_gap(tmp_path):
    # Frames for seconds 0, 1, 4 and 5: a recorder paused for two seconds.
    path = write_mkv(tmp_path / "gap.mkv", [[(0, noise(2, 1)), (4, noise(2, 2))]], frame=RATE)
    timing = Timing()
    decoded = decode_track_audio(path, None, None, timing)
    assert len(decoded) == 6 * SAMPLE_RATE
    assert loud_seconds(decoded) == [0, 1, 4, 5]
    assert timing == Timing()           # a gap is no jump back
    power = power_of(path)
    assert len(power) == 6 * 8000
    assert loud_seconds(np.sqrt(power), 8000) == [0, 1, 4, 5]


def test_a_stream_that_began_late_keeps_its_leading_silence(tmp_path):
    # Stream 2 began a minute after stream 1, in the same file: read from its
    # own first packet, its speaker's words all landed a minute early.
    path = write_mkv(tmp_path / "late.mkv", [[(0, noise(90, 3))], [(60, noise(30, 4))]])
    first = decode_track_audio(path, 0, None)
    second = decode_track_audio(path, 1, None)
    assert len(first) == len(second)
    assert loud_seconds(first) == list(range(90))
    assert loud_seconds(second) == list(range(60, 90))
    assert loud_seconds(np.sqrt(power_of(path, 1)), 8000) == list(range(60, 90))


def test_a_file_whose_clock_starts_at_ten_seconds_starts_at_zero(tmp_path):
    # The container's own start is 0:00: no ten seconds of silence first.
    path = write_mkv(tmp_path / "ten.mkv", [[(10, noise(5, 5))]])
    decoded = decode_track_audio(path, None, None)
    assert len(decoded) == 5 * SAMPLE_RATE
    assert loud_seconds(decoded) == [0, 1, 2, 3, 4]


def test_one_channel_keeps_the_same_time_as_the_whole_stream(tmp_path):
    stereo = noise(2, 6, channels=2)
    stereo[:, 1] = 0                    # only the left channel talks
    later = noise(2, 7, channels=2)
    later[:, 0] = 0                     # then only the right
    path = write_mkv(tmp_path / "pair.mkv", [[(0, stereo), (5, later)]])
    whole = decode_track_audio(path, None, None)
    left = decode_track_audio(path, None, 0)
    right = decode_track_audio(path, None, 1)
    assert len(whole) == len(left) == len(right)
    assert loud_seconds(whole) == [0, 1, 5, 6]
    assert loud_seconds(left) == [0, 1]
    assert loud_seconds(right) == [5, 6]


def test_frame_size_and_rate_move_nothing_beyond_rounding(tmp_path):
    # The same session cut into frames of three sizes and read at two rates
    # (Align's power at 8 kHz, the engines' audio at 16 kHz): where the sound
    # starts again after the gap agrees to within the timing tolerance,
    # whatever the cut.
    found = []
    for frame in (480, 1024, 4800):
        path = write_mkv(tmp_path / f"cut{frame}.mkv",
                         [[(0, noise(3, 8)), (7.3, noise(2, 9))]], frame=frame)
        for rate in (8000, SAMPLE_RATE):
            power = (power_of(path, rate=rate) if rate == 8000
                     else np.square(decode_track_audio(path, None, None)))
            back = int(np.flatnonzero(power[3 * rate:] > 1e-4)[0]) + 3 * rate
            found.append(back / rate)
            assert len(power) / rate == pytest.approx(9.3, abs=1 / rate)
    assert max(found) - min(found) <= TIMING_TOLERANCE
    assert np.mean(found) == pytest.approx(7.3, abs=TIMING_TOLERANCE)


def test_an_ordinary_wav_decodes_exactly_as_before(tmp_path):
    # Zero-based, no gaps: sample for sample what laying the resampled audio
    # end to end gave, at 16 kHz mono and as power at 8 kHz.
    import av
    from av.audio.resampler import AudioResampler

    path = tmp_path / "plain.wav"
    samples = noise(7.5, 10, channels=2, rate=44100)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(44100)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())

    def laid_end_to_end(**kwargs):
        chunks = []
        with av.open(str(path)) as container:
            resampler = AudioResampler(format="flt", **kwargs)
            for frame in container.decode(audio=0):
                chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
            chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
        return np.concatenate(chunks)

    np.testing.assert_array_equal(decode_track_audio(path, None, None),
                                  laid_end_to_end(layout="mono", rate=SAMPLE_RATE))
    both = laid_end_to_end(layout="stereo", rate=8000).astype(np.float64).reshape(-1, 2)
    np.testing.assert_array_equal(power_of(path), np.sum(both * both, axis=1))


def test_joined_recordings_keep_every_sample_and_say_where(tmp_path):
    # The second recording's clock starts again from zero. All of both is
    # kept, the second after the first, and the jump is reported.
    timing = Timing()
    decoded = decode_track_audio(joined_ts(tmp_path, 5, 3), None, None, timing)
    assert len(decoded) / SAMPLE_RATE == pytest.approx(8.0, abs=0.05)
    assert loud_seconds(decoded) == list(range(8))
    ((at, back),) = timing.jumps
    assert at == pytest.approx(5.0, abs=0.05)
    assert back == pytest.approx(5.0, abs=0.05)
    assert "jump back at 0:05" in timing.summary()


def test_a_long_gap_is_handed_out_a_little_at_a_time(tmp_path):
    # An hour between two bursts: never an hour of samples in one piece.
    path = write_mkv(tmp_path / "hour.mkv", [[(0, noise(1, 13)), (3601, noise(1, 14))]])
    chunks = list(iter_stream_power(path, None, 8000))
    assert max(len(c) for c in chunks) <= audio.SILENCE_CHUNK_SECONDS * 8000
    assert sum(len(c) for c in chunks) == 3602 * 8000


def test_a_timestamp_past_the_limit_is_an_error_not_an_allocation(tmp_path, monkeypatch):
    path = write_mkv(tmp_path / "far.mkv", [[(0, noise(1, 15)), (120, noise(1, 16))]])
    monkeypatch.setattr(audio, "MAX_TIMELINE_SECONDS", 60)
    zeros = Mock(wraps=np.zeros)
    monkeypatch.setattr(np, "zeros", zeros)
    with pytest.raises(AudioFileError, match="timing looks damaged") as caught:
        decode_track_audio(path, None, None)
    with pytest.raises(AudioFileError, match="timing looks damaged"):
        list(iter_stream_power(path, None, 8000))
    assert "far.mkv" in str(caught.value)
    assert all(np.prod(c.args[0]) <= 2 * 60 * SAMPLE_RATE for c in zeros.call_args_list)


def test_a_missing_stream_is_an_audio_file_error(tmp_path):
    path = write_mkv(tmp_path / "one.mkv", [[(0, noise(1, 17))]])
    with pytest.raises(AudioFileError, match="has no audio stream 4"):
        decode_track_audio(path, 3, None)


# ---- _placed's rules, on bare positions in samples at 1 kHz -----------------

def _blocks(*runs):
    """(position, samples) blocks of `length` ones, as _timed_blocks gives them."""
    return [(position, np.ones((length, 1), np.float32)) for position, length in runs]


def place(*runs):
    timing = Timing()
    placed = list(audio._placed(_blocks(*runs), 1000, Path("f.wav"), timing))
    return [(silence, len(samples)) for silence, samples in placed], timing


def test_within_the_tolerance_it_follows_on():
    tolerance = round(TIMING_TOLERANCE * 1000)
    placed, timing = place((0, 100), (100 + tolerance, 50), (150 - tolerance, 50))
    assert placed == [(0, 100), (0, 50), (0, 50)]
    assert timing == Timing()


def test_later_is_a_gap_filled_with_silence():
    placed, timing = place((0, 100), (500, 50))
    assert placed == [(0, 100), (400, 50)]
    assert timing.jumps == []


def test_earlier_plays_on_and_is_noted():
    # The clock went back 0.8 s at 1.0 s: read as restarted there, so the
    # 0.6 s gap after it is still a 0.6 s gap.
    placed, timing = place((0, 1000), (200, 500), (1300, 100))
    assert placed == [(0, 1000), (0, 500), (600, 100)]
    assert timing.jumps == [(1.0, 0.8)]


def test_before_zero_at_the_start_plays_from_zero_and_is_noted():
    # Normalised against the file's start already: what is left before it is
    # kept, never dropped, and said.
    placed, timing = place((-300, 1000))
    assert placed == [(0, 1000)]
    assert timing.jumps == [(0.0, 0.3)]


def test_a_block_with_no_position_follows_on():
    placed, timing = place((None, 100), (None, 100), (900, 10))
    assert placed == [(0, 100), (0, 100), (700, 10)]
    assert timing == Timing()


def test_an_absurd_position_stops_before_making_its_silence():
    limit = audio.MAX_TIMELINE_SECONDS * 1000
    at_limit, _timing = place((0, 10), (limit, 10))
    assert at_limit == [(0, 10), (limit - 10, 10)]
    placed = audio._placed(_blocks((0, 10), (limit + 1, 10)), 1000, Path("f.wav"), Timing())
    assert next(placed)[0] == 0
    with pytest.raises(AudioFileError):
        next(placed)


# ---- damaged packets and missing timestamps, through a stand-in container ---

class _Packet:
    def __init__(self, frames=(), error=None):
        self.frames, self.error = list(frames), error

    def decode(self):
        if self.error is not None:
            raise self.error
        return self.frames


def _frame(pts, samples: int = 480, rate: int = RATE):
    import av

    frame = av.AudioFrame.from_ndarray(np.full((1, samples), 0.1, np.float32),
                                       format="flt", layout="mono")
    frame.sample_rate = rate
    frame.pts = pts
    if pts is not None:
        frame.time_base = Fraction(1, rate)
    return frame


class _Container:
    """Just enough of an open container for _frames and _timed_blocks."""

    def __init__(self, packets, start_time=0):
        self.packets, self.start_time = packets, start_time

    def demux(self, *_streams):
        return iter(self.packets)


def _damage():
    import av

    return av.error.InvalidDataError(1094995529, "Invalid data")


def layout(packets):
    """(where it starts, length) of each block, in samples, once placed."""
    stream = SimpleNamespace(time_base=Fraction(1, RATE))
    blocks = audio._timed_blocks(_Container(packets), stream, Path("f.mka"),
                                 RATE, "mono", None, 0, Timing())
    placed, end = [], 0
    for silence, samples in audio._placed(blocks, RATE, Path("f.mka"), Timing()):
        end += silence
        placed.append((end, len(samples)))
        end += len(samples)
    return placed


def test_a_damaged_packet_is_skipped_and_counted():
    packets = [_Packet([_frame(0)]), _Packet(error=_damage()), _Packet([_frame(5280)])]
    timing = Timing()
    frames = list(audio._frames(_Container(packets), None, Path("f.mka"), timing))
    assert [f.pts for f in frames] == [0, 5280]
    assert timing.damaged == 1
    assert "1 damaged packet skipped" in timing.summary()
    # What came after it is still placed by its timestamp: 0.1 s of silence
    # where the damaged packet was, instead of the file stopping there.
    assert layout(packets) == [(0, 480), (5280, 480)]


def test_a_file_where_nothing_decodes_is_an_audio_file_error():
    packets = [_Packet(error=_damage()), _Packet(error=_damage())]
    with pytest.raises(AudioFileError, match="broken.mka: nothing in it would decode"):
        list(audio._frames(_Container(packets), None, Path("broken.mka"), Timing()))


def test_frames_without_timestamps_follow_on_until_one_has_one(monkeypatch):
    # Two frames with no timestamp, then the clock says 0.5 s: the first two
    # follow on from 0:00, and the rest goes where its clock says.
    packets = [_Packet([_frame(None)]), _Packet([_frame(None)]),
               _Packet([_frame(RATE // 2)]), _Packet([_frame(RATE // 2 + 480)])]
    assert layout(packets) == [(0, 480), (480, 480), (RATE // 2, 480), (RATE // 2 + 480, 480)]
    # And in a file with no 0:00 of its own, the first timestamp starts the
    # clock where the audio before it ended.
    monkeypatch.setattr(audio, "_source_origin", lambda *_args: None)
    assert layout(packets) == [(0, 480), (480, 480), (960, 480), (1440, 480)]


def test_nothing_to_say():
    assert Timing().summary() == ""


def test_jumps_and_damage_in_one_line():
    text = Timing(jumps=[(3723.4, 12.3), (5000.0, 2.0)], damaged=3).summary()
    assert "jump back in 2 places, first at 1:02:03" in text
    assert "off by up to 14.3 s" in text
    assert "3 damaged packets skipped" in text


# ---- what the worker hands a model ------------------------------------------

class _Whisper:
    """Stands in for a loaded faster-whisper model: keeps what it was given."""

    def transcribe(self, source, **_options):
        self.source = source
        return [], SimpleNamespace(duration=len(source) / 16000, language="en")


@pytest.fixture
def run_worker(monkeypatch):
    """Run a TranscribeWorker on one track with a stand-in model, synchronously.

    Returns (worker, notes, failures, reloads, logged). Nothing is
    loaded or downloaded, and nothing is written to the real diagnostics log.
    """
    logged = []
    monkeypatch.setattr(transcriber.diagnostics, "log_exception",
                        lambda what, exc, **fields: logged.append((what, fields)))
    monkeypatch.setattr(transcriber.diagnostics, "log_event",
                        lambda message, **fields: logged.append((message, fields)))
    reloads = Mock()
    monkeypatch.setattr(transcriber, "_get_model", reloads)

    def run(track: Track, model_name: str, device: str, model):
        worker = TranscribeWorker([track], [0], model_name, device, None)

        def load(backend: Backend):
            worker.backend = backend
            return model

        monkeypatch.setattr(worker, "_load_model", load)
        notes, failures = [], []
        worker.track_note.connect(lambda _i, note: notes.append(note))
        worker.track_failed.connect(lambda _i, message: failures.append(message))
        worker.run()
        return worker, notes, failures, reloads, logged

    return run


@pytest.fixture
def mlx_whisper(monkeypatch):
    """A stand-in mlx_whisper module that keeps what it was given."""
    module = SimpleNamespace(calls=[])

    def transcribe(source, **options):
        module.calls.append(source)
        return {"segments": [], "language": "en"}

    module.transcribe = transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    monkeypatch.setattr(transcriber, "mlx_available", lambda: True)
    return module


def test_faster_whisper_reads_through_the_apps_decoder(tmp_path, run_worker):
    # Given the path, faster-whisper would decode it itself and close the
    # two-second gap: "later" would be heard a second after "earlier".
    path = write_mkv(tmp_path / "gap.mkv", [[(0, noise(1, 1)), (3, noise(1, 2))]])
    model = _Whisper()
    worker, notes, failures, _reloads, _logged = run_worker(
        Track(path, "Alice"), "large-v3", "cpu", model)
    assert worker.backend == Backend("whisper-ct2", "cpu")
    assert isinstance(model.source, np.ndarray)
    assert len(model.source) == 4 * SAMPLE_RATE
    assert (notes, failures) == ([], [])


def test_mlx_whisper_gets_the_same_timed_array(tmp_path, run_worker, mlx_whisper):
    path = write_mkv(tmp_path / "gap.mkv", [[(0, noise(1, 1)), (3, noise(1, 2))]])
    worker, _notes, failures, _reloads, _logged = run_worker(
        Track(path, "Alice"), "large-v3", "mlx", transcriber._MlxModel("repo"))
    assert worker.backend == Backend("whisper-mlx", "metal")
    (source,) = mlx_whisper.calls
    assert isinstance(source, np.ndarray)       # never a path: no ffmpeg CLI here
    assert loud_seconds(source) == [0, 3]
    assert failures == []


def test_a_timing_note_goes_out_with_the_track(tmp_path, run_worker):
    path = joined_ts(tmp_path, 2, 1)
    _worker, notes, failures, _reloads, logged = run_worker(
        Track(path, "Alice"), "large-v3", "cpu", _Whisper())
    assert failures == []
    (note,) = notes
    assert "jump back at 0:02" in note
    assert ("decode note", {"track": str(path), "note": note}) in logged


def test_a_problem_with_the_file_fails_the_track_without_moving_off_the_gpu(
        tmp_path, run_worker, mlx_whisper):
    # mlx-whisper's fallback is faster-whisper on the CPU, which would read
    # the file no better and keep every later track off the GPU.
    path = write_mkv(tmp_path / "one.mkv", [[(0, noise(1, 4))]])
    track = Track(path, "Alice", stream_index=3)
    worker, _notes, failures, reloads, logged = run_worker(
        track, "large-v3", "mlx", transcriber._MlxModel("repo"))
    (failure,) = failures
    assert "has no audio stream 4" in failure
    assert worker.backend == Backend("whisper-mlx", "metal")
    reloads.assert_not_called()
    assert mlx_whisper.calls == []
    assert [what for what, _fields in logged] == ["decode"]
