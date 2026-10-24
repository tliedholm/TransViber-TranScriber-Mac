"""Probing and decoding of audio files, including multitrack containers.

A single file can hold several speakers two ways:
  - multiple audio streams in one container (OBS/REAPER/Craig exports), or
  - one stream with multiple channels (multichannel WAV/FLAC).
Both are expanded into one Track per speaker.

Decoding follows the file's own clock. Every stream in a file is measured
from one 0:00, the container's start, and a stream's audio goes where its
timestamps put it: a stream that began a minute after the others starts
with a minute of silence, and a stretch the recorder lost is silence too.
Laying decoded audio end to end instead, as faster-whisper's own decoder
does, moves everything after a gap earlier, and a late stream to 0:00 —
and breaks the rule merging relies on, that every track's 0:00 is the same
instant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from models import Track, default_speaker_name

SAMPLE_RATE = 16000

# A frame's timestamp up to this far from where the audio before it ended is
# rounding, and the audio runs on unbroken: Matroska keeps times to the
# millisecond, so a 1024-sample frame lands up to half a millisecond either
# side of where the samples before it put it. Further ahead is a gap (a lost
# packet, a paused recorder, a stream that began late) and silence fills it.
# Further back, the clock is taken to have restarted (two recordings joined
# end to end): the audio plays on from where it had got to, so nothing is
# lost, what follows keeps its own spacing from there, and the jump is noted
# in `Timing.jumps` for the caller to report.
TIMING_TOLERANCE = 0.02     # seconds
# Gaps are filled with silence only up to this far into the file: a timestamp
# beyond it is damage, not a recording that long, and decoding stops there
# with an error instead of allocating silence for it.
MAX_TIMELINE_SECONDS = 24 * 3600


class AudioFileError(ValueError):
    """The file can't be read as asked. Something about the file itself:
    trying again, or on another backend, gives the same answer."""


@dataclass
class Timing:
    """What following a stream's timestamps ran into, for the caller to report."""
    jumps: list[tuple[float, float]] = field(default_factory=list)  # (at s, back by s)
    damaged: int = 0            # packets that wouldn't decode, skipped

    def summary(self) -> str:
        """One line for the user, or "" when there is nothing to say."""
        from exporters import _hms

        notes = []
        if self.jumps:
            at = self.jumps[0][0]
            where = (f"at {_hms(at)}" if len(self.jumps) == 1
                     else f"in {len(self.jumps)} places, first at {_hms(at)}")
            total = sum(back for _at, back in self.jumps)
            notes.append(f"Its timestamps jump back {where}. The audio there carries on "
                         f"from where it was, so later times may be off by up to "
                         f"{total:.1f} s.")
        if self.damaged:
            packets = "packet" if self.damaged == 1 else "packets"
            notes.append(f"{self.damaged} damaged {packets} skipped, left as silence.")
        return " ".join(notes)


def _stream_channels(stream) -> int:
    channels = getattr(stream, "channels", None)
    if channels:
        return channels
    try:
        return len(stream.layout.channels)
    except Exception:
        return 1


def probe_streams(path: Path) -> list[dict]:
    """Audio streams in a file: [{'channels': int, 'duration': float}, ...]."""
    import av

    with av.open(str(path)) as container:
        fallback = container.duration / av.time_base if container.duration else 0.0
        streams = []
        for stream in container.streams.audio:
            if stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            else:
                duration = fallback
            streams.append({"channels": _stream_channels(stream), "duration": duration})
        return streams


def _channel_suffixes(count: int) -> list[str]:
    if count == 2:
        return [" L", " R"]
    return [f" Ch{n + 1}" for n in range(count)]


def tracks_for_file(path: Path, split_stereo: bool = False) -> list[Track]:
    """Expand a file into per-speaker Tracks.

    Multiple audio streams -> one track per stream. A stream with more than
    two channels -> one track per channel. Stereo is kept as one speaker
    unless split_stereo is set (right-click a track to split it).
    """
    base = default_speaker_name(path)
    try:
        streams = probe_streams(path)
    except Exception as exc:
        # A container this app genuinely cannot read used to be added as an
        # ordinary zero-duration track, and the codec error only surfaced
        # much later as a transcription failure. Keep the optimistic
        # single-stream guess, but carry the reason with it.
        return [Track(path=path, speaker=base,
                      probe_error=f"{type(exc).__name__}: {exc}")]
    if not streams:
        return [Track(path=path, speaker=base,
                      probe_error="No audio stream found in this file.")]

    multi_stream = len(streams) > 1
    tracks = []
    for si, stream in enumerate(streams):
        name = f"{base} T{si + 1}" if multi_stream else base
        channels = stream["channels"]
        if channels > 2 or (channels == 2 and split_stereo):
            for ch, suffix in enumerate(_channel_suffixes(channels)):
                tracks.append(Track(path=path, speaker=name + suffix,
                                    duration=stream["duration"],
                                    stream_index=si, channel=ch))
        else:
            tracks.append(Track(path=path, speaker=name, duration=stream["duration"],
                                stream_index=si if multi_stream else None))
    return tracks


def _audio_stream(container, path: Path, stream_index: int | None):
    """The audio stream a track names in an open container."""
    # The file can have changed since it was probed; say so instead of
    # raising a bare IndexError into the worker's generic handler.
    index = stream_index if stream_index is not None else 0
    streams = container.streams.audio
    if index >= len(streams):
        raise AudioFileError(f"{path.name} has no audio stream {index + 1} "
                             f"({len(streams)} found) — has the file changed?")
    return streams[index]


def _source_origin(container, path: Path) -> Fraction | None:
    """The file's 0:00 in seconds of its own timestamps, the same for every stream.

    The container's start time: when its earliest stream begins, codec
    priming the decoder trims already counted. Never each stream's own first
    timestamp, which would start a stream that began late at 0:00 too. A
    file that declares no start (WAV, raw AAC) is measured from the earliest
    timestamp any of its audio streams has, read from its first packets;
    None when it has no timestamps at all.
    """
    import av

    if container.start_time is not None:
        return Fraction(container.start_time, av.time_base)
    firsts: dict[int, Fraction] = {}
    with av.open(str(path)) as peek:
        audio = list(peek.streams.audio)
        for packet in peek.demux(*audio):
            if packet.pts is not None and packet.time_base is not None:
                firsts.setdefault(packet.stream_index, packet.pts * packet.time_base)
                if len(firsts) == len(audio):
                    break
    return min(firsts.values(), default=None)


def _frames(container, stream, path: Path, timing: Timing):
    """The stream's decoded frames, a packet that won't decode skipped.

    The timestamps after it put the rest of the audio back in place, with
    silence where that packet was; faster-whisper's decoder stopped there,
    and the file was cut short without a word. Only a stream where nothing
    decodes at all fails, with the first error.
    """
    import av

    first_error = None
    decoded = False
    for packet in container.demux(stream):
        try:
            frames = packet.decode()
        except av.error.InvalidDataError as exc:
            timing.damaged += 1
            first_error = first_error or exc
            continue
        for frame in frames:
            decoded = True
            yield frame
    if first_error is not None and not decoded:
        raise AudioFileError(f"{path.name}: nothing in it would decode "
                             f"({first_error})") from first_error


def _resampler(frame, rate: int, layout: str | None, channel: int | None,
               frame_size: int):
    """A function from frames (None flushes) to resampled ones, for one run.

    Packed float at `rate`, in `layout` (None: the frame's own). With
    `channel`, just that channel, as mono: converted to planar float at the
    source rate first so it can be picked out.
    """
    import av
    from av.audio.resampler import AudioResampler

    out = AudioResampler(format="flt", layout=layout or frame.layout, rate=rate,
                         frame_size=frame_size)
    if channel is None:
        return out.resample
    planar = AudioResampler(format="fltp", layout=frame.layout, rate=frame.sample_rate)

    def resample(frame):
        picked = []
        for part in planar.resample(frame):
            data = part.to_ndarray()
            row = np.ascontiguousarray(data[min(channel, data.shape[0] - 1)][None, :])
            mono = av.AudioFrame.from_ndarray(row, format="flt", layout="mono")
            mono.sample_rate = part.sample_rate
            picked += out.resample(mono)
        if frame is None:
            picked += out.resample(None)
        return picked

    return resample


def _timed_blocks(container, stream, path: Path, rate: int, layout: str | None,
                  channel: int | None, frame_size: int, timing: Timing):
    """(position, samples) for one stream decoded at `rate`, in decoding order.

    `samples` has one row per sample, a column per channel. `position` is
    where a run of blocks starts, in samples at `rate` from the file's 0:00
    (`_source_origin`), and None for every block that follows straight on
    from the one before. A run ends wherever a frame's timestamp strays more
    than TIMING_TOLERANCE from where the frames before it put it: the
    resampler is flushed and started again, so it never blends the audio
    either side of a gap into one. A frame without a timestamp follows on,
    and a run that began without one ends at the first frame that has one.
    """
    origin = _source_origin(container, path)
    # Decoded frames keep the stream's time base. Per frame the check is in
    # floats, which are exact enough to tell 20 ms from rounding; a run's
    # position is worked out exactly, once.
    time_base = stream.time_base
    tick = float(time_base) if time_base else None
    resample = None
    run_start: float | None = None    # seconds: the run's first timestamp
    run_samples = 0                   # source samples since then
    position = None                   # the run's position, until handed out
    emitted = 0                       # samples handed out so far

    def blocks(frames):
        nonlocal position, emitted
        for out in frames:
            samples = out.to_ndarray().reshape(-1, len(out.layout.channels))
            if samples.size:
                yield position, samples
                position = None
                emitted += len(samples)

    for frame in _frames(container, stream, path, timing):
        pts = frame.pts
        start = pts * tick if pts is not None and tick is not None else None
        if resample is not None and start is not None and (
                run_start is None
                or abs(start - run_start - run_samples / frame.sample_rate) > TIMING_TOLERANCE):
            yield from blocks(resample(None))
            resample = None
        if resample is None:
            resample = _resampler(frame, rate, layout, channel, frame_size)
            run_start, run_samples, position = start, 0, None
            if start is not None:
                exact = pts * time_base
                if origin is None:
                    # No clock until now: this frame follows on from
                    # whatever came before it untimed, and starts the clock.
                    origin = exact - Fraction(emitted, rate)
                position = round((exact - origin) * rate)
        resampled = resample(frame)
        if resampled:
            yield from blocks(resampled)
        run_samples += frame.samples
    if resample is not None:
        yield from blocks(resample(None))


def _placed(blocks, rate: int, path: Path, timing: Timing):
    """(silence, samples): each block with the silence that goes before it.

    The decision for every run's position (`_timed_blocks`): within
    TIMING_TOLERANCE of where the audio so far ended, it follows on; later,
    silence fills the gap; earlier, including before 0:00, it plays on from
    where the audio had got to, the jump is noted, and every position after
    it is read that much later, as from a clock that restarted. Audio is
    never dropped, so a damaged clock costs timing, never words.
    """
    end = 0
    lag = 0             # how much later than its timestamps the audio now runs
    tolerance = round(TIMING_TOLERANCE * rate)
    for position, samples in blocks:
        silence = 0
        if position is not None:
            position += lag
            gap = position - end
            if gap > tolerance:
                if position > MAX_TIMELINE_SECONDS * rate:
                    raise AudioFileError(
                        f"{path.name}: its timestamps put audio "
                        f"{position / rate / 3600:,.0f} hours into the file, past the "
                        f"{MAX_TIMELINE_SECONDS // 3600} hours this app reads. "
                        f"Its timing looks damaged.")
                silence = gap
            elif gap < -tolerance:
                timing.jumps.append((end / rate, -gap / rate))
                lag -= gap
        yield silence, samples
        end += silence + len(samples)


def _expected_seconds(container, stream) -> float:
    """How long the stream says it is, or 0 when it doesn't."""
    if stream.duration and stream.time_base:
        return float(stream.duration * stream.time_base)
    if container.duration:
        return container.duration / 1_000_000
    return 0.0


def decode_track_audio(path: Path, stream_index: int | None, channel: int | None,
                       timing: Timing | None = None) -> np.ndarray:
    """Decode one stream (and optionally one channel of it) to mono float32 16 kHz.

    On the file's own clock (see the module notes): a stream that began late
    keeps its leading silence, and a gap in its timestamps stays a gap. What
    that ran into is added to `timing`, when given.
    """
    import av

    timing = timing if timing is not None else Timing()
    with av.open(str(path)) as container:
        stream = _audio_stream(container, path, stream_index)
        # Written in place, not gathered and joined: for a four-hour track
        # that is one copy of nearly a gigabyte instead of two at once.
        expected = min(_expected_seconds(container, stream), MAX_TIMELINE_SECONDS)
        audio = np.zeros(int(expected * 1.01 * SAMPLE_RATE) + SAMPLE_RATE, np.float32)
        size = 0
        blocks = _timed_blocks(container, stream, path, SAMPLE_RATE, "mono", channel,
                               SAMPLE_RATE, timing)
        for silence, samples in _placed(blocks, SAMPLE_RATE, path, timing):
            needed = size + silence + len(samples)
            if needed > audio.size:
                grown = np.zeros(max(needed, audio.size * 3 // 2), np.float32)
                grown[:size] = audio[:size]
                audio = grown
            size += silence             # already zeros
            audio[size:size + len(samples)] = samples[:, 0]
            size += len(samples)
    audio.resize(size, refcheck=False)
    return audio
