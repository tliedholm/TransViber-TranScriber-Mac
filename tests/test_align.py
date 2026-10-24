"""Track offsets: what the Offset column accepts, and how Align suggests them.

The matcher tests run on synthetic frame energies: talkers made of syllables
of random length and loudness, mixed and cropped the way real sessions are.
They pin the logic — lag direction, what counts as strong, what may chain —
not the thresholds, which are still to be calibrated on real recordings.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import align
from align import (
    AUTOMATIC_REFERENCE,
    FRAME_RATE,
    SESSION_REFERENCE,
    AlignWorker,
    Cancelled,
    Envelope,
    Match,
    find_offset,
    fingerprint,
    format_offset,
    parse_offset,
    suggest_offsets,
)


# ---------- The Offset column ----------

@pytest.mark.parametrize("text, expected", [
    ("150", 150.0),
    ("150.25", 150.25),
    ("2:30", 150.0),
    ("2:30.5", 150.5),
    ("2:30,5", 150.5),          # decimal comma
    ("1:02:30", 3750.0),
    ("1:02:30.250", 3750.25),
    ("90:00", 5400.0),          # the first field can be any size
    ("30:00:00", 108000.0),     # hours are not limited to 24
    ("0", 0.0),
    ("  2:30  ", 150.0),
    ("0:00", 0.0),
])
def test_accepted_offset_forms(text, expected):
    assert parse_offset(text) == expected


def test_offsets_round_to_milliseconds():
    assert parse_offset("1.23456") == 1.235


@pytest.mark.parametrize("text, reason", [
    ("", "0 to reset"),
    ("   ", "0 to reset"),
    ("-5", "negative"),
    ("-0:30", "negative"),
    ("abc", "isn't a time"),
    ("nan", "isn't a time"),
    ("inf", "isn't a time"),
    ("1:75", "under 60"),
    ("1:60", "under 60"),
    ("1:02:60", "under 60"),
    ("1:2:3:4", "at most"),
    ("1.5:30", "isn't a time"),  # only the last field takes a fraction
    ("2:", "isn't a time"),
    (":30", "isn't a time"),
    ("2:30.", "isn't a time"),
    ("1e3", "isn't a time"),
    ("+30", "isn't a time"),
    ("١٥٠", "isn't a time"),  # non-ASCII digits
    ("9" * 400, "too large"),
    ("9" * 400 + ":00", "too large"),
    ("1" + "0" * 309, "too large"),
    ("999999999", "too large"),   # over 100 days
])
def test_rejected_offset_forms_say_why(text, reason):
    with pytest.raises(ValueError, match=reason):
        parse_offset(text)


@pytest.mark.parametrize("seconds, expected", [
    (0, "0:00"),
    (5, "0:05"),
    (150, "2:30"),
    (150.5, "2:30.5"),
    (312.4, "5:12.4"),
    (150.25, "2:30.25"),
    (150.125, "2:30.125"),
    (3750, "1:02:30"),
    (108000, "30:00:00"),       # no 24-hour wrap
    (59.9999, "1:00"),          # rounds to milliseconds first
])
def test_offsets_are_shown_in_standard_form(seconds, expected):
    assert format_offset(seconds) == expected


@pytest.mark.parametrize("seconds",
                         [0.0, 0.001, 1.5, 59.999, 61.0, 150.25, 3599.5, 3750.125, 90061.0])
def test_offsets_round_trip(seconds):
    assert parse_offset(format_offset(seconds)) == seconds


# ---------- Synthetic sessions ----------

def speech(seconds: float, seed: int, talk: float = 0.35):
    """Frame energies of one talker: turns of 1-8 s made of uneven syllables."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FRAME_RATE)
    energy = np.zeros(n)
    t, on = 0, rng.random() < talk
    while t < n:
        run = int(rng.uniform(1, 8) * FRAME_RATE)
        if on:
            s, stop = t, min(n, t + run)
            while s < stop:
                length = int(rng.uniform(0.08, 0.3) * FRAME_RATE)
                energy[s:min(s + length, stop)] = rng.uniform(0.1, 1.0)
                s += length + int(rng.uniform(0.02, 0.15) * FRAME_RATE)
        t += run
        on = rng.random() < (talk if not on else 0.3)
    return energy * rng.gamma(10, 0.1, n) * 1e-2


def hiss(seconds: float, seed: int):
    """A quiet microphone's own noise floor, about -60 dBFS."""
    return np.random.default_rng(seed).gamma(40, 1 / 40, int(seconds * FRAME_RATE)) * 1e-6


SESSION = 600          # seconds
LATE = 7345            # frames: 73.45 s


@pytest.fixture(scope="module")
def session():
    """Four talkers over ten minutes; a mix of three of them; the third's own
    file, begun 73.45 s in."""
    a, b, c, d = (speech(SESSION, seed) for seed in (1, 2, 3, 4))
    return SimpleNamespace(a=a, b=b, c=c, d=d,
                           mix=a + 0.5 * b + 0.3 * c + hiss(SESSION, 10),
                           late=(c + hiss(SESSION, 11))[LATE:])


def not_wrongly_strong(match: Match, expected_lag: float | None) -> bool:
    """A strong match must be the right one; with no right one, none is strong."""
    if expected_lag is None:
        return match.strength != "strong"
    return match.strength != "strong" or round(match.lag - expected_lag, 2) == 0


# ---------- Envelope ----------

def test_envelope_is_the_same_in_one_piece_or_in_chunks():
    samples = np.random.default_rng(0).standard_normal(10007)
    whole = Envelope()
    whole.feed(samples)
    for size in (1, 79, 333):
        pieces = Envelope()
        for i in range(0, samples.size, size):
            pieces.feed(samples[i:i + size])
        np.testing.assert_allclose(pieces.energy(), whole.energy())
    assert whole.energy().size == 10007 // 80


def test_frame_energy_is_the_mean_square():
    envelope = Envelope()
    envelope.feed(np.full(800, 0.5))
    np.testing.assert_allclose(envelope.energy(), np.full(10, 0.25))


def test_power_fed_directly_matches_samples():
    samples = np.random.default_rng(1).standard_normal(4000)
    from_samples, from_power = Envelope(), Envelope()
    from_samples.feed(samples)
    from_power.feed_power(samples * samples)
    np.testing.assert_allclose(from_power.energy(), from_samples.energy())


def test_a_damaged_sample_costs_one_frame_not_the_rest_of_the_file():
    samples = np.full(800, 0.5)
    samples[85] = np.nan
    samples[400] = np.inf
    energy = Envelope()
    energy.feed(samples)
    assert np.all(np.isfinite(energy.energy()))
    assert int(np.sum(energy.energy() == 0.25)) == 8


def test_streams_of_one_file_are_heard_together():
    np.testing.assert_allclose(align.sum_energies([np.ones(3), np.ones(5)]), [2, 2, 2, 1, 1])


# ---------- find_offset ----------

def test_finds_a_late_start_in_a_full_session_mix(session):
    match = find_offset(fingerprint(session.mix), fingerprint(session.late),
                        require_window_agreement=True)
    assert match.strength == "strong"
    assert match.lag == pytest.approx(73.45, abs=0.005)
    assert (match.windows_agreeing, match.windows_checked) == (3, 3)


def test_lag_direction(session):
    # The cropped copy began 30 s after the original did, not before.
    original = session.a + hiss(SESSION, 12)
    cropped = original[30 * FRAME_RATE:]
    assert find_offset(fingerprint(original), fingerprint(cropped)).lag == pytest.approx(30.0)
    assert find_offset(fingerprint(cropped), fingerprint(original)).lag == pytest.approx(-30.0)


def test_recordings_that_share_no_sound_are_no_match(session):
    match = find_offset(fingerprint(session.a + hiss(SESSION, 13)),
                        fingerprint(session.b + hiss(SESSION, 14)))
    assert match.strength == "none"


def test_a_speaker_absent_from_the_mix_is_not_matched(session):
    absent = (session.d + hiss(SESSION, 15))[LATE:]
    match = find_offset(fingerprint(session.mix), fingerprint(absent),
                        require_window_agreement=True)
    assert not_wrongly_strong(match, None)


@pytest.mark.parametrize("quiet_start", [hiss(60, 17), np.zeros(6000)],
                         ids=["hiss", "digital silence"])
def test_leading_silence_is_kept_as_timing(session, quiet_start):
    own = session.c + hiss(SESSION, 16)
    own[:6000] = quiet_start
    match = find_offset(fingerprint(session.mix), fingerprint(own),
                        require_window_agreement=True)
    assert (match.strength, match.lag) == ("strong", 0.0)


def test_a_speaker_who_also_left_early(session):
    own = (session.c + hiss(SESSION, 18))[LATE:LATE + 300 * FRAME_RATE]
    match = find_offset(fingerprint(session.mix), fingerprint(own),
                        require_window_agreement=True)
    assert match.strength == "strong"
    assert match.lag == pytest.approx(73.45, abs=0.005)


@pytest.mark.parametrize("gain", [0.05, 0.01, 0.002])
def test_a_quiet_speaker_is_never_placed_wrongly(session, gain):
    mix = session.a + 0.5 * session.b + gain * session.c + hiss(SESSION, 10)
    match = find_offset(fingerprint(mix), fingerprint(session.late),
                        require_window_agreement=True)
    assert not_wrongly_strong(match, 73.45)


@pytest.fixture(scope="module")
def bobs_room():
    """Two people in one room for 25 minutes: the room, and Bob's own file."""
    length = 1500
    alice, bob = speech(length, 40), speech(length, 41)
    return (fingerprint(alice + 0.1 * bob + hiss(length, 42)),
            bob + 0.1 * alice + hiss(length, 43))


def test_consistency_check_flags_an_edited_file(bobs_room):
    # Bob's file lost 0.3 s two thirds of the way in.
    room, bobs_file = bobs_room
    whole = find_offset(room, fingerprint(bobs_file[5000:125000]))
    assert (whole.strength, whole.lag) == ("strong", 50.0)
    edited = np.concatenate([bobs_file[5000:85000], bobs_file[85030:125000]])
    match = find_offset(room, fingerprint(edited))
    assert match.strength == "weak"
    assert "start and end disagree by 0.3 s" in match.detail


# Long cuts are caught too, wherever they fall: each part searches every lag,
# and a part straddling the cut isn't outvoted by the agreeing side of it. A
# quarter of the way in, two thirds, and late enough that the last third
# mostly agrees and only the sixth doesn't.
@pytest.mark.parametrize("cut_at", [35000, 85000, 105000])
@pytest.mark.parametrize("cut", [30, 6000])
def test_a_cut_is_caught_wherever_it_falls(bobs_room, cut_at, cut):
    room, bobs_file = bobs_room
    edited = np.concatenate([bobs_file[5000:cut_at], bobs_file[cut_at + cut:125000]])
    match = find_offset(room, fingerprint(edited))
    assert match.strength == "weak"
    assert f"start and end disagree by {cut / FRAME_RATE:.1f} s" in match.detail


# 30 s lost 90 s in: the first sixth straddles the cut almost evenly, two real
# peaks with neither standing out from the other. And 60 s from either end,
# where only the 80 s end part sees it.
@pytest.mark.parametrize("kept", [(14000, 17000), (11000, 14000), (116000, 119000)])
def test_a_cut_near_either_end_is_caught(bobs_room, kept):
    room, bobs_file = bobs_room
    edited = np.concatenate([bobs_file[5000:kept[0]], bobs_file[kept[1]:125000]])
    match = find_offset(room, fingerprint(edited))
    assert match.strength == "weak"
    assert "start and end disagree by 30.0 s" in match.detail


@pytest.mark.parametrize("length", [162, 180, 238, 242])
def test_a_short_edited_recording_is_checked(length):
    # A file that lost 5 s 50 s in. Under 240 s of overlap no third is 80 s
    # long, but the first and last 80 s fit without overlapping, and the first
    # finds the cut. The same files unedited stay strong.
    energy = speech(length + 5, 1, talk=0.8) + hiss(length + 5, 50)
    edited = np.concatenate([energy[:5000], energy[5500:]])
    match = find_offset(fingerprint(energy), fingerprint(edited))
    assert (match.strength, match.lag) == ("weak", 5.0)
    assert "start and end disagree by 5.0 s" in match.detail
    clean = find_offset(fingerprint(energy), fingerprint(energy[500:]))
    assert (clean.strength, clean.lag) == ("strong", 5.0)


@pytest.mark.parametrize("edited", [True, False], ids=["edited", "unedited"])
def test_a_match_too_short_to_check_is_never_strong(edited):
    # Under 160 s of overlap there is no room for two 80 s parts, so an edit
    # would go unseen: edited or not, the match is only offered.
    energy = speech(163, 1, talk=0.8) + hiss(163, 50)
    other = np.concatenate([energy[:5000], energy[5500:]]) if edited else energy[500:]
    match = find_offset(fingerprint(energy), fingerprint(other))
    assert (match.strength, match.lag) == ("weak", 5.0)
    assert match.detail == "too short to check for an edit"


@pytest.mark.parametrize("overlap, sizes", [
    (15999, []), (16000, [8000]), (23999, [8000]), (24000, [8000]),
    (24003, [8001, 8000]), (48000, [16000, 8000]), (96000, [32000, 16000, 8000]),
])
def test_part_sizes_around_the_boundaries(overlap, sizes):
    # 80 s parts once two fit; thirds and smaller once a third is 80 s. A
    # fraction that comes out at 80 s is the same part, checked once.
    assert align._part_sizes(overlap) == sizes


def test_a_sparse_speaker_is_not_contradicted_by_a_chance_peak():
    # Says little, mostly late on: in the last third a chance lag comes first
    # and the true one only second. That is too little to go on, not a part
    # of the file that matches somewhere else.
    base = 13_022_360
    a, b, c = speech(1200, base), speech(1200, base + 1), speech(1200, base + 2, talk=0.03)
    mix = a + 0.5 * b + 0.3 * c + hiss(1200, base + 3)
    own = (c + 10 * hiss(1200, base + 4))[6321:]
    match = find_offset(fingerprint(mix), fingerprint(own), require_window_agreement=True)
    assert (match.strength, match.lag) == ("strong", 63.21)


def test_a_part_with_little_sound_cannot_contradict():
    # A gated headset, silent but for a few words in its first and last
    # 221 s: the small end parts hold only those, and their chance peaks are
    # no evidence against the rest.
    n, edge = 1400 * FRAME_RATE, 22130
    a, c = speech(1400, 1477), speech(1400, 1479)
    keep = np.zeros(n, bool)
    keep[edge:n - edge] = True
    for word in (38307, 133628, 115608, 37745, 9803, 124772, 76737):
        keep[word:word + 40] = True
    c = np.where(keep, c, 0.0)
    room = a + 0.1 * c + hiss(1400, 1480)
    match = find_offset(fingerprint(room), fingerprint(c[3372:]))
    assert (match.strength, match.lag) == ("strong", 33.72)


@pytest.mark.parametrize("seed, talk, late, cut, at", [
    (335, 0.08, 7253, 500, 9605),
    (109, 0.04, 4726, 3000, 7252),
])
def test_edits_found_by_the_smaller_parts(seed, talk, late, cut, at):
    # Sparse talkers against a session reference whose files lost 5 s or 30 s
    # some 75 s in. The first is seen by a twelfth (the sixth straddles the
    # cut and mostly agrees); in the second the part's match elsewhere is
    # clear but not strong, and that is enough.
    a, b = speech(1600, 5 * seed + 31), speech(1600, 5 * seed + 32)
    c = speech(1600, 5 * seed + 33, talk=talk)
    mix = a + 0.5 * b + 0.3 * c + hiss(1600, 5 * seed + 34)
    own = (c + hiss(1600, 5 * seed + 35))[late:]
    edited = np.concatenate([own[:at], own[at + cut:]])
    match = find_offset(fingerprint(mix), fingerprint(edited), require_window_agreement=True)
    assert match.strength == "weak"
    assert f"start and end disagree by {cut / FRAME_RATE:.1f} s" in match.detail


def test_two_peaks_means_the_true_lag_is_one_of_them():
    # Stand-in fingerprints of bare onsets, every stretch of the reference
    # holding 20. The part finds 16 of its onsets at one lag and 10 at a
    # third; at its true lag, 15 or 12. Neither leader stands out from the
    # other. With 15 both stand out from everything else: a cut. With 12 the
    # true lag is no clearer than chance, so the leader is a lone chance
    # peak, and that is too little to go on, not a contradiction.
    rng = np.random.default_rng(0)
    size, length, true_lag = 20000, 80000, 30000
    onsets = rng.choice(size, 100, replace=False)
    part = np.zeros(size)
    part[onsets] = 1.0
    verdicts = []
    for at_true in (15, 12):
        ref = np.zeros(length)
        for lag, count in ((5000, 16), (true_lag, at_true), (55000, 10)):
            ref[lag + onsets[:count]] = 1.0
            spare = np.setdiff1d(np.arange(size), onsets)
            ref[lag + rng.choice(spare, 20 - count, replace=False)] = 1.0
        verdicts.append(align._part_verdict(ref, part, 0, size, true_lag)[0])
    assert verdicts == ["disagrees", "unclear"]


def test_a_part_is_only_scored_where_it_lies_inside_the_reference(session):
    # The first 60 s of this part are a copy of the reference's last 60 s:
    # slid to the reference's very end, half the part matches perfectly.
    # Scored on that sliver it would contradict the true lag.
    ref = fingerprint(session.mix)
    other = fingerprint(session.late).copy()
    other[:6000] = ref[-6000:]
    assert align._part_verdict(ref, other, 0, 12000, LATE)[0] != "disagrees"


def _gated(seed, words):
    """A gated headset: a handful of words, digital silence between."""
    rng = np.random.default_rng(seed)
    energy = np.zeros(600 * FRAME_RATE)
    for at in rng.uniform(0, 599, words):
        start = int(at * FRAME_RATE)
        energy[start:start + int(rng.uniform(0.2, 0.6) * FRAME_RATE)] = rng.uniform(0.1, 1)
    return energy * 1e-2


# The silence between can be digital zeros, or a floor far too quiet to hear
# (dither, a noise suppressor's residue): neither is sound.
@pytest.mark.parametrize("floor", [0.0, 1e-10])
@pytest.mark.parametrize("seed, words", [(24, 3), (109, 5), (139, 3)])
def test_two_mostly_silent_files_are_not_a_strong_match(floor, seed, words):
    # Two onsets meeting by chance outscore one; scored alone, that has passed
    # for strong.
    ref, other = (_gated(s, words) + hiss(600, s) * floor * 1e6 for s in (seed, 10000 + seed))
    assert find_offset(fingerprint(ref), fingerprint(other)).strength != "strong"


@pytest.mark.parametrize("floor", [0.0, 1.0], ids=["digital silence", "hiss"])
def test_seconds_of_sound_are_counted_however_little_there_is(floor):
    # Half a minute of speech in an hour, between digital silence or a
    # microphone's hiss: either way it is the seconds of speech that count,
    # not none of them (the busiest 1% of seconds are silent) and not every
    # second of hiss.
    speaking = np.zeros(3600 * FRAME_RATE)
    for second in np.random.default_rng(3).choice(3600, 30, replace=False):
        speaking[second * FRAME_RATE:second * FRAME_RATE + 30] = 0.01
    fp = fingerprint(speaking + floor * hiss(3600, 4))
    active = align._active_seconds(fp, 0, fp.size).size
    assert 30 <= active <= 60    # a word's rise can spill into the next second


def test_a_peak_resting_on_little_overlap_is_not_strong(session):
    # They share 150 s of sound: the end of one and the start of the other.
    shared = session.c[-15000:] + hiss(150, 23)
    ref = np.concatenate([speech(450, 24) + hiss(450, 25), shared])
    other = np.concatenate([shared, speech(450, 26)])
    match = find_offset(fingerprint(ref), fingerprint(other))
    assert (match.lag, match.strength) == (450.0, "weak")
    assert "only 150 s of overlap" in match.detail


def test_repeated_audio_is_not_a_strong_match():
    repeated = np.tile(speech(120, 30), 5)
    match = find_offset(fingerprint(repeated + hiss(600, 31)),
                        fingerprint((repeated + hiss(600, 32))[4000:40000]),
                        require_window_agreement=True)
    assert not_wrongly_strong(match, None)


# ---------- Window agreement: against a session reference, separate parts
# ---------- must each find the lag again

def test_too_little_audio_to_confirm_stays_weak(session):
    short = (session.c + hiss(SESSION, 19))[LATE:LATE + 60 * FRAME_RATE]
    whole = find_offset(fingerprint(session.mix), fingerprint(short))
    assert whole.detail == "too short to check for an edit"
    match = find_offset(fingerprint(session.mix), fingerprint(short),
                        require_window_agreement=True)
    assert match.strength == "weak"
    assert match.lag == pytest.approx(73.45, abs=0.005)   # the same peak, unconfirmed
    assert "too little audio" in match.detail     # what the stretches found comes first


def test_stretches_come_from_where_the_speaker_talks(session):
    # Talking only in the first third, hiss after: confirmed there, not
    # failed by stretches of hiss.
    own = session.late.copy()
    third = own.size // 3
    own[third:] = hiss(SESSION, 20)[:own.size - third]
    fp = fingerprint(own)
    assert align._active_seconds(fp, 0, fp.size).max() < third // FRAME_RATE + 1
    match = find_offset(fingerprint(session.mix), fp, require_window_agreement=True)
    assert match.strength == "strong"
    assert (match.windows_agreeing, match.windows_checked) == (3, 3)


@pytest.mark.parametrize("quiet", [hiss(SESSION, 27), np.zeros(60000)],
                         ids=["hiss", "digital silence"])
def test_silence_never_counts_as_activity(session, quiet):
    busy = np.concatenate([fingerprint(session.mix)[:3000], fingerprint(quiet + 0)[3000:]])
    assert align._active_seconds(busy, 0, busy.size).max() < 30


def test_a_speaker_heard_in_the_mix_for_only_part_of_their_file_is_weak(session):
    # Muted in the call a quarter of the way into their own file: only one
    # stretch can find them, and one is not enough.
    in_mix = session.c.copy()
    in_mix[LATE + (session.c.size - LATE) // 4:] = 0
    mix = session.a + 0.5 * session.b + 0.3 * in_mix + hiss(SESSION, 10)
    match = find_offset(fingerprint(mix), fingerprint(session.late),
                        require_window_agreement=True)
    assert match.strength == "weak"
    assert (match.windows_agreeing, match.windows_checked) == (1, 3)
    assert "only 1 of 3 separate parts of the file confirm it" in match.detail


def _conflict(ref, other):
    """find_offset's verdict, and what the stretches alone make of it."""
    match = find_offset(ref, other, require_window_agreement=True)
    lag = int(round(match.lag * FRAME_RATE))
    start, end = max(0, -lag), min(other.size, ref.size - lag)
    return match, align._window_agreement(ref, other, lag, start, end)[3]


def test_a_short_cut_is_a_conflict(session):
    own = session.c + hiss(SESSION, 11)
    cut = np.concatenate([own[LATE:30000], own[30030:]])      # 0.3 s lost
    match, conflict = _conflict(fingerprint(session.mix), fingerprint(cut))
    assert match.strength == "weak"
    assert "0.3 s" in match.detail
    assert conflict == pytest.approx(0.3)


@pytest.mark.parametrize("seed, start, cut_at", [(677, 4241, 33786), (11, 3246, 8625)])
def test_each_disagreeing_stretch_counts_however_it_scores(seed, start, cut_at):
    # 30 s cut out of a 500 s file. In the first the stretch holding the cut
    # finds the other lag less clearly than a strong match would; in the
    # second it splits nearly evenly across the cut, two real peaks neither
    # of which stands out from the other.
    a, b, c = (speech(500, seed * 3 + 700000 + k) for k in range(3))
    mix = a + 0.5 * b + 0.3 * c + hiss(500, seed + 800000)
    own = c + hiss(500, seed + 900000)
    edited = np.concatenate([own[start:cut_at], own[cut_at + 3000:]])
    match, conflict = _conflict(fingerprint(mix), fingerprint(edited))
    assert match.strength == "weak"
    assert "30.0 s" in match.detail
    assert conflict == pytest.approx(30.0)


# 300 s, then a 60 s gap cut out, then 200 s more; and 140 s, the gap, then 400 s.
@pytest.mark.parametrize("first, after", [(30000, 20000), (14000, 40000)])
def test_a_part_matching_somewhere_else_is_a_conflict(first, after):
    length, start = 900, 2000
    mix = speech(length, 1) + 0.5 * speech(length, 2) + 0.3 * speech(length, 3) + hiss(length, 10)
    own = speech(length, 3) + hiss(length, 21)
    edited = np.concatenate([own[start:start + first],
                             own[start + first + 6000:start + first + 6000 + after]])
    match, conflict = _conflict(fingerprint(mix), fingerprint(edited))
    assert match.strength == "weak"
    assert "60.0 s" in match.detail
    assert conflict == pytest.approx(60.0)


def test_parts_from_different_places_are_not_strong(session):
    own = session.c + hiss(SESSION, 21)
    spliced = np.concatenate([own[1000:15000], own[40000:54000], own[20000:34000]])
    match = find_offset(fingerprint(session.mix), fingerprint(spliced),
                        require_window_agreement=True)
    assert not_wrongly_strong(match, None)


# ---------- suggest_offsets, through a scripted matcher ----------

class _Fp:
    """A stand-in fingerprint: suggest_offsets only measures it and passes it on."""

    def __init__(self, name: str, seconds: float) -> None:
        self.name, self.frames = name, int(seconds * FRAME_RATE)

    def __len__(self) -> int:
        return self.frames


class _Matcher:
    """Scripted matches between stand-in fingerprints, keyed (reference, other)."""

    def __init__(self, **edges: tuple[str, float]) -> None:
        self.edges = {tuple(key.split("_")): Match(lag=lag, strength=strength, ratio=2.5)
                      for key, (strength, lag) in edges.items()}
        self.calls: list[tuple[str, str, bool]] = []

    def __call__(self, ref, other, require_window_agreement=False) -> Match:
        self.calls.append((ref.name, other.name, require_window_agreement))
        return self.edges.get((ref.name, other.name), Match())


def _run(lengths: dict, matcher: _Matcher, offsets: dict | None = None,
         reference: str | None = None, broken: dict | None = None):
    paths = {name: Path(f"{name}.wav") for name in lengths}
    fingerprints = {paths[n]: _Fp(n, sec) for n, sec in lengths.items()}
    for name, exc in (broken or {}).items():
        fingerprints[paths[name]] = exc
    units = [p for n, p in paths.items() if n != reference]
    result = suggest_offsets(units, fingerprints,
                             {paths[n]: v for n, v in (offsets or {}).items()},
                             paths[reference] if reference else None, matcher=matcher)
    return result, {s.path.stem: s for s in result.suggestions}


def test_chains_through_a_strong_match():
    matcher = _Matcher(a_b=("strong", 5.0), b_c=("strong", 2.0))
    result, rows = _run({"a": 100, "b": 90, "c": 80}, matcher)
    assert result.mode == AUTOMATIC_REFERENCE
    assert rows["a"].method == "reference"
    assert (rows["b"].offset, rows["c"].offset) == (5.0, 7.0)
    assert "Matched sound with b.wav" in rows["c"].reason
    assert all(not agree for *_names, agree in matcher.calls)


def test_never_chains_through_a_weak_match():
    matcher = _Matcher(a_b=("weak", 5.0), b_c=("strong", 2.0))
    _result, rows = _run({"a": 100, "b": 90, "c": 80}, matcher)
    assert (rows["b"].method, rows["b"].strength) == ("matched", "weak")
    assert rows["c"].method == "guess"
    assert ("b", "c", False) not in matcher.calls


def test_a_weak_match_is_kept_rather_than_replaced_by_a_guess():
    _result, rows = _run({"a": 100, "b": 40}, _Matcher(a_b=("weak", 5.0)))
    assert (rows["b"].offset, rows["b"].strength) == (5.0, "weak")


def test_the_guess_assumes_everyone_stopped_together():
    _result, rows = _run({"a": 100, "b": 40}, _Matcher(), offsets={"a": 10.0})
    assert (rows["b"].method, rows["b"].offset) == ("guess", 70.0)
    assert "stopped when a.wav ended" in rows["b"].reason


def test_anchors_on_the_automatic_references_current_offset():
    # The reference starts at 10:00; an event 3:00 into it is 1:00 into b.
    _result, rows = _run({"a": 1000, "b": 500}, _Matcher(a_b=("strong", 120.0)),
                         offsets={"a": 600.0})
    assert rows["a"].offset == 600.0
    assert format_offset(rows["b"].offset) == "12:00"


def test_without_a_reference_everything_moves_up_to_zero():
    result, rows = _run({"a": 100, "b": 90}, _Matcher(a_b=("strong", -30.0)))
    assert (rows["a"].offset, rows["b"].offset) == (30.0, 0.0)
    assert "moved 0:30 later" in result.note


def test_a_weak_match_does_not_move_everyone_else():
    result, rows = _run({"a": 100, "b": 90, "c": 80},
                        _Matcher(a_b=("weak", -30.0), a_c=("strong", 5.0)))
    assert (result.shift, result.note) == (0.0, "")
    assert (rows["a"].offset, rows["c"].offset) == (0.0, 5.0)
    assert rows["b"].offset == -30.0
    assert not rows["b"].applicable
    assert "would start 0:30 before 0:00" in rows["b"].reason


def test_rebasing_follows_a_new_reference_offset():
    result, _rows = _run({"a": 100, "b": 90, "c": 80},
                         _Matcher(a_b=("strong", -30.0), a_c=("strong", 5.0)))
    assert result.shift == 30.0
    moved = align.rebase(result, 120.0)
    assert {s.path.stem: s.offset for s in moved.suggestions} == {"a": 120.0, "b": 90.0,
                                                                   "c": 125.0}
    assert (moved.anchor, moved.shift, moved.note) == (120.0, 0.0, "")
    # The original result is untouched, and the reference result can't move.
    assert {s.path.stem: s.offset for s in result.suggestions} == {"a": 30.0, "b": 0.0,
                                                                    "c": 35.0}
    session_result, _rows = _run({"session": 100, "b": 90},
                                 _Matcher(session_b=("strong", 5.0)), reference="session")
    assert align.rebase(session_result, 50.0) is session_result


def test_a_session_reference_wins_even_when_shorter():
    result, rows = _run({"session": 1000, "a": 2000}, _Matcher(session_a=("strong", 0.0)),
                        reference="session")
    assert result.mode == SESSION_REFERENCE
    assert result.reference == Path("session.wav")
    assert list(rows) == ["a"]  # the reference is no speaker row
    assert rows["a"].method == "matched"


def test_the_session_reference_fixes_zero():
    # A sentence 17:10 into the reference is 2:10 into Bob's file.
    matcher = _Matcher(session_bob=("strong", 17 * 60 + 10 - (2 * 60 + 10)))
    result, rows = _run({"session": 3600, "bob": 3000}, matcher,
                        offsets={"bob": 99.0}, reference="session")
    assert format_offset(rows["bob"].offset) == "15:00"
    assert result.note == ""
    assert matcher.calls == [("session", "bob", True)]


def test_every_edge_needs_window_agreement_with_a_session_reference():
    matcher = _Matcher(session_b=("strong", 5.0), b_c=("strong", 2.0))
    _result, rows = _run({"session": 100, "b": 90, "c": 80}, matcher, reference="session")
    assert rows["c"].offset == 7.0
    assert all(agree for *_names, agree in matcher.calls)


def test_a_file_starting_before_the_session_reference_cannot_apply():
    matcher = _Matcher(session_b=("strong", -5.0), b_c=("strong", 3.0),
                       session_d=("strong", 10.0))
    result, rows = _run({"session": 100, "b": 90, "c": 50, "d": 60}, matcher,
                        reference="session")
    assert rows["b"].offset == -5.0
    assert not rows["b"].applicable
    assert "Starts before the session reference by 0:05" in rows["b"].reason
    # Not a stepping stone, and no guess hides it.
    assert rows["b"].method == "matched"
    assert ("b", "c", True) not in matcher.calls
    assert rows["c"].method == "guess"
    # Zero stays put and the valid match still applies.
    assert (rows["d"].offset, rows["d"].applicable) == (10.0, True)
    assert result.note == ""


# b is strongly before the reference; d would place it after it. In either
# table order: b tried before d is placed, or after.
@pytest.mark.parametrize("lengths", [{"session": 100, "b": 90, "d": 60},
                                     {"session": 100, "d": 60, "b": 90}],
                         ids=["b first", "d first"])
def test_a_strong_start_before_the_reference_is_not_chained_around(lengths):
    matcher = _Matcher(session_b=("strong", -5.0), session_d=("strong", 10.0),
                       d_b=("strong", 30.0))
    _result, rows = _run(lengths, matcher, reference="session")
    assert rows["b"].offset == -5.0
    assert not rows["b"].applicable
    assert "Starts before the session reference" in rows["b"].reason
    assert ("d", "b", True) not in matcher.calls


def test_a_negative_guess_against_the_session_reference_cannot_apply():
    _result, rows = _run({"session": 100, "long": 150}, _Matcher(), reference="session")
    assert rows["long"].offset == -50.0
    assert not rows["long"].applicable
    assert ("Starts before the session reference by 0:50 if it stopped when "
            "session.wav ended") in rows["long"].reason


def test_an_unreadable_file_only_fails_its_own_row():
    _result, rows = _run({"a": 100, "b": 90, "c": 80}, _Matcher(a_c=("strong", 4.0)),
                         broken={"b": OSError("bad header")})
    assert (rows["b"].method, rows["b"].offset) == ("error", None)
    assert not rows["b"].applicable
    assert "bad header" in rows["b"].reason
    assert rows["c"].offset == 4.0


def test_offsets_are_kept_to_milliseconds():
    _result, rows = _run({"a": 100, "b": 90}, _Matcher(a_b=("strong", 0.1 + 0.2)))
    assert rows["b"].offset == 0.3


def test_cancelling_stops_the_comparisons():
    paths = [Path("a.wav"), Path("b.wav")]
    fingerprints = {p: _Fp(p.stem, 10) for p in paths}
    with pytest.raises(Cancelled):
        suggest_offsets(paths, fingerprints, {}, matcher=_Matcher(), should_stop=lambda: True)


# ---------- AlignWorker.run(), with decoding replaced by energies ----------

def _worker(energies: dict, reference: str | None = None, units=("a", "b")):
    def decode(path, stream, should_stop):
        value = energies[path.stem]
        if isinstance(value, Exception):
            raise value
        return value
    worker = AlignWorker([(Path(f"{u}.wav"), [None]) for u in units], {},
                         (Path(f"{reference}.wav"), [None]) if reference else None)
    with patch("align.decode_energy", side_effect=decode):
        worker.run()
    return worker


@pytest.mark.parametrize("value", [OSError("no such file"), np.zeros(60000)],
                         ids=["unreadable", "silent"])
def test_an_unreadable_session_reference_fails_the_run(session, value):
    worker = _worker({"session": value, "a": session.late, "b": session.mix},
                     reference="session")
    assert worker.result is None
    assert "session reference session.wav" in worker.error
    assert "No offsets were changed" in worker.error


def test_an_unreadable_speaker_file_only_fails_its_row(session):
    worker = _worker({"session": session.mix, "a": OSError("bad header"), "b": session.late},
                     reference="session")
    rows = {s.path.stem: s for s in worker.result.suggestions}
    assert rows["a"].method == "error"
    assert rows["b"].offset == pytest.approx(73.45, abs=0.005)


def test_a_silent_speaker_file_gets_a_reason(session):
    worker = _worker({"a": session.mix, "b": np.zeros(60000)})
    rows = {s.path.stem: s for s in worker.result.suggestions}
    assert "no usable audio" in rows["b"].reason


def test_a_file_that_cannot_be_looked_at_is_not_kept(session):
    # These paths don't exist on disk, so there is no telling later whether
    # they changed.
    worker = _worker({"a": session.mix, "b": session.late})
    assert worker.result is not None
    assert worker.fingerprints == {}


def test_nothing_readable_is_an_error():
    worker = _worker({"a": OSError("x"), "b": OSError("y")})
    assert worker.result is None
    assert "None of the files could be read" in worker.error


def test_cancelling_while_reading_the_reference_emits_nothing():
    worker = AlignWorker([(Path("a.wav"), [None])], {}, (Path("session.wav"), [None]))

    def chunks(path, stream, rate):
        yield np.zeros(800)
        worker.stop()
        yield np.zeros(800)
        pytest.fail("decoding went on after cancel")

    with patch("audio.iter_stream_power", side_effect=chunks):
        worker.run()
    assert worker.cancelled
    assert (worker.result, worker.error) == (None, "")


# ---------- The whole path through PyAV: files on disk to a suggestion ----------

def _shaped_noise(energy, seed: int):
    """8 kHz noise whose frame energies are `energy`."""
    rng = np.random.default_rng(seed)
    hop = align.DECODE_RATE // FRAME_RATE
    return rng.standard_normal(energy.size * hop) * np.sqrt(np.repeat(energy, hop))


def _write_wav(path: Path, energy, seed: int, channels: int = 1, channel: int = 0) -> Path:
    """Noise shaped by frame energies, as 8 kHz 16-bit, on one of `channels`."""
    samples = _shaped_noise(energy, seed)
    frames = np.zeros((samples.size, channels))
    frames[:, channel] = samples
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(align.DECODE_RATE)
        out.writeframes((np.clip(frames, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _write_float_wav(path: Path, frames) -> Path:
    """32-bit float WAV (which `wave` can't write): NaN and inf survive in it."""
    import struct
    data = np.asarray(frames, dtype="<f4").tobytes()
    channels = frames.shape[1]
    with open(path, "wb") as out:
        out.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        out.write(b"fmt " + struct.pack("<IHHIIHH", 16, 3, channels, align.DECODE_RATE,
                                        align.DECODE_RATE * 4 * channels, 4 * channels, 32))
        out.write(b"data" + struct.pack("<I", len(data)) + data)
    return path


def _late_session(folder: Path, channels: int = 1, channel: int = 0):
    """A five-minute session mix, and one speaker's file begun 41.37 s in."""
    mix = speech(300, 50) + 0.5 * speech(300, 51) + hiss(300, 52)
    own = speech(300, 50) + hiss(300, 53)
    return (_write_wav(folder / "session.wav", mix, 1),
            _write_wav(folder / "late.wav", own[4137:], 2, channels, channel))


def test_a_second_run_reuses_what_the_first_read(tmp_path):
    reference, late = _late_session(tmp_path)
    first = AlignWorker([(late, [None])], {}, (reference, [None]))
    first.run()
    assert set(first.fingerprints) == {(reference, (None,)), (late, (None,))}
    again = AlignWorker([(late, [None])], {}, (reference, [None]), known=first.fingerprints)
    with patch("align.decode_energy", side_effect=AssertionError("decoded again")):
        again.run()
    assert again.error == ""
    assert again.result.suggestions == first.result.suggestions
    assert again.fingerprints == first.fingerprints


def test_a_file_changed_since_is_read_again(tmp_path):
    reference, late = _late_session(tmp_path)
    first = AlignWorker([(late, [None])], {}, (reference, [None]))
    first.run()
    st = late.stat()
    os.utime(late, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    again = AlignWorker([(late, [None])], {}, (reference, [None]), known=first.fingerprints)
    with patch("align.decode_energy", wraps=align.decode_energy) as decode:
        again.run()
    assert [c.args[0] for c in decode.call_args_list] == [late]
    # Kept against the file as it is now, not as it was.
    assert again.fingerprints[(late, (None,))][0] == (late.stat().st_mtime_ns,
                                                       late.stat().st_size)
    assert again.result.suggestions == first.result.suggestions


def test_other_streams_of_the_same_file_are_read_again(tmp_path):
    reference, late = _late_session(tmp_path)
    first = AlignWorker([(late, [None])], {}, (reference, [None]))
    first.run()
    again = AlignWorker([(late, [0])], {}, (reference, [None]), known=first.fingerprints)
    with patch("align.decode_energy", wraps=align.decode_energy) as decode:
        again.run()
    assert [c.args[:2] for c in decode.call_args_list] == [(late, 0)]


def test_aligns_a_late_file_against_a_session_reference(tmp_path):
    reference, late = _late_session(tmp_path)
    worker = AlignWorker([(late, [None])], {}, (reference, [None]))
    worker.run()
    assert worker.error == ""
    (row,) = worker.result.suggestions
    assert (row.method, row.strength) == ("matched", "strong")
    assert row.offset == pytest.approx(41.37, abs=0.005)


def test_an_edited_short_file_is_not_strong(tmp_path):
    # Three minutes that lost 5 s 50 s in, against the whole recording.
    energy = speech(180, 1, talk=0.8) + hiss(180, 50)
    whole = _write_wav(tmp_path / "whole.wav", energy, 1)
    edited = _write_wav(tmp_path / "edited.wav",
                        np.concatenate([energy[:5000], energy[5500:]]), 2)
    worker = AlignWorker([(whole, [None]), (edited, [None])], {})
    worker.run()
    assert worker.error == ""
    rows = {s.path.stem: s for s in worker.result.suggestions}
    assert rows["edited"].strength == "weak"
    assert "start and end disagree by 5.0 s" in rows["edited"].reason


def test_a_stream_that_began_late_keeps_its_place(tmp_path):
    # One file holding two speakers' streams, the second begun a minute in,
    # against a room recording that heard both. Read from 0:00, the late
    # stream (the first speaker says little) made the whole file look a
    # minute late — strong, with all three parts agreeing — and applying that
    # moved the first speaker's words a minute late too.
    from test_audio import write_mkv

    a, b = speech(360, 70, talk=0.1), speech(300, 71)
    room = a + np.concatenate([np.zeros(60 * FRAME_RATE), b]) + hiss(360, 72)
    reference = _write_wav(tmp_path / "room.wav", room, 1)
    both = write_mkv(tmp_path / "both.mkv",
                     [[(0, _shaped_noise(a + hiss(360, 73), 2)[:, None])],
                      [(60, _shaped_noise(b + hiss(300, 74), 3)[:, None])]],
                     rate=align.DECODE_RATE)
    worker = AlignWorker([(both, [0, 1])], {}, (reference, [None]))
    worker.run()
    assert worker.error == ""
    (row,) = worker.result.suggestions
    assert (row.strength, row.offset) == ("strong", 0.0)


@pytest.mark.parametrize("channels, speaker", [(3, 2), (6, 3)])
def test_every_channel_of_a_multichannel_file_is_heard(tmp_path, channels, speaker):
    # A mono downmix weights the LFE position (the 3rd of three channels, the
    # 4th of six) at zero; that can be someone's microphone.
    reference, late = _late_session(tmp_path, channels, speaker)
    worker = AlignWorker([(late, [0])], {}, (reference, [None]))
    worker.run()
    assert worker.error == ""
    (row,) = worker.result.suggestions
    assert row.strength == "strong"
    assert row.offset == pytest.approx(41.37, abs=0.005)


def test_a_channel_gone_to_nan_leaves_the_others_heard(tmp_path):
    # A filter gone unstable on one channel: NaN from 60 s to the end.
    session = speech(300, 50) + 0.5 * speech(300, 51) + hiss(300, 52)
    own = (speech(300, 50) + hiss(300, 53))[4137:]
    reference = _write_wav(tmp_path / "session.wav", session, 1)
    noise = _shaped_noise(own, 2)
    frames = np.zeros((noise.size, 3))
    frames[:, 0] = noise
    frames[60 * align.DECODE_RATE:, 1] = np.nan
    late = _write_float_wav(tmp_path / "late.wav", frames)
    worker = AlignWorker([(late, [None])], {}, (reference, [None]))
    worker.run()
    assert worker.error == ""
    (row,) = worker.result.suggestions
    assert row.strength == "strong"
    assert row.offset == pytest.approx(41.37, abs=0.005)
