# -*- coding: utf-8 -*-

import io
import math

import numpy as np
import pytest
import soundfile

from audinota.utils import (
    segment_audio_by_count,
    get_audio_duration,
    segment_audio_by_duration,
    segment_audio_at_silences,
    _load_mono,
)

from audinota.tests.audio_files import AudioFileEnum


def _make_wav_stream(samples: np.ndarray, sample_rate: int) -> io.BytesIO:
    buf = io.BytesIO()
    soundfile.write(buf, samples, sample_rate, format="WAV")
    buf.seek(0)
    return buf


def _tone(seconds: float, sample_rate: int = 16000, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds: float, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(sample_rate * seconds), dtype=np.float32)


def test_segment_audio_by_count():
    for audio_file in AudioFileEnum.iter():
        segments = segment_audio_by_count(audio=audio_file.audio, n_seg=4)
        assert len(segments) == 4
        assert all(isinstance(b, (bytes, bytearray)) and len(b) > 0 for b in segments)


def test_get_audio_duration():
    # Each fixture is named with its approximate duration; allow a ±1 minute
    # window so re-encoding the artifacts doesn't break the suite.
    cases = [
        (AudioFileEnum.a01_ai_karen_hao_agi_openai_ai_10min, (9, 11)),
        (AudioFileEnum.a02_stablecoin_15min, (14, 16)),
        (AudioFileEnum.a03_luo_ji_si_wei_kai_hui_60min, (57, 60)),
    ]
    for audio_file, (lower_min, upper_min) in cases:
        duration_s = get_audio_duration(audio_file.audio)
        assert (lower_min * 60) <= duration_s <= (upper_min * 60)


def test_segment_audio_by_duration():
    # Derive the expected count from the actual duration so the test is robust
    # against fixture re-encodes.
    for audio_file in AudioFileEnum.iter():
        duration_s = get_audio_duration(audio_file.audio)
        expected = max(1, math.ceil(duration_s / 60.0))
        segments = segment_audio_by_duration(audio=audio_file.audio, duration=60)
        assert len(segments) == expected


def test_segment_audio_at_silences():
    audio_file = AudioFileEnum.a01_ai_karen_hao_agi_openai_ai_10min
    duration_s = get_audio_duration(audio_file.audio)
    target = 120.0

    segments = segment_audio_at_silences(audio=audio_file.audio, target_duration=target)

    # Expect roughly duration/target segments, with some slack because silence
    # snapping can merge or split spans at boundaries.
    naive = max(1, math.ceil(duration_s / target))
    assert naive - 1 <= len(segments) <= naive + 2
    assert all(len(s) > 0 for s in segments)


# ---------------------------------------------------------------------------
# Synthetic-signal tests for the new silence-aligned chunker.
#
# These tests do not depend on the downloaded audio fixtures and complete in
# milliseconds. They lock in the correctness invariant that motivates the
# function existing in the first place: cut points must fall inside silences,
# not across speech.
# ---------------------------------------------------------------------------


def test_segment_audio_at_silences_cuts_land_in_silence():
    """Cuts must fall inside the silent regions, never across a tone."""
    sample_rate = 16000
    signal = np.concatenate([
        _tone(5.0, sample_rate),      # 0-5s: tone
        _silence(1.0, sample_rate),   # 5-6s: silence
        _tone(5.0, sample_rate),      # 6-11s: tone
        _silence(1.0, sample_rate),   # 11-12s: silence
        _tone(5.0, sample_rate),      # 12-17s: tone
    ])
    stream = _make_wav_stream(signal, sample_rate)

    chunks = segment_audio_at_silences(audio=stream, target_duration=6.0)
    assert len(chunks) >= 2  # signal is 17s, target 6s -> at least 2 cuts

    # Reconstruct the cut sample positions by walking through chunk durations.
    cut_positions = []
    cursor = 0
    for blob in chunks[:-1]:
        info = soundfile.info(io.BytesIO(blob))
        cursor += int(info.duration * sample_rate)
        cut_positions.append(cursor)

    # Every cut must land in a quiet window (RMS near zero in surrounding samples).
    quiet_threshold = 0.05
    for cut in cut_positions:
        window = signal[max(0, cut - 100): cut + 100]
        rms = float(np.sqrt(np.mean(window * window)))
        assert rms < quiet_threshold, (
            f"cut at sample {cut} ({cut/sample_rate:.2f}s) "
            f"is not in a silence: RMS={rms:.4f}"
        )


def test_segment_audio_at_silences_short_input_returns_single_chunk():
    """An input shorter than target_duration is returned as one chunk."""
    sample_rate = 16000
    signal = _tone(2.0, sample_rate)
    stream = _make_wav_stream(signal, sample_rate)
    chunks = segment_audio_at_silences(audio=stream, target_duration=10.0)
    assert len(chunks) == 1


def test_segment_audio_at_silences_rejects_nonpositive_duration():
    sample_rate = 16000
    stream = _make_wav_stream(_tone(1.0, sample_rate), sample_rate)
    with pytest.raises(ValueError):
        segment_audio_at_silences(audio=stream, target_duration=0)
    stream.seek(0)
    with pytest.raises(ValueError):
        segment_audio_at_silences(audio=stream, target_duration=-5.0)


def test_segment_audio_by_count_rejects_zero():
    sample_rate = 16000
    stream = _make_wav_stream(_tone(1.0, sample_rate), sample_rate)
    with pytest.raises(ValueError):
        segment_audio_by_count(audio=stream, n_seg=0)


def test_load_mono_downmixes_stereo():
    """Stereo input must be averaged to a single channel."""
    sample_rate = 16000
    left = _tone(1.0, sample_rate, freq=440.0)
    right = _tone(1.0, sample_rate, freq=880.0)
    stereo = np.stack([left, right], axis=1)
    assert stereo.shape == (sample_rate, 2)

    stream = _make_wav_stream(stereo, sample_rate)
    mono, sr = _load_mono(stream)

    assert sr == sample_rate
    assert mono.ndim == 1
    assert mono.shape == (sample_rate,)

    # WAV round-trip is lossy (PCM_16 quantization), so allow modest tolerance.
    expected = ((left + right) / 2.0).astype(np.float32)
    np.testing.assert_allclose(mono, expected, atol=1e-3)


if __name__ == "__main__":
    from audinota.tests import run_cov_test

    run_cov_test(
        __file__,
        "audinota.utils",
        preview=False,
    )
