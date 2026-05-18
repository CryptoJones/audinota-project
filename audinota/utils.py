# -*- coding: utf-8 -*-

"""
Audio segmentation and metadata utilities.

Uses ``soundfile`` for I/O to avoid the deprecated ``audioread`` path.

Three segmentation strategies are provided:

- :func:`segment_audio_by_count`     -- equal-sample slicing into ``n`` chunks.
- :func:`segment_audio_by_duration`  -- equal-sample slicing targeting a per-chunk
  duration. Both of the above are content-blind and will cut across words; use
  them only for non-speech audio or when you control the boundaries.
- :func:`segment_audio_at_silences`  -- duration-targeted slicing that snaps
  each cut to the nearest silence inside a search window. This is the right
  choice for speech, because boundaries land between utterances rather than
  mid-word.
"""

import typing as T
import io
import math

import numpy as np
import soundfile


def _to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    soundfile.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


def _load_mono(audio: T.BinaryIO) -> tuple[np.ndarray, int]:
    audio.seek(0)
    data, sample_rate = soundfile.read(audio, always_2d=False)
    audio.seek(0)
    if data.ndim > 1:
        # Downmix to mono for silence detection -- transcription quality is
        # not meaningfully affected and the analysis is much cheaper.
        data = data.mean(axis=1)
    return data.astype(np.float32, copy=False), sample_rate


def segment_audio_by_count(
    audio: T.BinaryIO,
    n_seg: int,
) -> list[bytes]:
    """
    Split audio into a fixed number of segments with equal sample counts.

    The last segment absorbs any remainder so no samples are dropped. This
    function is content-blind: cuts land at arbitrary sample offsets, so it
    is unsuitable for speech audio. Use :func:`segment_audio_at_silences` for
    that case.

    :param audio: Audio data as a binary stream (e.g. ``io.BytesIO``).
    :param n_seg: Number of segments to create (positive integer).

    :return: List of WAV-encoded segment bytes.
    """
    if n_seg < 1:
        raise ValueError(f"n_seg must be >= 1, got {n_seg}")

    audio.seek(0)
    audio_data, sample_rate = soundfile.read(audio)
    audio.seek(0)

    total_samples = len(audio_data)
    samples_per_segment = total_samples // n_seg

    segments = []
    for segment_idx in range(n_seg):
        start_sample = segment_idx * samples_per_segment
        end_sample = (
            total_samples
            if segment_idx == n_seg - 1
            else (segment_idx + 1) * samples_per_segment
        )
        segments.append(_to_wav_bytes(audio_data[start_sample:end_sample], sample_rate))

    return segments


def get_audio_duration(audio: T.BinaryIO) -> float:
    """
    Return audio duration in seconds from the file header without decoding samples.

    :param audio: Audio data as a binary stream.
    :return: Duration in seconds.
    """
    audio.seek(0)
    audio_info = soundfile.info(audio)
    audio.seek(0)
    return audio_info.duration


def segment_audio_by_duration(
    audio: T.BinaryIO,
    duration: float,
) -> list[bytes]:
    """
    Split audio into equal-count chunks sized to approximately ``duration`` seconds.

    The exact chunk length is ``total_duration / ceil(total_duration / duration)``,
    so all chunks have identical length. This is content-blind -- see
    :func:`segment_audio_at_silences` for a speech-safe alternative.

    :param audio: Audio data as a binary stream.
    :param duration: Target duration per segment in seconds.
    """
    total_duration = get_audio_duration(audio)
    num_segments = max(1, math.ceil(total_duration / duration))
    return segment_audio_by_count(audio, num_segments)


def _silence_centers(
    samples: np.ndarray,
    sample_rate: int,
    window_ms: float = 30.0,
    rms_floor_ratio: float = 0.02,
    min_silence_ms: float = 200.0,
) -> np.ndarray:
    """Return sample indices in the middle of each silent run."""
    win = max(1, int(sample_rate * window_ms / 1000))
    n_windows = len(samples) // win
    if n_windows == 0:
        return np.empty(0, dtype=np.int64)

    framed = samples[: n_windows * win].reshape(n_windows, win)
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    peak = rms.max()
    if peak == 0:
        return np.empty(0, dtype=np.int64)
    silent = rms < (peak * rms_floor_ratio)

    min_silence_windows = max(1, int(min_silence_ms / window_ms))
    centers: list[int] = []
    i = 0
    n = len(silent)
    while i < n:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < n and silent[j]:
            j += 1
        if (j - i) >= min_silence_windows:
            center_window = (i + j) // 2
            centers.append(center_window * win + win // 2)
        i = j
    return np.array(centers, dtype=np.int64)


def segment_audio_at_silences(
    audio: T.BinaryIO,
    target_duration: float,
    search_fraction: float = 0.25,
    min_silence_ms: float = 200.0,
) -> list[bytes]:
    """
    Split audio into chunks of roughly ``target_duration`` seconds, snapping
    each cut to the nearest silence inside a search window.

    The output is speech-safe: chunk boundaries land between utterances rather
    than across them. If no silence is found inside the search window for a
    particular boundary, the cut falls back to the exact target offset.

    :param audio: Audio data as a binary stream.
    :param target_duration: Approximate chunk duration in seconds.
    :param search_fraction: Fraction of ``target_duration`` to search on each
        side for a silence (e.g. ``0.25`` searches ±25%).
    :param min_silence_ms: Minimum silence length that counts as a cut point.

    :return: List of WAV-encoded segment bytes.
    """
    if target_duration <= 0:
        raise ValueError(f"target_duration must be > 0, got {target_duration}")

    samples, sample_rate = _load_mono(audio)
    total = len(samples)
    target_samples = int(target_duration * sample_rate)
    if total <= target_samples or target_samples <= 0:
        return [_to_wav_bytes(samples, sample_rate)]

    silence_idx = _silence_centers(samples, sample_rate, min_silence_ms=min_silence_ms)
    search = int(target_samples * search_fraction)

    cuts = [0]
    pos = target_samples
    while pos < total:
        if silence_idx.size:
            window = silence_idx[(silence_idx >= pos - search) & (silence_idx <= pos + search)]
            if window.size:
                pos = int(window[np.argmin(np.abs(window - pos))])
        cuts.append(pos)
        pos += target_samples
    cuts.append(total)

    # Drop any zero-length spans that could arise if a snap collapsed two boundaries.
    parts = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        if end > start:
            parts.append(_to_wav_bytes(samples[start:end], sample_rate))
    return parts
