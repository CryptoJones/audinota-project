# -*- coding: utf-8 -*-

"""
Integration test for the high-level transcription entry point.

This test downloads a fixture audio file from the project release artifacts
on first run and runs real faster-whisper inference. It is slow but it is
the only check that the pipeline end-to-end actually produces speech text
rather than failing silently.
"""

from audinota.transcribe import transcribe_audio_in_parallel
from audinota.tests.audio_files import AudioFileEnum


def test_transcribe_audio_in_parallel():
    text = transcribe_audio_in_parallel(
        audio=AudioFileEnum.a01_ai_karen_hao_agi_openai_ai_10min.audio,
    )
    # The fixture is a ~10 minute spoken-word clip; ``tiny`` Whisper on this
    # input reliably produces well over a few hundred characters of English
    # text. A near-empty result almost always means the pipeline silently
    # broke (VAD ate everything, decoder timed out, etc.).
    assert isinstance(text, str)
    assert len(text) > 500, f"Transcription suspiciously short: {len(text)} chars"
    # Strip whitespace -- a result that is whitespace-only would otherwise sneak
    # past the length check.
    assert text.strip(), "Transcription is whitespace-only"


def test_transcribe_audio_in_parallel_silence_chunked():
    """Same input, exercised through the silence-aligned pre-chunk path."""
    text = transcribe_audio_in_parallel(
        audio=AudioFileEnum.a01_ai_karen_hao_agi_openai_ai_10min.audio,
        seg_duration=120,
    )
    assert isinstance(text, str)
    assert len(text) > 500, f"Chunked transcription suspiciously short: {len(text)} chars"


if __name__ == "__main__":
    from audinota.tests import run_cov_test

    run_cov_test(
        __file__,
        "audinota.transcribe",
        preview=False,
    )
