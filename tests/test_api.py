# -*- coding: utf-8 -*-

"""
Smoke test for the public API surface.

Verifies that the symbols promised by README and the changelog are actually
importable from ``audinota.api``. This catches accidental renames before they
ship to PyPI.
"""

import inspect

from audinota import api


EXPECTED_CALLABLES = (
    "transcribe_audio",
    "transcribe_audio_in_parallel",
    "segment_audio_by_count",
    "segment_audio_by_duration",
    "segment_audio_at_silences",
    "get_audio_duration",
)


def test_public_api_surface():
    for name in EXPECTED_CALLABLES:
        attr = getattr(api, name, None)
        assert attr is not None, f"audinota.api missing public symbol: {name}"
        assert callable(attr), f"audinota.api.{name} is not callable"

    # TranscribeAudioResult is the documented return shape.
    assert inspect.isclass(api.TranscribeAudioResult)


def test_transcribe_in_parallel_signature():
    sig = inspect.signature(api.transcribe_audio_in_parallel)
    # README and CLI rely on these knobs being exposed.
    expected = {
        "audio",
        "seg_duration",
        "model_size",
        "device",
        "compute_type",
        "cpu_threads",
        "batch_size",
        "language",
        "task",
        "vad_filter",
        "min_silence_duration_ms",
    }
    missing = expected - set(sig.parameters)
    assert not missing, f"transcribe_audio_in_parallel missing params: {missing}"


def test_transcribe_audio_signature():
    sig = inspect.signature(api.transcribe_audio)
    expected = {
        "audio",
        "model_size",
        "device",
        "compute_type",
        "cpu_threads",
        "num_workers",
        "batch_size",
        "language",
        "task",
        "vad_filter",
        "min_silence_duration_ms",
        "model",
    }
    missing = expected - set(sig.parameters)
    assert not missing, f"transcribe_audio missing params: {missing}"


if __name__ == "__main__":
    from audinota.tests import run_cov_test

    run_cov_test(
        __file__,
        "audinota.api",
        preview=False,
    )
