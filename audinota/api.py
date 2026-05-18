# -*- coding: utf-8 -*-

"""
Public API surface for audinota.

Import directly from this module:

    from audinota.api import transcribe_audio_in_parallel
"""

from .utils import segment_audio_by_count
from .utils import segment_audio_by_duration
from .utils import segment_audio_at_silences
from .utils import get_audio_duration
from .transcribe import transcribe_audio
from .transcribe import transcribe_audio_in_parallel
from .transcribe import TranscribeAudioResult

__all__ = [
    "segment_audio_by_count",
    "segment_audio_by_duration",
    "segment_audio_at_silences",
    "get_audio_duration",
    "transcribe_audio",
    "transcribe_audio_in_parallel",
    "TranscribeAudioResult",
]
