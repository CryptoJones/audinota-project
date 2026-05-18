# -*- coding: utf-8 -*-

"""
Transcription pipeline built on faster-whisper.

The library exposes two entry points:

- ``transcribe_audio`` -- run one audio stream through a Whisper model
  (loading the model lazily, or reusing one passed in).
- ``transcribe_audio_in_parallel`` -- the high-level convenience wrapper
  that loads a model once and either streams the whole file through
  ``BatchedInferencePipeline`` (default) or pre-chunks at silences for
  very long inputs that won't fit in memory.

Faster-whisper's ``BatchedInferencePipeline`` already performs VAD-aware
chunking and batched decoding internally, so callers rarely need to
pre-chunk. The ``seg_duration`` knob exists for the multi-hour archive
case where loading the entire decoded waveform into RAM is undesirable.
"""

import typing as T
import io
import dataclasses

from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.transcribe import TranscriptionInfo, Segment

from .utils import segment_audio_at_silences


@dataclasses.dataclass
class TranscribeAudioResult:
    text: str
    segments: list[Segment]
    info: TranscriptionInfo


def _load_model(
    model_size: str,
    device: str,
    compute_type: str,
    cpu_threads: int,
    num_workers: int,
) -> WhisperModel:
    return WhisperModel(
        model_size_or_path=model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
    )


def transcribe_audio(
    audio: T.BinaryIO,
    model_size: str = "tiny",
    device: str = "cpu",
    compute_type: str = "int8",
    cpu_threads: int = 0,
    num_workers: int = 1,
    batch_size: int = 16,
    language: T.Optional[str] = None,
    task: str = "transcribe",
    vad_filter: bool = True,
    min_silence_duration_ms: int = 500,
    model: T.Optional[WhisperModel] = None,
) -> TranscribeAudioResult:
    """
    Transcribe an audio stream and return text plus structured segments.

    :param audio: Audio data as a binary stream.
    :param model_size: Whisper model name ("tiny", "base", "small", "medium",
        "large-v2", "large-v3") or a local model path. Ignored when ``model``
        is provided.
    :param device: "cpu", "cuda", or "auto".
    :param compute_type: ctranslate2 compute type ("int8", "float16", ...).
    :param cpu_threads: CPU threads. ``0`` lets faster-whisper choose.
    :param num_workers: Concurrent CTranslate2 workers.
    :param batch_size: Decoder batch size. Higher uses more memory.
    :param language: ISO code (e.g. "en", "zh") or ``None`` to auto-detect.
    :param task: ``"transcribe"`` (default) or ``"translate"`` -- the latter
        transcribes any source language directly into English text.
    :param vad_filter: Run Silero VAD to skip non-speech regions.
    :param min_silence_duration_ms: VAD parameter -- minimum silence span
        that counts as a break.
    :param model: Optional pre-loaded :class:`WhisperModel` to reuse, avoiding
        load overhead when transcribing many streams.
    """
    if model is None:
        model = _load_model(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
    pipeline = BatchedInferencePipeline(model=model)
    segments_iter, info = pipeline.transcribe(
        audio=audio,
        batch_size=batch_size,
        language=language,
        task=task,
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=min_silence_duration_ms),
    )
    segments = list(segments_iter)
    text = "".join(seg.text for seg in segments)
    return TranscribeAudioResult(text=text, segments=segments, info=info)


def transcribe_audio_in_parallel(
    audio: T.BinaryIO,
    seg_duration: T.Optional[float] = None,
    n_jobs: T.Optional[int] = None,  # kept for backwards-compat signature
    model_size: str = "tiny",
    device: str = "cpu",
    compute_type: str = "int8",
    cpu_threads: int = 0,
    batch_size: int = 16,
    language: T.Optional[str] = None,
    task: str = "transcribe",
    vad_filter: bool = True,
    min_silence_duration_ms: int = 500,
) -> str:
    """
    Transcribe an audio stream and return the concatenated text.

    By default the whole stream is fed to faster-whisper's batched inference
    pipeline, which performs VAD-aware chunking and batched decoding inside a
    single model instance. That avoids the model-reload overhead and word
    boundary truncation of a naive pre-chunking strategy.

    Pass ``seg_duration`` to enable silence-aligned pre-chunking for very long
    files that don't fit in memory. Cut points snap to the nearest silence
    inside a search window, so words are never split mid-utterance.

    :param audio: Audio data as a binary stream.
    :param seg_duration: If set, pre-chunk the audio at silences near every
        ``seg_duration`` seconds before transcribing. Default ``None``.
    :param n_jobs: Accepted for backwards compatibility. Ignored -- batched
        inference is intra-model parallel; multi-process parallelism over
        multiple model copies is not worth the load overhead for typical
        inputs.
    :param model_size: Whisper model size or path.
    :param device: "cpu", "cuda", or "auto".
    :param compute_type: ctranslate2 compute type.
    :param cpu_threads: CPU threads. ``0`` lets faster-whisper choose.
    :param batch_size: Decoder batch size.
    :param language: ISO language code or ``None`` to auto-detect.
    :param task: ``"transcribe"`` (default) or ``"translate"`` -- the latter
        transcribes any source language directly into English text.
    :param vad_filter: Run Silero VAD to skip non-speech regions.
    :param min_silence_duration_ms: VAD silence threshold.
    """
    del n_jobs  # see docstring

    model = _load_model(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=1,
    )

    common = dict(
        model=model,
        batch_size=batch_size,
        language=language,
        task=task,
        vad_filter=vad_filter,
        min_silence_duration_ms=min_silence_duration_ms,
    )

    if seg_duration is None:
        return transcribe_audio(audio=audio, **common).text

    chunks = segment_audio_at_silences(audio=audio, target_duration=seg_duration)
    parts = [transcribe_audio(audio=io.BytesIO(chunk), **common).text for chunk in chunks]
    return "".join(parts)
