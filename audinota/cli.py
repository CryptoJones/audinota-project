# -*- coding: utf-8 -*-

"""
Command Line Interface for Audinota.

Built on Python Fire. Exposes the full transcription knob set on the CLI so
that callers can pick a model size, language, device, and chunking strategy
without forking the library.
"""

import io
import sys
from pathlib import Path
from typing import Optional

import fire

from audinota.api import transcribe_audio_in_parallel


def resolve_output_path(
    input_path: str,
    output_path: Optional[str],
    overwrite: bool,
) -> Path:
    """
    Resolve the final output file path.

    Three cases:

    1. ``output_path`` is None -- write a ``.txt`` next to the input file,
       auto-numbering on conflict.
    2. ``output_path`` is an existing directory -- write a ``.txt`` named after
       the input stem inside it, auto-numbering on conflict.
    3. ``output_path`` is a file path -- use it verbatim; raise
       :class:`FileExistsError` if it exists and ``overwrite`` is False.

    :raises FileExistsError: when case 3 hits an existing file without overwrite.
    """
    input_file = Path(input_path).absolute()

    if output_path is None:
        return _find_unique_filename(input_file.parent, input_file.stem, ".txt")

    output_path_obj = Path(output_path)

    if output_path_obj.is_dir():
        return _find_unique_filename(output_path_obj, input_file.stem, ".txt")

    if output_path_obj.exists() and not overwrite:
        raise FileExistsError(
            f"Output file '{output_path_obj}' already exists. "
            "Use --overwrite to overwrite the existing file."
        )

    return output_path_obj


def _find_unique_filename(
    directory: Path,
    base_name: str,
    extension: str,
) -> Path:
    candidate = directory / f"{base_name}{extension}"
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = directory / f"{base_name}_{counter:02d}{extension}"
        if not candidate.exists():
            return candidate
        counter += 1
        if counter > 999:
            raise RuntimeError("Cannot find unique filename after 999 attempts")


class AudioTranscriber:
    """Command-line interface for audinota."""

    def transcribe(
        self,
        input: str,
        output: Optional[str] = None,
        overwrite: bool = False,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        batch_size: int = 16,
        language: Optional[str] = None,
        task: str = "transcribe",
        vad_filter: bool = True,
        min_silence_duration_ms: int = 500,
        seg_duration: Optional[float] = None,
        quiet: bool = False,
    ) -> None:
        """
        Transcribe an audio file to text.

        :param input: Path to the input audio file (required).
        :param output: Output file or directory path. Default: ``.txt`` next to input.
        :param overwrite: Overwrite an existing output file. Default False.
        :param model_size: Whisper model name or local path. Default "tiny".
        :param device: "cpu", "cuda", or "auto". Default "cpu".
        :param compute_type: ctranslate2 compute type. Default "int8".
        :param cpu_threads: CPU threads (0 = faster-whisper picks). Default 0.
        :param batch_size: Decoder batch size. Default 16.
        :param language: ISO code (e.g. "en", "zh") or None to auto-detect.
        :param task: "transcribe" (default) or "translate" -- the latter
            transcribes any source language directly into English text.
        :param vad_filter: Enable Silero VAD silence skipping. Default True.
        :param min_silence_duration_ms: VAD silence threshold in milliseconds.
            Default 500.
        :param seg_duration: Optional pre-chunk size in seconds (silence-aligned).
            Default None -- feed the whole file to batched inference.
        :param quiet: Suppress progress output. Default False.

        Examples::

            audinota transcribe --input=podcast.mp3
            audinota transcribe --input=lecture.mp4 --output=transcripts/ --model_size=base
            audinota transcribe --input=zh.wav --language=zh --vad_filter=false
            audinota transcribe --input=spanish.mp3 --task=translate
        """
        def say(msg: str) -> None:
            if not quiet:
                print(msg)

        input_path = Path(input)
        if not input_path.exists():
            print(f"Input audio file not found: {input}", file=sys.stderr)
            sys.exit(2)
        if not input_path.is_file():
            print(f"Input path is not a file: {input}", file=sys.stderr)
            sys.exit(2)

        say(f"Transcribing audio file: {input_path}")

        try:
            output_path = resolve_output_path(input, output, overwrite)
        except FileExistsError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        say(f"Output will be saved to: {output_path}")

        say("Loading audio data...")
        audio_stream = io.BytesIO(input_path.read_bytes())

        say(f"Transcribing (model={model_size}, device={device}, compute_type={compute_type}, task={task})...")
        transcribed_text = transcribe_audio_in_parallel(
            audio=audio_stream,
            seg_duration=seg_duration,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            batch_size=batch_size,
            language=language,
            task=task,
            vad_filter=vad_filter,
            min_silence_duration_ms=min_silence_duration_ms,
        )

        say("Saving transcription...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(transcribed_text, encoding="utf-8")

        say(f"Transcription complete: {output_path.resolve().as_uri()}")
        say(f"Text length: {len(transcribed_text)} characters")


def main():
    """Entry point for the ``audinota`` console script."""
    fire.Fire(AudioTranscriber)
