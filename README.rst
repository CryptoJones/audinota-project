
.. image:: https://readthedocs.org/projects/audinota/badge/?version=latest
    :target: https://audinota.readthedocs.io/en/latest/
    :alt: Documentation Status

.. image:: https://github.com/MacHu-GWU/audinota-project/actions/workflows/main.yml/badge.svg
    :target: https://github.com/MacHu-GWU/audinota-project/actions?query=workflow:CI

.. image:: https://codecov.io/gh/MacHu-GWU/audinota-project/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/MacHu-GWU/audinota-project

.. image:: https://img.shields.io/pypi/v/audinota.svg
    :target: https://pypi.python.org/pypi/audinota

.. image:: https://img.shields.io/pypi/l/audinota.svg
    :target: https://pypi.python.org/pypi/audinota

.. image:: https://img.shields.io/pypi/pyversions/audinota.svg
    :target: https://pypi.python.org/pypi/audinota

.. image:: https://img.shields.io/badge/✍️_Release_History!--None.svg?style=social&logo=github
    :target: https://github.com/MacHu-GWU/audinota-project/blob/main/release-history.rst

.. image:: https://img.shields.io/badge/⭐_Star_me_on_GitHub!--None.svg?style=social&logo=github
    :target: https://github.com/MacHu-GWU/audinota-project

------

.. image:: https://img.shields.io/badge/Link-API-blue.svg
    :target: https://audinota.readthedocs.io/en/latest/py-modindex.html

.. image:: https://img.shields.io/badge/Link-Install-blue.svg
    :target: `install`_

.. image:: https://img.shields.io/badge/Link-GitHub-blue.svg
    :target: https://github.com/MacHu-GWU/audinota-project

.. image:: https://img.shields.io/badge/Link-Submit_Issue-blue.svg
    :target: https://github.com/MacHu-GWU/audinota-project/issues

.. image:: https://img.shields.io/badge/Link-Request_Feature-blue.svg
    :target: https://github.com/MacHu-GWU/audinota-project/issues

.. image:: https://img.shields.io/badge/Link-Download-blue.svg
    :target: https://pypi.org/pypi/audinota#files


Welcome to ``audinota`` Documentation
==============================================================================
.. image:: https://audinota.readthedocs.io/en/latest/_static/audinota-logo.png
    :target: https://audinota.readthedocs.io/en/latest/

**Audinota** (Latin for "taking notes from audio") is a small Python library
that wraps `faster-whisper <https://github.com/SYSTRAN/faster-whisper>`_ with
a friendly API and CLI for audio-to-text transcription.

The library focuses on plain text output -- no subtitles, no timestamp
management, no audio editing. Inputs go through faster-whisper's
``BatchedInferencePipeline``, which performs VAD-aware chunking and batched
decoding inside a single model instance. For very long files, an optional
silence-aligned pre-chunker is available so that boundaries land between
utterances rather than across words.

Audinota is a thin convenience layer; quality and speed are inherited from
the underlying ``faster-whisper`` / ``ctranslate2`` runtime and the Whisper
model you choose. ``tiny`` and ``base`` are fast and small enough for cheap
CPU-only deployments (e.g. AWS Lambda); ``medium`` and ``large-v3`` are the
right choice when transcription quality matters more than latency.


Quick Start
------------------------------------------------------------------------------
Audinota makes audio transcription incredibly simple with just a few lines of code:

.. code-block:: python

    import io
    from pathlib import Path
    from audinota.api import transcribe_audio_in_parallel

    # Transcribe any audio file to text
    text = transcribe_audio_in_parallel(
        audio=io.BytesIO(Path("podcast_episode.mp3").read_bytes()),
    )
    print(text)

**What happens under the hood:**

1. **Format support**: Whatever ``faster-whisper`` / ``ffmpeg`` can decode --
   WAV, FLAC, OGG, and (on systems with ffmpeg available) MP3, MP4, M4A, and
   other compressed containers.
2. **Language detection**: Whisper auto-detects the spoken language unless
   you pass ``language="en"``, ``language="zh"``, etc.
3. **VAD-aware chunking**: ``BatchedInferencePipeline`` runs Silero VAD on
   the input and feeds speech-only spans to the decoder in batches.
4. **Optional silence-aligned pre-chunking**: pass ``seg_duration=120`` to
   split a multi-hour file into ~2-minute speech-aligned chunks before
   transcription, so peak memory stays bounded.
5. **Text assembly**: chunk texts are concatenated; segment objects are also
   exposed via :class:`TranscribeAudioResult` if you want timestamps or
   per-segment metadata.


Command Line Interface
------------------------------------------------------------------------------
Audinota provides a powerful command-line interface for easy audio transcription without writing code:

Basic Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: console

    # Simple transcription - output saved next to input file
    $ audinota transcribe --input="podcast.mp3"

    # Specify output directory
    $ audinota transcribe --input="lecture.mp4" --output="/path/to/transcripts/"

    # Specify exact output file
    $ audinota transcribe --input="interview.wav" --output="result.txt"

    # Overwrite existing files
    $ audinota transcribe --input="audio.m4a" --output="existing.txt" --overwrite

Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**--input** (required)
    Path to the input audio file. Anything ``faster-whisper`` / ``ffmpeg`` can
    decode is accepted -- WAV, FLAC, OGG, and (with ffmpeg installed) the
    common compressed formats MP3, MP4, M4A, and so on.

**--output** (optional)
    Controls where the transcription is saved:

    - **Not specified**: Creates a .txt file next to the input file
      
      .. code-block:: console
      
          $ audinota transcribe --input="podcast.mp3"
          # Creates: podcast.txt

    - **Directory path**: Creates a .txt file in the specified directory
      
      .. code-block:: console
      
          $ audinota transcribe --input="podcast.mp3" --output="/transcripts/"
          # Creates: /transcripts/podcast.txt

    - **File path**: Uses the exact specified file path
      
      .. code-block:: console
      
          $ audinota transcribe --input="podcast.mp3" --output="my_transcript.txt"
          # Creates: my_transcript.txt

**--overwrite** (optional, default: False)
    Boolean flag that controls file overwriting behavior:

    - **False** (default): If output file exists, shows error and stops
    - **True**: Overwrites existing output files without asking

    .. note::
        This only applies when --output specifies a file path. Directory outputs use automatic numbering instead.

Model and inference options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**--model_size** (default: ``"tiny"``)
    Whisper model name (``"tiny"``, ``"base"``, ``"small"``, ``"medium"``,
    ``"large-v2"``, ``"large-v3"``) or a local model path.

**--device** (default: ``"cpu"``)
    ``"cpu"``, ``"cuda"``, or ``"auto"``.

**--compute_type** (default: ``"int8"``)
    ctranslate2 compute type (``"int8"``, ``"float16"``, ``"float32"``, ...).
    Defaults are CPU-friendly; on GPU use ``"float16"``.

**--cpu_threads** (default: ``0``)
    Number of CPU threads. ``0`` lets faster-whisper choose.

**--batch_size** (default: ``16``)
    Decoder batch size. Higher uses more memory; lower is safer on tiny VMs.

**--language** (default: auto-detect)
    ISO language code (``"en"``, ``"zh"``, ``"ja"``, ...). Skipping detection
    is a meaningful speedup on short files.

**--task** (default: ``"transcribe"``)
    ``"transcribe"`` (same-language speech-to-text) or ``"translate"``
    (any-source-language speech-to-English-text).

**--vad_filter** (default: ``True``)
    Run Silero VAD to skip non-speech regions before decoding.

**--min_silence_duration_ms** (default: ``500``)
    Silero VAD silence threshold in milliseconds.

**--seg_duration** (default: ``None``)
    If set, pre-chunk the audio at silences near every ``seg_duration``
    seconds. Use for multi-hour files where loading the entire decoded
    waveform into RAM is undesirable. The default path (``None``) feeds the
    whole file to batched inference, which is faster for normal inputs.

**--quiet** (default: ``False``)
    Suppress progress output.

File Conflict Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Audinota intelligently handles file name conflicts:

**Automatic Numbering**
    When output goes to a directory and files already exist:

    .. code-block:: console

        $ audinota transcribe --input="audio.mp3" --output="/transcripts/"
        # If /transcripts/audio.txt exists, creates /transcripts/audio_01.txt
        # If both exist, creates /transcripts/audio_02.txt, etc.

**File Path Conflicts**
    When --output specifies an existing file:

    .. code-block:: console

        $ audinota transcribe --input="audio.mp3" --output="existing.txt"
        # Error: Output file 'existing.txt' already exists. Use --overwrite

        $ audinota transcribe --input="audio.mp3" --output="existing.txt" --overwrite
        # Overwrites existing.txt

Real-World Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: console

    # Transcribe a podcast episode
    $ audinota transcribe --input="episode_042.mp3"
    # Output: episode_042.txt

    # Batch processing to organized directory
    $ mkdir transcripts
    $ audinota transcribe --input="meeting_2024_01.m4a" --output="transcripts/"
    $ audinota transcribe --input="meeting_2024_02.m4a" --output="transcripts/"
    # Output: transcripts/meeting_2024_01.txt, transcripts/meeting_2024_02.txt

    # Process lecture with custom naming
    $ audinota transcribe --input="cs101_lecture.mp4" --output="notes/week1_lecture.txt"

    # Replace previous transcription
    $ audinota transcribe --input="revised_audio.wav" --output="final_transcript.txt" --overwrite

Behavior notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Batched decoding**: ``BatchedInferencePipeline`` runs multiple speech
  segments through the decoder in parallel inside a single Whisper model.
- **VAD-aware chunking**: Silero VAD is used to skip non-speech regions and
  align internal chunks to silence.
- **Single-model design**: the default path loads one model and reuses it for
  the whole file. There is no multi-process model-per-chunk fan-out by
  design -- the model load cost dwarfs the benefit on typical inputs.

.. code-block:: console

    $ audinota transcribe --input=long_podcast.mp3
    Transcribing audio file: long_podcast.mp3
    Output will be saved to: long_podcast.txt
    Loading audio data...
    Transcribing (model=tiny, device=cpu, compute_type=int8)...
    Saving transcription...
    Transcription complete: file:///path/to/long_podcast.txt
    Text length: 15847 characters


.. _install:

Install
------------------------------------------------------------------------------

``audinota`` is released on PyPI, so all you need is to:

.. code-block:: console

    $ pip install audinota

To upgrade to latest version:

.. code-block:: console

    $ pip install --upgrade audinota
