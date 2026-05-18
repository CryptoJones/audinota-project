.. _release_history:

Release and Version History
==============================================================================


x.y.z (Backlog)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Features and Improvements**

**Minor Improvements**

**Bugfixes**

**Miscellaneous**


0.2.0 (2026-05-18)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Breaking Changes**

- ``transcribe_audio`` now takes flat keyword arguments (``model_size``,
  ``device``, ``compute_type``, ``language``, ``batch_size``, ``vad_filter``,
  ...) instead of the previous ``whisper_model_kwargs`` and ``transcribe_kwargs``
  dict bundles. Callers that constructed those dicts must unpack them.
- ``transcribe_audio_in_parallel`` no longer fans out to ``mpire`` worker
  processes by default. The whole stream is fed to a single
  ``BatchedInferencePipeline``, which avoids the per-chunk model-load overhead
  that previously made the function slower than serial on small VMs. The
  ``n_jobs`` parameter is accepted for signature compatibility but ignored.

**Features and Improvements**

- Exposed the full model and inference parameter set on both the Python API
  (``transcribe_audio``, ``transcribe_audio_in_parallel``) and the CLI
  (``--model_size``, ``--device``, ``--compute_type``, ``--language``,
  ``--batch_size``, ``--vad_filter``, ``--seg_duration``, ``--cpu_threads``,
  ``--quiet``). The library is now usable for non-default models.
- New :func:`audinota.utils.segment_audio_at_silences` -- duration-targeted
  chunking that snaps each cut to the nearest silence inside a search window.
  Chunk boundaries land between utterances rather than across words.
- The optional pre-chunking path of ``transcribe_audio_in_parallel`` (enabled
  via ``seg_duration=...``) uses the new silence-aligned chunker and reuses a
  single loaded ``WhisperModel`` across chunks.
- Added ``py.typed`` marker so downstream type-checkers see the inline hints.

**Bugfixes**

- CLI now exits non-zero (1 for ``FileExistsError``, 2 for missing input) so
  scripting against ``audinota transcribe`` can detect failures. The previous
  implementation printed an error and exited 0.
- ``tests/vendor/pytest_cov_helper.py`` test runners now propagate pytest's
  exit code. ``make test`` previously reported success even when pytest
  failed.
- Fixed ``classifier`` -> ``classifiers`` typo in ``pyproject.toml``. The
  project was shipping with zero PyPI classifiers because the wrong key was
  silently ignored. Added relevant Topic and Audience classifiers.
- ``audinota/tests/audio_files.py`` now passes a 60s timeout to
  ``urllib.request.urlopen`` so a hung GitHub release server cannot stall a
  CI run indefinitely.

**Miscellaneous**

- Dropped the unused ``librosa`` runtime dependency (and its transitive
  graph: numba, scikit-learn, scipy, joblib, soxr, audioread, pooch,
  lazy-loader, msgpack, decorator, llvmlite). Also dropped ``mpire`` since
  the multi-process pool path was removed. Installs are dramatically smaller.
- All ``requirements*.txt`` files regenerated from the trimmed lockfile.
- Excluded ``audinota/vendor/*`` from the wheel -- the vendored pytest helper
  no longer ships to end users.
- ``USING_COVERAGE`` in the GitHub Actions workflow now matches the actual
  matrix (3.10, 3.11, 3.12) instead of listing untested versions.
- Bumped Read the Docs build image from the EOL ``ubuntu-20.04`` to
  ``ubuntu-22.04``.
- ``docs/source/conf.py`` no longer uses the deprecated
  ``datetime.utcnow()`` and no longer swallows ``KeyboardInterrupt`` via a
  bare ``except:``.
- Trimmed README and release-history of unsupported marketing claims
  ("120x cheaper than AWS Transcribe", "smart segmentation",
  "multi-format support including MP4/M4A through libsndfile"). Format
  support is now hedged on what ``faster-whisper`` / ``ffmpeg`` can actually
  decode; ``BatchedInferencePipeline`` is described accurately.
- Test assertions added/tightened: ``test_api`` now verifies the public
  symbol set and signature; ``test_transcribe`` asserts the output is
  non-empty and exceeds a sanity-length threshold; ``test_utils`` derives
  expected chunk counts from actual audio duration instead of hard-coded
  fixture-specific magic numbers.


0.1.1 (2024-08-11)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Features and Improvements**

- Initial public release.
- Wraps ``faster-whisper`` with a parallel pre-chunking pipeline and a
  ``fire``-based CLI.
- Provides ``transcribe_audio`` and ``transcribe_audio_in_parallel``, plus
  ``segment_audio_by_count`` / ``segment_audio_by_duration`` utilities.
- CLI entry point ``audinota transcribe --input=...`` with directory output,
  custom file naming, and conflict-resolution numbering.

.. note::

    The original 0.1.1 release notes claimed automatic MP4/M4A support, smart
    speech-aware segmentation, and "120x cheaper than AWS Transcribe"
    economics. Those claims overstated the implementation: chunking was
    content-blind equal-sample slicing, format support depended entirely on
    whatever libsndfile understood (no MP4/M4A), and the cost comparison was
    aspirational with no benchmark. See the 0.2.0 entry above for the
    cleanup.
