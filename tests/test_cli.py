# -*- coding: utf-8 -*-

from unittest import mock

import pytest

from audinota.cli import (
    AudioTranscriber,
    resolve_output_path,
)
from audinota.paths import dir_unit_test

dir_tmp = dir_unit_test / "tmp"
dir_tmp.mkdir(parents=True, exist_ok=True)


def test_resolve_output_path():
    """
    Test the resolve_output_path function with various scenarios.
    """
    # Setup test files and directories
    p_input_audio = dir_tmp / "test_audio.mp3"
    p_input_audio.write_text("fake audio content", encoding="utf-8")

    output_dir = dir_tmp / "output_dir"
    output_dir.mkdir(exist_ok=True)

    # Test Case 1: input output both files, no conflict
    def test_case_1():
        """Test: input output both files, output has no conflict"""
        output_file = dir_tmp / "no_conflict.txt"
        # Ensure the file doesn't exist
        if output_file.exists():
            output_file.unlink()

        result = resolve_output_path(
            input_path=str(p_input_audio), output_path=str(output_file), overwrite=False
        )

        assert result == output_file
        assert result.is_absolute()

    # Test Case 2: input output both files, conflict exists, overwrite=False
    def test_case_2():
        """Test: input output both files, output has conflict, overwrite=False (should raise error)"""
        conflict_file = dir_tmp / "existing_conflict.txt"
        conflict_file.write_text("existing content", encoding="utf-8")

        with pytest.raises(FileExistsError) as exc_info:
            resolve_output_path(
                input_path=str(p_input_audio),
                output_path=str(conflict_file),
                overwrite=False,
            )

        assert "already exists" in str(exc_info.value)
        assert "Use --overwrite" in str(exc_info.value)

    # Test Case 3: input output both files, conflict exists, overwrite=True
    def test_case_3():
        """Test: input output both files, output has conflict, overwrite=True (should succeed)"""
        conflict_file = dir_tmp / "overwrite_conflict.txt"
        conflict_file.write_text("existing content", encoding="utf-8")

        result = resolve_output_path(
            input_path=str(p_input_audio),
            output_path=str(conflict_file),
            overwrite=True,
        )

        assert result == conflict_file
        assert result.exists()  # File should still exist

    # Test Case 4: output is directory, no filename conflict
    def test_case_4():
        """Test: output is directory, no filename conflict"""
        # Ensure target file doesn't exist
        target_file = output_dir / "test_audio.txt"
        if target_file.exists():
            target_file.unlink()

        result = resolve_output_path(
            input_path=str(p_input_audio), output_path=str(output_dir), overwrite=False
        )

        expected = output_dir / "test_audio.txt"
        assert result == expected
        assert result.name == "test_audio.txt"

    # Test Case 5: output is directory, filename conflict (should auto-number)
    def test_case_5():
        """Test: output is directory, filename conflict (should add _01, _02 suffixes)"""
        # Create conflicting files
        base_file = output_dir / "conflict_audio.txt"
        conflict_file_01 = output_dir / "conflict_audio_01.txt"
        conflict_file_02 = output_dir / "conflict_audio_02.txt"

        base_file.write_text("content 0", encoding="utf-8")
        conflict_file_01.write_text("content 1", encoding="utf-8")
        conflict_file_02.write_text("content 2", encoding="utf-8")

        # Create input file with matching stem
        conflict_input = dir_tmp / "conflict_audio.mp3"
        conflict_input.write_text("fake audio", encoding="utf-8")

        result = resolve_output_path(
            input_path=str(conflict_input),
            output_path=str(output_dir),
            overwrite=False,  # overwrite doesn't matter for directory case
        )

        expected = output_dir / "conflict_audio_03.txt"
        assert result == expected
        assert result.name == "conflict_audio_03.txt"
        assert not result.exists()  # Should return non-existing file

    # Test Case 6: No output path specified (should create next to input)
    def test_case_6():
        """Test: No output path specified - should create .txt next to input file"""
        result = resolve_output_path(
            input_path=str(p_input_audio), output_path=None, overwrite=False
        )

        expected = p_input_audio.parent / "test_audio.txt"
        assert result == expected
        assert result.name == "test_audio.txt"

    # Test Case 7: No output path, input filename conflicts
    def test_case_7():
        """Test: No output path, but input.txt already exists (should auto-number)"""
        # Create conflicting file next to input
        conflict_input = dir_tmp / "auto_number_test.mp3"
        conflict_input.write_text("fake audio", encoding="utf-8")

        existing_txt = dir_tmp / "auto_number_test.txt"
        existing_txt.write_text("existing", encoding="utf-8")

        result = resolve_output_path(
            input_path=str(conflict_input), output_path=None, overwrite=False
        )

        expected = dir_tmp / "auto_number_test_01.txt"
        assert result == expected
        assert result.name == "auto_number_test_01.txt"

    # Run all test cases
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()
    test_case_6()
    test_case_7()

    print("✅ All test cases passed!")
    print("📋 Test Summary:")
    print("   ✅ Case 1: No file conflict - ✓")
    print("   ✅ Case 2: File conflict, overwrite=False (FileExistsError) - ✓")
    print("   ✅ Case 3: File conflict, overwrite=True - ✓")
    print("   ✅ Case 4: Directory output, no conflict - ✓")
    print("   ✅ Case 5: Directory output, auto-numbering (_03) - ✓")
    print("   ✅ Case 6: No output path (default behavior) - ✓")
    print("   ✅ Case 7: No output path, auto-numbering (_01) - ✓")


# ---------------------------------------------------------------------------
# Behavioral tests for AudioTranscriber.transcribe with the heavy transcription
# call mocked. These exercise the CLI plumbing (flag propagation, exit codes,
# quiet-mode output suppression) without paying the Whisper model load cost.
# ---------------------------------------------------------------------------


def _make_fake_input(name: str) -> str:
    """Create a placeholder file whose bytes are never decoded (Whisper mocked)."""
    path = dir_tmp / name
    path.write_bytes(b"fake-audio-bytes")
    return str(path)


def test_cli_propagates_flags_to_transcribe_in_parallel():
    """Every CLI flag must reach transcribe_audio_in_parallel with its value."""
    input_path = _make_fake_input("flag_prop_input.mp3")
    output_path = dir_tmp / "flag_prop_output.txt"
    if output_path.exists():
        output_path.unlink()

    with mock.patch(
        "audinota.cli.transcribe_audio_in_parallel", return_value="hello world"
    ) as mock_transcribe:
        AudioTranscriber().transcribe(
            input=input_path,
            output=str(output_path),
            model_size="base",
            device="cuda",
            compute_type="float16",
            cpu_threads=4,
            batch_size=8,
            language="en",
            task="translate",
            vad_filter=False,
            min_silence_duration_ms=750,
            seg_duration=60.0,
            quiet=True,
        )

    assert mock_transcribe.call_count == 1
    kwargs = mock_transcribe.call_args.kwargs
    assert kwargs["model_size"] == "base"
    assert kwargs["device"] == "cuda"
    assert kwargs["compute_type"] == "float16"
    assert kwargs["cpu_threads"] == 4
    assert kwargs["batch_size"] == 8
    assert kwargs["language"] == "en"
    assert kwargs["task"] == "translate"
    assert kwargs["vad_filter"] is False
    assert kwargs["min_silence_duration_ms"] == 750
    assert kwargs["seg_duration"] == 60.0
    assert output_path.read_text(encoding="utf-8") == "hello world"


def test_cli_exits_2_on_missing_input(capsys):
    with pytest.raises(SystemExit) as exc:
        AudioTranscriber().transcribe(input="/path/that/does/not/exist.mp3", quiet=True)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "not found" in captured.err.lower()


def test_cli_exits_1_on_existing_output_without_overwrite(capsys):
    input_path = _make_fake_input("exit1_input.mp3")
    output_path = dir_tmp / "exit1_output.txt"
    output_path.write_text("already here", encoding="utf-8")

    with mock.patch("audinota.cli.transcribe_audio_in_parallel"):
        with pytest.raises(SystemExit) as exc:
            AudioTranscriber().transcribe(
                input=input_path,
                output=str(output_path),
                overwrite=False,
                quiet=True,
            )
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "already exists" in captured.err.lower()
    # Existing file must be left untouched on the error path.
    assert output_path.read_text(encoding="utf-8") == "already here"


def test_cli_overwrite_replaces_existing_file():
    input_path = _make_fake_input("ow_input.mp3")
    output_path = dir_tmp / "ow_output.txt"
    output_path.write_text("stale content", encoding="utf-8")

    with mock.patch(
        "audinota.cli.transcribe_audio_in_parallel", return_value="fresh content"
    ):
        AudioTranscriber().transcribe(
            input=input_path,
            output=str(output_path),
            overwrite=True,
            quiet=True,
        )
    assert output_path.read_text(encoding="utf-8") == "fresh content"


def test_cli_quiet_suppresses_progress_lines(capsys):
    input_path = _make_fake_input("quiet_input.mp3")
    output_path = dir_tmp / "quiet_output.txt"
    if output_path.exists():
        output_path.unlink()

    with mock.patch(
        "audinota.cli.transcribe_audio_in_parallel", return_value="text"
    ):
        AudioTranscriber().transcribe(
            input=input_path, output=str(output_path), quiet=True
        )
    captured = capsys.readouterr()
    assert captured.out == "", f"--quiet should silence stdout, got: {captured.out!r}"
    assert captured.err == "", f"successful run should not write stderr, got: {captured.err!r}"


def test_cli_non_quiet_emits_progress(capsys):
    input_path = _make_fake_input("noisy_input.mp3")
    output_path = dir_tmp / "noisy_output.txt"
    if output_path.exists():
        output_path.unlink()

    with mock.patch(
        "audinota.cli.transcribe_audio_in_parallel", return_value="text"
    ):
        AudioTranscriber().transcribe(input=input_path, output=str(output_path))
    captured = capsys.readouterr()
    assert "Transcribing" in captured.out
    assert "Transcription complete" in captured.out


if __name__ == "__main__":
    from audinota.tests import run_cov_test

    run_cov_test(
        __file__,
        "audinota.cli",
        preview=False,
    )
