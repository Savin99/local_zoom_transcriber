from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import watch_zoom_transcripts as watcher


def write_file(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def make_recording(
    zoom_root: Path,
    name: str = "2026-06-05 20.00.00 Test Meeting",
    *,
    transcript: bool = False,
    speaker_count: int = 0,
) -> Path:
    folder = zoom_root / name
    folder.mkdir(parents=True)
    (folder / "audio123.m4a").write_bytes(b"source audio")
    write_file(folder / "recording.conf", '{"magic_number": "123"}')
    if transcript:
        write_file(folder / "transcripts" / "all_calls.md", "# Zoom Transcription\n\nhello\n")
    if speaker_count:
        speaker_dir = folder / "Audio Record"
        speaker_dir.mkdir()
        for index in range(speaker_count):
            (speaker_dir / f"audioSpeaker{index}123.m4a").write_bytes(b"speaker audio")
    return folder


def make_fake_transcriber(tmp_path: Path, *, exit_code: int = 0, write_transcript: bool = True) -> Path:
    script = tmp_path / "fake_transcriber.py"
    lines = [
        "from __future__ import annotations",
        "",
        "import json",
        "import sys",
        "from pathlib import Path",
        "",
        "folder = Path(sys.argv[1])",
        "(folder / 'argv.json').write_text(",
        "    json.dumps(sys.argv, ensure_ascii=False),",
        "    encoding='utf-8',",
        ")",
    ]
    if write_transcript:
        lines.extend(
            [
                "transcripts = folder / 'transcripts'",
                "transcripts.mkdir(parents=True, exist_ok=True)",
                "(transcripts / 'call_01.md').write_text(",
                "    '# Call 01\\n\\nfake body\\n',",
                "    encoding='utf-8',",
                ")",
                "(transcripts / 'all_calls.md').write_text(",
                "    '# Zoom Transcription\\n\\nfake body\\n',",
                "    encoding='utf-8',",
                ")",
            ]
        )
    lines.append(f"raise SystemExit({exit_code})")
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script


def make_args(zoom_root: Path, cowork_dir: Path, transcriber: Path, **overrides) -> argparse.Namespace:
    values = {
        "zoom_root": zoom_root,
        "cowork_dir": cowork_dir,
        "once": True,
        "plan_only": False,
        "force": False,
        "poll_sec": 60.0,
        "stable_sec": 0.0,
        "state_file": cowork_dir / ".zoom_transcriber_state.json",
        "transcriber": transcriber,
        "model": "v3_e2e_rnnt",
        "device": "cpu",
        "batch_size": 1,
        "num_workers": 0,
        "save_every_chunks": 5,
        "speaker_dir": "Audio Record",
        "no_diarization": False,
        "verbose": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def process_quietly(args: argparse.Namespace, state: dict) -> int:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        return watcher.process_ready_recordings(args, state)


class WatchZoomTranscriptsTests(unittest.TestCase):
    def test_discover_recordings_finds_zoom_meeting_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zoom_root = root / "Zoom"
            zoom_root.mkdir()
            make_recording(zoom_root, "meeting 10", speaker_count=2)
            (zoom_root / "not a meeting").mkdir()

            snapshots = watcher.discover_recordings(zoom_root, "Audio Record")

            self.assertEqual([item.folder.name for item in snapshots], ["meeting 10"])
            self.assertEqual(snapshots[0].source_audio_count, 1)
            self.assertEqual(snapshots[0].speaker_audio_count, 2)

    def test_should_process_waits_until_files_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = make_recording(root)
            snapshot = watcher.snapshot_recording(folder, "Audio Record")
            assert snapshot is not None

            with patch.object(watcher.time, "time", return_value=snapshot.latest_mtime + 10):
                process, reason = watcher.should_process(
                    snapshot,
                    {"processed": {}},
                    root / "Cowork",
                    stable_sec=120,
                    force=False,
                )

            self.assertFalse(process)
            self.assertIn("waiting until stable", reason)

    def test_plan_only_reports_candidates_without_running_transcriber(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zoom_root = root / "Zoom"
            cowork_dir = root / "Cowork"
            zoom_root.mkdir()
            make_recording(zoom_root)
            transcriber = make_fake_transcriber(root, exit_code=99, write_transcript=False)
            args = make_args(zoom_root, cowork_dir, transcriber, plan_only=True)
            state = watcher.load_state(args.state_file)

            processed = process_quietly(args, state)

            self.assertEqual(processed, 1)
            self.assertFalse(args.state_file.exists())
            self.assertEqual(list(cowork_dir.glob("*.md")), [])

    def test_process_ready_recordings_exports_markdown_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zoom_root = root / "Zoom"
            cowork_dir = root / "Cowork"
            zoom_root.mkdir()
            meeting = make_recording(zoom_root, speaker_count=1)
            transcriber = make_fake_transcriber(root)
            args = make_args(zoom_root, cowork_dir, transcriber)
            state = watcher.load_state(args.state_file)

            processed = process_quietly(args, state)

            self.assertEqual(processed, 1)
            export_file = cowork_dir / "2026-06-05_20.00.00_Test_Meeting_call_01_transcript.md"
            self.assertTrue(export_file.exists())
            exported = export_file.read_text(encoding="utf-8")
            self.assertIn("# 2026-06-05 20.00.00 Test Meeting", exported)
            self.assertIn("fake body", exported)
            self.assertIn("Speaker audio files: `1`", exported)

            saved_state = json.loads(args.state_file.read_text(encoding="utf-8"))
            item = saved_state["processed"][str(meeting.resolve())]
            self.assertEqual(item["status"], "done")
            self.assertEqual(item["export_files"], [str(export_file.resolve())])

            argv = json.loads((meeting / "argv.json").read_text(encoding="utf-8"))
            self.assertIn("--speaker-dir", argv)
            self.assertIn("Audio Record", argv)

    def test_completed_recording_is_not_processed_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zoom_root = root / "Zoom"
            cowork_dir = root / "Cowork"
            zoom_root.mkdir()
            meeting = make_recording(zoom_root)
            transcriber = make_fake_transcriber(root)
            args = make_args(zoom_root, cowork_dir, transcriber)
            state = watcher.load_state(args.state_file)
            self.assertEqual(process_quietly(args, state), 1)
            first_argv_mtime = (meeting / "argv.json").stat().st_mtime_ns
            time.sleep(0.001)

            state = watcher.load_state(args.state_file)
            self.assertEqual(process_quietly(args, state), 0)

            self.assertEqual((meeting / "argv.json").stat().st_mtime_ns, first_argv_mtime)

    def test_force_processes_completed_recording_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zoom_root = root / "Zoom"
            cowork_dir = root / "Cowork"
            zoom_root.mkdir()
            meeting = make_recording(zoom_root)
            transcriber = make_fake_transcriber(root)
            args = make_args(zoom_root, cowork_dir, transcriber)
            state = watcher.load_state(args.state_file)
            self.assertEqual(process_quietly(args, state), 1)
            first_argv_mtime = (meeting / "argv.json").stat().st_mtime_ns
            time.sleep(0.001)

            state = watcher.load_state(args.state_file)
            forced_args = make_args(zoom_root, cowork_dir, transcriber, force=True)
            self.assertEqual(process_quietly(forced_args, state), 1)

            self.assertGreater((meeting / "argv.json").stat().st_mtime_ns, first_argv_mtime)

    def test_failed_transcriber_marks_state_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zoom_root = root / "Zoom"
            cowork_dir = root / "Cowork"
            zoom_root.mkdir()
            meeting = make_recording(zoom_root)
            transcriber = make_fake_transcriber(root, exit_code=7, write_transcript=False)
            args = make_args(zoom_root, cowork_dir, transcriber)
            state = watcher.load_state(args.state_file)

            processed = process_quietly(args, state)

            self.assertEqual(processed, 0)
            saved_state = json.loads(args.state_file.read_text(encoding="utf-8"))
            item = saved_state["processed"][str(meeting.resolve())]
            self.assertEqual(item["status"], "failed")
            self.assertIn("code 7", item["error"])

    def test_export_transcript_uses_safe_output_name_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meeting = make_recording(root, "2026-06-05 20:00 / weird name", transcript=True)
            snapshot = watcher.snapshot_recording(meeting, "Audio Record")
            assert snapshot is not None

            output = watcher.export_transcripts(snapshot, root / "Cowork")

            self.assertEqual(len(output), 1)
            self.assertEqual(output[0].name, "weird_name_transcript.md")
            text = output[0].read_text(encoding="utf-8")
            self.assertIn("Source Zoom folder:", text)
            self.assertIn("# Zoom Transcription", text)

    def test_export_transcripts_splits_existing_call_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meeting = make_recording(root, transcript=True)
            write_file(meeting / "transcripts" / "call_01.md", "# Call 01\n\nfirst\n")
            write_file(meeting / "transcripts" / "call_02.md", "# Call 02\n\nsecond\n")
            snapshot = watcher.snapshot_recording(meeting, "Audio Record")
            assert snapshot is not None

            output = watcher.export_transcripts(snapshot, root / "Cowork")

            self.assertEqual(
                [path.name for path in output],
                [
                    "2026-06-05_20.00.00_Test_Meeting_call_01_transcript.md",
                    "2026-06-05_20.00.00_Test_Meeting_call_02_transcript.md",
                ],
            )
            self.assertIn("first", output[0].read_text(encoding="utf-8"))
            self.assertIn("second", output[1].read_text(encoding="utf-8"))

    def test_load_state_recovers_from_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_file = write_file(Path(tmp) / "state.json", "not-json")

            state = watcher.load_state(state_file)

            self.assertEqual(state, {"version": watcher.STATE_VERSION, "processed": {}})


if __name__ == "__main__":
    unittest.main()
