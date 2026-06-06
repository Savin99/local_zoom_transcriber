from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import transcribe_zoom_gigaam as transcriber


class TranscribeZoomGigaamTests(unittest.TestCase):
    def test_format_time_uses_hh_mm_ss(self) -> None:
        self.assertEqual(transcriber.format_time(0), "00:00:00")
        self.assertEqual(transcriber.format_time(65), "00:01:05")
        self.assertEqual(transcriber.format_time(3661), "01:01:01")

    def test_discover_recordings_prefers_top_level_zoom_audio_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "notes.m4a").write_bytes(b"ignore when audio files exist")
            (folder / "audio10.m4a").write_bytes(b"second")
            (folder / "audio2.m4a").write_bytes(b"first")

            recordings = transcriber.discover_recordings(folder)

            self.assertEqual([path.name for path in recordings], ["audio2.m4a", "audio10.m4a"])

    def test_discover_recordings_falls_back_to_any_m4a(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "part2.m4a").write_bytes(b"second")
            (folder / "part1.m4a").write_bytes(b"first")

            recordings = transcriber.discover_recordings(folder)

            self.assertEqual([path.name for path in recordings], ["part1.m4a", "part2.m4a"])

    def test_parse_speaker_name_removes_zoom_prefix_magic_and_suffix_numbers(self) -> None:
        name = transcriber.parse_speaker_name(Path("audioIvanPetrov3123.m4a"), "123")

        self.assertEqual(name, "Ivan Petrov")

    def test_parse_speaker_name_handles_zoom_long_numeric_suffix_without_magic(self) -> None:
        name = transcriber.parse_speaker_name(Path("audioAlexD41075103314.m4a"), None)

        self.assertEqual(name, "Alex D")

    def test_read_zoom_magic_number_returns_none_for_missing_or_invalid_conf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertIsNone(transcriber.read_zoom_magic_number(folder))
            (folder / "recording.conf").write_text("not-json", encoding="utf-8")

            self.assertIsNone(transcriber.read_zoom_magic_number(folder))

    def test_read_zoom_magic_number_from_recording_conf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "recording.conf").write_text(
                json.dumps({"magic_number": 1075103314}),
                encoding="utf-8",
            )

            self.assertEqual(transcriber.read_zoom_magic_number(folder), "1075103314")

    def test_apply_existing_transcript_restores_matching_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            call = transcriber.CallPlan(
                call_index=1,
                source_index=1,
                source_file="/tmp/audio.m4a",
                start=0.0,
                end=10.0,
                chunks=[
                    transcriber.AsrChunk(start=0.0, end=5.0),
                    transcriber.AsrChunk(start=5.0, end=10.0),
                ],
            )
            manifest = {
                "calls": [
                    {
                        "call_index": 1,
                        "source_index": 1,
                        "source_file": "/tmp/audio.m4a",
                        "start": 0.0,
                        "end": 10.0,
                        "chunks": [
                            {
                                "start": 0.0,
                                "end": 5.0,
                                "text": "hello",
                                "speaker": "Alex",
                                "speaker_confidence": 0.9,
                            }
                        ],
                    }
                ]
            }
            (output_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )

            restored = transcriber.apply_existing_transcript(output_dir, [call])

            self.assertEqual(restored, 1)
            self.assertEqual(call.chunks[0].text, "hello")
            self.assertEqual(call.chunks[0].speaker, "Alex")
            self.assertEqual(call.chunks[0].speaker_confidence, 0.9)
            self.assertEqual(call.chunks[1].text, "")

    def test_existing_manifest_complete_requires_calls_and_partial_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            self.assertFalse(transcriber.existing_manifest_complete(output_dir))
            (output_dir / "manifest.json").write_text(
                json.dumps({"partial": True, "calls": [{"call_index": 1}]}),
                encoding="utf-8",
            )
            self.assertFalse(transcriber.existing_manifest_complete(output_dir))
            (output_dir / "manifest.json").write_text(
                json.dumps({"partial": False, "calls": [{"call_index": 1}]}),
                encoding="utf-8",
            )

            self.assertTrue(transcriber.existing_manifest_complete(output_dir))

    def test_save_dry_run_plan_does_not_overwrite_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "call_01.md").write_text("real transcript\n", encoding="utf-8")
            (output_dir / "manifest.json").write_text('{"partial": false}\n', encoding="utf-8")
            call = transcriber.CallPlan(
                call_index=1,
                source_index=1,
                source_file="/tmp/audio.m4a",
                start=0.0,
                end=3.0,
                chunks=[transcriber.AsrChunk(start=0.0, end=3.0)],
            )

            plan_path = transcriber.save_dry_run_plan(output_dir, [call], {"dry_run": True})

            self.assertEqual((output_dir / "call_01.md").read_text(encoding="utf-8"), "real transcript\n")
            self.assertEqual((output_dir / "manifest.json").read_text(encoding="utf-8"), '{"partial": false}\n')
            self.assertEqual(plan_path.name, "dry_run_plan.json")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertTrue(plan["dry_run"])
            self.assertTrue(plan["partial"])

    def test_normalize_single_duplicate_labels_removes_lonely_zoom_duplicate_suffix(self) -> None:
        call = transcriber.CallPlan(
            call_index=1,
            source_index=1,
            source_file="/tmp/audio.m4a",
            start=0.0,
            end=10.0,
            chunks=[transcriber.AsrChunk(start=0.0, end=1.0, text="hi", speaker="Ivan (1) / Alex")],
        )

        transcriber.normalize_single_duplicate_labels([call])

        self.assertEqual(call.chunks[0].speaker, "Ivan / Alex")

    def test_render_call_markdown_includes_metadata_speaker_and_text(self) -> None:
        call = transcriber.CallPlan(
            call_index=1,
            source_index=1,
            source_file="/tmp/audio123.m4a",
            start=0.0,
            end=3.0,
            chunks=[transcriber.AsrChunk(start=0.0, end=3.0, text="hello", speaker="Alex")],
        )

        markdown = transcriber.render_call_markdown(call, partial=False)

        self.assertIn("# Call 01", markdown)
        self.assertIn("- Source: `audio123.m4a`", markdown)
        self.assertIn("- Speakers: Alex", markdown)
        self.assertIn("[00:00:00 - 00:00:03] **Alex:** hello", markdown)


if __name__ == "__main__":
    unittest.main()
