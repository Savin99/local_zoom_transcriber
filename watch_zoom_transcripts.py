#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent
DEFAULT_TRANSCRIBER = ROOT / "transcribe_zoom_gigaam.py"
STATE_VERSION = 1


@dataclass(frozen=True)
class RecordingSnapshot:
    folder: Path
    fingerprint: str
    latest_mtime: float
    source_audio_count: int
    speaker_audio_count: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def source_audio_files(folder: Path) -> list[Path]:
    audio = sorted([path for path in folder.glob("audio*.m4a") if path.is_file()], key=natural_key)
    if audio:
        return audio
    return sorted([path for path in folder.glob("*.m4a") if path.is_file()], key=natural_key)


def snapshot_recording(folder: Path, speaker_dir_name: str) -> Optional[RecordingSnapshot]:
    sources = source_audio_files(folder)
    if not sources:
        return None

    speaker_dir = folder / speaker_dir_name
    speakers = (
        sorted([path for path in speaker_dir.glob("*.m4a") if path.is_file()], key=natural_key)
        if speaker_dir.is_dir()
        else []
    )
    metadata = [folder / "recording.conf", folder / "zoomver.tag"]
    tracked = sources + speakers + [path for path in metadata if path.is_file()]

    items: list[dict[str, object]] = []
    latest_mtime = 0.0
    for path in tracked:
        stat = path.stat()
        latest_mtime = max(latest_mtime, stat.st_mtime)
        items.append(
            {
                "path": str(path.relative_to(folder)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    payload = json.dumps(items, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return RecordingSnapshot(
        folder=folder,
        fingerprint=hashlib.sha256(payload).hexdigest(),
        latest_mtime=latest_mtime,
        source_audio_count=len(sources),
        speaker_audio_count=len(speakers),
    )


def discover_recordings(zoom_root: Path, speaker_dir_name: str) -> list[RecordingSnapshot]:
    candidates: list[Path] = []
    if source_audio_files(zoom_root):
        candidates.append(zoom_root)
    candidates.extend(sorted([path for path in zoom_root.iterdir() if path.is_dir()], key=natural_key))

    snapshots: list[RecordingSnapshot] = []
    seen: set[Path] = set()
    for folder in candidates:
        resolved = folder.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        snapshot = snapshot_recording(folder, speaker_dir_name)
        if snapshot:
            snapshots.append(snapshot)
    return snapshots


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"version": STATE_VERSION, "processed": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": STATE_VERSION, "processed": {}}
    if not isinstance(state, dict):
        return {"version": STATE_VERSION, "processed": {}}
    state.setdefault("version", STATE_VERSION)
    state.setdefault("processed", {})
    return state


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def slugify(value: str) -> str:
    value = re.sub(r"\s+", "_", value.strip())
    value = re.sub(r"[^\w.\-]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "zoom_recording"


def exported_path(cowork_dir: Path, zoom_folder: Path) -> Path:
    return cowork_dir / f"{slugify(zoom_folder.name)}_transcript.md"


def call_transcript_paths(zoom_folder: Path) -> list[Path]:
    return sorted((zoom_folder / "transcripts").glob("call_*.md"), key=natural_key)


def expected_export_paths(cowork_dir: Path, zoom_folder: Path) -> list[Path]:
    calls = call_transcript_paths(zoom_folder)
    if not calls:
        return [exported_path(cowork_dir, zoom_folder)]
    prefix = slugify(zoom_folder.name)
    return [cowork_dir / f"{prefix}_{path.stem}_transcript.md" for path in calls]


def should_process(
    snapshot: RecordingSnapshot,
    state: dict,
    cowork_dir: Path,
    stable_sec: float,
    force: bool,
) -> tuple[bool, str]:
    age = time.time() - snapshot.latest_mtime
    if age < stable_sec:
        return False, f"waiting until stable ({int(age)}/{int(stable_sec)} sec)"

    key = str(snapshot.folder.resolve())
    previous = state["processed"].get(key) or {}
    export_files = expected_export_paths(cowork_dir, snapshot.folder)

    if force:
        return True, "forced"
    if previous.get("fingerprint") != snapshot.fingerprint:
        return True, "new or changed"
    if previous.get("status") != "done":
        return True, "previous run was not completed"
    if any(not path.exists() for path in export_files):
        return True, "export file is missing"
    return False, "already done"


def run_transcriber(snapshot: RecordingSnapshot, args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        str(args.transcriber),
        str(snapshot.folder),
        "--model",
        args.model,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--save-every-chunks",
        str(args.save_every_chunks),
        "--speaker-dir",
        args.speaker_dir,
    ]
    if args.no_diarization:
        command.append("--no-diarization")
    print(f"[{utc_now()}] Transcribing: {snapshot.folder}")
    return subprocess.run(command, cwd=str(ROOT)).returncode


def export_one_transcript(
    snapshot: RecordingSnapshot,
    cowork_dir: Path,
    transcript_path: Path,
    output_path: Path,
) -> Path:
    cowork_dir.mkdir(parents=True, exist_ok=True)
    transcript = transcript_path.read_text(encoding="utf-8").lstrip()
    body = "\n".join(
        [
            f"# {snapshot.folder.name}",
            "",
            f"- Source Zoom folder: `{snapshot.folder.resolve()}`",
            f"- Exported at: `{utc_now()}`",
            f"- Source audio files: `{snapshot.source_audio_count}`",
            f"- Speaker audio files: `{snapshot.speaker_audio_count}`",
            "",
            transcript,
        ]
    ).rstrip() + "\n"
    output_path.write_text(body, encoding="utf-8")
    return output_path


def export_transcripts(snapshot: RecordingSnapshot, cowork_dir: Path) -> list[Path]:
    call_paths = call_transcript_paths(snapshot.folder)
    if call_paths:
        prefix = slugify(snapshot.folder.name)
        return [
            export_one_transcript(
                snapshot,
                cowork_dir,
                transcript_path,
                cowork_dir / f"{prefix}_{transcript_path.stem}_transcript.md",
            )
            for transcript_path in call_paths
        ]

    transcript_path = snapshot.folder / "transcripts" / "all_calls.md"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found after ASR: {transcript_path}")
    return [export_one_transcript(snapshot, cowork_dir, transcript_path, exported_path(cowork_dir, snapshot.folder))]


def mark_state(
    state: dict,
    snapshot: RecordingSnapshot,
    status: str,
    export_files: Optional[list[Path]] = None,
    error: Optional[str] = None,
) -> None:
    key = str(snapshot.folder.resolve())
    item = {
        "status": status,
        "fingerprint": snapshot.fingerprint,
        "updated_at": utc_now(),
        "source_audio_count": snapshot.source_audio_count,
        "speaker_audio_count": snapshot.speaker_audio_count,
    }
    if export_files:
        item["export_files"] = [str(path.resolve()) for path in export_files]
    if error:
        item["error"] = error
    state["processed"][key] = item


def process_ready_recordings(args: argparse.Namespace, state: dict) -> int:
    snapshots = discover_recordings(args.zoom_root, args.speaker_dir)
    if not snapshots:
        print(f"[{utc_now()}] No Zoom recording folders with .m4a files found in {args.zoom_root}")
        return 0

    processed = 0
    for snapshot in snapshots:
        process, reason = should_process(snapshot, state, args.cowork_dir, args.stable_sec, args.force)
        if args.verbose or process or args.plan_only:
            print(f"[{utc_now()}] {snapshot.folder.name}: {reason}")
        if not process:
            continue
        if args.plan_only:
            processed += 1
            continue

        try:
            returncode = run_transcriber(snapshot, args)
            if returncode != 0:
                raise RuntimeError(f"transcriber exited with code {returncode}")
            export_files = export_transcripts(snapshot, args.cowork_dir)
            mark_state(state, snapshot, "done", export_files=export_files)
            save_state(args.state_file, state)
            processed += 1
            print(f"[{utc_now()}] Exported {len(export_files)} file(s) to {args.cowork_dir}")
        except Exception as exc:
            mark_state(state, snapshot, "failed", error=str(exc))
            save_state(args.state_file, state)
            print(f"[{utc_now()}] Failed: {snapshot.folder} ({exc})", file=sys.stderr)
    return processed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watch a common Zoom recordings folder and export transcripts to a Cowork folder."
    )
    parser.add_argument("zoom_root", type=Path, help="Common Zoom folder that receives per-meeting folders.")
    parser.add_argument("cowork_dir", type=Path, help="Folder watched by Cowork / knowledge base.")
    parser.add_argument("--once", action="store_true", help="Scan once and exit.")
    parser.add_argument("--plan-only", action="store_true", help="Print what would be processed without running ASR.")
    parser.add_argument("--force", action="store_true", help="Reprocess recordings even when state says they are done.")
    parser.add_argument("--poll-sec", type=float, default=60.0)
    parser.add_argument("--stable-sec", type=float, default=120.0)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--transcriber", type=Path, default=DEFAULT_TRANSCRIBER)
    parser.add_argument("--model", default="v3_e2e_rnnt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every-chunks", type=int, default=5)
    parser.add_argument("--speaker-dir", default="Audio Record")
    parser.add_argument("--no-diarization", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.zoom_root = args.zoom_root.expanduser().resolve()
    args.cowork_dir = args.cowork_dir.expanduser().resolve()
    args.transcriber = args.transcriber.expanduser().resolve()
    args.state_file = (
        args.state_file.expanduser().resolve()
        if args.state_file
        else args.cowork_dir / ".zoom_transcriber_state.json"
    )

    if not args.zoom_root.is_dir():
        print(f"Zoom root not found: {args.zoom_root}", file=sys.stderr)
        return 2
    if not args.transcriber.exists():
        print(f"Transcriber script not found: {args.transcriber}", file=sys.stderr)
        return 2

    state = load_state(args.state_file)
    print(f"[{utc_now()}] Watching Zoom root: {args.zoom_root}")
    print(f"[{utc_now()}] Exporting transcripts to: {args.cowork_dir}")
    print(f"[{utc_now()}] State file: {args.state_file}")

    while True:
        process_ready_recordings(args, state)
        if args.once:
            return 0
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
