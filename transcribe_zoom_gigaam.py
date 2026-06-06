#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
from collections import Counter
import json
import math
import re
import subprocess
import sys
import unicodedata
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
GIGAAM_DIR = ROOT / "GigaAM"
if GIGAAM_DIR.exists():
    sys.path.insert(0, str(GIGAAM_DIR))

import gigaam  # noqa: E402
from gigaam.preprocess import SAMPLE_RATE, load_audio  # noqa: E402
from gigaam.utils import AudioDataset  # noqa: E402

ASR_JOIN_GAP_SEC = 1.20
ASR_BOUNDARY_SEARCH_SEC = 2.50


@dataclass
class SpeechInterval:
    start: float
    end: float


@dataclass
class AsrChunk:
    start: float
    end: float
    text: str = ""
    speaker: Optional[str] = None
    speaker_confidence: Optional[float] = None


@dataclass
class CallPlan:
    call_index: int
    source_index: int
    source_file: str
    start: float
    end: float
    chunks: list[AsrChunk]


@dataclass
class SpeakerTrack:
    path: str
    speaker: str
    duration: float
    intervals: list[SpeechInterval]
    threshold_db: float


def format_time(seconds: float) -> str:
    whole = int(max(0.0, float(seconds)))
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def discover_recordings(folder: Path) -> list[Path]:
    audio = sorted(
        [p for p in folder.glob("audio*.m4a") if p.is_file()],
        key=natural_key,
    )
    if audio:
        return audio

    return sorted(
        [p for p in folder.glob("*.m4a") if p.is_file()],
        key=natural_key,
    )


def get_audio_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def read_zoom_magic_number(folder: Path) -> Optional[str]:
    conf_path = folder / "recording.conf"
    if not conf_path.exists():
        return None
    try:
        data = json.loads(conf_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    magic = data.get("magic_number")
    return str(magic) if magic else None


def pretty_speaker_name(raw: str) -> str:
    raw = unicodedata.normalize("NFC", raw).strip("_- ")
    raw = re.sub(r"(?<=[a-zа-яё])(?=[A-ZА-ЯЁ])", " ", raw)
    raw = re.sub(r"(?<=[A-ZА-ЯЁ])(?=[A-ZА-ЯЁ][a-zа-яё])", " ", raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw or "Unknown"


def parse_speaker_name(path: Path, magic_number: Optional[str]) -> str:
    stem = unicodedata.normalize("NFC", path.stem)
    if stem.lower().startswith("audio"):
        stem = stem[5:]
    if magic_number and stem.endswith(magic_number):
        stem = stem[: -len(magic_number)]
    else:
        stem = re.sub(r"\d{6,}$", "", stem)
    stem = re.sub(r"\d+$", "", stem)
    return pretty_speaker_name(stem)


def parse_threshold(value: str) -> Optional[float]:
    if value.lower() == "auto":
        return None
    return float(value)


def choose_threshold(db: torch.Tensor) -> float:
    finite = db[torch.isfinite(db)]
    if finite.numel() == 0:
        return -42.0
    noise = torch.quantile(finite, 0.20).item()
    loud = torch.quantile(finite, 0.95).item()
    adaptive = noise + (loud - noise) * 0.35
    return max(-52.0, min(-32.0, adaptive))


def frame_db(audio: torch.Tensor, frame_sec: float, hop_sec: float) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    frame_len = max(1, int(SAMPLE_RATE * frame_sec))
    hop_len = max(1, int(SAMPLE_RATE * hop_sec))
    if audio.numel() < frame_len:
        rms = torch.sqrt(torch.mean(audio.float().pow(2)).clamp_min(1e-12))
        return torch.tensor([20.0 * torch.log10(rms).item()]), torch.tensor([0.0]), frame_len / SAMPLE_RATE, hop_len / SAMPLE_RATE

    power = audio.float().pow(2)
    cumulative = torch.nn.functional.pad(torch.cumsum(power, dim=0), (1, 0))
    starts = torch.arange(0, audio.numel() - frame_len + 1, hop_len)
    sums = cumulative[starts + frame_len] - cumulative[starts]
    rms = torch.sqrt((sums / frame_len).clamp_min(1e-12))
    db = 20.0 * torch.log10(rms)
    times = starts.float() / SAMPLE_RATE
    return db, times, frame_len / SAMPLE_RATE, hop_len / SAMPLE_RATE


def merge_intervals(intervals: Iterable[SpeechInterval], max_gap: float) -> list[SpeechInterval]:
    merged: list[SpeechInterval] = []
    for interval in sorted(intervals, key=lambda item: item.start):
        if not merged or interval.start - merged[-1].end > max_gap:
            merged.append(SpeechInterval(interval.start, interval.end))
        else:
            merged[-1].end = max(merged[-1].end, interval.end)
    return merged


def interval_overlap(start: float, end: float, interval: SpeechInterval) -> float:
    return max(0.0, min(end, interval.end) - max(start, interval.start))


def overlap_score(start: float, end: float, intervals: list[SpeechInterval]) -> float:
    if not intervals:
        return 0.0
    starts = [item.start for item in intervals]
    idx = max(0, bisect.bisect_left(starts, start) - 1)
    score = 0.0
    while idx < len(intervals) and intervals[idx].start < end:
        score += interval_overlap(start, end, intervals[idx])
        idx += 1
    return score


def detect_speech_intervals(
    audio: torch.Tensor,
    threshold_db: Optional[float],
    frame_sec: float,
    hop_sec: float,
    min_speech_sec: float,
    merge_silence_sec: float,
    pad_sec: float,
) -> tuple[list[SpeechInterval], float]:
    db, times, frame_sec_real, _ = frame_db(audio, frame_sec, hop_sec)
    threshold = choose_threshold(db) if threshold_db is None else threshold_db
    active = db > threshold
    duration = audio.numel() / SAMPLE_RATE

    intervals: list[SpeechInterval] = []
    run_start: Optional[float] = None
    last_end = 0.0
    for is_active, start in zip(active.tolist(), times.tolist()):
        end = min(duration, start + frame_sec_real)
        if is_active and run_start is None:
            run_start = start
        if is_active:
            last_end = end
        elif run_start is not None:
            if last_end - run_start >= min_speech_sec:
                intervals.append(
                    SpeechInterval(
                        max(0.0, run_start - pad_sec),
                        min(duration, last_end + pad_sec),
                    )
                )
            run_start = None

    if run_start is not None and last_end - run_start >= min_speech_sec:
        intervals.append(
            SpeechInterval(max(0.0, run_start - pad_sec), min(duration, last_end + pad_sec))
        )

    return merge_intervals(intervals, merge_silence_sec), threshold


def find_quiet_chunk_boundary(audio: torch.Tensor, lower: float, upper: float) -> float:
    lower = max(0.0, lower)
    upper = min(audio.numel() / SAMPLE_RATE, upper)
    if upper <= lower:
        return lower

    frame_sec = 0.18
    hop_sec = 0.06
    start_sample = int(lower * SAMPLE_RATE)
    end_sample = int(upper * SAMPLE_RATE)
    window = audio[start_sample:end_sample]
    db, times, frame_real, _ = frame_db(window, frame_sec, hop_sec)
    idx = int(torch.argmin(db).item())
    return lower + float(times[idx].item()) + frame_real / 2.0


def split_long_speech_interval(audio: torch.Tensor, interval: SpeechInterval, max_chunk_sec: float) -> list[SpeechInterval]:
    parts: list[SpeechInterval] = []
    start = interval.start
    while interval.end - start > max_chunk_sec:
        target = start + max_chunk_sec
        lower = max(start + max_chunk_sec * 0.60, target - ASR_BOUNDARY_SEARCH_SEC)
        upper = min(interval.end, target)
        split = find_quiet_chunk_boundary(audio, lower, upper)
        if split <= start + 1.0:
            split = target
        parts.append(SpeechInterval(start, min(split, interval.end)))
        start = min(split, interval.end)
    if interval.end - start > 0.2:
        parts.append(SpeechInterval(start, interval.end))
    return parts


def build_chunks(
    audio: torch.Tensor,
    intervals: list[SpeechInterval],
    max_chunk_sec: float,
) -> list[AsrChunk]:
    base: list[SpeechInterval] = []
    for interval in intervals:
        if interval.end - interval.start > max_chunk_sec:
            base.extend(split_long_speech_interval(audio, interval, max_chunk_sec))
        else:
            base.append(interval)

    chunks: list[SpeechInterval] = []
    for interval in base:
        if not chunks:
            chunks.append(SpeechInterval(interval.start, interval.end))
            continue
        candidate_duration = interval.end - chunks[-1].start
        gap = interval.start - chunks[-1].end
        if gap <= ASR_JOIN_GAP_SEC and candidate_duration <= max_chunk_sec:
            chunks[-1].end = interval.end
        else:
            chunks.append(SpeechInterval(interval.start, interval.end))

    return [AsrChunk(round(item.start, 3), round(item.end, 3)) for item in chunks if item.end - item.start > 0.2]


def build_recording_call(
    source_index: int,
    source_file: Path,
    speech: list[SpeechInterval],
    audio: torch.Tensor,
    max_chunk_sec: float,
) -> Optional[CallPlan]:
    if not speech:
        return None

    chunks = build_chunks(audio, speech, max_chunk_sec)
    if not chunks:
        return None

    return CallPlan(
        call_index=source_index,
        source_index=source_index,
        source_file=str(source_file),
        start=round(speech[0].start, 3),
        end=round(speech[-1].end, 3),
        chunks=chunks,
    )


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def transcribe_chunks(
    model: gigaam.GigaAMASR,
    audio: torch.Tensor,
    chunks: list[AsrChunk],
    batch_size: int,
    num_workers: int,
    max_chunks: Optional[int],
    call_index: int,
    save_every_chunks: int,
    after_save: Optional[Callable[[], None]] = None,
) -> int:
    pending = [chunk for chunk in chunks if not chunk.text]
    selected = pending if max_chunks is None else pending[:max_chunks]
    if not selected:
        return 0

    tensors = [
        audio[int(chunk.start * SAMPLE_RATE) : int(chunk.end * SAMPLE_RATE)]
        for chunk in selected
    ]
    dataset = AudioDataset(tensors, tokenizer=None)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=AudioDataset.collate,
        num_workers=num_workers,
    )

    written = 0
    for batch_index, (wav_pad, wav_lens) in enumerate(loader, start=1):
        wav_pad = wav_pad.to(model._device).to(model._dtype)
        wav_lens = wav_lens.to(model._device)
        encoded, encoded_len = model.forward(wav_pad, wav_lens)
        for text, _words in model._decode(encoded, encoded_len, wav_lens, False):
            selected[written].text = text.strip()
            written += 1
        print(
            f"  call {call_index:02d}: transcribed {written}/{len(selected)} chunks "
            f"(batch {batch_index}/{math.ceil(len(selected) / batch_size)})",
            flush=True,
        )
        if after_save and (
            written == len(selected)
            or (save_every_chunks > 0 and written % save_every_chunks == 0)
        ):
            after_save()
    return written


def render_call_markdown(call: CallPlan, partial: bool) -> str:
    speakers = call_speaker_list(call)
    lines = [
        f"# Call {call.call_index:02d}",
        "",
        f"- Source: `{Path(call.source_file).name}`",
        f"- Range: `{format_time(call.start)} - {format_time(call.end)}`",
    ]
    if speakers:
        lines.append(f"- Speakers: {', '.join(speakers)}")
    if partial:
        lines.append("- Status: partial / in progress")
    lines.append("")

    for chunk in call.chunks:
        if not chunk.text:
            continue
        prefix = f"[{format_time(chunk.start)} - {format_time(chunk.end)}]"
        if chunk.speaker:
            lines.append(f"{prefix} **{chunk.speaker}:** {chunk.text}")
        else:
            lines.append(f"{prefix} {chunk.text}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_call_text(call: CallPlan) -> str:
    lines = []
    for chunk in call.chunks:
        if not chunk.text:
            continue
        if chunk.speaker:
            lines.append(f"{chunk.speaker}: {chunk.text}")
        else:
            lines.append(chunk.text)
    return "\n\n".join(lines).strip() + "\n"


def save_outputs(output_dir: Path, calls: list[CallPlan], metadata: dict, partial: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for call in calls:
        stem = f"call_{call.call_index:02d}"
        (output_dir / f"{stem}.md").write_text(render_call_markdown(call, partial), encoding="utf-8")
        (output_dir / f"{stem}.txt").write_text(render_call_text(call), encoding="utf-8")

    all_lines: list[str] = ["# Zoom Transcription", ""]
    for call in calls:
        all_lines.append(render_call_markdown(call, partial).rstrip())
        all_lines.append("")
    (output_dir / "all_calls.md").write_text("\n".join(all_lines).rstrip() + "\n", encoding="utf-8")

    manifest = {
        **metadata,
        "partial": partial,
        "calls": [
            {
                **asdict(call),
                "source_file": str(Path(call.source_file).resolve()),
            }
            for call in calls
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_dry_run_plan(output_dir: Path, calls: list[CallPlan], metadata: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "dry_run_plan.json"
    plan = {
        **metadata,
        "partial": True,
        "calls": [
            {
                **asdict(call),
                "source_file": str(Path(call.source_file).resolve()),
            }
            for call in calls
        ],
    }
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return plan_path


def apply_existing_transcript(output_dir: Path, calls: list[CallPlan]) -> int:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return 0

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0

    existing: dict[tuple[int, float, float], dict] = {}
    for call_data in manifest.get("calls", []):
        call_index = int(call_data.get("call_index", -1))
        for chunk in call_data.get("chunks", []):
            text = str(chunk.get("text") or "").strip()
            if not text:
                continue
            key = (
                call_index,
                round(float(chunk.get("start", 0.0)), 3),
                round(float(chunk.get("end", 0.0)), 3),
            )
            existing[key] = chunk

    restored = 0
    for call in calls:
        for chunk in call.chunks:
            key = (call.call_index, round(chunk.start, 3), round(chunk.end, 3))
            if key in existing and not chunk.text:
                saved = existing[key]
                chunk.text = str(saved.get("text") or "").strip()
                chunk.speaker = saved.get("speaker")
                chunk.speaker_confidence = saved.get("speaker_confidence")
                restored += 1
    return restored


def calls_from_manifest(manifest: dict) -> list[CallPlan]:
    calls: list[CallPlan] = []
    for call_data in manifest.get("calls", []):
        chunks = [
            AsrChunk(
                start=round(float(chunk.get("start", 0.0)), 3),
                end=round(float(chunk.get("end", 0.0)), 3),
                text=str(chunk.get("text") or "").strip(),
                speaker=chunk.get("speaker"),
                speaker_confidence=chunk.get("speaker_confidence"),
            )
            for chunk in call_data.get("chunks", [])
        ]
        calls.append(
            CallPlan(
                call_index=int(call_data["call_index"]),
                source_index=int(call_data["source_index"]),
                source_file=str(call_data["source_file"]),
                start=round(float(call_data["start"]), 3),
                end=round(float(call_data["end"]), 3),
                chunks=chunks,
            )
        )
    return calls


def load_manifest_calls(output_dir: Path) -> tuple[dict, list[CallPlan]]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = {key: value for key, value in manifest.items() if key != "calls"}
    return metadata, calls_from_manifest(manifest)


def call_speaker_list(call: CallPlan) -> list[str]:
    speakers: set[str] = set()
    for chunk in call.chunks:
        if not chunk.speaker:
            continue
        speakers.update(part.strip() for part in chunk.speaker.split("/") if part.strip())
    return sorted(speakers)


def normalize_single_duplicate_labels(calls: list[CallPlan]) -> None:
    duplicate_pattern = re.compile(r"^(?P<base>.+) \((?P<number>\d+)\)$")
    for call in calls:
        used_parts: set[str] = set()
        for chunk in call.chunks:
            if chunk.speaker:
                used_parts.update(part.strip() for part in chunk.speaker.split("/") if part.strip())

        by_base: dict[str, list[str]] = {}
        for speaker in used_parts:
            match = duplicate_pattern.match(speaker)
            if match:
                by_base.setdefault(match.group("base"), []).append(speaker)

        replacements = {
            variants[0]: base
            for base, variants in by_base.items()
            if base not in used_parts and len(variants) == 1
        }
        if not replacements:
            continue

        for chunk in call.chunks:
            if not chunk.speaker:
                continue
            parts = [part.strip() for part in chunk.speaker.split("/")]
            chunk.speaker = " / ".join(replacements.get(part, part) for part in parts)


def collect_speaker_tracks(
    folder: Path,
    calls: list[CallPlan],
    speaker_dir_name: str,
    duration_tolerance_sec: float,
    threshold_db: Optional[float],
    frame_sec: float,
    hop_sec: float,
    min_speech_sec: float,
    merge_silence_sec: float,
    pad_sec: float,
) -> tuple[dict[str, list[SpeakerTrack]], list[dict]]:
    speaker_dir = folder / speaker_dir_name
    if not speaker_dir.is_dir():
        return {}, []

    magic_number = read_zoom_magic_number(folder)
    audio_files = sorted(speaker_dir.glob("*.m4a"), key=natural_key)
    candidates: list[dict] = []
    for path in audio_files:
        try:
            duration = get_audio_duration(path)
        except (subprocess.CalledProcessError, ValueError):
            continue
        candidates.append(
            {
                "path": path,
                "speaker": parse_speaker_name(path, magic_number),
                "duration": duration,
            }
        )

    source_files = sorted({Path(call.source_file).resolve() for call in calls}, key=natural_key)
    tracks_by_source: dict[str, list[SpeakerTrack]] = {}
    summary: list[dict] = []

    for source_file in source_files:
        source_duration = get_audio_duration(source_file)
        matched = [
            item
            for item in candidates
            if abs(float(item["duration"]) - source_duration) <= duration_tolerance_sec
        ]
        name_counts = Counter(str(item["speaker"]) for item in matched)
        seen: Counter[str] = Counter()
        tracks: list[SpeakerTrack] = []
        for item in matched:
            base_name = str(item["speaker"])
            seen[base_name] += 1
            speaker_name = (
                f"{base_name} ({seen[base_name]})"
                if name_counts[base_name] > 1
                else base_name
            )
            print(
                f"  diarization: {source_file.name} <- {speaker_name} "
                f"({Path(item['path']).name})",
                flush=True,
            )
            audio = load_audio(str(item["path"]))
            intervals, used_threshold = detect_speech_intervals(
                audio,
                threshold_db,
                frame_sec,
                hop_sec,
                min_speech_sec,
                merge_silence_sec,
                pad_sec,
            )
            tracks.append(
                SpeakerTrack(
                    path=str(Path(item["path"]).resolve()),
                    speaker=speaker_name,
                    duration=round(float(item["duration"]), 3),
                    intervals=intervals,
                    threshold_db=round(used_threshold, 2),
                )
            )
            summary.append(
                {
                    "source_file": str(source_file),
                    "speaker": speaker_name,
                    "track_file": str(Path(item["path"]).resolve()),
                    "duration_sec": round(float(item["duration"]), 3),
                    "speech_intervals": len(intervals),
                    "threshold_db": round(used_threshold, 2),
                }
            )
        tracks_by_source[str(source_file)] = tracks

    return tracks_by_source, summary


def apply_diarization(
    folder: Path,
    calls: list[CallPlan],
    metadata: dict,
    args: argparse.Namespace,
) -> int:
    tracks_by_source, summary = collect_speaker_tracks(
        folder,
        calls,
        args.speaker_dir,
        args.speaker_match_tolerance_sec,
        parse_threshold(args.speaker_threshold_db),
        args.speaker_frame_sec,
        args.speaker_hop_sec,
        args.speaker_min_speech_sec,
        args.speaker_merge_silence_sec,
        args.speaker_pad_sec,
    )
    assigned = 0
    for call in calls:
        tracks = tracks_by_source.get(str(Path(call.source_file).resolve()), [])
        for chunk in call.chunks:
            chunk.speaker = None
            chunk.speaker_confidence = None
            if not chunk.text or not tracks:
                continue
            scores = []
            for track in tracks:
                score = overlap_score(chunk.start, chunk.end, track.intervals)
                if score >= args.speaker_min_overlap_sec:
                    scores.append((track.speaker, score))
            if not scores:
                continue
            scores.sort(key=lambda item: item[1], reverse=True)
            top_score = scores[0][1]
            selected = [scores[0]]
            for speaker, score in scores[1:]:
                if len(selected) >= args.speaker_max_labels:
                    break
                if score >= args.speaker_min_overlap_sec and score / top_score >= args.speaker_secondary_ratio:
                    selected.append((speaker, score))
            chunk.speaker = " / ".join(speaker for speaker, _score in selected)
            chunk_duration = max(0.001, chunk.end - chunk.start)
            chunk.speaker_confidence = round(min(1.0, top_score / chunk_duration), 3)
            assigned += 1

    normalize_single_duplicate_labels(calls)
    metadata["diarization"] = {
        "method": "zoom_separate_audio_tracks",
        "speaker_dir": args.speaker_dir,
        "assigned_chunks": assigned,
        "track_count": len(summary),
        "tracks": summary,
        "settings": {
            "speaker_match_tolerance_sec": args.speaker_match_tolerance_sec,
            "speaker_threshold_db": args.speaker_threshold_db,
            "speaker_min_overlap_sec": args.speaker_min_overlap_sec,
            "speaker_secondary_ratio": args.speaker_secondary_ratio,
            "speaker_max_labels": args.speaker_max_labels,
        },
    }
    return assigned


def text_chunk_count(calls: list[CallPlan]) -> int:
    return sum(1 for call in calls for chunk in call.chunks if chunk.text)


def existing_manifest_complete(output_dir: Path) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(manifest.get("calls")) and manifest.get("partial") is False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe local Zoom recording folders with GigaAM."
    )
    parser.add_argument("zoom_folder", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default="v3_e2e_rnnt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--vad-threshold-db", default="auto", help="Use a number like -42 or 'auto'.")
    parser.add_argument("--vad-frame-sec", type=float, default=0.50)
    parser.add_argument("--vad-hop-sec", type=float, default=0.10)
    parser.add_argument("--min-speech-sec", type=float, default=0.45)
    parser.add_argument("--merge-silence-sec", type=float, default=0.80)
    parser.add_argument("--speech-pad-sec", type=float, default=0.25)
    parser.add_argument("--chunk-sec", type=float, default=22.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--save-every-chunks", type=int, default=5)
    parser.add_argument("--diarize-only", action="store_true")
    parser.add_argument("--no-diarization", action="store_true")
    parser.add_argument("--speaker-dir", default="Audio Record")
    parser.add_argument("--speaker-match-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--speaker-threshold-db", default="auto")
    parser.add_argument("--speaker-frame-sec", type=float, default=0.30)
    parser.add_argument("--speaker-hop-sec", type=float, default=0.10)
    parser.add_argument("--speaker-min-speech-sec", type=float, default=0.20)
    parser.add_argument("--speaker-merge-silence-sec", type=float, default=0.20)
    parser.add_argument("--speaker-pad-sec", type=float, default=0.05)
    parser.add_argument("--speaker-min-overlap-sec", type=float, default=0.20)
    parser.add_argument("--speaker-secondary-ratio", type=float, default=0.35)
    parser.add_argument("--speaker-max-labels", type=int, default=2)
    return parser


def main() -> int:
    warnings.filterwarnings(
        "ignore",
        message=r"An output with one or more elements was resized.*",
        category=UserWarning,
    )
    args = build_arg_parser().parse_args()
    folder = args.zoom_folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"Zoom folder not found: {folder}", file=sys.stderr)
        return 2

    recordings = discover_recordings(folder)
    if not recordings:
        print(f"No top-level .m4a Zoom audio files found in {folder}", file=sys.stderr)
        return 2

    threshold_arg = parse_threshold(args.vad_threshold_db)
    output_dir = (args.output or (folder / "transcripts")).expanduser().resolve()

    if args.diarize_only:
        metadata, calls = load_manifest_calls(output_dir)
        print(f"Applying diarization from {folder / args.speaker_dir}...", flush=True)
        assigned = apply_diarization(folder, calls, metadata, args)
        save_outputs(output_dir, calls, metadata, partial=bool(metadata.get("partial")))
        print(f"Diarization assigned speakers to {assigned} chunks in {output_dir}", flush=True)
        return 0

    calls: list[CallPlan] = []
    source_meta: list[dict] = []

    for source_index, source_file in enumerate(recordings, start=1):
        print(f"[{source_index}/{len(recordings)}] Loading {source_file.name}...", flush=True)
        audio = load_audio(str(source_file))
        duration = audio.numel() / SAMPLE_RATE
        speech, threshold = detect_speech_intervals(
            audio,
            threshold_arg,
            args.vad_frame_sec,
            args.vad_hop_sec,
            args.min_speech_sec,
            args.merge_silence_sec,
            args.speech_pad_sec,
        )
        call = build_recording_call(
            source_index,
            source_file,
            speech,
            audio,
            args.chunk_sec,
        )
        if call:
            calls.append(call)
        source_meta.append(
            {
                "source_index": source_index,
                "source_file": str(source_file.resolve()),
                "duration_sec": round(duration, 3),
                "threshold_db": round(threshold, 2),
                "speech_intervals": len(speech),
                "calls": 1 if call else 0,
                "chunks": len(call.chunks) if call else 0,
            }
        )
        print(
            f"  duration={format_time(duration)} speech_intervals={len(speech)} "
            f"calls={1 if call else 0} chunks={len(call.chunks) if call else 0} "
            f"threshold={threshold:.1f}dB",
            flush=True,
        )

    partial = args.dry_run or args.max_chunks is not None
    resumed_chunks = 0 if args.dry_run else apply_existing_transcript(output_dir, calls)
    if resumed_chunks:
        print(f"Resumed {resumed_chunks} already transcribed chunks from {output_dir}", flush=True)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "zoom_folder": str(folder),
        "model": args.model,
        "sample_rate": SAMPLE_RATE,
        "dry_run": args.dry_run,
        "partial": partial,
        "resumed_chunks": resumed_chunks,
        "sources": source_meta,
        "settings": {
            "chunk_sec": args.chunk_sec,
            "merge_silence_sec": args.merge_silence_sec,
            "vad_threshold_db": args.vad_threshold_db,
        },
    }

    if args.dry_run:
        plan_path = save_dry_run_plan(output_dir, calls, metadata)
        print(f"Dry run plan written to {plan_path}", flush=True)
        return 0

    if existing_manifest_complete(output_dir) and args.max_chunks is None:
        print(f"Transcript already complete in {output_dir}; skipping ASR.", flush=True)
        if not args.no_diarization:
            print(f"Applying diarization from {folder / args.speaker_dir}...", flush=True)
            assigned = apply_diarization(folder, calls, metadata, args)
            print(f"Diarization assigned speakers to {assigned} chunks.", flush=True)
        save_outputs(output_dir, calls, metadata, partial=False)
        return 0

    device = resolve_device(args.device)
    print(f"Loading GigaAM model {args.model} on {device}...", flush=True)
    model = gigaam.load_model(args.model, device=device, use_flash=False)

    chunks_left = args.max_chunks
    loaded_source_file: Optional[str] = None
    loaded_audio: Optional[torch.Tensor] = None
    for index, call in enumerate(calls, start=1):
        print(f"Transcribing call {call.call_index:02d} ({len(call.chunks)} chunks)...", flush=True)
        if loaded_source_file != call.source_file:
            loaded_source_file = call.source_file
            loaded_audio = load_audio(call.source_file)
        if loaded_audio is None:
            raise RuntimeError(f"Failed to load audio: {call.source_file}")
        used_limit = chunks_left
        def checkpoint() -> None:
            save_outputs(output_dir, calls, metadata, partial=True)

        done = transcribe_chunks(
            model,
            loaded_audio,
            call.chunks,
            args.batch_size,
            args.num_workers,
            used_limit,
            call.call_index,
            args.save_every_chunks,
            checkpoint,
        )
        if args.max_chunks is None:
            save_outputs(output_dir, calls, metadata, partial=index < len(calls))
        if chunks_left is not None:
            chunks_left = max(0, chunks_left - done)
            if chunks_left == 0:
                break

    if not args.no_diarization and not partial:
        print(f"Applying diarization from {folder / args.speaker_dir}...", flush=True)
        assigned = apply_diarization(folder, calls, metadata, args)
        print(f"Diarization assigned speakers to {assigned} chunks.", flush=True)

    save_outputs(output_dir, calls, metadata, partial=partial)
    total_chunks = sum(1 for call in calls for chunk in call.chunks if chunk.text)
    print(f"Written {len(calls)} calls, {total_chunks} transcribed chunks to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
