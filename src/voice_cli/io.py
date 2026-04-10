from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import unicodedata
import wave
import zipfile


class AudioProbeError(RuntimeError):
    pass


@dataclass(slots=True)
class AudioInfo:
    sample_rate: int
    channels: int
    duration_sec: float


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text_file(path: Path) -> str:
    encodings = ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "cp1258", "cp1252", "latin-1"]
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_text(text: str, normalize_unicode: bool = True) -> str:
    cleaned = text.replace("\ufeff", " ")
    if normalize_unicode:
        cleaned = unicodedata.normalize("NFC", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def parse_srt_text(raw: str, normalize_unicode: bool = True) -> str:
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.isdigit():
            continue
        if "-->" in stripped:
            continue
        lines.append(stripped)
    return normalize_text(" ".join(lines), normalize_unicode=normalize_unicode)


def probe_wav(path: Path) -> AudioInfo:
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            frame_count = handle.getnframes()
    except (wave.Error, FileNotFoundError, OSError) as exc:
        raise AudioProbeError(str(exc)) from exc
    duration = frame_count / float(sample_rate) if sample_rate else 0.0
    return AudioInfo(sample_rate=sample_rate, channels=channels, duration_sec=duration)


def extract_zip(source: Path, target_dir: Path, overwrite: bool = False) -> Path:
    if overwrite and target_dir.exists():
        shutil.rmtree(target_dir)
    if not target_dir.exists():
        ensure_dir(target_dir)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(target_dir)
    return target_dir


def find_first(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.rglob(pattern))
    return matches[0] if matches else None


def resolve_relative_audio(candidate: str, base_dir: Path) -> Path:
    audio_path = Path(candidate)
    if audio_path.is_absolute():
        return audio_path
    candidates = [
        base_dir / audio_path,
        base_dir / "wavs" / audio_path,
        base_dir.parent / audio_path,
        base_dir.parent / "wavs" / audio_path,
    ]
    for item in candidates:
        if item.exists():
            return item
    return candidates[0]


def sanitize_name(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = lowered.strip("_")
    return lowered or "run"


def timestamp_id(prefix: str) -> str:
    return f"{sanitize_name(prefix)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
