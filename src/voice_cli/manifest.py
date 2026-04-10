from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass(slots=True)
class ManifestRecord:
    audio_file: str
    text: str
    speaker: str = "speaker01"
    language: str = "vi"
    duration_sec: float | None = None
    sample_rate: int | None = None
    split: str = "train"
    source_format: str = "unknown"
    source_id: str = ""
    status: str = "imported"
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def audio_path(self) -> Path:
        return Path(self.audio_file)

    @property
    def canonical_key(self) -> str:
        return str(self.audio_path.resolve()).lower()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "ManifestRecord":
        payload = dict(payload)
        payload.setdefault("speaker", "speaker01")
        payload.setdefault("language", "vi")
        payload.setdefault("split", "train")
        payload.setdefault("source_format", "unknown")
        payload.setdefault("source_id", "")
        payload.setdefault("status", "imported")
        payload.setdefault("error", None)
        payload.setdefault("notes", [])
        return cls(**payload)


def load_jsonl(path: Path) -> list[ManifestRecord]:
    if not path.exists():
        return []
    records: list[ManifestRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(ManifestRecord.from_dict(json.loads(stripped)))
    return records


def write_jsonl(path: Path, records: list[ManifestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False))
            handle.write("\n")


def merge_records(existing: list[ManifestRecord], new: list[ManifestRecord]) -> list[ManifestRecord]:
    merged: dict[str, ManifestRecord] = {record.canonical_key: record for record in existing}
    for record in new:
        merged[record.canonical_key] = record
    return sorted(merged.values(), key=lambda item: item.audio_file.lower())


def write_f5_csv(path: Path, records: list[ManifestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(["audio_file", "text"])
        for record in records:
            text = record.text.replace("\r", " ").replace("\n", " ").strip()
            writer.writerow([str(record.audio_path.resolve()), text])
