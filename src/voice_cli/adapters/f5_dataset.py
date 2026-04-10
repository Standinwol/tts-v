from __future__ import annotations

import csv
import json
from pathlib import Path

from voice_cli.io import extract_zip, find_first, normalize_text, probe_wav, resolve_relative_audio
from voice_cli.manifest import ManifestRecord


def import_from_f5_zip(
    source: Path,
    imports_root: Path,
    speaker: str | None = None,
    language: str = "vi",
    overwrite: bool = False,
) -> list[ManifestRecord]:
    extracted_root = extract_zip(source, imports_root / source.stem, overwrite=overwrite)
    metadata_json = find_first(extracted_root, "metadata.json")
    metadata_csv = find_first(extracted_root, "metadata.csv")
    if metadata_json:
        return import_from_f5_json(metadata_json, speaker=speaker, language=language)
    if metadata_csv:
        return import_from_f5_csv(metadata_csv, speaker=speaker, language=language)
    raise FileNotFoundError(f"No metadata.json or metadata.csv found in {source}")


def import_from_f5_csv(
    source: Path,
    speaker: str | None = None,
    language: str = "vi",
) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        for index, row in enumerate(reader, start=1):
            raw_audio = row.get("audio_file") or row.get("file_name")
            raw_text = row.get("text")
            if not raw_audio or raw_text is None:
                continue
            audio_path = resolve_relative_audio(raw_audio, source.parent)
            duration = None
            sample_rate = None
            try:
                audio_info = probe_wav(audio_path)
                duration = round(audio_info.duration_sec, 6)
                sample_rate = audio_info.sample_rate
            except Exception:
                pass
            records.append(
                ManifestRecord(
                    audio_file=str(audio_path.resolve()),
                    text=normalize_text(raw_text),
                    speaker=speaker or "speaker01",
                    language=language,
                    duration_sec=duration,
                    sample_rate=sample_rate,
                    split="train",
                    source_format="f5_csv",
                    source_id=audio_path.stem or f"row_{index:04d}",
                )
            )
    return records


def import_from_f5_json(
    source: Path,
    speaker: str | None = None,
    language: str = "vi",
) -> list[ManifestRecord]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {source}")
    records: list[ManifestRecord] = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            continue
        raw_audio = row.get("audio_path") or row.get("audio_file")
        raw_text = row.get("text")
        if not raw_audio or raw_text is None:
            continue
        audio_path = resolve_relative_audio(str(raw_audio), source.parent)
        duration = row.get("duration")
        sample_rate = row.get("sample_rate")
        if duration is None or sample_rate is None:
            try:
                audio_info = probe_wav(audio_path)
                duration = round(audio_info.duration_sec, 6)
                sample_rate = audio_info.sample_rate
            except Exception:
                duration = duration if duration is not None else None
                sample_rate = sample_rate if sample_rate is not None else None
        records.append(
            ManifestRecord(
                audio_file=str(audio_path.resolve()),
                text=normalize_text(str(raw_text)),
                speaker=str(row.get("speaker") or speaker or "speaker01"),
                language=str(row.get("language") or language),
                duration_sec=float(duration) if duration is not None else None,
                sample_rate=int(sample_rate) if sample_rate is not None else None,
                split=str(row.get("split") or "train"),
                source_format="f5_json",
                source_id=str(row.get("source_id") or audio_path.stem or f"row_{index:04d}"),
            )
        )
    return records
