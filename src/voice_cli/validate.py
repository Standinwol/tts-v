from __future__ import annotations

from collections import Counter
from dataclasses import replace
import csv
from pathlib import Path

from voice_cli.config import ProjectConfig, load_project_config
from voice_cli.io import normalize_text, probe_wav, write_json
from voice_cli.manifest import ManifestRecord, load_jsonl, write_jsonl


def _report_paths(project: ProjectConfig) -> tuple[Path, Path]:
    return (
        project.paths.reports_dir / "validate_summary.json",
        project.paths.reports_dir / "validate_summary.csv",
    )


def _validate_record(
    record: ManifestRecord,
    expected_sample_rate: int,
    expected_channels: int,
    min_sec: float,
    max_sec: float,
    seen_paths: set[str],
    normalize_unicode: bool,
) -> tuple[ManifestRecord, list[str]]:
    reasons: list[str] = []
    normalized_text = normalize_text(record.text, normalize_unicode=normalize_unicode)

    if not normalized_text:
        reasons.append("empty_text")

    if record.canonical_key in seen_paths:
        reasons.append("duplicate_audio_path")
    else:
        seen_paths.add(record.canonical_key)

    audio_path = record.audio_path
    if not audio_path.exists():
        reasons.append("missing_audio")
        return replace(record, text=normalized_text), reasons

    try:
        audio_info = probe_wav(audio_path)
    except Exception:
        reasons.append("unreadable_audio")
        return replace(record, text=normalized_text), reasons

    if audio_info.sample_rate != expected_sample_rate:
        reasons.append("sample_rate_mismatch")
    if audio_info.channels != expected_channels:
        reasons.append("channel_mismatch")
    if audio_info.duration_sec < min_sec:
        reasons.append("too_short")
    if audio_info.duration_sec > max_sec:
        reasons.append("too_long")

    updated = replace(
        record,
        text=normalized_text,
        duration_sec=round(audio_info.duration_sec, 6),
        sample_rate=audio_info.sample_rate,
    )
    return updated, reasons


def validate_manifest(
    project_config_path: Path,
    manifest_path: Path | None = None,
    min_sec: float | None = None,
    max_sec: float | None = None,
) -> dict:
    project = load_project_config(project_config_path)
    source_manifest = manifest_path or project.manifest.master
    records = load_jsonl(source_manifest)
    validated: list[ManifestRecord] = []
    invalid: list[ManifestRecord] = []
    reasons_counter: Counter[str] = Counter()
    text_counter: Counter[str] = Counter()
    seen_paths: set[str] = set()

    lower = min_sec if min_sec is not None else project.audio.min_sec
    upper = max_sec if max_sec is not None else project.audio.max_sec

    for record in records:
        checked, reasons = _validate_record(
            record=record,
            expected_sample_rate=project.audio.sample_rate,
            expected_channels=project.audio.channels,
            min_sec=lower,
            max_sec=upper,
            seen_paths=seen_paths,
            normalize_unicode=project.text.normalize_unicode,
        )
        text_counter[checked.text] += 1
        if reasons:
            reasons_counter.update(reasons)
            invalid.append(replace(checked, status="invalid", error=";".join(reasons)))
        else:
            validated.append(replace(checked, status="ok", error=None))

    write_jsonl(project.manifest.validated, validated)
    write_jsonl(project.manifest.invalid, invalid)

    duplicate_text_entries = sum(count - 1 for count in text_counter.values() if count > 1)
    durations = [record.duration_sec for record in validated if record.duration_sec is not None]
    summary = {
        "source_manifest": str(source_manifest),
        "validated_manifest": str(project.manifest.validated),
        "invalid_manifest": str(project.manifest.invalid),
        "total_records": len(records),
        "validated_records": len(validated),
        "invalid_records": len(invalid),
        "expected_sample_rate": project.audio.sample_rate,
        "expected_channels": project.audio.channels,
        "duration_min_sec": min(durations) if durations else None,
        "duration_max_sec": max(durations) if durations else None,
        "duration_avg_sec": round(sum(durations) / len(durations), 6) if durations else None,
        "duplicate_text_entries": duplicate_text_entries,
        "invalid_reasons": dict(sorted(reasons_counter.items())),
    }

    summary_json_path, summary_csv_path = _report_paths(project)
    write_json(summary_json_path, summary)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reason", "count"])
        for reason, count in sorted(reasons_counter.items()):
            writer.writerow([reason, count])

    return summary
