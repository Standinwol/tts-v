from __future__ import annotations

from pathlib import Path

from voice_cli.io import normalize_text, probe_wav, read_text_file, resolve_relative_audio
from voice_cli.manifest import ManifestRecord


def import_from_esd_list(
    source: Path,
    speaker: str | None = None,
    language: str | None = None,
    normalize_unicode: bool = True,
) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for index, line in enumerate(read_text_file(source).splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split("|", 3)
        if len(parts) != 4:
            continue
        raw_audio, row_speaker, row_language, row_text = parts
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
                text=normalize_text(row_text, normalize_unicode=normalize_unicode),
                speaker=speaker or row_speaker or "speaker01",
                language=(language or row_language or "vi").lower(),
                duration_sec=duration,
                sample_rate=sample_rate,
                split="train",
                source_format="gpt_sovits",
                source_id=audio_path.stem or f"row_{index:04d}",
            )
        )
    return records
