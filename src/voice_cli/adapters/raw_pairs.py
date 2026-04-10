from __future__ import annotations

from pathlib import Path

from voice_cli.io import normalize_text, parse_srt_text, probe_wav, read_text_file
from voice_cli.manifest import ManifestRecord


def import_from_raw_pairs(
    source: Path,
    speaker: str | None = None,
    language: str = "vi",
    normalize_unicode: bool = True,
) -> list[ManifestRecord]:
    if source.is_file():
        wav_files = [source]
    else:
        wav_files = sorted({path.resolve() for path in source.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"})

    records: list[ManifestRecord] = []
    for wav_path in wav_files:
        stem = wav_path.with_suffix("")
        txt_path = stem.with_suffix(".txt")
        srt_path = stem.with_suffix(".srt")
        transcript = None
        if txt_path.exists():
            transcript = normalize_text(read_text_file(txt_path), normalize_unicode=normalize_unicode)
        elif srt_path.exists():
            transcript = parse_srt_text(read_text_file(srt_path), normalize_unicode=normalize_unicode)
        if transcript is None:
            continue
        duration = None
        sample_rate = None
        try:
            audio_info = probe_wav(wav_path)
            duration = round(audio_info.duration_sec, 6)
            sample_rate = audio_info.sample_rate
        except Exception:
            pass
        records.append(
            ManifestRecord(
                audio_file=str(wav_path.resolve()),
                text=transcript,
                speaker=speaker or "speaker01",
                language=language,
                duration_sec=duration,
                sample_rate=sample_rate,
                split="train",
                source_format="raw_pairs",
                source_id=wav_path.stem,
            )
        )
    return records
