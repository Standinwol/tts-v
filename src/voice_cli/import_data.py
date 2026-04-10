from __future__ import annotations

from pathlib import Path

from voice_cli.adapters.f5_dataset import import_from_f5_csv, import_from_f5_json, import_from_f5_zip
from voice_cli.adapters.gpt_sovits import import_from_esd_list
from voice_cli.adapters.raw_pairs import import_from_raw_pairs
from voice_cli.config import load_project_config
from voice_cli.manifest import ManifestRecord, load_jsonl, merge_records, write_jsonl


def run_import(
    project_config_path: Path,
    source: Path,
    source_format: str,
    speaker: str | None,
    language: str | None,
    append: bool,
    overwrite_imports: bool,
) -> dict:
    project = load_project_config(project_config_path)
    effective_language = language or project.text.language
    format_name = source_format.lower()

    if format_name == "f5_zip":
        imported = import_from_f5_zip(
            source=source,
            imports_root=project.paths.imports_root,
            speaker=speaker,
            language=effective_language,
            overwrite=overwrite_imports,
        )
    elif format_name == "f5_csv":
        imported = import_from_f5_csv(source, speaker=speaker, language=effective_language)
    elif format_name == "f5_json":
        imported = import_from_f5_json(source, speaker=speaker, language=effective_language)
    elif format_name == "gpt_sovits":
        imported = import_from_esd_list(source, speaker=speaker, language=effective_language)
    elif format_name == "raw_pairs":
        imported = import_from_raw_pairs(source, speaker=speaker, language=effective_language)
    else:
        raise ValueError(f"Unsupported format: {source_format}")

    existing: list[ManifestRecord] = load_jsonl(project.manifest.master) if append else []
    merged = merge_records(existing, imported)
    write_jsonl(project.manifest.master, merged)
    return {
        "source": str(source),
        "format": format_name,
        "imported_records": len(imported),
        "total_records": len(merged),
        "manifest": str(project.manifest.master),
    }
