from __future__ import annotations

import csv
import json
from pathlib import Path

from voice_cli.config import load_project_config, load_train_config
from voice_cli.f5_wrapper import (
    build_prepare_command,
    build_train_command,
    resolve_runtime,
    run_command,
    sanitize_dataset_name,
)
from voice_cli.io import ensure_dir, sanitize_name, timestamp_id, write_json
from voice_cli.manifest import ManifestRecord, load_jsonl, write_f5_csv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _split_records(records: list[ManifestRecord], val_ratio: float) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    explicit_val = [record for record in records if record.split.lower() in {"val", "valid", "validation"}]
    if explicit_val:
        train = [record for record in records if record not in explicit_val]
        return train, explicit_val

    sorted_records = sorted(records, key=lambda item: item.audio_file.lower())
    if len(sorted_records) < 2:
        return sorted_records, []
    val_count = max(1, int(round(len(sorted_records) * val_ratio)))
    val_count = min(val_count, len(sorted_records) - 1)
    return sorted_records[:-val_count], sorted_records[-val_count:]


def _write_prepare_input(metadata_path: Path, records: list[ManifestRecord]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(["audio_file", "text"])
        for record in records:
            writer.writerow([str(record.audio_path.resolve()), record.text])


def _find_resume_target(runs_dir: Path, run_name: str, resume: str | None) -> dict | None:
    if not resume:
        return None
    candidates: list[Path] = []
    if resume == "latest":
        candidates = sorted(runs_dir.glob(f"{run_name}_*"), reverse=True)
    else:
        candidate = runs_dir / resume
        if candidate.exists():
            candidates = [candidate]
    for candidate in candidates:
        metadata_path = candidate / "run.json"
        if metadata_path.exists():
            return json.loads(metadata_path.read_text(encoding="utf-8"))
    return None


def _should_use_vi_prepare(language: str, tokenizer: str) -> bool:
    return language.lower() == "vi" and tokenizer.lower() == "char"


def _vi_prepare_script_path() -> Path:
    return _repo_root() / "scripts" / "prepare_vi_csv_wavs.py"


def _build_vi_prepare_command(
    *,
    python_exe: str,
    script_path: Path,
    input_csv: Path,
    output_dir: Path,
    pretrain: Path,
) -> list[str]:
    return [
        python_exe,
        str(script_path),
        str(input_csv),
        str(output_dir),
        "--pretrain-checkpoint",
        str(pretrain),
    ]


def run_train(
    project_config_path: Path,
    train_config_path: Path,
    manifest_path: Path,
    pretrain: Path,
    run_name: str,
    f5_root: Path | None,
    python_exe: str | None,
    dry_run: bool,
    resume: str | None,
) -> dict:
    project = load_project_config(project_config_path)
    train_config = load_train_config(train_config_path)
    runtime = resolve_runtime(project.f5, python_exe=python_exe, f5_root=f5_root)
    records = [record for record in load_jsonl(manifest_path) if record.status == "ok"]
    if not records:
        raise ValueError(f"No validated records with status=ok found in {manifest_path}")

    safe_run_name = sanitize_name(run_name)
    resume_metadata = _find_resume_target(project.paths.runs_dir, safe_run_name, resume)
    if resume_metadata:
        dataset_name = str(resume_metadata["dataset_name"])
    else:
        dataset_name = sanitize_dataset_name(timestamp_id(safe_run_name))

    run_id = timestamp_id(safe_run_name)
    run_dir = project.paths.runs_dir / run_id
    ensure_dir(run_dir)

    train_records, val_records = _split_records(records, train_config.val_ratio)
    run_train_csv = run_dir / "train_f5.csv"
    run_val_csv = run_dir / "val_f5.csv"
    write_f5_csv(run_train_csv, train_records)
    write_f5_csv(run_val_csv, val_records)
    write_f5_csv(project.manifest.train_f5, train_records)
    write_f5_csv(project.manifest.val_f5, val_records)

    prepare_input_dir = run_dir / "f5_prepare_input"
    prepare_input_csv = prepare_input_dir / "metadata.csv"
    _write_prepare_input(prepare_input_csv, train_records)

    runtime_root = runtime.root
    if runtime_root is None:
        raise ValueError("Training requires --f5-root or f5.root in configs/project.yaml")

    use_vi_prepare = _should_use_vi_prepare(project.text.language, train_config.tokenizer)
    if resume_metadata:
        prepare_output_dir = Path(str(resume_metadata.get("prepare_output_dir") or (runtime_root / "data" / dataset_name)))
        effective_pretrain = Path(str(resume_metadata.get("effective_pretrain") or pretrain))
        prepare_command = None
    elif use_vi_prepare:
        prepare_script = _vi_prepare_script_path()
        if not prepare_script.exists():
            raise FileNotFoundError(f"Vietnamese prepare script not found: {prepare_script}")
        prepare_output_dir = runtime_root / "data" / f"{dataset_name}_char"
        effective_pretrain = prepare_output_dir / f"pretrained_{pretrain.name}"
        prepare_command = _build_vi_prepare_command(
            python_exe=runtime.python_exe,
            script_path=prepare_script,
            input_csv=prepare_input_csv,
            output_dir=prepare_output_dir,
            pretrain=pretrain,
        )
    else:
        prepare_output_dir = runtime_root / "data" / dataset_name
        effective_pretrain = pretrain
        prepare_command = build_prepare_command(runtime, prepare_input_csv, prepare_output_dir)

    train_command = build_train_command(runtime, train_config, dataset_name=dataset_name, pretrain=effective_pretrain)

    metadata = {
        "kind": "train",
        "run_id": run_id,
        "run_name": safe_run_name,
        "dataset_name": dataset_name,
        "manifest_path": str(manifest_path),
        "pretrain": str(pretrain),
        "effective_pretrain": str(effective_pretrain),
        "prepare_profile": "vi_char_fresh_vocab" if use_vi_prepare else "upstream",
        "prepare_output_dir": str(prepare_output_dir),
        "expected_checkpoint_dir": str(runtime_root / "ckpts" / dataset_name),
        "resume": resume,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "project_config": str(project_config_path),
        "train_config": str(train_config_path),
        "runtime": {
            "python_exe": runtime.python_exe,
            "root": str(runtime.root) if runtime.root else None,
            "prepare_script": str(runtime.prepare_script) if runtime.prepare_script else None,
            "train_script": str(runtime.train_script) if runtime.train_script else None,
        },
        "commands": {
            "prepare": prepare_command,
            "train": train_command,
        },
    }
    write_json(run_dir / "run.json", metadata)

    if not resume_metadata and prepare_command is not None:
        run_command(
            command=prepare_command,
            cwd=runtime_root,
            log_path=run_dir / "prepare.log",
            dry_run=dry_run,
        )
    run_command(
        command=train_command,
        cwd=runtime_root,
        log_path=run_dir / "train.log",
        dry_run=dry_run,
    )
    return metadata
