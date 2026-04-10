from __future__ import annotations

from pathlib import Path

import typer

from voice_cli.console import console
from voice_cli.import_data import run_import
from voice_cli.infer import run_infer
from voice_cli.normalize_numbers import run_normalize_numbers
from voice_cli.train import run_train
from voice_cli.validate import validate_manifest

app = typer.Typer(add_completion=False, help="CLI workflow for Vietnamese F5-TTS projects.")


@app.command("import-data")
def import_data_command(
    source: Path = typer.Option(..., help="Source file or directory."),
    format: str = typer.Option(..., "--format", help="One of: f5_zip, f5_csv, f5_json, gpt_sovits, raw_pairs."),
    speaker: str | None = typer.Option(None, help="Override speaker id."),
    language: str | None = typer.Option(None, help="Override language code."),
    config: Path = typer.Option(Path("configs/project.yaml"), help="Project config path."),
    append: bool = typer.Option(False, help="Append to the existing master manifest."),
    overwrite_imports: bool = typer.Option(False, help="Re-extract imported archives if needed."),
) -> None:
    try:
        summary = run_import(
            project_config_path=config,
            source=source,
            source_format=format,
            speaker=speaker,
            language=language,
            append=append,
            overwrite_imports=overwrite_imports,
        )
    except Exception as exc:
        console.print(f"[red]Import failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(summary)


@app.command("validate-data")
def validate_data_command(
    manifest: Path = typer.Option(Path("data/manifests/master.jsonl"), help="Master manifest to validate."),
    config: Path = typer.Option(Path("configs/project.yaml"), help="Project config path."),
    min_sec: float | None = typer.Option(None, help="Override minimum duration."),
    max_sec: float | None = typer.Option(None, help="Override maximum duration."),
) -> None:
    try:
        summary = validate_manifest(
            project_config_path=config,
            manifest_path=manifest,
            min_sec=min_sec,
            max_sec=max_sec,
        )
    except Exception as exc:
        console.print(f"[red]Validation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(summary)


@app.command("train")
def train_command(
    manifest: Path = typer.Option(Path("data/manifests/validated.jsonl"), help="Validated manifest path."),
    pretrain: Path = typer.Option(..., help="Path to the upstream pretrained checkpoint."),
    run_name: str = typer.Option(..., help="Logical run name."),
    project_config: Path = typer.Option(Path("configs/project.yaml"), help="Project config path."),
    train_config: Path = typer.Option(Path("configs/train.yaml"), help="Train config path."),
    f5_root: Path | None = typer.Option(None, help="Local F5-TTS repository root."),
    python_exe: str | None = typer.Option(None, help="Python executable used to run upstream scripts."),
    dry_run: bool = typer.Option(False, help="Build commands without executing them."),
    resume: str | None = typer.Option(None, help='Resume from a previous run id, or use "latest".'),
) -> None:
    try:
        summary = run_train(
            project_config_path=project_config,
            train_config_path=train_config,
            manifest_path=manifest,
            pretrain=pretrain,
            run_name=run_name,
            f5_root=f5_root,
            python_exe=python_exe,
            dry_run=dry_run,
            resume=resume,
        )
    except Exception as exc:
        console.print(f"[red]Train setup failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(summary)


@app.command("normalize-vi-numbers")
def normalize_vi_numbers_command(
    source: Path = typer.Option(..., help="Source text file. Supports .txt, .srt, .csv, .jsonl"),
    output: Path | None = typer.Option(None, help="Optional output path. Defaults to <source>.normalized.<ext>"),
    format: str = typer.Option("auto", "--format", help="One of: auto, text, srt, csv, jsonl."),
    overwrite: bool = typer.Option(False, help="Rewrite the source file in place."),
) -> None:
    try:
        summary = run_normalize_numbers(
            source=source,
            output=output,
            overwrite=overwrite,
            format_name=format,
        )
    except Exception as exc:
        console.print(f"[red]Normalize failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(summary)


@app.command("infer")
def infer_command(
    checkpoint: Path = typer.Option(..., help="Fine-tuned checkpoint path."),
    ref_audio: Path = typer.Option(..., help="Reference audio path."),
    text: str = typer.Option(..., help="Generated text."),
    out: Path = typer.Option(..., help="Output wav path."),
    ref_text: str | None = typer.Option(None, help="Reference transcription."),
    project_config: Path = typer.Option(Path("configs/project.yaml"), help="Project config path."),
    infer_config: Path = typer.Option(Path("configs/infer.yaml"), help="Infer config path."),
    f5_root: Path | None = typer.Option(None, help="Local F5-TTS repository root."),
    python_exe: str | None = typer.Option(None, help="Python executable used to run upstream scripts."),
    seed: int | None = typer.Option(None, help="Optional random seed."),
    speed: float | None = typer.Option(None, help="Optional speech speed override."),
    nfe_step: int | None = typer.Option(None, help="Optional denoising steps override."),
    remove_silence: bool = typer.Option(False, help="Trim silence from generated wav."),
    auto_transcribe_ref: bool = typer.Option(False, help="Allow upstream ASR for missing ref_text."),
    vocab_file: Path | None = typer.Option(None, help="Optional vocab.txt override."),
    vocoder_name: str | None = typer.Option(None, help="Optional vocoder override."),
    model: str | None = typer.Option(None, help="Optional upstream model override."),
    dry_run: bool = typer.Option(False, help="Build commands without executing them."),
) -> None:
    try:
        summary = run_infer(
            project_config_path=project_config,
            infer_config_path=infer_config,
            checkpoint=checkpoint,
            ref_audio=ref_audio,
            ref_text=ref_text,
            text=text,
            out=out,
            f5_root=f5_root,
            python_exe=python_exe,
            seed=seed,
            speed=speed,
            nfe_step=nfe_step,
            remove_silence=remove_silence,
            auto_transcribe_ref=auto_transcribe_ref,
            vocab_file=vocab_file,
            vocoder_name=vocoder_name,
            model=model,
            dry_run=dry_run,
        )
    except Exception as exc:
        console.print(f"[red]Infer setup failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(summary)


if __name__ == "__main__":
    app()
