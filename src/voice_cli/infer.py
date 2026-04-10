from __future__ import annotations

from pathlib import Path

from voice_cli.config import load_infer_config, load_project_config
from voice_cli.f5_wrapper import build_infer_command, resolve_runtime, run_command
from voice_cli.io import ensure_dir, timestamp_id, write_json


def run_infer(
    project_config_path: Path,
    infer_config_path: Path,
    checkpoint: Path,
    ref_audio: Path,
    ref_text: str | None,
    text: str,
    out: Path,
    f5_root: Path | None,
    python_exe: str | None,
    seed: int | None,
    speed: float | None,
    nfe_step: int | None,
    remove_silence: bool,
    auto_transcribe_ref: bool,
    vocab_file: Path | None,
    vocoder_name: str | None,
    model: str | None,
    dry_run: bool,
) -> dict:
    project = load_project_config(project_config_path)
    infer_config = load_infer_config(infer_config_path)
    runtime = resolve_runtime(project.f5, python_exe=python_exe, f5_root=f5_root)
    missing_ref_text = ref_text is None or not ref_text.strip()

    if missing_ref_text:
        if auto_transcribe_ref:
            if not infer_config.allow_auto_transcribe_ref:
                raise ValueError("Auto transcription is disabled by configs/infer.yaml.")
        elif infer_config.require_ref_text:
            raise ValueError("ref_text is required unless --auto-transcribe-ref is enabled.")

    ensure_dir(out.parent)
    run_id = timestamp_id("infer")
    run_dir = project.paths.runs_dir / run_id
    ensure_dir(run_dir)

    command = build_infer_command(
        runtime=runtime,
        infer_config=infer_config,
        checkpoint=checkpoint,
        ref_audio=ref_audio,
        ref_text=ref_text,
        gen_text=text,
        output_file=out,
        seed=seed,
        speed=speed,
        nfe_step=nfe_step,
        remove_silence=remove_silence,
        auto_transcribe_ref=auto_transcribe_ref,
        vocab_file=vocab_file,
        vocoder_name=vocoder_name,
        model=model,
    )
    metadata = {
        "kind": "infer",
        "run_id": run_id,
        "checkpoint": str(checkpoint),
        "ref_audio": str(ref_audio),
        "ref_text": ref_text,
        "text": text,
        "out": str(out),
        "seed": seed,
        "speed": speed if speed is not None else infer_config.default_speed,
        "nfe_step": nfe_step if nfe_step is not None else infer_config.default_nfe_step,
        "remove_silence": remove_silence,
        "auto_transcribe_ref": auto_transcribe_ref,
        "command": command,
    }
    write_json(run_dir / "run.json", metadata)
    write_json(out.with_suffix(".json"), metadata)
    run_command(
        command=command,
        cwd=runtime.root,
        log_path=run_dir / "infer.log",
        dry_run=dry_run,
    )
    return metadata
