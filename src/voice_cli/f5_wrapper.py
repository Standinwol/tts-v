from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
from typing import Iterable

from voice_cli.config import F5Config, InferConfig, TrainConfig
from voice_cli.console import console
from voice_cli.io import ensure_parent, sanitize_name, write_json


@dataclass(slots=True)
class F5Runtime:
    python_exe: str
    root: Path | None
    prepare_script: Path | None
    train_script: Path | None
    infer_script: Path | None


def resolve_runtime(
    config: F5Config,
    python_exe: str | None = None,
    f5_root: Path | None = None,
    prepare_script: Path | None = None,
    train_script: Path | None = None,
    infer_script: Path | None = None,
) -> F5Runtime:
    root = f5_root or config.root
    derived_prepare = prepare_script or config.prepare_script
    derived_train = train_script or config.train_script
    derived_infer = infer_script or config.infer_script
    if root is not None:
        derived_prepare = derived_prepare or (root / "src" / "f5_tts" / "train" / "datasets" / "prepare_csv_wavs.py")
        derived_train = derived_train or (root / "src" / "f5_tts" / "train" / "finetune_cli.py")
        derived_infer = derived_infer or (root / "src" / "f5_tts" / "infer" / "infer_cli.py")
    return F5Runtime(
        python_exe=python_exe or config.python_exe,
        root=root,
        prepare_script=derived_prepare,
        train_script=derived_train,
        infer_script=derived_infer,
    )


def _command_display(command: Iterable[str]) -> str:
    return shlex.join([str(part) for part in command])


def run_command(command: list[str], cwd: Path | None, log_path: Path, dry_run: bool = False) -> dict:
    ensure_parent(log_path)
    payload = {
        "command": [str(part) for part in command],
        "cwd": str(cwd) if cwd else None,
        "dry_run": dry_run,
    }
    if dry_run:
        write_json(log_path.with_suffix(".json"), payload)
        return payload

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {_command_display(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            console.print(line, end="")
            handle.write(line)
        return_code = process.wait()
        payload["return_code"] = return_code
        write_json(log_path.with_suffix(".json"), payload)
        if return_code != 0:
            raise RuntimeError(f"Command failed with exit code {return_code}: {_command_display(command)}")
    return payload


def build_prepare_command(runtime: F5Runtime, input_csv: Path, output_dir: Path) -> list[str]:
    if runtime.prepare_script is None:
        raise ValueError("Unable to resolve F5 prepare_csv_wavs.py. Pass --f5-root or --prepare-script.")
    return [
        runtime.python_exe,
        str(runtime.prepare_script),
        str(input_csv),
        str(output_dir),
    ]


def build_train_command(
    runtime: F5Runtime,
    train_config: TrainConfig,
    dataset_name: str,
    pretrain: Path,
) -> list[str]:
    if runtime.train_script is None:
        raise ValueError("Unable to resolve F5 finetune_cli.py. Pass --f5-root or --train-script.")
    command = [
        runtime.python_exe,
        str(runtime.train_script),
        "--exp_name",
        train_config.exp_name,
        "--dataset_name",
        dataset_name,
        "--learning_rate",
        str(train_config.learning_rate),
        "--batch_size_per_gpu",
        str(train_config.batch_size_per_gpu),
        "--batch_size_type",
        train_config.batch_size_type,
        "--max_samples",
        str(train_config.max_samples),
        "--grad_accumulation_steps",
        str(train_config.grad_accumulation_steps),
        "--max_grad_norm",
        str(train_config.max_grad_norm),
        "--epochs",
        str(train_config.epochs),
        "--num_warmup_updates",
        str(train_config.num_warmup_updates),
        "--save_per_updates",
        str(train_config.save_per_updates),
        "--keep_last_n_checkpoints",
        str(train_config.keep_last_n_checkpoints),
        "--last_per_updates",
        str(train_config.last_per_updates),
        "--pretrain",
        str(pretrain),
        "--tokenizer",
        train_config.tokenizer,
    ]
    if train_config.finetune:
        command.append("--finetune")
    if train_config.logger:
        command.extend(["--logger", train_config.logger])
    if train_config.log_samples:
        command.append("--log_samples")
    if train_config.bnb_optimizer:
        command.append("--bnb_optimizer")
    return command


def build_infer_command(
    runtime: F5Runtime,
    infer_config: InferConfig,
    checkpoint: Path,
    ref_audio: Path,
    ref_text: str | None,
    gen_text: str,
    output_file: Path,
    seed: int | None,
    speed: float | None,
    nfe_step: int | None,
    remove_silence: bool,
    auto_transcribe_ref: bool,
    vocab_file: Path | None,
    vocoder_name: str | None,
    model: str | None,
) -> list[str]:
    if runtime.infer_script is None:
        raise ValueError("Unable to resolve F5 infer_cli.py. Pass --f5-root or --infer-script.")
    effective_model = model or infer_config.model
    if ref_text is None and auto_transcribe_ref:
        ref_text = ""
    command = [
        runtime.python_exe,
        str(runtime.infer_script),
        "--model",
        effective_model,
        "--ckpt_file",
        str(checkpoint),
        "--ref_audio",
        str(ref_audio),
        "--ref_text",
        ref_text or "",
        "--gen_text",
        gen_text,
        "--output_dir",
        str(output_file.parent),
        "--output_file",
        output_file.name,
        "--nfe_step",
        str(nfe_step if nfe_step is not None else infer_config.default_nfe_step),
        "--speed",
        str(speed if speed is not None else infer_config.default_speed),
        "--vocoder_name",
        vocoder_name or infer_config.default_vocoder_name,
    ]
    if vocab_file is not None:
        command.extend(["--vocab_file", str(vocab_file)])
    if seed is not None:
        command.extend(["--seed", str(seed)])
    if remove_silence:
        command.append("--remove_silence")
    return command


def sanitize_dataset_name(run_id: str) -> str:
    return sanitize_name(run_id)
