from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load config files.") from exc

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return loaded


def _path(value: str | None, default: str) -> Path:
    return Path(value or default)


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 24000
    channels: int = 1
    format: str = "wav"
    min_sec: float = 3.0
    max_sec: float = 20.0


@dataclass(slots=True)
class TextConfig:
    language: str = "vi"
    normalize_unicode: bool = True


@dataclass(slots=True)
class ManifestConfig:
    master: Path = Path("./data/manifests/master.jsonl")
    validated: Path = Path("./data/manifests/validated.jsonl")
    invalid: Path = Path("./data/manifests/invalid.jsonl")
    train_f5: Path = Path("./data/manifests/train_f5.csv")
    val_f5: Path = Path("./data/manifests/val_f5.csv")


@dataclass(slots=True)
class RuntimePathsConfig:
    imports_root: Path = Path("./data/imports")
    reports_dir: Path = Path("./reports")
    runs_dir: Path = Path("./runs")
    logs_dir: Path = Path("./logs")


@dataclass(slots=True)
class F5Config:
    python_exe: str = "python"
    root: Path | None = None
    prepare_script: Path | None = None
    train_script: Path | None = None
    infer_script: Path | None = None


@dataclass(slots=True)
class ProjectConfig:
    audio: AudioConfig
    text: TextConfig
    manifest: ManifestConfig
    paths: RuntimePathsConfig
    f5: F5Config


@dataclass(slots=True)
class TrainConfig:
    exp_name: str = "F5TTS_v1_Base"
    tokenizer: str = "char"
    learning_rate: float = 1.0e-5
    batch_size_per_gpu: int = 3200
    batch_size_type: str = "frame"
    max_samples: int = 64
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    epochs: int = 100
    num_warmup_updates: int = 20000
    save_per_updates: int = 2000
    keep_last_n_checkpoints: int = 5
    last_per_updates: int = 500
    finetune: bool = True
    logger: str | None = "tensorboard"
    log_samples: bool = False
    bnb_optimizer: bool = False
    val_ratio: float = 0.05


@dataclass(slots=True)
class InferConfig:
    model: str = "F5TTS_v1_Base"
    require_ref_text: bool = True
    allow_auto_transcribe_ref: bool = False
    default_speed: float = 1.0
    default_remove_silence: bool = False
    default_nfe_step: int = 32
    default_vocoder_name: str = "vocos"


def load_project_config(path: Path) -> ProjectConfig:
    raw = _load_yaml(path)
    audio = raw.get("audio", {})
    text = raw.get("text", {})
    manifest = raw.get("manifest", {})
    paths = raw.get("paths", {})
    f5 = raw.get("f5", {})
    return ProjectConfig(
        audio=AudioConfig(
            sample_rate=int(audio.get("sample_rate", 24000)),
            channels=int(audio.get("channels", 1)),
            format=str(audio.get("format", "wav")),
            min_sec=float(audio.get("min_sec", 3.0)),
            max_sec=float(audio.get("max_sec", 20.0)),
        ),
        text=TextConfig(
            language=str(text.get("language", "vi")),
            normalize_unicode=bool(text.get("normalize_unicode", True)),
        ),
        manifest=ManifestConfig(
            master=_path(manifest.get("master"), "./data/manifests/master.jsonl"),
            validated=_path(manifest.get("validated"), "./data/manifests/validated.jsonl"),
            invalid=_path(manifest.get("invalid"), "./data/manifests/invalid.jsonl"),
            train_f5=_path(manifest.get("train_f5"), "./data/manifests/train_f5.csv"),
            val_f5=_path(manifest.get("val_f5"), "./data/manifests/val_f5.csv"),
        ),
        paths=RuntimePathsConfig(
            imports_root=_path(paths.get("imports_root"), "./data/imports"),
            reports_dir=_path(paths.get("reports_dir"), "./reports"),
            runs_dir=_path(paths.get("runs_dir"), "./runs"),
            logs_dir=_path(paths.get("logs_dir"), "./logs"),
        ),
        f5=F5Config(
            python_exe=str(f5.get("python_exe") or "python"),
            root=Path(f5["root"]) if f5.get("root") else None,
            prepare_script=Path(f5["prepare_script"]) if f5.get("prepare_script") else None,
            train_script=Path(f5["train_script"]) if f5.get("train_script") else None,
            infer_script=Path(f5["infer_script"]) if f5.get("infer_script") else None,
        ),
    )


def load_train_config(path: Path) -> TrainConfig:
    raw = _load_yaml(path)
    return TrainConfig(
        exp_name=str(raw.get("exp_name", "F5TTS_v1_Base")),
        tokenizer=str(raw.get("tokenizer", "char")),
        learning_rate=float(raw.get("learning_rate", 1.0e-5)),
        batch_size_per_gpu=int(raw.get("batch_size_per_gpu", 3200)),
        batch_size_type=str(raw.get("batch_size_type", "frame")),
        max_samples=int(raw.get("max_samples", 64)),
        grad_accumulation_steps=int(raw.get("grad_accumulation_steps", 1)),
        max_grad_norm=float(raw.get("max_grad_norm", 1.0)),
        epochs=int(raw.get("epochs", 100)),
        num_warmup_updates=int(raw.get("num_warmup_updates", 20000)),
        save_per_updates=int(raw.get("save_per_updates", 2000)),
        keep_last_n_checkpoints=int(raw.get("keep_last_n_checkpoints", 5)),
        last_per_updates=int(raw.get("last_per_updates", 500)),
        finetune=bool(raw.get("finetune", True)),
        logger=raw.get("logger"),
        log_samples=bool(raw.get("log_samples", False)),
        bnb_optimizer=bool(raw.get("bnb_optimizer", False)),
        val_ratio=float(raw.get("val_ratio", 0.05)),
    )


def load_infer_config(path: Path) -> InferConfig:
    raw = _load_yaml(path)
    return InferConfig(
        model=str(raw.get("model", "F5TTS_v1_Base")),
        require_ref_text=bool(raw.get("require_ref_text", True)),
        allow_auto_transcribe_ref=bool(raw.get("allow_auto_transcribe_ref", False)),
        default_speed=float(raw.get("default_speed", 1.0)),
        default_remove_silence=bool(raw.get("default_remove_silence", False)),
        default_nfe_step=int(raw.get("default_nfe_step", 32)),
        default_vocoder_name=str(raw.get("default_vocoder_name", "vocos")),
    )
