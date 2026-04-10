from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing
import shutil
import subprocess
import sys
import unicodedata
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path

import soundfile as sf
import torchaudio
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


DEFAULT_DATASET_NAME = "Emilia_ZH_EN"
DEFAULT_BASE_TOKENIZER = "pinyin"
SUMMARY_FILENAME = "prepare_vi_summary.json"
BATCH_SIZE = 100
CHUNK_SIZE = 100
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)
THREAD_NAME_PREFIX = "ViAudioProcessor"

executor = None


def normalize_text_vi(text: str) -> str:
    cleaned = text.replace("\ufeff", " ")
    cleaned = unicodedata.normalize("NFC", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def get_default_base_vocab_path() -> Path:
    return Path(files("f5_tts").joinpath(f"../../data/{DEFAULT_DATASET_NAME}_{DEFAULT_BASE_TOKENIZER}/vocab.txt"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an F5-TTS dataset for Vietnamese char training without pinyin conversion."
    )
    parser.add_argument("input_csv", type=Path, help="CSV with header audio_file|text and absolute audio paths.")
    parser.add_argument("output_dir", type=Path, help="Prepared dataset output directory, usually data/<dataset_name>_char.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of worker threads. Default: min({MAX_WORKERS}, file count).",
    )
    parser.add_argument(
        "--base-vocab",
        type=Path,
        default=None,
        help="Base vocab.txt used as the stable char index order. Defaults to F5-TTS Emilia_ZH_EN_pinyin vocab.",
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=None,
        help="Optional pretrained checkpoint to copy or extend so it matches the generated Vietnamese char vocab.",
    )
    parser.add_argument(
        "--allow-missing-vocab",
        action="store_true",
        help="Allow missing chars without preparing an adjusted checkpoint. Unsafe for finetuning unless you know the base vocab already covers them.",
    )
    return parser


@contextmanager
def graceful_exit():
    def signal_handler(signum, frame):  # type: ignore[unused-argument]
        print("\nReceived termination signal. Cleaning up...")
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    import signal

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        yield
    finally:
        if executor is not None:
            executor.shutdown(wait=False)


def is_csv_input(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".csv"


def read_audio_text_pairs(csv_file_path: Path) -> list[tuple[str, str]]:
    if not is_csv_input(csv_file_path):
        raise ValueError(f"input must be a .csv file: {csv_file_path}")

    pairs: list[tuple[str, str]] = []
    with csv_file_path.open("r", encoding="utf-8-sig", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter="|")
        header = next(reader, None)
        if header is None:
            return pairs
        if len(header) < 2 or header[0].strip() != "audio_file" or header[1].strip() != "text":
            raise ValueError("CSV header must be: audio_file|text")

        for row_idx, row in enumerate(reader, start=2):
            if len(row) < 2:
                continue
            audio_file = row[0].strip()
            text = normalize_text_vi(row[1])
            if not audio_file or not text:
                continue
            audio_path = Path(audio_file).expanduser()
            if not audio_path.is_absolute():
                raise ValueError(f"audio_file must be an absolute path (row {row_idx}): {audio_file}")
            pairs.append((audio_path.as_posix(), text))
    return pairs


def get_audio_duration(audio_path: str, timeout: int = 5) -> float:
    try:
        return sf.info(audio_path).duration
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=timeout,
        )
        duration_str = result.stdout.strip()
        if duration_str:
            return float(duration_str)
    except Exception:
        pass

    info = torchaudio.info(audio_path)
    if info.sample_rate <= 0:
        raise RuntimeError(f"failed to get duration for {audio_path}: invalid sample rate")
    return info.num_frames / info.sample_rate


def process_audio_file(audio_path: str, text: str) -> tuple[str, str, float] | None:
    if not Path(audio_path).exists():
        print(f"audio {audio_path} not found, skipping")
        return None
    try:
        duration = get_audio_duration(audio_path)
        if duration <= 0:
            raise ValueError(f"Duration {duration} is non-positive.")
        return (audio_path, text, duration)
    except Exception as exc:
        print(f"Warning: failed to process {audio_path}: {exc}. Skipping corrupt file.")
        return None


def prepare_vi_csv_wavs(input_csv: Path, num_workers: int | None = None) -> tuple[list[dict], list[float], set[str]]:
    global executor

    audio_path_text_pairs = read_audio_text_pairs(input_csv)
    total_files = len(audio_path_text_pairs)
    if total_files == 0:
        raise RuntimeError("No valid rows found in CSV.")

    worker_count = num_workers if num_workers is not None else min(MAX_WORKERS, total_files)
    print(f"\nProcessing {total_files} audio files using {worker_count} workers...")

    with graceful_exit():
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix=THREAD_NAME_PREFIX,
        ) as exec:
            executor = exec
            results: list[tuple[str, str, float]] = []
            for index in range(0, len(audio_path_text_pairs), CHUNK_SIZE):
                chunk = audio_path_text_pairs[index : index + CHUNK_SIZE]
                futures = [executor.submit(process_audio_file, pair[0], pair[1]) for pair in chunk]
                for future in tqdm(
                    futures,
                    total=len(chunk),
                    desc=f"Processing chunk {index // CHUNK_SIZE + 1}/{(total_files + CHUNK_SIZE - 1) // CHUNK_SIZE}",
                ):
                    result = future.result()
                    if result is not None:
                        results.append(result)
            executor = None

    processed = [result for result in results if result is not None]
    if not processed:
        raise RuntimeError("No valid audio files were processed.")

    prepared_rows: list[dict] = []
    durations: list[float] = []
    vocab_set: set[str] = set()
    for audio_path, text, duration in processed:
        prepared_rows.append({"audio_path": audio_path, "text": text, "duration": duration})
        durations.append(duration)
        vocab_set.update(text)
    return prepared_rows, durations, vocab_set


def read_vocab_chars(path: Path) -> list[str]:
    chars: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            chars.append(line.rstrip("\n"))
    return chars


def merge_vocab(base_vocab_chars: list[str], text_vocab_set: set[str]) -> tuple[list[str], list[str]]:
    base_vocab = list(base_vocab_chars)
    if not base_vocab:
        base_vocab = [" "]
    if base_vocab[0] != " ":
        if " " in base_vocab:
            base_vocab.remove(" ")
        base_vocab.insert(0, " ")

    seen = set(base_vocab)
    missing_chars = sorted(char for char in text_vocab_set if char not in seen)
    merged_vocab = base_vocab + missing_chars
    return merged_vocab, missing_chars


def write_vocab(path: Path, vocab_chars: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for char in vocab_chars:
            handle.write(char)
            handle.write("\n")


def extend_embedding_rows(tensor, new_size: int):
    import torch

    old_size, embedding_dim = tensor.shape
    if new_size < old_size:
        raise ValueError(f"new vocab size {new_size} cannot be smaller than current size {old_size}")
    if new_size == old_size:
        return tensor

    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True, unbiased=False)
    std = torch.where(std < 1e-5, torch.full_like(std, 0.02), std)
    extra = mean + torch.randn((new_size - old_size, embedding_dim), dtype=tensor.dtype) * std
    return torch.cat([tensor, extra], dim=0)


def should_resize_tensor(name: str, tensor, checkpoint_vocab_size: int) -> bool:
    return (
        hasattr(tensor, "ndim")
        and tensor.ndim == 2
        and tensor.shape[0] == checkpoint_vocab_size
        and name.endswith("text_embed.text_embed.weight")
    )


def resize_checkpoint_object(obj, checkpoint_vocab_size: int, new_vocab_size: int, prefix: str = "") -> list[str]:
    changed: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                changed.extend(resize_checkpoint_object(value, checkpoint_vocab_size, new_vocab_size, dotted))
            elif should_resize_tensor(dotted, value, checkpoint_vocab_size):
                obj[key] = extend_embedding_rows(value, new_vocab_size)
                changed.append(dotted)
    return changed


def detect_checkpoint_vocab_size(payload) -> int | None:
    candidates: list[int] = []

    def collect(obj, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                dotted = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, dict):
                    collect(value, dotted)
                elif hasattr(value, "ndim") and value.ndim == 2 and dotted.endswith("text_embed.text_embed.weight"):
                    candidates.append(int(value.shape[0]))

    collect(payload)
    if not candidates:
        return None
    return max(candidates)


def prepare_adjusted_pretrain(pretrain_path: Path, output_path: Path, merged_vocab: list[str], missing_chars: list[str]) -> dict:
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = pretrain_path.suffix.lower()

    if suffix == ".safetensors":
        from safetensors.torch import load_file, save_file

        tensors = load_file(str(pretrain_path))
        checkpoint_vocab_size = detect_checkpoint_vocab_size(tensors)
        if checkpoint_vocab_size is None:
            raise RuntimeError(f"Could not detect text embedding weights in {pretrain_path}")
        changed_keys = resize_checkpoint_object(tensors, checkpoint_vocab_size, len(merged_vocab))
        if missing_chars and not changed_keys:
            raise RuntimeError(f"No text embedding weights were resized in {pretrain_path}")
        save_file(tensors, str(output_path))
        return {
            "checkpoint_format": "safetensors",
            "checkpoint_vocab_size": checkpoint_vocab_size,
            "changed_keys": changed_keys,
        }

    if suffix == ".pt":
        payload = torch.load(pretrain_path, map_location="cpu")
        checkpoint_vocab_size = detect_checkpoint_vocab_size(payload)
        if checkpoint_vocab_size is None:
            raise RuntimeError(f"Could not detect text embedding weights in {pretrain_path}")
        changed_keys = resize_checkpoint_object(payload, checkpoint_vocab_size, len(merged_vocab))
        if missing_chars and not changed_keys:
            raise RuntimeError(f"No text embedding weights were resized in {pretrain_path}")
        torch.save(payload, output_path)
        return {
            "checkpoint_format": "pt",
            "checkpoint_vocab_size": checkpoint_vocab_size,
            "changed_keys": changed_keys,
        }

    raise ValueError(f"Unsupported checkpoint format for {pretrain_path}. Expected .pt or .safetensors")


def save_prepared_dataset(
    output_dir: Path,
    rows: list[dict],
    durations: list[float],
    vocab_chars: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {output_dir} ...")

    with ArrowWriter(path=(output_dir / "raw.arrow").as_posix()) as writer:
        for row in tqdm(rows, desc="Writing to raw.arrow ..."):
            writer.write(row)
        writer.finalize()

    with (output_dir / "duration.json").open("w", encoding="utf-8") as handle:
        json.dump({"duration": durations}, handle, ensure_ascii=False)

    write_vocab(output_dir / "vocab.txt", vocab_chars)


def main() -> int:
    args = build_parser().parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    base_vocab_path = (args.base_vocab.expanduser().resolve() if args.base_vocab else get_default_base_vocab_path().resolve())
    pretrain_path = args.pretrain_checkpoint.expanduser().resolve() if args.pretrain_checkpoint else None

    if not base_vocab_path.exists():
        raise SystemExit(f"Base vocab not found: {base_vocab_path}")
    if pretrain_path is not None and not pretrain_path.exists():
        raise SystemExit(f"Pretrained checkpoint not found: {pretrain_path}")

    rows, durations, text_vocab = prepare_vi_csv_wavs(input_csv, num_workers=args.workers)
    base_vocab_chars = read_vocab_chars(base_vocab_path)
    merged_vocab, missing_chars = merge_vocab(base_vocab_chars, text_vocab)
    save_prepared_dataset(output_dir, rows, durations, merged_vocab)

    prepared_pretrain_path = None
    checkpoint_info: dict | None = None
    if pretrain_path is not None:
        prepared_pretrain_path = output_dir / f"pretrained_{pretrain_path.name}"
        if missing_chars:
            checkpoint_info = prepare_adjusted_pretrain(
                pretrain_path=pretrain_path,
                output_path=prepared_pretrain_path,
                merged_vocab=merged_vocab,
                missing_chars=missing_chars,
            )
        else:
            shutil.copy2(pretrain_path, prepared_pretrain_path)
            checkpoint_info = {
                "checkpoint_format": pretrain_path.suffix.lower().lstrip("."),
                "checkpoint_vocab_size": len(base_vocab_chars),
                "changed_keys": [],
            }
    elif missing_chars and not args.allow_missing_vocab:
        raise SystemExit(
            "The Vietnamese dataset contains chars not present in the base vocab. "
            "Re-run with --pretrain-checkpoint so the checkpoint can be adjusted, "
            "or pass --allow-missing-vocab if you only want the prepared dataset."
        )

    summary = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "records": len(rows),
        "total_hours": round(sum(durations) / 3600, 6),
        "base_vocab_path": str(base_vocab_path),
        "base_vocab_size": len(base_vocab_chars),
        "prepared_vocab_size": len(merged_vocab),
        "missing_chars": missing_chars,
        "prepared_pretrain_checkpoint": str(prepared_pretrain_path) if prepared_pretrain_path else None,
        "checkpoint_info": checkpoint_info,
    }
    (output_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_name = output_dir.stem
    print(f"\nFor {dataset_name}, sample count: {len(rows)}")
    print(f"For {dataset_name}, vocab size is: {len(merged_vocab)}")
    print(f"For {dataset_name}, missing chars appended: {len(missing_chars)}")
    print(f"For {dataset_name}, total {sum(durations) / 3600:.2f} hours")
    if prepared_pretrain_path is not None:
        print(f"Prepared pretrain checkpoint: {prepared_pretrain_path}")
    if missing_chars:
        print(f"Missing chars appended to vocab: {missing_chars}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
