from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert one audio file or a directory tree to 24kHz mono PCM16 WAV with ffmpeg."
    )
    parser.add_argument("input", type=Path, help="Input audio file or directory.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output wav path for a single input file. Defaults to <stem>_24k.wav next to the input.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for directory mode. Relative layout is preserved.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories when the input is a directory.",
    )
    parser.add_argument(
        "--copy-sidecars",
        action="store_true",
        help="Copy matching .txt/.srt sidecar files alongside converted wav files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable to use. Default: ffmpeg",
    )
    parser.add_argument(
        "--resampler",
        default="soxr",
        help="ffmpeg resampler name. Default: soxr",
    )
    return parser


def iter_audio_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    pattern = "**/*" if recursive else "*"
    files = [path for path in input_path.glob(pattern) if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES]
    return sorted(files)


def resolve_output_path(input_file: Path, input_root: Path, output: Path | None, output_dir: Path | None) -> Path:
    if input_root.is_file():
        if output is not None:
            return output
        return input_file.with_name(f"{input_file.stem}_24k.wav")

    if output_dir is None:
        output_dir = input_root / "resampled_24k"
    relative_parent = input_file.parent.relative_to(input_root)
    return output_dir / relative_parent / f"{input_file.stem}_24k.wav"


def copy_sidecars(input_file: Path, output_file: Path, overwrite: bool) -> None:
    for suffix in (".txt", ".srt"):
        source = input_file.with_suffix(suffix)
        if not source.exists():
            continue
        target = output_file.with_suffix(suffix)
        if target.exists() and not overwrite:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def run_ffmpeg(ffmpeg_bin: str, input_file: Path, output_file: Path, overwrite: bool, resampler: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y" if overwrite else "-n",
        "-i",
        str(input_file),
        "-af",
        f"aresample=resampler={resampler}",
        "-ar",
        "24000",
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(output_file),
    ]
    subprocess.run(command, check=True)


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    files = iter_audio_files(input_path, recursive=args.recursive)
    if not files:
        raise SystemExit(f"No supported audio files found under: {input_path}")

    converted = 0
    for input_file in files:
        output_file = resolve_output_path(
            input_file=input_file,
            input_root=input_path,
            output=args.output,
            output_dir=args.output_dir,
        )
        run_ffmpeg(
            ffmpeg_bin=args.ffmpeg_bin,
            input_file=input_file,
            output_file=output_file,
            overwrite=args.overwrite,
            resampler=args.resampler,
        )
        if args.copy_sidecars:
            copy_sidecars(input_file, output_file, overwrite=args.overwrite)
        converted += 1
        print(f"[ok] {input_file} -> {output_file}")

    print(f"Converted {converted} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
