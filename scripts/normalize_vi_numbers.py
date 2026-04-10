from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice_cli.normalize_numbers import run_normalize_numbers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize Arabic numerals into spoken Vietnamese text for training transcripts.")
    parser.add_argument("source", type=Path, help="Source file. Supports .txt, .srt, .csv, .jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Optional output file path. Defaults to <source>.normalized.<ext>")
    parser.add_argument(
        "--format",
        choices=["auto", "text", "srt", "csv", "jsonl"],
        default="auto",
        help="Force the input format. Default: auto",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite the source file in place.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_normalize_numbers(
        source=args.source,
        output=args.output,
        overwrite=args.overwrite,
        format_name=args.format,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
