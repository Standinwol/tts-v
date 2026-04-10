from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap_repo_src() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))


def main() -> int:
    _bootstrap_repo_src()
    from voice_cli.prepare_vi_csv_wavs import main as package_main

    return package_main()


if __name__ == "__main__":
    raise SystemExit(main())
