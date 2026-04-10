from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


TIME_PATTERN = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})"
)
SENTENCE_END_PATTERN = re.compile(r"[.!?\u2026][\"'\u201d\u2019)\]]*\s*$")


@dataclass
class SubtitleSegment:
    index: int
    start_raw: str
    end_raw: str
    text: str

    @property
    def start_ffmpeg(self) -> str:
        return self.start_raw.replace(",", ".")

    @property
    def end_ffmpeg(self) -> str:
        return self.end_raw.replace(",", ".")

    @property
    def duration_seconds(self) -> float:
        return parse_timestamp_seconds(self.end_raw) - parse_timestamp_seconds(self.start_raw)

    @property
    def start_seconds(self) -> float:
        return parse_timestamp_seconds(self.start_raw)

    @property
    def end_seconds(self) -> float:
        return parse_timestamp_seconds(self.end_raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split an audio file into clips using subtitle timestamps from an SRT file."
    )
    parser.set_defaults(prefer_sentence_end=True)
    parser.add_argument("audio", type=Path, help="Input audio file.")
    parser.add_argument("srt", type=Path, help="Input SRT subtitle file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to <audio_stem>_srt_split next to the input audio.",
    )
    parser.add_argument(
        "--prefix",
        default="clip",
        help="Output filename prefix. Default: clip",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable to use. Default: ffmpeg",
    )
    parser.add_argument(
        "--encoding",
        help="Force subtitle file encoding. If omitted, several encodings are tried.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Skip subtitle segments shorter than this duration in seconds. Default: 0",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=0.0,
        help="Skip subtitle segments longer than this duration in seconds. Default: 0 (disabled)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Parse subtitles and print a summary without writing output files.",
    )
    parser.add_argument(
        "--merge-sentences",
        action="store_true",
        help="Merge consecutive subtitle cues until the combined text ends with sentence punctuation. If a finished sentence is shorter than --min-duration, keep merging with the next sentence when the timestamp gap allows it.",
    )
    parser.add_argument(
        "--max-merge-gap",
        type=float,
        default=1.5,
        help="When --merge-sentences is enabled, break the group if the gap between cues exceeds this many seconds. Default: 1.5",
    )
    parser.add_argument(
        "--max-merged-duration",
        type=float,
        default=15.0,
        help="When --merge-sentences is enabled, use this as the target merged duration in seconds. By default the script keeps merging until sentence-ending punctuation; use --force-mid-sentence-split to make this a hard cap. Default: 15",
    )
    parser.add_argument(
        "--prefer-sentence-end",
        dest="prefer_sentence_end",
        action="store_true",
        help="When --merge-sentences is enabled, treat --max-merged-duration as a soft target and keep merging until sentence-ending punctuation. Default: enabled.",
    )
    parser.add_argument(
        "--force-mid-sentence-split",
        dest="prefer_sentence_end",
        action="store_false",
        help="When --merge-sentences is enabled, treat --max-merged-duration as a hard cap and allow splitting before sentence-ending punctuation.",
    )
    return parser


def parse_timestamp_seconds(raw: str) -> float:
    hours, minutes, seconds = raw.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def load_text(path: Path, forced_encoding: str | None) -> str:
    encodings = [forced_encoding] if forced_encoding else ["utf-8-sig", "utf-8", "cp1258", "cp1252", "latin-1"]
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        if encoding is None:
            continue
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise SystemExit(f"Could not read subtitle file: {path}")


def parse_srt(content: str) -> list[SubtitleSegment]:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    segments: list[SubtitleSegment] = []
    blocks = re.split(r"\n\s*\n", normalized)
    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.split("\n") if line.strip()]
        if not lines:
            continue

        index = len(segments) + 1
        cursor = 0
        if lines[0].isdigit():
            index = int(lines[0])
            cursor = 1

        if cursor >= len(lines):
            continue
        match = TIME_PATTERN.match(lines[cursor])
        if not match:
            continue
        cursor += 1

        text = " ".join(line.strip() for line in lines[cursor:]).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue

        segments.append(
            SubtitleSegment(
                index=index,
                start_raw=match.group("start"),
                end_raw=match.group("end"),
                text=text,
            )
        )
    return segments


def sanitize_text_for_txt(text: str) -> str:
    return text.strip() + "\n"


def join_segment_text(segments: list[SubtitleSegment]) -> str:
    return re.sub(r"\s+", " ", " ".join(segment.text.strip() for segment in segments)).strip()


def ends_sentence(text: str) -> bool:
    return bool(SENTENCE_END_PATTERN.search(text))


def merge_segment_group(segments: list[SubtitleSegment]) -> SubtitleSegment:
    if not segments:
        raise ValueError("Cannot merge an empty subtitle group.")
    return SubtitleSegment(
        index=segments[0].index,
        start_raw=segments[0].start_raw,
        end_raw=segments[-1].end_raw,
        text=join_segment_text(segments),
    )


def merge_sentence_segments(
    segments: list[SubtitleSegment],
    max_gap: float,
    max_merged_duration: float,
    prefer_sentence_end: bool,
    min_sentence_duration: float,
) -> list[SubtitleSegment]:
    if not segments:
        return []

    merged: list[SubtitleSegment] = []
    current_group: list[SubtitleSegment] = []

    def flush_group() -> None:
        nonlocal current_group
        if current_group:
            merged.append(merge_segment_group(current_group))
            current_group = []

    for segment in segments:
        if not current_group:
            current_group = [segment]
        else:
            gap_seconds = segment.start_seconds - current_group[-1].end_seconds
            if gap_seconds > max_gap:
                flush_group()
                current_group = [segment]
            else:
                current_group.append(segment)

        combined = merge_segment_group(current_group)
        if ends_sentence(combined.text):
            if combined.duration_seconds >= min_sentence_duration:
                flush_group()
                continue

        if (
            max_merged_duration > 0
            and combined.duration_seconds >= max_merged_duration
            and not prefer_sentence_end
        ):
            flush_group()

    flush_group()
    return merged


def should_keep(segment: SubtitleSegment, min_duration: float, max_duration: float) -> bool:
    duration = segment.duration_seconds
    if duration < min_duration:
        return False
    if max_duration > 0 and duration > max_duration:
        return False
    return True


def run_ffmpeg(ffmpeg_bin: str, audio_path: Path, segment: SubtitleSegment, output_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y" if overwrite else "-n",
        "-i",
        str(audio_path),
        "-ss",
        segment.start_ffmpeg,
        "-to",
        segment.end_ffmpeg,
        "-ar",
        "24000",
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def write_metadata_csv(output_dir: Path, rows: list[tuple[str, str]]) -> None:
    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow(["audio_file", "text"])
        writer.writerows(rows)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args()
    audio_path = args.audio.resolve()
    srt_path = args.srt.resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")
    if not srt_path.exists():
        raise SystemExit(f"SRT file not found: {srt_path}")

    output_dir = args.output_dir.resolve() if args.output_dir else audio_path.with_name(f"{audio_path.stem}_srt_split")
    content = load_text(srt_path, args.encoding)
    segments = parse_srt(content)
    processed_segments = (
        merge_sentence_segments(
            segments,
            max_gap=args.max_merge_gap,
            max_merged_duration=args.max_merged_duration,
            prefer_sentence_end=args.prefer_sentence_end,
            min_sentence_duration=args.min_duration,
        )
        if args.merge_sentences
        else segments
    )
    kept = [segment for segment in processed_segments if should_keep(segment, args.min_duration, args.max_duration)]

    if not kept:
        raise SystemExit("No usable subtitle segments found.")

    print(
        f"Parsed {len(segments)} subtitle cue(s); "
        f"after processing: {len(processed_segments)} segment(s); "
        f"keeping {len(kept)} segment(s)."
    )
    if args.preview:
        for segment in kept[:10]:
            print(f"[{segment.index}] {segment.start_raw} -> {segment.end_raw} ({segment.duration_seconds:.2f}s) {segment.text}")
        if len(kept) > 10:
            print(f"... {len(kept) - 10} more segment(s)")
        return 0

    metadata_rows: list[tuple[str, str]] = []
    for output_index, segment in enumerate(kept, start=1):
        basename = f"{args.prefix}_{output_index:04d}"
        output_audio = output_dir / "wavs" / f"{basename}.wav"
        output_text = output_dir / "wavs" / f"{basename}.txt"
        run_ffmpeg(
            ffmpeg_bin=args.ffmpeg_bin,
            audio_path=audio_path,
            segment=segment,
            output_path=output_audio,
            overwrite=args.overwrite,
        )
        output_text.write_text(sanitize_text_for_txt(segment.text), encoding="utf-8")
        metadata_rows.append((output_audio.relative_to(output_dir).as_posix(), segment.text))
        print(f"[ok] {segment.index} -> {output_audio}")

    write_metadata_csv(output_dir, metadata_rows)
    print(f"Wrote {len(metadata_rows)} clip(s) and metadata to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
