from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import re

from voice_cli.io import read_text_file

DIGIT_WORDS = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
}

GROUP_UNITS = [
    "",
    "nghìn",
    "triệu",
    "tỷ",
    "nghìn tỷ",
    "triệu tỷ",
    "tỷ tỷ",
]

NUMBER_TOKEN = r"\d(?:[\d.,]*\d)?"
CURRENCY_RE = re.compile(rf"(?<![\w/.:-])(?P<number>{NUMBER_TOKEN})\s*(?P<unit>đ|₫|vnd|vnđ)\b", re.IGNORECASE)
PERCENT_RE = re.compile(rf"(?<![\w/.:-])(?P<number>{NUMBER_TOKEN})\s*%")
PHONE_RE = re.compile(r"(?<![\w/.:-])(?P<number>0(?:[\s.]*\d){8,10})(?![\w/:-])")
GENERIC_NUMBER_RE = re.compile(rf"(?<![\w/.:-])(?P<number>{NUMBER_TOKEN})(?![\w/:-])")


def _read_three_digits(value: int, *, force_full: bool) -> str:
    hundreds = value // 100
    tens = (value % 100) // 10
    ones = value % 10
    parts: list[str] = []

    if hundreds:
        parts.extend([DIGIT_WORDS[str(hundreds)], "trăm"])
    elif force_full and value:
        parts.extend(["không", "trăm"])

    if tens == 0:
        if ones:
            if hundreds or force_full:
                parts.append("lẻ")
            parts.append(DIGIT_WORDS[str(ones)])
        return " ".join(parts)

    if tens == 1:
        parts.append("mười")
        if ones == 5:
            parts.append("lăm")
        elif ones:
            parts.append(DIGIT_WORDS[str(ones)])
        return " ".join(parts)

    parts.extend([DIGIT_WORDS[str(tens)], "mươi"])
    if ones == 1:
        parts.append("mốt")
    elif ones == 4:
        parts.append("tư")
    elif ones == 5:
        parts.append("lăm")
    elif ones:
        parts.append(DIGIT_WORDS[str(ones)])
    return " ".join(parts)


def integer_to_vietnamese(value: int) -> str:
    if value < 0:
        raise ValueError("Only non-negative integers are supported.")
    if value == 0:
        return DIGIT_WORDS["0"]

    groups: list[int] = []
    remaining = value
    while remaining:
        groups.append(remaining % 1000)
        remaining //= 1000

    if len(groups) > len(GROUP_UNITS):
        raise ValueError("Number is too large to convert safely.")

    parts: list[str] = []
    higher_seen = False
    for index in range(len(groups) - 1, -1, -1):
        group_value = groups[index]
        if group_value == 0:
            continue
        force_full = higher_seen and group_value < 100
        chunk = _read_three_digits(group_value, force_full=force_full)
        unit = GROUP_UNITS[index]
        parts.append(chunk if not unit else f"{chunk} {unit}")
        higher_seen = True
    return " ".join(parts)


def digit_sequence_to_vietnamese(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if not digits:
        raise ValueError("Digit sequence is empty.")
    return " ".join(DIGIT_WORDS[digit] for digit in digits)


def _parse_grouped_integer(token: str) -> str | None:
    compact = token.strip()
    if re.fullmatch(r"\d+", compact):
        return compact
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", compact):
        return compact.replace(".", "").replace(",", "")
    return None


def _parse_decimal(token: str) -> tuple[str, str] | None:
    compact = token.strip()
    if _parse_grouped_integer(compact) is not None:
        return None
    match = re.fullmatch(r"(\d+)[.,](\d+)", compact)
    if not match:
        return None
    return match.group(1), match.group(2)


def _number_token_to_vietnamese(token: str, *, digit_sequence_for_leading_zero: bool) -> str | None:
    integer_digits = _parse_grouped_integer(token)
    if integer_digits is not None:
        if digit_sequence_for_leading_zero and len(integer_digits) > 1 and integer_digits.startswith("0"):
            return digit_sequence_to_vietnamese(integer_digits)
        return integer_to_vietnamese(int(integer_digits))

    decimal_parts = _parse_decimal(token)
    if decimal_parts is None:
        return None

    integer_part, fraction_part = decimal_parts
    integer_words = integer_to_vietnamese(int(integer_part))
    fraction_words = digit_sequence_to_vietnamese(fraction_part)
    return f"{integer_words} phẩy {fraction_words}"


def normalize_vi_numbers_in_text(text: str) -> tuple[str, int]:
    replacements = 0

    def replace_currency(match: re.Match[str]) -> str:
        nonlocal replacements
        words = _number_token_to_vietnamese(match.group("number"), digit_sequence_for_leading_zero=False)
        if words is None:
            return match.group(0)
        replacements += 1
        return f"{words} đồng"

    def replace_percent(match: re.Match[str]) -> str:
        nonlocal replacements
        words = _number_token_to_vietnamese(match.group("number"), digit_sequence_for_leading_zero=False)
        if words is None:
            return match.group(0)
        replacements += 1
        return f"{words} phần trăm"

    def replace_phone(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 1
        return digit_sequence_to_vietnamese(match.group("number"))

    def replace_generic(match: re.Match[str]) -> str:
        nonlocal replacements
        words = _number_token_to_vietnamese(match.group("number"), digit_sequence_for_leading_zero=True)
        if words is None:
            return match.group(0)
        replacements += 1
        return words

    normalized = CURRENCY_RE.sub(replace_currency, text)
    normalized = PERCENT_RE.sub(replace_percent, normalized)
    normalized = PHONE_RE.sub(replace_phone, normalized)
    normalized = GENERIC_NUMBER_RE.sub(replace_generic, normalized)
    return normalized, replacements


def _normalize_srt_content(content: str) -> tuple[str, int, int]:
    replacements = 0
    changed_lines = 0
    output_lines: list[str] = []
    for line in content.splitlines(keepends=True):
        line_ending = ""
        raw_line = line
        if raw_line.endswith("\r\n"):
            line_ending = "\r\n"
            raw_line = raw_line[:-2]
        elif raw_line.endswith("\n"):
            line_ending = "\n"
            raw_line = raw_line[:-1]

        stripped = raw_line.strip()
        if not stripped or stripped.isdigit() or "-->" in stripped:
            output_lines.append(raw_line + line_ending)
            continue

        leading = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        trailing = raw_line[len(raw_line.rstrip()) :]
        normalized_line, line_replacements = normalize_vi_numbers_in_text(stripped)
        if normalized_line != stripped:
            changed_lines += 1
        replacements += line_replacements
        output_lines.append(f"{leading}{normalized_line}{trailing}{line_ending}")
    return "".join(output_lines), replacements, changed_lines


def _normalize_text_content(content: str) -> tuple[str, int, int]:
    replacements = 0
    changed_lines = 0
    output_lines: list[str] = []
    for line in content.splitlines(keepends=True):
        line_ending = ""
        raw_line = line
        if raw_line.endswith("\r\n"):
            line_ending = "\r\n"
            raw_line = raw_line[:-2]
        elif raw_line.endswith("\n"):
            line_ending = "\n"
            raw_line = raw_line[:-1]

        normalized_line, line_replacements = normalize_vi_numbers_in_text(raw_line)
        if normalized_line != raw_line:
            changed_lines += 1
        replacements += line_replacements
        output_lines.append(normalized_line + line_ending)
    return "".join(output_lines), replacements, changed_lines


def _normalize_csv_content(content: str) -> tuple[str, int, int]:
    header = content.splitlines()[0] if content.splitlines() else ""
    delimiter = "|" if "|" in header else "\t" if "\t" in header else ","
    reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
    if reader.fieldnames is None:
        raise ValueError("CSV is missing a header row.")
    text_field = "text" if "text" in reader.fieldnames else "transcript" if "transcript" in reader.fieldnames else None
    if text_field is None:
        raise ValueError("CSV must contain a 'text' or 'transcript' column.")

    replacements = 0
    changed_rows = 0
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=reader.fieldnames, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    for row in reader:
        raw_text = row.get(text_field) or ""
        normalized_text, row_replacements = normalize_vi_numbers_in_text(raw_text)
        if normalized_text != raw_text:
            changed_rows += 1
        replacements += row_replacements
        row[text_field] = normalized_text
        writer.writerow(row)
    return buffer.getvalue(), replacements, changed_rows


def _normalize_jsonl_content(content: str) -> tuple[str, int, int]:
    replacements = 0
    changed_rows = 0
    output_lines: list[str] = []
    for raw_line in content.splitlines():
        if not raw_line.strip():
            output_lines.append(raw_line)
            continue
        payload = json.loads(raw_line)
        text_value = payload.get("text")
        if isinstance(text_value, str):
            normalized_text, row_replacements = normalize_vi_numbers_in_text(text_value)
            if normalized_text != text_value:
                changed_rows += 1
            replacements += row_replacements
            payload["text"] = normalized_text
        output_lines.append(json.dumps(payload, ensure_ascii=False))
    trailing_newline = "\n" if content.endswith("\n") else ""
    return "\n".join(output_lines) + trailing_newline, replacements, changed_rows


def detect_text_format(path: Path, explicit_format: str | None = None) -> str:
    if explicit_format and explicit_format != "auto":
        return explicit_format
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return "srt"
    if suffix == ".csv":
        return "csv"
    if suffix == ".jsonl":
        return "jsonl"
    return "text"


def default_output_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}.normalized{source.suffix}")


def run_normalize_numbers(
    *,
    source: Path,
    output: Path | None,
    overwrite: bool,
    format_name: str = "auto",
) -> dict:
    source_path = source.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    effective_output = source_path if overwrite else (output.expanduser().resolve() if output else default_output_path(source_path))
    format_resolved = detect_text_format(source_path, explicit_format=format_name)
    content = read_text_file(source_path)

    if format_resolved == "srt":
        normalized, replacements, changed_records = _normalize_srt_content(content)
    elif format_resolved == "csv":
        normalized, replacements, changed_records = _normalize_csv_content(content)
    elif format_resolved == "jsonl":
        normalized, replacements, changed_records = _normalize_jsonl_content(content)
    elif format_resolved == "text":
        normalized, replacements, changed_records = _normalize_text_content(content)
    else:
        raise ValueError(f"Unsupported format: {format_resolved}")

    effective_output.parent.mkdir(parents=True, exist_ok=True)
    effective_output.write_text(normalized, encoding="utf-8")
    return {
        "kind": "normalize_vi_numbers",
        "source": str(source_path),
        "output": str(effective_output),
        "format": format_resolved,
        "replacements": replacements,
        "changed_records": changed_records,
        "overwrite": overwrite,
    }
