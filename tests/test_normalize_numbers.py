from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from voice_cli.normalize_numbers import integer_to_vietnamese, normalize_vi_numbers_in_text, run_normalize_numbers


class VietnameseNumberNormalizationTests(unittest.TestCase):
    def test_integer_to_vietnamese(self) -> None:
        self.assertEqual(integer_to_vietnamese(24), "hai mươi tư")
        self.assertEqual(integer_to_vietnamese(105), "một trăm lẻ năm")
        self.assertEqual(integer_to_vietnamese(2024), "hai nghìn không trăm hai mươi tư")

    def test_normalize_text_rewrites_common_number_forms(self) -> None:
        normalized, replacements = normalize_vi_numbers_in_text(
            "Giảm 25% còn 120.000đ, gọi 0912345678, hết 24 tiếng."
        )
        self.assertEqual(
            normalized,
            "Giảm hai mươi lăm phần trăm còn một trăm hai mươi nghìn đồng, gọi không chín một hai ba bốn năm sáu bảy tám, hết hai mươi tư tiếng.",
        )
        self.assertEqual(replacements, 4)

    def test_dates_and_times_are_left_untouched(self) -> None:
        normalized, replacements = normalize_vi_numbers_in_text("Hẹn ngày 12/03/2024 lúc 14:30.")
        self.assertEqual(normalized, "Hẹn ngày 12/03/2024 lúc 14:30.")
        self.assertEqual(replacements, 0)

    def test_srt_processing_skips_indices_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "sample.srt"
            source.write_text(
                "1\n00:00:00,000 --> 00:00:02,000\nTôi có 24 quả.\n\n2\n00:00:02,500 --> 00:00:04,000\nGọi 0912345678 nhé.\n",
                encoding="utf-8",
            )

            summary = run_normalize_numbers(source=source, output=None, overwrite=False, format_name="srt")

            output = Path(summary["output"])
            self.assertTrue(output.exists())
            self.assertEqual(summary["changed_records"], 2)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "1\n00:00:00,000 --> 00:00:02,000\nTôi có hai mươi tư quả.\n\n2\n00:00:02,500 --> 00:00:04,000\nGọi không chín một hai ba bốn năm sáu bảy tám nhé.\n",
            )

    def test_csv_processing_only_changes_text_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "metadata.csv"
            source.write_text(
                "audio_file|text\n/audio/24.wav|Tôi có 24 quả\n/audio/25.wav|Giảm 25%\n",
                encoding="utf-8",
            )

            summary = run_normalize_numbers(source=source, output=None, overwrite=False, format_name="csv")

            output = Path(summary["output"])
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "audio_file|text\n/audio/24.wav|Tôi có hai mươi tư quả\n/audio/25.wav|Giảm hai mươi lăm phần trăm\n",
            )
            self.assertEqual(summary["changed_records"], 2)


if __name__ == "__main__":
    unittest.main()
