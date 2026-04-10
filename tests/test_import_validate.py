from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_cli.adapters.raw_pairs import import_from_raw_pairs
from voice_cli.manifest import ManifestRecord, write_f5_csv
from voice_cli.validate import _validate_record


def _write_silent_wav(path: Path, sample_rate: int = 24000, channels: int = 1, seconds: float = 1.0) -> None:
    frame_count = int(sample_rate * seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frame_count * channels)


class ImportValidateTests(unittest.TestCase):
    def test_import_raw_pairs_prefers_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wav_path = root / "sample.wav"
            txt_path = root / "sample.txt"
            _write_silent_wav(wav_path, seconds=1.5)
            txt_path.write_text("Hello   world\n", encoding="utf-8")

            records = import_from_raw_pairs(root, speaker="speakerA", language="vi")

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].speaker, "speakerA")
            self.assertEqual(records[0].text, "Hello world")
            self.assertAlmostEqual(records[0].duration_sec or 0.0, 1.5, places=2)

    def test_validate_record_flags_sample_rate_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wav_path = root / "bad.wav"
            _write_silent_wav(wav_path, sample_rate=22050, seconds=4.0)
            record = ManifestRecord(audio_file=str(wav_path), text="hello")

            updated, reasons = _validate_record(
                record=record,
                expected_sample_rate=24000,
                expected_channels=1,
                min_sec=3.0,
                max_sec=20.0,
                seen_paths=set(),
            )

            self.assertIn("sample_rate_mismatch", reasons)
            self.assertEqual(updated.sample_rate, 22050)

    def test_write_f5_csv_quotes_pipe_in_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wav_path = root / "sample.wav"
            csv_path = root / "train.csv"
            _write_silent_wav(wav_path, seconds=4.0)

            write_f5_csv(csv_path, [ManifestRecord(audio_file=str(wav_path), text="xin|chao")])

            contents = csv_path.read_text(encoding="utf-8")
            self.assertIn('"xin|chao"', contents)


if __name__ == "__main__":
    unittest.main()
