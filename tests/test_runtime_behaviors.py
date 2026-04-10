from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unicodedata
import unittest
from unittest import mock
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_cli.adapters.f5_dataset import import_from_f5_csv
from voice_cli.adapters.raw_pairs import import_from_raw_pairs
from voice_cli.config import InferConfig, load_project_config
from voice_cli.f5_wrapper import F5Runtime, build_infer_command
import voice_cli.prepare_vi_csv_wavs as vi_prepare
from voice_cli.train import _should_use_vi_prepare


def _write_silent_wav(path: Path, sample_rate: int = 24000, channels: int = 1, seconds: float = 1.0) -> None:
    frame_count = int(sample_rate * seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frame_count * channels)


def _load_split_by_srt_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "split_by_srt.py"
    spec = importlib.util.spec_from_file_location("split_by_srt_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(importlib.util.find_spec("yaml") is not None, "PyYAML not installed")
class ConfigPathResolutionTests(unittest.TestCase):
    def test_project_config_resolves_relative_paths_from_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "project.yaml"

            config_path.write_text(
                "\n".join(
                    [
                        "audio:",
                        "  sample_rate: 24000",
                        "  channels: 1",
                        "text:",
                        "  language: vi",
                        "manifest:",
                        "  master: ./data/master.jsonl",
                        "paths:",
                        "  reports_dir: ./reports",
                        "f5:",
                        "  root: ../F5-TTS",
                    ]
                ),
                encoding="utf-8",
            )

            project = load_project_config(config_path)

            self.assertEqual(project.manifest.master, (config_dir / "data" / "master.jsonl").resolve())
            self.assertEqual(project.paths.reports_dir, (config_dir / "reports").resolve())
            self.assertEqual(project.f5.root, (root / "F5-TTS").resolve())


class RuntimeBehaviorTests(unittest.TestCase):
    def test_should_use_vi_prepare_accepts_common_vietnamese_language_aliases(self) -> None:
        self.assertTrue(_should_use_vi_prepare("vi", "char"))
        self.assertTrue(_should_use_vi_prepare("vi-VN", "char"))
        self.assertTrue(_should_use_vi_prepare("vi_vn", "char"))
        self.assertTrue(_should_use_vi_prepare("vietnamese", "char"))
        self.assertFalse(_should_use_vi_prepare("vi", "pinyin"))

    def test_import_f5_csv_normalizes_but_keeps_vietnamese_diacritics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wav_path = root / "sample.wav"
            csv_path = root / "metadata.csv"
            _write_silent_wav(wav_path, seconds=1.0)
            decomposed_text = "To\u0302i ye\u0302u Tie\u0302\u0301ng Vie\u0323\u0302t va\u0300 chu\u031b\u0303 đa\u0300ng hoa\u0323t đo\u0323\u0302ng"

            csv_path.write_text(
                "audio_file|text\n" + f"{wav_path.name}|{decomposed_text}\n",
                encoding="utf-8",
            )

            records = import_from_f5_csv(csv_path, language="vi", normalize_unicode=True)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].text, unicodedata.normalize("NFC", decomposed_text))
            self.assertIn("đ", records[0].text)
            self.assertIn("ế", records[0].text)
            self.assertIn("ữ", records[0].text)

    def test_import_raw_pairs_preserves_text_when_normalization_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wav_path = root / "sample.wav"
            txt_path = root / "sample.txt"
            _write_silent_wav(wav_path, seconds=1.0)
            decomposed_text = "a\u0301"
            txt_path.write_text(decomposed_text, encoding="utf-8")

            records = import_from_raw_pairs(root, normalize_unicode=False)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].text, decomposed_text)

    def test_build_infer_command_omits_ref_text_without_auto_transcribe(self) -> None:
        command = build_infer_command(
            runtime=F5Runtime("python", Path("F5"), None, None, Path("infer_cli.py")),
            infer_config=InferConfig(require_ref_text=False, allow_auto_transcribe_ref=False),
            checkpoint=Path("checkpoint.pt"),
            ref_audio=Path("ref.wav"),
            ref_text=None,
            gen_text="hello",
            output_file=Path("outputs/out.wav"),
            seed=None,
            speed=None,
            nfe_step=None,
            remove_silence=False,
            auto_transcribe_ref=False,
            vocab_file=None,
            vocoder_name=None,
            model=None,
        )

        self.assertNotIn("--ref_text", command)

    def test_build_infer_command_uses_empty_ref_text_only_for_auto_transcribe(self) -> None:
        command = build_infer_command(
            runtime=F5Runtime("python", Path("F5"), None, None, Path("infer_cli.py")),
            infer_config=InferConfig(require_ref_text=False, allow_auto_transcribe_ref=True),
            checkpoint=Path("checkpoint.pt"),
            ref_audio=Path("ref.wav"),
            ref_text=None,
            gen_text="hello",
            output_file=Path("outputs/out.wav"),
            seed=None,
            speed=None,
            nfe_step=None,
            remove_silence=False,
            auto_transcribe_ref=True,
            vocab_file=None,
            vocoder_name=None,
            model=None,
        )

        ref_text_index = command.index("--ref_text")
        self.assertEqual(command[ref_text_index + 1], "")

    def test_get_audio_duration_uses_ffprobe_without_soundfile(self) -> None:
        completed_process = types.SimpleNamespace(stdout="1.25\n")
        with (
            mock.patch.object(vi_prepare, "sf", None),
            mock.patch.object(vi_prepare, "torchaudio", None),
            mock.patch.object(vi_prepare.subprocess, "run", return_value=completed_process),
        ):
            duration = vi_prepare.get_audio_duration("sample.wav")

        self.assertEqual(duration, 1.25)

    def test_prepare_vi_csv_wavs_keeps_vietnamese_diacritics_in_rows_and_vocab(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wav_path = (root / "sample.wav").resolve()
            csv_path = root / "metadata.csv"
            source_text = "To\u0302i ye\u0302u Tie\u0302\u0301ng Vie\u0323\u0302t"
            expected_text = unicodedata.normalize("NFC", source_text)
            csv_path.write_text(
                "audio_file|text\n" + f"{wav_path}|{source_text}\n",
                encoding="utf-8",
            )

            def fake_tqdm(iterable, **kwargs):
                return iterable

            def fake_process_audio_file(audio_path: str, text: str):
                return (audio_path, text, 1.0)

            with (
                mock.patch.object(vi_prepare, "tqdm", fake_tqdm),
                mock.patch.object(vi_prepare, "process_audio_file", side_effect=fake_process_audio_file),
            ):
                rows, durations, vocab_chars = vi_prepare.prepare_vi_csv_wavs(csv_path, num_workers=1)

            self.assertEqual(durations, [1.0])
            self.assertEqual(rows[0]["text"], expected_text)
            self.assertIn("ô", vocab_chars)
            self.assertIn("ê", vocab_chars)
            self.assertIn("ế", vocab_chars)
            self.assertIn("ệ", vocab_chars)

    def test_split_by_srt_recognizes_unicode_sentence_endings(self) -> None:
        split_by_srt = _load_split_by_srt_module()

        self.assertTrue(split_by_srt.ends_sentence("Xin chao\u2026"))
        self.assertTrue(split_by_srt.ends_sentence("Xin chao!\u201d"))
        self.assertFalse(split_by_srt.ends_sentence("Xin chao"))

    def test_split_by_srt_prefers_sentence_end_over_hard_duration_cap(self) -> None:
        split_by_srt = _load_split_by_srt_module()
        segments = [
            split_by_srt.SubtitleSegment(
                index=1,
                start_raw="00:00:00,000",
                end_raw="00:00:06,000",
                text="Xin chao",
            ),
            split_by_srt.SubtitleSegment(
                index=2,
                start_raw="00:00:06,100",
                end_raw="00:00:10,500",
                text="the gioi",
            ),
            split_by_srt.SubtitleSegment(
                index=3,
                start_raw="00:00:10,600",
                end_raw="00:00:12,000",
                text="roi day.",
            ),
        ]

        soft_merged = split_by_srt.merge_sentence_segments(
            segments,
            max_gap=0.8,
            max_merged_duration=10.0,
            prefer_sentence_end=True,
            min_sentence_duration=0.0,
        )
        hard_merged = split_by_srt.merge_sentence_segments(
            segments,
            max_gap=0.8,
            max_merged_duration=10.0,
            prefer_sentence_end=False,
            min_sentence_duration=0.0,
        )

        self.assertEqual(len(soft_merged), 1)
        self.assertEqual(soft_merged[0].text, "Xin chao the gioi roi day.")
        self.assertEqual(len(hard_merged), 2)

    def test_split_by_srt_merges_short_sentence_with_following_sentence(self) -> None:
        split_by_srt = _load_split_by_srt_module()
        segments = [
            split_by_srt.SubtitleSegment(
                index=1,
                start_raw="00:00:00,000",
                end_raw="00:00:02,000",
                text="Cau dau tien.",
            ),
            split_by_srt.SubtitleSegment(
                index=2,
                start_raw="00:00:02,100",
                end_raw="00:00:04,500",
                text="Cau thu hai.",
            ),
        ]

        merged = split_by_srt.merge_sentence_segments(
            segments,
            max_gap=0.8,
            max_merged_duration=10.0,
            prefer_sentence_end=True,
            min_sentence_duration=3.0,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "Cau dau tien. Cau thu hai.")


if __name__ == "__main__":
    unittest.main()
