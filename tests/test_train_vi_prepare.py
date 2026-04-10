from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_cli.train import run_train


class TrainVietnamesePrepareTests(unittest.TestCase):
    def test_run_train_uses_vi_prepare_script_for_char_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_config = root / "project.yaml"
            train_config = root / "train.yaml"
            manifest_path = root / "validated.jsonl"
            pretrain_path = root / "model_1250000.safetensors"
            f5_root = root / "F5-TTS"

            (f5_root / "data").mkdir(parents=True)
            pretrain_path.write_bytes(b"fake")

            project_config.write_text(
                "\n".join(
                    [
                        "audio:",
                        "  sample_rate: 24000",
                        "  channels: 1",
                        "  min_sec: 3",
                        "  max_sec: 20",
                        "text:",
                        "  language: vi",
                        "manifest:",
                        f"  master: {root / 'master.jsonl'}",
                        f"  validated: {root / 'validated.jsonl'}",
                        f"  invalid: {root / 'invalid.jsonl'}",
                        f"  train_f5: {root / 'train_f5.csv'}",
                        f"  val_f5: {root / 'val_f5.csv'}",
                        "paths:",
                        f"  imports_root: {root / 'imports'}",
                        f"  reports_dir: {root / 'reports'}",
                        f"  runs_dir: {root / 'runs'}",
                        f"  logs_dir: {root / 'logs'}",
                        "f5:",
                        "  python_exe: python",
                    ]
                ),
                encoding="utf-8",
            )

            train_config.write_text(
                "\n".join(
                    [
                        "exp_name: F5TTS_v1_Base",
                        "tokenizer: char",
                        "learning_rate: 1.0e-5",
                        "batch_size_per_gpu: 400",
                        "batch_size_type: frame",
                        "max_samples: 8",
                        "grad_accumulation_steps: 1",
                        "epochs: 1",
                        "num_warmup_updates: 1",
                        "save_per_updates: 1",
                        "keep_last_n_checkpoints: 2",
                        "last_per_updates: 1",
                        "finetune: true",
                        "logger:",
                        "log_samples: false",
                        "bnb_optimizer: false",
                        "val_ratio: 0.1",
                    ]
                ),
                encoding="utf-8",
            )

            audio_path = root / "sample.wav"
            audio_path.write_bytes(b"fake")
            manifest_path.write_text(
                json.dumps(
                    {
                        "audio_file": str(audio_path),
                        "text": "Xin chao",
                        "speaker": "speaker01",
                        "language": "vi",
                        "status": "ok",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            metadata = run_train(
                project_config_path=project_config,
                train_config_path=train_config,
                manifest_path=manifest_path,
                pretrain=pretrain_path,
                run_name="speaker01_ft",
                f5_root=f5_root,
                python_exe="python",
                dry_run=True,
                resume=None,
            )

            prepare_command = metadata["commands"]["prepare"]
            train_command = metadata["commands"]["train"]
            self.assertIsNotNone(prepare_command)
            assert prepare_command is not None
            self.assertTrue(str(prepare_command[1]).endswith("scripts\\prepare_vi_csv_wavs.py") or str(prepare_command[1]).endswith("scripts/prepare_vi_csv_wavs.py"))
            self.assertTrue(str(prepare_command[2]).endswith("metadata.csv"))
            self.assertTrue(Path(metadata["prepare_output_dir"]).name.endswith("_char"))
            self.assertEqual(metadata["prepare_profile"], "vi_char_fresh_vocab")
            self.assertNotIn("--base-vocab", prepare_command)
            self.assertIn("--pretrain-checkpoint", prepare_command)
            effective_pretrain = Path(metadata["effective_pretrain"])
            self.assertEqual(effective_pretrain.name, f"pretrained_{pretrain_path.name}")
            self.assertEqual(train_command[train_command.index("--pretrain") + 1], str(effective_pretrain))


if __name__ == "__main__":
    unittest.main()
