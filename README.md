# voice-cli

CLI-first workflow for importing, validating, training, and inferencing Vietnamese voice cloning datasets with F5-TTS.

## Status

This repository includes the MVP commands:

- `voice-cli import-data`
- `voice-cli validate-data`
- `voice-cli train`
- `voice-cli infer`

## Quick start

1. Install the package:

```bash
pip install -e .
```

2. Import a local dataset:

```bash
voice-cli import-data --source ./exp/dataset_v3_f5_tts.zip --format f5_zip
```

3. Validate it:

```bash
voice-cli validate-data --manifest ./data/manifests/master.jsonl
```

4. Train with a local F5-TTS checkout:

```bash
voice-cli train \
  --manifest ./data/manifests/validated.jsonl \
  --pretrain /path/to/model.safetensors \
  --run-name speaker01_ft \
  --f5-root /path/to/F5-TTS
```

For Vietnamese datasets, `voice-cli train` now switches to a `char` / no-pinyin prepare flow automatically when `configs/project.yaml` has `text.language: vi` and `configs/train.yaml` uses `tokenizer: char`. The prepared dataset is written to `data/<dataset_name>_char/`, and the copied or adjusted pretrain checkpoint is written next to it as `pretrained_<name>`.

5. Infer:

```bash
voice-cli infer \
  --checkpoint /path/to/checkpoint.pt \
  --ref-audio ./ref.wav \
  --ref-text "Reference text." \
  --text "Generated text." \
  --out ./outputs/infer/test.wav \
  --f5-root /path/to/F5-TTS
```

## Vietnamese Training

Do not use the upstream `prepare_csv_wavs.py` directly for Vietnamese if you want a stable `char` tokenizer workflow. The upstream prepare step converts text to pinyin, which corrupts Vietnamese transcripts.

Use the local prepare helper instead:

```bash
python scripts/prepare_vi_csv_wavs.py \
  /abs/path/to/metadata.csv \
  /path/to/F5-TTS/data/speaker01_char \
  --base-vocab /path/to/F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt \
  --pretrain-checkpoint /path/to/model_1250000.safetensors
```

This script:

- keeps Vietnamese text in UTF-8 / NFC form
- never converts text to pinyin
- writes `raw.arrow`, `duration.json`, and `vocab.txt`
- appends missing Vietnamese characters to the base vocab
- prepares a matching `pretrained_<checkpoint>` file in the output directory when needed

Then train with:

```bash
python src/f5_tts/train/finetune_cli.py \
  --exp_name F5TTS_v1_Base \
  --dataset_name speaker01 \
  --tokenizer char \
  --pretrain /path/to/F5-TTS/data/speaker01_char/pretrained_model_1250000.safetensors \
  --finetune
```

Important details:

- `metadata.csv` must use the header `audio_file|text`
- `audio_file` values must be absolute paths
- for `tokenizer=char`, F5-TTS resolves vocab from `data/<dataset_name>_char/vocab.txt`, so the output directory name must end with `_char`
