# CLI Implementation Plan v2 cho Vietnamese TTS Voice Cloning

## Mục tiêu

Xây một project **CLI-first** để:

- nhập và chuẩn hóa dataset đang có;
- validate dữ liệu trước khi train;
- fine-tune **F5-TTS** cho tiếng Việt trên Vast.ai;
- infer ra 1 file WAV từ text với voice reference rõ ràng;
- giữ kiến trúc đủ sạch để sau này mở rộng sang `batch-infer`, `evaluate`, `export-best`, hoặc desktop app.

Mục tiêu của bản v2 này là chuyển từ một roadmap tốt sang một **implementation spec đủ chặt để bắt đầu code**.

## Phạm vi triển khai

### MVP cần làm trước

Chỉ làm 4 lệnh sau:

1. `import-data`
2. `validate-data`
3. `train`
4. `infer`

### Phase 2 làm sau khi MVP chạy ổn

- `prepare-data` từ raw long-form audio
- `batch-infer`
- `evaluate`
- `export-best`

Lý do: dữ liệu trong workspace hiện tại đã có sẵn ở nhiều format khác nhau (`dataset_v3_f5_tts.zip`, `gpt_sovits/esd.list`, `.wav + .srt/.txt`), nên bước đầu cần giải quyết **contract dữ liệu và train/infer ổn định** trước khi mở rộng.

## Các quyết định kỹ thuật đã chốt

### 1. Audio training chuẩn

- **sample rate:** `24000`
- **channels:** mono
- **format:** WAV PCM16

Plan cũ ghi `22050`, nhưng upstream `finetune_cli.py` của F5-TTS đang dùng `target_sample_rate = 24000`, nên toàn bộ pipeline nội bộ phải bám `24 kHz`.

### 2. Base model mặc định

- **default experiment:** `F5TTS_v1_Base`
- **fine-tune mode:** bật `--finetune`

`F5TTS_Base` vẫn là choice hợp lệ, nhưng mặc định của plan phải thống nhất với upstream để giảm mơ hồ.

### 3. Tokenizer cho tiếng Việt

- **default tokenizer:** `char`
- **không dùng mặc định:** `pinyin`
- **`custom` tokenizer:** để phase sau nếu thật sự cần

Lý do: CLI gốc mặc định `pinyin`, nhưng với tiếng Việt thì không phù hợp. `char` là điểm khởi đầu đơn giản, ít rủi ro hơn `custom`.

### 4. Policy cho reference text khi infer

- `ref_text` là **required** mặc định
- chỉ cho phép auto-ASR khi bật cờ rõ ràng: `--auto-transcribe-ref`

Mục tiêu là giữ hành vi infer **deterministic, dễ debug, ít tốn tài nguyên hơn**. Auto-ASR là opt-in, không phải default.

### 5. Duration policy

- **ngưỡng mặc định để train:** `3s` đến `20s`
- đây là **config mặc định**, không phải hard rule hardcode
- clip ngoài ngưỡng sẽ bị cảnh báo hoặc filter theo config

Plan cũ dùng `5-15s`, nhưng dataset hiện có trong repo đang có clip khoảng `3.35s` đến `18.97s`, nên ngưỡng đó quá cứng.

## Contract dữ liệu chuẩn nội bộ

Project phải chốt **1 manifest canonical** nội bộ, rồi viết adapter từ các format khác sang manifest này.

### Canonical manifest

File chuẩn nội bộ:

```text
data/manifests/master.jsonl
```

Mỗi dòng là một object JSON:

```json
{
  "audio_file": "E:/vibe/TTS-vast/data/processed/wavs/clip_0001.wav",
  "text": "Đã đến lúc đi Bắc Hải Sai rồi.",
  "speaker": "speaker01",
  "language": "vi",
  "duration_sec": 4.77,
  "sample_rate": 24000,
  "split": "train",
  "source_format": "f5_csv",
  "source_id": "clip_0001",
  "status": "ok"
}
```

### Vì sao không dùng thẳng `metadata.csv`

Vì workspace hiện có nhiều format đầu vào:

- F5-TTS style `metadata.csv`
- F5-TTS style `metadata.json`
- GPT-SoVITS `esd.list`
- raw `.wav + .srt/.txt`

Nếu không có 1 manifest canonical, các lệnh `validate`, `train`, `infer`, `evaluate` sẽ mỗi chỗ hiểu dữ liệu theo một kiểu khác nhau.

### Export format cho F5-TTS

Khi train, CLI sẽ export canonical manifest sang CSV đúng format F5-TTS yêu cầu:

```text
audio_file|text
E:/abs/path/clip_0001.wav|Đã đến lúc đi Bắc Hải Sai rồi.
```

Các file export nên nằm ở:

```text
data/manifests/train_f5.csv
data/manifests/val_f5.csv
```

## Nguồn dữ liệu đầu vào cần hỗ trợ ở MVP

### 1. F5 dataset zip

Ví dụ dữ liệu đang có:

```text
dataset_v3_f5_tts.zip
```

CLI phải đọc được:

- `metadata.csv`
- `metadata.json`
- thư mục `wavs/`

### 2. GPT-SoVITS dataset

Ví dụ dữ liệu đang có:

```text
gpt_sovits/esd.list
gpt_sovits/wavs/speaker01/*.wav
```

CLI phải parse được `esd.list` và map về canonical manifest.

### 3. Raw local files

Cho phép import các cặp:

- `.wav + .txt`
- `.wav + .srt`

nhưng phần split long-form audio và auto-transcription từ file lớn sẽ để **phase 2**.

## Cấu trúc thư mục đề xuất

```text
tts-viet-cli/
├── README.md
├── pyproject.toml
├── .env.example
├── configs/
│   ├── project.yaml
│   ├── train.yaml
│   └── infer.yaml
├── data/
│   ├── imports/
│   ├── processed/
│   │   └── wavs/
│   └── manifests/
│       ├── master.jsonl
│       ├── validated.jsonl
│       ├── invalid.jsonl
│       ├── train_f5.csv
│       └── val_f5.csv
├── models/
│   ├── base/
│   ├── checkpoints/
│   └── exported/
├── outputs/
│   └── infer/
├── logs/
├── reports/
├── runs/
├── src/
│   └── voice_cli/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── manifest.py
│       ├── io.py
│       ├── adapters/
│       │   ├── f5_dataset.py
│       │   ├── gpt_sovits.py
│       │   └── raw_pairs.py
│       ├── validate.py
│       ├── train.py
│       ├── infer.py
│       └── f5_wrapper.py
└── tests/
```

So với plan cũ, cấu trúc này bỏ bớt module chưa cần cho MVP và tập trung vào adapter + contract dữ liệu.

## Stack kỹ thuật

- Python 3.11
- Typer cho CLI
- PyYAML cho config
- Rich cho logging terminal
- FFmpeg để kiểm tra/chuẩn hóa audio
- upstream **SWivid/F5-TTS** để train và infer

### Ghi chú quan trọng về F5-TTS

- Nếu chỉ infer: `pip install f5-tts` là đủ
- Nếu train/fine-tune: phải dùng **local editable install** của repo F5-TTS

CLI nội bộ không nên tự re-implement train loop hay inference core; nó chỉ nên **wrap upstream F5-TTS**.

## Thiết kế lệnh CLI cho MVP

```bash
voice-cli import-data
voice-cli validate-data
voice-cli train
voice-cli infer
```

## 1. `import-data`

### Mục tiêu

Nhập dataset từ nhiều nguồn và chuyển về `master.jsonl`.

### Input format cần hỗ trợ

- `f5_csv`
- `f5_json`
- `gpt_sovits`
- `raw_pairs`
- `f5_zip`

### Chức năng

- đọc dataset nguồn;
- resolve path về absolute path;
- lấy hoặc tính `duration_sec` nếu thiếu;
- kiểm tra sample rate / channels;
- normalize metadata text về UTF-8;
- gán `split` mặc định;
- ghi ra `master.jsonl`.

### Ví dụ

```bash
voice-cli import-data \
  --source ./dataset_v3_f5_tts.zip \
  --format f5_zip \
  --speaker speaker01 \
  --language vi
```

```bash
voice-cli import-data \
  --source ./gpt_sovits/esd.list \
  --format gpt_sovits \
  --speaker speaker01 \
  --language vi
```

### Output

- `data/manifests/master.jsonl`
- log import

## 2. `validate-data`

### Mục tiêu

Xác thực canonical manifest trước khi train.

### Rule validate

- file audio tồn tại;
- đọc được bằng backend audio;
- sample rate đúng `24000`;
- mono;
- duration nằm trong ngưỡng config;
- transcript không rỗng;
- transcript không chỉ toàn ký tự lạ hoặc whitespace;
- path không trùng;
- text không trùng quá mức bất thường;
- tỉ lệ invalid được báo rõ.

### Policy

- record lỗi sẽ sang `invalid.jsonl`
- record hợp lệ được xuất ra `validated.jsonl`
- không xóa file nguồn

### Ví dụ

```bash
voice-cli validate-data \
  --manifest ./data/manifests/master.jsonl \
  --min-sec 3 \
  --max-sec 20
```

### Output

- `data/manifests/invalid.jsonl`
- `reports/validate_summary.json`
- `reports/validate_summary.csv`

## 3. `train`

### Mục tiêu

Wrap upstream `finetune_cli.py` của F5-TTS bằng một command ổn định và có khả năng resume.

### Nguyên tắc

- chỉ train trên dữ liệu đã validate;
- export manifest canonical sang `train_f5.csv` và `val_f5.csv`;
- map config YAML sang đúng CLI args của upstream;
- checkpoint và log phải nằm trong thư mục theo `run_id`;
- hỗ trợ resume.

### Các tham số quan trọng phải được chốt trong wrapper

- `exp_name: F5TTS_v1_Base`
- `tokenizer: char`
- `finetune: true`
- `sample_rate: 24000`
- `dataset_name`: theo `run_name` hoặc `speaker_id`
- `pretrain`: required

### Ví dụ

```bash
voice-cli train \
  --config ./configs/train.yaml \
  --manifest ./data/manifests/master.jsonl \
  --pretrain ./models/base/f5tts_v1_base/model.pt \
  --run-name speaker01_ft
```

### Wrapper cần làm gì

1. lọc record hợp lệ từ canonical manifest;
2. xuất `train_f5.csv` và `val_f5.csv`;
3. dựng thư mục `runs/<run_id>/`;
4. gọi upstream F5-TTS training command;
5. ghi lại full command, config snapshot, stdout, stderr;
6. hỗ trợ `--resume latest`.

### Vast.ai operational requirements

- mount volume bền vững cho `data/`, `models/`, `runs/`
- preprocess/import nên làm trước, không làm trên instance train nếu không cần
- mỗi lần save checkpoint phải ghi log rõ đường dẫn
- resume phải lấy từ checkpoint gần nhất trong `runs/<run_id>/`

## 4. `infer`

### Mục tiêu

Sinh 1 file WAV từ text với checkpoint fine-tuned.

### Chính sách bắt buộc

- `ref_audio`: required
- `ref_text`: required
- `text`: required
- chỉ khi bật `--auto-transcribe-ref` mới cho phép bỏ `ref_text`

### Chức năng

- gọi upstream inference của F5-TTS;
- lưu output WAV;
- lưu metadata JSON của lần infer;
- cho phép `seed`, `speed`, `nfe_step`, `remove_silence`.

### Ví dụ

```bash
voice-cli infer \
  --checkpoint ./models/checkpoints/speaker01_ft/model_last.pt \
  --ref-audio ./refs/ref_8s.wav \
  --ref-text "Xin chào, đây là giọng tham chiếu." \
  --text "Xin chào, đây là bản kiểm tra giọng nói." \
  --out ./outputs/infer/test_01.wav
```

### Output

- file WAV
- file metadata `.json` đi kèm:

```json
{
  "checkpoint": "./models/checkpoints/speaker01_ft/model_last.pt",
  "ref_audio": "./refs/ref_8s.wav",
  "ref_text": "Xin chào, đây là giọng tham chiếu.",
  "text": "Xin chào, đây là bản kiểm tra giọng nói.",
  "seed": 1234
}
```

## Config chuẩn

### `configs/project.yaml`

```yaml
audio:
  sample_rate: 24000
  channels: 1
  format: wav
  min_sec: 3
  max_sec: 20

text:
  language: vi
  normalize_unicode: true

manifest:
  master: ./data/manifests/master.jsonl
  validated: ./data/manifests/validated.jsonl
  invalid: ./data/manifests/invalid.jsonl
  train_f5: ./data/manifests/train_f5.csv
  val_f5: ./data/manifests/val_f5.csv
```

### `configs/train.yaml`

```yaml
exp_name: F5TTS_v1_Base
tokenizer: char
learning_rate: 1.0e-5
batch_size_per_gpu: 3200
batch_size_type: frame
max_samples: 64
grad_accumulation_steps: 1
epochs: 100
num_warmup_updates: 20000
save_per_updates: 2000
keep_last_n_checkpoints: 5
last_per_updates: 500
finetune: true
logger: tensorboard
```

Đây là config xuất phát điểm. Giá trị train thực tế vẫn phải tune theo VRAM và chất lượng dữ liệu.

### `configs/infer.yaml`

```yaml
require_ref_text: true
allow_auto_transcribe_ref: false
default_speed: 1.0
default_remove_silence: false
default_nfe_step: 32
```

## Những gì chưa làm trong MVP

Các phần sau **không nên code ngay ở vòng đầu**:

- split long-form audio bằng silence/VAD
- auto-transcription hàng loạt bằng Whisper
- batch inference
- speaker similarity scoring
- CER/WER evaluation
- auto-select checkpoint tốt nhất
- export-best

Đây là phase sau khi `import -> validate -> train -> infer` chạy ổn end-to-end.

## Milestone triển khai

### Milestone 1: Data contract ready

Điều kiện đạt:

- import được `dataset_v3_f5_tts.zip`
- import được `gpt_sovits/esd.list`
- sinh ra `master.jsonl` thống nhất

### Milestone 2: Validation ready

Điều kiện đạt:

- validate được sample rate, duration, transcript
- sinh `invalid.jsonl` và report
- export được `train_f5.csv`

### Milestone 3: Train ready

Điều kiện đạt:

- `voice-cli train` gọi được upstream F5-TTS
- checkpoint được lưu trong `runs/<run_id>/`
- resume thành công

### Milestone 4: Infer ready

Điều kiện đạt:

- `voice-cli infer` sinh được WAV
- `ref_text` được enforce đúng policy
- metadata infer được lưu lại đầy đủ

## Quy tắc chất lượng

- mọi command phải có `--help`
- không hardcode path
- mọi run có `run_id`
- mọi command ghi log ra `logs/` hoặc `runs/<run_id>/`
- mọi lỗi phải nói rõ thiếu gì: audio file, manifest, checkpoint, `ffmpeg`, upstream F5-TTS
- wrapper không được phụ thuộc vào format nội bộ mơ hồ

## Kết luận chốt

Hướng đúng cho vòng này là:

1. chốt **24k + `F5TTS_v1_Base` + tokenizer `char`**
2. chốt **canonical manifest**
3. chốt **`ref_text` required, auto-ASR là opt-in**
4. chỉ build **`import-data` / `validate-data` / `train` / `infer`** trước

Sau khi 4 bước này chạy ổn, mới mở rộng sang `prepare-data`, `batch-infer`, `evaluate`, `export-best`.

## Reference links

- F5-TTS repository: https://github.com/SWivid/F5-TTS
- F5-TTS training README: https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/train
- F5-TTS `finetune_cli.py`: https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/train/finetune_cli.py
- F5-TTS `infer_cli.py`: https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/infer_cli.py
