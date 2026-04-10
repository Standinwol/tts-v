# voice-cli

CLI-first workflow để import, validate, train và infer voice cloning tiếng Việt với F5-TTS.

Project này không tự re-implement training loop của F5-TTS. Nó làm 2 việc chính:

- chuẩn hóa dữ liệu đầu vào về một manifest thống nhất
- wrap các script train/infer của upstream F5-TTS theo một contract ổn định hơn

## Trạng thái hiện tại

Repository đang có 4 lệnh MVP:

- `voice-cli import-data`
- `voice-cli validate-data`
- `voice-cli train`
- `voice-cli infer`

## Điểm quan trọng cho tiếng Việt có dấu

Luồng hiện tại được thiết kế để train tiếng Việt có dấu chuẩn:

- transcript được giữ ở UTF-8
- mặc định được chuẩn hóa về Unicode `NFC`
- không đi qua pinyin khi train tiếng Việt với `tokenizer: char`
- vocab ký tự được build trực tiếp từ dữ liệu tiếng Việt của bạn
- checkpoint pretrain được rewrite lại phần text embedding để khớp vocab mới

Nếu bạn chỉ train tiếng Việt có dấu chuẩn, đây là luồng nên dùng.

## Cài đặt

Cài package local:

```bash
pip install -e .
```

Bản cài tối thiểu này đủ cho CLI wrapper và các luồng `dry-run`.

Để train thật, bạn vẫn cần một môi trường F5-TTS đầy đủ, thường bao gồm:

- `torch`
- `torchaudio`
- `datasets`
- `soundfile`
- `tqdm`
- và các dependency khác của upstream F5-TTS

## Quick Start

### 1. Import dataset

```bash
voice-cli import-data \
  --source ./exp/dataset_v3_f5_tts.zip \
  --format f5_zip
```

Các format đầu vào đang hỗ trợ:

- `f5_zip`
- `f5_csv`
- `f5_json`
- `gpt_sovits`
- `raw_pairs`

### 2. Validate dữ liệu

```bash
voice-cli validate-data \
  --manifest ./data/manifests/master.jsonl
```

Validate hiện kiểm tra các điểm chính:

- file audio có tồn tại hay không
- WAV có đọc được hay không
- sample rate
- số kênh
- duration min/max
- transcript rỗng
- path audio trùng nhau

### 3. Train với F5-TTS local

```bash
voice-cli train \
  --manifest ./data/manifests/validated.jsonl \
  --pretrain /path/to/model.safetensors \
  --run-name speaker01_ft \
  --f5-root /path/to/F5-TTS
```

### 4. Infer

```bash
voice-cli infer \
  --checkpoint /path/to/checkpoint.pt \
  --ref-audio ./ref.wav \
  --ref-text "Đây là câu tham chiếu." \
  --text "Đây là câu sinh thử nghiệm." \
  --out ./outputs/infer/test.wav \
  --f5-root /path/to/F5-TTS
```

## Luồng train tiếng Việt

Khi thỏa cả 2 điều kiện sau:

- `configs/project.yaml` có `text.language: vi`
- `configs/train.yaml` có `tokenizer: char`

thì `voice-cli train` sẽ tự chuyển sang luồng prepare tiếng Việt riêng, thay vì dùng `prepare_csv_wavs.py` upstream.

Luồng này:

- giữ nguyên tiếng Việt có dấu
- không convert transcript sang pinyin
- chuẩn hóa text về `NFC`
- build `vocab.txt` mới từ chính dataset
- ghi `raw.arrow`, `duration.json`, `vocab.txt`
- tạo checkpoint pretrain mới dạng `pretrained_<ten_file>`

Dataset prepared sẽ được ghi vào:

```text
<F5_ROOT>/data/<dataset_name>_char/
```

Checkpoint pretrain đã adapt sẽ nằm cạnh đó:

```text
<F5_ROOT>/data/<dataset_name>_char/pretrained_<checkpoint_goc>
```

Nếu bạn train thật trên RTX 3090 và muốn giữ checkpoint gọn hơn, repo có sẵn profile:

- `configs/train_3090_quality.yaml`

Profile này giữ mục tiêu fine-tune chất lượng, nhưng giảm tốc độ phình dung lượng ổ đĩa bằng cách:

- `save_per_updates: 5000`
- `keep_last_n_checkpoints: 1`
- `last_per_updates: 1000`

Ví dụ:

```bash
voice-cli train \
  --manifest ./data/manifests/validated.jsonl \
  --pretrain /path/to/model_1250000.safetensors \
  --run-name speaker01_ft \
  --train-config ./configs/train_3090_quality.yaml \
  --f5-root /path/to/F5-TTS
```

## Chạy helper prepare tiếng Việt trực tiếp

Nếu bạn muốn chạy prepare thủ công trong môi trường F5-TTS:

```bash
python scripts/prepare_vi_csv_wavs.py \
  /abs/path/to/metadata.csv \
  /path/to/F5-TTS/data/speaker01_char \
  --pretrain-checkpoint /path/to/model_1250000.safetensors
```

Yêu cầu:

- `metadata.csv` phải có header `audio_file|text`
- `audio_file` phải là absolute path
- output directory nên kết thúc bằng `_char`
- nếu transcript có chứa `|`, hãy dùng CSV writer đúng chuẩn

## Chuẩn hóa số tiếng Việt trước khi train

Nếu subtitle hoặc transcript còn các chữ số như `24`, `120.000đ`, `25%`, `0912345678`, nên đổi chúng về cách đọc ra thật trước khi train.

Repo có sẵn lệnh:

```bash
voice-cli normalize-vi-numbers \
  --source ./subs/sample.srt
```

Hoặc dùng script trực tiếp:

```bash
python scripts/normalize_vi_numbers.py ./subs/sample.srt
```

Lệnh này hiện hỗ trợ:

- `.txt`
- `.srt`
- `.csv` có cột `text` hoặc `transcript`
- `.jsonl` có field `text`

Ví dụ:

- `24` -> `hai mươi tư`
- `120.000đ` -> `một trăm hai mươi nghìn đồng`
- `25%` -> `hai mươi lăm phần trăm`
- `0912345678` -> `không chín một hai ba bốn năm sáu bảy tám`

Lưu ý:

- script cố ý không tự động đổi các mẫu như ngày tháng `12/03/2024` hoặc giờ `14:30`
- vẫn nên rà lại output trước khi train, nhất là với mã sản phẩm, version, biển số, hoặc câu có nhiều ký hiệu đặc biệt

## Chính sách infer

Mặc định:

- `ref_audio` là bắt buộc
- `ref_text` là bắt buộc
- chỉ được bỏ `ref_text` khi bật `--auto-transcribe-ref`
- và `configs/infer.yaml` cho phép `allow_auto_transcribe_ref: true`

Wrapper hiện chỉ truyền `--ref_text ""` xuống upstream khi bạn thực sự bật `--auto-transcribe-ref`.

## Config

Các file config mặc định:

- `configs/project.yaml`
- `configs/train.yaml`
- `configs/train_3090_quality.yaml`
- `configs/infer.yaml`

Lưu ý:

- path tương đối trong YAML được resolve theo vị trí của chính file config, không theo thư mục bạn đang đứng để chạy lệnh
- mặc định transcript sẽ được normalize Unicode với `text.normalize_unicode: true`
- nếu muốn giữ nguyên raw form của Unicode, có thể đặt `text.normalize_unicode: false`

## Output chính

Trong quá trình chạy, project sẽ sinh ra các file quan trọng như:

- `data/manifests/master.jsonl`
- `data/manifests/validated.jsonl`
- `data/manifests/invalid.jsonl`
- `data/manifests/train_f5.csv`
- `data/manifests/val_f5.csv`
- `reports/validate_summary.json`
- `reports/validate_summary.csv`
- `runs/<run_id>/run.json`
- `runs/<run_id>/*.log`
- `outputs/infer/*.wav`
- `outputs/infer/*.json`

## Test

Chạy unit test:

```bash
pip install -e .
python -m unittest discover -s tests -v
```

Repository cũng có GitHub Actions workflow để chạy test tự động:

```text
.github/workflows/tests.yml
```

## Ghi chú vận hành

- `voice-cli train` cần `--f5-root` hoặc `f5.root` trong `project.yaml`
- `voice-cli infer` cũng cần F5-TTS local nếu bạn muốn chạy inference thật
- lệnh `train` và `infer` đều hỗ trợ `--dry-run` để chỉ build command mà chưa thực thi

## Khi nào không nên dùng luồng này

Không nên dùng luồng train tiếng Việt hiện tại nếu:

- bạn muốn tokenizer khác `char`
- bạn muốn dùng pinyin tokenizer của upstream
- bạn đang chuẩn bị dữ liệu cho ngôn ngữ khác nhưng vẫn để `text.language: vi`

## Tóm tắt

Nếu mục tiêu của bạn là fine-tune F5-TTS cho tiếng Việt có dấu chuẩn, quy trình khuyến nghị là:

1. import dataset vào manifest canonical
2. validate dataset
3. train với `text.language: vi` và `tokenizer: char`
4. infer với `ref_text` rõ ràng

Luồng này hiện ưu tiên giữ nguyên tiếng Việt có dấu, tránh pinyin hóa và giảm mismatch giữa transcript, vocab và checkpoint.
