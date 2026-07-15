# Prediction CSV schema

## Mục đích

Mọi prediction của zero-shot, LoRA, lambda ablation, multi-seed và external test phải dùng cùng một schema để các bước aggregate và error analysis không cần xử lý riêng theo từng người hoặc từng model.

Schema này chỉ chuẩn hóa cấu trúc và metadata. Việc chuyển đổi không được thay đổi nội dung nhận dạng `ref`/`hyp`, không chạy lại inference và không làm thay đổi metric.

## Schema chuẩn (version 1)

CSV phải dùng UTF-8, có header và đúng thứ tự 11 cột sau:

```text
utt_id,dataset,model,model_size,train_type,lambda,seed,snr,noise_type,ref,hyp
```

| Cột | Bắt buộc | Kiểu/quy ước | Ý nghĩa |
| --- | --- | --- | --- |
| `utt_id` | Có | Chuỗi không rỗng, duy nhất trong một file | ID của mẫu prediction. |
| `dataset` | Có | Chuỗi, ví dụ `vivos`, `fleurs` | Dataset evaluation. |
| `model` | Có | Chuỗi, ví dụ `whisper`, `phowhisper` | Họ model. |
| `model_size` | Có | Chuỗi, ví dụ `tiny`, `base`, `small` | Kích thước model. |
| `train_type` | Có | Chuỗi `snake_case` | Cách model được train/adapt. |
| `lambda` | Có cột | Số không âm hoặc để trống theo quy tắc bên dưới | Trọng số tone loss. |
| `seed` | Có | Số nguyên không âm | Run/training seed, cố định trong một prediction run. |
| `snr` | Có | `clean` hoặc số dB như `20`, `10`, `5`, `0` | Điều kiện SNR của mẫu. |
| `noise_type` | Có | Ví dụ `clean`, `music`, `noise`, `speech`, `babble` | Loại noise đúng theo benchmark metadata. |
| `ref` | Có | Chuỗi Unicode, không rỗng | Reference transcript. |
| `hyp` | Có cột | Chuỗi Unicode; được phép rỗng | Model hypothesis. Hypothesis rỗng là một kết quả ASR hợp lệ và sẽ được tính là deletion. |

### Quy ước `train_type` và `lambda`

Các giá trị `train_type` hiện dùng:

- `zero_shot`
- `ordinary_lora`
- `tone_lora`
- `clean_only_lora`
- `noisy_only_lora`
- `clean_noisy_lora`
- `tone_lora_clean_noisy`

Quy tắc `lambda`:

- `zero_shot`: để trống.
- `ordinary_lora`: dùng `0` vì loss chỉ có `L_ASR`.
- Train type có tone supervision: bắt buộc ghi lambda thực tế, ví dụ `0.05`, `0.1`, `0.3`, `0.5`.
- Không ghi các chuỗi như `none`, `null`, `N/A`.

### Quy ước `seed`

`seed` trong prediction là **run/training seed**. Một file prediction đại diện cho một run nên chỉ có một giá trị seed.

Không lấy `outputs/benchmark/benchmark_manifest.csv.seed` để điền cột này. Seed trong benchmark là seed trộn noise riêng cho từng utterance/SNR và thay đổi theo từng dòng. Dùng seed đó làm run seed sẽ khiến aggregate chia một run thành hàng nghìn nhóm sai.

Với sáu zero-shot prediction legacy hiện tại của Phúc, team đã duyệt backfill metadata sau qua CLI:

```text
train_type=zero_shot
lambda=<empty>
seed=42
```

Giá trị `42` ở đây là declared run/experiment seed theo quy ước của batch legacy. Zero-shot decode hiện dùng `do_sample=False` và code cũ không truyền seed vào inference, nên không diễn giải giá trị này là training seed hoặc bằng chứng về inference RNG. `normalization_report.csv` sẽ ghi nguồn metadata là `cli_backfill`.

## Khóa và tính nhất quán

- `utt_id` phải duy nhất trong từng file prediction.
- Khi gộp nhiều file, khóa run-level là `(dataset, utt_id, model, model_size, train_type, lambda, seed)`.
- Trong một file, `model`, `model_size`, `train_type`, `lambda` và `seed` phải nhất quán.
- `ref`, `snr` và `noise_type` của cùng `utt_id` phải khớp benchmark.
- Lưu `ref/hyp` nguyên văn. Text normalization chỉ được áp dụng trong lúc tính metric.

## Mapping format cũ

### Zero-shot 8 cột

Format cũ của Phúc:

```text
utt_id,dataset,model,model_size,snr,noise_type,ref,hyp
```

Giữ nguyên cả 8 trường và chèn thêm `train_type`, `lambda`, `seed` vào đúng vị trí schema chuẩn.

### Midterm/LoRA 7 cột

Format cũ từ `scripts/infer.py` hoặc `evaluate.py`:

```text
utt_id,audio,text,prediction,snr,noise_type,dataset
```

Mapping:

- `text` -> `ref`
- `prediction` -> `hyp`
- Bỏ `audio` vì không thuộc prediction schema.
- `model`, `model_size`, `train_type`, `lambda`, `seed` phải được truyền rõ khi chạy script; không tự suy đoán từ checkpoint hoặc tên file.

## Script chuẩn hóa

Script dùng chung: `scripts/normalize_predictions.py`.

Chuẩn hóa sáu file zero-shot của Phúc mà không ghi đè file gốc:

```powershell
python scripts/normalize_predictions.py --input_glob "outputs/zero_shot/pred_*.csv" --train_type zero_shot --seed 42 --benchmark_manifest outputs/benchmark/benchmark_manifest.csv --expected_rows 1500 --output_dir outputs/predictions/zero_shot
```

Ví dụ một file tone-aware LoRA:

```powershell
python scripts/normalize_predictions.py --input outputs/lora/pred_tone_lora_lambda_0.1.csv --model phowhisper --model_size base --train_type tone_lora --lambda 0.1 --seed 42 --output_dir outputs/predictions/tone_lora
```

Script sẽ:

1. Chỉ chấp nhận schema chuẩn hoặc hai legacy schema được mô tả ở trên.
2. Kiểm tra cột, metadata, duplicate `utt_id`, lambda/seed và các trường bắt buộc.
3. Ghi file cùng basename vào `output_dir`; mặc định từ chối ghi đè.
4. Đọc lại output và xác nhận nội dung sau khi ghi.
5. Tạo `normalization_report.csv` ghi source/output SHA-256, schema nguồn, số dòng, metadata origin, benchmark hash và số hypothesis rỗng.

Raw prediction là dữ liệu đầu vào bất biến. Chỉ dùng `--overwrite` để tạo lại file đã chuẩn hóa trong output directory; không đặt `output_dir` trùng thư mục raw input.
