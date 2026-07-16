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
- `tone_aware_lora`
- `clean_only_lora`
- `noisy_only_lora`
- `clean_noisy_lora`
- `tone_aware_lora_clean_noisy`

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
python scripts/normalize_predictions.py --input outputs/lora/pred_tone_lora_lambda_0.1.csv --model phowhisper --model_size base --train_type tone_aware_lora --lambda 0.1 --seed 42 --output_dir outputs/predictions/tone_aware_lora
```

Script sẽ:

1. Chỉ chấp nhận schema chuẩn hoặc hai legacy schema được mô tả ở trên.
2. Kiểm tra cột, metadata, duplicate `utt_id`, lambda/seed và các trường bắt buộc.
3. Ghi file cùng basename vào `output_dir`; mặc định từ chối ghi đè.
4. Đọc lại output và xác nhận nội dung sau khi ghi.
5. Tạo `normalization_report.csv` ghi source/output SHA-256, schema nguồn, số dòng, metadata origin, benchmark hash và số hypothesis rỗng.

Raw prediction là dữ liệu đầu vào bất biến. Chỉ dùng `--overwrite` để tạo lại file đã chuẩn hóa trong output directory; không đặt `output_dir` trùng thư mục raw input.

Hai lệnh chuẩn hóa ở trên phục vụ chuyển đổi artifact legacy. Các prediction từ
benchmark 1.500 dòng cũ không được dùng để chọn lambda trong paper-v2.

## Hợp đồng metric `aligned_v1`

`aligned_v1` normalize NFC, lowercase, bỏ punctuation theo
`src/vitonesr/text_norm.py`, sau đó align từng utterance riêng. Không
được align xuyên ranh giới utterance.

- WER là word edit distance chia cho số reference word.
- CER là character edit distance chia cho số reference character **sau
  normalize, có tính khoảng trắng đơn giữa các từ**. Vì vậy không so
  trực tiếp với CER của tool loại bỏ whitespace.
- TER chỉ xét deletion hoặc cặp align có cùng lexical base sau khi bỏ
  thanh. Lexical substitution không nằm trong denominator TER.
- DER chỉ xét cặp align có cùng plain base sau khi bỏ thanh, dấu chữ
  cái và gộp `đ/d`. Deletion và lexical substitution không nằm trong
  denominator DER.
- FCER xét coda `ch/ng/nh/n/t/c/m/p` tại reference position đã align khi
  reference hoặc hypothesis có coda; hypothesis insertion không tham gia.
- SWDR là deletion rate trên đúng danh sách từ cố định `đã`, `có`,
  `là`, `một`, `và`, không phải metric cho mọi từ ngắn.

WER/CER/SWDR có denominator chỉ phụ thuộc reference. Denominator của
TER/DER/FCER phụ thuộc cả hypothesis, nên ba metric này là
**conditional diagnostic**, không được tuyên bố cải thiện độc lập mà
thiếu numerator, denominator và coverage.
`CorpusMetricResult.to_dict(include_counts=True)` xuất `ter_coverage`,
`der_coverage`, `fcer_coverage` bằng
eligible denominator chia số reference word. Denominator bằng 0 là không đủ
bằng chứng cho claim, dù scalar API giữ giá trị `0` để tương thích
`aligned_v1`.

Lambda selection fail closed khi low-SNR denominator TER, DER hoặc FCER bằng
0. Candidate phải giữ ít nhất `0.98` denominator coverage so với ordinary
LoRA cho **cả ba** metric. Đây là guard chống giảm denominator, không biến
conditional diagnostic thành metric denominator cố định. Trong paper, dùng
WER/CER là endpoint chính; TER/DER/FCER và SWDR là phân tích hỗ trợ có
nêu rõ phạm vi.

Các bảng paper-v2 `results_by_snr.csv`, `results_by_noise_type.csv` và
`external_fleurs_results.csv` giữ nguyên các cột scalar hiện hữu, sau đó
nối evidence fields ổn định:

```text
wer_numerator,wer_denominator,cer_numerator,cer_denominator,
ter_numerator,ter_denominator,der_numerator,der_denominator,
fcer_numerator,fcer_denominator,swdr_numerator,swdr_denominator,
ter_coverage,der_coverage,fcer_coverage
```

Rate trong cùng dòng phải bằng numerator chia denominator theo
ratio-of-totals của nhóm; khi denominator bằng 0, scalar `aligned_v1` là 0.
Ba coverage phải bằng eligible denominator chia `wer_denominator`. Không
tính trung bình các rate theo utterance.

## Provenance cho paper-v2

CSV prediction vẫn giữ đúng 11 cột chuẩn, không chèn metadata giao thức vào
header. Mỗi prediction do evaluator paper-v2 tạo phải có sidecar cùng tên:

```text
<prediction>.csv.provenance.json
```

Sidecar tối thiểu khóa các trường:

- `evaluation_split`: chỉ một trong `dev`, `test`, `external`;
- `manifest_sha256`: SHA-256 của manifest đánh giá;
- `prediction_sha256`: SHA-256 của CSV 11 cột;
- `num_rows`: số prediction trong CSV;
- `checkpoint` và `checkpoint_sha256`: checkpoint thực sự đã evaluate và
  fingerprint của adapter, processor, tone-head (nếu có), resolved config và
  trainer state dùng cho inference;
- `resolved_config_sha256`: hash config đã lưu trong checkpoint;
- `training_scope` và `training_contract_sha256`: phân biệt run `formal` với
  `smoke`, đồng thời khóa method/train type, lambda, seed, immutable model
  revision, hyperparameter, train/dev manifest và noise contract;
- `evaluation_contract` và `evaluation_contract_sha256`: khóa model revision,
  audio preprocessing, batch/precision và deterministic decoding contract;
- `runtime_environment`: ghi device/dtype, phiên bản PyTorch, Transformers và
  CUDA thực tế của inference;
- `evaluation_scope`, `filters`, `selected_rows_sha256`: chứng minh đây là
  full-manifest hay partial/smoke run;
- `metric_version`: bắt buộc là `aligned_v1` cho paper-v2;
- `split_lock_sha256` và, khi mở test, `decision_lock_sha256`.

Aggregate paper-v2 đọc lại manifest, tái tạo tập dòng từ filters, rồi đối chiếu
ID/reference/condition theo đúng thứ tự với CSV. Aggregate cũng hash lại
prediction/manifest và re-fingerprint checkpoint cùng `resolved_config.yaml`;
artifact đã bị thay sau inference sẽ bị từ chối.

Chọn lambda chỉ nhận run `training_scope=formal`, full
`evaluation_split=dev`, cùng `manifest_sha256`, `selected_rows_sha256`, locked
evaluation-contract hash, checkpoint/training-contract hash hợp lệ và metric
`aligned_v1`. Ngoài WER/CER guard, TER, DER và FCER của candidate phải có
denominator coverage tối thiểu `0.98` so với ordinary LoRA ở low-SNR. Input test,
mixed/partial split, smoke hoặc provenance legacy đều bị từ chối.

Khi mở final test, decision lock version 2 không dùng một danh sách checkpoint
vô danh. Mỗi cấu hình được đặt tên và khóa duy nhất theo `method_id`,
`train_type`, lambda, seed, backbone + immutable revision,
`checkpoint_sha256`, `resolved_config_sha256` và
`training_contract_sha256`; evaluator chỉ chấp nhận exact match. Việc chọn
checkpoint/method/lambda chỉ diễn ra trên dev. Final test chỉ được evaluate sau
khi decision-v2 đã khóa và không được dùng để chọn lại cấu hình.
