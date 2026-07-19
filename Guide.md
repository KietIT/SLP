# Hướng dẫn chạy và bàn giao paper-v2

Tài liệu này là runbook chung cho Trung, Phúc và Phát sau khi source paper-v2 đã được đẩy lên Git. Mục tiêu là tạo lại toàn bộ bằng chứng từ checkpoint mới mà không dùng official test để train, chọn checkpoint hoặc chọn lambda.

Source, checkpoint, benchmark và decision lock hiện đã được bàn giao; các job inference cuối có thể chạy trên máy Trung theo protocol inference chéo máy bên dưới. Không sửa hoặc tái tạo `environment_lock.json`/`method_lock.json` của Phát: chúng tiếp tục là provenance bất biến của quá trình train.

## Cập nhật ưu tiên — formal inference chéo máy

Mục này **thay thế mọi câu cũ ở phần dưới yêu cầu final LoRA/FLEURS phải chạy trên đúng GPU hoặc exact package environment đã train của Phát**. Ranh giới mới tách hai provenance độc lập:

- `environment_lock.json` và `method_lock.json` chứng minh checkpoint được train bằng môi trường/source đã khóa của Phát; máy Trung không được giả vờ khớp môi trường train.
- `inference_runtime_lock.json` chứng minh final LoRA và FLEURS được infer trên môi trường hiện tại của Trung. Lock này hash đúng ba source bắt buộc: module runtime và hai wrapper final/FLEURS.
- Final LoRA phải ghi `inference_runtime_verified=true` và `training_runtime_verified_as_current=false` trong execution receipt. Giá trị `false` thứ hai là đúng vì Trung không chạy đúng runtime train của Phát.
- Provenance gốc của FLEURS tiếp tục ghi trung thực `method_runtime_verified=false`. Không sửa cờ này. Mỗi prediction/result có thêm sidecar `*.inference_runtime.json`, và `fleurs_execution_receipt.json` mới là bằng chứng runtime inference hiện tại.
- Phân tích FLEURS downstream chạy ở non-formal mode vì verifier formal cũ đòi cờ runtime train nói trên. Receipt bàn giao cuối xác minh cả hai execution receipt theo runtime hiện tại rồi bind SHA-256 của từng artifact downstream; vì vậy đây không phải là bỏ provenance.
- VietSpeech tạm bỏ theo scope hiện tại.

Sau khi source của ba file inference-runtime đã chốt, capture lock đúng **một lần** trên máy Trung rồi verify ngay. Nếu lock đã tồn tại thì chỉ chạy lệnh `--verify-existing`, không overwrite/recapture để hợp thức hóa source hoặc runtime đã đổi:

```powershell
$Python = (Get-Command python).Source
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
if (-not (Test-Path outputs/paper_v2/protocol/inference_runtime_lock.json)) {
  & $Python scripts/capture_inference_runtime.py `
    --training-environment outputs/paper_v2/protocol/environment_lock.json `
    --output outputs/paper_v2/protocol/inference_runtime_lock.json
}
& $Python scripts/capture_inference_runtime.py `
  --output outputs/paper_v2/protocol/inference_runtime_lock.json `
  --verify-existing
```

Nếu source wrapper, Python/package, CUDA hoặc GPU thay đổi sau capture thì verifier phải dừng; review thay đổi, hủy output inference chưa bàn giao và tạo một transaction lock mới có chủ đích. Không thay đổi training lock của Phát.

## 1. Ranh giới khoa học bắt buộc

- VIVOS `train` có 8.835 câu/36 speaker; `dev` có 2.825 câu/10 speaker. Hai tập tách speaker hoàn toàn.
- Trong 760 câu official test, 300 câu đã xuất hiện trong benchmark lịch sử nên chỉ là `test_legacy_exposed`; 460 câu còn lại là `test_locked`. Có thể build và khóa **benchmark data-only** trước quyết định lambda để chạy baseline song song, nhưng prediction/metric final tuyệt đối không được dùng để chọn lambda.
- Năm lambda `0, 0.05, 0.1, 0.3, 0.5` chỉ được so sánh trên noisy-dev 14.125 dòng. Không chạy cả năm lambda trên final test.
- Final test chỉ so sánh đúng ba vai trò đã ghi trong decision lock: `ordinary_baseline`, `selected_method`, `locked_control`.
- Final robustness benchmark có 460 nguồn × `(clean, 20, 10, 5, 0 dB)` = 2.300 dòng và chỉ dùng MUSAN `test`.
- MUSAN `train/dev/test` tách theo SHA-256 raw bytes. Train augmentation dùng MUSAN `train`; chọn lambda dùng MUSAN `dev`; final benchmark dùng MUSAN `test`.
- FLEURS Vietnamese gồm 857 câu clean. Vì tập này đã được xem ở vòng cũ, kết quả mới phải ghi là `legacy_exposed_external_replication`, không gọi là untouched external test.
- Checkpoint cũ, prediction cũ và benchmark 1.500 dòng chỉ dùng để so sánh lịch sử. Không đưa chúng vào selection hoặc paper-v2 aggregate.

Nếu bất kỳ lệnh nào báo hash, lock, schema, row count, checkpoint identity hoặc decision role không khớp thì dừng. Không sửa CSV/JSON thủ công để vượt qua lỗi.

## 2. Artifact đã khóa trên máy Trung

Đây là các artifact đã khóa ở local máy Trung. Các manifest/lock/audit nhỏ và tone-audit summary chỉ đi cùng source commit tương lai sau khi Trung review/stage; không mặc định rằng chúng đã có trên remote. Sau khi nhận bằng Git hoặc handoff bundle, đối chiếu lại SHA-256 trước khi chạy. Riêng row-level tone audit CSV 59,9 MB là artifact local có thể tái tạo, cố ý không commit; mỗi máy phải sinh lại file LF và ghi SHA-256 của lần audit đó vào handoff manifest, không dùng một hash row-level cố định giữa Windows/Linux.

| Artifact | Quy mô/trạng thái | SHA-256 |
| --- | ---: | --- |
| `data/manifests/paper_v2/vivos_train.jsonl` | 8.835 câu | `98dc7029d2e9794d68cc1905cf08970ca59b07d653f53493d527ff2dc7ccf7cb` |
| `data/manifests/paper_v2/vivos_dev.jsonl` | 2.825 câu | `3dc1afaaf4aedcaf5e5f472d93bb718df5776630e2e04db59e32f8fd8ef1af79` |
| `data/manifests/paper_v2/vivos_test_legacy_exposed.jsonl` | 300 câu, historical only | `99ffd07a85b19ecbd3e42915e6f3af735f82f8033e512862ac683e5becc98d82` |
| `data/manifests/paper_v2/vivos_test_locked.jsonl` | 460 câu, sealed | `e85bb3dc20d383da19804adeac5dfefd884acf9675fc547e912822880e2ab6ac` |
| `outputs/paper_v2/protocol/split_lock.json` | `SEALED`, audit 15/15 PASS | `a2b186e8f2b4d65bfa4ce5ac3511e23a9e000296e242dd12962c6fcfffc20de2` |
| `outputs/paper_v2/protocol/legacy_test_exposure.csv` | 300 exposed IDs | `800ef84711e1a8944e47a5c7d0e6e8d59520c7ec7ef7b9e9156e316c6860f09e` |
| `data/manifests/noise/paper_v2/musan_registry.jsonl` | 2.016 file | `c62de50e5c523c4c8c5b5e4d669793f4e369ec2a2a241e3cc9c2833b36976e28` |
| `data/manifests/noise/paper_v2/musan_train.jsonl` | 1.612 file | `9f0f840214c4960310c4f6ec20b2da2d5e0cf6cf3dcdf96e6d6d448351951848` |
| `data/manifests/noise/paper_v2/musan_dev.jsonl` | 202 file | `471c0f94a8f29eefa462bc92caee0270a911ab889dc26662a0a7fe6e74aeac00` |
| `data/manifests/noise/paper_v2/musan_test.jsonl` | 202 file | `6345d11536a99f0584bc96d58d10ad236e4b6995f692748e8eebe12489be9719` |
| `outputs/paper_v2/protocol/noise_split_lock.json` | `LOCKED`, audit 15/15 PASS | `5ebb46494c5613e586b9c9f93cffb8afbcbd4cdec21d198d02f198a1ee764cc9` |
| `data/manifests/paper_v2/vivos_dev_noisy.jsonl` | 14.125 dòng | `e9bb6cdb4934d90f10b0782b7960a609b33476ffbfcd946a08568a46de8c853b` |
| `outputs/paper_v2/protocol/noisy_dev_lock.json` | `LOCKED`, selection-only | `db263c9dc565a498261a1f912d7389ad4c947015d1b03f12befb89618417cda7` |
| `outputs/paper_v2/audits/tone_alignment_train_dev_audit.csv` | 11.660/11.660 exact, local/regenerable; sinh lại sau checkout để chuẩn hóa LF | xem hash trong lần audit mới |
| `outputs/paper_v2/audits/tone_alignment_train_dev_summary.md` | PASS | `00f0ce5330fe470fd86ff7fa2d09ec2814fb2789a353e769fc5a8fec1d7d1253` |

MUSAN archive chính thức đã dùng có 11.086.114.085 bytes và SHA-256 `86d1061c7e15b5c9e906777685c519701df51bfde3001e1070dcc9ffac955ee1`.

Kiểm tra nhanh các hash sau khi nhận artifact:

```powershell
$Expected = @{
  'data/manifests/paper_v2/vivos_train.jsonl' = '98dc7029d2e9794d68cc1905cf08970ca59b07d653f53493d527ff2dc7ccf7cb'
  'data/manifests/paper_v2/vivos_dev.jsonl' = '3dc1afaaf4aedcaf5e5f472d93bb718df5776630e2e04db59e32f8fd8ef1af79'
  'data/manifests/paper_v2/vivos_test_locked.jsonl' = 'e85bb3dc20d383da19804adeac5dfefd884acf9675fc547e912822880e2ab6ac'
  'outputs/paper_v2/protocol/split_lock.json' = 'a2b186e8f2b4d65bfa4ce5ac3511e23a9e000296e242dd12962c6fcfffc20de2'
  'outputs/paper_v2/protocol/noise_split_lock.json' = '5ebb46494c5613e586b9c9f93cffb8afbcbd4cdec21d198d02f198a1ee764cc9'
  'outputs/paper_v2/protocol/noisy_dev_lock.json' = 'db263c9dc565a498261a1f912d7389ad4c947015d1b03f12befb89618417cda7'
  'data/manifests/paper_v2/vivos_dev_noisy.jsonl' = 'e9bb6cdb4934d90f10b0782b7960a609b33476ffbfcd946a08568a46de8c853b'
}
foreach ($Item in $Expected.GetEnumerator()) {
  if (-not (Test-Path -LiteralPath $Item.Key)) { throw "Missing: $($Item.Key)" }
  $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Item.Key).Hash.ToLowerInvariant()
  if ($Actual -ne $Item.Value) { throw "SHA mismatch: $($Item.Key)" }
}
'PASS: received protocol hashes match'
```

## 3. Phân công khuyến nghị

| Giai đoạn | Owner chính | Output | Phụ thuộc |
| --- | --- | --- | --- |
| Preflight, formal environment và method lock | Phát trên máy train chuẩn | `environment_lock.json`, `method_lock.json` | Source commit + VIVOS + MUSAN + noisy-dev hoàn chỉnh |
| Train năm lambda | Phát, cùng một máy/GPU | 5 thư mục checkpoint + log | Method lock |
| Evaluate noisy-dev năm lambda | Phát, cùng một máy inference | 5 prediction + `lambda_ablation_results.csv` | 5 checkpoint |
| Build và publish final benchmark đúng một lần | Phúc | self-contained 2.300-WAV bundle + manifest/lock/audit/handoff SHA | VIVOS locked split + MUSAN test lock; **không phụ thuộc model/lambda** |
| Sáu zero-shot | Phúc, ngay sau khi benchmark được khóa | 6 prediction + provenance | Cùng final benchmark bundle v2 |
| Chọn method và khóa decision | Trung review/approve; Phát chạy lệnh trên reference machine | report + `best_lambda_decision.json` | Full five-lambda results + exact method environment + checkpoints; không dùng final prediction/metric |
| Ba LoRA role trên final | Trung trên inference runtime đã khóa riêng | 3 prediction + provenance/results + execution receipt | Decision + final benchmark + 3 checkpoint + inference runtime lock |
| FLEURS 857 | Trung trên cùng inference runtime đã khóa riêng | 3 prediction + result/provenance/runtime sidecar + execution receipt | Decision + method/training provenance + inference runtime lock + 3 checkpoint + portable FLEURS bundle |
| Aggregate, error analysis, confusion và bootstrap | Trung | CSV/PNG/report cuối | Tất cả prediction cần thiết |

Nguyên tắc tốt nhất là một máy chuẩn train cả năm lambda. Như vậy GPU, package, thứ tự dữ liệu, preprocessing và runtime giống nhau; lambda là biến khác biệt duy nhất. Phúc build/publish benchmark data-only trước và chạy sáu zero-shot; cùng lúc Phát train/evaluate năm lambda và chọn best chỉ trên noisy-dev. Sau decision lock, Phát pull đúng bundle byte-identical rồi chạy ba final LoRA role. Không rebuild benchmark trên máy Phát.

Máy Trung không cần giả lập exact GPU/package runtime đã train của Phát. Final LoRA/FLEURS được chấp nhận khi training provenance cũ vẫn byte-identical, checkpoint/decision/benchmark được verify transitively, inference runtime riêng của Trung khớp lock hiện tại và hai execution receipt PASS. Không trộn quy tắc này với fallback chia **training** giữa nhiều máy bên dưới.

### Fallback nếu buộc phải chia lambda cho nhiều máy

Chỉ dùng khi không đủ thời gian trên một máy:

- Mọi máy phải checkout cùng Git commit, nhận cùng `environment_lock.json`, `method_lock.json`, split/noise/noisy-dev locks và cùng raw audio bytes.
- Python/platform/GPU/package/CUDA runtime phải khớp exact environment lock; đây là điều kiện verifier, không chỉ là khuyến nghị. GPU/runtime khác thì worker không được tham gia formal run.
- Mỗi máy chỉ chạy lambda được giao bằng `--lambda-value`; không tự đổi batch size, accumulation, epoch, seed, LoRA rank hoặc decode.
- Coordinator phải evaluate cả năm checkpoint trên cùng một máy và cùng noisy-dev.
- Mỗi worker tạo file SHA handoff cho toàn bộ checkpoint/log trước khi chuyển.
- Ghi rõ đây là distributed-training fallback trong provenance. Trước khi nộp paper, nên retrain cả năm trên một reference machine; nếu không, phải báo hạn chế này và chạy ít nhất một lambda lặp trên hai máy để định lượng machine effect.

Không chạy một lambda trên hai máy rồi chọn checkpoint tốt hơn. Đó là thêm một vòng selection không đăng ký trước.

## 4. Chuẩn bị mỗi máy

Chạy từ root repository trong PowerShell. Dùng đúng một Conda environment cho toàn bộ một run.

```powershell
conda activate slp
$Python = (Get-Command python).Source
& $Python --version
& $Python -m pip install -r requirements.txt
& $Python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Không nâng package sau khi tạo `environment_lock.json`. Nếu thay package, source hoặc config thì method lock và mọi checkpoint sau nó phải làm lại.

Chạy test source trước khi dùng GPU:

```powershell
& $Python -m unittest discover -s tests -v
```

Tất cả test phải PASS. Test unit không thay thế smoke test với audio/model thật.

### Data layout bắt buộc

```text
data/raw/vivos/vivos/train/waves/<speaker>/*.wav
data/raw/vivos/vivos/train/prompts.txt
data/raw/vivos/vivos/test/waves/<speaker>/*.wav
data/raw/vivos/vivos/test/prompts.txt
data/raw/musan/musan/music/**/*.wav
data/raw/musan/musan/noise/**/*.wav
data/raw/musan/musan/speech/**/*.wav
data/derived/paper_v2/noisy_dev/snr_20/*.wav
data/derived/paper_v2/noisy_dev/snr_10/*.wav
data/derived/paper_v2/noisy_dev/snr_5/*.wav
data/derived/paper_v2/noisy_dev/snr_0/*.wav
```

Không đổi tên hoặc đặt dataset ở layout khác vì manifest dùng path tương đối này.

Nếu worker chưa có raw VIVOS, tải bằng script của repository rồi chỉ xác
minh transaction split đã publish (không `--overwrite`):

```powershell
bash scripts/download_vivos.sh
& $Python scripts/make_vivos_manifest.py `
  --vivos-root data/raw/vivos `
  --out-dir data/manifests/paper_v2 `
  --protocol-dir outputs/paper_v2/protocol `
  --legacy-benchmark-manifest outputs/benchmark/benchmark_manifest.csv `
  --expected-legacy-exposed 300 `
  --seed 42 `
  --dev-speaker-fraction 0.20
```

Lệnh phải báo `verified_existing` và giữ nguyên các hash ở mục 2.

Nếu cần tải lại MUSAN:

```powershell
New-Item -ItemType Directory -Force data/downloads | Out-Null
curl.exe -L --retry 5 --output data/downloads/musan.tar.gz https://www.openslr.org/resources/17/musan.tar.gz
$MusanSha = (Get-FileHash -Algorithm SHA256 data/downloads/musan.tar.gz).Hash.ToLowerInvariant()
if ($MusanSha -ne '86d1061c7e15b5c9e906777685c519701df51bfde3001e1070dcc9ffac955ee1') { throw 'MUSAN archive SHA mismatch' }
New-Item -ItemType Directory -Force data/raw/musan | Out-Null
tar.exe -xzf data/downloads/musan.tar.gz -C data/raw/musan
```

Sau khi raw MUSAN đã đúng layout, lệnh sau xác minh hoặc tái tạo đúng registry đã khóa:

```powershell
& $Python scripts/lock_musan_noise_protocol.py `
  --musan-root data/raw/musan/musan `
  --source-url https://www.openslr.org/resources/17/musan.tar.gz `
  --source-revision 86d1061c7e15b5c9e906777685c519701df51bfde3001e1070dcc9ffac955ee1 `
  --license-id 'MIXED-CREATIVE-COMMONS-AND-US-PUBLIC-DOMAIN; SEE-ARCHIVE-LICENSE-FILES' `
  --license-url https://www.openslr.org/resources/17/musan.tar.gz
```

Kết quả phải là `verified_existing` hoặc tạo lại đúng các hash ở mục 2. Không dùng `--overwrite` khi hash khác.

Khuyến nghị chuyển nguyên thư mục `data/derived/paper_v2/noisy_dev` từ máy Trung. Khi đã có đủ manifest, lock, audit và audio, lệnh dưới đây chỉ xác minh existing transaction:

```powershell
& $Python scripts/build_noisy_dev_benchmark.py `
  --source-dev-manifest data/manifests/paper_v2/vivos_dev.jsonl `
  --source-dev-sha256 3dc1afaaf4aedcaf5e5f472d93bb718df5776630e2e04db59e32f8fd8ef1af79
```

Nếu clone mới chỉ có manifest/lock được Git track nhưng chưa có ignored audio directory, không chạy đè ngay. Ưu tiên nhận audio bundle. Chỉ coordinator được tái tạo transaction bằng `--overwrite`, rồi bắt buộc kiểm tra manifest/lock/audit vẫn đúng hash ở mục 2; nếu khác thì dừng.

## 5. Preflight thật và khóa môi trường/method

Materialize đúng snapshot đã khóa, rồi ép Transformers/Hugging Face dùng cache offline trong toàn bộ training session:

```powershell
& $Python -c "from huggingface_hub import snapshot_download; print(snapshot_download('vinai/PhoWhisper-base', revision='7ebdb9e88f5cc5271fb88f4d642c82ff9388650e'))"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
```

Sinh lại full tone-alignment audit trên checkout hiện tại. Bước này cố ý thay
row-level CSV local cũ để mọi byte dùng LF trên cả Windows/Linux; ghi SHA-256
mới vào biên bản handoff, không dùng hash lịch sử ở một máy khác:

```powershell
& $Python scripts/audit_tone_alignment.py `
  --input data/manifests/paper_v2/vivos_train.jsonl `
  --input data/manifests/paper_v2/vivos_dev.jsonl `
  --text-column text `
  --id-column utt_id `
  --tokenizer-source vinai/PhoWhisper-base `
  --tokenizer-revision 7ebdb9e88f5cc5271fb88f4d642c82ff9388650e `
  --policy last_subtoken `
  --keep-duplicates `
  --output-csv outputs/paper_v2/audits/tone_alignment_train_dev_audit.csv `
  --output-summary outputs/paper_v2/audits/tone_alignment_train_dev_summary.md `
  --scope-note 'Full paper-v2 VIVOS train and speaker-disjoint dev manifests; every row audited without transcript deduplication.' `
  --overwrite
```

Lệnh phải vẫn báo 11.660/11.660 exact và summary `PASS`. Nếu summary thay đổi
ngoài metadata line-ending thì dừng review trước khi smoke/train.

Smoke train cả ordinary và tone-aware path, dùng output riêng và không ghi vào checkpoint formal. Mỗi smoke chỉ train một optimizer step nhưng trainer vẫn chạy full clean-dev evaluation để chọn `best`, nên có thể mất vài phút:

```powershell
& $Python scripts/train_phat_lora.py `
  --config configs/phat/lambda_0.yaml `
  --output-dir outputs/paper_v2/smoke/train_lambda0 `
  --max-train-samples 16 `
  --max-train-steps 1 `
  --device cuda
& $Python scripts/train_phat_lora.py `
  --config configs/phat/lambda_005.yaml `
  --output-dir outputs/paper_v2/smoke/train_lambda005 `
  --max-train-samples 16 `
  --max-train-steps 1 `
  --device cuda
```

Hai smoke trên đã đi qua đường đọc audio, noise augmentation, tokenizer, ordinary loss, tone alignment/loss, forward/backward, clean-dev evaluation và checkpoint thật. Không dùng checkpoint smoke để tạo prediction formal; evaluation giới hạn chỉ chạy sau khi đã có checkpoint formal và phải ghi vào output smoke riêng.

Trước formal capture, source tree phải sạch. Raw data và smoke output phải nằm trong các path đã ignore.

```powershell
$Dirty = git status --porcelain
if ($Dirty) { $Dirty; throw 'Repository must be clean before formal environment capture' }
& $Python scripts/capture_paper_v2_environment.py `
  --formal `
  --revision base_model=7ebdb9e88f5cc5271fb88f4d642c82ff9388650e `
  --output outputs/paper_v2/protocol/environment_lock.json
```

Không dùng `--allow-dirty-repository` cho run chính. Sau đó khóa method; lệnh formal sẽ xác minh split, toàn bộ MUSAN/VIVOS/noisy-dev audio, config, decode, metrics và source components:

```powershell
& $Python scripts/lock_paper_v2_method.py `
  --config configs/phat/lambda_0.yaml `
  --environment outputs/paper_v2/protocol/environment_lock.json `
  --output outputs/paper_v2/protocol/method_lock.json
```

Hai lệnh phải báo `mode=formal`. Không chỉnh file nào nằm trong source-component inventory sau thời điểm này. Nếu cần sửa source/config, xóa bỏ run chưa công bố theo quy trình review, tạo lại environment/method lock rồi train lại từ đầu.

Inventory này không chỉ khóa trainer/tone loss: nó còn khóa builder
benchmark, zero-shot/final-LoRA/FLEURS runners, aggregate, error analysis,
bootstrap và generator old-vs-new. Vì vậy phải hoàn tất review source trước
khi tạo method lock; không vá script phân tích sau khi đã xem kết quả.

Formal environment ghi Git commit sạch của source/method trước experiment. Sau này Trung có thể tạo một commit artifact/result riêng; commit đó không phải captured method-source commit và không được dùng để âm thầm thay method lock. Khi bàn giao, luôn giữ cả source commit identity trong environment lock lẫn artifact commit identity.

## 6. Train năm lambda và resume

### Phương án chính: một máy train cả năm

```powershell
& $Python scripts/train_all_lambdas.py `
  --config configs/phat/phat_pipeline.yaml `
  --device cuda
```

Nếu bị ngắt sau khi đã có `checkpoint_step_*`, resume toàn pipeline:

```powershell
& $Python scripts/train_all_lambdas.py `
  --config configs/phat/phat_pipeline.yaml `
  --resume `
  --device cuda
```

Lệnh resume bỏ qua run đã có cả `best` và `final`, còn run dở sẽ lấy checkpoint step mới nhất. Không thêm `--overwrite` vào lệnh resume.

Output formal:

```text
outputs/paper_v2/checkpoints/ckpt_lora_ordinary_lambda0/{best,final,...}
outputs/paper_v2/checkpoints/ckpt_tone_lora_lambda_005/{best,final,...}
outputs/paper_v2/checkpoints/ckpt_tone_lora_lambda_01/{best,final,...}
outputs/paper_v2/checkpoints/ckpt_tone_lora_lambda_03/{best,final,...}
outputs/paper_v2/checkpoints/ckpt_tone_lora_lambda_05/{best,final,...}
outputs/paper_v2/logs/
```

### Fallback: giao đúng một lambda cho một worker

Ví dụ worker lambda 0.1:

```powershell
& $Python scripts/train_all_lambdas.py `
  --config configs/phat/phat_pipeline.yaml `
  --lambda-value 0.1 `
  --device cuda
```

Resume worker đó:

```powershell
& $Python scripts/train_all_lambdas.py `
  --config configs/phat/phat_pipeline.yaml `
  --lambda-value 0.1 `
  --resume `
  --device cuda
```

Lặp với đúng một trong `0`, `0.05`, `0.1`, `0.3`, `0.5`. Không dùng `--seed` vì paper-v2 single-seed hiện tại đã khóa seed 42; multi-seed là task riêng của Kiệt.

## 7. Evaluate noisy-dev và khóa quyết định

Evaluate đầy đủ cả năm checkpoint trên cùng máy:

```powershell
& $Python scripts/evaluate_all_lambdas.py `
  --config configs/phat/phat_pipeline.yaml `
  --device cuda
```

Job noisy-dev có 14.125 dòng cho mỗi lambda và ghi checkpoint prediction sau
mỗi batch. Nếu máy bị ngắt, chạy lại đúng command với `--resume`:

```powershell
& $Python scripts/evaluate_all_lambdas.py `
  --config configs/phat/phat_pipeline.yaml `
  --device cuda `
  --resume
```

`--resume` xác minh lại config, manifest/noisy-dev lock, checkpoint SHA,
thứ tự prefix và SHA-256 của CSV trước khi tiếp tục. Lambda đã hoàn tất đủ CSV
và provenance sẽ được reuse mà không load model/audio; lambda còn dở tiếp tục
từ batch kế tiếp. Crash đúng khoảng giữa lúc publish CSV và resume state chỉ
được phục hồi khi `.csv.recovery.json` khớp chính xác; orphan hoặc file bị sửa
sẽ fail closed. Không ghép `--resume` với `--overwrite`.

Chạy riêng một checkpoint cũng dùng cùng cơ chế:

```powershell
& $Python scripts/evaluate_phat_checkpoint.py `
  --config configs/phat/lambda_005.yaml `
  --checkpoint outputs/paper_v2/checkpoints/ckpt_tone_lora_lambda_005/best `
  --device cuda `
  --resume
```

Output:

```text
outputs/paper_v2/dev_screen/pred_lora_ordinary_lambda0.csv
outputs/paper_v2/dev_screen/pred_tone_lora_lambda_005.csv
outputs/paper_v2/dev_screen/pred_tone_lora_lambda_01.csv
outputs/paper_v2/dev_screen/pred_tone_lora_lambda_03.csv
outputs/paper_v2/dev_screen/pred_tone_lora_lambda_05.csv
outputs/paper_v2/dev_screen/lambda_ablation_results.csv
```

Mỗi prediction phải có sidecar `.csv.provenance.json`, 14.125 dòng và provenance `full_manifest`; kết quả phải chứa đủ năm lambda, cùng seed, manifest hash, evaluation contract, method lock và `metric_version=aligned_v1`.

`lambda_ablation_results.csv` phải giữ numerator, denominator và coverage
cho TER/DER/FCER. Ba denominator này phụ thuộc hypothesis; selection vì
thế fail closed khi denominator bằng 0 và yêu cầu mỗi candidate giữ ít nhất
`0.98` denominator so với ordinary LoRA cho cả TER, DER và FCER. Guard
này chống candidate làm metric đẹp giả bằng cách giảm eligibility;
WER/CER vẫn là endpoint chính.

Khi bàn giao một job chưa xong, phải copy cùng relative path cả CSV,
`.csv.resume.json` và `.csv.recovery.json` nếu file recovery đang tồn tại. Khi
job đã hoàn tất, chỉ bàn giao CSV + `.csv.provenance.json`; resume/recovery
sidecar phải không còn. `lambda_ablation_results.csv` chỉ được reuse khi nội
dung đúng bằng aggregate tái tính từ đủ năm prediction đã verify.

Sau khi Trung xem đủ năm lambda và approve bảng selection, Phát chạy lệnh khóa selected method và locked control trên chính reference machine. Riêng bước **selection** này xác minh exact formal environment, method/audio lock và checkpoint identities nên không chuyển sang máy Trung; sau khi decision đã `LOCKED`, final inference mới được chuyển sang protocol inference riêng ở đầu tài liệu:

```powershell
& $Python scripts/select_best_lambda.py `
  --config configs/phat/phat_pipeline.yaml
```

Output:

```text
outputs/paper_v2/dev_screen/best_lambda_report.md
outputs/paper_v2/protocol/best_lambda_decision.json
```

Decision phải `LOCKED`, có đúng ba role riêng biệt và selected lambda phải dương. Lambda được chọn có thể không còn là 0.05. Từ đây tuyệt đối không quay lại đổi selection rule sau khi xem final test.

Lấy role/config động, không hard-code 0.05 hay 0.1:

```powershell
$Decision = Get-Content -Raw outputs/paper_v2/protocol/best_lambda_decision.json | ConvertFrom-Json
$RunsByRole = @{}
foreach ($Run in $Decision.locked_configurations) { $RunsByRole[$Run.role] = $Run }
$ConfigByLambda = @{
  '0' = 'configs/phat/lambda_0.yaml'
  '0.05' = 'configs/phat/lambda_005.yaml'
  '0.1' = 'configs/phat/lambda_01.yaml'
  '0.3' = 'configs/phat/lambda_03.yaml'
  '0.5' = 'configs/phat/lambda_05.yaml'
}
function Get-LambdaKey([double] $Value) {
  $Value.ToString('0.##', [Globalization.CultureInfo]::InvariantCulture)
}
$SelectedLambda = Get-LambdaKey ([double] $Decision.selected_lambda)
$ControlLambda = Get-LambdaKey ([double] $Decision.locked_control_lambda)
$SelectedConfig = $ConfigByLambda[$SelectedLambda]
$ControlConfig = $ConfigByLambda[$ControlLambda]
```

## 8. Build final benchmark đúng một lần

Phúc chạy ngay khi đã có đúng VIVOS locked split và MUSAN test lock. Builder v2 là **data-only**: identity chỉ phụ thuộc split/noise locks, source audio, seed và mixing contract; không đọc decision, method lock hay checkpoint. Vì vậy job này chạy song song an toàn với việc Phát train/chọn lambda trên noisy-dev:

```powershell
& $Python scripts/build_final_benchmark.py
```

Output phải có đúng 2.300 dòng:

```text
outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl
data/derived/paper_v2/final_benchmark/
outputs/paper_v2/protocol/final_benchmark_lock.json
outputs/paper_v2/protocol/final_benchmark_audit.csv
```

`data/derived/paper_v2/final_benchmark/` phải self-contained đúng 2.300 WAV: `clean/` có 460 file và bốn thư mục SNR có 1.840 file. Không dùng `--overwrite`. Phúc tạo bundle một lần, sinh handoff SHA cho manifest + lock + audit + đủ 2.300 WAV, rồi Phát/Trung chỉ nhận bản byte-identical; không rebuild trên máy khác.

Trước khi public audio lên GitHub, Trung phải xác nhận quyền tái phân phối và attribution của VIVOS/MUSAN-derived audio. Nếu chưa có xác nhận, dùng private repository/Git LFS hoặc chuyển bundle ngoài Git; không public raw VIVOS/MUSAN. Phần Git LFS và verify sau `pull` nằm ở mục 15.

## 9. Sáu zero-shot trên final benchmark

`configs/paper_v2_zero_shot.yaml` là template fail-closed. Tạo một runtime copy dưới `outputs`, rồi điền các SHA động sau khi benchmark v2 được khóa; zero-shot không chờ hoặc đọc `best_lambda_decision.json`:

Trước khi validate, máy Phúc phải có đúng relative path của final-benchmark lock, manifest và 2.300 WAV. Các source/noise hashes đã được builder v2 khóa trong final lock; runner zero-shot xác minh bundle nhận được nhưng không yêu cầu decision, method lock, LoRA checkpoint, raw VIVOS hay raw MUSAN.

- SHA của `final_benchmark_lock.json` và final manifest.
- Immutable Hugging Face commit SHA cho năm model còn placeholder. PhoWhisper-base đã khóa revision `7ebdb9e88f5cc5271fb88f4d642c82ff9388650e`.
- Xác nhận protocol v2/data-only; missing/tampered lock, manifest hoặc WAV làm runner fail-closed.
- Sau khi đủ sáu exact snapshots trong cache, đặt `runtime.local_files_only: true`.

Lấy commit SHA hiện tại rồi khóa nó; không ghi `main`:

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
& $Python -c "from huggingface_hub import model_info; print(model_info('openai/whisper-tiny').sha)"
& $Python -c "from huggingface_hub import model_info; print(model_info('openai/whisper-base').sha)"
& $Python -c "from huggingface_hub import model_info; print(model_info('openai/whisper-small').sha)"
& $Python -c "from huggingface_hub import model_info; print(model_info('vinai/PhoWhisper-tiny').sha)"
& $Python -c "from huggingface_hub import model_info; print(model_info('vinai/PhoWhisper-small').sha)"
```

Ghi năm SHA vừa lấy cùng PhoWhisper-base SHA vào map, tải đúng từng snapshot, rồi mới chuyển inference sang offline:

```powershell
$ZeroShotRevisions = @{
  'openai/whisper-tiny' = '<IMMUTABLE_SHA>'
  'openai/whisper-base' = '<IMMUTABLE_SHA>'
  'openai/whisper-small' = '<IMMUTABLE_SHA>'
  'vinai/PhoWhisper-tiny' = '<IMMUTABLE_SHA>'
  'vinai/PhoWhisper-base' = '7ebdb9e88f5cc5271fb88f4d642c82ff9388650e'
  'vinai/PhoWhisper-small' = '<IMMUTABLE_SHA>'
}
foreach ($Item in $ZeroShotRevisions.GetEnumerator()) {
  if ($Item.Value -eq '<IMMUTABLE_SHA>') { throw "Unresolved model revision: $($Item.Key)" }
  & $Python -c "from huggingface_hub import snapshot_download; import sys; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" $Item.Key $Item.Value
}
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
```

```powershell
New-Item -ItemType Directory -Force outputs/paper_v2/protocol/runtime | Out-Null
$RuntimeZeroShot = 'outputs/paper_v2/protocol/runtime/paper_v2_zero_shot.locked.yaml'
Copy-Item configs/paper_v2_zero_shot.yaml $RuntimeZeroShot
$BenchmarkLockSha = (Get-FileHash -Algorithm SHA256 outputs/paper_v2/protocol/final_benchmark_lock.json).Hash.ToLowerInvariant()
$BenchmarkManifestSha = (Get-FileHash -Algorithm SHA256 outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl).Hash.ToLowerInvariant()
$Yaml = [IO.File]::ReadAllText($RuntimeZeroShot)
$Yaml = [regex]::Replace($Yaml, '(?m)^  expected_lock_sha256:.*$', "  expected_lock_sha256: $BenchmarkLockSha")
$Yaml = [regex]::Replace($Yaml, '(?m)^  expected_manifest_sha256:.*$', "  expected_manifest_sha256: $BenchmarkManifestSha")
[IO.File]::WriteAllText($RuntimeZeroShot, $Yaml, [Text.UTF8Encoding]::new($false))
```

Sau đó mở runtime YAML và điền năm immutable model SHA ở các placeholder còn lại. Không sửa template trong `configs/`.

Sau khi điền runtime YAML, validate mà chưa mở test/model:

```powershell
& $Python scripts/run_zero_shot_paper_v2.py `
  --config outputs/paper_v2/protocol/runtime/paper_v2_zero_shot.locked.yaml `
  --validate-config
```

Chạy đủ sáu model:

```powershell
& $Python scripts/run_zero_shot_paper_v2.py `
  --config outputs/paper_v2/protocol/runtime/paper_v2_zero_shot.locked.yaml
```

Nếu bị ngắt, chỉ resume partial đã hash-bound:

```powershell
& $Python scripts/run_zero_shot_paper_v2.py `
  --config outputs/paper_v2/protocol/runtime/paper_v2_zero_shot.locked.yaml `
  --resume
```

Nếu chuyển một model dở sang máy khác, copy cùng relative path cả
`pred_*.csv`, `.csv.resume.json` và `.csv.recovery.json` nếu receipt đang tồn
tại; máy nhận chỉ được chạy `--resume` với đúng runtime YAML
bytes. Khi model hoàn tất, resume/recovery sidecar sẽ bị xóa và bundle
bàn giao chỉ gồm prediction + provenance.

Phúc có thể chia model giữa nhiều GPU clone bằng `--models`, ví dụ:

```powershell
& $Python scripts/run_zero_shot_paper_v2.py `
  --config outputs/paper_v2/protocol/runtime/paper_v2_zero_shot.locked.yaml `
  --models whisper_tiny whisper_base whisper_small
```

Nhóm còn lại dùng `phowhisper_tiny phowhisper_base phowhisper_small`. Hai máy phải dùng cùng runtime config bytes và final benchmark bundle. Sau đó chuyển cả CSV lẫn provenance sidecar về coordinator.

Output đủ phải là sáu file `outputs/paper_v2/predictions/zero_shot/pred_*.csv` và sáu sidecar `pred_*.csv.provenance.json`; mỗi prediction có đúng 2.300 dòng. Không bàn giao file `.resume.json` như một kết quả hoàn tất.

## 10. Ba role LoRA trên final benchmark

Phần này dùng runner final-LoRA fail-closed. Runner không nhận lambda trên CLI; nó chỉ resolve đúng `ordinary_baseline`, `selected_method`, `locked_control` từ decision v3. Không tự đổi các config dev thành test và không hard-code lambda.

Chỉ bắt đầu bước này sau khi decision đã được khóa từ noisy-dev. Máy Trung dùng nguyên checkpoint của Phát và bundle benchmark do Phúc publish, verify handoff SHA/LFS rồi chạy bằng inference runtime đã khóa riêng; decision được verifier đối chiếu với source-test identity trong benchmark nhưng không làm thay đổi hoặc rebuild benchmark.

Sau decision v3 và final-benchmark lock, materialize runtime config tự động. Bước này pin SHA của split/noise/method/decision/final lock và final manifest, nhưng chưa mở final manifest/audio/model:

```powershell
& $Python scripts/prepare_final_lora_config.py `
  --template configs/paper_v2_final_lora.yaml `
  --output outputs/paper_v2/protocol/final_lora_runtime.yaml
```

Không sửa immutable template vì template cũng nằm trong method source lock. Không dùng `--overwrite` trừ khi một lock đã được thay đổi có chủ đích và toàn bộ downstream output cũ đã bị hủy theo review. Runtime mặc định dùng `device: auto`; trên máy chính phải kiểm tra nó resolve CUDA. Nếu muốn preflight mạnh nhất, đặt `verify_method_audio_sha256: true` trong runtime YAML vừa sinh trước lần chạy đầu và bàn giao chính runtime YAML theo SHA. Runner ghi exact runtime-YAML path/SHA-256 vào provenance của từng role và aggregate; method lock đồng thời bao phủ template, builder, evaluator và các runner/analysis source của paper-v2.

Trước khi chạy, lệnh verify runtime phải PASS trên đúng terminal hiện tại. Sau đó chạy ba role bằng wrapper chéo máy; wrapper vẫn verify transitive method/checkpoint/config/decision/benchmark của Phát nhưng không tuyên bố runtime train đang là runtime hiện tại:

```powershell
& $Python scripts/capture_inference_runtime.py `
  --output outputs/paper_v2/protocol/inference_runtime_lock.json `
  --verify-existing
& $Python scripts/run_final_lora_inference_runtime.py `
  --config outputs/paper_v2/protocol/final_lora_runtime.yaml `
  --inference-runtime-lock outputs/paper_v2/protocol/inference_runtime_lock.json
```

Nếu bị ngắt, chạy lại đúng runtime YAML với `--resume`. Runner reuse role đã
hoàn tất và tiếp tục role đang dở từ **batch kế tiếp**, không chạy lại 2.300
dòng từ đầu:

```powershell
& $Python scripts/run_final_lora_inference_runtime.py `
  --config outputs/paper_v2/protocol/final_lora_runtime.yaml `
  --inference-runtime-lock outputs/paper_v2/protocol/inference_runtime_lock.json `
  --resume
```

Trong lúc chạy, mỗi role dở dùng thư mục ẩn
`outputs/paper_v2/final_predictions/.<role>.partial/`. Sau mỗi inference batch,
runner ghi canonical prefix CSV cùng state hash-bound. Write-ahead recovery
receipt cho phép phục hồi đúng crash window CSV đã commit nhưng state chưa
commit. Prefix được khóa vào role/configuration, decision, final manifest,
checkpoint, resolved/training config, runtime config và prediction schema; nếu
hash, schema, thứ tự `utt_id/ref`, metadata hoặc runtime khác thì resume dừng
trước khi load model. Không sửa/xóa riêng lẻ `predictions.partial.csv`,
`resume.json` hay `recovery.json`, và không dùng `--resume` để overwrite output
đã hoàn tất.

Output:

```text
outputs/paper_v2/final_predictions/ordinary_baseline/predictions.csv
outputs/paper_v2/final_predictions/ordinary_baseline/provenance.json
outputs/paper_v2/final_predictions/selected_method/predictions.csv
outputs/paper_v2/final_predictions/selected_method/provenance.json
outputs/paper_v2/final_predictions/locked_control/predictions.csv
outputs/paper_v2/final_predictions/locked_control/provenance.json
outputs/paper_v2/final_predictions/aggregate/final_lora_results.csv
outputs/paper_v2/final_predictions/aggregate/provenance.json
outputs/paper_v2/protocol/inference_runtime_lock.json
outputs/paper_v2/protocol/final_lora_execution_receipt.json
```

Output bắt buộc là ba prediction canonical 2.300 dòng, ba provenance sidecar và aggregate 30 dòng cho đúng các role trong decision lock. Checkpoint/configuration ID phải khớp byte-for-byte với `locked_configurations`; không được thay bằng `final/` nếu decision khóa `best/`. Khi hoàn tất, runner tự xóa thư mục `.partial`; không bàn giao partial/state/recovery như kết quả cuối. Xác minh lại toàn bộ receipt và current runtime mà không infer:

```powershell
& $Python scripts/run_final_lora_inference_runtime.py `
  --config outputs/paper_v2/protocol/final_lora_runtime.yaml `
  --inference-runtime-lock outputs/paper_v2/protocol/inference_runtime_lock.json `
  --verify-only
```

## 11. FLEURS Vietnamese 857 câu

Formal preparation dùng exact cached Hugging Face dataset revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`; dataset card tại revision đó khai báo `CC-BY-4.0`. Manifest mới dùng repo-relative path, ghi SHA-256 từng WAV và có preparation lock/audit nên có thể bàn giao sang root khác nếu bundle byte-identical.

Ba role FLEURS chạy trên cùng inference runtime riêng đã khóa ở máy Trung. Wrapper mới xác minh transitive method artifact, source/config/checkpoint/decision của Phát mà không yêu cầu GPU/package hiện tại phải giả làm môi trường train; sau đó nó xác minh `inference_runtime_lock.json` trước khi mở model/audio và tạo execution receipt bất biến.

Authorization nền giữ `verify_current_method=False`, vì exact runtime train không phải runtime hiện tại. Wrapper bù lại bằng verifier inference-runtime độc lập. Do đó provenance gốc phải giữ `method_runtime_verified=false`; chỉ sidecar `*.inference_runtime.json` và receipt mới được ghi `inference_runtime_verified=true`. Hai cờ nói về hai câu hỏi khác nhau và không được sửa tay để giống nhau.

Không dùng `data/manifests/fleurs/test.jsonl`: đó là manifest lịch sử chứa absolute path và không có per-audio hash. Formal FLEURS chỉ nhận bundle `data/manifests/fleurs/paper_v2/` được khóa dưới đây.

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
$FleursRevision = '70bb2e84b976b7e960aa89f1c648e09c59f894dd'
& $Python scripts/download_fleurs.py `
  --revision $FleursRevision
$FleursManifest = 'data/manifests/fleurs/paper_v2/test.jsonl'
$FleursPreparationLock = 'outputs/paper_v2/protocol/fleurs_test_lock.json'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
```

Nếu download bị ngắt, chạy lại cùng exact revision với `--resume`; formal mode không cho `--overwrite`:

```powershell
& $Python scripts/download_fleurs.py `
  --revision $FleursRevision `
  --resume
```

Output preparation:

```text
data/manifests/fleurs/paper_v2/test.jsonl
data/manifests/fleurs/paper_v2/audio/test/*.wav
outputs/paper_v2/protocol/fleurs_test_lock.json
outputs/paper_v2/protocol/fleurs_test_audit.csv
```

Lock ghi preparation contract, license `CC-BY-4.0`, dataset revision, manifest/audit hash và canonical audio-inventory hash. Mỗi WAV được kiểm tra raw-byte SHA-256. Nhãn evidence vẫn là legacy-exposed replication vì vấn đề exposure, không phải vì thiếu content provenance.

Trên chính terminal/máy chạy FLEURS, rehydrate config từ decision; không dựa vào biến PowerShell của mục 7 trên máy khác:

```powershell
$Decision = Get-Content -Raw outputs/paper_v2/protocol/best_lambda_decision.json | ConvertFrom-Json
$ConfigByLambda = @{
  '0' = 'configs/phat/lambda_0.yaml'
  '0.05' = 'configs/phat/lambda_005.yaml'
  '0.1' = 'configs/phat/lambda_01.yaml'
  '0.3' = 'configs/phat/lambda_03.yaml'
  '0.5' = 'configs/phat/lambda_05.yaml'
}
function Get-LambdaKey([double] $Value) {
  $Value.ToString('0.##', [Globalization.CultureInfo]::InvariantCulture)
}
$SelectedConfig = $ConfigByLambda[(Get-LambdaKey ([double] $Decision.selected_lambda))]
$ControlConfig = $ConfigByLambda[(Get-LambdaKey ([double] $Decision.locked_control_lambda))]
if (-not $SelectedConfig -or -not $ControlConfig) { throw 'Decision lambda has no registered config' }
```

Registry là artifact bất biến. Nếu `outputs/paper_v2/protocol/fleurs_run_registry.json` đã có thì không tạo lại; wrapper sẽ verify nó khi chạy. Trên Windows, tạo mới bằng API với các path POSIX bên dưới để tránh `Path` CLI đổi `/` thành `\` trước portable-path verifier:

```powershell
if (-not (Test-Path outputs/paper_v2/protocol/fleurs_run_registry.json)) {
  $CreateFleursRegistry = @"
from scripts.run_external_fleurs import create_run_registry
create_run_registry(
    'outputs/paper_v2/protocol/fleurs_run_registry.json',
    preparation_lock_path='outputs/paper_v2/protocol/fleurs_test_lock.json',
    split_lock_path='outputs/paper_v2/protocol/split_lock.json',
    decision_lock_path='outputs/paper_v2/protocol/best_lambda_decision.json',
    config_paths_by_role={
        'ordinary_baseline': 'configs/phat/lambda_0.yaml',
        'selected_method': '$SelectedConfig',
        'locked_control': '$ControlConfig',
    },
)
"@
  & $Python -c $CreateFleursRegistry
}
```

Smoke riêng năm câu, dùng output và receipt riêng để không chiếm canonical path:

```powershell
& $Python scripts/run_external_fleurs_inference_runtime.py `
  --run-registry outputs/paper_v2/protocol/fleurs_run_registry.json `
  --inference-runtime-lock outputs/paper_v2/protocol/inference_runtime_lock.json `
  --output-dir outputs/paper_v2/smoke/fleurs `
  --receipt outputs/paper_v2/smoke/fleurs_execution_receipt.json `
  --limit 5 `
  --device auto
```

Chạy đủ 857 câu:

```powershell
& $Python scripts/run_external_fleurs_inference_runtime.py `
  --run-registry outputs/paper_v2/protocol/fleurs_run_registry.json `
  --inference-runtime-lock outputs/paper_v2/protocol/inference_runtime_lock.json `
  --device auto
```

Resume an toàn:

```powershell
& $Python scripts/run_external_fleurs_inference_runtime.py `
  --run-registry outputs/paper_v2/protocol/fleurs_run_registry.json `
  --inference-runtime-lock outputs/paper_v2/protocol/inference_runtime_lock.json `
  --device auto `
  --resume
```

Nếu máy inference bị ngắt và cần restore từ backup, giữ nguyên relative path của registry, inference/training locks, checkpoint, FLEURS bundle và từng `pred_*.csv`; copy kèm `.csv.resume.json` và `.csv.recovery.json` nếu receipt tồn tại. Chỉ chạy `--resume` khi current runtime vẫn khớp `inference_runtime_lock.json`. Khi role đã hoàn tất, resume/recovery sidecar phải biến mất.

Output:

```text
outputs/paper_v2/protocol/fleurs_run_registry.json
outputs/paper_v2/external/fleurs/predictions/pred_<configuration_id>.csv
outputs/paper_v2/external/fleurs/predictions/pred_<configuration_id>.csv.provenance.json
outputs/paper_v2/external/fleurs/predictions/pred_<configuration_id>.csv.inference_runtime.json
outputs/paper_v2/external/fleurs/external_fleurs_results.csv
outputs/paper_v2/external/fleurs/external_fleurs_results.csv.provenance.json
outputs/paper_v2/external/fleurs/external_fleurs_results.csv.inference_runtime.json
outputs/paper_v2/protocol/fleurs_execution_receipt.json
```

Ba prediction phải có đúng 857 dòng, cùng thứ tự `utt_id/ref`; result có đúng ba dòng role và đủ `WER/CER/TER/DER/FCER/SWDR` với `metric_version=aligned_v1`. Sau prefix scalar cũ, CSV còn ghi `*_numerator`, `*_denominator` cho cả sáu metric và `ter_coverage/der_coverage/fcer_coverage`; result sidecar phải có `provenance_version=paper_v2_fleurs_results_v4`. Base provenance giữ `method_runtime_verified=false` nhưng vẫn bind method environment/source identity; extension sidecar và receipt phải bind SHA-256 của base artifact/provenance và current inference-runtime identity. Verify lại receipt mà không infer:

```powershell
& $Python scripts/run_external_fleurs_inference_runtime.py `
  --run-registry outputs/paper_v2/protocol/fleurs_run_registry.json `
  --inference-runtime-lock outputs/paper_v2/protocol/inference_runtime_lock.json `
  --verify-receipt-only
```

Khi viết paper, CER `aligned_v1` có tính khoảng trắng sau normalize.
TER/DER/FCER là conditional diagnostic, còn SWDR chỉ bao phủ năm từ
`đã/có/là/một/và`. Không kết luận một system tốt hơn chỉ từ scalar
TER/DER/FCER trong `external_fleurs_results.csv`; phải báo cáo kèm
numerator, denominator/coverage đã ghi trong cùng dòng và WER/CER.
Chi tiết định nghĩa nằm trong `docs/prediction_schema.md`.

## 12. Aggregate final results

Aggregate cả sáu zero-shot và ba final LoRA, dùng chính final manifest để kiểm tra ID/reference/condition:

```powershell
& $Python scripts/aggregate_results.py `
  --input-glob 'outputs/paper_v2/predictions/zero_shot/pred_*.csv' `
  --input-glob 'outputs/paper_v2/final_predictions/*/predictions.csv' `
  --benchmark-manifest outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl `
  --formal-paper-v2 `
  --split-lock outputs/paper_v2/protocol/split_lock.json `
  --decision-lock outputs/paper_v2/protocol/best_lambda_decision.json `
  --final-benchmark-lock outputs/paper_v2/protocol/final_benchmark_lock.json `
  --metric-version aligned_v1 `
  --output-dir outputs/paper_v2/analysis/final
```

Nếu tiến trình bị ngắt giữa hai CSV, chạy lại **đúng lệnh trên** và thêm
`--resume`. Script chỉ phục hồi khi canonical file, stage và transaction journal
đều khớp SHA-256 của đúng phép tính này; không dùng `--overwrite` để phục hồi:

```powershell
& $Python scripts/aggregate_results.py `
  --input-glob 'outputs/paper_v2/predictions/zero_shot/pred_*.csv' `
  --input-glob 'outputs/paper_v2/final_predictions/*/predictions.csv' `
  --benchmark-manifest outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl `
  --formal-paper-v2 `
  --split-lock outputs/paper_v2/protocol/split_lock.json `
  --decision-lock outputs/paper_v2/protocol/best_lambda_decision.json `
  --final-benchmark-lock outputs/paper_v2/protocol/final_benchmark_lock.json `
  --metric-version aligned_v1 `
  --output-dir outputs/paper_v2/analysis/final `
  --resume
```

Output:

```text
outputs/paper_v2/analysis/final/results_by_snr.csv
outputs/paper_v2/analysis/final/results_by_noise_type.csv
outputs/paper_v2/analysis/final/aggregate_results.provenance.json
outputs/paper_v2/analysis/final/results.bundle.commit.json
```

Với đủ chín run, `results_by_snr.csv` phải có 9 × 7 = 63 dòng và `results_by_noise_type.csv` phải có 9 × 4 = 36 dòng. Mỗi run có 460 dòng ở từng nhóm `clean/20/10/5/0`, 1.840 ở `noisy_all`, 2.300 ở `all`. Noise types hiện là `clean/music/noise/speech`; script tự nhận thêm type mới nếu protocol sau này thay đổi.

Mỗi dòng aggregate còn ghi `prediction_sha256`, `benchmark_manifest_sha256` và `benchmark_manifest_format`; không nhận result thiếu ba provenance field này cho paper-v2. Schema giữ nguyên prefix scalar/provenance cũ và nối thêm `*_numerator`, `*_denominator` cho WER/CER/TER/DER/FCER/SWDR cùng `ter_coverage`, `der_coverage`, `fcer_coverage`. Mọi rate phải khớp numerator chia denominator theo ratio-of-totals; coverage phải khớp eligible denominator chia `wer_denominator`.

## 13. Error analysis, confusion matrix và breakdown

Phân tích chính tập trung vào ba LoRA role đã khóa:

```powershell
& $Python scripts/error_analysis.py `
  --pred-glob 'outputs/paper_v2/final_predictions/*/predictions.csv' `
  --formal-paper-v2 `
  --benchmark-manifest outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl `
  --split-lock outputs/paper_v2/protocol/split_lock.json `
  --decision-lock outputs/paper_v2/protocol/best_lambda_decision.json `
  --final-benchmark-lock outputs/paper_v2/protocol/final_benchmark_lock.json `
  --out-dir outputs/paper_v2/error_analysis/final
```

Sinh WER decomposition, TER/DER breakdown, mất dấu, sai tone, sai vowel quality và `đ/d`:

```powershell
& $Python scripts/build_error_breakdowns.py `
  --events outputs/paper_v2/error_analysis/final/error_events.csv `
  --formal-paper-v2 `
  --benchmark-manifest outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl `
  --split-lock outputs/paper_v2/protocol/split_lock.json `
  --decision-lock outputs/paper_v2/protocol/best_lambda_decision.json `
  --final-benchmark-lock outputs/paper_v2/protocol/final_benchmark_lock.json `
  --out-dir outputs/paper_v2/error_analysis/final/breakdowns
```

Sinh tone confusion matrix, final-coda matrix và short-word deletion examples. Rehydrate lambda động ngay trong terminal phân tích:

```powershell
$Decision = Get-Content -Raw outputs/paper_v2/protocol/best_lambda_decision.json | ConvertFrom-Json
function Get-LambdaKey([double] $Value) {
  $Value.ToString('0.##', [Globalization.CultureInfo]::InvariantCulture)
}
$SelectedLambda = Get-LambdaKey ([double] $Decision.selected_lambda)
$ControlLambda = Get-LambdaKey ([double] $Decision.locked_control_lambda)
$Focus = @(
  '--focus-run', 'ordinary_lora:0',
  '--focus-run', "tone_aware_lora:$SelectedLambda",
  '--focus-run', "tone_aware_lora:$ControlLambda"
)
$FormalFinal = @(
  '--formal-paper-v2',
  '--benchmark-manifest', 'outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl',
  '--split-lock', 'outputs/paper_v2/protocol/split_lock.json',
  '--decision-lock', 'outputs/paper_v2/protocol/best_lambda_decision.json',
  '--final-benchmark-lock', 'outputs/paper_v2/protocol/final_benchmark_lock.json'
)
& $Python scripts/build_error_artifacts.py --artifact tone --events outputs/paper_v2/error_analysis/final/error_events.csv @Focus @FormalFinal --out-dir outputs/paper_v2/error_analysis/final/artifacts
& $Python scripts/build_error_artifacts.py --artifact coda --events outputs/paper_v2/error_analysis/final/error_events.csv @Focus @FormalFinal --out-dir outputs/paper_v2/error_analysis/final/artifacts
& $Python scripts/build_error_artifacts.py --artifact short-word --events outputs/paper_v2/error_analysis/final/error_events.csv @Focus @FormalFinal --out-dir outputs/paper_v2/error_analysis/final/artifacts
```

Nếu một bundle bị ngắt giữa các lần commit, giữ nguyên input và toàn bộ tham số,
sau đó chạy lại đúng command với `--resume`:

```powershell
& $Python scripts/error_analysis.py `
  --pred-glob 'outputs/paper_v2/final_predictions/*/predictions.csv' `
  --formal-paper-v2 `
  --benchmark-manifest outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl `
  --split-lock outputs/paper_v2/protocol/split_lock.json `
  --decision-lock outputs/paper_v2/protocol/best_lambda_decision.json `
  --final-benchmark-lock outputs/paper_v2/protocol/final_benchmark_lock.json `
  --out-dir outputs/paper_v2/error_analysis/final `
  --resume
& $Python scripts/build_error_breakdowns.py `
  --events outputs/paper_v2/error_analysis/final/error_events.csv `
  --formal-paper-v2 `
  --benchmark-manifest outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl `
  --split-lock outputs/paper_v2/protocol/split_lock.json `
  --decision-lock outputs/paper_v2/protocol/best_lambda_decision.json `
  --final-benchmark-lock outputs/paper_v2/protocol/final_benchmark_lock.json `
  --out-dir outputs/paper_v2/error_analysis/final/breakdowns `
  --resume
& $Python scripts/build_error_artifacts.py --artifact tone --events outputs/paper_v2/error_analysis/final/error_events.csv @Focus @FormalFinal --out-dir outputs/paper_v2/error_analysis/final/artifacts --resume
& $Python scripts/build_error_artifacts.py --artifact coda --events outputs/paper_v2/error_analysis/final/error_events.csv @Focus @FormalFinal --out-dir outputs/paper_v2/error_analysis/final/artifacts --resume
& $Python scripts/build_error_artifacts.py --artifact short-word --events outputs/paper_v2/error_analysis/final/error_events.csv @Focus @FormalFinal --out-dir outputs/paper_v2/error_analysis/final/artifacts --resume
```

`--resume` chỉ hoàn tất transaction có journal/commit-marker và SHA-256 input
khớp chính xác. Orphan stage, input đổi hoặc output bị sửa đều phải dừng; không
dùng `--overwrite` như một cách recovery.

Output chính:

```text
error_events.csv
error_summary.csv
error_analysis.provenance.json
error_analysis.bundle.commit.json
breakdowns/wer_decomposition.csv
breakdowns/ter_breakdown.csv
breakdowns/der_breakdown.csv
breakdowns/orthographic_breakdown.csv
breakdowns/diacritic_error_events.csv
breakdowns/error_breakdowns.provenance.json
breakdowns/error_breakdowns.bundle.commit.json
artifacts/tone_confusion_matrix.csv
artifacts/tone_confusion_matrix.png
artifacts/tone_confusion_matrix.provenance.json
artifacts/tone_confusion_matrix.bundle.commit.json
artifacts/final_coda_confusion_matrix.csv
artifacts/final_coda_confusion_matrix.png
artifacts/final_coda_confusion_matrix.provenance.json
artifacts/final_coda_confusion_matrix.bundle.commit.json
artifacts/short_word_deletion_examples.csv
artifacts/short_word_deletion_examples.provenance.json
artifacts/short_word_deletion_examples.bundle.commit.json
```

FLEURS error analysis là replication phụ. Vì base provenance cố ý giữ `method_runtime_verified=false`, chạy downstream này ở non-formal mode; tính toàn vẹn formal của inference sẽ do FLEURS execution receipt và receipt bàn giao cuối xác minh. Dùng directory riêng và thêm `--overall-only` cho ba artifact vì FLEURS chỉ clean:

```powershell
& $Python scripts/error_analysis.py `
  --pred-glob 'outputs/paper_v2/external/fleurs/predictions/pred_*.csv' `
  --benchmark-manifest data/manifests/fleurs/paper_v2/test.jsonl `
  --out-dir outputs/paper_v2/external/fleurs/error_analysis
& $Python scripts/build_error_breakdowns.py `
  --events outputs/paper_v2/external/fleurs/error_analysis/error_events.csv `
  --benchmark-manifest data/manifests/fleurs/paper_v2/test.jsonl `
  --out-dir outputs/paper_v2/external/fleurs/error_analysis/breakdowns
$FleursAnalysis = @(
  '--benchmark-manifest', 'data/manifests/fleurs/paper_v2/test.jsonl'
)
& $Python scripts/build_error_artifacts.py --artifact tone --events outputs/paper_v2/external/fleurs/error_analysis/error_events.csv @Focus @FleursAnalysis --overall-only --out-dir outputs/paper_v2/external/fleurs/error_analysis/artifacts
& $Python scripts/build_error_artifacts.py --artifact coda --events outputs/paper_v2/external/fleurs/error_analysis/error_events.csv @Focus @FleursAnalysis --overall-only --out-dir outputs/paper_v2/external/fleurs/error_analysis/artifacts
& $Python scripts/build_error_artifacts.py --artifact short-word --events outputs/paper_v2/external/fleurs/error_analysis/error_events.csv @Focus @FleursAnalysis --overall-only --out-dir outputs/paper_v2/external/fleurs/error_analysis/artifacts
```

## 14. Paired bootstrap 1.000 lần

Tạo mapping theo `configuration_id` trong decision, không theo tên file/lambda đoán tay:

```powershell
$Decision = Get-Content -Raw outputs/paper_v2/protocol/best_lambda_decision.json | ConvertFrom-Json
$RunsByRole = @{}
foreach ($Run in $Decision.locked_configurations) { $RunsByRole[$Run.role] = $Run }
if ($RunsByRole.Count -ne 3) { throw 'Decision does not contain exactly three roles' }
$OrdinaryId = [string] $RunsByRole['ordinary_baseline'].configuration_id
$SelectedId = [string] $RunsByRole['selected_method'].configuration_id
$ControlId = [string] $RunsByRole['locked_control'].configuration_id
```

Final robustness bootstrap lấy `source_utt_id` làm cluster để clean và bốn noisy replica của cùng câu luôn được resample cùng nhau:

```powershell
& $Python scripts/cluster_bootstrap_ci.py `
  --decision-lock outputs/paper_v2/protocol/best_lambda_decision.json `
  --benchmark-manifest outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl `
  --formal-paper-v2 `
  --split-lock outputs/paper_v2/protocol/split_lock.json `
  --final-benchmark-lock outputs/paper_v2/protocol/final_benchmark_lock.json `
  --prediction "$OrdinaryId=outputs/paper_v2/final_predictions/ordinary_baseline/predictions.csv" `
  --prediction "$SelectedId=outputs/paper_v2/final_predictions/selected_method/predictions.csv" `
  --prediction "$ControlId=outputs/paper_v2/final_predictions/locked_control/predictions.csv" `
  --cluster-unit source_utt_id `
  --n-bootstrap 1000 `
  --ci-level 0.95 `
  --bootstrap-seed 42 `
  --output outputs/paper_v2/statistics/bootstrap_ci_final.csv
```

Output final bootstrap là một transaction ba file; CSV không có provenance/commit marker đi kèm thì không phải paper evidence:

```text
outputs/paper_v2/statistics/bootstrap_ci_final.csv
outputs/paper_v2/statistics/bootstrap_ci_final.csv.provenance.json
outputs/paper_v2/statistics/cluster_bootstrap.bundle.commit.json
```

FLEURS có một quan sát độc lập trên mỗi utt, nên dùng singleton cluster. Chạy bootstrap FLEURS ở non-formal mode vì base provenance giữ cờ runtime train trung thực; receipt bàn giao cuối sẽ bind output này với FLEURS execution receipt đã xác minh:

```powershell
& $Python scripts/cluster_bootstrap_ci.py `
  --decision-lock outputs/paper_v2/protocol/best_lambda_decision.json `
  --benchmark-manifest data/manifests/fleurs/paper_v2/test.jsonl `
  --prediction "$OrdinaryId=outputs/paper_v2/external/fleurs/predictions/pred_$OrdinaryId.csv" `
  --prediction "$SelectedId=outputs/paper_v2/external/fleurs/predictions/pred_$SelectedId.csv" `
  --prediction "$ControlId=outputs/paper_v2/external/fleurs/predictions/pred_$ControlId.csv" `
  --cluster-unit utt_id_singleton_external `
  --n-bootstrap 1000 `
  --ci-level 0.95 `
  --bootstrap-seed 42 `
  --output outputs/paper_v2/external/fleurs/bootstrap_ci_results.csv
```

Output FLEURS bootstrap:

```text
outputs/paper_v2/external/fleurs/bootstrap_ci_results.csv
outputs/paper_v2/external/fleurs/bootstrap_ci_results.csv.provenance.json
outputs/paper_v2/external/fleurs/cluster_bootstrap.bundle.commit.json
```

Nếu final robustness bootstrap formal bị ngắt trong transaction, chạy lại **toàn bộ exact command** với thêm `--resume` sau `--output`, ví dụ phần cuối:

```powershell
  --output outputs/paper_v2/statistics/bootstrap_ci_final.csv `
  --resume
```

`--resume` chỉ hợp lệ cho formal paper-v2 và loại trừ `--overwrite`. Nó chỉ phục hồi khi decision, benchmark, ba prediction + sidecar, cluster unit và toàn bộ tham số bootstrap khớp SHA-256; canonical/stage/marker bị sửa phải fail closed. FLEURS bootstrap hiện chạy non-formal nên không dùng `--resume`; nếu bị ngắt, kiểm tra transaction chưa tạo canonical bundle rồi chạy lại từ đầu vào output sạch.

Mỗi output có 3 pair × 4 metric (`ΔWER`, `ΔCER`, `ΔTER`, `ΔDER`) = 12 dòng. Quy ước delta là role B trừ role A; đọc dấu delta cùng CI, không chỉ đọc `ci_excludes_zero`.

## 15. Bàn giao artifact giữa các máy bằng SHA

Luôn giữ nguyên relative path trong repository. Trước khi copy một bundle, tạo manifest SHA:

### Final benchmark qua Git LFS

Bundle mà Phúc publish và Phát/Trung pull phải gồm đúng các path sau:

```text
data/derived/paper_v2/final_benchmark/                         # 2.300 WAV, Git LFS
outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl
outputs/paper_v2/protocol/final_benchmark_lock.json
outputs/paper_v2/protocol/final_benchmark_audit.csv
outputs/paper_v2/protocol/final_benchmark_handoff_sha256.csv
```

Chỉ **một máy builder được chạy** `build_final_benchmark.py`. Sau khi
publish, Phúc và Phát chỉ pull và đọc bundle; không chạy builder lại. Metadata
benchmark là **một file JSONL duy nhất, đúng 2.300 dòng**:
`outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl`. Mỗi dòng trỏ tới
một WAV trong LFS, vì vậy bundle vẫn có 2.300 WAV riêng; không nhúng binary
audio vào JSONL.

Trước khi public, builder phải đọc
`docs/final_benchmark_attribution.md` và xác nhận repository/kênh phân phối
phù hợp CC BY-NC-SA 4.0 của VIVOS cùng license/attribution theo từng
nguồn MUSAN. Nếu chưa xác nhận, chỉ dùng private LFS hoặc external
handoff.

Builder kiểm tra allow-list/LFS, stage đúng bundle và xác nhận JSONL
không bị chia nhỏ:

```powershell
git lfs install

$Manifest = 'outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl'
$AudioDir = 'data/derived/paper_v2/final_benchmark'
$Rows = @(Get-Content -LiteralPath $Manifest)
$Wavs = @(Get-ChildItem -LiteralPath $AudioDir -Recurse -File -Filter *.wav)
if ($Rows.Count -ne 2300) { throw "Expected one 2300-line JSONL, got $($Rows.Count)" }
if ($Wavs.Count -ne 2300) { throw "Expected 2300 WAVs, got $($Wavs.Count)" }

$Repo = (Resolve-Path .).Path.TrimEnd('\')
$Probe = $Wavs[0].FullName.Substring($Repo.Length + 1).Replace('\', '/')
git check-ignore -q -- $Probe
if ($LASTEXITCODE -eq 0) { throw 'Final benchmark WAV allow-list is not active' }
$Attr = git check-attr filter -- $Probe
if ($Attr -notmatch ': filter: lfs$') { throw "Final benchmark WAV is not Git LFS: $Attr" }

git add -- `
  .gitattributes `
  .gitignore `
  docs/final_benchmark_attribution.md `
  $AudioDir `
  $Manifest `
  outputs/paper_v2/protocol/final_benchmark_lock.json `
  outputs/paper_v2/protocol/final_benchmark_audit.csv `
  outputs/paper_v2/protocol/final_benchmark_handoff_sha256.csv

$LfsCount = @(git lfs ls-files | Select-String 'data/derived/paper_v2/final_benchmark/.+\.wav$').Count
if ($LfsCount -ne 2300) { throw "Expected 2300 staged LFS WAVs, got $LfsCount" }
git lfs fsck
git diff --cached --check
git diff --cached --stat
```

Chỉ commit/push sau khi Trung duyệt staged diff. Không dùng `git add .`,
`git add -A` hoặc `git add -f`.

Sau khi staged diff được duyệt:

```powershell
git commit -m "data(benchmark): publish locked self-contained final benchmark"
$Branch = git branch --show-current
git push origin $Branch
```

Không add raw VIVOS, raw MUSAN hoặc toàn bộ `data/`. Audio final chỉ được track ở allow-list trên và phải là Git LFS, không phải Git blob thường. Mỗi máy nhận cài LFS rồi materialize object trước khi verify:

```powershell
git lfs install
git pull --ff-only
git lfs pull --include="data/derived/paper_v2/final_benchmark/**"

$Manifest = 'outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl'
$Rows = @(Get-Content -LiteralPath $Manifest)
if ($Rows.Count -ne 2300) { throw "Expected one 2300-line JSONL, got $($Rows.Count)" }
$Wavs = @(Get-ChildItem data/derived/paper_v2/final_benchmark -Recurse -File -Filter *.wav)
if ($Wavs.Count -ne 2300) { throw "Expected 2300 benchmark WAVs, got $($Wavs.Count)" }
$FirstLine = Get-Content -LiteralPath $Wavs[0].FullName -TotalCount 1 -ErrorAction SilentlyContinue
if ($FirstLine -eq 'version https://git-lfs.github.com/spec/v1') { throw 'Git LFS objects were not materialized' }
```

Nếu `git lfs pull` báo quota/permission hoặc quyền tái phân phối audio chưa được xác nhận, dừng; dùng private LFS/external handoff, không force-add WAV vào Git thường.

```powershell
function New-HandoffManifest {
  param(
    [Parameter(Mandatory)] [string[]] $InputPath,
    [Parameter(Mandatory)] [string] $OutputCsv
  )
  $Repo = (Resolve-Path .).Path.TrimEnd('\')
  $Files = foreach ($Path in $InputPath) {
    Get-ChildItem -LiteralPath $Path -Recurse -File
  }
  $Rows = foreach ($File in ($Files | Sort-Object FullName -Unique)) {
    [pscustomobject]@{
      path = $File.FullName.Substring($Repo.Length + 1).Replace('\', '/')
      bytes = $File.Length
      sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $File.FullName).Hash.ToLowerInvariant()
    }
  }
  $Parent = Split-Path -Parent $OutputCsv
  if ($Parent) { New-Item -ItemType Directory -Force $Parent | Out-Null }
  $Rows | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation -Encoding utf8
}
```

Ví dụ Phát bàn giao checkpoints và dev-screen:

```powershell
New-HandoffManifest `
  -InputPath outputs/paper_v2/checkpoints, outputs/paper_v2/logs, outputs/paper_v2/dev_screen, outputs/paper_v2/protocol/environment_lock.json, outputs/paper_v2/protocol/method_lock.json, outputs/paper_v2/protocol/best_lambda_decision.json `
  -OutputCsv outputs/paper_v2/handoff/phat_to_trung_sha256.csv
```

Sau khi build final benchmark, Phúc tạo manifest handoff riêng; transaction self-contained phải có 2.300 WAV cộng manifest, lock và audit:

```powershell
New-HandoffManifest `
  -InputPath outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl, outputs/paper_v2/protocol/final_benchmark_lock.json, outputs/paper_v2/protocol/final_benchmark_audit.csv, data/derived/paper_v2/final_benchmark `
  -OutputCsv outputs/paper_v2/protocol/final_benchmark_handoff_sha256.csv

$Handoff = @(Import-Csv outputs/paper_v2/protocol/final_benchmark_handoff_sha256.csv)
if ($Handoff.Count -ne 2303) { throw "Expected 2303 handoff entries, got $($Handoff.Count)" }
```

Bundle FLEURS chuyển sang máy inference của Trung phải gồm cả WAV, manifest, lock và audit; checkpoint/training locks vẫn giữ nguyên provenance của Phát:

```powershell
New-HandoffManifest `
  -InputPath data/manifests/fleurs/paper_v2, outputs/paper_v2/protocol/fleurs_test_lock.json, outputs/paper_v2/protocol/fleurs_test_audit.csv `
  -OutputCsv outputs/paper_v2/handoff/fleurs_bundle_sha256.csv
```

Tương tự, Phúc bàn giao nguyên directory `outputs/paper_v2/predictions/zero_shot`; máy chạy final LoRA/FLEURS bàn giao prediction, provenance và result tương ứng. Không chỉ copy CSV mà bỏ sidecar/registry/runtime config.

Máy nhận đặt file đúng relative path rồi verify:

```powershell
$Manifest = Import-Csv outputs/paper_v2/protocol/final_benchmark_handoff_sha256.csv
$Bad = foreach ($Row in $Manifest) {
  if (-not (Test-Path -LiteralPath $Row.path)) {
    $Row.path
    continue
  }
  $File = Get-Item -LiteralPath $Row.path
  $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Row.path).Hash.ToLowerInvariant()
  if ([string] $File.Length -ne [string] $Row.bytes -or $Hash -ne $Row.sha256) { $Row.path }
}
if ($Bad) { $Bad; throw 'Handoff verification failed' }
'PASS: handoff is byte-identical'
```

Không mở/sửa CSV bằng Excel trước khi hash; Excel có thể đổi encoding, newline hoặc số thập phân.

## 16. Stop conditions

Dừng pipeline và báo Trung nếu gặp một trong các trường hợp sau:

- Git commit/source tree khác reference hoặc repository dirty trước formal capture.
- Bất kỳ completed hash ở mục 2 khác giá trị khóa.
- Test hoặc real-data smoke fail; CUDA/FP16 không sẵn sàng trên máy đã chọn.
- MUSAN archive, split hoặc audio hash không khớp; train/dev/test có overlap.
- Noisy-dev không đúng 14.125 dòng hoặc bị gắn `final_test_eligible=true`.
- Formal method/environment lock không được tạo hoặc source thay đổi sau lock.
- Một trong năm training run dùng seed/config/batch/decode khác.
- Ablation thiếu lambda, prediction không đủ 14.125 dòng hoặc provenance không phải full noisy-dev.
- Decision không khóa đúng ordinary + selected positive tone-aware + distinct locked control.
- Final benchmark v2 không data-only, không self-contained đủ 2.300 WAV, bị rebuild khác hash hoặc không chỉ dùng MUSAN test.
- Phát dùng bất kỳ final prediction/metric nào để chọn lambda thay vì chỉ dùng noisy-dev.
- Zero-shot config còn placeholder/floating revision hoặc prediction thiếu provenance.
- Inference runtime hiện tại không khớp `inference_runtime_lock.json`, hoặc final/FLEURS execution receipt không verify được toàn bộ artifact/transitive training provenance.
- Base provenance FLEURS bị sửa thành `method_runtime_verified=true`, thiếu `*.inference_runtime.json`, hoặc extension/receipt không ghi và verify `inference_runtime_verified=true`.
- Final/FLEURS ba role khác decision lock, checkpoint SHA khác hoặc ref/utt_id không paired.
- Aggregate không cho 63/36 dòng khi đủ chín final runs.
- Error operations không reconcile hoặc bootstrap không có đúng 12 dòng/full pairing.

## 17. Bảng old-vs-new sau khi thu đủ kết quả

Generator deterministic đã có sẵn; chỉ chạy sau khi toàn bộ canonical output cũ và paper-v2 output mới ở đúng path mặc định:

```powershell
& $Python scripts/build_old_vs_new_comparison.py
```

Nếu bị ngắt giữa CSV/Markdown/provenance, chạy lại đúng command với
`--resume`; chỉ bundle có cùng deterministic bytes mới được hoàn tất:

```powershell
& $Python scripts/build_old_vs_new_comparison.py --resume
```

Output immutable/no-overwrite:

```text
outputs/paper_v2/reports/old_vs_new_comparison.csv
outputs/paper_v2/reports/old_vs_new_comparison.md
outputs/paper_v2/reports/old_vs_new_comparison.provenance.json
outputs/paper_v2/reports/old_vs_new_comparison.bundle.commit.json
```

Formal mode đọc các nguồn canonical sau:

- Cũ: `outputs/analysis/results_by_snr.csv`, `results_by_noise_type.csv`, benchmark cũ, FLEURS results/bootstrap/predictions cũ.
- Mới: aggregate final, final benchmark, three-role bootstrap, FLEURS portable manifest + preparation lock + results/provenance/bootstrap.
- Protocol: split/noise/noisy-dev/environment/method/decision/final locks.

CSV ghi `section`, `artifact`, old/new scope, run identity, metric/statistic, old/new/delta, comparability/reason, counts, hashes và protocol label. Generator tự rehydrate selected/control từ decision lock; không nhập lambda hoặc số metric bằng tay.

Chỉ để xem trước khi output mới chưa đủ, dùng diagnostic mode và bắt buộc đổi cả ba output path để không thể bị nhầm với bảng formal:

```powershell
& $Python scripts/build_old_vs_new_comparison.py `
  --diagnostic-allow-partial `
  --output-csv outputs/paper_v2/reports/preview_old_vs_new.csv `
  --output-md outputs/paper_v2/reports/preview_old_vs_new.md `
  --output-provenance outputs/paper_v2/reports/preview_old_vs_new.provenance.json
```

Quy tắc diễn giải ba block:

- Internal robustness: historical 300-source/1.500-row benchmark so với final 460-source/2.300-row benchmark. Generator phải **withhold delta** vì sample/protocol khác nhau.
- Lambda screen: best cũ 0.05 so với selected/locked-control mới trên noisy-dev. Không gọi đây là paired delta vì tập chọn cũ và mới khác nhau.
- FLEURS: chỉ tính direct delta khi chứng minh được đúng 857 `utt_id` và `ref` ở cả hai phía; nếu không thì withhold.

Ghi rõ checkpoint cũ được train trước tone-alignment/protocol fix. Không diễn giải việc metric thay đổi là do riêng tone loss khi data split, noise partition, checkpoint và protocol đều đã đổi.

### Receipt bàn giao cuối trên máy Trung

Chỉ tạo receipt này sau khi hai wrapper `--verify-only`/`--verify-receipt-only` đã PASS và toàn bộ aggregate, error artifacts, bootstrap cùng old-vs-new đã hoàn tất. Lệnh dưới đây liệt kê deterministic mọi file trong đúng các output roots canonical, loại trừ state tạm, rồi truyền từng path bằng `--artifact`; không dùng glob bên trong verifier:

```powershell
$DeliveryRoots = @(
  'outputs/paper_v2/analysis/final',
  'outputs/paper_v2/error_analysis/final',
  'outputs/paper_v2/statistics',
  'outputs/paper_v2/external/fleurs/error_analysis',
  'outputs/paper_v2/external/fleurs/bootstrap_ci_results.csv',
  'outputs/paper_v2/external/fleurs/bootstrap_ci_results.csv.provenance.json',
  'outputs/paper_v2/external/fleurs/cluster_bootstrap.bundle.commit.json',
  'outputs/paper_v2/reports/old_vs_new_comparison.csv',
  'outputs/paper_v2/reports/old_vs_new_comparison.md',
  'outputs/paper_v2/reports/old_vs_new_comparison.provenance.json',
  'outputs/paper_v2/reports/old_vs_new_comparison.bundle.commit.json'
)
$Repo = (Resolve-Path .).Path.TrimEnd('\')
$DeliveryFiles = foreach ($Root in $DeliveryRoots) {
  if (-not (Test-Path -LiteralPath $Root)) { throw "Missing delivery root/artifact: $Root" }
  Get-Item -LiteralPath $Root | ForEach-Object {
    if ($_.PSIsContainer) { Get-ChildItem -LiteralPath $_.FullName -Recurse -File } else { $_ }
  }
}
$DeliveryFiles = @($DeliveryFiles | Where-Object {
  $_.Name -notmatch '\.(tmp|resume\.json|recovery\.json)$' -and
  $_.FullName -notmatch '[\\/]\.[^\\/]+\.partial[\\/]'
} | Sort-Object FullName -Unique)
$DeliveryArgs = foreach ($File in $DeliveryFiles) {
  '--artifact'
  $File.FullName.Substring($Repo.Length + 1).Replace('\', '/')
}
& $Python scripts/verify_paper_v2_inference_delivery.py @DeliveryArgs `
  --inference-runtime-lock outputs/paper_v2/protocol/inference_runtime_lock.json `
  --final-lora-receipt outputs/paper_v2/protocol/final_lora_execution_receipt.json `
  --fleurs-receipt outputs/paper_v2/protocol/fleurs_execution_receipt.json `
  --output outputs/paper_v2/protocol/trung_delivery_receipt.json
& $Python scripts/verify_paper_v2_inference_delivery.py @DeliveryArgs `
  --inference-runtime-lock outputs/paper_v2/protocol/inference_runtime_lock.json `
  --final-lora-receipt outputs/paper_v2/protocol/final_lora_execution_receipt.json `
  --fleurs-receipt outputs/paper_v2/protocol/fleurs_execution_receipt.json `
  --output outputs/paper_v2/protocol/trung_delivery_receipt.json `
  --verify-existing
```

Receipt cuối phải trả `status=VERIFIED`. Bất kỳ artifact nào thay đổi byte, bị thiếu hoặc current inference runtime lệch lock đều làm lần verify sau dừng; khi đó không sửa receipt thủ công.

## 18. Không commit các file sau

- Raw VIVOS, MUSAN, FLEURS audio và derived noisy-dev. Ngoại lệ duy nhất là self-contained final-benchmark WAV đã được Trung duyệt, allow-list đúng path và track bằng Git LFS theo mục 15.
- Archive dataset, Hugging Face cache hoặc model snapshot cache.
- Checkpoint weights/adapters bị cấm, ngoại trừ đúng năm bundle `best/` inference-only đã allow-list theo phê duyệt branch ngày 2026-07-18. Kể cả trong năm run đó, `optimizer.pt`, `scheduler.pt`, `scaler.pt`, `rng_state.pt`, `checkpoint_step_*`, `final/` và mọi state phục vụ resume vẫn không được commit.
- Smoke outputs, partial/resume/temp/backup files, lock file tạm, log chứa path/token riêng của máy.
- Conda/venv, cache Python, IDE settings, secrets, Hugging Face token.
- Diagnostic preview old-vs-new; chỉ canonical formal CSV/Markdown/provenance mới được đưa ra review.

Chỉ chuẩn bị để review/commit: source, tests, config/template, `Guide.md`, manifest/lock/audit nhỏ đã được hash, canonical result CSV/Markdown/PNG cuối và provenance không chứa secret. Việc `git add`, commit, push hoặc force-add output phải do Trung duyệt từng nhóm thay đổi.

Phê duyệt ngày 2026-07-18 chỉ cho phép commit và push đúng năm bundle `best/` inference-only cùng kết quả formal lên branch `codex/phat-results-checkpoints` bằng Git LFS; đây không phải là phê duyệt merge vào `main`. Mỗi bundle chỉ gồm `adapter/{adapter_config.json,adapter_model.safetensors,README.md}`, `processor/{processor_config.json,tokenizer.json,tokenizer_config.json}`, `resolved_config.yaml`, `trainer_state.json`; bốn bundle tone-aware có thêm `tone_head.pt`. Trước khi merge, Trung vẫn phải review riêng staged diff, LFS objects và hash provenance.

## 19. Checklist hoàn tất

- [ ] Source commit và test suite PASS trên từng máy chạy.
- [ ] VIVOS/MUSAN/noisy-dev bytes khớp locks.
- [ ] Formal environment + method lock được khóa trước training.
- [ ] Năm lambda được train lại và evaluate đủ trên noisy-dev.
- [ ] Final benchmark v2 self-contained 2.300 WAV được Phúc build đúng một lần, handoff SHA PASS và publish qua kênh đã duyệt.
- [ ] Decision khóa selected lambda và locked control chỉ từ noisy-dev; final prediction/metric không tham gia selection.
- [ ] Inference runtime của máy Trung được capture riêng, `--verify-existing` PASS; training environment của Phát không bị sửa.
- [ ] Sáu zero-shot và ba final LoRA predictions hoàn chỉnh; final execution receipt PASS với inference=true/training-current=false.
- [ ] FLEURS 857 được chạy lại, ghi đúng nhãn legacy-exposed replication, base runtime flag=false và inference extensions/receipt PASS.
- [ ] Aggregate, error artifacts, WER/DER breakdown và hai bootstrap hoàn chỉnh.
- [ ] FLEURS downstream chạy non-formal và receipt bàn giao cuối bind đủ SHA-256 của artifact canonical.
- [ ] Handoff SHA của mọi artifact lớn PASS trên máy Trung.
- [ ] Bảng old-vs-new formal được tạo từ output canonical, provenance PASS và nêu rõ protocol/sample/checkpoint thay đổi.
- [ ] Chỉ sau review mới git add theo từng logical commit.
