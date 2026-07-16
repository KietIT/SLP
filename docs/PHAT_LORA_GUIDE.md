# Phat PhoWhisper LoRA Ablation Guide

> **Paper-v2 validity boundary:** the active protocol uses the speaker-disjoint
> VIVOS train/dev split and uses dev only for checkpoint/method/lambda
> selection. Final test evaluation is permitted only after decision-v3 locks
> the selected identities. The selected checkpoint is stored under `best/`.
> The old 1,500-row benchmark and `final/` commands
> documented in repository history used the official VIVOS test partition for
> model selection. Those artifacts are historical diagnostics, not paper-v2
> evidence. Checkpoints trained before the exact tone-alignment fix must also be
> retrained.

## Scope

This guide covers only Nguyen Thanh Phat's deliverables:

- ordinary PhoWhisper-base LoRA with `lambda = 0`
- tone-aware LoRA with `L_total = L_ASR + lambda * L_tone`
- lambda values `0`, `0.05`, `0.1`, `0.3`, and `0.5`
- checkpoint/resume, prediction export, metric aggregation, and best-lambda selection

The benchmark builder, zero-shot baselines, team-wide error analysis, and multi-seed experiments remain owned by the other assigned members.

## Required environment

Use the existing `slp` conda environment. Do not create another environment and do not use the default Python interpreter.

```powershell
conda activate slp
$Python = (Get-Command python).Source
& $Python --version
& $Python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
& $Python -m pip install -r requirements.txt
```

Keep using the same `$Python` variable in that activated PowerShell session for
all commands below. This avoids binding the guide to one user's Conda path.

The verified development machine used Python 3.11.15, PyTorch 2.11.0+cu128, CUDA 12.8, and an NVIDIA GeForce RTX 4060 Laptop GPU.

## File structure

```text
configs/phat/
  base.yaml
  lambda_0.yaml
  lambda_005.yaml
  lambda_01.yaml
  lambda_03.yaml
  lambda_05.yaml
  phat_pipeline.yaml

src/vitonesr/phat/
  config.py
  evaluation.py
  losses.py
  modeling.py
  protocol.py
  reproducibility.py
  selection.py
  trainer.py
  training_data.py

scripts/
  train_phat_lora.py
  train_all_lambdas.py
  evaluate_phat_checkpoint.py
  evaluate_all_lambdas.py
  select_best_lambda.py
  run_phat_pipeline.py

tests/phat/
```

Active paper-v2 artifacts are written under `outputs/paper_v2/` and are ignored
by Git except for explicitly reviewed protocol manifests/audits. Historical
artifacts remain under `outputs/phat/` and must not be mixed into paper-v2 runs.

## Dataset contracts

Training uses the locked paper-v2 JSONL manifests. Each row must contain real
`audio`, `text`, and `split` values. Every locked VIVOS audio SHA-256 is
validated before model loading or output-directory creation. Gate 1.2 only
validates training-noise paths plus the noise-manifest hash; Gate 2 must add
content hashes for every noise file before formal training is allowed.

```text
data/manifests/paper_v2/vivos_train.jsonl
data/manifests/paper_v2/vivos_dev.jsonl
data/manifests/paper_v2/vivos_test_legacy_exposed.jsonl
data/manifests/paper_v2/vivos_test_locked.jsonl
outputs/paper_v2/protocol/legacy_test_exposure.csv
data/manifests/noise/musan_noise.jsonl  # legacy/unlocked; not formal Gate-2 input
```

Create and verify the split with:

```powershell
python scripts/make_vivos_manifest.py `
  --vivos-root data/raw/vivos `
  --out-dir data/manifests/paper_v2 `
  --protocol-dir outputs/paper_v2/protocol `
  --legacy-benchmark-manifest outputs/benchmark/benchmark_manifest.csv `
  --expected-legacy-exposed 300 `
  --seed 42 `
  --dev-speaker-fraction 0.20
```

The reviewed seed-42 lock contains 8,835 train utterances from 36 speakers and
2,825 dev utterances from 10 speakers. It partitions all 760 official-test
utterances from 19 speakers into 300 `legacy_exposed` utterances found in the
historical 1,500-row benchmark and a disjoint 460-utterance unseen
`test_locked` complement. The exposure registry and source benchmark are
hash-bound by the split lock. The audit reports zero cross-split speaker,
utterance, or audio-hash overlap, proves exposed/locked disjointness, and proves
their union equals the official test inventory. Re-running the builder must
return `verified_existing`.

The official test manifest is sealed until a reviewed
`best_lambda_decision.json` with
`decision_version=paper_v2_method_lambda_decision_v3` binds the split lock,
dev-screen manifest, selection results/rule, selection evaluation contract,
and allowed test evaluation contracts. Each named `locked_configurations`
entry binds a unique `configuration_id` and role to its `method_id`,
`train_type`, lambda, seed, backbone name and immutable revision, checkpoint
fingerprint, resolved-config hash, and training-contract hash. The evaluator
requires one exact identity match; a bare or anonymous checkpoint allow-list is
not accepted.

The evaluator rejects test while `protocol.final_test_unlocked: false` and also
rejects it when that flag is true but the reviewed decision-v3 artifact is
missing or mismatched. For locked VIVOS dev/test evaluation, the configured
manifest path must also equal the canonical path recorded by the split lock;
a config cannot relabel or redirect the sealed test manifest as dev or external.
Both its canonical path and locked SHA-256 are rejected for every non-test
evaluation before the manifest loader runs. The lambda selector
also rejects test. Gate 1.2 exposes the clean dev manifest, but the configured
0/5-dB rule needs a Gate-2 noise-disjoint dev screen. Gate 2 is not a config-only
manifest swap: it must create and lock a derived noisy-dev benchmark whose
provenance binds the source `vivos_dev.jsonl` hash, a locked/disjoint dev-noise
registry, builder parameters/seed, output-manifest hash, and derived-audio
hashes. Its verifier must reject source-utterance drift, noise-pool overlap,
missing/changed audio, invalid SNR, and clipping.

The Gate-2 noise registry and noisy-dev benchmark are now locked, so the active
paper-v2 configs set `protocol.formal_training_unlocked: true`. The flag alone
does not authorize a formal run: the trainer still verifies the reviewed
environment/method locks, exact split/noise/noisy-dev hashes, source identity,
and audio inventory before loading a model or creating formal output. Clean-dev
evaluation is diagnostic only, must write to a separate smoke/diagnostic output,
and must never feed the low-SNR selector.

The evaluator accepts JSONL `audio`/`text` or CSV `audio_path`/`transcript`, but
every row must explicitly declare one of `split=dev`, `split=test`, or
`split=external`. One evaluation manifest cannot mix those splits.

## Tone labels and tone-aware loss

The shared `src/vitonesr/tone.py` implementation defines six classes:

| ID | Tone |
| ---: | --- |
| 0 | ngang |
| 1 | sac |
| 2 | huyen |
| 3 | hoi |
| 4 | nga |
| 5 | nang |

Transcripts are Unicode-normalized and split into syllable-like whitespace tokens. Tone marks are extracted from Unicode combining marks. Digits, punctuation, uppercase acronyms, and tokens without a Vietnamese vowel use `ignore_index = -100`.

Tone alignment tokenizes the complete normalized transcript and reconstructs
the same BPE sequence word by word: the first word has no synthetic prefix
space, and later words do. Any token-sequence mismatch raises before training;
there is no truncating `zip`. The `last_subtoken` policy attaches the tone to
the final BPE piece of each aligned word. Special tokens and padding remain
`-100`.

The decoder's final hidden state feeds a trainable `LayerNorm + Linear(512, 6)` tone head. Cross-entropy is computed only on valid labels. If a batch contains no valid tone positions, the function returns differentiable zero instead of `NaN`.

For `lambda = 0`, no tone head is created and the total loss is exactly the ASR loss. For positive lambda values, only PEFT LoRA weights and the tone head are trainable; the backbone remains frozen.

## Train one lambda

The command shape below is documented for use after Gates 1-3. It is not an
authorization to start a definitive run while the Gate-2 noise lock is absent.

```powershell
& $Python scripts\train_phat_lora.py `
  --config configs\phat\lambda_01.yaml `
  --device cuda
```

Useful overrides:

```text
--lambda-value
--seed
--manifest
--output-dir
--device
--overwrite
--max-train-samples
--max-train-steps
```

`--max-train-samples` and `--max-train-steps` are for real-data smoke validation. Outputs from limited runs must use a separate path containing `smoke` and must not be reported as the official ablation.

## Train all five lambdas

The command shape below is likewise blocked for definitive training until
Gates 1-3 are complete.

```powershell
& $Python scripts\train_all_lambdas.py `
  --config configs\phat\phat_pipeline.yaml `
  --device cuda
```

Default checkpoint roots:

```text
outputs/paper_v2/checkpoints/ckpt_lora_ordinary_lambda0/
outputs/paper_v2/checkpoints/ckpt_tone_lora_lambda_005/
outputs/paper_v2/checkpoints/ckpt_tone_lora_lambda_01/
outputs/paper_v2/checkpoints/ckpt_tone_lora_lambda_03/
outputs/paper_v2/checkpoints/ckpt_tone_lora_lambda_05/
```

Dev evaluation runs without training augmentation after every epoch. The
strictly lowest finite `dev_asr_loss` is saved under `best/`; a tie keeps the
earlier checkpoint. `final/` is retained only as an archive/resume state and is
not the default evaluation candidate.

Every saved checkpoint contains:

```text
best/ or final/
  adapter/
  processor/
  tone_head.pt          # positive lambda only
  optimizer.pt
  scheduler.pt
  scaler.pt
  rng_state.pt
  trainer_state.json
  resolved_config.yaml
```

Periodic `checkpoint_step_XXXXXX/` directories use the same layout.

## Resume training

Resume from a periodic checkpoint with the same model, lambda, and LoRA rank:

```powershell
& $Python scripts\train_phat_lora.py `
  --config configs\phat\lambda_01.yaml `
  --resume `
  --checkpoint outputs\paper_v2\checkpoints\ckpt_tone_lora_lambda_01\checkpoint_step_000500 `
  --device cuda
```

The adapter, tone head, optimizer, scheduler, gradient scaler, Python/NumPy/PyTorch RNG state, epoch, batch position, and global step are restored. Existing output directories are rejected unless `--resume` or `--overwrite` is explicit.

## Diagnostic clean-dev evaluation

This command evaluates clean dev only as a diagnostic after a compatible
checkpoint exists. Its output is deliberately isolated under `smoke/`; it is
not a lambda-selection artifact and cannot unlock final test.

```powershell
& $Python scripts\evaluate_phat_checkpoint.py `
  --config configs\phat\lambda_01.yaml `
  --checkpoint outputs\paper_v2\checkpoints\ckpt_tone_lora_lambda_01\best `
  --manifest data\manifests\paper_v2\vivos_dev.jsonl `
  --output-dir outputs\paper_v2\smoke\predictions\pred_tone_lora_lambda_01_clean_dev.csv `
  --device cuda
```

Filters are available through `--subset`, `--snr`, and `--noise-type`.

Every prediction CSV uses exactly:

```text
utt_id,dataset,model,model_size,train_type,lambda,seed,snr,noise_type,ref,hyp
```

The exact 11-column CSV is accompanied by
`<prediction>.csv.provenance.json`, which binds the prediction SHA-256 to its
evaluation split, manifest and selected-row hashes, full/partial filters,
checkpoint/config hashes, immutable backbone revision through the training and
evaluation contracts, training scope/contract, split/decision locks, row
count, and `metric_version=aligned_v1`. It records the complete runtime
evaluation contract (audio preprocessing and deterministic decoding included)
plus the runtime device/dtype, PyTorch, Transformers, and CUDA environment.
Aggregation reopens and re-hashes the manifest/prediction, reconstructs the
selected rows, and re-fingerprints the checkpoint and resolved config before
computing metrics. A filtered run must use a separate smoke path.

## Evaluate and aggregate all lambdas

```powershell
& $Python scripts\evaluate_all_lambdas.py `
  --config configs\phat\phat_pipeline.yaml `
  --device cuda
```

This command requires real `best/` checkpoints and evaluates the manifest
declared in each paper-v2 config. Once Gate 2 supplies the noise-disjoint dev
screen and its lock/verifier passes, it exports five prediction files and
writes:

```text
outputs/paper_v2/dev_screen/lambda_ablation_results.csv
```

The result contains WER, CER, TER, DER, FCER, and SWDR when the shared metric
implementation provides them. Every row also records each metric's numerator
and denominator plus TER/DER/FCER eligible-reference coverage; the scalar
columns retain their aligned-v1 semantics. It includes aggregate rows for all
data, clean, noisy-all, each SNR, each noise type, and each SNR/noise-type
combination.

After formal checkpoints exist, a limited diagnostic evaluation of those same
checkpoints is isolated explicitly:

```powershell
& $Python scripts\evaluate_all_lambdas.py `
  --config configs\phat\phat_pipeline.yaml `
  --output-dir outputs\paper_v2\smoke\predictions `
  --results-path outputs\paper_v2\smoke\lambda_ablation_results.csv `
  --limit 5 `
  --allow-partial `
  --device cuda
```

A checkpoint produced by smoke training has a different canonical training
contract. It can only be evaluated with that matching resolved smoke config and
remains ineligible for formal lambda selection.

## Select the best lambda

```powershell
& $Python scripts\select_best_lambda.py `
  --config configs\phat\phat_pipeline.yaml
```

The selector:

1. aggregates aligned-v1 TER and DER numerators/denominators at 0 and 5 dB;
2. compares aggregate WER/CER against ordinary LoRA (`lambda = 0`);
3. rejects candidates exceeding the configured absolute WER/CER increases;
4. rejects candidates whose low-SNR TER, DER, or FCER denominator coverage is
   below `0.98` of the ordinary-LoRA baseline;
5. chooses the lowest weighted low-SNR TER/DER score;
6. breaks ties by WER, CER, then smaller lambda.

All thresholds, SNR values, and weights are in `configs/phat/base.yaml`. The report is written to:

```text
outputs/paper_v2/dev_screen/best_lambda_report.md
```

The selector accepts only `evaluation_split=dev`, requires one shared manifest
SHA-256 and one locked evaluation-contract hash, and refuses
missing/mixed/test provenance before it reads the metrics. It also refuses a
complete report if any configured lambda or required low-SNR row is missing.
`--allow-partial` relaxes missing-lambda completeness only; it never bypasses
the dev/test, formal-training-scope, full-manifest, selected-row, checkpoint,
training/evaluation-contract, metric-version, or `0.98` TER/DER/FCER coverage
guards.

## End-to-end command

Do not run the formal end-to-end command or definitive training at the current
Gate-1.2 snapshot. The pipeline must fail closed until Gate 2 provides the
locked training-noise registry and locked derived noisy-dev screen, and Gate 3
locks the method candidates. Definitive runs start only after Gates 1-3.

After those gates are reviewed and their hashes/contracts are installed, the
complete command is:

```powershell
& $Python scripts\run_phat_pipeline.py `
  --config configs\phat\phat_pipeline.yaml `
  --device cuda
```

## Tests

The tests do not download a model or dataset. They use small tensors, a fake tokenizer, temporary CSV files, and in-memory selection rows.

```powershell
& $Python -m unittest discover -s tests\phat -v
```

Covered behavior:

- six Vietnamese tone classes and invalid-token handling
- token-level tone labels and ignored positions
- finite zero loss for all-ignored batches
- tone loss depends on logits and labels
- exact `lambda = 0` loss identity
- exact prediction schema
- dev-only/hash-bound best-lambda selection and WER/CER guards
- `0.98` low-SNR TER/DER/FCER denominator-coverage guards
- speaker-disjoint manifest locking and leakage audits
- dev evaluation, strict best-checkpoint selection, and safe checkpoint replacement
- loading all five configs
- deterministic Python, NumPy, and PyTorch seeds

## Historical pre-paper-v2 run

The following run is retained only for coursework traceability. It selected
lambda on the old 1,500-row benchmark derived from the official VIVOS test and
used checkpoints trained before the exact tone-alignment correction. It must
not be cited as paper-v2 evidence:

- all five lambdas (`0`, `0.05`, `0.1`, `0.3`, and `0.5`) completed three epochs and `2,187` optimizer steps
- every historical final checkpoint contains the adapter, processor, optimizer, scheduler, scaler, RNG state, resolved config, and trainer state; positive-lambda checkpoints also contain `tone_head.pt`
- every lambda was evaluated on all `1,500` benchmark rows: `300` clean and `300` for each SNR in `20`, `10`, `5`, and `0` dB
- all five prediction files contain the exact 11-column shared schema with no empty reference or hypothesis
- `outputs/phat/reports/lambda_ablation_results.csv` contains the real WER, CER, TER, DER, FCER, and SWDR aggregates
- the historical selector chose `lambda = 0.05`; this is not the locked paper-v2 choice

The active paper-v2 run has not yet been trained. It starts only after Gates 1
through 3 are complete. Historical runtime artifacts must never be copied into
`outputs/paper_v2/`.

## Troubleshooting

- `Output directory is not empty`: use a new path, `--resume` with a periodic checkpoint, or explicit `--overwrite`.
- `Missing PEFT adapter`: pass the paper-v2 experiment root containing `best/adapter/`, the `best/` checkpoint, or the adapter directory itself.
- `Resume config mismatch`: keep the same backbone, lambda, and LoRA rank as the saved checkpoint.
- CUDA out of memory: reduce the configured batch size from 16 and increase gradient accumulation so the effective batch remains 16. Record identical changes for all lambdas before comparing them.
- Missing audio path: regenerate/fix the manifest. The pipeline intentionally does not fabricate replacement data.
- Formal training/selection blocked before Gate 2: expected; build and verify the locked training-noise registry and source-dev-bound noisy-dev benchmark first. Never substitute the unlocked legacy noise manifest, clean dev, `vivos_test_locked.jsonl`, or the historical 1,500-row benchmark.
- NumPy/PyTorch DLL `Access is denied` under a restricted shell: run the same exact `slp` interpreter in a normal authorized terminal; do not switch to default Python or create a new environment.
