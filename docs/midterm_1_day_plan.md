# Midterm 1-Day Adjusted Plan

## Decision

Keep the current research direction. Do not pivot.

For midterm, the safe claim is: the group has built a reproducible Vietnamese noisy-ASR pipeline, preliminary zero-shot baseline evaluation, tone-label extraction, Vietnamese-specific metric prototypes, and a compute-aware plan for LoRA/tone-aware MTL training.

Do not claim that tone-aware MTL improves ASR unless there are real post-training results.

## One-Day Priority

1. Make the repo runnable end-to-end on a small subset.
2. Prepare clean and noisy VIVOS manifests with fixed seed and SNR 20/10/5/0.
3. Run zero-shot Whisper baseline first.
4. Run PhoWhisper zero-shot only if Colab/GPU time is available.
5. Produce tables: dataset stats, WER/CER by SNR, TER/DER prototype by SNR.
6. Prepare slide and short report around progress, results, risks, and next steps.

## Cut From the Original 3-Day Plan

- Full LoRA training.
- Full tone-aware MTL training.
- Lambda sensitivity experiments.
- Speech enhancement experiments.
- FLEURS/VietSuperSpeech integration.
- Large benchmark claims.

These remain after-midterm work.

## Schedule

| Time | Focus | Output |
|---|---|---|
| 08:00-10:00 | Setup environment, VIVOS/MUSAN, manifests | `data/manifests/vivos/*.jsonl`, `data/manifests/noise/*.jsonl` |
| 10:00-12:00 | Generate fixed noisy subset | `test_noisy.jsonl`, audio samples, `dataset_stats.csv` |
| 13:00-16:00 | Run zero-shot baseline | prediction CSV for clean/noisy subset |
| 16:00-18:00 | Score metrics and collect case studies | `metrics_clean.csv`, `metrics_noisy_by_snr.csv`, 2-3 examples |
| 18:00-21:00 | Build slide + report | 10-12 slides, 2-4 page report |
| 21:00-22:00 | Dry-run and freeze | no new features, only wording fixes |

## Minimum Commands

```bash
python scripts/make_vivos_manifest.py --vivos_root data/raw/vivos --out_dir data/manifests/vivos
python scripts/make_noise_manifest.py --noise_root data/raw/musan/noise --out data/manifests/noise/musan_noise.jsonl
python scripts/make_noisy_test.py --manifest data/manifests/vivos/test.jsonl --noise_manifest data/manifests/noise/musan_noise.jsonl --out_manifest data/manifests/vivos/test_noisy.jsonl --limit 50 --snrs 20 10 5 0 --seed 42
python scripts/dataset_stats.py data/manifests/vivos/test.jsonl data/manifests/vivos/test_noisy.jsonl --out outputs/dataset_stats.csv
python scripts/infer.py --manifest data/manifests/vivos/test.jsonl --model openai/whisper-base --out outputs/whisper_clean.csv --limit 30
python scripts/infer.py --manifest data/manifests/vivos/test_noisy.jsonl --model openai/whisper-base --out outputs/whisper_noisy.csv --limit 120
python scripts/score_predictions.py --pred outputs/whisper_clean.csv --out outputs/metrics_clean.csv
python scripts/score_predictions.py --pred outputs/whisper_noisy.csv --out outputs/metrics_noisy_by_snr.csv --group_by snr
```

## Slide Outline

1. Title, members, mentor.
2. Problem: Vietnamese ASR under noise is hard because tones matter.
3. Research gap: WER/CER hide tone and diacritic errors.
4. Research question and hypothesis.
5. Dataset and controlled noisy benchmark.
6. Pipeline: manifest -> noise mixing -> ASR inference -> metrics.
7. Architecture plan: PhoWhisper + LoRA + decoder-side tone head.
8. Current implementation status.
9. Preliminary WER/CER/TER/DER results by SNR.
10. Error examples under higher noise.
11. Risks and mitigation: GPU, dataset, tone labels, overclaiming.
12. Next steps: Colab LoRA, noisy LoRA vs tone-aware MTL.

## Report Structure

1. Project overview and research question.
2. Progress completed before midterm.
3. Dataset/noise benchmark setup.
4. Current source code and pipeline.
5. Preliminary results.
6. Risk assessment.
7. Post-midterm plan.

## Colab Note

Use local machine for preprocessing and smoke tests. Use Colab for PhoWhisper/LoRA training with batch size 1, FP16, gradient accumulation, max audio 15 seconds, and a small subset first. Training is a next-step deliverable, not required for the midterm claim.
