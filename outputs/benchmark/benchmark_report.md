# Robust Benchmark Report

## Configuration
- VIVOS manifest: `data/manifests/vivos/test.jsonl`
- Noise manifest: `data/manifests/noise/musan_noise_typed.jsonl`
- Seed: `42`
- Pool size: `500`
- Eval size: `300`
- SNR levels: `20, 10, 5, 0`
- Sample rate: `16000`

## Counts
| condition | snr | count |
| --- | --- | --- |
| clean | clean | 300 |
| noisy | 20 | 300 |
| noisy | 10 | 300 |
| noisy | 5 | 300 |
| noisy | 0 | 300 |

## Noise Type Distribution
| snr | noise_type | count |
| --- | --- | --- |
| 0 | music | 93 |
| 0 | noise | 145 |
| 0 | speech | 62 |
| 10 | music | 97 |
| 10 | noise | 139 |
| 10 | speech | 64 |
| 20 | music | 96 |
| 20 | noise | 145 |
| 20 | speech | 59 |
| 5 | music | 108 |
| 5 | noise | 127 |
| 5 | speech | 65 |
| clean | clean | 300 |

## Duration Summary
| condition | snr | total_hours | avg_seconds | min_seconds | max_seconds |
| --- | --- | --- | --- | --- | --- |
| clean | clean | 0.290630 | 3.488 | 1.375 | 6.781 |
| noisy | 0 | 0.290630 | 3.488 | 1.375 | 6.781 |
| noisy | 10 | 0.290630 | 3.488 | 1.375 | 6.781 |
| noisy | 20 | 0.290630 | 3.488 | 1.375 | 6.781 |
| noisy | 5 | 0.290630 | 3.488 | 1.375 | 6.781 |

## Manifest Schema
- Pool columns: `source_utt_id, dataset, split, clean_path, transcript, duration, seed, pool_rank`
- Benchmark columns: `utt_id, dataset, split, condition, clean_path, noisy_path, audio_path, snr, noise_type, noise_path, transcript, duration, seed, source_utt_id`

## Reproducibility
Rows are selected by sorting VIVOS utterance IDs, shuffling with the master seed, taking the pool, then taking the eval subset. Each noisy sample uses a stable SHA-256 seed derived from the master seed, source utterance ID, and SNR condition.

## Validation Status
PASS

## Notes
- Babble was not generated for this benchmark.

## Output Files
- Pool manifest: `outputs/benchmark/benchmark_pool_manifest.csv` (500 rows)
- Benchmark manifest: `outputs/benchmark/benchmark_manifest.csv` (1500 rows)
- Noisy audio directory: `data/noisy_eval`
