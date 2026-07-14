# Best Lambda Report

- Source results: `outputs\phat\smoke\reports\lambda_ablation_results.csv`
- Low-SNR priority: `0, 5 dB`
- Maximum absolute WER increase: `0.050000`
- Maximum absolute CER increase: `0.030000`

## Lambda comparison

| lambda | train type | clean WER | clean CER | guard WER | guard CER | delta WER | delta CER | low-SNR TER | low-SNR DER | score | eligible |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | ordinary_lora | 0.153846 | 0.052632 | 0.184615 | 0.084211 | +0.000000 | +0.000000 | 0.000000 | 0.000000 | 0.000000 | yes |
| 0.05 | tone_aware_lora | 0.153846 | 0.052632 | 0.184615 | 0.084211 | +0.000000 | +0.000000 | 0.000000 | 0.000000 | 0.000000 | yes |
| 0.1 | tone_aware_lora | 0.153846 | 0.052632 | 0.184615 | 0.084211 | +0.000000 | +0.000000 | 0.000000 | 0.000000 | 0.000000 | yes |
| 0.3 | tone_aware_lora | 0.153846 | 0.052632 | 0.184615 | 0.084211 | +0.000000 | +0.000000 | 0.000000 | 0.000000 | 0.000000 | yes |
| 0.5 | tone_aware_lora | 0.153846 | 0.052632 | 0.184615 | 0.084211 | +0.000000 | +0.000000 | 0.000000 | 0.000000 | 0.000000 | yes |

## Selected lambda

**lambda = 0**

This lambda passes the configured WER/CER degradation guard and has the best weighted low-SNR TER/DER score. Ties are resolved by lower WER, then lower CER, then smaller lambda.

- Low-SNR TER: `0.000000`
- Low-SNR DER: `0.000000`
- WER delta versus lambda 0: `+0.000000`
- CER delta versus lambda 0: `+0.000000`

## Limitations and warnings

- lambda=0: guard row has 5 samples, expected 1500
- lambda=0: clean row has 1 samples, expected 300
- lambda=0: SNR 5 row has 1 samples, expected 300
- lambda=0: SNR 0 row has 1 samples, expected 300
- lambda=0.05: guard row has 5 samples, expected 1500
- lambda=0.05: clean row has 1 samples, expected 300
- lambda=0.05: SNR 5 row has 1 samples, expected 300
- lambda=0.05: SNR 0 row has 1 samples, expected 300
- lambda=0.1: guard row has 5 samples, expected 1500
- lambda=0.1: clean row has 1 samples, expected 300
- lambda=0.1: SNR 5 row has 1 samples, expected 300
- lambda=0.1: SNR 0 row has 1 samples, expected 300
- lambda=0.3: guard row has 5 samples, expected 1500
- lambda=0.3: clean row has 1 samples, expected 300
- lambda=0.3: SNR 5 row has 1 samples, expected 300
- lambda=0.3: SNR 0 row has 1 samples, expected 300
- lambda=0.5: guard row has 5 samples, expected 1500
- lambda=0.5: clean row has 1 samples, expected 300
- lambda=0.5: SNR 5 row has 1 samples, expected 300
- lambda=0.5: SNR 0 row has 1 samples, expected 300
