# Best Lambda Report

- Source results: `outputs/phat/reports/lambda_ablation_results.csv`
- Low-SNR priority: `0, 5 dB`
- Maximum absolute WER increase: `0.050000`
- Maximum absolute CER increase: `0.030000`

## Lambda comparison

| lambda | train type | clean WER | clean CER | guard WER | guard CER | delta WER | delta CER | low-SNR TER | low-SNR DER | score | eligible |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | ordinary_lora | 0.074461 | 0.032719 | 0.132462 | 0.072891 | +0.000000 | +0.000000 | 0.153854 | 0.067242 | 0.110548 | yes |
| 0.05 | tone_aware_lora | 0.073481 | 0.032796 | 0.128870 | 0.070175 | -0.003592 | -0.002716 | 0.144544 | 0.056159 | 0.100351 | yes |
| 0.1 | tone_aware_lora | 0.074788 | 0.033413 | 0.130372 | 0.071688 | -0.002090 | -0.001204 | 0.148953 | 0.061297 | 0.105125 | yes |
| 0.3 | tone_aware_lora | 0.080340 | 0.035265 | 0.139517 | 0.075978 | +0.007054 | +0.003087 | 0.159571 | 0.076822 | 0.118196 | yes |
| 0.5 | tone_aware_lora | 0.087851 | 0.038738 | 0.154801 | 0.088031 | +0.022338 | +0.015140 | 0.184847 | 0.100751 | 0.142799 | yes |

## Selected lambda

**lambda = 0.05**

This lambda passes the configured WER/CER degradation guard and has the best weighted low-SNR TER/DER score. Ties are resolved by lower WER, then lower CER, then smaller lambda.

- Low-SNR TER: `0.144544`
- Low-SNR DER: `0.056159`
- WER delta versus lambda 0: `-0.003592`
- CER delta versus lambda 0: `-0.002716`

## Limitations and warnings

- All five configured lambda values were present with the required aggregate rows.
