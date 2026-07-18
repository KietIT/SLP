# Best Lambda Report

- Source results: `outputs/paper_v2/dev_screen/lambda_ablation_results.csv`
- Evaluation split: `dev`
- Evaluation manifest SHA-256: `e9bb6cdb4934d90f10b0782b7960a609b33476ffbfcd946a08568a46de8c853b`
- Evaluation contract SHA-256: `647101a414b86db0a8a9896821630904399b2fbb122035692f4d6e19c497a589`
- Low-SNR priority: `0, 5 dB`
- Maximum absolute WER increase: `0.050000`
- Maximum absolute CER increase: `0.030000`
- Minimum TER denominator ratio versus ordinary LoRA: `0.980000`
- Minimum DER denominator ratio versus ordinary LoRA: `0.980000`
- Minimum FCER denominator ratio versus ordinary LoRA: `0.980000`

## Lambda comparison

| lambda | train type | clean WER | clean CER | guard WER | guard CER | delta WER | delta CER | low-SNR TER | low-SNR DER | TER coverage | DER coverage | FCER coverage | score | eligible |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | ordinary_lora | 0.033629 | 0.019059 | 0.070043 | 0.039941 | +0.000000 | +0.000000 | 0.017200 | 0.004763 | 1.000000 | 1.000000 | 1.000000 | 0.010982 | no |
| 0.05 | tone_aware_lora | 0.034829 | 0.019634 | 0.071152 | 0.040502 | +0.001109 | +0.000560 | 0.016805 | 0.004962 | 0.998180 | 0.998525 | 1.000265 | 0.010884 | yes |
| 0.1 | tone_aware_lora | 0.035123 | 0.019259 | 0.072688 | 0.041476 | +0.002646 | +0.001535 | 0.016748 | 0.005145 | 0.997197 | 0.997537 | 0.999973 | 0.010946 | yes |
| 0.3 | tone_aware_lora | 0.038776 | 0.020773 | 0.075915 | 0.042297 | +0.005872 | +0.002356 | 0.017261 | 0.005343 | 0.993088 | 0.993718 | 1.000133 | 0.011302 | yes |
| 0.5 | tone_aware_lora | 0.041550 | 0.021886 | 0.078460 | 0.043430 | +0.008417 | +0.003489 | 0.017975 | 0.005443 | 0.991209 | 0.991697 | 1.000557 | 0.011709 | yes |

## Selected lambda

**lambda = 0.05**

This lambda passes the configured WER/CER degradation guard and has the best weighted low-SNR TER/DER score. Ties are resolved by lower WER, then lower CER, then smaller lambda.

- Low-SNR TER: `0.016805`
- Low-SNR DER: `0.004962`
- TER coverage ratio versus ordinary LoRA: `0.998180`
- DER coverage ratio versus ordinary LoRA: `0.998525`
- FCER coverage ratio versus ordinary LoRA: `1.000265`
- WER delta versus lambda 0: `+0.001109`
- CER delta versus lambda 0: `+0.000560`
- Locked control lambda: `0.1`
- Locked control strategy: `best_eligible_non_selected_tone_aware`

## Limitations and warnings

- All five configured lambda values were present with the required aggregate rows.
