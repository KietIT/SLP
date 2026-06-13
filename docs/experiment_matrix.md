# Experiment matrix

| ID | Train data | Test data | Noise train | Noise test | Model | LoRA | Tone MTL | Enhancement | Purpose |
|---|---|---|---|---|---|---|---|---|---|
| E0 | none | clean/noisy | no | yes | Whisper base | no | no | no | multilingual zero-shot |
| E1 | none | clean/noisy | no | yes | PhoWhisper base | no | no | no | Vietnamese baseline |
| E2 | clean | clean/noisy | no | yes | PhoWhisper base | yes | no | no | clean LoRA |
| E3 | clean+on-the-fly noise | clean/noisy | yes | yes | PhoWhisper base | yes | no | no | robust LoRA |
| E4 | clean+on-the-fly noise | clean/noisy | yes | yes | PhoWhisper base | yes | yes | no | proposed |
| E5 | clean+on-the-fly noise | enhanced noisy | yes | yes | best model | yes | optional | yes | enhancement trade-off |

Primary result should compare E3 vs E4 under the same data, seed, LoRA config, and compute budget.
