# Final benchmark licensing and attribution

This notice applies to the self-contained paper-v2 benchmark distributed at:

```text
data/derived/paper_v2/final_benchmark/
outputs/paper_v2/benchmark/final_benchmark_manifest.jsonl
```

The single JSONL manifest contains 2,300 rows: 460 clean VIVOS utterances and
the same 460 utterances mixed at 20, 10, 5, and 0 dB SNR. The audio is stored
as 2,300 separate WAV files; it is not embedded in the JSONL.

## VIVOS

The speech comes from **AILAB VIVOS Corpus, Vietnamese - Voices of Southern
Corpus for Speech Recognition, version 1.00 (December 2016)**, prepared by the
Artificial Intelligence Laboratory, VNUHCM - University of Science.

- Copyright: AILAB, 2016.
- License stated in the locally downloaded `VIVOS/COPYING`: Creative Commons
  Attribution-NonCommercial-ShareAlike 4.0 International
  ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).
- Official distribution used by this project:
  [VIVOS on Zenodo](https://zenodo.org/records/7068130).

The 460 clean files in this benchmark are byte-identical copies selected from
the locked VIVOS test complement. The 1,840 noisy files are modified versions:
the VIVOS speech is decoded/resampled to 16 kHz as needed, combined with a
deterministically selected and length-fitted MUSAN signal at a target SNR,
anti-clipped when required, and written as PCM-16 WAV. The manifest and lock
record the source, noise, parameters, and SHA-256 values.

The VIVOS terms require attribution, non-commercial use, an indication of
changes, and ShareAlike for adapted material. Publishing this benchmark in a
public GitHub repository is distribution. Do not use the bundle commercially
or remove this notice, the manifest, or the protocol lock.

## MUSAN

Noise signals come from **MUSAN: A Music, Speech, and Noise Corpus** by David
Snyder, Guoguo Chen, and Daniel Povey (2015).

The locally downloaded top-level MUSAN README says that every source
subdirectory carries its own `LICENSE`, connecting individual files to their
governing license and attribution. It also says that MUSAN content is under a
Creative Commons license or is considered public domain in the United States,
and that content forbidding commercial use was excluded. This is not one
blanket license: applicable terms and attribution remain source-specific.

The paper-v2 JSONL records `noise_id`, `noise_type`, `noise_path`, and
`noise_audio_sha256` for every noisy row. Use these fields to trace a selected
signal back to its authoritative MUSAN subdirectory `LICENSE`. Examples in the
local archive include CC BY, CC BY-SA, and public-domain material. This notice
does not replace those per-source entries or relax their terms.

Suggested MUSAN citation:

```bibtex
@misc{Snyder2015MUSAN,
  author = {David Snyder and Guoguo Chen and Daniel Povey},
  title = {{MUSAN}: {A} Music, Speech, and Noise Corpus},
  year = {2015},
  eprint = {1510.08484},
  archivePrefix = {arXiv}
}
```

Official corpus page: [OpenSLR 17](https://www.openslr.org/17/).

## Required distribution practice

Before publishing or mirroring the LFS bundle:

1. Keep this notice with the benchmark manifest, lock, audit, and handoff hash
   inventory.
2. Keep the VIVOS attribution and CC BY-NC-SA 4.0 notice visible.
3. Retain the applicable MUSAN per-source credits/license information for all
   selected `noise_id` values; the original MUSAN archive is authoritative.
4. Mark the noisy WAVs as modified mixtures and do not imply endorsement by
   VIVOS, AILAB, MUSAN, or the original MUSAN contributors.
5. Use a private Git LFS repository or a controlled external handoff until the
   project owner has reviewed the redistribution terms and attribution
   inventory. Git LFS does not alter the underlying license obligations.

This project note summarizes the license files shipped with the locally
downloaded corpora; it is not legal advice and does not grant rights beyond
the original licenses.
