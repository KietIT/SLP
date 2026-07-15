# Tone-label alignment audit

**Verdict: PASS**

This audit reconstructs each normalized transcript from word-level BPE pieces, using no leading space for the first word and one leading space for every later word. A transcript passes only when that sequence exactly equals one-shot tokenization of the complete transcript.

## Scope and provenance

- Audit version: `tone_alignment_v1`
- Inputs: `outputs/phat/predictions/pred_lora_ordinary_lambda0.csv`
- Tokenizer: `vinai/PhoWhisper-base`
- Tokenizer vocabulary SHA-256: `a148dd2e364cd5b8a3109bde5f2b12f7805ef49e26282acbab8f7c31f3aaefd6`
- Tone policy: `last_subtoken`
- Transcript deduplication: `normalized_text`
- Scope note: Current local corpus contains 1,500 VIVOS benchmark prediction rows; normalized-text deduplication audits unique references only, not full VIVOS train/dev manifests.

## Alignment checks

| Check | Value |
| --- | --- |
| Raw input rows | 1500 |
| Audited transcripts | 296 |
| Duplicate normalized transcripts skipped | 1204 |
| Exact BPE reconstructions | 296 |
| Exact reconstruction rate | 100.000000% |
| Alignment mismatches | 0 |
| BPE tokens | 5781 |
| Supervised tone-token targets | 3040 |
| Words audited | 3041 |
| Words with valid tone targets | 3040 |
| Words masked from tone loss | 1 |

## Six-tone distribution

| Tone | Tone ID | Word count |
| --- | --- | --- |
| ngang | 0 | 832 |
| sac | 1 | 709 |
| huyen | 2 | 576 |
| hoi | 3 | 292 |
| nga | 4 | 139 |
| nang | 5 | 492 |

## Word audit categories

| Status | Word count |
| --- | --- |
| all_caps_unmarked_or_acronym_candidate | 385 |
| marked_tone | 2208 |
| masked_no_vowel | 1 |
| unmarked_or_foreign_candidate | 447 |

`unmarked_or_foreign_candidate` and `all_caps_unmarked_or_acronym_candidate` are deliberately review buckets. Latin spelling alone cannot distinguish Vietnamese ngang-tone words from foreign words or acronyms without a lexicon or language-identification policy.

## Most frequent unmarked/foreign candidates

| Normalized word | Count |
| --- | --- |
| không | 38 |
| tôi | 27 |
| mươi | 20 |
| cho | 19 |
| như | 17 |
| trên | 16 |
| trong | 15 |
| công | 15 |
| ra | 14 |
| khi | 13 |
| em | 13 |
| anh | 13 |
| hai | 13 |
| đi | 12 |
| nhưng | 12 |
| con | 12 |
| cô | 11 |
| ba | 9 |
| lên | 8 |
| tư | 8 |

## Alignment errors

None.

## Interpretation

A PASS establishes exact word-piece/token-label alignment for the audited transcripts and tokenizer. It does not retroactively validate checkpoints trained with the previous alignment implementation; tone-aware checkpoints must be retrained after this fix.
