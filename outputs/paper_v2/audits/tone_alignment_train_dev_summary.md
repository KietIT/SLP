# Tone-label alignment audit

**Verdict: PASS**

This audit reconstructs each normalized transcript from word-level BPE pieces, using no leading space for the first word and one leading space for every later word. A transcript passes only when that sequence exactly equals one-shot tokenization of the complete transcript.

## Scope and provenance

- Audit version: `tone_alignment_v2`
- Inputs: `data/manifests/paper_v2/vivos_train.jsonl`, `data/manifests/paper_v2/vivos_dev.jsonl`
- Tokenizer: `vinai/PhoWhisper-base`
- Tokenizer immutable revision: `7ebdb9e88f5cc5271fb88f4d642c82ff9388650e`
- Tokenizer vocabulary SHA-256: `a148dd2e364cd5b8a3109bde5f2b12f7805ef49e26282acbab8f7c31f3aaefd6`
- Tone policy: `last_subtoken`
- Transcript deduplication: `disabled`
- Scope note: Full paper-v2 VIVOS train and speaker-disjoint dev manifests; every row audited without transcript deduplication.

## Alignment checks

| Check | Value |
| --- | --- |
| Raw input rows | 11660 |
| Audited transcripts | 11660 |
| Duplicate normalized transcripts skipped | 0 |
| Exact BPE reconstructions | 11660 |
| Exact reconstruction rate | 100.000000% |
| Alignment mismatches | 0 |
| BPE tokens | 291046 |
| Supervised tone-token targets | 154241 |
| Words audited | 154268 |
| Words with valid tone targets | 154241 |
| Words masked from tone loss | 27 |

## Six-tone distribution

| Tone | Tone ID | Word count |
| --- | --- | --- |
| ngang | 0 | 41703 |
| sac | 1 | 35671 |
| huyen | 2 | 29923 |
| hoi | 3 | 14803 |
| nga | 4 | 7905 |
| nang | 5 | 24236 |

## Word audit categories

| Status | Word count |
| --- | --- |
| all_caps_unmarked_or_acronym_candidate | 19838 |
| marked_tone | 112538 |
| masked_digit_or_symbol | 1 |
| masked_no_vowel | 26 |
| unmarked_or_foreign_candidate | 21865 |

`unmarked_or_foreign_candidate` and `all_caps_unmarked_or_acronym_candidate` are deliberately review buckets. Latin spelling alone cannot distinguish Vietnamese ngang-tone words from foreign words or acronyms without a lexicon or language-identification policy.

## Most frequent unmarked/foreign candidates

| Normalized word | Count |
| --- | --- |
| không | 1743 |
| tôi | 1222 |
| cho | 1120 |
| trong | 941 |
| mươi | 818 |
| như | 753 |
| ra | 694 |
| khi | 682 |
| anh | 674 |
| con | 653 |
| đi | 617 |
| công | 616 |
| hai | 608 |
| nhưng | 595 |
| ông | 517 |
| em | 487 |
| ba | 454 |
| năm | 416 |
| trên | 416 |
| ta | 380 |

## Alignment errors

None.

## Interpretation

A PASS establishes exact word-piece/token-label alignment for the audited transcripts and tokenizer. It does not retroactively validate checkpoints trained with the previous alignment implementation; tone-aware checkpoints must be retrained after this fix.
