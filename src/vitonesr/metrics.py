from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

try:
    import jiwer
except ModuleNotFoundError:
    jiwer = None

from .text_norm import normalize_vi_text
from .tone import extract_tone, strip_tone_marks


def _edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = cur
    return prev[-1]


def wer(refs: List[str], hyps: List[str]) -> float:
    norm_refs = [normalize_vi_text(x) for x in refs]
    norm_hyps = [normalize_vi_text(x) for x in hyps]
    if jiwer is not None:
        return jiwer.wer(norm_refs, norm_hyps)
    ref_words = [word for sent in norm_refs for word in sent.split()]
    hyp_words = [word for sent in norm_hyps for word in sent.split()]
    return _edit_distance(ref_words, hyp_words) / max(len(ref_words), 1)


def cer(refs: List[str], hyps: List[str]) -> float:
    norm_refs = [normalize_vi_text(x) for x in refs]
    norm_hyps = [normalize_vi_text(x) for x in hyps]
    if jiwer is not None:
        return jiwer.cer(norm_refs, norm_hyps)
    ref_chars = list(" ".join(norm_refs))
    hyp_chars = list(" ".join(norm_hyps))
    return _edit_distance(ref_chars, hyp_chars) / max(len(ref_chars), 1)


def simple_tone_error_rate(refs: List[str], hyps: List[str]) -> float:
    # A simple first-pass metric. For the paper, replace with edit-aligned syllable TER.
    total = 0
    err = 0
    for r, h in zip(refs, hyps):
        r_syl = normalize_vi_text(r).split()
        h_syl = normalize_vi_text(h).split()
        n = min(len(r_syl), len(h_syl))
        for i in range(n):
            rt, rok = extract_tone(r_syl[i])
            ht, hok = extract_tone(h_syl[i])
            if rok:
                total += 1
                if (not hok) or rt != ht:
                    err += 1
        err += abs(len(r_syl) - len(h_syl))
        total += max(0, len(r_syl) - n)
    return err / max(total, 1)


def diacritic_error_rate(refs: List[str], hyps: List[str]) -> float:
    total = 0
    err = 0
    for r, h in zip(refs, hyps):
        r_syl = normalize_vi_text(r).split()
        h_syl = normalize_vi_text(h).split()
        n = min(len(r_syl), len(h_syl))
        for i in range(n):
            if strip_tone_marks(r_syl[i]) == strip_tone_marks(h_syl[i]):
                total += 1
                if r_syl[i] != h_syl[i]:
                    err += 1
        err += abs(len(r_syl) - len(h_syl))
        total += max(0, len(r_syl) - n)
    return err / max(total, 1)


def compute_all(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    return {
        "wer": wer(refs, hyps),
        "cer": cer(refs, hyps),
        "ter_simple": simple_tone_error_rate(refs, hyps),
        "der_simple": diacritic_error_rate(refs, hyps),
    }
