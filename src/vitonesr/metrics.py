from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

try:
    import jiwer
except ModuleNotFoundError:
    jiwer = None

from .text_norm import normalize_vi_text
from .tone import extract_tone, strip_tone_marks


FINAL_CONSONANTS = ("ch", "ng", "nh", "c", "m", "n", "p", "t")


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


def _align_sequences(ref: Sequence[str], hyp: Sequence[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    rows = len(ref) + 1
    cols = len(hyp) + 1
    dp = [[0] * cols for _ in range(rows)]
    back = [[""] * cols for _ in range(rows)]

    for i in range(1, rows):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(1, cols):
        dp[0][j] = j
        back[0][j] = "insert"

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            candidates = [
                (dp[i - 1][j - 1] + cost, "match" if cost == 0 else "replace"),
                (dp[i - 1][j] + 1, "delete"),
                (dp[i][j - 1] + 1, "insert"),
            ]
            dp[i][j], back[i][j] = min(candidates, key=lambda x: x[0])

    aligned: List[Tuple[Optional[str], Optional[str]]] = []
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        op = back[i][j]
        if op in {"match", "replace"}:
            aligned.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif op == "delete":
            aligned.append((ref[i - 1], None))
            i -= 1
        else:
            aligned.append((None, hyp[j - 1]))
            j -= 1
    aligned.reverse()
    return aligned


def _final_consonant(word: str) -> Optional[str]:
    base = strip_tone_marks(word)
    for suffix in FINAL_CONSONANTS:
        if base.endswith(suffix):
            return suffix
    return None


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


def final_consonant_error_rate(refs: List[str], hyps: List[str]) -> float:
    total = 0
    err = 0
    for r, h in zip(refs, hyps):
        r_words = normalize_vi_text(r).split()
        h_words = normalize_vi_text(h).split()
        for r_word, h_word in _align_sequences(r_words, h_words):
            if r_word is None:
                continue
            r_final = _final_consonant(r_word)
            if r_final is None:
                continue
            total += 1
            if h_word is None or _final_consonant(h_word) != r_final:
                err += 1
    return err / max(total, 1)


def short_word_deletion_rate(refs: List[str], hyps: List[str]) -> float:
    total = 0
    deleted = 0
    for r, h in zip(refs, hyps):
        r_words = normalize_vi_text(r).split()
        h_words = normalize_vi_text(h).split()
        for r_word, h_word in _align_sequences(r_words, h_words):
            if r_word is None or len(r_word) > 2:
                continue
            total += 1
            if h_word is None:
                deleted += 1
    return deleted / max(total, 1)


def compute_all(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    return {
        "wer": wer(refs, hyps),
        "cer": cer(refs, hyps),
        "ter_simple": simple_tone_error_rate(refs, hyps),
        "der_simple": diacritic_error_rate(refs, hyps),
        "fcer_simple": final_consonant_error_rate(refs, hyps),
        "swdr_simple": short_word_deletion_rate(refs, hyps),
    }
