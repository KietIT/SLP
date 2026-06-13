"""Vietnamese tone extraction and token-label alignment.

Tone classes:
0 ngang, 1 sac, 2 huyen, 3 hoi, 4 nga, 5 nang
ignore_index = -100
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

from .text_norm import normalize_vi_text

TONE_TO_ID: Dict[str, int] = {
    "ngang": 0,
    "sac": 1,
    "huyen": 2,
    "hoi": 3,
    "nga": 4,
    "nang": 5,
}
ID_TO_TONE = {v: k for k, v in TONE_TO_ID.items()}
COMBINING_TONE_TO_NAME = {
    "\u0301": "sac",     # acute
    "\u0300": "huyen",   # grave
    "\u0309": "hoi",     # hook above
    "\u0303": "nga",     # tilde
    "\u0323": "nang",    # dot below
}
VI_VOWELS = set("aăâeêioôơuưyAĂÂEÊIOÔƠUƯY")
WORD_RE = re.compile(r"^[\wÀ-ỹĐđ]+$", re.UNICODE)
IGNORE_INDEX = -100


def strip_tone_marks(token: str) -> str:
    chars = []
    for ch in unicodedata.normalize("NFD", token):
        if ch not in COMBINING_TONE_TO_NAME:
            chars.append(ch)
    return unicodedata.normalize("NFC", "".join(chars))


def has_vi_vowel(token: str) -> bool:
    base = strip_tone_marks(token)
    return any(ch in VI_VOWELS for ch in base)


def extract_tone(word: str, *, lexicon: Optional[set[str]] = None, accept_unmarked: bool = True) -> Tuple[int, bool]:
    """Return (tone_id, is_valid). Conservative handling for noisy conversational text.

    - Words with digits/symbols are ignored.
    - Uppercase acronyms such as AI, GPU, FPT are ignored.
    - Marked Vietnamese syllables are accepted.
    - Unmarked syllables are classed as 'ngang' only if accept_unmarked=True or word is in lexicon.
      For a stricter paper experiment, set accept_unmarked=False and pass a curated lexicon.
    """
    if not word:
        return IGNORE_INDEX, False
    raw = unicodedata.normalize("NFC", word.strip())
    if len(raw) > 1 and raw.isupper():
        return IGNORE_INDEX, False
    w = normalize_vi_text(word, remove_punct=True)
    if not w or any(ch.isdigit() for ch in w) or not WORD_RE.match(w):
        return IGNORE_INDEX, False
    if not has_vi_vowel(w):
        return IGNORE_INDEX, False
    nfd = unicodedata.normalize("NFD", w)
    marks = [COMBINING_TONE_TO_NAME[ch] for ch in nfd if ch in COMBINING_TONE_TO_NAME]
    if marks:
        return TONE_TO_ID[marks[-1]], True
    if accept_unmarked or (lexicon is not None and w in lexicon):
        return TONE_TO_ID["ngang"], True
    return IGNORE_INDEX, False


def syllable_tone_sequence(text: str, **kwargs) -> List[int]:
    return [extract_tone(w, **kwargs)[0] for w in normalize_vi_text(text).split()]


def build_token_tone_labels(text: str, tokenizer, policy: str = "last_subtoken", max_length: Optional[int] = None) -> List[int]:
    """Approximate syllable-to-token alignment for Whisper BPE.

    We tokenize syllable-by-syllable with a leading space to mimic word-boundary tokenization.
    This avoids needing character offsets, which Whisper tokenizer usually does not expose.
    Use the same tokenizer settings as training and validate alignment manually on samples.
    """
    assert policy in {"last_subtoken", "all_subtokens"}
    labels: List[int] = []
    for syl in normalize_vi_text(text).split():
        tone_id, ok = extract_tone(syl)
        pieces = tokenizer(" " + syl, add_special_tokens=False).input_ids
        if not pieces:
            continue
        if not ok:
            labels.extend([IGNORE_INDEX] * len(pieces))
        elif policy == "all_subtokens":
            labels.extend([tone_id] * len(pieces))
        else:
            labels.extend([IGNORE_INDEX] * (len(pieces) - 1) + [tone_id])
    if max_length is not None:
        labels = labels[:max_length]
    return labels


def tone_error_rate(ref_text: str, hyp_text: str) -> float:
    """Position-based TER over aligned syllable count; use edit-aligned version in final experiments."""
    ref = syllable_tone_sequence(ref_text)
    hyp = syllable_tone_sequence(hyp_text)
    n = min(len(ref), len(hyp))
    if n == 0:
        return 0.0 if len(ref) == len(hyp) else 1.0
    subs = sum(1 for i in range(n) if ref[i] != hyp[i])
    # count insertion/deletion as tone errors too
    return (subs + abs(len(ref) - len(hyp))) / max(len(ref), 1)
