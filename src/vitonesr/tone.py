"""Vietnamese tone extraction and token-label alignment.

Tone classes:
0 ngang, 1 sac, 2 huyen, 3 hoi, 4 nga, 5 nang
ignore_index = -100
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
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


class ToneAlignmentError(ValueError):
    """Raised when word-wise tone labels cannot be aligned exactly to ASR BPE IDs."""


@dataclass(frozen=True)
class ToneWordAlignment:
    """Audit metadata for one normalized whitespace-delimited word."""

    word_index: int
    source_word: str
    normalized_word: str
    piece_text: str
    token_start: int
    token_end: int
    token_ids: tuple[int, ...]
    tone_id: int
    is_valid: bool
    status: str
    all_caps_context: bool


@dataclass(frozen=True)
class TokenToneAlignment:
    """Exact word-piece IDs and their tone labels before Whisper special tokens."""

    normalized_text: str
    token_ids: tuple[int, ...]
    tone_labels: tuple[int, ...]
    words: tuple[ToneWordAlignment, ...]


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
    # In mixed-case transcripts, an ASCII all-caps token is a conservative acronym
    # signal. Uppercase Vietnamese words such as "TÔI" retain diacritics and must not
    # be discarded merely because their casing is uppercase.
    if len(raw) > 1 and raw.isascii() and raw.isalpha() and raw.isupper():
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


def _tone_status(
    source_word: str,
    tone_id: int,
    is_valid: bool,
    *,
    all_caps_context: bool = False,
) -> str:
    """Return a deterministic audit category without pretending to detect language ID."""

    raw = unicodedata.normalize("NFC", source_word.strip())
    normalized = normalize_vi_text(source_word, remove_punct=True)
    if (
        all_caps_context
        and len(raw) > 1
        and raw.isascii()
        and raw.isalpha()
        and raw.isupper()
    ):
        # In an ALL-CAPS corpus, casing cannot separate an ordinary unmarked
        # Vietnamese word from an acronym. Keep the target for compatibility with
        # normalized training text, but expose the ambiguity in the audit.
        return "all_caps_unmarked_or_acronym_candidate"
    if len(raw) > 1 and raw.isascii() and raw.isalpha() and raw.isupper():
        return "masked_acronym"
    if not normalized or any(ch.isdigit() for ch in normalized) or not WORD_RE.match(normalized):
        return "masked_digit_or_symbol"
    if not has_vi_vowel(normalized):
        return "masked_no_vowel"
    if not is_valid:
        return "masked_unmarked"
    has_tone_mark = any(
        ch in COMBINING_TONE_TO_NAME for ch in unicodedata.normalize("NFD", normalized)
    )
    if has_tone_mark:
        return "marked_tone"
    if tone_id == TONE_TO_ID["ngang"]:
        # Vietnamese ngang words and foreign words share the same Latin spellings in many
        # cases. Keep this as a manual-review bucket instead of claiming language ID.
        return "unmarked_or_foreign_candidate"
    return "valid_tone"


def _first_difference(left: List[int], right: List[int]) -> int:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return min(len(left), len(right))


def build_token_tone_alignment(
    text: str,
    tokenizer,
    policy: str = "last_subtoken",
    max_length: Optional[int] = None,
    *,
    source_text: Optional[str] = None,
) -> TokenToneAlignment:
    """Build an exact word-to-BPE tone alignment for Whisper tokenizers.

    Whisper byte-level BPE tokenizes the first word without a boundary space and later
    words with one. The concatenated word-piece IDs must exactly equal tokenizing the
    complete normalized transcript. Any mismatch is a validity error, never something to
    repair later with positional ``zip``.

    ``source_text`` preserves original casing for acronym auditing. VIVOS prompts are
    commonly formatted as whole-transcript uppercase, so casing is treated as formatting
    when a multi-word source transcript is entirely uppercase; such words remain visible
    through ``all_caps_context`` in the audit metadata.
    """

    if policy not in {"last_subtoken", "all_subtokens"}:
        raise ValueError(f"Unsupported tone label policy: {policy}")
    if max_length is not None and max_length < 0:
        raise ValueError("max_length must be non-negative")

    normalized_text = normalize_vi_text(text)
    normalized_words = normalized_text.split()
    if not normalized_words:
        raise ToneAlignmentError("Transcript is empty after Vietnamese text normalization")

    source_value = text if source_text is None else source_text
    normalized_source = normalize_vi_text(source_value, lowercase=False)
    source_words = normalized_source.split()
    if len(source_words) != len(normalized_words):
        raise ToneAlignmentError(
            "Source and normalized transcript have different word counts: "
            f"source={len(source_words)}, normalized={len(normalized_words)}"
        )

    all_caps_context = len(source_words) > 1 and normalized_source.isupper()
    token_ids: List[int] = []
    tone_labels: List[int] = []
    word_alignments: List[ToneWordAlignment] = []

    for word_index, (source_word, normalized_word) in enumerate(
        zip(source_words, normalized_words)
    ):
        piece_text = normalized_word if word_index == 0 else " " + normalized_word
        pieces = list(tokenizer(piece_text, add_special_tokens=False).input_ids)
        if not pieces:
            raise ToneAlignmentError(
                f"Tokenizer emitted no IDs for word {word_index}: {normalized_word!r}"
            )

        # Do not mask every word in corpora whose transcription convention is ALL CAPS.
        tone_source = normalized_word if all_caps_context else source_word
        tone_id, is_valid = extract_tone(tone_source)
        status = _tone_status(
            source_word,
            tone_id,
            is_valid,
            all_caps_context=all_caps_context,
        )
        start = len(token_ids)
        token_ids.extend(pieces)
        if not is_valid:
            word_labels = [IGNORE_INDEX] * len(pieces)
        elif policy == "all_subtokens":
            word_labels = [tone_id] * len(pieces)
        else:
            word_labels = [IGNORE_INDEX] * (len(pieces) - 1) + [tone_id]
        tone_labels.extend(word_labels)
        word_alignments.append(
            ToneWordAlignment(
                word_index=word_index,
                source_word=source_word,
                normalized_word=normalized_word,
                piece_text=piece_text,
                token_start=start,
                token_end=len(token_ids),
                token_ids=tuple(pieces),
                tone_id=tone_id,
                is_valid=is_valid,
                status=status,
                all_caps_context=all_caps_context,
            )
        )

    full_text_ids = list(tokenizer(normalized_text, add_special_tokens=False).input_ids)
    if token_ids != full_text_ids:
        mismatch_index = _first_difference(token_ids, full_text_ids)
        piece_id = token_ids[mismatch_index] if mismatch_index < len(token_ids) else None
        full_id = full_text_ids[mismatch_index] if mismatch_index < len(full_text_ids) else None
        raise ToneAlignmentError(
            "Word-wise BPE IDs do not match full-transcript IDs: "
            f"index={mismatch_index}, piece_id={piece_id}, full_id={full_id}, "
            f"piece_length={len(token_ids)}, full_length={len(full_text_ids)}"
        )
    if len(token_ids) != len(tone_labels):
        raise ToneAlignmentError(
            f"Token/tone length mismatch: token_ids={len(token_ids)}, labels={len(tone_labels)}"
        )

    if max_length is not None:
        token_ids = token_ids[:max_length]
        tone_labels = tone_labels[:max_length]
        clipped_words: List[ToneWordAlignment] = []
        for word in word_alignments:
            if word.token_start >= max_length:
                break
            clipped_end = min(word.token_end, max_length)
            kept_piece_count = clipped_end - word.token_start
            clipped_words.append(
                replace(
                    word,
                    token_end=clipped_end,
                    token_ids=word.token_ids[:kept_piece_count],
                )
            )
        word_alignments = clipped_words

    return TokenToneAlignment(
        normalized_text=normalized_text,
        token_ids=tuple(token_ids),
        tone_labels=tuple(tone_labels),
        words=tuple(word_alignments),
    )


def build_token_tone_labels(
    text: str,
    tokenizer,
    policy: str = "last_subtoken",
    max_length: Optional[int] = None,
    *,
    source_text: Optional[str] = None,
) -> List[int]:
    """Backward-compatible label-only wrapper around exact BPE alignment."""

    alignment = build_token_tone_alignment(
        text,
        tokenizer,
        policy=policy,
        max_length=max_length,
        source_text=source_text,
    )
    return list(alignment.tone_labels)


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
