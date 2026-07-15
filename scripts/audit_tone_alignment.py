from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitonesr.prediction import atomic_write_csv  # noqa: E402
from src.vitonesr.text_norm import normalize_vi_text  # noqa: E402
from src.vitonesr.tone import (  # noqa: E402
    ID_TO_TONE,
    IGNORE_INDEX,
    TONE_TO_ID,
    ToneAlignmentError,
    build_token_tone_alignment,
)


AUDIT_VERSION = "tone_alignment_v1"
DEFAULT_INPUT = "outputs/phat/predictions/pred_lora_ordinary_lambda0.csv"
DEFAULT_OUTPUT_CSV = "outputs/paper_v2/audits/tone_alignment_audit.csv"
DEFAULT_OUTPUT_SUMMARY = "outputs/paper_v2/audits/tone_alignment_summary.md"
TEXT_COLUMN_CANDIDATES = ("transcript", "ref", "text")
ID_COLUMN_CANDIDATES = ("source_utt_id", "utt_id", "id")
AUDIT_COLUMNS = [
    "audit_version",
    "input_file",
    "input_sha256",
    "input_row",
    "source_id",
    "text_sha256",
    "normalized_text",
    "word_index",
    "source_word",
    "normalized_word",
    "piece_text",
    "token_start",
    "token_end",
    "token_count",
    "token_ids",
    "tone_id",
    "tone_name",
    "is_valid",
    "status",
    "all_caps_context",
    "exact_bpe_match",
    "alignment_error",
]


class ToneAuditError(ValueError):
    """Raised when an audit input or output contract is invalid."""


@dataclass(frozen=True)
class TranscriptRecord:
    input_file: str
    input_sha256: str
    input_row: int
    source_id: str
    text: str


@dataclass(frozen=True)
class AuditResult:
    rows: tuple[dict[str, object], ...]
    input_rows: int
    selected_transcripts: int
    duplicate_transcripts_skipped: int
    exact_transcripts: int
    mismatched_transcripts: int
    token_count: int
    targeted_token_count: int
    status_counts: Counter[str]
    tone_counts: Counter[str]
    candidate_counts: Counter[str]
    errors: tuple[tuple[str, str], ...]

    @property
    def passed(self) -> bool:
        return (
            self.selected_transcripts > 0
            and self.exact_transcripts == self.selected_transcripts
            and self.mismatched_transcripts == 0
            and not self.errors
        )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _select_column(
    columns: Sequence[str],
    explicit: str | None,
    candidates: Sequence[str],
    *,
    label: str,
    path: Path,
) -> str:
    if explicit is not None:
        if explicit not in columns:
            raise ToneAuditError(f"{path}: {label} column {explicit!r} does not exist")
        return explicit
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ToneAuditError(
        f"{path}: cannot infer {label} column; expected one of {list(candidates)}"
    )


def _records_from_csv(
    path: Path,
    *,
    text_column: str | None,
    id_column: str | None,
) -> list[TranscriptRecord]:
    try:
        raw = path.read_bytes()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ToneAuditError(f"{path}: CSV has no header")
            columns = list(reader.fieldnames)
            if len(columns) != len(set(columns)):
                raise ToneAuditError(f"{path}: CSV has duplicate header columns")
            selected_text = _select_column(
                columns,
                text_column,
                TEXT_COLUMN_CANDIDATES,
                label="text",
                path=path,
            )
            selected_id = _select_column(
                columns,
                id_column,
                ID_COLUMN_CANDIDATES,
                label="ID",
                path=path,
            )
            digest = _sha256_bytes(raw)
            display = _display_path(path)
            records: list[TranscriptRecord] = []
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ToneAuditError(f"{path}: malformed CSV row {row_number}")
                text = str(row[selected_text])
                source_id = str(row[selected_id])
                if not source_id.strip():
                    raise ToneAuditError(f"{path}: blank source ID at row {row_number}")
                records.append(
                    TranscriptRecord(display, digest, row_number, source_id, text)
                )
    except UnicodeDecodeError as exc:
        raise ToneAuditError(f"{path}: input must be UTF-8") from exc
    if not records:
        raise ToneAuditError(f"{path}: input contains no data rows")
    return records


def _records_from_jsonl(
    path: Path,
    *,
    text_column: str | None,
    id_column: str | None,
) -> list[TranscriptRecord]:
    try:
        raw = path.read_bytes()
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise ToneAuditError(f"{path}: input must be UTF-8") from exc
    parsed: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ToneAuditError(f"{path}: invalid JSON at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ToneAuditError(f"{path}: line {line_number} must be a JSON object")
        parsed.append((line_number, value))
    if not parsed:
        raise ToneAuditError(f"{path}: input contains no data rows")
    columns = sorted({str(key) for _, row in parsed for key in row})
    selected_text = _select_column(
        columns,
        text_column,
        TEXT_COLUMN_CANDIDATES,
        label="text",
        path=path,
    )
    selected_id = _select_column(
        columns,
        id_column,
        ID_COLUMN_CANDIDATES,
        label="ID",
        path=path,
    )
    digest = _sha256_bytes(raw)
    display = _display_path(path)
    records = []
    for line_number, row in parsed:
        if selected_text not in row or selected_id not in row:
            raise ToneAuditError(
                f"{path}: line {line_number} is missing {selected_text!r} or {selected_id!r}"
            )
        source_id = str(row[selected_id])
        if not source_id.strip():
            raise ToneAuditError(f"{path}: blank source ID at line {line_number}")
        records.append(
            TranscriptRecord(display, digest, line_number, source_id, str(row[selected_text]))
        )
    return records


def load_records(
    input_paths: Sequence[str | Path],
    *,
    text_column: str | None = None,
    id_column: str | None = None,
) -> list[TranscriptRecord]:
    records: list[TranscriptRecord] = []
    seen_paths: set[Path] = set()
    for value in input_paths:
        path = Path(value)
        if not path.is_file():
            raise ToneAuditError(f"input file does not exist: {path}")
        resolved = path.resolve()
        if resolved in seen_paths:
            raise ToneAuditError(f"duplicate input path: {path}")
        seen_paths.add(resolved)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            records.extend(
                _records_from_csv(path, text_column=text_column, id_column=id_column)
            )
        elif suffix in {".jsonl", ".json"}:
            records.extend(
                _records_from_jsonl(path, text_column=text_column, id_column=id_column)
            )
        else:
            raise ToneAuditError(f"unsupported input format for {path}; use CSV or JSONL")
    if not records:
        raise ToneAuditError("at least one input file is required")
    return records


def _base_audit_row(record: TranscriptRecord, normalized_text: str) -> dict[str, object]:
    return {
        "audit_version": AUDIT_VERSION,
        "input_file": record.input_file,
        "input_sha256": record.input_sha256,
        "input_row": record.input_row,
        "source_id": record.source_id,
        "text_sha256": _sha256_text(normalized_text),
        "normalized_text": normalized_text,
    }


def audit_records(
    records: Sequence[TranscriptRecord],
    tokenizer: Any,
    *,
    policy: str = "last_subtoken",
    deduplicate_text: bool = True,
) -> AuditResult:
    selected: list[TranscriptRecord] = []
    seen_texts: set[str] = set()
    duplicate_count = 0
    for record in records:
        normalized_text = normalize_vi_text(record.text)
        if deduplicate_text and normalized_text in seen_texts:
            duplicate_count += 1
            continue
        seen_texts.add(normalized_text)
        selected.append(record)

    rows: list[dict[str, object]] = []
    exact_transcripts = 0
    mismatched_transcripts = 0
    token_count = 0
    targeted_token_count = 0
    status_counts: Counter[str] = Counter()
    tone_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    errors: list[tuple[str, str]] = []

    for record in selected:
        normalized_text = normalize_vi_text(record.text)
        base = _base_audit_row(record, normalized_text)
        try:
            alignment = build_token_tone_alignment(
                record.text,
                tokenizer,
                policy=policy,
                source_text=record.text,
            )
            full_ids = tuple(
                tokenizer(alignment.normalized_text, add_special_tokens=False).input_ids
            )
            exact = alignment.token_ids == full_ids
            if not exact:
                raise ToneAlignmentError(
                    "independent full-transcript tokenization check did not match"
                )
            if len(alignment.token_ids) != len(alignment.tone_labels):
                raise ToneAlignmentError("token and tone-label counts differ after alignment")
            exact_transcripts += 1
            token_count += len(alignment.token_ids)
            targeted_token_count += sum(
                label != IGNORE_INDEX for label in alignment.tone_labels
            )
            for word in alignment.words:
                status_counts[word.status] += 1
                if word.is_valid:
                    tone_counts[ID_TO_TONE[word.tone_id]] += 1
                if word.status in {
                    "unmarked_or_foreign_candidate",
                    "all_caps_unmarked_or_acronym_candidate",
                }:
                    candidate_counts[word.normalized_word] += 1
                rows.append(
                    {
                        **base,
                        "word_index": word.word_index,
                        "source_word": word.source_word,
                        "normalized_word": word.normalized_word,
                        "piece_text": word.piece_text,
                        "token_start": word.token_start,
                        "token_end": word.token_end,
                        "token_count": len(word.token_ids),
                        "token_ids": json.dumps(
                            list(word.token_ids), ensure_ascii=False, separators=(",", ":")
                        ),
                        "tone_id": word.tone_id,
                        "tone_name": ID_TO_TONE.get(word.tone_id, "ignored"),
                        "is_valid": str(word.is_valid).lower(),
                        "status": word.status,
                        "all_caps_context": str(word.all_caps_context).lower(),
                        "exact_bpe_match": "true",
                        "alignment_error": "",
                    }
                )
        except (ToneAlignmentError, ValueError, KeyError, AssertionError) as exc:
            mismatched_transcripts += 1
            message = str(exc)
            errors.append((record.source_id, message))
            rows.append(
                {
                    **base,
                    "word_index": "",
                    "source_word": "",
                    "normalized_word": "",
                    "piece_text": "",
                    "token_start": "",
                    "token_end": "",
                    "token_count": "",
                    "token_ids": "",
                    "tone_id": "",
                    "tone_name": "",
                    "is_valid": "false",
                    "status": "alignment_error",
                    "all_caps_context": "",
                    "exact_bpe_match": "false",
                    "alignment_error": message,
                }
            )

    return AuditResult(
        rows=tuple(rows),
        input_rows=len(records),
        selected_transcripts=len(selected),
        duplicate_transcripts_skipped=duplicate_count,
        exact_transcripts=exact_transcripts,
        mismatched_transcripts=mismatched_transcripts,
        token_count=token_count,
        targeted_token_count=targeted_token_count,
        status_counts=status_counts,
        tone_counts=tone_counts,
        candidate_counts=candidate_counts,
        errors=tuple(errors),
    )


def tokenizer_vocab_sha256(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()
    payload = json.dumps(
        sorted((str(token), int(token_id)) for token, token_id in vocab.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _sha256_text(payload)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    def escape(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    output = [
        "| " + " | ".join(escape(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend(
        "| " + " | ".join(escape(value) for value in row) + " |" for row in rows
    )
    return output


def build_summary(
    result: AuditResult,
    *,
    input_paths: Sequence[str | Path],
    tokenizer_source: str,
    vocab_sha256: str,
    policy: str,
    deduplicate_text: bool,
    scope_note: str = "",
) -> str:
    valid_words = sum(result.tone_counts.values())
    total_words = sum(result.status_counts.values())
    masked_words = total_words - valid_words
    exact_rate = (
        result.exact_transcripts / result.selected_transcripts
        if result.selected_transcripts
        else 0.0
    )
    lines = [
        "# Tone-label alignment audit",
        "",
        f"**Verdict: {'PASS' if result.passed else 'FAIL'}**",
        "",
        "This audit reconstructs each normalized transcript from word-level BPE pieces, "
        "using no leading space for the first word and one leading space for every later "
        "word. A transcript passes only when that sequence exactly equals one-shot "
        "tokenization of the complete transcript.",
        "",
        "## Scope and provenance",
        "",
        f"- Audit version: `{AUDIT_VERSION}`",
        f"- Inputs: {', '.join(f'`{_display_path(Path(path))}`' for path in input_paths)}",
        f"- Tokenizer: `{tokenizer_source}`",
        f"- Tokenizer vocabulary SHA-256: `{vocab_sha256}`",
        f"- Tone policy: `{policy}`",
        f"- Transcript deduplication: `{'normalized_text' if deduplicate_text else 'disabled'}`",
    ]
    if scope_note:
        lines.extend([f"- Scope note: {scope_note}"])
    lines.extend(
        [
            "",
            "## Alignment checks",
            "",
            *_markdown_table(
                ["Check", "Value"],
                [
                    ["Raw input rows", result.input_rows],
                    ["Audited transcripts", result.selected_transcripts],
                    ["Duplicate normalized transcripts skipped", result.duplicate_transcripts_skipped],
                    ["Exact BPE reconstructions", result.exact_transcripts],
                    ["Exact reconstruction rate", f"{exact_rate:.6%}"],
                    ["Alignment mismatches", result.mismatched_transcripts],
                    ["BPE tokens", result.token_count],
                    ["Supervised tone-token targets", result.targeted_token_count],
                    ["Words audited", total_words],
                    ["Words with valid tone targets", valid_words],
                    ["Words masked from tone loss", masked_words],
                ],
            ),
            "",
            "## Six-tone distribution",
            "",
            *_markdown_table(
                ["Tone", "Tone ID", "Word count"],
                [
                    [tone_name, TONE_TO_ID[tone_name], result.tone_counts[tone_name]]
                    for tone_name in ("ngang", "sac", "huyen", "hoi", "nga", "nang")
                ],
            ),
            "",
            "## Word audit categories",
            "",
            *_markdown_table(
                ["Status", "Word count"],
                [[status, count] for status, count in sorted(result.status_counts.items())],
            ),
            "",
            "`unmarked_or_foreign_candidate` and "
            "`all_caps_unmarked_or_acronym_candidate` are deliberately review buckets. "
            "Latin spelling alone cannot distinguish Vietnamese ngang-tone words from "
            "foreign words or acronyms without a lexicon or language-identification policy.",
            "",
            "## Most frequent unmarked/foreign candidates",
            "",
            *_markdown_table(
                ["Normalized word", "Count"],
                result.candidate_counts.most_common(20) or [["(none)", 0]],
            ),
            "",
            "## Alignment errors",
            "",
        ]
    )
    if result.errors:
        lines.extend(
            _markdown_table(
                ["Source ID", "Error"],
                [[source_id, error] for source_id, error in result.errors[:20]],
            )
        )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A PASS establishes exact word-piece/token-label alignment for the audited "
            "transcripts and tokenizer. It does not retroactively validate checkpoints "
            "trained with the previous alignment implementation; tone-aware checkpoints "
            "must be retrained after this fix.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write_text(path: str | Path, content: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit exact Vietnamese word-to-BPE tone-label alignment."
    )
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        help=f"CSV/JSONL transcript source; repeatable (default: {DEFAULT_INPUT})",
    )
    parser.add_argument("--text-column", help="Override transcript column detection")
    parser.add_argument("--id-column", help="Override utterance-ID column detection")
    parser.add_argument("--tokenizer-source", default="vinai/PhoWhisper-base")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Transformers to download a missing tokenizer; disabled by default",
    )
    parser.add_argument(
        "--policy",
        choices=("last_subtoken", "all_subtokens"),
        default="last_subtoken",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Audit repeated normalized transcripts instead of retaining the first one",
    )
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-summary", default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument("--scope-note", default="")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing audit outputs",
    )
    return parser


def run(args: argparse.Namespace) -> AuditResult:
    input_paths = list(args.inputs or [DEFAULT_INPUT])
    output_csv = Path(args.output_csv)
    output_summary = Path(args.output_summary)
    if output_csv.resolve() == output_summary.resolve():
        raise ToneAuditError("CSV and Markdown outputs must be different files")
    existing = [path for path in (output_csv, output_summary) if path.exists()]
    if existing and not args.overwrite:
        raise ToneAuditError(
            "refusing to overwrite existing outputs without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    records = load_records(
        input_paths,
        text_column=args.text_column,
        id_column=args.id_column,
    )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ToneAuditError("transformers is required for the alignment audit") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_source,
        local_files_only=not args.allow_download,
    )
    result = audit_records(
        records,
        tokenizer,
        policy=args.policy,
        deduplicate_text=not args.keep_duplicates,
    )
    vocab_sha256 = tokenizer_vocab_sha256(tokenizer)
    summary = build_summary(
        result,
        input_paths=input_paths,
        tokenizer_source=args.tokenizer_source,
        vocab_sha256=vocab_sha256,
        policy=args.policy,
        deduplicate_text=not args.keep_duplicates,
        scope_note=args.scope_note,
    )
    atomic_write_csv(output_csv, result.rows, AUDIT_COLUMNS)
    atomic_write_text(output_summary, summary)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (ToneAuditError, OSError) as exc:
        parser.error(str(exc))
    print(
        f"tone alignment audit: {'PASS' if result.passed else 'FAIL'}; "
        f"exact={result.exact_transcripts}/{result.selected_transcripts}; "
        f"mismatches={result.mismatched_transcripts}"
    )
    print(f"CSV: {args.output_csv}")
    print(f"Summary: {args.output_summary}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
