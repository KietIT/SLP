from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
import unicodedata
from collections import Counter
from pathlib import Path
from unittest import mock

from PIL import Image

from scripts.build_error_artifacts import (
    CODA_CSV_NAME,
    CODA_MATRIX_COLUMNS,
    CODA_NONE,
    CODA_ORDER,
    CODA_OUTPUT_LOCK_NAME,
    CODA_PNG_NAME,
    ErrorArtifactError,
    OUTPUT_LOCK_NAME,
    SHORT_WORD_COLUMNS,
    SHORT_WORD_CSV_NAME,
    SHORT_WORD_ORDER,
    SHORT_WORD_OUTPUT_LOCK_NAME,
    TONE_MATRIX_COLUMNS,
    TONE_ORDER,
    build_coda_matrix_rows,
    build_tone_matrix_rows,
    canonical_candidate_lambdas,
    canonical_focus_runs,
    canonical_low_snrs,
    load_coda_aggregation,
    load_short_word_aggregation,
    load_tone_aggregation,
    resolve_coda_candidate_runs,
    resolve_candidate_runs,
    run_coda_build,
    run_short_word_build,
    run_build,
)
from scripts.error_analysis import EVENT_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_error_artifacts.py"
EXPECTED_TONE_COLUMNS = [
    "metric_version",
    "dataset",
    "model",
    "model_size",
    "train_type",
    "lambda",
    "seed",
    "group_type",
    "group_value",
    "ref_tone",
    "hyp_tone",
    "count",
    "ref_total",
    "row_rate",
]
EXPECTED_CODA_COLUMNS = [
    "metric_version",
    "dataset",
    "model",
    "model_size",
    "train_type",
    "lambda",
    "seed",
    "group_type",
    "group_value",
    "ref_coda",
    "hyp_coda",
    "count",
    "ref_total",
    "row_rate",
]
EXPECTED_SHORT_WORD_COLUMNS = [
    "metric_version",
    "dataset",
    "model",
    "model_size",
    "train_type",
    "lambda",
    "seed",
    "utt_id",
    "snr",
    "noise_type",
    "low_snr_scope",
    "deleted_word",
    "ref_index",
    "context_window",
    "left_context",
    "right_context",
    "context",
    "ref",
    "hyp",
]


def event_row(**updates: str) -> dict[str, str]:
    row = {
        "metric_version": "aligned_v1",
        "dataset": "vivos",
        "model": "phowhisper",
        "model_size": "base",
        "train_type": "tone_aware_lora",
        "lambda": "0.05",
        "seed": "42",
        "utt_id": "utt-001",
        "snr": "clean",
        "noise_type": "clean",
        "operation": "match",
        "ref_token": "ma",
        "hyp_token": "ma",
        "ref_index": "0",
        "hyp_index": "0",
        "ref": "ma",
        "hyp": "ma",
        "ref_tone_base": "ma",
        "hyp_tone_base": "ma",
        "ref_plain_base": "ma",
        "hyp_plain_base": "ma",
        "ref_tone": "ngang",
        "hyp_tone": "ngang",
        "ref_coda": "",
        "hyp_coda": "",
        "tone_eligible": "true",
        "tone_error": "false",
        "diacritic_eligible": "true",
        "diacritic_error": "false",
        "final_consonant_eligible": "false",
        "final_consonant_error": "false",
        "short_word_deletion": "false",
    }
    row.update(updates)
    return row


def fixture_rows() -> list[dict[str, str]]:
    return [
        event_row(utt_id="005-clean", snr="clean"),
        event_row(
            utt_id="005-zero-tone",
            snr="0",
            noise_type="noise",
            operation="substitution",
            ref_tone="sac",
            hyp_tone="huyen",
            tone_error="true",
        ),
        event_row(
            utt_id="005-five-tone",
            snr="5",
            noise_type="music",
            operation="substitution",
            ref_tone="ngang",
            hyp_tone="sac",
            tone_error="true",
        ),
        event_row(
            utt_id="005-zero-lexical",
            snr="0",
            noise_type="noise",
            operation="substitution",
            ref_token="má",
            hyp_token="màu",
            ref_tone="sac",
            hyp_tone="huyen",
            tone_eligible="false",
            tone_error="false",
        ),
        event_row(
            utt_id="005-zero-deletion",
            snr="0",
            noise_type="speech",
            operation="deletion",
            hyp_token="",
            hyp_index="",
            hyp="",
            ref_tone="nang",
            hyp_tone="",
            tone_error="true",
            diacritic_eligible="false",
        ),
        event_row(
            utt_id="005-zero-insertion",
            snr="0",
            noise_type="speech",
            operation="insertion",
            ref_token="",
            ref_index="",
            ref="",
            ref_tone="",
            hyp_tone="ngang",
            tone_eligible="false",
        ),
        event_row(
            utt_id="005-ten",
            snr="10",
            noise_type="music",
            ref_tone="hoi",
            hyp_tone="hoi",
        ),
        event_row(
            utt_id="01-clean",
            snr="clean",
            **{"lambda": "0.1"},
            ref_tone="huyen",
            hyp_tone="huyen",
        ),
        event_row(
            utt_id="01-zero-tone",
            snr="0",
            noise_type="noise",
            operation="substitution",
            **{"lambda": "0.1"},
            ref_tone="ngang",
            hyp_tone="huyen",
            tone_error="true",
        ),
        event_row(
            utt_id="01-five-match",
            snr="5",
            noise_type="music",
            **{"lambda": "0.1"},
        ),
        event_row(
            utt_id="01-zero-deletion",
            snr="0",
            noise_type="speech",
            operation="deletion",
            **{"lambda": "0.1"},
            hyp_token="",
            hyp_index="",
            hyp="",
            ref_tone="sac",
            hyp_tone="",
            tone_error="true",
            diacritic_eligible="false",
        ),
        event_row(
            utt_id="ordinary-clean",
            train_type="ordinary_lora",
            **{"lambda": "0"},
            ref_tone="hoi",
            hyp_tone="hoi",
        ),
        event_row(
            utt_id="ordinary-zero",
            snr="0",
            noise_type="noise",
            train_type="ordinary_lora",
            **{"lambda": "0"},
            ref_tone="sac",
            hyp_tone="sac",
        ),
        event_row(
            utt_id="ordinary-five",
            snr="5",
            noise_type="noise",
            train_type="ordinary_lora",
            operation="substitution",
            **{"lambda": "0"},
            ref_tone="nga",
            hyp_tone="nang",
            tone_error="true",
        ),
    ]


def coda_event_row(**updates: str) -> dict[str, str]:
    row = event_row(
        ref_token="ban",
        hyp_token="ban",
        ref="ban",
        hyp="ban",
        ref_coda="n",
        hyp_coda="n",
        final_consonant_eligible="true",
        final_consonant_error="false",
    )
    row.update(updates)
    return row


def coda_fixture_rows() -> list[dict[str, str]]:
    return [
        coda_event_row(utt_id="005-clean", snr="clean"),
        coda_event_row(
            utt_id="005-zero-gain",
            snr="0",
            noise_type="noise",
            operation="substitution",
            ref_token="ba",
            hyp_token="bang",
            ref_coda="",
            hyp_coda="ng",
            final_consonant_error="true",
        ),
        coda_event_row(
            utt_id="005-five-loss",
            snr="5",
            noise_type="music",
            operation="substitution",
            ref_token="bang",
            hyp_token="ba",
            ref_coda="ng",
            hyp_coda="",
            final_consonant_error="true",
        ),
        coda_event_row(
            utt_id="005-zero-swap",
            snr="0",
            noise_type="speech",
            operation="substitution",
            ref_coda="ch",
            hyp_coda="c",
            final_consonant_error="true",
        ),
        coda_event_row(
            utt_id="005-zero-deletion",
            snr="0",
            noise_type="speech",
            operation="deletion",
            ref_coda="p",
            hyp_coda="",
            hyp_token="",
            hyp_index="",
            hyp="",
            final_consonant_error="true",
        ),
        coda_event_row(
            utt_id="005-zero-insertion",
            snr="0",
            noise_type="speech",
            operation="insertion",
            ref_token="",
            ref_index="",
            ref="",
            ref_coda="",
            hyp_coda="n",
            final_consonant_eligible="false",
        ),
        coda_event_row(
            utt_id="005-ten",
            snr="10",
            noise_type="noise",
            ref_coda="m",
            hyp_coda="m",
        ),
        coda_event_row(
            utt_id="01-clean",
            snr="clean",
            **{"lambda": "0.1"},
            ref_coda="nh",
            hyp_coda="nh",
        ),
        coda_event_row(
            utt_id="01-zero-swap",
            snr="0",
            noise_type="noise",
            operation="substitution",
            **{"lambda": "0.1"},
            ref_coda="c",
            hyp_coda="ch",
            final_consonant_error="true",
        ),
        coda_event_row(
            utt_id="01-five-match",
            snr="5",
            noise_type="music",
            **{"lambda": "0.1"},
            ref_coda="t",
            hyp_coda="t",
        ),
        coda_event_row(
            utt_id="01-zero-deletion",
            snr="0",
            noise_type="speech",
            operation="deletion",
            **{"lambda": "0.1"},
            ref_coda="n",
            hyp_coda="",
            hyp_token="",
            hyp_index="",
            hyp="",
            final_consonant_error="true",
        ),
        coda_event_row(
            utt_id="ordinary-clean",
            train_type="ordinary_lora",
            **{"lambda": "0"},
            ref_coda="ng",
            hyp_coda="ng",
        ),
        coda_event_row(
            utt_id="ordinary-zero",
            snr="0",
            noise_type="noise",
            train_type="ordinary_lora",
            operation="substitution",
            **{"lambda": "0"},
            ref_coda="",
            hyp_coda="p",
            final_consonant_error="true",
        ),
        coda_event_row(
            utt_id="ordinary-five",
            snr="5",
            noise_type="music",
            train_type="ordinary_lora",
            **{"lambda": "0"},
            ref_coda="p",
            hyp_coda="p",
        ),
    ]


def short_word_event_row(**updates: str) -> dict[str, str]:
    row = event_row(
        ref_token="có",
        hyp_token="có",
        ref_index="0",
        hyp_index="0",
        ref="có",
        hyp="có",
        short_word_deletion="false",
    )
    row.update(updates)
    return row


def short_word_fixture_rows() -> list[dict[str, str]]:
    return [
        short_word_event_row(
            utt_id="005-clean",
            snr="clean",
            ref_token="đã",
            hyp_token="đã",
            ref="đã ở đây",
            hyp="đã ở đây",
        ),
        short_word_event_row(
            utt_id="005-zero-middle",
            snr="0",
            noise_type="noise",
            operation="deletion",
            ref_token="và",
            hyp_token="",
            ref_index="2",
            hyp_index="",
            ref="TÔI ĐÃ VÀ ĐI LÀM",
            hyp="tôi đã đi làm",
            short_word_deletion="true",
        ),
        short_word_event_row(
            utt_id="005-five-match",
            snr="5",
            noise_type="music",
        ),
        short_word_event_row(
            utt_id="005-ten-start",
            snr="10",
            noise_type="speech",
            operation="deletion",
            ref_token="một",
            hyp_token="",
            ref_index="0",
            hyp_index="",
            ref=unicodedata.normalize("NFD", "MỘT ngày đẹp"),
            hyp="ngày đẹp",
            short_word_deletion="true",
        ),
        short_word_event_row(
            utt_id="01-clean",
            snr="clean",
            **{"lambda": "0.1"},
        ),
        short_word_event_row(
            utt_id="01-zero-match",
            snr="0",
            noise_type="noise",
            **{"lambda": "0.1"},
            ref_token="đã",
            hyp_token="đã",
            ref="đã xong",
            hyp="đã xong",
        ),
        short_word_event_row(
            utt_id="01-five-end",
            snr="5",
            noise_type="music",
            operation="deletion",
            **{"lambda": "0.1"},
            ref_token="và",
            hyp_token="",
            ref_index="2",
            hyp_index="",
            ref="tôi đi và",
            hyp="tôi đi",
            short_word_deletion="true",
        ),
        short_word_event_row(
            utt_id="ordinary-clean",
            snr="clean",
            train_type="ordinary_lora",
            **{"lambda": "0"},
            ref_token="một",
            hyp_token="một",
            ref="một lần",
            hyp="một lần",
        ),
        short_word_event_row(
            utt_id="ordinary-zero",
            snr="0",
            noise_type="noise",
            train_type="ordinary_lora",
            **{"lambda": "0"},
        ),
        short_word_event_row(
            utt_id="ordinary-five-repeat",
            snr="5",
            noise_type="music",
            operation="deletion",
            train_type="ordinary_lora",
            **{"lambda": "0"},
            ref_token="là",
            hyp_token="",
            ref_index="2",
            hyp_index="",
            ref="là đây là",
            hyp="là đây",
            short_word_deletion="true",
        ),
    ]


def write_events(
    path: Path,
    rows: list[dict[str, str]],
    *,
    columns: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_columns = columns or EVENT_COLUMNS
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=selected_columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def find_cell(
    rows: list[dict[str, object]],
    *,
    lambda_value: str,
    scope: str,
    ref_tone: str,
    hyp_tone: str,
) -> dict[str, object]:
    matches = [
        row
        for row in rows
        if row["lambda"] == lambda_value
        and row["group_value"] == scope
        and row["ref_tone"] == ref_tone
        and row["hyp_tone"] == hyp_tone
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one matrix cell, found {len(matches)}")
    return matches[0]


def find_coda_cell(
    rows: list[dict[str, object]],
    *,
    lambda_value: str,
    scope: str,
    ref_coda: str,
    hyp_coda: str,
) -> dict[str, object]:
    matches = [
        row
        for row in rows
        if row["lambda"] == lambda_value
        and row["group_value"] == scope
        and row["ref_coda"] == ref_coda
        and row["hyp_coda"] == hyp_coda
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one coda matrix cell, found {len(matches)}")
    return matches[0]


class BuildErrorArtifactsTest(unittest.TestCase):
    def test_gate_d_schema_and_tone_order_are_fixed(self) -> None:
        self.assertEqual(TONE_MATRIX_COLUMNS, EXPECTED_TONE_COLUMNS)
        self.assertEqual(
            TONE_ORDER,
            ("ngang", "sac", "huyen", "hoi", "nga", "nang"),
        )

    def test_dense_matrices_filter_events_and_keep_scopes_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            event_path = Path(temporary) / "error_events.csv"
            write_events(event_path, fixture_rows())

            aggregation = load_tone_aggregation(event_path, low_snrs=("0", "5"))
            rows = build_tone_matrix_rows(aggregation)

        self.assertEqual(aggregation.event_rows, 14)
        self.assertEqual(len(aggregation.run_keys), 3)
        self.assertEqual(len(rows), 3 * 2 * 6 * 6)
        self.assertEqual(
            {(row["group_type"], row["group_value"]) for row in rows},
            {("scope", "overall"), ("scope", "low_snr")},
        )

        overall_ngang_ngang = find_cell(
            rows,
            lambda_value="0.05",
            scope="overall",
            ref_tone="ngang",
            hyp_tone="ngang",
        )
        overall_ngang_sac = find_cell(
            rows,
            lambda_value="0.05",
            scope="overall",
            ref_tone="ngang",
            hyp_tone="sac",
        )
        low_ngang_ngang = find_cell(
            rows,
            lambda_value="0.05",
            scope="low_snr",
            ref_tone="ngang",
            hyp_tone="ngang",
        )
        low_ngang_sac = find_cell(
            rows,
            lambda_value="0.05",
            scope="low_snr",
            ref_tone="ngang",
            hyp_tone="sac",
        )
        self.assertEqual(overall_ngang_ngang["count"], 1)
        self.assertEqual(overall_ngang_sac["count"], 1)
        self.assertEqual(overall_ngang_ngang["ref_total"], 2)
        self.assertEqual(float(overall_ngang_ngang["row_rate"]), 0.5)
        self.assertEqual(float(overall_ngang_sac["row_rate"]), 0.5)
        self.assertEqual(low_ngang_ngang["count"], 0)
        self.assertEqual(low_ngang_sac["count"], 1)
        self.assertEqual(float(low_ngang_sac["row_rate"]), 1.0)

        zero_cell = find_cell(
            rows,
            lambda_value="0.05",
            scope="low_snr",
            ref_tone="nang",
            hyp_tone="nang",
        )
        self.assertEqual(zero_cell["count"], 0)
        self.assertEqual(zero_cell["ref_total"], 0)
        self.assertEqual(float(zero_cell["row_rate"]), 0.0)

        run_005 = next(run for run in aggregation.run_keys if run[4] == "0.05")
        self.assertEqual(aggregation.eligible_deletions[(run_005, "overall")], 1)
        self.assertEqual(aggregation.eligible_deletions[(run_005, "low_snr")], 1)
        selected = resolve_candidate_runs(
            aggregation, canonical_candidate_lambdas(("0.050", "0.10"))
        )
        self.assertEqual([candidate for candidate, _ in selected], ["0.05", "0.1"])

    def test_invalid_event_contract_is_rejected(self) -> None:
        mutations = [
            ("metric_version", "simple_v1", "metric_version"),
            ("tone_eligible", "yes", "true or false"),
            ("operation", "replace", "unknown operation"),
            ("ref_tone", "unknown", "unknown ref_tone"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (field, value, message) in enumerate(mutations):
                with self.subTest(field=field):
                    rows = fixture_rows()
                    rows[0][field] = value
                    path = root / f"bad_{index}.csv"
                    write_events(path, rows)
                    with self.assertRaisesRegex(ErrorArtifactError, message):
                        load_tone_aggregation(path, low_snrs=("0", "5"))

            short_header = root / "short_header.csv"
            write_events(short_header, fixture_rows(), columns=EVENT_COLUMNS[:-1])
            with self.assertRaisesRegex(ErrorArtifactError, "exact Gate C event columns"):
                load_tone_aggregation(short_header, low_snrs=("0", "5"))

    def test_low_snr_accepts_negative_values_and_is_complete_for_every_run(self) -> None:
        self.assertEqual(canonical_low_snrs(("-5.0", "0.00")), ("-5", "0"))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            negative_rows = fixture_rows()
            for row in negative_rows:
                if row["snr"] == "0":
                    row["snr"] = "-5"
            negative_path = root / "negative.csv"
            write_events(negative_path, negative_rows)
            aggregation = load_tone_aggregation(
                negative_path,
                low_snrs=("-5", "5"),
            )
            self.assertEqual(aggregation.low_snrs, ("-5", "5"))

            incomplete_rows = [
                row
                for row in fixture_rows()
                if not (row["lambda"] == "0.05" and row["snr"] == "5")
            ]
            incomplete_path = root / "incomplete.csv"
            write_events(incomplete_path, incomplete_rows)
            with self.assertRaisesRegex(
                ErrorArtifactError,
                r"lambda='0\.05'.*missing 5",
            ):
                load_tone_aggregation(incomplete_path, low_snrs=("0", "5"))

            complete_path = root / "complete.csv"
            write_events(complete_path, fixture_rows())
            canonicalized = load_tone_aggregation(
                complete_path,
                low_snrs=("0.00", "5.0"),
            )
            self.assertEqual(canonicalized.low_snrs, ("0", "5"))
            with self.assertRaisesRegex(ErrorArtifactError, r"missing 50"):
                load_tone_aggregation(complete_path, low_snrs=("50",))

    def test_missing_duplicate_and_ambiguous_candidates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ErrorArtifactError, "unique"):
            canonical_candidate_lambdas(("0.05", "0.050"))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_path = root / "error_events.csv"
            write_events(event_path, fixture_rows())
            aggregation = load_tone_aggregation(event_path, low_snrs=("0", "5"))
            with self.assertRaisesRegex(ErrorArtifactError, "found 0"):
                resolve_candidate_runs(aggregation, ("0.05", "0.3"))

            ambiguous_rows = fixture_rows()
            ambiguous_rows.append(
                event_row(
                    utt_id="005-seed-43",
                    seed="43",
                    **{"lambda": "0.05"},
                )
            )
            ambiguous_rows.extend(
                [
                    event_row(
                        utt_id=f"005-seed-43-{snr}",
                        seed="43",
                        snr=snr,
                        noise_type="noise",
                        **{"lambda": "0.05"},
                    )
                    for snr in ("0", "5")
                ]
            )
            ambiguous_path = root / "ambiguous.csv"
            write_events(ambiguous_path, ambiguous_rows)
            ambiguous = load_tone_aggregation(ambiguous_path, low_snrs=("0", "5"))
            with self.assertRaisesRegex(ErrorArtifactError, "found 2"):
                resolve_candidate_runs(ambiguous, ("0.05", "0.1"))

    def test_cli_writes_valid_deterministic_outputs_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_path = root / "error_events.csv"
            output_dir = root / "artifacts"
            write_events(event_path, fixture_rows())
            command = [
                sys.executable,
                "-X",
                "utf8",
                str(SCRIPT),
                "--events",
                str(event_path),
                "--candidate-lambda",
                "0.05",
                "--candidate-lambda",
                "0.1",
                "--low-snr",
                "0",
                "--low-snr",
                "5",
                "--out-dir",
                str(output_dir),
            ]

            first = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("candidate lambda=0.05 scope=overall", first.stdout)
            self.assertIn("candidate lambda=0.1 scope=low_snr", first.stdout)

            csv_path = output_dir / "tone_confusion_matrix.csv"
            png_path = output_dir / "tone_confusion_matrix.png"
            header, rows = read_csv(csv_path)
            self.assertEqual(header, EXPECTED_TONE_COLUMNS)
            self.assertEqual(len(rows), 3 * 2 * 6 * 6)
            csv_before = csv_path.read_bytes()
            png_before = png_path.read_bytes()
            self.assertTrue(
                (output_dir / "tone_confusion_matrix.provenance.json").is_file()
            )
            self.assertTrue(
                (output_dir / "tone_confusion_matrix.bundle.commit.json").is_file()
            )
            resumed = subprocess.run(
                [*command, "--resume"], capture_output=True, text=True, check=False
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(csv_path.read_bytes(), csv_before)
            self.assertEqual(png_path.read_bytes(), png_before)
            with Image.open(png_path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreaterEqual(image.width, 4000)
                self.assertGreaterEqual(image.height, 2800)
                dpi = image.info.get("dpi", (0, 0))
                self.assertGreaterEqual(dpi[0], 299)
                self.assertGreaterEqual(dpi[1], 299)

            refused = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(refused.returncode, 2)
            self.assertIn("output already exists", refused.stderr)
            self.assertEqual(csv_path.read_bytes(), csv_before)
            self.assertEqual(png_path.read_bytes(), png_before)

            overwritten = subprocess.run(
                [*command, "--overwrite"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(overwritten.returncode, 0, overwritten.stderr)
            self.assertEqual(csv_path.read_bytes(), csv_before)
            self.assertEqual(png_path.read_bytes(), png_before)
            self.assertFalse((output_dir / ".tone_confusion_matrix.csv.tmp").exists())
            self.assertFalse((output_dir / ".tone_confusion_matrix.png.tmp").exists())
            self.assertFalse((output_dir / ".tone_confusion_matrix.csv.bak").exists())
            self.assertFalse((output_dir / ".tone_confusion_matrix.png.bak").exists())
            self.assertFalse((output_dir / OUTPUT_LOCK_NAME).exists())

    def test_existing_output_lock_blocks_a_second_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_path = root / "error_events.csv"
            output_dir = root / "artifacts"
            output_dir.mkdir()
            write_events(event_path, fixture_rows())
            lock_path = output_dir / OUTPUT_LOCK_NAME
            lock_contents = "pid=external\n"
            lock_path.write_text(lock_contents, encoding="utf-8")

            with self.assertRaisesRegex(ErrorArtifactError, "locked by another build"):
                run_build(
                    event_path,
                    output_dir,
                    candidate_lambdas=("0.05", "0.1"),
                    low_snrs=("0", "5"),
                )

            self.assertEqual(lock_path.read_text(encoding="utf-8"), lock_contents)
            self.assertFalse((output_dir / "tone_confusion_matrix.csv").exists())
            self.assertFalse((output_dir / "tone_confusion_matrix.png").exists())

    def test_second_commit_failure_restores_the_previous_output_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_path = root / "error_events.csv"
            output_dir = root / "artifacts"
            output_dir.mkdir()
            write_events(event_path, fixture_rows())

            csv_path = output_dir / "tone_confusion_matrix.csv"
            png_path = output_dir / "tone_confusion_matrix.png"
            old_csv = b"previous csv generation\n"
            old_png = b"previous png generation\n"
            csv_path.write_bytes(old_csv)
            png_path.write_bytes(old_png)

            original_replace = Path.replace
            failed = False

            def replace_with_second_commit_failure(
                source: Path,
                target: str | Path,
            ) -> Path:
                nonlocal failed
                target_path = Path(target)
                if (
                    not failed
                    and source.name == ".tone_confusion_matrix.png.tmp"
                    and target_path == png_path
                ):
                    failed = True
                    raise OSError("injected PNG commit failure")
                return original_replace(source, target)

            with mock.patch.object(
                Path,
                "replace",
                autospec=True,
                side_effect=replace_with_second_commit_failure,
            ):
                with self.assertRaisesRegex(OSError, "injected PNG commit failure"):
                    run_build(
                        event_path,
                        output_dir,
                        candidate_lambdas=("0.05", "0.1"),
                        low_snrs=("0", "5"),
                        overwrite=True,
                    )

            self.assertTrue(failed)
            self.assertEqual(csv_path.read_bytes(), old_csv)
            self.assertEqual(png_path.read_bytes(), old_png)
            for suffix in ("tmp", "bak"):
                self.assertFalse(
                    (output_dir / f".tone_confusion_matrix.csv.{suffix}").exists()
                )
                self.assertFalse(
                    (output_dir / f".tone_confusion_matrix.png.{suffix}").exists()
                )
            self.assertFalse((output_dir / OUTPUT_LOCK_NAME).exists())

    def test_render_failure_cleans_temporary_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_path = root / "error_events.csv"
            output_dir = root / "artifacts"
            write_events(event_path, fixture_rows())

            def fail_after_partial_render(path: Path, *_: object) -> None:
                path.write_bytes(b"partial PNG")
                raise RuntimeError("render failed")

            with mock.patch(
                "scripts.build_error_artifacts.render_tone_confusion_png",
                side_effect=fail_after_partial_render,
            ):
                with self.assertRaisesRegex(RuntimeError, "render failed"):
                    run_build(
                        event_path,
                        output_dir,
                        candidate_lambdas=("0.05", "0.1"),
                        low_snrs=("0", "5"),
                    )

            self.assertFalse((output_dir / "tone_confusion_matrix.csv").exists())
            self.assertFalse((output_dir / "tone_confusion_matrix.png").exists())
            self.assertFalse((output_dir / ".tone_confusion_matrix.csv.tmp").exists())
            self.assertFalse((output_dir / ".tone_confusion_matrix.png.tmp").exists())
            self.assertFalse((output_dir / OUTPUT_LOCK_NAME).exists())

    def test_gate_e_schema_order_and_dense_coda_filtering(self) -> None:
        self.assertEqual(CODA_MATRIX_COLUMNS, EXPECTED_CODA_COLUMNS)
        self.assertEqual(
            CODA_ORDER,
            ("none", "n", "ng", "nh", "t", "c", "ch", "m", "p"),
        )

        with tempfile.TemporaryDirectory() as temporary:
            event_path = Path(temporary) / "error_events.csv"
            write_events(event_path, coda_fixture_rows())
            aggregation = load_coda_aggregation(event_path, low_snrs=("0", "5"))
            rows = build_coda_matrix_rows(aggregation)

        self.assertEqual(aggregation.event_rows, 14)
        self.assertEqual(len(aggregation.run_keys), 3)
        self.assertEqual(len(rows), 3 * 2 * 9 * 9)
        self.assertEqual(
            {(row["group_type"], row["group_value"]) for row in rows},
            {("scope", "overall"), ("scope", "low_snr")},
        )

        overall_gain = find_coda_cell(
            rows,
            lambda_value="0.05",
            scope="overall",
            ref_coda=CODA_NONE,
            hyp_coda="ng",
        )
        low_loss = find_coda_cell(
            rows,
            lambda_value="0.05",
            scope="low_snr",
            ref_coda="ng",
            hyp_coda=CODA_NONE,
        )
        low_deletion = find_coda_cell(
            rows,
            lambda_value="0.05",
            scope="low_snr",
            ref_coda="p",
            hyp_coda=CODA_NONE,
        )
        empty_cell = find_coda_cell(
            rows,
            lambda_value="0.05",
            scope="overall",
            ref_coda=CODA_NONE,
            hyp_coda=CODA_NONE,
        )
        self.assertEqual(overall_gain["count"], 1)
        self.assertEqual(overall_gain["ref_total"], 1)
        self.assertEqual(float(overall_gain["row_rate"]), 1.0)
        self.assertEqual(low_loss["count"], 1)
        self.assertEqual(low_deletion["count"], 1)
        self.assertEqual(empty_cell["count"], 0)
        self.assertEqual(empty_cell["ref_total"], 1)

        run_005 = next(run for run in aggregation.run_keys if run[4] == "0.05")
        self.assertEqual(sum(aggregation.counts[(run_005, "overall")].values()), 6)
        self.assertEqual(sum(aggregation.counts[(run_005, "low_snr")].values()), 4)
        self.assertEqual(aggregation.word_deletions[(run_005, "overall")], 1)
        self.assertEqual(aggregation.word_deletions[(run_005, "low_snr")], 1)
        selected = resolve_coda_candidate_runs(
            aggregation,
            canonical_candidate_lambdas(("0.050", "0.10")),
        )
        self.assertEqual([candidate for candidate, _ in selected], ["0.05", "0.1"])

    def test_invalid_coda_event_invariants_are_rejected(self) -> None:
        cases: list[tuple[str, list[dict[str, str]], str]] = []

        bad_bool = coda_fixture_rows()
        bad_bool[0]["final_consonant_eligible"] = "yes"
        cases.append(("bool", bad_bool, "true or false"))

        error_without_eligibility = coda_fixture_rows()
        error_without_eligibility[0]["final_consonant_eligible"] = "false"
        error_without_eligibility[0]["final_consonant_error"] = "true"
        cases.append(("eligibility", error_without_eligibility, "requires"))

        unknown_coda = coda_fixture_rows()
        unknown_coda[0]["ref_coda"] = "z"
        cases.append(("label", unknown_coda, "unknown ref_coda"))

        empty_pair = coda_fixture_rows()
        empty_pair[0]["ref_coda"] = ""
        empty_pair[0]["hyp_coda"] = ""
        cases.append(("empty", empty_pair, "at least one coda"))

        eligible_insertion = coda_fixture_rows()
        insertion = next(row for row in eligible_insertion if row["operation"] == "insertion")
        insertion["final_consonant_eligible"] = "true"
        insertion["final_consonant_error"] = "true"
        cases.append(("insertion", eligible_insertion, "insertion cannot"))

        mismatched_error = coda_fixture_rows()
        mismatched_error[0]["final_consonant_error"] = "true"
        cases.append(("error", mismatched_error, "disagrees with the coda pair"))

        missing_eligibility = coda_fixture_rows()
        missing_eligibility[0]["final_consonant_eligible"] = "false"
        cases.append(
            (
                "missing_eligibility",
                missing_eligibility,
                "final_consonant_eligible disagrees",
            )
        )

        malformed_deletion = coda_fixture_rows()
        deletion = next(row for row in malformed_deletion if row["operation"] == "deletion")
        deletion["hyp_coda"] = "n"
        cases.append(("deletion", malformed_deletion, "deletion must have empty hyp_coda"))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, rows, message in cases:
                with self.subTest(name=name):
                    path = root / f"{name}.csv"
                    write_events(path, rows)
                    with self.assertRaisesRegex(ErrorArtifactError, message):
                        load_coda_aggregation(path, low_snrs=("0", "5"))

    def test_coda_cli_is_deterministic_and_does_not_touch_tone_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_path = root / "error_events.csv"
            output_dir = root / "artifacts"
            output_dir.mkdir()
            write_events(event_path, coda_fixture_rows())

            tone_csv = output_dir / "tone_confusion_matrix.csv"
            tone_png = output_dir / "tone_confusion_matrix.png"
            tone_csv.write_bytes(b"approved tone csv\n")
            tone_png.write_bytes(b"approved tone png\n")
            command = [
                sys.executable,
                "-X",
                "utf8",
                str(SCRIPT),
                "--artifact",
                "coda",
                "--events",
                str(event_path),
                "--candidate-lambda",
                "0.05",
                "--candidate-lambda",
                "0.1",
                "--low-snr",
                "0",
                "--low-snr",
                "5",
                "--out-dir",
                str(output_dir),
            ]

            first = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("matrix=6 off_diagonal=4 word_deletions=1", first.stdout)
            csv_path = output_dir / CODA_CSV_NAME
            png_path = output_dir / CODA_PNG_NAME
            header, rows = read_csv(csv_path)
            self.assertEqual(header, EXPECTED_CODA_COLUMNS)
            self.assertEqual(len(rows), 3 * 2 * 9 * 9)
            csv_before = csv_path.read_bytes()
            png_before = png_path.read_bytes()
            with Image.open(png_path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (4500, 3600))
                dpi = image.info.get("dpi", (0, 0))
                self.assertGreaterEqual(dpi[0], 299)
                self.assertGreaterEqual(dpi[1], 299)

            self.assertEqual(tone_csv.read_bytes(), b"approved tone csv\n")
            self.assertEqual(tone_png.read_bytes(), b"approved tone png\n")
            refused = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(refused.returncode, 2)
            self.assertIn("output already exists", refused.stderr)
            self.assertEqual(csv_path.read_bytes(), csv_before)
            self.assertEqual(png_path.read_bytes(), png_before)

            overwritten = subprocess.run(
                [*command, "--overwrite"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(overwritten.returncode, 0, overwritten.stderr)
            self.assertEqual(csv_path.read_bytes(), csv_before)
            self.assertEqual(png_path.read_bytes(), png_before)
            self.assertEqual(tone_csv.read_bytes(), b"approved tone csv\n")
            self.assertEqual(tone_png.read_bytes(), b"approved tone png\n")
            for suffix in ("tmp", "bak"):
                self.assertFalse((output_dir / f".{CODA_CSV_NAME}.{suffix}").exists())
                self.assertFalse((output_dir / f".{CODA_PNG_NAME}.{suffix}").exists())
            self.assertFalse((output_dir / CODA_OUTPUT_LOCK_NAME).exists())

    def test_coda_lock_and_render_failure_leave_no_partial_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_path = root / "error_events.csv"
            output_dir = root / "artifacts"
            output_dir.mkdir()
            write_events(event_path, coda_fixture_rows())
            lock_path = output_dir / CODA_OUTPUT_LOCK_NAME
            lock_contents = "pid=external\n"
            lock_path.write_text(lock_contents, encoding="utf-8")

            with self.assertRaisesRegex(ErrorArtifactError, "locked by another build"):
                run_coda_build(
                    event_path,
                    output_dir,
                    candidate_lambdas=("0.05", "0.1"),
                    low_snrs=("0", "5"),
                )
            self.assertEqual(lock_path.read_text(encoding="utf-8"), lock_contents)
            lock_path.unlink()

            with mock.patch.object(
                csv.DictWriter,
                "writerows",
                autospec=True,
                side_effect=RuntimeError("coda CSV write failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "coda CSV write failed"):
                    run_coda_build(
                        event_path,
                        output_dir,
                        candidate_lambdas=("0.05", "0.1"),
                        low_snrs=("0", "5"),
                    )
            self.assertFalse((output_dir / f".{CODA_CSV_NAME}.tmp").exists())
            self.assertFalse((output_dir / CODA_OUTPUT_LOCK_NAME).exists())

            def fail_after_partial_render(path: Path, *_: object) -> None:
                path.write_bytes(b"partial PNG")
                raise RuntimeError("coda render failed")

            with mock.patch(
                "scripts.build_error_artifacts.render_coda_confusion_png",
                side_effect=fail_after_partial_render,
            ):
                with self.assertRaisesRegex(RuntimeError, "coda render failed"):
                    run_coda_build(
                        event_path,
                        output_dir,
                        candidate_lambdas=("0.05", "0.1"),
                        low_snrs=("0", "5"),
                    )

            self.assertFalse((output_dir / CODA_CSV_NAME).exists())
            self.assertFalse((output_dir / CODA_PNG_NAME).exists())
            self.assertFalse((output_dir / f".{CODA_CSV_NAME}.tmp").exists())
            self.assertFalse((output_dir / f".{CODA_PNG_NAME}.tmp").exists())
            self.assertFalse((output_dir / CODA_OUTPUT_LOCK_NAME).exists())

    def test_coda_second_commit_failure_restores_previous_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_path = root / "error_events.csv"
            output_dir = root / "artifacts"
            output_dir.mkdir()
            write_events(event_path, coda_fixture_rows())

            csv_path = output_dir / CODA_CSV_NAME
            png_path = output_dir / CODA_PNG_NAME
            old_csv = b"previous coda csv\n"
            old_png = b"previous coda png\n"
            csv_path.write_bytes(old_csv)
            png_path.write_bytes(old_png)
            original_replace = Path.replace
            failed = False

            def replace_with_png_failure(source: Path, target: str | Path) -> Path:
                nonlocal failed
                if (
                    not failed
                    and source.name == f".{CODA_PNG_NAME}.tmp"
                    and Path(target) == png_path
                ):
                    failed = True
                    raise OSError("injected coda PNG commit failure")
                return original_replace(source, target)

            with mock.patch.object(
                Path,
                "replace",
                autospec=True,
                side_effect=replace_with_png_failure,
            ):
                with self.assertRaisesRegex(OSError, "injected coda PNG commit failure"):
                    run_coda_build(
                        event_path,
                        output_dir,
                        candidate_lambdas=("0.05", "0.1"),
                        low_snrs=("0", "5"),
                        overwrite=True,
                    )

            self.assertTrue(failed)
            self.assertEqual(csv_path.read_bytes(), old_csv)
            self.assertEqual(png_path.read_bytes(), old_png)
            for suffix in ("tmp", "bak"):
                self.assertFalse((output_dir / f".{CODA_CSV_NAME}.{suffix}").exists())
                self.assertFalse((output_dir / f".{CODA_PNG_NAME}.{suffix}").exists())
            self.assertFalse((output_dir / CODA_OUTPUT_LOCK_NAME).exists())

    def test_gate_f_schema_context_sorting_and_lambda_separation(self) -> None:
        self.assertEqual(SHORT_WORD_COLUMNS, EXPECTED_SHORT_WORD_COLUMNS)
        self.assertEqual(SHORT_WORD_ORDER, ("đã", "có", "là", "một", "và"))

        with tempfile.TemporaryDirectory() as temporary:
            event_path = Path(temporary) / "error_events.csv"
            write_events(event_path, short_word_fixture_rows())
            aggregation = load_short_word_aggregation(
                event_path,
                low_snrs=("0", "5"),
                context_window=2,
            )

        self.assertEqual(aggregation.event_rows, 10)
        self.assertEqual(len(aggregation.run_keys), 3)
        self.assertEqual(len(aggregation.examples), 4)
        examples = {row["utt_id"]: row for row in aggregation.examples}
        middle = examples["005-zero-middle"]
        self.assertEqual(middle["left_context"], "tôi đã")
        self.assertEqual(middle["right_context"], "đi làm")
        self.assertEqual(middle["context"], "tôi đã ⟦và⟧ đi làm")
        self.assertEqual(middle["low_snr_scope"], "true")

        start = examples["005-ten-start"]
        self.assertEqual(start["context"], "⟦một⟧ ngày đẹp")
        self.assertEqual(start["left_context"], "")
        self.assertEqual(start["low_snr_scope"], "false")
        self.assertTrue(unicodedata.is_normalized("NFC", str(start["ref"])))
        self.assertEqual(examples["01-five-end"]["context"], "tôi đi ⟦và⟧")
        self.assertEqual(
            examples["ordinary-five-repeat"]["context"],
            "là đây ⟦là⟧",
        )

        run_005 = next(run for run in aggregation.run_keys if run[4] == "0.05")
        run_01 = next(run for run in aggregation.run_keys if run[4] == "0.1")
        self.assertEqual(aggregation.deletion_counts[(run_005, "overall")], 2)
        self.assertEqual(aggregation.deletion_counts[(run_005, "low_snr")], 1)
        self.assertEqual(aggregation.reference_units[(run_005, "overall")], 4)
        self.assertEqual(aggregation.reference_units[(run_005, "low_snr")], 2)
        self.assertEqual(aggregation.deletion_counts[(run_01, "overall")], 1)
        self.assertEqual(aggregation.deletion_counts[(run_01, "low_snr")], 1)
        self.assertEqual(
            aggregation.word_counts[(run_005, "overall")],
            Counter({"và": 1, "một": 1}),
        )

        observed_sort = [
            (
                str(row["train_type"]).casefold(),
                str(row["lambda"]).casefold(),
                str(row["utt_id"]).casefold(),
                int(row["ref_index"]),
            )
            for row in aggregation.examples
        ]
        self.assertEqual(observed_sort, sorted(observed_sort))

    def test_invalid_short_word_event_contract_is_rejected(self) -> None:
        cases: list[tuple[str, list[dict[str, str]], str]] = []

        missing_flag = short_word_fixture_rows()
        missing_flag[1]["short_word_deletion"] = "false"
        cases.append(("missing_flag", missing_flag, "disagrees"))

        false_positive = short_word_fixture_rows()
        false_positive[0]["short_word_deletion"] = "true"
        cases.append(("false_positive", false_positive, "disagrees"))

        nonempty_hyp = short_word_fixture_rows()
        nonempty_hyp[1]["hyp_token"] = "và"
        cases.append(("hyp", nonempty_hyp, "empty hyp_token and hyp_index"))

        invalid_index = short_word_fixture_rows()
        invalid_index[1]["ref_index"] = "x"
        cases.append(("index", invalid_index, "non-negative integer ref_index"))

        mismatched_index = short_word_fixture_rows()
        mismatched_index[1]["ref_index"] = "1"
        cases.append(("mismatch", mismatched_index, "not ref_token"))

        duplicate = short_word_fixture_rows()
        duplicate.append(dict(duplicate[1]))
        cases.append(("duplicate", duplicate, "duplicate short-word deletion key"))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, rows, message in cases:
                with self.subTest(name=name):
                    path = root / f"{name}.csv"
                    write_events(path, rows)
                    with self.assertRaisesRegex(ErrorArtifactError, message):
                        load_short_word_aggregation(
                            path,
                            low_snrs=("0", "5"),
                            context_window=2,
                        )

            valid_path = root / "valid.csv"
            write_events(valid_path, short_word_fixture_rows())
            with self.assertRaisesRegex(ErrorArtifactError, "non-negative"):
                load_short_word_aggregation(
                    valid_path,
                    low_snrs=("0", "5"),
                    context_window=-1,
                )

    def test_short_word_cli_is_deterministic_and_preserves_gate_d_e(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_path = root / "error_events.csv"
            output_dir = root / "artifacts"
            output_dir.mkdir()
            write_events(event_path, short_word_fixture_rows())

            approved_files = {
                output_dir / "tone_confusion_matrix.csv": b"approved tone csv\n",
                output_dir / "tone_confusion_matrix.png": b"approved tone png\n",
                output_dir / CODA_CSV_NAME: b"approved coda csv\n",
                output_dir / CODA_PNG_NAME: b"approved coda png\n",
            }
            for path, contents in approved_files.items():
                path.write_bytes(contents)

            command = [
                sys.executable,
                "-X",
                "utf8",
                str(SCRIPT),
                "--artifact",
                "short-word",
                "--events",
                str(event_path),
                "--candidate-lambda",
                "0.05",
                "--candidate-lambda",
                "0.1",
                "--low-snr",
                "0",
                "--low-snr",
                "5",
                "--context-window",
                "2",
                "--out-dir",
                str(output_dir),
            ]

            first = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn(
                "candidate lambda=0.05 scope=overall examples=2 "
                "reference_units=4 swdr=0.500000000000",
                first.stdout,
            )
            self.assertIn(
                "candidate lambda=0.1 scope=low_snr examples=1 "
                "reference_units=2 swdr=0.500000000000",
                first.stdout,
            )
            csv_path = output_dir / SHORT_WORD_CSV_NAME
            header, rows = read_csv(csv_path)
            self.assertEqual(header, EXPECTED_SHORT_WORD_COLUMNS)
            self.assertEqual(len(rows), 4)
            self.assertEqual(sum(row["lambda"] == "0.05" for row in rows), 2)
            self.assertEqual(sum(row["lambda"] == "0.1" for row in rows), 1)
            csv_before = csv_path.read_bytes()
            self.assertFalse(
                (output_dir / "short_word_deletion_examples.png").exists()
            )
            for path, contents in approved_files.items():
                self.assertEqual(path.read_bytes(), contents)

            refused = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(refused.returncode, 2)
            self.assertIn("output already exists", refused.stderr)
            self.assertEqual(csv_path.read_bytes(), csv_before)

            overwritten = subprocess.run(
                [*command, "--overwrite"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(overwritten.returncode, 0, overwritten.stderr)
            self.assertEqual(csv_path.read_bytes(), csv_before)
            for path, contents in approved_files.items():
                self.assertEqual(path.read_bytes(), contents)
            self.assertFalse((output_dir / f".{SHORT_WORD_CSV_NAME}.tmp").exists())
            self.assertFalse((output_dir / f".{SHORT_WORD_CSV_NAME}.bak").exists())
            self.assertFalse((output_dir / SHORT_WORD_OUTPUT_LOCK_NAME).exists())

    def test_short_word_lock_write_failure_and_rollback_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_path = root / "error_events.csv"
            output_dir = root / "artifacts"
            output_dir.mkdir()
            write_events(event_path, short_word_fixture_rows())
            lock_path = output_dir / SHORT_WORD_OUTPUT_LOCK_NAME
            lock_contents = "pid=external\n"
            lock_path.write_text(lock_contents, encoding="utf-8")

            with self.assertRaisesRegex(ErrorArtifactError, "locked by another build"):
                run_short_word_build(
                    event_path,
                    output_dir,
                    candidate_lambdas=("0.05", "0.1"),
                    low_snrs=("0", "5"),
                    context_window=2,
                )
            self.assertEqual(lock_path.read_text(encoding="utf-8"), lock_contents)
            lock_path.unlink()

            with mock.patch.object(
                csv.DictWriter,
                "writerows",
                autospec=True,
                side_effect=RuntimeError("short CSV write failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "short CSV write failed"):
                    run_short_word_build(
                        event_path,
                        output_dir,
                        candidate_lambdas=("0.05", "0.1"),
                        low_snrs=("0", "5"),
                        context_window=2,
                    )
            csv_path = output_dir / SHORT_WORD_CSV_NAME
            self.assertFalse(csv_path.exists())
            self.assertFalse((output_dir / f".{SHORT_WORD_CSV_NAME}.tmp").exists())
            self.assertFalse((output_dir / SHORT_WORD_OUTPUT_LOCK_NAME).exists())

            old_csv = b"previous short-word csv\n"
            csv_path.write_bytes(old_csv)
            original_replace = Path.replace
            failed = False

            def replace_with_commit_failure(source: Path, target: str | Path) -> Path:
                nonlocal failed
                if (
                    not failed
                    and source.name == f".{SHORT_WORD_CSV_NAME}.tmp"
                    and Path(target) == csv_path
                ):
                    failed = True
                    raise OSError("injected short-word commit failure")
                return original_replace(source, target)

            with mock.patch.object(
                Path,
                "replace",
                autospec=True,
                side_effect=replace_with_commit_failure,
            ):
                with self.assertRaisesRegex(OSError, "injected short-word commit failure"):
                    run_short_word_build(
                        event_path,
                        output_dir,
                        candidate_lambdas=("0.05", "0.1"),
                        low_snrs=("0", "5"),
                        context_window=2,
                        overwrite=True,
                    )

            self.assertTrue(failed)
            self.assertEqual(csv_path.read_bytes(), old_csv)
            self.assertFalse((output_dir / f".{SHORT_WORD_CSV_NAME}.tmp").exists())
            self.assertFalse((output_dir / f".{SHORT_WORD_CSV_NAME}.bak").exists())
            self.assertFalse((output_dir / SHORT_WORD_OUTPUT_LOCK_NAME).exists())

    def test_overall_only_supports_explicit_ordinary_and_two_lambda_focus_runs(self) -> None:
        focus_runs = (
            "ordinary_lora:0",
            "tone_aware_lora:0.05",
            "tone_aware_lora:0.1",
        )
        self.assertEqual(
            canonical_focus_runs(focus_runs),
            (
                ("ordinary_lora", "0"),
                ("tone_aware_lora", "0.05"),
                ("tone_aware_lora", "0.1"),
            ),
        )
        tone_rows = [row for row in fixture_rows() if row["snr"] == "clean"]
        coda_rows = [row for row in coda_fixture_rows() if row["snr"] == "clean"]
        short_rows = []
        for train_type, lambda_value in canonical_focus_runs(focus_runs):
            short_rows.append(
                short_word_event_row(
                    utt_id=f"{train_type}-{lambda_value}-clean-deletion",
                    train_type=train_type,
                    **{"lambda": lambda_value},
                    operation="deletion",
                    ref_token="đã",
                    hyp_token="",
                    ref_index="0",
                    hyp_index="",
                    ref="đã",
                    hyp="",
                    short_word_deletion="true",
                )
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tone_events = root / "tone_events.csv"
            coda_events = root / "coda_events.csv"
            short_events = root / "short_events.csv"
            write_events(tone_events, tone_rows)
            write_events(coda_events, coda_rows)
            write_events(short_events, short_rows)

            tone_result = run_build(
                tone_events,
                root / "tone",
                focus_runs=focus_runs,
                low_snrs=(),
                overall_only=True,
            )
            coda_result = run_coda_build(
                coda_events,
                root / "coda",
                focus_runs=focus_runs,
                low_snrs=(),
                overall_only=True,
            )
            short_result = run_short_word_build(
                short_events,
                root / "short",
                focus_runs=focus_runs,
                low_snrs=(),
                context_window=1,
                overall_only=True,
            )

            self.assertEqual(tone_result.aggregation.low_snrs, ())
            self.assertEqual(coda_result.aggregation.low_snrs, ())
            self.assertEqual(short_result.aggregation.low_snrs, ())
            self.assertEqual(len(tone_result.candidate_runs), 3)
            self.assertEqual(len(coda_result.candidate_runs), 3)
            self.assertEqual(len(short_result.candidate_runs), 3)
            self.assertEqual(tone_result.matrix_rows, 3 * len(TONE_ORDER) ** 2)
            self.assertEqual(coda_result.matrix_rows, 3 * len(CODA_ORDER) ** 2)
            self.assertEqual(short_result.example_rows, 3)
            _, tone_csv = read_csv(tone_result.csv_path)
            _, coda_csv = read_csv(coda_result.csv_path)
            _, short_csv = read_csv(short_result.csv_path)
            self.assertEqual({row["group_value"] for row in tone_csv}, {"overall"})
            self.assertEqual({row["group_value"] for row in coda_csv}, {"overall"})
            self.assertTrue(all(row["low_snr_scope"] == "false" for row in short_csv))
            self.assertTrue(tone_result.png_path.is_file())
            self.assertTrue(coda_result.png_path.is_file())

    def test_overall_only_cli_focus_is_explicit_and_conflicts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_path = root / "events.csv"
            output_dir = root / "output"
            write_events(event_path, [row for row in fixture_rows() if row["snr"] == "clean"])
            command = [
                sys.executable,
                "-X",
                "utf8",
                str(SCRIPT),
                "--events",
                str(event_path),
                "--overall-only",
                "--focus-run",
                "ordinary_lora:0",
                "--focus-run",
                "tone_aware_lora:0.05",
                "--focus-run",
                "tone_aware_lora:0.1",
                "--out-dir",
                str(output_dir),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("PASS scope: overall-only", completed.stdout)
            self.assertIn("focus=ordinary_lora:0 scope=overall", completed.stdout)
            _, rows = read_csv(output_dir / "tone_confusion_matrix.csv")
            self.assertEqual({row["group_value"] for row in rows}, {"overall"})

            conflicting_scope = subprocess.run(
                [*command[:-2], "--low-snr", "0", *command[-2:]],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(conflicting_scope.returncode, 2)
            self.assertIn("cannot be combined", conflicting_scope.stderr)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_path = root / "events.csv"
            write_events(event_path, [row for row in fixture_rows() if row["snr"] == "clean"])
            with self.assertRaisesRegex(ErrorArtifactError, "either --focus-run"):
                run_build(
                    event_path,
                    root / "output",
                    candidate_lambdas=("0.05", "0.1"),
                    focus_runs=("ordinary_lora:0",),
                    low_snrs=(),
                    overall_only=True,
                )


if __name__ == "__main__":
    unittest.main()
