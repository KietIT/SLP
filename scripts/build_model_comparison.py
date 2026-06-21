from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_FIELDS = [
    "wer",
    "cer",
    "ter_simple",
    "der_simple",
    "fcer_simple",
    "swdr_simple",
]

MODEL_FILES = [
    ("Whisper-base zero-shot", "whisper"),
    ("PhoWhisper-base zero-shot", "phowhisper"),
    ("PhoWhisper tone-aware LoRA", "phowhisper_lora"),
]

SNR_ORDER = ["20", "10", "5", "0"]


def read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metric file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"group", "n", *METRIC_FIELDS}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return rows


def find_group(rows: list[dict[str, str]], group: str) -> dict[str, str]:
    for row in rows:
        if str(row["group"]) == group:
            return row
    raise ValueError(f"Missing group '{group}'")


def normalize_snr_group(value: str) -> str:
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return str(number).replace(".", "p")


def comparison_row(model: str, condition: str, row: dict[str, str]) -> dict[str, str]:
    return {
        "model": model,
        "condition": condition,
        "n": row["n"],
        **{field: row[field] for field in METRIC_FIELDS},
    }


def add_model_rows(outputs_dir: Path, model_name: str, tag: str) -> list[dict[str, str]]:
    clean_rows = read_metrics(outputs_dir / f"metrics_{tag}_clean.csv")
    noisy_rows = read_metrics(outputs_dir / f"metrics_{tag}_noisy_by_snr.csv")

    rows = [
        comparison_row(model_name, "clean", find_group(clean_rows, "all")),
        comparison_row(model_name, "noisy_all", find_group(noisy_rows, "all")),
    ]
    by_snr = {normalize_snr_group(row["group"]): row for row in noisy_rows if row["group"] != "all"}
    for snr in SNR_ORDER:
        if snr in by_snr:
            rows.append(comparison_row(model_name, f"snr_{snr}", by_snr[snr]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a final 6-metric model comparison CSV.")
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_path = Path(args.out) if args.out else outputs_dir / "model_comparison_6metrics.csv"
    all_rows: list[dict[str, str]] = []
    for model_name, tag in MODEL_FILES:
        all_rows.extend(add_model_rows(outputs_dir, model_name, tag))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "condition", "n", *METRIC_FIELDS]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
