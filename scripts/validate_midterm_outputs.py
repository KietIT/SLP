from __future__ import annotations

import argparse
import csv
from pathlib import Path


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def check_file(path: Path, min_rows: int | None = None) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing: {path}"
    if path.stat().st_size == 0:
        return False, f"empty: {path}"
    if min_rows is not None:
        rows = count_csv_rows(path)
        if rows < min_rows:
            return False, f"too few rows: {path} has {rows}, expected >= {min_rows}"
        return True, f"ok: {path} ({rows} rows)"
    return True, f"ok: {path}"


def main() -> None:
    p = argparse.ArgumentParser(description="Validate that midterm pipeline outputs are ready for report/slides.")
    p.add_argument("--outputs_dir", default="outputs")
    p.add_argument("--model_tag", default="openai_whisper-base")
    p.add_argument("--min_clean_predictions", type=int, default=1)
    p.add_argument("--min_noisy_predictions", type=int, default=1)
    args = p.parse_args()

    outputs = Path(args.outputs_dir)
    checks = [
        check_file(outputs / "dataset_stats.csv", min_rows=1),
        check_file(outputs / f"{args.model_tag}_clean.csv", min_rows=args.min_clean_predictions),
        check_file(outputs / f"{args.model_tag}_noisy.csv", min_rows=args.min_noisy_predictions),
        check_file(outputs / f"metrics_{args.model_tag}_clean.csv", min_rows=1),
        check_file(outputs / f"metrics_{args.model_tag}_noisy_by_snr.csv", min_rows=1),
        check_file(outputs / "midterm_summary.md"),
    ]

    ok = True
    for passed, message in checks:
        ok = ok and passed
        print(("[PASS] " if passed else "[FAIL] ") + message)

    if not ok:
        raise SystemExit(1)
    print("Midterm outputs are ready for report/slides.")


if __name__ == "__main__":
    main()
