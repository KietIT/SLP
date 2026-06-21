from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.metrics import compute_all


def main() -> None:
    p = argparse.ArgumentParser(description="Score prediction CSV with WER/CER/TER/DER/FCER/SWDR, optionally grouped by a metadata column.")
    p.add_argument("--pred", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--group_by", default="snr")
    args = p.parse_args()

    with Path(args.pred).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Prediction CSV is empty.")

    groups: dict[str, list[dict]] = defaultdict(list)
    groups["all"] = rows
    if args.group_by:
        for row in rows:
            groups[str(row.get(args.group_by, "unknown"))].append(row)

    summary = []
    for group, items in groups.items():
        refs = [x.get("text", "") for x in items]
        hyps = [x.get("prediction", "") for x in items]
        metrics = compute_all(refs, hyps)
        summary.append({"group": group, "n": len(items), **metrics})

    fieldnames = ["group", "n", "wer", "cer", "ter_simple", "der_simple", "fcer_simple", "swdr_simple"]
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary)
        print(f"wrote {out}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
