from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import soundfile as sf


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def duration_seconds(path: str) -> float:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def summarize(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = str(row.get("snr", "clean"))
        try:
            dur = duration_seconds(row["audio"])
        except Exception:
            dur = float(row.get("duration", 0.0) or 0.0)
        groups[key].append(dur)

    out = []
    for key, durations in sorted(groups.items(), key=lambda kv: kv[0]):
        total = sum(durations)
        n = len(durations)
        out.append({
            "manifest": str(path),
            "snr": key,
            "utterances": n,
            "hours": round(total / 3600.0, 4),
            "avg_seconds": round(total / max(n, 1), 2),
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize ASR manifest size and duration by SNR.")
    p.add_argument("manifests", nargs="+")
    p.add_argument("--out", default=None, help="Optional CSV output path.")
    args = p.parse_args()

    rows = []
    for manifest in args.manifests:
        rows.extend(summarize(Path(manifest)))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["manifest", "snr", "utterances", "hours", "avg_seconds"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out}")
    else:
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
