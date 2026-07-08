from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.noise_manifest import (  # noqa: E402
    build_musan_noise_manifest,
    generate_babble_noise,
    write_noise_manifest,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Create a typed MUSAN noise manifest.")
    p.add_argument("--musan_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_music", action="store_true")
    p.add_argument("--no_noise", action="store_true")
    p.add_argument("--no_speech", action="store_true")
    p.add_argument("--babble_out_dir", default=None)
    p.add_argument("--num_babble_files", type=int, default=200)
    p.add_argument("--min_speakers", type=int, default=3)
    p.add_argument("--max_speakers", type=int, default=6)
    p.add_argument("--babble_duration_seconds", type=float, default=15.0)
    p.add_argument("--sample_rate", type=int, default=16000)
    args = p.parse_args()

    rows = build_musan_noise_manifest(
        musan_root=args.musan_root,
        out_path=args.out,
        include_music=not args.no_music,
        include_noise=not args.no_noise,
        include_speech=not args.no_speech,
        seed=args.seed,
    )

    if args.babble_out_dir:
        speech_items = [row for row in rows if row.get("noise_type") == "speech"]
        babble_rows = generate_babble_noise(
            speech_items=speech_items,
            out_dir=args.babble_out_dir,
            num_files=args.num_babble_files,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            duration_seconds=args.babble_duration_seconds,
            sample_rate=args.sample_rate,
            seed=args.seed,
        )
        rows.extend(babble_rows)
        write_noise_manifest(args.out, rows)

    counts = Counter(row.get("noise_type", "unknown") for row in rows)
    print(f"wrote {len(rows)} noise rows to {args.out}")
    for noise_type, count in sorted(counts.items()):
        print(f"{noise_type}: {count}")


if __name__ == "__main__":
    main()
