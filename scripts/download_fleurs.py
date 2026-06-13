"""Download/prepare Vietnamese FLEURS manifests via Hugging Face Datasets."""
import argparse
import json
from pathlib import Path
from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/manifests/fleurs")
    p.add_argument("--cache_dir", default="data/raw/hf_cache")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation", "test"]:
        ds = load_dataset("google/fleurs", "vi_vn", split=split, cache_dir=args.cache_dir)
        out_path = out_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in ds:
                item = {
                    "audio": row["audio"]["path"],
                    "text": row["transcription"],
                    "dataset": "fleurs",
                    "split": split,
                    "snr": "clean",
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
