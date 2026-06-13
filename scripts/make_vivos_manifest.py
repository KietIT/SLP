import argparse
import json
from pathlib import Path


def read_prompts(path: Path):
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def write_split(root: Path, split: str, out_dir: Path):
    # VIVOS commonly uses train/prompts.txt, test/prompts.txt and wav subdirs.
    split_dir = root / split
    if not split_dir.exists():
        # tolerate nested vivos/vivos layout
        candidates = list(root.rglob(split))
        split_dir = candidates[0] if candidates else split_dir
    prompts_path = split_dir / "prompts.txt"
    if not prompts_path.exists():
        print(f"warning: missing {prompts_path}; skipping {split}")
        return None
    prompts = read_prompts(prompts_path)
    out = out_dir / ("dev.jsonl" if split == "test" else f"{split}.jsonl")
    wavs = list((split_dir / "waves").rglob("*.wav")) if (split_dir / "waves").exists() else list(split_dir.rglob("*.wav"))
    with out.open("w", encoding="utf-8") as f:
        for wav in sorted(wavs):
            utt_id = wav.stem
            text = prompts.get(utt_id)
            if text:
                f.write(json.dumps({
                    "audio": str(wav),
                    "text": text,
                    "utt_id": utt_id,
                    "dataset": "vivos",
                    "split": split,
                    "snr": "clean",
                }, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vivos_root", default="data/raw/vivos")
    p.add_argument("--out_dir", default="data/manifests/vivos")
    args = p.parse_args()
    root = Path(args.vivos_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Try to find extracted directory containing train/test.
    if not (root / "train").exists():
        matches = [p for p in root.rglob("train") if p.is_dir()]
        if matches:
            root = matches[0].parent
    write_split(root, "train", out_dir)
    write_split(root, "test", out_dir)
    # duplicate dev to test initially; replace by clean/noisy official split if needed
    test = out_dir / "dev.jsonl"
    if test.exists():
        (out_dir / "test.jsonl").write_text(test.read_text(encoding="utf-8"), encoding="utf-8")

if __name__ == "__main__":
    main()
