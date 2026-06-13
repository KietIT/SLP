import argparse
import json
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise_root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.noise_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted([x for x in root.rglob("*") if x.suffix.lower() in AUDIO_EXTS])
    with out.open("w", encoding="utf-8") as f:
        for path in paths:
            item = {"audio": str(path), "noise_type": path.parent.name}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(paths)} noise files to {out}")

if __name__ == "__main__":
    main()
