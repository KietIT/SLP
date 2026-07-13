from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.zero_shot import ZeroShotConfig, ZeroShotInferencer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Run zero-shot Whisper/PhoWhisper inference over the robust benchmark CSV.")
    p.add_argument("--benchmark_manifest", required=True)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model_size", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--language", default="vi")
    p.add_argument("--task", default="transcribe")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_audio_seconds", type=float, default=30.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    config = ZeroShotConfig(
        benchmark_manifest=Path(args.benchmark_manifest),
        model_name_or_path=args.model_name_or_path,
        model=args.model,
        model_size=args.model_size,
        out=Path(args.out),
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        device=args.device,
        language=args.language,
        task=args.task,
        max_new_tokens=args.max_new_tokens,
        resume=args.resume,
        overwrite=args.overwrite,
        max_audio_seconds=args.max_audio_seconds,
    )
    result = ZeroShotInferencer(config).run()
    print(result)


if __name__ == "__main__":
    main()
