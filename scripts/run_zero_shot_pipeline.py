from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vitonesr.prediction import ZERO_SHOT_MODEL_SPECS, selected_model_specs  # noqa: E402


PYTHON = sys.executable


def run_step(name: str, args: list[str]) -> None:
    print(f"\n==> {name}", flush=True)
    command = [PYTHON, *args]
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the robust benchmark and zero-shot baseline pipeline.")
    p.add_argument("--vivos_root", default="data/raw/vivos")
    p.add_argument("--musan_root", default="data/raw/musan/musan")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pool_size", type=int, default=500)
    p.add_argument("--eval_size", type=int, default=300)
    p.add_argument("--snrs", type=float, nargs="+", default=[20, 10, 5, 0])
    p.add_argument("--models", nargs="+", choices=sorted(ZERO_SHOT_MODEL_SPECS), default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()

    benchmark_dir = Path("outputs/benchmark_smoke" if args.smoke_test else "outputs/benchmark")
    pred_dir = Path("outputs/zero_shot_smoke" if args.smoke_test else "outputs/zero_shot")
    noisy_dir = Path("data/noisy_eval_smoke" if args.smoke_test else "data/noisy_eval")
    noise_manifest = Path("data/manifests/noise/musan_noise_typed_smoke.jsonl" if args.smoke_test else "data/manifests/noise/musan_noise_typed.jsonl")
    vivos_manifest = Path("data/manifests/vivos/test.jsonl")
    expected_rows = args.eval_size * (1 + len(args.snrs))

    if not vivos_manifest.exists():
        run_step(
            "Create VIVOS manifests",
            [
                "scripts/make_vivos_manifest.py",
                "--vivos_root",
                args.vivos_root,
                "--out_dir",
                "data/manifests/vivos",
            ],
        )

    run_step(
        "Create typed MUSAN noise manifest",
        [
            "scripts/make_musan_noise_manifest_typed.py",
            "--musan_root",
            args.musan_root,
            "--out",
            str(noise_manifest),
            "--seed",
            str(args.seed),
        ],
    )

    run_step(
        "Build robust benchmark",
        [
            "scripts/build_robust_benchmark.py",
            "--vivos_manifest",
            str(vivos_manifest),
            "--noise_manifest",
            str(noise_manifest),
            "--out_manifest",
            str(benchmark_dir / "benchmark_manifest.csv"),
            "--pool_manifest",
            str(benchmark_dir / "benchmark_pool_manifest.csv"),
            "--report_out",
            str(benchmark_dir / "benchmark_report.md"),
            "--out_noisy_dir",
            str(noisy_dir),
            "--pool_size",
            str(args.pool_size),
            "--eval_size",
            str(args.eval_size),
            "--snrs",
            *[str(snr) for snr in args.snrs],
            "--seed",
            str(args.seed),
            "--sample_rate",
            str(args.sample_rate),
        ],
    )

    specs = selected_model_specs(args.models)
    for key, spec in specs.items():
        run_step(
            f"Run zero-shot inference for {key}",
            [
                "scripts/infer_zero_shot.py",
                "--benchmark_manifest",
                str(benchmark_dir / "benchmark_manifest.csv"),
                "--model_name_or_path",
                spec["model_name_or_path"],
                "--model",
                spec["model"],
                "--model_size",
                spec["model_size"],
                "--out",
                str(pred_dir / spec["filename"]),
                "--sample_rate",
                str(args.sample_rate),
                "--batch_size",
                str(args.batch_size),
                "--device",
                args.device,
                "--language",
                "vi",
                "--task",
                "transcribe",
                "--max_new_tokens",
                "128",
                "--resume",
            ],
        )

    aggregate_args = [
        "scripts/aggregate_zero_shot.py",
        "--pred_dir",
        str(pred_dir),
        "--out_by_snr",
        str(pred_dir / "zero_shot_results_by_snr.csv"),
        "--out_by_noise_type",
        str(pred_dir / "zero_shot_results_by_noise_type.csv"),
        "--expected_rows",
        str(expected_rows),
    ]
    if args.smoke_test:
        aggregate_args.append("--smoke_test")
    if args.models:
        aggregate_args.extend(["--models", *args.models])
    run_step("Aggregate zero-shot results", aggregate_args)

    validate_args = [
        "scripts/validate_robust_benchmark.py",
        "--benchmark_manifest",
        str(benchmark_dir / "benchmark_manifest.csv"),
        "--pool_manifest",
        str(benchmark_dir / "benchmark_pool_manifest.csv"),
        "--pred_dir",
        str(pred_dir),
        "--expected_eval_size",
        str(args.eval_size),
        "--expected_pool_size",
        str(args.pool_size),
        "--snrs",
        *[str(snr) for snr in args.snrs],
    ]
    if args.models:
        validate_args.extend(["--models", *args.models])
    run_step("Validate robust benchmark outputs", validate_args)

    print(f"\nDone. Main result: {pred_dir / 'zero_shot_results_by_snr.csv'}", flush=True)


if __name__ == "__main__":
    main()
