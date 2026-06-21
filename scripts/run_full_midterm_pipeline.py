from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

CLEAN_LIMIT = 30
NOISY_CLEAN_LIMIT = 50
NOISY_INFER_LIMIT = 120
TRAINING_CONFIG = "configs/phowhisper_base_lora.yaml"
CHECKPOINT_DIR = "experiments/phowhisper_base_lora_mtl"


def run_step(name: str, args: list[str]) -> None:
    print(f"\n==> {name}", flush=True)
    command = [PYTHON, *args]
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def require_path(path: str, message: str) -> None:
    if not (ROOT / path).exists():
        raise FileNotFoundError(f"{message} Missing path: {path}")


def find_noise_root() -> str:
    candidates = [
        "data/raw/musan/musan/noise",
        "data/raw/musan/noise",
    ]
    for candidate in candidates:
        if (ROOT / candidate).exists():
            return candidate
    raise FileNotFoundError("MUSAN noise data is required before running the pipeline.")


def require_vivos_data() -> None:
    if not list((ROOT / "data/raw/vivos").rglob("prompts.txt")):
        raise FileNotFoundError(
            "VIVOS data is required before running the pipeline. "
            "Expected extracted prompts.txt files under data/raw/vivos."
        )


def ensure_dirs() -> None:
    for path in [
        "data/raw/vivos",
        "data/raw/musan",
        "data/manifests/noise",
        "data/manifests/vivos",
        "outputs",
        "experiments",
    ]:
        (ROOT / path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"Using Python: {PYTHON}", flush=True)
    ensure_dirs()
    require_vivos_data()
    noise_root = find_noise_root()

    run_step(
        "Check Python environment",
        [
            "-c",
            "import sys; print(sys.executable); import torch, soundfile, yaml; "
            "print('torch_cuda_available=', torch.cuda.is_available())",
        ],
    )

    run_step(
        "Create VIVOS manifests",
        [
            "scripts/make_vivos_manifest.py",
            "--vivos_root",
            "data/raw/vivos",
            "--out_dir",
            "data/manifests/vivos",
        ],
    )

    run_step(
        "Create MUSAN noise manifest",
        [
            "scripts/make_noise_manifest.py",
            "--noise_root",
            noise_root,
            "--out",
            "data/manifests/noise/musan_noise.jsonl",
        ],
    )

    run_step(
        "Create fixed noisy VIVOS test set",
        [
            "scripts/make_noisy_test.py",
            "--manifest",
            "data/manifests/vivos/test.jsonl",
            "--noise_manifest",
            "data/manifests/noise/musan_noise.jsonl",
            "--out_manifest",
            "data/manifests/vivos/test_noisy.jsonl",
            "--limit",
            str(NOISY_CLEAN_LIMIT),
            "--snrs",
            "20",
            "10",
            "5",
            "0",
            "--seed",
            "42",
        ],
    )

    run_step(
        "Run Whisper-base clean baseline",
        [
            "scripts/infer.py",
            "--manifest",
            "data/manifests/vivos/test.jsonl",
            "--model",
            "openai/whisper-base",
            "--out",
            "outputs/whisper_clean.csv",
            "--limit",
            str(CLEAN_LIMIT),
            "--language",
            "vi",
            "--task",
            "transcribe",
        ],
    )

    run_step(
        "Run Whisper-base noisy baseline",
        [
            "scripts/infer.py",
            "--manifest",
            "data/manifests/vivos/test_noisy.jsonl",
            "--model",
            "openai/whisper-base",
            "--out",
            "outputs/whisper_noisy.csv",
            "--limit",
            str(NOISY_INFER_LIMIT),
            "--language",
            "vi",
            "--task",
            "transcribe",
        ],
    )

    run_step(
        "Run PhoWhisper-base clean baseline",
        [
            "scripts/infer.py",
            "--manifest",
            "data/manifests/vivos/test.jsonl",
            "--model",
            "vinai/PhoWhisper-base",
            "--out",
            "outputs/phowhisper_clean.csv",
            "--limit",
            str(CLEAN_LIMIT),
            "--language",
            "vi",
            "--task",
            "transcribe",
        ],
    )

    run_step(
        "Run PhoWhisper-base noisy baseline",
        [
            "scripts/infer.py",
            "--manifest",
            "data/manifests/vivos/test_noisy.jsonl",
            "--model",
            "vinai/PhoWhisper-base",
            "--out",
            "outputs/phowhisper_noisy.csv",
            "--limit",
            str(NOISY_INFER_LIMIT),
            "--language",
            "vi",
            "--task",
            "transcribe",
        ],
    )

    for pred, out, group_by in [
        ("outputs/whisper_clean.csv", "outputs/metrics_whisper_clean.csv", None),
        ("outputs/whisper_noisy.csv", "outputs/metrics_whisper_noisy_by_snr.csv", "snr"),
        ("outputs/phowhisper_clean.csv", "outputs/metrics_phowhisper_clean.csv", None),
        ("outputs/phowhisper_noisy.csv", "outputs/metrics_phowhisper_noisy_by_snr.csv", "snr"),
    ]:
        args = ["scripts/score_predictions.py", "--pred", pred, "--out", out]
        if group_by is not None:
            args.extend(["--group_by", group_by])
        run_step(f"Score {pred}", args)

    run_step("Train PhoWhisper tone-aware LoRA", ["train.py", "--config", TRAINING_CONFIG])

    run_step(
        "Evaluate PhoWhisper tone-aware LoRA on clean set",
        [
            "evaluate.py",
            "--config",
            TRAINING_CONFIG,
            "--checkpoint",
            CHECKPOINT_DIR,
            "--split",
            "test",
            "--out",
            "outputs/phowhisper_lora_clean.csv",
            "--limit",
            str(CLEAN_LIMIT),
        ],
    )

    run_step(
        "Evaluate PhoWhisper tone-aware LoRA on noisy set",
        [
            "evaluate.py",
            "--config",
            TRAINING_CONFIG,
            "--checkpoint",
            CHECKPOINT_DIR,
            "--split",
            "test_noisy",
            "--out",
            "outputs/phowhisper_lora_noisy.csv",
            "--limit",
            str(NOISY_INFER_LIMIT),
        ],
    )

    for pred, out, group_by in [
        ("outputs/phowhisper_lora_clean.csv", "outputs/metrics_phowhisper_lora_clean.csv", None),
        ("outputs/phowhisper_lora_noisy.csv", "outputs/metrics_phowhisper_lora_noisy_by_snr.csv", "snr"),
    ]:
        args = ["scripts/score_predictions.py", "--pred", pred, "--out", out]
        if group_by is not None:
            args.extend(["--group_by", group_by])
        run_step(f"Score {pred}", args)

    run_step(
        "Build final model comparison table",
        [
            "scripts/build_model_comparison.py",
            "--outputs_dir",
            "outputs",
            "--out",
            "outputs/model_comparison_6metrics.csv",
        ],
    )

    print("\nDone. Main output: outputs/model_comparison_6metrics.csv", flush=True)


if __name__ == "__main__":
    main()
