from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def md_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No data available._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        values = [str(row.get(col, "")).replace("\n", " ").strip() for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def disagreement_score(row: dict) -> int:
    ref = normalize(row.get("text", ""))
    hyp = normalize(row.get("prediction", ""))
    if ref == hyp:
        return 0
    ref_words = set(ref.split())
    hyp_words = set(hyp.split())
    return len(ref_words.symmetric_difference(hyp_words)) + abs(len(ref) - len(hyp))


def select_examples(rows: list[dict], limit: int = 3) -> list[dict]:
    candidates = [row for row in rows if normalize(row.get("text", "")) != normalize(row.get("prediction", ""))]
    candidates.sort(key=disagreement_score, reverse=True)
    return candidates[:limit]


def find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def pct(value: str | float) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return str(value)


def metric_row(rows: list[dict], group: str) -> dict | None:
    return next((row for row in rows if row.get("group") == group), None)


def comparison_table(base_rows: list[dict], forced_rows: list[dict], label: str) -> str:
    clean_base = metric_row(base_rows, "all")
    clean_forced = metric_row(forced_rows, "all")
    if not clean_base or not clean_forced:
        return "_No comparison data available._"
    return md_table(
        [{
            "set": label,
            "metric": "WER",
            "before": pct(clean_base["wer"]),
            "forced_vi": pct(clean_forced["wer"]),
            "delta_pp": f"{(float(clean_forced['wer']) - float(clean_base['wer'])) * 100:+.2f}",
        }, {
            "set": label,
            "metric": "CER",
            "before": pct(clean_base["cer"]),
            "forced_vi": pct(clean_forced["cer"]),
            "delta_pp": f"{(float(clean_forced['cer']) - float(clean_base['cer'])) * 100:+.2f}",
        }, {
            "set": label,
            "metric": "TER simple",
            "before": pct(clean_base["ter_simple"]),
            "forced_vi": pct(clean_forced["ter_simple"]),
            "delta_pp": f"{(float(clean_forced['ter_simple']) - float(clean_base['ter_simple'])) * 100:+.2f}",
        }, {
            "set": label,
            "metric": "DER simple",
            "before": pct(clean_base["der_simple"]),
            "forced_vi": pct(clean_forced["der_simple"]),
            "delta_pp": f"{(float(clean_forced['der_simple']) - float(clean_base['der_simple'])) * 100:+.2f}",
        }],
        ["set", "metric", "before", "forced_vi", "delta_pp"],
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Build a Markdown summary from midterm CSV outputs.")
    p.add_argument("--outputs_dir", default="outputs")
    p.add_argument("--model_tag", default="openai_whisper-base")
    p.add_argument("--out", default="outputs/midterm_summary.md")
    args = p.parse_args()

    outputs = Path(args.outputs_dir)
    dataset_stats = read_csv(outputs / "dataset_stats.csv")

    base_clean_metrics_path = find_first_existing([
        outputs / "metrics_whisper_clean.csv",
        outputs / "metrics_clean.csv",
    ])
    base_noisy_metrics_path = find_first_existing([
        outputs / "metrics_whisper_noisy_by_snr.csv",
        outputs / "metrics_noisy_by_snr.csv",
    ])
    clean_metrics_path = find_first_existing([
        outputs / "metrics_whisper_clean_forced.csv",
        outputs / f"metrics_{args.model_tag}_clean.csv",
        outputs / "metrics_whisper_clean.csv",
        outputs / "metrics_clean.csv",
    ])
    noisy_metrics_path = find_first_existing([
        outputs / "metrics_whisper_noisy_forced_by_snr.csv",
        outputs / f"metrics_{args.model_tag}_noisy_by_snr.csv",
        outputs / "metrics_whisper_noisy_by_snr.csv",
        outputs / "metrics_noisy_by_snr.csv",
    ])
    noisy_predictions_path = find_first_existing([
        outputs / "whisper_noisy_forced.csv",
        outputs / f"{args.model_tag}_noisy.csv",
        outputs / "whisper_noisy.csv",
    ])

    base_clean_metrics = read_csv(base_clean_metrics_path) if base_clean_metrics_path else []
    base_noisy_metrics = read_csv(base_noisy_metrics_path) if base_noisy_metrics_path else []
    clean_metrics = read_csv(clean_metrics_path) if clean_metrics_path else []
    noisy_metrics = read_csv(noisy_metrics_path) if noisy_metrics_path else []
    noisy_predictions = read_csv(noisy_predictions_path) if noisy_predictions_path else []
    examples = select_examples(noisy_predictions, limit=3)

    lines = [
        "# Midterm Result Summary",
        "",
        "## Safe Midterm Claim",
        "",
        "We completed a reproducible Vietnamese noisy-ASR pipeline with clean/noisy manifests, controlled SNR noise injection, zero-shot Whisper-base inference, forced Vietnamese decoding, and prototype Vietnamese-specific metrics. LoRA and tone-aware MTL training are prepared as the next Colab step.",
        "",
        "## Dataset / Noise Statistics",
        "",
        md_table(dataset_stats, ["manifest", "snr", "utterances", "hours", "avg_seconds"]),
        "",
        "## Forced Vietnamese Decoding Comparison",
        "",
        "The inference script now passes `language=vi` and `task=transcribe` into `model.generate()`. This removes occasional non-Vietnamese decoding, but only slightly improves aggregate error rates, so model adaptation is still needed.",
        "",
        comparison_table(base_clean_metrics, clean_metrics, "Clean"),
        "",
        comparison_table(base_noisy_metrics, noisy_metrics, "Noisy all"),
        "",
        "## Clean Baseline Metrics (Forced VI)",
        "",
        md_table(clean_metrics, ["group", "n", "wer", "cer", "ter_simple", "der_simple"]),
        "",
        "## Noisy Baseline Metrics By SNR (Forced VI)",
        "",
        md_table(noisy_metrics, ["group", "n", "wer", "cer", "ter_simple", "der_simple"]),
        "",
        "## Qualitative Error Examples",
        "",
    ]

    if examples:
        for i, row in enumerate(examples, 1):
            lines.extend([
                f"### Example {i}",
                "",
                f"- SNR: `{row.get('snr', '')}`",
                f"- Noise type: `{row.get('noise_type', '')}`",
                f"- Reference: {row.get('text', '')}",
                f"- Prediction: {row.get('prediction', '')}",
                "",
            ])
    else:
        lines.append("_No mismatched prediction examples found yet._")
        lines.append("")

    lines.extend([
        "## Slide Mapping",
        "",
        "- Dataset slide: use `outputs/dataset_stats.csv`.",
        "- Result slide: use the forced-VI noisy metrics table above.",
        "- Error analysis slide: use the qualitative examples above.",
        "- Next-step slide: compare Noisy LoRA vs Tone-aware MTL on Colab after midterm.",
        "- Keep the before/after decoding comparison as a small source-code fix, not a final model improvement claim.",
        "",
        "## Wording To Avoid",
        "",
        "- Do not claim tone-aware MTL improves results until trained-model metrics exist.",
        "- If using demo noise instead of MUSAN, call it a controlled synthetic noise subset.",
    ])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
