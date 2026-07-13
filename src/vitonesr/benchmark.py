from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .noise import fit_noise, mix_at_snr, read_audio
from .prediction import (
    BENCHMARK_COLUMNS,
    POOL_COLUMNS,
    PREDICTION_COLUMNS,
    ZERO_SHOT_MODEL_SPECS,
    atomic_write_csv,
    normalize_snr,
    prediction_path_for,
    read_csv_rows,
    read_jsonl,
    selected_model_specs,
    validate_columns,
)


def stable_seed(master_seed: int, utt_id: str, condition: str) -> int:
    raw = f"{master_seed}|{utt_id}|{condition}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value[:120] or "utt"


def snr_label(snr: float | int | str) -> str:
    if str(snr).lower() == "clean":
        return "clean"
    value = float(snr)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


@dataclass
class BenchmarkConfig:
    vivos_manifest: Path
    noise_manifest: Path
    out_manifest: Path
    pool_manifest: Path
    report_out: Path
    out_noisy_dir: Path
    pool_size: int = 500
    eval_size: int = 300
    snrs: tuple[float, ...] = (20, 10, 5, 0)
    seed: int = 42
    sample_rate: int = 16000


class RobustBenchmarkBuilder:
    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def build(self) -> dict:
        clean_items = self._load_clean_items()
        if len(clean_items) < self.config.pool_size:
            raise ValueError(
                f"VIVOS manifest has {len(clean_items)} usable rows, "
                f"but pool_size={self.config.pool_size}"
            )
        if self.config.eval_size > self.config.pool_size:
            raise ValueError("eval_size must be less than or equal to pool_size")

        rng = random.Random(self.config.seed)
        sorted_items = sorted(clean_items, key=lambda item: item["source_utt_id"])
        rng.shuffle(sorted_items)
        pool_items = sorted_items[: self.config.pool_size]
        eval_items = pool_items[: self.config.eval_size]
        noise_items = self._load_noise_items()

        pool_rows = [self._pool_row(item, rank) for rank, item in enumerate(pool_items, 1)]
        benchmark_rows = self._build_benchmark_rows(eval_items, noise_items)
        self.validate(benchmark_rows)

        atomic_write_csv(self.config.pool_manifest, pool_rows, POOL_COLUMNS)
        atomic_write_csv(self.config.out_manifest, benchmark_rows, BENCHMARK_COLUMNS)
        self._write_report(benchmark_rows, pool_rows, noise_items, validation_status="PASS")

        return {
            "pool_rows": len(pool_rows),
            "benchmark_rows": len(benchmark_rows),
            "out_manifest": str(self.config.out_manifest),
            "pool_manifest": str(self.config.pool_manifest),
            "report_out": str(self.config.report_out),
        }

    def validate(self, rows: list[dict]) -> None:
        expected_total = self.config.eval_size * (1 + len(self.config.snrs))
        errors: list[str] = []
        if len(rows) != expected_total:
            errors.append(f"benchmark rows: expected {expected_total}, found {len(rows)}")

        utt_ids = [row.get("utt_id", "") for row in rows]
        if len(set(utt_ids)) != len(utt_ids):
            errors.append("utt_id values must be unique")

        condition_counts = Counter(row.get("condition", "") for row in rows)
        if condition_counts.get("clean", 0) != self.config.eval_size:
            errors.append(f"clean rows: expected {self.config.eval_size}, found {condition_counts.get('clean', 0)}")
        expected_noisy = self.config.eval_size * len(self.config.snrs)
        if condition_counts.get("noisy", 0) != expected_noisy:
            errors.append(f"noisy rows: expected {expected_noisy}, found {condition_counts.get('noisy', 0)}")

        snr_counts = Counter(normalize_snr(row.get("snr", "")) for row in rows)
        if snr_counts.get("clean", 0) != self.config.eval_size:
            errors.append(f"clean SNR rows: expected {self.config.eval_size}, found {snr_counts.get('clean', 0)}")
        for snr in self.config.snrs:
            label = snr_label(snr)
            if snr_counts.get(label, 0) != self.config.eval_size:
                errors.append(f"SNR {label} rows: expected {self.config.eval_size}, found {snr_counts.get(label, 0)}")

        for row in rows:
            audio_path = Path(row.get("audio_path", ""))
            if not audio_path.exists():
                errors.append(f"audio_path does not exist: {audio_path}")
                break
            for name in ("transcript", "duration", "seed", "source_utt_id"):
                if str(row.get(name, "")).strip() == "":
                    errors.append(f"missing {name} for row {row.get('utt_id', '')}")
                    break

        if errors:
            raise ValueError("Benchmark validation failed: " + "; ".join(errors))

    def _load_clean_items(self) -> list[dict]:
        raw_rows = read_jsonl(self.config.vivos_manifest)
        rows: list[dict] = []
        for row in raw_rows:
            audio = row.get("audio") or row.get("clean_path")
            text = row.get("text") or row.get("transcript")
            if not audio or not text:
                continue
            source_utt_id = str(row.get("utt_id") or Path(audio).stem)
            rows.append({
                "source_utt_id": source_utt_id,
                "dataset": row.get("dataset", "vivos"),
                "split": row.get("split", "test"),
                "clean_path": str(audio),
                "transcript": str(text),
                "duration": self._duration_seconds(Path(audio)),
            })
        return rows

    def _load_noise_items(self) -> list[dict]:
        rows = read_jsonl(self.config.noise_manifest)
        rows = [row for row in rows if row.get("audio")]
        if not rows:
            raise ValueError("Noise manifest is empty.")
        return rows

    def _pool_row(self, item: dict, rank: int) -> dict:
        return {
            "source_utt_id": item["source_utt_id"],
            "dataset": item["dataset"],
            "split": item["split"],
            "clean_path": item["clean_path"],
            "transcript": item["transcript"],
            "duration": f"{float(item['duration']):.6f}",
            "seed": self.config.seed,
            "pool_rank": rank,
        }

    def _build_benchmark_rows(self, eval_items: Sequence[dict], noise_items: Sequence[dict]) -> list[dict]:
        rows: list[dict] = []
        for item in eval_items:
            source_utt_id = item["source_utt_id"]
            clean_seed = stable_seed(self.config.seed, source_utt_id, "clean")
            rows.append({
                "utt_id": f"{source_utt_id}_clean",
                "dataset": item["dataset"],
                "split": item["split"],
                "condition": "clean",
                "clean_path": item["clean_path"],
                "noisy_path": "",
                "audio_path": item["clean_path"],
                "snr": "clean",
                "noise_type": "clean",
                "noise_path": "",
                "transcript": item["transcript"],
                "duration": f"{float(item['duration']):.6f}",
                "seed": clean_seed,
                "source_utt_id": source_utt_id,
            })

            clean = read_audio(item["clean_path"], sr=self.config.sample_rate)
            for snr in self.config.snrs:
                label = snr_label(snr)
                item_seed = stable_seed(self.config.seed, source_utt_id, f"snr{label}")
                item_rng = random.Random(item_seed)
                noise_item = item_rng.choice(list(noise_items))
                noise = read_audio(noise_item["audio"], sr=self.config.sample_rate)
                fitted = fit_noise(noise, len(clean), item_rng)
                mixed = mix_at_snr(clean, fitted, float(snr))

                snr_dir = self.config.out_noisy_dir / f"snr_{label}"
                snr_dir.mkdir(parents=True, exist_ok=True)
                noisy_path = snr_dir / f"{safe_stem(source_utt_id)}_snr{label}.wav"
                import soundfile as sf

                sf.write(noisy_path, mixed, self.config.sample_rate)

                rows.append({
                    "utt_id": f"{source_utt_id}_snr{label}",
                    "dataset": item["dataset"],
                    "split": item["split"],
                    "condition": "noisy",
                    "clean_path": item["clean_path"],
                    "noisy_path": str(noisy_path),
                    "audio_path": str(noisy_path),
                    "snr": label,
                    "noise_type": noise_item.get("noise_type", Path(noise_item["audio"]).parent.name),
                    "noise_path": noise_item["audio"],
                    "transcript": item["transcript"],
                    "duration": f"{float(item['duration']):.6f}",
                    "seed": item_seed,
                    "source_utt_id": source_utt_id,
                })
        return rows

    def _duration_seconds(self, path: Path) -> float:
        if not path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {path}")
        try:
            import soundfile as sf

            info = sf.info(path)
            return float(info.frames) / float(info.samplerate)
        except Exception:
            wav = read_audio(str(path), sr=self.config.sample_rate)
            return len(wav) / float(self.config.sample_rate)

    def _write_report(
        self,
        benchmark_rows: Sequence[dict],
        pool_rows: Sequence[dict],
        noise_items: Sequence[dict],
        validation_status: str,
    ) -> None:
        counts = self._counts_table(benchmark_rows)
        noise_distribution = self._noise_distribution_table(benchmark_rows)
        duration_summary = self._duration_summary_table(benchmark_rows)
        available_noise_types = sorted({str(row.get("noise_type", "")) for row in noise_items if row.get("noise_type")})
        missing_native = [name for name in ("music", "noise", "speech") if name not in available_noise_types]

        notes = []
        if "babble" not in available_noise_types:
            notes.append("Babble was not generated for this benchmark.")
        if missing_native:
            notes.append("Missing native MUSAN noise types: " + ", ".join(missing_native) + ".")
        if not notes:
            notes.append("All native MUSAN noise types were available.")

        lines = [
            "# Robust Benchmark Report",
            "",
            "## Configuration",
            f"- VIVOS manifest: `{self.config.vivos_manifest}`",
            f"- Noise manifest: `{self.config.noise_manifest}`",
            f"- Seed: `{self.config.seed}`",
            f"- Pool size: `{self.config.pool_size}`",
            f"- Eval size: `{self.config.eval_size}`",
            f"- SNR levels: `{', '.join(snr_label(snr) for snr in self.config.snrs)}`",
            f"- Sample rate: `{self.config.sample_rate}`",
            "",
            "## Counts",
            self._md_table(counts, ["condition", "snr", "count"]),
            "",
            "## Noise Type Distribution",
            self._md_table(noise_distribution, ["snr", "noise_type", "count"]),
            "",
            "## Duration Summary",
            self._md_table(duration_summary, ["condition", "snr", "total_hours", "avg_seconds", "min_seconds", "max_seconds"]),
            "",
            "## Manifest Schema",
            "- Pool columns: `" + ", ".join(POOL_COLUMNS) + "`",
            "- Benchmark columns: `" + ", ".join(BENCHMARK_COLUMNS) + "`",
            "",
            "## Reproducibility",
            "Rows are selected by sorting VIVOS utterance IDs, shuffling with the master seed, taking the pool, then taking the eval subset. Each noisy sample uses a stable SHA-256 seed derived from the master seed, source utterance ID, and SNR condition.",
            "",
            "## Validation Status",
            validation_status,
            "",
            "## Notes",
        ]
        lines.extend(f"- {note}" for note in notes)
        lines.extend([
            "",
            "## Output Files",
            f"- Pool manifest: `{self.config.pool_manifest}` ({len(pool_rows)} rows)",
            f"- Benchmark manifest: `{self.config.out_manifest}` ({len(benchmark_rows)} rows)",
            f"- Noisy audio directory: `{self.config.out_noisy_dir}`",
        ])

        self.config.report_out.parent.mkdir(parents=True, exist_ok=True)
        self.config.report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _counts_table(self, rows: Sequence[dict]) -> list[dict]:
        counter = Counter((row.get("condition", ""), normalize_snr(row.get("snr", ""))) for row in rows)
        output = [{"condition": "clean", "snr": "clean", "count": counter.get(("clean", "clean"), 0)}]
        for snr in self.config.snrs:
            label = snr_label(snr)
            output.append({"condition": "noisy", "snr": label, "count": counter.get(("noisy", label), 0)})
        return output

    def _noise_distribution_table(self, rows: Sequence[dict]) -> list[dict]:
        counter = Counter((normalize_snr(row.get("snr", "")), row.get("noise_type", "")) for row in rows)
        return [
            {"snr": snr, "noise_type": noise_type, "count": count}
            for (snr, noise_type), count in sorted(counter.items(), key=lambda item: (item[0][0], item[0][1]))
        ]

    def _duration_summary_table(self, rows: Sequence[dict]) -> list[dict]:
        groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            groups[(row.get("condition", ""), normalize_snr(row.get("snr", "")))].append(float(row.get("duration", 0.0)))
        output = []
        for (condition, snr), values in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
            total = sum(values)
            output.append({
                "condition": condition,
                "snr": snr,
                "total_hours": f"{total / 3600.0:.6f}",
                "avg_seconds": f"{total / max(len(values), 1):.3f}",
                "min_seconds": f"{min(values):.3f}",
                "max_seconds": f"{max(values):.3f}",
            })
        return output

    def _md_table(self, rows: Sequence[dict], columns: Sequence[str]) -> str:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = []
        for row in rows:
            body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
        return "\n".join([header, sep, *body])


def validate_robust_benchmark_files(
    benchmark_manifest: str | Path,
    pool_manifest: str | Path,
    pred_dir: str | Path | None,
    expected_eval_size: int = 300,
    expected_pool_size: int = 500,
    snrs: Sequence[float] = (20, 10, 5, 0),
    model_keys: Sequence[str] | None = None,
    require_results: bool = True,
) -> dict:
    benchmark_manifest = Path(benchmark_manifest)
    pool_manifest = Path(pool_manifest)
    pred_dir = Path(pred_dir) if pred_dir is not None else None

    pool_rows, pool_columns = read_csv_rows(pool_manifest)
    validate_columns(pool_manifest, pool_columns, POOL_COLUMNS)
    if len(pool_rows) != expected_pool_size:
        raise ValueError(f"{pool_manifest} has {len(pool_rows)} rows, expected {expected_pool_size}")

    benchmark_rows, benchmark_columns = read_csv_rows(benchmark_manifest)
    validate_columns(benchmark_manifest, benchmark_columns, BENCHMARK_COLUMNS)
    expected_total = expected_eval_size * (1 + len(snrs))
    if len(benchmark_rows) != expected_total:
        raise ValueError(f"{benchmark_manifest} has {len(benchmark_rows)} rows, expected {expected_total}")

    condition_counts = Counter(row.get("condition", "") for row in benchmark_rows)
    if condition_counts.get("clean", 0) != expected_eval_size:
        raise ValueError(f"clean rows: expected {expected_eval_size}, found {condition_counts.get('clean', 0)}")
    if condition_counts.get("noisy", 0) != expected_eval_size * len(snrs):
        raise ValueError(f"noisy rows: expected {expected_eval_size * len(snrs)}, found {condition_counts.get('noisy', 0)}")

    snr_counts = Counter(normalize_snr(row.get("snr", "")) for row in benchmark_rows)
    for snr in ("clean", *[snr_label(value) for value in snrs]):
        if snr_counts.get(snr, 0) != expected_eval_size:
            raise ValueError(f"SNR {snr}: expected {expected_eval_size}, found {snr_counts.get(snr, 0)}")

    for row in benchmark_rows:
        if not Path(row.get("audio_path", "")).exists():
            raise FileNotFoundError(f"Missing audio_path: {row.get('audio_path', '')}")
        for name in ("transcript", "duration", "seed"):
            if str(row.get(name, "")).strip() == "":
                raise ValueError(f"Missing {name} for {row.get('utt_id', '')}")

    checked_predictions: list[str] = []
    if pred_dir is not None:
        specs = selected_model_specs(model_keys)
        for key, spec in specs.items():
            path = prediction_path_for(pred_dir, spec)
            if not path.exists():
                raise FileNotFoundError(f"Missing prediction file for {key}: {path}")
            rows, columns = read_csv_rows(path)
            validate_columns(path, columns, PREDICTION_COLUMNS, exact=True)
            if len(rows) != expected_total:
                raise ValueError(f"{path} has {len(rows)} rows, expected {expected_total}")
            checked_predictions.append(str(path))
        if require_results:
            result_path = pred_dir / "zero_shot_results_by_snr.csv"
            if not result_path.exists():
                raise FileNotFoundError(f"Missing result file: {result_path}")

    return {
        "pool_rows": len(pool_rows),
        "benchmark_rows": len(benchmark_rows),
        "prediction_files": checked_predictions,
        "model_keys": list(selected_model_specs(model_keys).keys()) if pred_dir is not None else list(ZERO_SHOT_MODEL_SPECS),
    }
