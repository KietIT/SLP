from __future__ import annotations

import dataclasses
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import numpy as np
import soundfile as sf

from src.vitonesr.demo_inference import DemoConfig, DemoEngine, DemoInferenceError


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "demo.yaml"
METRICS = ("wer", "cer", "ter", "der", "fcer", "swdr")


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return value


def _field(value: Any, *names: str) -> Any:
    plain = _plain(value)
    if not isinstance(plain, Mapping):
        raise AssertionError(f"Expected mapping-like value, found {type(value).__name__}")
    for name in names:
        if name in plain:
            return plain[name]
    raise AssertionError(f"Missing all fields {names!r}; found {sorted(plain)}")


def _row_field(row: Any, name: str) -> Any:
    plain = _plain(row)
    if not isinstance(plain, Mapping):
        raise AssertionError(f"Demo row is not mapping-like: {type(row).__name__}")
    if name in plain:
        return plain[name]
    for container in ("metrics", "errors", "error_counts"):
        nested = _plain(plain.get(container, {}))
        if isinstance(nested, Mapping) and name in nested:
            return nested[name]
    raise AssertionError(f"Demo row does not expose {name!r}: {sorted(plain)}")


def _role_id(role: Any) -> str:
    if isinstance(role, str):
        return role
    plain = _plain(role)
    if isinstance(plain, Mapping):
        for key in ("id", "role", "role_id"):
            if key in plain:
                return str(plain[key])
    raise AssertionError(f"Cannot determine role id from {role!r}")


class DemoAudioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DemoConfig.from_yaml(CONFIG_PATH)
        # Predictor injection is a hard boundary: constructing a unit-test engine
        # must not import Transformers or load any checkpoint.
        self.engine = DemoEngine(self.config, predictor=lambda *_args: {})

    def test_prepare_audio_downmixes_resamples_and_clips_to_protocol_limit(self) -> None:
        source_rate = 8_000
        seconds = 16.25
        t = np.arange(int(source_rate * seconds), dtype=np.float32) / source_rate
        stereo = np.column_stack(
            (
                0.10 * np.sin(2 * np.pi * 220 * t),
                0.06 * np.sin(2 * np.pi * 330 * t),
            )
        ).astype(np.float32)

        def fake_resample(
            waveform: np.ndarray, *, orig_sr: int, target_sr: int
        ) -> np.ndarray:
            size = int(round(waveform.size * target_sr / orig_sr))
            old_axis = np.linspace(0.0, 1.0, waveform.size, endpoint=False)
            new_axis = np.linspace(0.0, 1.0, size, endpoint=False)
            return np.interp(new_axis, old_axis, waveform).astype(np.float32)

        fake_librosa = types.SimpleNamespace(resample=fake_resample)
        with patch.dict(sys.modules, {"librosa": fake_librosa}):
            prepared = self.engine.prepare_audio(stereo, sample_rate=source_rate)

        waveform = np.asarray(_field(prepared, "waveform", "audio"))
        sample_rate = int(_field(prepared, "sample_rate"))
        self.assertEqual(sample_rate, 16_000)
        self.assertEqual(waveform.dtype, np.float32)
        self.assertEqual(waveform.ndim, 1)
        self.assertEqual(waveform.size, 15 * 16_000)
        self.assertTrue(np.all(np.isfinite(waveform)))
        self.assertTrue(bool(_field(prepared, "truncated")))

    def test_synthetic_noise_is_deterministic_and_close_to_requested_snr(self) -> None:
        sample_rate = 16_000
        t = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
        clean = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

        noise = np.random.default_rng(11).normal(0.0, 0.05, sample_rate * 3)
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            noise_path = Path(temporary) / "noise.wav"
            sf.write(noise_path, noise.astype(np.float32), sample_rate)
            with patch(
                "src.vitonesr.demo_inference.read_audio",
                return_value=noise.astype(np.float32),
            ):
                mixed_a, metadata_a = self.engine.apply_noise(
                    clean, "fan", 10, noise_path=noise_path, seed=2026
                )
                mixed_b, metadata_b = self.engine.apply_noise(
                    clean, "fan", 10, noise_path=noise_path, seed=2026
                )

        np.testing.assert_array_equal(mixed_a, mixed_b)
        self.assertEqual(_plain(metadata_a), _plain(metadata_b))
        self.assertEqual(mixed_a.shape, clean.shape)
        self.assertEqual(mixed_a.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(mixed_a)))
        self.assertLessEqual(float(np.max(np.abs(mixed_a))), 1.000001)
        self.assertFalse(np.array_equal(mixed_a, clean))

        # The chosen low-amplitude signal does not trigger peak normalization,
        # so the residual is the added noise and its measured SNR is testable.
        residual = mixed_a.astype(np.float64) - clean.astype(np.float64)
        measured_snr = 10.0 * np.log10(
            np.mean(clean.astype(np.float64) ** 2) / np.mean(residual**2)
        )
        self.assertAlmostEqual(measured_snr, 10.0, delta=0.25)
        self.assertEqual(_field(metadata_a, "noise_type"), "fan")
        self.assertEqual(float(_field(metadata_a, "snr", "snr_db")), 10.0)

    def test_noise_validation_rejects_unknown_type_and_missing_snr(self) -> None:
        clean = np.ones(1_600, dtype=np.float32) * 0.01
        with self.assertRaises((TypeError, ValueError, DemoInferenceError)):
            self.engine.apply_noise(clean, "unknown", 10)
        with self.assertRaises((TypeError, ValueError)):
            self.engine.apply_noise(clean, "fan", None)  # type: ignore[arg-type]

    def test_prepare_audio_rejects_empty_and_nonfinite_arrays(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            self.engine.prepare_audio(np.array([], dtype=np.float32), sample_rate=16_000)
        with self.assertRaises((TypeError, ValueError)):
            self.engine.prepare_audio(
                np.array([0.0, np.nan], dtype=np.float32), sample_rate=16_000
            )


class DemoInferenceTests(unittest.TestCase):
    def test_fake_predictor_runs_three_roles_and_reuses_aligned_metrics(self) -> None:
        calls: list[tuple[np.ndarray, int, list[str]]] = []

        def fake_predictor(
            waveform: np.ndarray,
            sample_rate: int,
            roles: Sequence[Any],
        ) -> dict[str, Any]:
            role_ids = [_role_id(role) for role in roles]
            calls.append((waveform.copy(), sample_rate, role_ids))
            return {
                "ordinary_baseline": {
                    "transcript": "đã có một",
                    "latency_seconds": 0.10,
                },
                "selected_method": {
                    "transcript": "đã có",
                    "latency_seconds": 0.11,
                },
                "locked_control": {
                    "transcript": "đã có một và",
                    "latency_seconds": 0.12,
                },
            }

        engine = DemoEngine(DemoConfig.from_yaml(CONFIG_PATH), predictor=fake_predictor)
        waveform = np.zeros(3_200, dtype=np.float32)
        result = engine.run(
            waveform,
            reference="đã có một",
            noise_type="none",
            snr=None,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 16_000)
        self.assertEqual(
            calls[0][2],
            ["ordinary_baseline", "selected_method", "locked_control"],
        )
        rows = list(_field(result, "rows", "results"))
        self.assertEqual(len(rows), 3)
        by_role = {str(_row_field(row, "role")): row for row in rows}
        self.assertEqual(set(by_role), set(calls[0][2]))

        exact = by_role["ordinary_baseline"]
        deletion = by_role["selected_method"]
        insertion = by_role["locked_control"]
        self.assertEqual(_row_field(exact, "transcript"), "đã có một")
        exact_metrics = _plain(_field(exact, "metrics"))
        deletion_metrics = _plain(_field(deletion, "metrics"))
        insertion_metrics = _plain(_field(insertion, "metrics"))
        for metric in METRICS:
            self.assertEqual(float(exact_metrics[metric]), 0.0)
        self.assertAlmostEqual(float(deletion_metrics["wer"]), 1 / 3)
        self.assertEqual(int(_row_field(deletion, "deletion")), 1)
        self.assertEqual(int(_row_field(deletion, "substitution")), 0)
        self.assertEqual(int(_row_field(deletion, "insertion")), 0)
        self.assertAlmostEqual(float(insertion_metrics["wer"]), 1 / 3)
        self.assertEqual(int(_row_field(insertion, "insertion")), 1)

    def test_reference_is_optional_and_does_not_fabricate_metrics(self) -> None:
        def fake_predictor(
            _waveform: np.ndarray,
            _sample_rate: int,
            roles: Sequence[Any],
        ) -> dict[str, str]:
            return {_role_id(role): "xin chào" for role in roles}

        engine = DemoEngine(DemoConfig.from_yaml(CONFIG_PATH), predictor=fake_predictor)
        result = engine.run(
            np.zeros(1_600, dtype=np.float32),
            reference=None,
            noise_type="none",
            snr=None,
        )
        rows = list(_field(result, "rows", "results"))
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(_row_field(row, "transcript"), "xin chào")
            plain = _plain(row)
            for metric in METRICS:
                if metric in plain:
                    self.assertIsNone(plain[metric])
                else:
                    nested = _plain(plain.get("metrics"))
                    self.assertTrue(
                        nested is None
                        or metric not in nested
                        or nested[metric] is None
                    )


if __name__ == "__main__":
    unittest.main()
