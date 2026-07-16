from __future__ import annotations

import random
from copy import deepcopy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.vitonesr.phat.config import (
    load_experiment_config,
    validate_experiment_config,
)
from src.vitonesr.phat.evaluation import _verify_configured_noisy_dev
from src.vitonesr.phat.protocol import evaluation_contract_sha256, sha256_file
from src.vitonesr.phat.reproducibility import set_global_seed


ROOT = Path(__file__).resolve().parents[2]


class ConfigAndSeedTests(unittest.TestCase):
    def test_all_five_lambda_configs_load(self) -> None:
        expected = {
            "lambda_0.yaml": 0.0,
            "lambda_005.yaml": 0.05,
            "lambda_01.yaml": 0.1,
            "lambda_03.yaml": 0.3,
            "lambda_05.yaml": 0.5,
        }
        output_dirs: set[str] = set()
        for filename, lambda_value in expected.items():
            with self.subTest(filename=filename):
                config = load_experiment_config(ROOT / "configs" / "phat" / filename)
                self.assertEqual(float(config["training"]["lambda_tone"]), lambda_value)
                output_dirs.add(str(config["training"]["output_dir"]))
        self.assertEqual(len(output_dirs), 5)

    def test_global_seed_is_repeatable(self) -> None:
        set_global_seed(123, deterministic=True)
        first = (random.random(), float(np.random.random()), torch.rand(3))
        set_global_seed(123, deterministic=True)
        second = (random.random(), float(np.random.random()), torch.rand(3))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_non_finite_lambda_is_rejected(self) -> None:
        config = load_experiment_config(
            ROOT / "configs" / "phat" / "lambda_005.yaml"
        )
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                changed = deepcopy(config)
                changed["training"]["lambda_tone"] = invalid
                with self.assertRaisesRegex(ValueError, "finite"):
                    validate_experiment_config(changed)

    def test_noisy_dev_config_is_hash_bound_and_fail_closed(self) -> None:
        config = load_experiment_config(
            ROOT / "configs" / "phat" / "lambda_005.yaml"
        )
        evaluation = config["evaluation"]
        self.assertEqual(evaluation["benchmark_protocol"], "noisy_dev")
        self.assertEqual(int(evaluation["expected_total_rows"]), 14125)
        self.assertEqual(
            config["selection"]["expected_evaluation_contract_sha256"],
            evaluation_contract_sha256(config),
        )
        for field in (
            "noisy_dev_lock",
            "expected_noisy_dev_lock_sha256",
            "expected_noise_split_lock_sha256",
            "expected_source_dev_sha256",
        ):
            with self.subTest(field=field):
                changed = deepcopy(config)
                changed["evaluation"].pop(field)
                with self.assertRaisesRegex(ValueError, "noisy-dev|SHA-256"):
                    validate_experiment_config(changed)

        changed = deepcopy(config)
        changed["evaluation"]["benchmark_protocol"] = "locked_vivos"
        with self.assertRaisesRegex(ValueError, "Noisy-dev lock fields"):
            validate_experiment_config(changed)

    def test_noisy_dev_runtime_binds_split_noise_manifest_and_method(self) -> None:
        config = load_experiment_config(
            ROOT / "configs" / "phat" / "lambda_005.yaml"
        )
        evaluation = config["evaluation"]
        protocol = config["protocol"]
        split_lock = {
            "splits": {
                "dev": {
                    "manifest_sha256": evaluation["expected_source_dev_sha256"]
                }
            }
        }
        method_integrity = {
            "protocol_split_lock_sha256": sha256_file(
                ROOT / str(protocol["split_lock"])
            ),
            "noise_split_lock_sha256": evaluation[
                "expected_noise_split_lock_sha256"
            ],
            "noisy_dev_lock_sha256": evaluation[
                "expected_noisy_dev_lock_sha256"
            ],
            "noisy_dev_manifest_sha256": evaluation[
                "expected_manifest_sha256"
            ],
        }
        verified = {
            "lock_sha256": evaluation["expected_noisy_dev_lock_sha256"],
            "manifest_path": ROOT / str(evaluation["manifest"]),
            "manifest_sha256": evaluation["expected_manifest_sha256"],
            "rows": evaluation["expected_total_rows"],
            "audio_hashes_verified": False,
        }
        with patch(
            "src.vitonesr.phat.evaluation.verify_noisy_dev_lock",
            return_value=verified,
        ) as verifier:
            integrity = _verify_configured_noisy_dev(
                config,
                split_lock=split_lock,
                method_integrity=method_integrity,
            )
        self.assertTrue(integrity["audio_hashes_verified"])
        self.assertFalse(verifier.call_args.kwargs["verify_audio"])

        stale_method = dict(method_integrity)
        stale_method["noisy_dev_manifest_sha256"] = "f" * 64
        with (
            patch(
                "src.vitonesr.phat.evaluation.verify_noisy_dev_lock",
                return_value=verified,
            ),
            self.assertRaisesRegex(ValueError, "method lock"),
        ):
            _verify_configured_noisy_dev(
                config,
                split_lock=split_lock,
                method_integrity=stale_method,
            )

    def test_canonical_clean_dev_contract_remains_supported(self) -> None:
        config = load_experiment_config(
            ROOT / "configs" / "phat" / "lambda_005.yaml"
        )
        evaluation = config["evaluation"]
        for field in (
            "noisy_dev_lock",
            "expected_noisy_dev_lock_sha256",
            "expected_noise_split_lock_sha256",
            "expected_source_dev_sha256",
        ):
            evaluation.pop(field)
        evaluation.update(
            {
                "benchmark_protocol": "locked_vivos",
                "manifest": "data/manifests/paper_v2/vivos_dev.jsonl",
                "expected_manifest_sha256": (
                    "3dc1afaaf4aedcaf5e5f472d93bb718df5776630e2e04db59e32f8fd8ef1af79"
                ),
                "expected_total_rows": 2825,
            }
        )
        config["selection"]["expected_manifest_sha256"] = evaluation[
            "expected_manifest_sha256"
        ]
        config["selection"]["expected_evaluation_contract_sha256"] = (
            evaluation_contract_sha256(config)
        )
        validate_experiment_config(config)


if __name__ == "__main__":
    unittest.main()
