from __future__ import annotations

import random
import unittest
from pathlib import Path

import numpy as np
import torch

from src.vitonesr.phat.config import load_experiment_config
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


if __name__ == "__main__":
    unittest.main()
