from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.train_all_lambdas import _latest_resumable_checkpoint


class TrainAllLambdasResumeTests(unittest.TestCase):
    def test_latest_checkpoint_is_sorted_by_numeric_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            for step in (9, 10, 900, 1000):
                (output_dir / f"checkpoint_step_{step}").mkdir()

            latest = _latest_resumable_checkpoint(output_dir)

            self.assertEqual(latest, output_dir / "checkpoint_step_1000")

    def test_malformed_checkpoint_directory_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "checkpoint_step_latest").mkdir()

            with self.assertRaisesRegex(ValueError, "Malformed resumable"):
                _latest_resumable_checkpoint(output_dir)

    def test_checkpoint_shaped_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "checkpoint_step_10").write_text("not a directory")

            with self.assertRaisesRegex(ValueError, "expected a directory"):
                _latest_resumable_checkpoint(output_dir)

    def test_missing_output_directory_has_no_resume_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "missing"
            self.assertIsNone(_latest_resumable_checkpoint(output_dir))


if __name__ == "__main__":
    unittest.main()
