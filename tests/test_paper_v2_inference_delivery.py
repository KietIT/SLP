from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from scripts.verify_paper_v2_inference_delivery import (
    DELIVERY_RECEIPT_VERSION,
    PaperV2DeliveryError,
    build_or_verify_delivery_receipt,
    build_parser,
)
from src.vitonesr.inference_runtime import INFERENCE_RUNTIME_SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]


def _repo_ref(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PaperV2InferenceDeliveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "inference_runtime_lock.json"
        self.runtime_identity = "a" * 64
        self.runtime.write_text(
            json.dumps(
                {
                    "identity_sha256": self.runtime_identity,
                    "schema_version": INFERENCE_RUNTIME_SCHEMA_VERSION,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        self.final_receipt = self.root / "final_lora_receipt.json"
        self.final_receipt.write_text(
            '{"kind":"final-lora"}\n', encoding="utf-8", newline="\n"
        )
        self.fleurs_receipt = self.root / "fleurs_receipt.json"
        self.fleurs_receipt.write_text(
            '{"kind":"fleurs"}\n', encoding="utf-8", newline="\n"
        )
        self.config = self.root / "final_lora.yaml"
        self.config.write_text("formal: true\n", encoding="utf-8", newline="\n")
        self.training = self.root / "training_environment.json"
        self.training.write_text("{}\n", encoding="utf-8", newline="\n")
        self.first = self.root / "z_result.csv"
        self.first.write_bytes(b"metric,value\nwer,1.0\n")
        self.second = self.root / "a_result.csv"
        self.second.write_bytes(b"metric,value\nder,0.5\n")
        self.output = self.root / "trung_delivery_receipt.json"
        self.expected_final_hash = _sha256(self.final_receipt)
        self.expected_fleurs_hash = _sha256(self.fleurs_receipt)
        self.final_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.fleurs_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def final_verifier(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        self.final_calls.append((args, kwargs))
        if not self.final_receipt.is_file() or _sha256(
            self.final_receipt
        ) != self.expected_final_hash:
            raise ValueError("tampered final-LoRA receipt")
        return {
            "aggregate_rows": 3,
            "execution_receipt_sha256": self.expected_final_hash,
            "inference_runtime_lock_sha256": _sha256(self.runtime),
            "inference_runtime_verified": True,
            "prediction_rows_per_role": 2300,
            "roles": ["ordinary", "tone_best", "tone_control"],
            "status": "VERIFIED",
        }

    def fleurs_verifier(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        self.fleurs_calls.append((args, kwargs))
        if not self.fleurs_receipt.is_file() or _sha256(
            self.fleurs_receipt
        ) != self.expected_fleurs_hash:
            raise ValueError("tampered FLEURS receipt")
        return {
            "inference_runtime_identity_sha256": self.runtime_identity,
            "prediction_count": 3,
            "receipt_sha256": self.expected_fleurs_hash,
            "status": "VERIFIED",
        }

    def run_delivery(
        self, *, verify_existing: bool = False, skip_fleurs: bool = False
    ) -> dict[str, Any]:
        return build_or_verify_delivery_receipt(
            artifacts=[_repo_ref(self.first), _repo_ref(self.second)],
            output=_repo_ref(self.output),
            final_lora_receipt=_repo_ref(self.final_receipt),
            final_lora_config=_repo_ref(self.config),
            fleurs_receipt=_repo_ref(self.fleurs_receipt),
            inference_runtime_lock=_repo_ref(self.runtime),
            training_environment_lock=_repo_ref(self.training),
            verify_existing=verify_existing,
            skip_fleurs=skip_fleurs,
            final_lora_verifier=self.final_verifier,
            fleurs_verifier=self.fleurs_verifier,
        )

    def test_writes_complete_canonical_receipt_and_calls_both_verifiers(self) -> None:
        result = self.run_delivery()

        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(result["receipt_status"], "written")
        self.assertEqual(result["artifact_count"], 2)
        self.assertEqual(len(self.final_calls), 1)
        self.assertEqual(len(self.fleurs_calls), 1)
        self.assertIs(self.final_calls[0][1]["verify_current"], True)
        self.assertEqual(
            self.fleurs_calls[0][1]["inference_runtime_lock"],
            _repo_ref(self.runtime),
        )

        value = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(value["schema_version"], DELIVERY_RECEIPT_VERSION)
        self.assertEqual(value["status"], "COMPLETE")
        self.assertIs(value["contract"]["fleurs_required"], True)
        self.assertEqual(
            [item["path"] for item in value["artifacts"]],
            sorted([_repo_ref(self.first), _repo_ref(self.second)]),
        )
        by_path = {item["path"]: item for item in value["artifacts"]}
        self.assertEqual(by_path[_repo_ref(self.first)]["bytes"], self.first.stat().st_size)
        self.assertEqual(by_path[_repo_ref(self.first)]["sha256"], _sha256(self.first))
        self.assertEqual(
            value["inference_runtime"]["identity_sha256"], self.runtime_identity
        )
        expected_text = json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2
        ) + "\n"
        self.assertEqual(self.output.read_text(encoding="utf-8"), expected_text)

    def test_verify_existing_rechecks_everything_and_detects_artifact_change(self) -> None:
        self.run_delivery()
        verified = self.run_delivery(verify_existing=True)
        self.assertEqual(verified["receipt_status"], "verified_existing")
        self.assertEqual(len(self.final_calls), 2)
        self.assertEqual(len(self.fleurs_calls), 2)

        self.first.write_bytes(b"metric,value\nwer,9.9\n")
        with self.assertRaisesRegex(PaperV2DeliveryError, "differs from current"):
            self.run_delivery(verify_existing=True)

    def test_existing_receipt_is_immutable_without_verify_flag(self) -> None:
        self.run_delivery()
        with self.assertRaises(FileExistsError):
            self.run_delivery()

    def test_tampered_final_or_fleurs_receipt_fails_before_delivery(self) -> None:
        self.final_receipt.write_text("{}\n", encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(ValueError, "tampered final-LoRA"):
            self.run_delivery()
        self.assertFalse(self.output.exists())
        self.assertEqual(len(self.fleurs_calls), 0)

        self.final_receipt.write_text(
            '{"kind":"final-lora"}\n', encoding="utf-8", newline="\n"
        )
        self.fleurs_receipt.unlink()
        with self.assertRaisesRegex(ValueError, "tampered FLEURS"):
            self.run_delivery()
        self.assertFalse(self.output.exists())

    def test_skip_fleurs_is_intermediate_only_and_never_writes_receipt(self) -> None:
        result = self.run_delivery(skip_fleurs=True)
        self.assertEqual(result["status"], "INTERMEDIATE_VERIFIED")
        self.assertIsNone(result["delivery_receipt"])
        self.assertEqual(len(self.final_calls), 1)
        self.assertEqual(len(self.fleurs_calls), 0)
        self.assertFalse(self.output.exists())

        with self.assertRaisesRegex(PaperV2DeliveryError, "intermediate"):
            self.run_delivery(skip_fleurs=True, verify_existing=True)

    def test_rejects_missing_duplicate_and_self_artifacts(self) -> None:
        common = {
            "output": _repo_ref(self.output),
            "final_lora_receipt": _repo_ref(self.final_receipt),
            "final_lora_config": _repo_ref(self.config),
            "fleurs_receipt": _repo_ref(self.fleurs_receipt),
            "inference_runtime_lock": _repo_ref(self.runtime),
            "training_environment_lock": _repo_ref(self.training),
            "final_lora_verifier": self.final_verifier,
            "fleurs_verifier": self.fleurs_verifier,
        }
        with self.assertRaisesRegex(PaperV2DeliveryError, "At least one"):
            build_or_verify_delivery_receipt(artifacts=[], **common)
        with self.assertRaisesRegex(PaperV2DeliveryError, "Duplicate"):
            build_or_verify_delivery_receipt(
                artifacts=[_repo_ref(self.first), _repo_ref(self.first)], **common
            )
        with self.assertRaisesRegex(PaperV2DeliveryError, "cannot bind itself"):
            self.output.write_text("temporary\n", encoding="utf-8")
            build_or_verify_delivery_receipt(
                artifacts=[_repo_ref(self.output)], **common
            )

    def test_runtime_identity_mismatch_is_rejected(self) -> None:
        def another_runtime(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
            value = dict(self.fleurs_verifier(*args, **kwargs))
            value["inference_runtime_identity_sha256"] = "b" * 64
            return value

        with self.assertRaisesRegex(PaperV2DeliveryError, "different inference"):
            build_or_verify_delivery_receipt(
                artifacts=[_repo_ref(self.first)],
                output=_repo_ref(self.output),
                final_lora_receipt=_repo_ref(self.final_receipt),
                final_lora_config=_repo_ref(self.config),
                fleurs_receipt=_repo_ref(self.fleurs_receipt),
                inference_runtime_lock=_repo_ref(self.runtime),
                training_environment_lock=_repo_ref(self.training),
                final_lora_verifier=self.final_verifier,
                fleurs_verifier=another_runtime,
            )

    def test_parser_requires_repeatable_artifacts(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["--artifact", "one.csv", "--artifact", "two.csv", "--skip-fleurs"]
        )
        self.assertEqual(args.artifact, ["one.csv", "two.csv"])
        self.assertTrue(args.skip_fleurs)
        with self.assertRaises(SystemExit):
            parser.parse_args([])


if __name__ == "__main__":
    unittest.main()
