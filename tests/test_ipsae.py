from __future__ import annotations

import unittest
import json
import os
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from evaluators.af2 import _metrics as compute_af2_metrics
from evaluators.af3 import (
    _backfill_sample_ipsae,
    _deduplicate_samples,
    _refresh_sample_iptm,
    collect_af3_output,
    compute_af3_metrics,
)
from evaluators.ipsae import compute_ipsae_min


class IpSAECalculatorTest(unittest.TestCase):
    def setUp(self) -> None:
        # Target group: indices 0, 1, 3. Binder group: indices 2, 4.
        self.pae = np.full((5, 5), 20.0)
        self.pae[2, [0, 1, 3]] = [1.0, 2.0, 11.0]
        self.pae[4, [0, 1, 3]] = [5.0, 11.0, 11.0]
        self.pae[0, [2, 4]] = [2.0, 11.0]
        self.pae[1, [2, 4]] = [3.0, 4.0]
        self.pae[3, [2, 4]] = [11.0, 11.0]
        self.binder = np.asarray([False, False, True, False, True])
        self.target = ~self.binder

    def test_uses_the_conservative_direction(self) -> None:
        d0 = max(1.0, 1.24 * (27.0 - 15.0) ** (1.0 / 3.0) - 1.8)
        transform = lambda value: 1.0 / (1.0 + (value / d0) ** 2.0)
        binder_to_target = (transform(1.0) + transform(2.0)) / 2.0
        target_to_binder = transform(2.0)
        self.assertGreater(binder_to_target, target_to_binder)
        self.assertAlmostEqual(
            compute_ipsae_min(self.pae, self.binder, self.target),
            target_to_binder,
        )

    def test_is_invariant_to_token_permutation(self) -> None:
        order = np.asarray([4, 0, 3, 2, 1])
        expected = compute_ipsae_min(self.pae, self.binder, self.target)
        observed = compute_ipsae_min(
            self.pae[np.ix_(order, order)],
            self.binder[order],
            self.target[order],
        )
        self.assertAlmostEqual(observed, expected)

    def test_strict_cutoff_and_empty_groups_return_zero(self) -> None:
        pae = np.full((3, 3), 10.0)
        self.assertEqual(
            compute_ipsae_min(pae, [True, False, False], [False, True, True]),
            0.0,
        )
        self.assertEqual(compute_ipsae_min(pae, [False] * 3, [True] * 3), 0.0)

    def test_uses_partner_count_above_the_d0_floor(self) -> None:
        pae = np.full((60, 60), 20.0)
        pae[:30, 30:] = 2.0
        pae[30:, :30] = 2.0
        binder = np.arange(60) < 30
        d0 = 1.24 * (30.0 - 15.0) ** (1.0 / 3.0) - 1.8
        expected = 1.0 / (1.0 + (2.0 / d0) ** 2.0)
        self.assertAlmostEqual(compute_ipsae_min(pae, binder, ~binder), expected)

    def test_rejects_invalid_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "square PAE"):
            compute_ipsae_min(np.zeros((2, 3)), [True, False], [False, True])
        with self.assertRaisesRegex(ValueError, "masks"):
            compute_ipsae_min(np.zeros((2, 2)), [True], [False, True])


class ReevaluatorIpSAETest(unittest.TestCase):
    def setUp(self) -> None:
        self.pae = np.full((5, 5), 20.0)
        self.pae[3:, :3] = [[1.0, 2.0, 11.0], [5.0, 11.0, 11.0]]
        self.pae[:3, 3:] = [[2.0, 11.0], [3.0, 4.0], [11.0, 11.0]]
        self.expected = compute_ipsae_min(
            self.pae,
            [False, False, False, True, True],
            [True, True, True, False, False],
        )

    def test_af2_uses_all_target_chains_as_one_group(self) -> None:
        model = SimpleNamespace(aux={
            "plddt": np.asarray([0.7, 0.7, 0.7, 0.8, 0.9]),
            "log": {"ptm": 0.6, "i_ptm": 0.7},
            "pae": self.pae,
        })
        self.assertAlmostEqual(compute_af2_metrics(model, 2)["ipsae_min"], self.expected)

    def test_af3_supports_multiple_noncontiguous_target_chains(self) -> None:
        # Reorder so target chains A and C, and binder chain Z, are interleaved.
        order = np.asarray([0, 3, 1, 4, 2])
        confidences = {
            "pae": self.pae[np.ix_(order, order)].tolist(),
            "token_chain_ids": ["A", "Z", "C", "Z", "A"],
        }
        metrics = compute_af3_metrics(
            {}, confidences, "Z", target_ids=["A", "C"]
        )
        self.assertAlmostEqual(metrics["ipsae_min"], self.expected)

    def test_af3_uses_the_binder_chain_iptm_for_multichain_targets(self) -> None:
        order = np.asarray([0, 3, 1, 4, 2])
        confidences = {
            "pae": self.pae[np.ix_(order, order)].tolist(),
            "token_chain_ids": ["A", "Z", "C", "Z", "A"],
        }
        metrics = compute_af3_metrics(
            {"iptm": 0.91, "chain_iptm": [0.81, 0.23, 0.72]},
            confidences,
            "Z",
            target_ids=["A", "C"],
        )
        self.assertAlmostEqual(metrics["i_ptm"], 0.23)
        self.assertAlmostEqual(metrics["global_i_ptm"], 0.91)

    def test_af3_falls_back_to_global_iptm_for_older_summaries(self) -> None:
        metrics = compute_af3_metrics(
            {"iptm": 0.71},
            {"token_chain_ids": ["A", "Z"]},
            "Z",
            target_ids=["A"],
        )
        self.assertAlmostEqual(metrics["i_ptm"], 0.71)
        self.assertAlmostEqual(metrics["global_i_ptm"], 0.71)

    def test_af3_rejects_malformed_modern_chain_iptm(self) -> None:
        with self.assertRaisesRegex(ValueError, "chain_iptm"):
            compute_af3_metrics(
                {"iptm": 0.91, "chain_iptm": [0.82]},
                {"token_chain_ids": ["A", "Z"]},
                "Z",
                target_ids=["A"],
            )

    def test_af3_excludes_unselected_chains(self) -> None:
        pae = np.pad(self.pae, ((0, 1), (0, 1)), constant_values=0.1)
        confidences = {
            "pae": pae.tolist(),
            "token_chain_ids": ["A", "A", "C", "Z", "Z", "L"],
        }
        metrics = compute_af3_metrics(
            {}, confidences, "Z", target_ids=["A", "C"]
        )
        self.assertAlmostEqual(metrics["ipsae_min"], self.expected)

    def test_af3_backfills_saved_confidence_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            confidence = run_dir / "confidences.json"
            confidence.write_text(json.dumps({
                "pae": self.pae.tolist(),
                "token_chain_ids": ["A", "C", "A", "Z", "Z"],
            }), encoding="utf-8")
            result = {
                "binder_chain_id": "Z",
                "target_chain_ids": ["A", "C"],
            }
            sample = {"metrics": {}, "confidences": "confidences.json"}
            self.assertTrue(_backfill_sample_ipsae(run_dir, result, sample))
            self.assertAlmostEqual(sample["metrics"]["ipsae_min"], self.expected)
            self.assertFalse(_backfill_sample_ipsae(run_dir, result, sample))

    def test_af3_refreshes_saved_binder_iptm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            summary = run_dir / "summary_confidences.json"
            summary.write_text(json.dumps({
                "iptm": 0.91,
                "chain_iptm": [0.82, 0.24],
            }), encoding="utf-8")
            confidence = run_dir / "confidences.json"
            confidence.write_text(json.dumps({
                "token_chain_ids": ["A", "A", "Z", "Z"],
            }), encoding="utf-8")
            result = {
                "binder_chain_id": "Z",
                "target_chain_ids": ["A"],
            }
            sample = {
                "metrics": {"i_ptm": 0.91},
                "summary_confidences": "summary_confidences.json",
                "confidences": "confidences.json",
            }
            self.assertTrue(_refresh_sample_iptm(run_dir, result, sample))
            self.assertAlmostEqual(sample["metrics"]["i_ptm"], 0.24)
            self.assertAlmostEqual(sample["metrics"]["global_i_ptm"], 0.91)
            self.assertFalse(_refresh_sample_iptm(run_dir, result, sample))

    def test_af3_keeps_only_the_newest_retry_for_each_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.json"
            new = root / "new.json"
            other = root / "other.json"
            for path in (old, new, other):
                path.write_text("{}", encoding="utf-8")
            os.utime(old, (1, 1))
            os.utime(new, (2, 2))
            os.utime(other, (3, 3))
            samples = [
                {"seed": 1, "sample": 0, "summary_confidences": "old.json"},
                {"seed": 1, "sample": 1, "summary_confidences": "other.json"},
                {"seed": 1, "sample": 0, "summary_confidences": "new.json"},
            ]
            observed = _deduplicate_samples(root, samples)
            self.assertEqual(
                [item["summary_confidences"] for item in observed],
                ["other.json", "new.json"],
            )

    def test_af3_does_not_mix_a_partial_retry_with_an_older_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def saved(attempt: str, sample: int, timestamp: int) -> dict:
                path = (
                    root / attempt / f"seed-1_sample-{sample}"
                    / "summary_confidences.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
                os.utime(path, (timestamp, timestamp))
                return {
                    "seed": 1,
                    "sample": sample,
                    "summary_confidences": str(path),
                }

            samples = [
                saved("complete", 0, 1),
                saved("complete", 1, 1),
                saved("retry", 0, 2),
            ]
            observed = _deduplicate_samples(root, samples)
            self.assertEqual(
                [(item["seed"], item["sample"]) for item in observed],
                [(1, 0), (1, 1)],
            )
            self.assertTrue(all("complete" in item["summary_confidences"] for item in observed))

    def test_af3_requires_every_expected_diffusion_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sample_dir = output / "prediction" / "seed-1_sample-0"
            sample_dir.mkdir(parents=True)
            (sample_dir / "summary_confidences.json").write_text(json.dumps({
                "iptm": 0.7,
                "chain_iptm": [0.6, 0.5],
            }), encoding="utf-8")
            (sample_dir / "confidences.json").write_text(json.dumps({
                "pae": [[0.0, 2.0], [2.0, 0.0]],
                "token_chain_ids": ["A", "Z"],
            }), encoding="utf-8")
            (sample_dir / "model.cif").write_text("data_test\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete seed/sample"):
                collect_af3_output(
                    output, "Z", [1], target_ids=["A"], expected_samples=2
                )


if __name__ == "__main__":
    unittest.main()
