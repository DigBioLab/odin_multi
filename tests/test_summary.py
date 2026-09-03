from __future__ import annotations

import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluators.summary import (
    Case,
    _build_curves,
    _build_scatter,
    _curve_rows,
    _design_case_id,
    _evaluation_identity,
    _figure_stem,
    _rank_candidates,
    summarize_run,
)
from odin_multi import build_parser
from run_layout import RunLayout, layout_for_run


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find(rows: list[dict[str, str]], **wanted: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in wanted.items()):
            return row
    raise AssertionError(f"No row matches {wanted}")


def _trajectory(i_ptm: float, i_pae: float, binder_plddt: float) -> dict:
    pae = [
        [0.0, i_pae - 3.0, i_pae - 1.0],
        [i_pae + 1.0, 0.0, 0.0],
        [i_pae + 3.0, 0.0, 0.0],
    ]
    return {
        "seq": [[0, 1], [0, 1]],
        "xyz": [[], []],
        "plddt": [
            [0.1, 0.1, 0.1],
            [0.5, binder_plddt - 0.1, binder_plddt + 0.1],
        ],
        "pae": [
            [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            pae,
        ],
        "ptm": [0.1, 0.7],
        "i_ptm": [0.1, i_ptm],
        "iteration": [0, 10],
        "stage": ["logits", "hard"],
    }


class SummaryFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "run"
        self.selection_name = "selected_test"
        self.contexts = [
            {"name": "target_a", "role": "target"},
            {"name": "off_a", "role": "offtarget"},
        ]
        self.layout = RunLayout.new(self.run_dir)
        _write_json(
            self.layout.manifest,
            {
                "layout_version": 2,
                "base_seed": 0,
                "requested_designs": 2,
                "contexts": self.contexts,
            },
        )
        selected = []
        for index in range(2):
            design_id = f"design_{index}"
            selected.append({
                "selection_status": "selected",
                "design_index": index,
                "design_id": design_id,
                "iteration": 10,
                "stage": "hard",
                "frame_index": 1,
                "sequence": "AC",
                "length": 2,
                "seed": 100 + index,
            })
            trajectory_path = self.layout.design_trajectory(index, design_id)
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            with trajectory_path.open("wb") as handle:
                pickle.dump({
                    "target_a": _trajectory(
                        0.8 - 0.1 * index, 13.0 + index, 0.9 - 0.05 * index
                    ),
                    "off_a": _trajectory(
                        0.2 + 0.1 * index, 21.0 - index, 0.8 - 0.05 * index
                    ),
                }, handle)
            _write_json(
                self.layout.design_status(index),
                {
                    "status": "complete",
                    "trajectory_pickle": str(trajectory_path.relative_to(self.run_dir)),
                },
            )
        _write_json(
            self.layout.selection_dir(self.selection_name) / "selection.json",
            {
                "status": "complete",
                "selection_name": self.selection_name,
                "selected": 2,
                "rows": selected,
            },
        )
        self._write_af2(selected)
        self._write_af3(selected)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_af2(self, selected: list[dict]) -> None:
        root = self.layout.evaluation_dir("af2", "af2_ensemble")
        _write_json(root / "evaluation.json", {
            "evaluator": "af2",
            "evaluation_name": "af2_ensemble",
            "selection_name": self.selection_name,
            "config": {"models": [0, 1], "seeds": [3, 4], "num_recycles": 3},
        })
        for design in selected:
            index = int(design["design_index"])
            for context in self.contexts:
                base_iptm = (
                    0.6 - 0.1 * index
                    if context["role"] == "target"
                    else 0.1 + 0.1 * index
                )
                base_ipae = 4.0 + index if context["role"] == "target" else 20.0
                for model_offset, model in enumerate((0, 1)):
                    for seed_offset, seed in enumerate((3, 4)):
                        replicate = model_offset * 2 + seed_offset
                        directory = (
                            self.layout.evaluation_jobs("af2", "af2_ensemble")
                            / f"t{index:05d}" / context["name"]
                            / f"model_{model + 1}_seed_{seed}"
                        )
                        _write_json(directory / "result.json", {
                            "status": "complete",
                            "design_index": index,
                            "design_id": design["design_id"],
                            "selection_name": self.selection_name,
                            "selected_iteration": 10,
                            "selected_stage": "hard",
                            "sequence": "AC",
                            "context": context,
                            "model_number": model + 1,
                            "model_name": f"model_{model + 1}",
                            "seed": seed,
                            "num_recycles": 3,
                            "metrics": {
                                "plddt": 0.8 + 0.01 * replicate,
                                "ptm": 0.7,
                                "i_ptm": base_iptm + 0.1 * replicate,
                                "ipsae_min": base_iptm + 0.1 * replicate,
                                "i_pae": base_ipae + replicate,
                            },
                            "structure": "prediction.pdb",
                        })
                        (directory / "prediction.pdb").write_text(
                            f"MODEL {replicate}\n", encoding="utf-8"
                        )

    def _write_af3(self, selected: list[dict]) -> None:
        root = self.layout.evaluation_dir("af3", "af3_ensemble")
        _write_json(root / "evaluation.json", {
            "evaluator": "af3",
            "evaluation_name": "af3_ensemble",
            "selection_name": self.selection_name,
            "config": {
                "seeds": [7],
                "extra_flags": {"num_diffusion_samples": 5},
            },
        })
        for design in selected:
            index = int(design["design_index"])
            for context in self.contexts:
                sample_count = 4 if index == 1 and context["role"] == "offtarget" else 5
                base_iptm = (
                    0.7 - 0.2 * index
                    if context["role"] == "target"
                    else 0.1 + 0.1 * index
                )
                base_ipae = 4.0 + index if context["role"] == "target" else 18.0
                samples = []
                directory = (
                    self.layout.evaluation_jobs("af3", "af3_ensemble")
                    / f"t{index:05d}" / context["name"]
                )
                for sample in range(sample_count):
                    structure = directory / f"sample_{sample}" / "model.cif"
                    structure.parent.mkdir(parents=True, exist_ok=True)
                    structure.write_text(f"data_sample_{sample}\n", encoding="utf-8")
                    samples.append({
                        "seed": 7,
                        "sample": sample,
                        "metrics": {
                            "plddt": 0.88,
                            "binder_plddt": 0.9 - 0.01 * sample,
                            "ptm": 0.75,
                            "i_ptm": base_iptm + 0.05 * sample,
                            "ipsae_min": base_iptm + 0.05 * sample,
                            "i_pae": base_ipae + sample,
                            "min_i_pae": base_ipae - 1.0 + sample,
                            "ranking_score": 0.6,
                        },
                        "structure": str(structure.relative_to(self.run_dir)),
                    })
                _write_json(directory / "result.json", {
                    "status": "complete",
                    "design_index": index,
                    "design_id": design["design_id"],
                    "selection_name": self.selection_name,
                    "selected_iteration": 10,
                    "selected_stage": "hard",
                    "sequence": "AC",
                    "context": context,
                    "samples": samples,
                })
        _write_json(
            self.layout.evaluation_jobs("af3", "af3_ensemble")
            / "t00001" / "off_a_failed" / "failure.json",
            {
                "status": "failed",
                "design_index": 1,
                "design_id": "design_1",
                "context": "off_a",
                "error": "one sample was not produced",
            },
        )

    def test_end_to_end_summary(self) -> None:
        result = summarize_run(self.run_dir)
        self.assertEqual(result, {
            "cases": 3,
            "source_rows": 39,
            "context_rows": 11,
            "design_rows": 6,
            "candidates": 3,
            "candidate_structures": 4,
            "figures": 36,
        })
        summary = self.run_dir / "04_summary"
        for path in (
            summary / "README.md",
            summary / "candidates.csv",
            summary / "candidates.fasta",
            summary / "data" / "replicates.csv",
            summary / "data" / "contexts.csv",
            summary / "data" / "designs.csv",
            summary / "data" / "curves.csv",
            summary / "data" / "scatters.csv",
            summary / "data" / "candidate_structures.csv",
        ):
            self.assertTrue(path.is_file(), path)

        source = _read_csv(summary / "data" / "replicates.csv")
        af3_raw = [row for row in source if row["source"] == "af3_reevaluation"]
        self.assertEqual(len(af3_raw), 19)
        self.assertEqual(
            {row["sample"] for row in af3_raw if row["design_index"] == "0"},
            {"0", "1", "2", "3", "4"},
        )

        contexts = _read_csv(summary / "data" / "contexts.csv")
        original = _find(
            contexts,
            case_id="design_selected_test",
            design_index="0",
            context="target_a",
        )
        self.assertAlmostEqual(float(original["binder_plddt"]), 0.9)
        self.assertAlmostEqual(float(original["i_ptm"]), 0.8)
        self.assertAlmostEqual(float(original["i_pae"]), 13.0)
        self.assertAlmostEqual(float(original["min_i_pae"]), 10.0)
        af2_mean = _find(
            contexts,
            case_id="af2",
            design_index="0",
            context="target_a",
        )
        self.assertEqual(af2_mean["replicate_count"], "4")
        self.assertAlmostEqual(float(af2_mean["i_ptm"]), 0.75)
        self.assertAlmostEqual(float(af2_mean["ipsae_min"]), 0.75)
        af3_mean = _find(
            contexts,
            case_id="af3",
            design_index="0",
            context="target_a",
        )
        self.assertEqual(af3_mean["replicate_count"], "5")
        self.assertAlmostEqual(float(af3_mean["i_ptm"]), 0.8)
        self.assertAlmostEqual(float(af3_mean["ipsae_min"]), 0.8)
        self.assertFalse(any(
            row["case_id"] == "af3"
            and row["design_index"] == "1"
            and row["context"] == "off_a"
            for row in contexts
        ))

        designs = _read_csv(summary / "data" / "designs.csv")
        incomplete = _find(
            designs,
            case_id="af3",
            design_index="1",
        )
        self.assertEqual(incomplete["contexts_observed"], "1")
        self.assertEqual(incomplete["contexts_complete"], "False")
        self.assertEqual(incomplete["candidate_status"], "incomplete_contexts")
        self.assertEqual(incomplete["candidate_rank"], "")
        complete = _find(
            designs,
            case_id="af3",
            design_index="0",
        )
        self.assertAlmostEqual(float(complete["target_min_ipsae_min"]), 0.8)
        self.assertAlmostEqual(float(complete["offtarget_max_ipsae_min"]), 0.2)
        self.assertAlmostEqual(float(complete["target_offtarget_ipsae_min_gap"]), 0.6)
        self.assertEqual(complete["candidate_status"], "ranked")
        self.assertEqual(complete["candidate_rank"], "1")

        below_ratio = _find(
            designs,
            case_id="design_selected_test",
            design_index="1",
        )
        self.assertEqual(
            below_ratio["candidate_status"], "below_specificity_ratio"
        )
        duplicate = _find(
            designs,
            case_id="af2",
            design_index="1",
        )
        self.assertEqual(duplicate["candidate_status"], "duplicate_sequence")
        self.assertEqual(duplicate["duplicate_of_design_id"], "design_0")

        candidates = _read_csv(summary / "candidates.csv")
        self.assertEqual(len(candidates), 3)
        self.assertEqual({row["design_id"] for row in candidates}, {"design_0"})
        self.assertEqual({row["candidate_rank"] for row in candidates}, {"1"})
        fasta = (summary / "candidates.fasta").read_text(encoding="utf-8")
        self.assertEqual(fasta.count(">"), 3)
        self.assertEqual(fasta.count("\nAC\n"), 3)


        curves = _read_csv(summary / "data" / "curves.csv")
        target_curve = _find(
            curves,
            case_id="af3",
            metric="ipsae_min",
            curve_kind="context",
            curve_label="target_a · target",
            threshold="0.5",
        )
        specificity = _find(
            curves,
            case_id="af3",
            metric="i_ptm",
            curve_kind="specificity",
            threshold="1.0",
        )
        self.assertEqual(target_curve["n_designs"], "2")
        self.assertEqual(specificity["n_designs"], "1")
        self.assertFalse(any(
            row["metric"] in {"i_ptm", "i_pae"}
            and row["curve_kind"] != "specificity"
            for row in curves
        ))

        scatters = _read_csv(summary / "data" / "scatters.csv")
        af3_scatter = [
            row for row in scatters
            if row["case_id"] == "af3"
        ]
        self.assertEqual(len(af3_scatter), 3)
        self.assertEqual({row["design_index"] for row in af3_scatter}, {"0"})
        self.assertEqual(
            {row["scatter_kind"] for row in af3_scatter},
            {"specificity_landscape"},
        )

        for case_id in (
            "design_selected_test",
            "af2",
            "af3",
        ):
            for metric in ("i_ptm", "i_pae"):
                for kind in ("specificity", "target_vs_offtarget"):
                    for suffix in ("svg", "png"):
                        self.assertTrue(
                            (
                                summary
                                / "figures"
                                / f"{case_id}_{metric.replace('_', '')}_{kind}.{suffix}"
                            ).is_file()
                        )
        for case_id in (
            "af2",
            "af3",
        ):
            for kind in ("thresholds", "specificity", "target_vs_offtarget"):
                for suffix in ("svg", "png"):
                    self.assertTrue(
                        (
                            summary
                            / "figures"
                            / f"{case_id}_ipsae_min_{kind}.{suffix}"
                        ).is_file()
                    )
        self.assertFalse(any(
            path.is_dir() for path in (summary / "figures").iterdir()
        ))
        structures = _read_csv(summary / "data" / "candidate_structures.csv")
        self.assertEqual(
            sum(row["status"] == "exported" for row in structures), 4
        )
        af3_target = _find(
            structures,
            case_id="af3",
            context="target_a",
        )
        self.assertEqual(af3_target["sample"], "0")
        self.assertTrue((self.run_dir / af3_target["exported_structure"]).is_file())

        report = (summary / "README.md").read_text(encoding="utf-8")
        self.assertIn("AF3 reevaluation", report)
        self.assertNotIn("AF3 reevaluation (af3_ensemble)", report)
        self.assertIn("| 1 | 19 |", report)

        first_curves = (summary / "data" / "curves.csv").read_text(encoding="utf-8")
        summarize_run(self.run_dir)
        self.assertEqual(
            first_curves,
            (summary / "data" / "curves.csv").read_text(encoding="utf-8"),
        )

    def test_cli_and_narrowing_require_a_pair(self) -> None:
        arguments = build_parser().parse_args([
            "summarize", "--run-dir", str(self.run_dir)
        ])
        self.assertIsNone(arguments.evaluator)
        self.assertIsNone(arguments.evaluation_name)
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            summarize_run(self.run_dir, evaluator="af2")

    def test_legacy_run_layout_is_still_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            _write_json(run / "odin_multi_run.json", {"contexts": [{}]})
            layout = layout_for_run(run)
            self.assertTrue(layout.legacy)
            self.assertEqual(layout.summary, run / "Summary")


class SummaryStrategiesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = Case(
            case_id="case",
            label="Case",
            source="af2_design",
            evaluator="af2",
            evaluation_name=None,
            selection_name="selection",
            expected_replicates=1,
            selected_rows=[],
            rows=[],
        )

    def test_public_case_names_are_compact_and_legacy_paths_are_unchanged(self) -> None:
        current = RunLayout.new(Path("/tmp/current_run"))
        legacy = RunLayout(Path("/tmp/legacy_run"), 1)

        self.assertEqual(
            _design_case_id(current, "best_clipped_i_pae_ratio"),
            "design_specificity",
        )
        self.assertEqual(
            _design_case_id(current, "best_i_ptm"), "design_best_i_ptm"
        )
        self.assertEqual(
            _evaluation_identity(current, "af3", "af3_v100", 1),
            ("af3", "AF3 reevaluation"),
        )
        self.assertEqual(
            _evaluation_identity(current, "af3", "alternative", 2),
            ("af3_alternative", "AF3 reevaluation (alternative)"),
        )
        self.assertEqual(
            _figure_stem(current, "af3", "iptm_thresholds"),
            current.figures / "af3_iptm_thresholds",
        )
        self.assertEqual(
            _design_case_id(legacy, "best_i_ptm"),
            "af2_design__best_i_ptm",
        )
        self.assertEqual(
            _figure_stem(legacy, "legacy_case", "iptm_curves"),
            legacy.figures / "legacy_case" / "iptm_curves",
        )

    def test_cross_reactivity_ranks_worst_target_and_collapses_duplicates(self) -> None:
        rows = [
            {
                "case_id": "case", "design_index": 1, "design_id": "d1",
                "sequence": "AA", "contexts_complete": True,
                "target_max_i_pae": 8.0,
                "offtarget_target_i_pae_ratio": None,
            },
            {
                "case_id": "case", "design_index": 2, "design_id": "d2",
                "sequence": "BB", "contexts_complete": True,
                "target_max_i_pae": 5.0,
                "offtarget_target_i_pae_ratio": None,
            },
            {
                "case_id": "case", "design_index": 3, "design_id": "d3",
                "sequence": "AA", "contexts_complete": True,
                "target_max_i_pae": 4.0,
                "offtarget_target_i_pae_ratio": None,
            },
            {
                "case_id": "case", "design_index": 4, "design_id": "d4",
                "sequence": "CC", "contexts_complete": False,
                "target_max_i_pae": 3.0,
                "offtarget_target_i_pae_ratio": None,
            },
        ]
        manifest = {"contexts": [
            {"name": "target_a", "role": "target"},
            {"name": "target_b", "role": "target"},
        ]}

        candidates = _rank_candidates(rows, manifest)

        self.assertEqual(
            [(row["design_id"], row["candidate_rank"]) for row in candidates],
            [("d3", 1), ("d2", 2)],
        )
        self.assertEqual(rows[0]["candidate_status"], "duplicate_sequence")
        self.assertEqual(rows[0]["duplicate_of_design_id"], "d3")
        self.assertEqual(rows[3]["candidate_status"], "incomplete_contexts")


    def test_specificity_ratio_is_inclusive_then_ranks_target_quality(self) -> None:
        rows = [
            {
                "case_id": "case", "design_index": 0, "design_id": "d0",
                "sequence": "AA", "contexts_complete": True,
                "target_max_i_pae": 7.0,
                "offtarget_target_i_pae_ratio": 1.5,
            },
            {
                "case_id": "case", "design_index": 1, "design_id": "d1",
                "sequence": "BB", "contexts_complete": True,
                "target_max_i_pae": 5.0,
                "offtarget_target_i_pae_ratio": 2.0,
            },
            {
                "case_id": "case", "design_index": 2, "design_id": "d2",
                "sequence": "CC", "contexts_complete": True,
                "target_max_i_pae": 4.0,
                "offtarget_target_i_pae_ratio": 1.49,
            },
            {
                "case_id": "case", "design_index": 3, "design_id": "d3",
                "sequence": "DD", "contexts_complete": True,
                "target_max_i_pae": 3.0,
                "offtarget_target_i_pae_ratio": None,
            },
        ]
        manifest = {"contexts": [
            {"name": "target", "role": "target"},
            {"name": "offtarget", "role": "offtarget"},
        ]}

        candidates = _rank_candidates(rows, manifest)

        self.assertEqual(
            [(row["design_id"], row["candidate_rank"]) for row in candidates],
            [("d1", 1), ("d0", 2)],
        )
        self.assertEqual(rows[2]["candidate_status"], "below_specificity_ratio")
        self.assertEqual(rows[3]["candidate_status"], "missing_specificity_ratio")


    def _scatter(
        self,
        targets: list[str],
        context_values: dict[str, tuple[float, float, float, float]],
    ) -> list[dict]:
        context_rows = [
            {
                "case_id": "case",
                "design_index": 0,
                "context": name,
                "role": "target",
                "binder_plddt": values[0],
                "i_ptm": values[1],
                "i_pae": values[2],
                "ipsae_min": values[3],
            }
            for name, values in context_values.items()
        ]
        iptm = [value[1] for value in context_values.values()]
        ipae = [value[2] for value in context_values.values()]
        ipsae = [value[3] for value in context_values.values()]
        design_rows = [{
            "case_id": "case",
            "design_index": 0,
            "design_id": "d0",
            "targets_complete": True,
            "contexts_complete": True,
            "target_min_i_ptm": min(iptm),
            "target_mean_i_ptm": sum(iptm) / len(iptm),
            "target_max_i_pae": max(ipae),
            "target_mean_i_pae": sum(ipae) / len(ipae),
            "target_min_ipsae_min": min(ipsae),
            "target_mean_ipsae_min": sum(ipsae) / len(ipsae),
        }]
        manifest = {
            "contexts": [{"name": name, "role": "target"} for name in targets]
        }
        return _build_scatter([self.case], context_rows, design_rows, manifest)

    def test_scatter_strategies_for_one_two_and_many_targets(self) -> None:
        single = self._scatter(["t1"], {"t1": (0.91, 0.77, 6.0, 0.72)})
        self.assertEqual({row["scatter_kind"] for row in single}, {
            "single_target_confidence"
        })
        single_iptm = next(row for row in single if row["metric"] == "i_ptm")
        self.assertEqual((single_iptm["x_name"], single_iptm["x"]), (
            "binder pLDDT", 0.91
        ))

        two = self._scatter(
            ["t1", "t2"],
            {
                "t1": (0.9, 0.7, 5.0, 0.7),
                "t2": (0.8, 0.8, 7.0, 0.8),
            },
        )
        self.assertEqual({row["scatter_kind"] for row in two}, {
            "two_target_landscape"
        })
        two_iptm = next(row for row in two if row["metric"] == "i_ptm")
        self.assertEqual(
            (two_iptm["x_name"], two_iptm["y_name"], two_iptm["x"], two_iptm["y"]),
            ("t1", "t2", 0.7, 0.8),
        )
        self.assertTrue(all(row["passes_both_targets"] for row in two))

        many = self._scatter(
            ["t1", "t2", "t3"],
            {
                "t1": (0.9, 0.6, 5.0, 0.61),
                "t2": (0.8, 0.7, 6.0, 0.72),
                "t3": (0.85, 0.8, 8.0, 0.83),
            },
        )
        self.assertEqual({row["scatter_kind"] for row in many}, {
            "multitarget_landscape"
        })
        many_iptm = next(row for row in many if row["metric"] == "i_ptm")
        many_ipae = next(row for row in many if row["metric"] == "i_pae")
        self.assertAlmostEqual(many_iptm["x"], 0.7)
        self.assertEqual(many_iptm["y"], 0.6)
        self.assertAlmostEqual(many_ipae["x"], 19.0 / 3.0)
        self.assertEqual(many_ipae["y"], 8.0)

        failing = self._scatter(
            ["t1", "t2"],
            {
                "t1": (0.9, 0.7, 5.0, 0.7),
                "t2": (0.8, 0.4, 9.0, 0.5),
            },
        )
        self.assertFalse(any(row["passes_both_targets"] for row in failing))

    def test_curve_direction_and_bootstrap_are_deterministic(self) -> None:
        thresholds = np.asarray([0.5])
        first = _curve_rows(
            self.case,
            "i_ptm",
            "context",
            "target",
            np.asarray([0.2, 0.8]),
            thresholds,
            higher_is_better=True,
        )
        second = _curve_rows(
            self.case,
            "i_ptm",
            "context",
            "target",
            np.asarray([0.2, 0.8]),
            thresholds,
            higher_is_better=True,
        )
        self.assertEqual(first, second)
        self.assertEqual(first[0]["fraction"], 0.5)
        low_is_good = _curve_rows(
            self.case,
            "i_pae",
            "context",
            "target",
            np.asarray([5.0, 15.0]),
            np.asarray([10.0]),
            higher_is_better=False,
        )
        self.assertEqual(low_is_good[0]["fraction"], 0.5)

    def test_cross_reactivity_keeps_threshold_curves_for_all_metrics(self) -> None:
        context_rows = []
        for name, iptm, ipae, ipsae in (
            ("t1", 0.7, 5.0, 0.71),
            ("t2", 0.8, 6.0, 0.81),
        ):
            context_rows.append({
                "case_id": "case",
                "design_index": 0,
                "context": name,
                "role": "target",
                "i_ptm": iptm,
                "i_pae": ipae,
                "ipsae_min": ipsae,
            })
        design_rows = [{
            "case_id": "case",
            "design_index": 0,
            "targets_complete": True,
            "contexts_complete": True,
            "target_min_i_ptm": 0.7,
            "target_max_i_pae": 6.0,
            "target_min_ipsae_min": 0.71,
        }]
        manifest = {"contexts": [
            {"name": "t1", "role": "target"},
            {"name": "t2", "role": "target"},
        ]}

        rows = _build_curves(
            [self.case], context_rows, design_rows, manifest
        )

        self.assertEqual({row["metric"] for row in rows}, {
            "i_ptm", "i_pae", "ipsae_min",
        })
        self.assertEqual({row["curve_kind"] for row in rows}, {
            "context", "all_targets",
        })


if __name__ == "__main__":
    unittest.main()
