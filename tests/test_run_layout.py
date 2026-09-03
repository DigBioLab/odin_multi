from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from odin_multi import _plot_selection_overview, init_run, load_run
from run_layout import RunLayout, layout_for_run


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class RunLayoutTest(unittest.TestCase):
    def test_only_the_two_shipped_examples_remain_and_initialize(self) -> None:
        target_root = REPOSITORY_ROOT / "settings_target"
        loss_root = REPOSITORY_ROOT / "settings_loss"
        advanced_root = REPOSITORY_ROOT / "settings_advanced"
        self.assertEqual(
            {path.name for path in target_root.glob("*.json")},
            {
                "specificity_target.json",
                "specificity_offtarget.json",
                "toxin_erabutoxin_a.json",
                "toxin_short_neurotoxin_alpha_nk.json",
            },
        )
        self.assertEqual(
            {path.name for path in loss_root.glob("*.json")},
            {"target.json", "offtarget.json"},
        )
        self.assertEqual(
            {path.name for path in advanced_root.glob("*.json")},
            {"general.json"},
        )

        examples = (
            (
                "specificity",
                ["specificity_target.json", "specificity_offtarget.json"],
                ["target.json", "offtarget.json"],
                ["target", "offtarget"],
            ),
            (
                "cross_reactivity",
                [
                    "toxin_erabutoxin_a.json",
                    "toxin_short_neurotoxin_alpha_nk.json",
                ],
                ["target.json", "target.json"],
                ["target", "target"],
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, settings_names, loss_names, roles in examples:
                manifest = init_run(
                    Path(directory) / name,
                    [target_root / item for item in settings_names],
                    advanced_root / "general.json",
                    [loss_root / item for item in loss_names],
                    42,
                )
                self.assertEqual(
                    [context["role"] for context in manifest["contexts"]], roles
                )
                for context in manifest["contexts"]:
                    self.assertTrue(
                        (Path(directory) / name / context["pdb"]).is_file()
                    )

    def test_selection_overviews_cover_each_method(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {
                "design_index": 0,
                "selection_status": "selected",
                "iteration": 120,
                "worst_target_i_ptm": 0.7,
                "worst_target_i_pae": 5.0,
                "strongest_offtarget_i_pae": 12.0,
            }
            for method in (
                "last", "best_i_ptm", "best_i_pae",
                "best_clipped_i_pae_ratio",
            ):
                output = root / method
                self.assertEqual(
                    _plot_selection_overview(output, method, [base]), 2
                )
                for suffix in (".png", ".svg"):
                    path = output / "figures" / f"selection_overview{suffix}"
                    self.assertGreater(path.stat().st_size, 0)

    def test_new_run_creates_numbered_pipeline_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            params = root / "params"
            params.mkdir()
            target = root / "target.pdb"
            target.write_text("END\n", encoding="utf-8")
            general = root / "general.json"
            settings = root / "target.json"
            loss = root / "loss.json"
            _write_json(general, {"af_params_dir": str(params), "lengths": [40]})
            _write_json(settings, {
                "binder_name": "target_a",
                "starting_pdb": str(target),
            })
            _write_json(loss, {"role": "target"})

            manifest = init_run(run, [settings], general, [loss], 7)
            layout = layout_for_run(run)

            self.assertEqual(manifest["layout_version"], 2)
            self.assertFalse(layout.legacy)
            self.assertEqual(layout.manifest, run / "run.json")
            for name in (
                "00_inputs", "01_designs", "02_selections",
                "03_evaluations", "04_summary", ".pipeline",
            ):
                self.assertTrue((run / name).is_dir(), name)
            self.assertFalse((run / "Inputs").exists())
            self.assertEqual(
                manifest["advanced"], "00_inputs/general.json"
            )
            self.assertTrue(
                (run / manifest["contexts"][0]["settings"]).is_file()
            )
            self.assertEqual(load_run(run)[1], manifest)

    def test_layout_paths_are_short_and_stage_local(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = RunLayout.new(Path(directory))
            self.assertEqual(
                layout.design_trajectory(3, "long_design_name"),
                layout.root / "01_designs/t00003/trajectory.pickle",
            )
            self.assertEqual(
                layout.selection_csv("best_i_ptm"),
                layout.root / "02_selections/best_i_ptm/selection.csv",
            )
            self.assertEqual(
                layout.evaluation_metrics("af3", "standard"),
                layout.root / "03_evaluations/af3/standard/metrics.csv",
            )
            self.assertEqual(
                layout.summary_report,
                layout.root / "04_summary/README.md",
            )

    def test_legacy_manifest_keeps_legacy_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            _write_json(run / "odin_multi_run.json", {"contexts": [{}]})
            layout = layout_for_run(run)
            self.assertTrue(layout.legacy)
            self.assertEqual(layout.selections, run / "Selections")
            self.assertEqual(layout.summary, run / "Summary")


if __name__ == "__main__":
    unittest.main()
