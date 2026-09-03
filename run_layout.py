"""Versioned filesystem layout for Odin-Multi run directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CURRENT_LAYOUT_VERSION = 2
RUN_FILE = "run.json"
LEGACY_RUN_FILE = "odin_multi_run.json"


@dataclass(frozen=True)
class RunLayout:
    """Resolve every pipeline stage without spreading path literals."""

    root: Path
    version: int

    @classmethod
    def new(cls, root: Path) -> "RunLayout":
        return cls(Path(root).resolve(), CURRENT_LAYOUT_VERSION)

    @classmethod
    def detect(cls, root: Path) -> "RunLayout":
        root = Path(root).resolve()
        if (root / RUN_FILE).is_file():
            return cls(root, CURRENT_LAYOUT_VERSION)
        if (root / LEGACY_RUN_FILE).is_file():
            return cls(root, 1)
        raise FileNotFoundError(
            f"Run is not configured: expected {root / RUN_FILE} "
            f"or {root / LEGACY_RUN_FILE}"
        )

    @property
    def legacy(self) -> bool:
        return self.version == 1

    @property
    def manifest(self) -> Path:
        return self.root / (LEGACY_RUN_FILE if self.legacy else RUN_FILE)

    @property
    def inputs(self) -> Path:
        return self.root / ("Inputs" if self.legacy else "00_inputs")

    @property
    def designs(self) -> Path:
        return self.root / ("Trajectory" if self.legacy else "01_designs")

    @property
    def selections(self) -> Path:
        return self.root / ("Selections" if self.legacy else "02_selections")

    @property
    def evaluations(self) -> Path:
        return self.root / ("Evaluations" if self.legacy else "03_evaluations")

    @property
    def summary(self) -> Path:
        return self.root / ("Summary" if self.legacy else "04_summary")

    @property
    def locks(self) -> Path:
        return self.root / ("Locks" if self.legacy else ".pipeline/locks")

    def design_dir(self, index: int) -> Path:
        if self.legacy:
            return self.designs
        return self.designs / f"t{index:05d}"

    def design_status(self, index: int) -> Path:
        if self.legacy:
            return self.root / "Status" / "Designs" / f"t{index:05d}.json"
        return self.design_dir(index) / "design.json"

    def design_checks(self, index: int) -> Path:
        if self.legacy:
            return self.root / "Status" / "Failures" / f"t{index:05d}.csv"
        return self.design_dir(index) / "checks.csv"

    def design_trajectory(self, index: int, design_id: str) -> Path:
        if self.legacy:
            return self.designs / "Pickle" / f"{design_id}_trajectory.pickle"
        return self.design_dir(index) / "trajectory.pickle"

    def design_auxiliary(self, index: int, design_id: str) -> Path:
        if self.legacy:
            return self.designs / "Pickle" / f"{design_id}.pickle"
        return self.design_dir(index) / "auxiliary.pickle"

    def selection_dir(self, name: str) -> Path:
        return self.selections / name

    def selection_csv(self, name: str) -> Path:
        filename = "selected_iterations.csv" if self.legacy else "selection.csv"
        return self.selection_dir(name) / filename

    def evaluation_dir(self, evaluator: str, name: str) -> Path:
        return self.evaluations / evaluator / name

    def evaluation_jobs(self, evaluator: str, name: str) -> Path:
        folder = "results" if self.legacy else "jobs"
        return self.evaluation_dir(evaluator, name) / folder

    def evaluation_metrics(self, evaluator: str, name: str) -> Path:
        filename = "evaluation_results.csv" if self.legacy else "metrics.csv"
        return self.evaluation_dir(evaluator, name) / filename

    def evaluation_failures(self, evaluator: str, name: str) -> Path:
        filename = "evaluation_failures.csv" if self.legacy else "failures.csv"
        return self.evaluation_dir(evaluator, name) / filename

    @property
    def figures(self) -> Path:
        return self.summary / ("Figures" if self.legacy else "figures")

    @property
    def summary_report(self) -> Path:
        return self.summary / ("summary.md" if self.legacy else "README.md")

    @property
    def summary_data(self) -> Path:
        return self.summary if self.legacy else self.summary / "data"

    def summary_table(self, kind: str) -> Path:
        legacy = {
            "replicates": "summary_source_data.csv",
            "contexts": "summary_by_context.csv",
            "designs": "summary_by_design.csv",
            "curves": "summary_curve_data.csv",
            "scatters": "summary_scatter_data.csv",
            "candidate_structures": "candidate_structures.csv",
        }
        current = {
            "replicates": "replicates.csv",
            "contexts": "contexts.csv",
            "designs": "designs.csv",
            "curves": "curves.csv",
            "scatters": "scatters.csv",
            "candidate_structures": "candidate_structures.csv",
        }
        names = legacy if self.legacy else current
        return self.summary_data / names[kind]

    @property
    def candidates_csv(self) -> Path:
        return self.summary / "candidates.csv"

    @property
    def candidates_fasta(self) -> Path:
        return self.summary / "candidates.fasta"

    @property
    def candidate_structures(self) -> Path:
        return self.summary / "candidate_structures"


def layout_for_run(root: Path, *, create: bool = False) -> RunLayout:
    """Return the existing layout, or version 2 for a not-yet-created run."""
    root = Path(root).resolve()
    if (root / RUN_FILE).is_file() or (root / LEGACY_RUN_FILE).is_file():
        return RunLayout.detect(root)
    if create:
        return RunLayout.new(root)
    return RunLayout.detect(root)
