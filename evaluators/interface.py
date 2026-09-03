"""PyRosetta interface metrics for re-evaluated structures.

The design loop scores only AlphaFold confidence terms. Shape complementarity,
interface dG/dSASA, packstat and the hydrogen-bond counts come from Rosetta's
InterfaceAnalyzerMover, so they are computed here, after prediction, on the
structure the evaluator just produced.

PyRosetta is imported lazily: an evaluation that leaves ``interface_metrics``
off runs unchanged on an installation without PyRosetta.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Iterable

# Metric names as returned by functions.pyrosetta_utils.score_interface, in the
# order they appear in the evaluation CSVs.
INTERFACE_FIELDS = [
    "interface_sc",
    "interface_dG",
    "interface_dSASA",
    "interface_dG_SASA_ratio",
    "interface_packstat",
    "interface_nres",
    "interface_interface_hbonds",
    "interface_hbond_percentage",
    "interface_delta_unsat_hbonds",
    "interface_delta_unsat_hbonds_percentage",
    "interface_hydrophobicity",
    "interface_fraction",
    "binder_score",
    "surface_hydrophobicity",
]

# Extra columns that describe the scored structure rather than the interface.
INTERFACE_EXTRA_FIELDS = ["interface_residues", "interface_structure"]


def _to_pdb(structure: Path, workdir: Path) -> Path:
    """Return a PDB path for ``structure``, converting mmCIF when needed.

    AF3 writes model.cif; PyRosetta's pose_from_pdb wants a PDB file.
    """
    if structure.suffix.lower() != ".cif":
        return structure
    from Bio.PDB import MMCIFParser, PDBIO

    parser = MMCIFParser(QUIET=True)
    model = parser.get_structure("model", str(structure))
    converted = workdir / f"{structure.stem}.pdb"
    io = PDBIO()
    io.set_structure(model)
    io.save(str(converted))
    return converted


def score_structure(
    structure: Path,
    binder_chain: str,
    target_chains: Iterable[str],
    *,
    relax: bool = True,
    workdir: Path | None = None,
) -> dict[str, Any]:
    """Score one predicted complex and return flat interface metrics.

    ``relax`` runs FastRelax first. It shifts the values substantially -- an
    interface that scores a positive dG unrelaxed usually scores a strongly
    negative one after relaxation -- so relaxed and unrelaxed numbers must not
    be compared with each other. Turning it off is much faster.
    """
    from functions.pyrosetta_utils import pr_relax, score_interface

    structure = Path(structure)
    if not structure.is_file():
        raise FileNotFoundError(f"Structure not found: {structure}")
    owned = workdir is None
    directory = Path(tempfile.mkdtemp(prefix=".interface.")) if owned else Path(workdir)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        pdb = _to_pdb(structure, directory)
        scored = pdb
        if relax:
            scored = directory / f"{pdb.stem}_relaxed.pdb"
            pr_relax(str(pdb), str(scored))
        scores, _, residues = score_interface(
            str(scored), binder_chain, list(target_chains)
        )
        metrics: dict[str, Any] = {key: scores.get(key) for key in INTERFACE_FIELDS}
        metrics["interface_residues"] = residues
        metrics["interface_structure"] = None if owned else str(scored)
        return metrics
    finally:
        if owned:
            import shutil

            shutil.rmtree(directory, ignore_errors=True)
