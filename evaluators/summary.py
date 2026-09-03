"""Run-level summaries and transparent candidate rankings for Odin-Multi."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import pickle
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from odin_multi import (
    _copy_atomic,
    _csv_text,
    _run_path,
    _slug,
    _trajectory_frames,
    atomic_write_text,
    file_lock,
    load_json,
    load_run,
)

from .interface import INTERFACE_FIELDS
from run_layout import RunLayout, layout_for_run


IPTM_THRESHOLDS = np.linspace(0.0, 1.0, 101)
IPAE_THRESHOLDS = np.linspace(0.0, 31.0, 125)
IPSAE_THRESHOLDS = np.linspace(0.0, 1.0, 101)
# Specificity is swept over the target/off-target ratio, not over the metric.
RATIO_THRESHOLDS = np.linspace(1.0, 5.0, 160)
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 0

# Figure palette and reference cutoffs follow the visual vocabulary used in the
# protein-design benchmark figures.
# Teal reads as "what you want", grey as "what you do not"; the benchmark uses
# the same vocabulary to separate treatment from control, whereas a single run
# has no control arm and uses it to separate target from off-target.
TEAL = "#2BBAAC"
DARK_TEAL = "#117269"
MID_TEAL = "#189486"
CHARCOAL = "#42403F"
MID_GREY = "#828080"
LIGHT_GREY = "#C2C0BF"

# Suggested cutoffs. Annotation only - nothing is filtered on them.
TARGET_IPTM_THRESHOLD = 0.5   # minimum; higher is better
TARGET_IPAE_THRESHOLD = 7.5   # maximum, Angstrom; lower is better
TARGET_IPSAE_THRESHOLD = 0.60  # minimum; higher is better
# 1.5, not the benchmark figures' 2.5 - set for this pipeline.
SPECIFICITY_RATIO_THRESHOLD = 1.5

REFERENCE_CUTOFFS = {
    "i_ptm": TARGET_IPTM_THRESHOLD,
    "i_pae": TARGET_IPAE_THRESHOLD,
    "ipsae_min": TARGET_IPSAE_THRESHOLD,
}

# Axis windows. The raw i_pae sweep runs to 31 A, which buries the region the
# cutoff lives in, so the drawn window is tighter than the computed grid.
AXIS_LIMITS = {
    "i_ptm": (0.0, 1.0), "i_pae": (0.0, 30.0),
    "ipsae_min": (0.0, 1.0),
}
AXIS_TICKS = {
    "i_ptm": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "i_pae": [0, 5, 10, 15, 20, 25, 30],
    "ipsae_min": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
}

CONFIDENCE_METRICS = (
    "plddt",
    "binder_plddt",
    "ptm",
    "i_ptm",
    "global_i_ptm",
    "ipsae_min",
    "i_pae",
    "min_i_pae",
    "ranking_score",
)
NUMERIC_METRICS = (*CONFIDENCE_METRICS, *INTERFACE_FIELDS)
SOURCE_FIELDS = (
    "case_id",
    "case_label",
    "source",
    "evaluator",
    "evaluation_name",
    "selection",
    "design_index",
    "design_id",
    "selected_iteration",
    "selected_stage",
    "sequence",
    "context",
    "role",
    "replicate",
    "replicate_count_expected",
    "model",
    "model_name",
    "seed",
    "sample",
    "num_recycles",
    *NUMERIC_METRICS,
    "structure",
)


@dataclass(frozen=True)
class Case:
    case_id: str
    label: str
    source: str
    evaluator: str
    evaluation_name: str | None
    selection_name: str
    expected_replicates: int
    selected_rows: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    failures: int = 0


def _finite(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[Any]) -> float | None:
    collected = [item for item in (_finite(value) for value in values) if item is not None]
    return sum(collected) / len(collected) if collected else None


def _median(values: Iterable[Any]) -> float | None:
    collected = [item for item in (_finite(value) for value in values) if item is not None]
    return float(statistics.median(collected)) if collected else None


def _format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _interface_pae_stats(
    frame: Any, binder_length: int
) -> tuple[float | None, float | None]:
    array = np.asarray(frame).squeeze()
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        return None, None
    target_length = array.shape[0] - binder_length
    if target_length <= 0:
        return None, None
    values = np.concatenate((
        np.asarray(array[:target_length, target_length:], dtype=float).reshape(-1),
        np.asarray(array[target_length:, :target_length], dtype=float).reshape(-1),
    ))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    return float(values.mean()), float(values.min())


def _plddt_stats(
    frame: Any, binder_length: int
) -> tuple[float | None, float | None]:
    values = np.asarray(frame, dtype=float).squeeze()
    if values.ndim == 0:
        score = _finite(values)
        return score, score
    values = values.reshape(-1)
    finite = values[np.isfinite(values)]
    overall = float(finite.mean()) if finite.size else None
    if values.size < binder_length:
        return overall, None
    binder = values[-binder_length:]
    binder = binder[np.isfinite(binder)]
    return overall, float(binder.mean()) if binder.size else None


def _selection(run_dir: Path, name: str) -> dict[str, Any]:
    path = layout_for_run(run_dir).selection_dir(name) / "selection.json"
    if not path.is_file():
        raise FileNotFoundError(f"Selection not found: {path}")
    selection = load_json(path)
    if selection.get("status") != "complete":
        raise ValueError(f"Selection {name!r} is not complete")
    return selection


def _selected_rows(selection: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row for row in selection.get("rows", [])
        if row.get("selection_status") == "selected"
    ]


def _design_case_id(layout: RunLayout, selection_name: str) -> str:
    """Return a compact publication identifier for one design selection."""
    if layout.legacy:
        return f"af2_design__{_slug(selection_name)}"
    name = (
        "specificity"
        if selection_name == "best_clipped_i_pae_ratio"
        else _slug(selection_name)
    )
    return f"design_{name}"


def _design_case(
    run_dir: Path,
    manifest: dict[str, Any],
    selection_name: str,
) -> Case:
    layout = layout_for_run(run_dir)
    selected_rows = _selected_rows(_selection(run_dir, selection_name))
    case_id = _design_case_id(layout, selection_name)
    label = f"AF2 design ({selection_name})"
    rows: list[dict[str, Any]] = []
    for selected in selected_rows:
        design_index = int(selected["design_index"])
        design_id = str(selected["design_id"])
        status_path = layout.design_status(design_index)
        artifact_value = None
        if status_path.is_file():
            artifact_value = load_json(status_path).get("trajectory_pickle")
        if not artifact_value:
            artifact_value = str(
                layout.design_trajectory(design_index, design_id).relative_to(run_dir)
            )
        artifact = _run_path(run_dir, str(artifact_value))
        if not artifact.is_file():
            raise FileNotFoundError(f"Selected trajectory not found: {artifact}")
        with artifact.open("rb") as handle:
            payload = pickle.load(handle)

        frame_index = int(selected["frame_index"])
        sequence = str(selected["sequence"])
        binder_length = int(selected.get("length") or len(sequence))
        for context in manifest["contexts"]:
            trajectory = _trajectory_frames(payload.get(context["name"]))
            if not 0 <= frame_index < len(trajectory["seq"]):
                raise ValueError(
                    f"Selected frame {frame_index} is absent for "
                    f"{design_id}/{context['name']}"
                )
            plddt, binder_plddt = _plddt_stats(
                trajectory["plddt"][frame_index], binder_length
            )
            i_pae, min_i_pae = _interface_pae_stats(
                trajectory["pae"][frame_index], binder_length
            )
            row = {
                "case_id": case_id,
                "case_label": label,
                "source": "af2_design",
                "evaluator": "af2",
                "evaluation_name": None,
                "selection": selection_name,
                "design_index": design_index,
                "design_id": design_id,
                "selected_iteration": selected.get("iteration"),
                "selected_stage": selected.get("stage"),
                "sequence": sequence,
                "context": context["name"],
                "role": context["role"],
                "replicate": "selected_frame",
                "replicate_count_expected": 1,
                "model": None,
                "model_name": None,
                "seed": selected.get("seed"),
                "sample": None,
                "num_recycles": None,
                "plddt": plddt,
                "binder_plddt": binder_plddt,
                "ptm": _finite(trajectory["ptm"][frame_index]),
                "i_ptm": _finite(trajectory["i_ptm"][frame_index]),
                "i_pae": i_pae,
                "min_i_pae": min_i_pae,
                "ranking_score": None,
                "structure": None,
            }
            for metric in INTERFACE_FIELDS:
                row[metric] = None
            rows.append(row)
    return Case(
        case_id=case_id,
        label=label,
        source="af2_design",
        evaluator="af2",
        evaluation_name=None,
        selection_name=selection_name,
        expected_replicates=1,
        selected_rows=selected_rows,
        rows=rows,
    )


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _expected_replicates(metadata: dict[str, Any]) -> int:
    evaluator = str(metadata.get("evaluator"))
    config = metadata.get("config", {})
    seeds = config.get("seeds") or [0]
    if evaluator == "af2":
        models = config.get("models") or [0]
        return max(1, len(models) * len(seeds))
    flags = config.get("extra_flags") or {}
    samples = _integer(flags.get("num_diffusion_samples")) or 5
    return max(1, len(seeds) * samples)


def _evaluation_identity(
    layout: RunLayout,
    evaluator: str,
    evaluation_name: str,
    evaluation_count: int,
) -> tuple[str, str]:
    """Return the public ID and label without exposing an unnecessary run name."""
    if layout.legacy:
        return (
            f"{evaluator}_reevaluation__{_slug(evaluation_name)}",
            f"{evaluator.upper()} reevaluation ({evaluation_name})",
        )
    if evaluation_count <= 1:
        return evaluator, f"{evaluator.upper()} reevaluation"
    return (
        f"{evaluator}_{_slug(evaluation_name)}",
        f"{evaluator.upper()} reevaluation ({evaluation_name})",
    )


def _evaluation_case(
    run_dir: Path, root: Path, *, evaluation_count: int = 1
) -> Case:
    layout = layout_for_run(run_dir)
    metadata = load_json(root / "evaluation.json")
    evaluator = str(metadata.get("evaluator") or root.parent.name)
    evaluation_name = str(metadata.get("evaluation_name") or root.name)
    selection_name = str(metadata.get("selection_name") or "")
    if evaluator not in {"af2", "af3"}:
        raise ValueError(f"Unsupported evaluator in {root}: {evaluator!r}")
    if not selection_name:
        raise ValueError(f"Evaluation metadata lacks selection_name: {root}")
    expected = _expected_replicates(metadata)
    case_id, label = _evaluation_identity(
        layout, evaluator, evaluation_name, evaluation_count
    )
    results_path = layout.evaluation_metrics(evaluator, evaluation_name)
    if not results_path.is_file():
        raise FileNotFoundError(f"Collected results not found: {results_path}")

    rows: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"design_index", "design_id", "context", "role", "i_ptm"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{results_path} is missing columns: {sorted(missing)}")
        for raw in reader:
            model = _integer(raw.get("model"))
            seed = _integer(raw.get("seed"))
            sample = _integer(raw.get("sample"))
            replicate = (
                f"model_{model}_seed_{seed}"
                if evaluator == "af2"
                else f"seed_{seed}_sample_{sample}"
            )
            row = {
                "case_id": case_id,
                "case_label": label,
                "source": f"{evaluator}_reevaluation",
                "evaluator": evaluator,
                "evaluation_name": evaluation_name,
                "selection": raw.get("selection") or selection_name,
                "design_index": _integer(raw.get("design_index")),
                "design_id": raw.get("design_id"),
                "selected_iteration": _integer(raw.get("selected_iteration")),
                "selected_stage": raw.get("selected_stage"),
                "sequence": raw.get("sequence"),
                "context": raw.get("context"),
                "role": raw.get("role"),
                "replicate": replicate,
                "replicate_count_expected": expected,
                "model": model,
                "model_name": raw.get("model_name"),
                "seed": seed,
                "sample": sample,
                "num_recycles": _integer(raw.get("num_recycles")),
                "structure": raw.get("structure"),
            }
            for metric in NUMERIC_METRICS:
                row[metric] = _finite(raw.get(metric))
            if evaluator == "af2" and row.get("binder_plddt") is None:
                # AF2 evaluation_results.csv defines plddt as the binder mean.
                row["binder_plddt"] = row.get("plddt")
            rows.append(row)
    return Case(
        case_id=case_id,
        label=label,
        source=f"{evaluator}_reevaluation",
        evaluator=evaluator,
        evaluation_name=evaluation_name,
        selection_name=selection_name,
        expected_replicates=expected,
        selected_rows=_selected_rows(_selection(run_dir, selection_name)),
        rows=rows,
        failures=_count_csv_rows(
            layout.evaluation_failures(evaluator, evaluation_name)
        ),
    )


def _discover_evaluations(
    run_dir: Path,
    evaluator: str | None,
    evaluation_name: str | None,
) -> list[Path]:
    layout = layout_for_run(run_dir)
    if (evaluator is None) != (evaluation_name is None):
        raise ValueError(
            "--evaluator and --evaluation-name must be supplied together"
        )
    if evaluator is not None:
        root = layout.evaluation_dir(evaluator, str(evaluation_name))
        if not (root / "evaluation.json").is_file():
            raise FileNotFoundError(f"Evaluation not found: {root}")
        return [root]
    return sorted(
        path.parent
        for path in layout.evaluations.glob("*/*/evaluation.json")
        if path.parent.parent.name in {"af2", "af3"}
    )


def _collect_evaluations(run_dir: Path, roots: list[Path]) -> None:
    for root in roots:
        metadata = load_json(root / "evaluation.json")
        evaluator = str(metadata.get("evaluator") or root.parent.name)
        name = str(metadata.get("evaluation_name") or root.name)
        if evaluator == "af2":
            from .af2 import collect_evaluation
        elif evaluator == "af3":
            from .af3 import collect_evaluation
        else:
            raise ValueError(f"Unsupported evaluator {evaluator!r}")
        collect_evaluation(run_dir, name)


def _selection_names(
    run_dir: Path, roots: list[Path], include_all: bool
) -> list[str]:
    layout = layout_for_run(run_dir)
    names = {
        str(load_json(root / "evaluation.json").get("selection_name") or "")
        for root in roots
    }
    names.discard("")
    if include_all or not roots:
        names.update(
            path.parent.name
            for path in layout.selections.glob("*/selection.json")
            if load_json(path).get("status") == "complete"
        )
    return sorted(names)


def _aggregate_contexts(cases: list[Case]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    identity_fields = (
        "case_id",
        "case_label",
        "source",
        "evaluator",
        "evaluation_name",
        "selection",
        "design_index",
        "design_id",
        "selected_iteration",
        "selected_stage",
        "sequence",
        "context",
        "role",
    )
    for case in cases:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in case.rows:
            grouped[tuple(row.get(field) for field in identity_fields)].append(row)
        for identity, replicates in grouped.items():
            if len(replicates) < case.expected_replicates:
                continue
            row = dict(zip(identity_fields, identity))
            row["replicate_count"] = len(replicates)
            row["replicate_count_expected"] = case.expected_replicates
            row["replicates_complete"] = len(replicates) >= case.expected_replicates
            for metric in NUMERIC_METRICS:
                row[metric] = _mean(item.get(metric) for item in replicates)
            result.append(row)
    result.sort(key=lambda row: (
        str(row["case_id"]), int(row["design_index"]), str(row["context"])
    ))
    return result


def _role_values(
    rows: list[dict[str, Any]], role: str, metric: str
) -> list[float]:
    return [
        value
        for value in (
            _finite(row.get(metric)) for row in rows if row.get("role") == role
        )
        if value is not None
    ]


def _aggregate_designs(
    cases: list[Case],
    context_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_targets = [
        item["name"] for item in manifest["contexts"] if item["role"] == "target"
    ]
    expected_offtargets = [
        item["name"] for item in manifest["contexts"] if item["role"] == "offtarget"
    ]
    by_case_context: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in context_rows:
        by_case_context[str(row["case_id"])][int(row["design_index"])].append(row)

    result: list[dict[str, Any]] = []
    for case in cases:
        for selected in case.selected_rows:
            index = int(selected["design_index"])
            contexts = by_case_context[case.case_id].get(index, [])
            observed_targets = {
                str(row["context"]) for row in contexts if row["role"] == "target"
            }
            observed_offtargets = {
                str(row["context"])
                for row in contexts
                if row["role"] == "offtarget"
            }
            target_iptm = _role_values(contexts, "target", "i_ptm")
            target_ipae = _role_values(contexts, "target", "i_pae")
            target_ipsae = _role_values(contexts, "target", "ipsae_min")
            target_plddt = _role_values(contexts, "target", "binder_plddt")
            offtarget_iptm = _role_values(contexts, "offtarget", "i_ptm")
            offtarget_ipae = _role_values(contexts, "offtarget", "i_pae")
            offtarget_ipsae = _role_values(contexts, "offtarget", "ipsae_min")
            target_min_iptm = min(target_iptm) if target_iptm else None
            target_min_ipsae = min(target_ipsae) if target_ipsae else None
            target_max_ipae = max(target_ipae) if target_ipae else None
            offtarget_max_iptm = max(offtarget_iptm) if offtarget_iptm else None
            offtarget_min_ipae = min(offtarget_ipae) if offtarget_ipae else None
            offtarget_max_ipsae = max(offtarget_ipsae) if offtarget_ipsae else None
            result.append({
                "case_id": case.case_id,
                "case_label": case.label,
                "source": case.source,
                "evaluator": case.evaluator,
                "evaluation_name": case.evaluation_name,
                "selection": case.selection_name,
                "design_index": index,
                "design_id": selected.get("design_id"),
                "selected_iteration": selected.get("iteration"),
                "selected_stage": selected.get("stage"),
                "sequence": selected.get("sequence"),
                "contexts_expected": len(manifest["contexts"]),
                "contexts_observed": len({str(row["context"]) for row in contexts}),
                "contexts_complete": (
                    observed_targets == set(expected_targets)
                    and observed_offtargets == set(expected_offtargets)
                ),
                "target_contexts_expected": len(expected_targets),
                "target_contexts_observed": len(observed_targets),
                "targets_complete": observed_targets == set(expected_targets),
                "offtarget_contexts_expected": len(expected_offtargets),
                "offtarget_contexts_observed": len(observed_offtargets),
                "offtargets_complete": observed_offtargets == set(expected_offtargets),
                "target_min_i_ptm": target_min_iptm,
                "target_mean_i_ptm": _mean(target_iptm),
                "target_min_ipsae_min": target_min_ipsae,
                "target_mean_ipsae_min": _mean(target_ipsae),
                "target_max_i_pae": target_max_ipae,
                "target_mean_i_pae": _mean(target_ipae),
                "target_min_binder_plddt": min(target_plddt) if target_plddt else None,
                "target_mean_binder_plddt": _mean(target_plddt),
                "offtarget_max_i_ptm": offtarget_max_iptm,
                "offtarget_mean_i_ptm": _mean(offtarget_iptm),
                "offtarget_max_ipsae_min": offtarget_max_ipsae,
                "offtarget_mean_ipsae_min": _mean(offtarget_ipsae),
                "offtarget_min_i_pae": offtarget_min_ipae,
                "offtarget_mean_i_pae": _mean(offtarget_ipae),
                "target_offtarget_i_ptm_gap": (
                    target_min_iptm - offtarget_max_iptm
                    if target_min_iptm is not None and offtarget_max_iptm is not None
                    else None
                ),
                "target_offtarget_ipsae_min_gap": (
                    target_min_ipsae - offtarget_max_ipsae
                    if target_min_ipsae is not None and offtarget_max_ipsae is not None
                    else None
                ),
                "offtarget_target_i_pae_ratio": (
                    offtarget_min_ipae / target_max_ipae
                    if offtarget_min_ipae is not None
                    and target_max_ipae is not None
                    and target_max_ipae > 0.0
                    else None
                ),
            })
    result.sort(key=lambda row: (str(row["case_id"]), int(row["design_index"])))
    return result


def _rank_candidates(
    design_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Annotate design rows and return ranked, sequence-unique candidates.

    Rankings are independent per result case. Target-only/cross-reactivity runs
    rank every complete design by worst-target iPAE (ascending). Specificity
    runs first require strongest-off-target / worst-target iPAE >= 1.5, then
    use the same target-quality ordering. Exact duplicate sequences retain only
    their best-ranked instance.
    """
    has_offtargets = any(
        context.get("role") == "offtarget" for context in manifest["contexts"]
    )
    target_count = sum(
        context.get("role") == "target" for context in manifest["contexts"]
    )
    if has_offtargets:
        mode = "specificity"
    else:
        mode = "cross_reactivity" if target_count > 1 else "target_only"
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in design_rows:
        row["candidate_mode"] = mode
        row["candidate_status"] = ""
        row["candidate_rank"] = None
        row["duplicate_of_design_id"] = None
        by_case[str(row["case_id"])].append(row)

    ranked: list[dict[str, Any]] = []
    for case_id in sorted(by_case):
        eligible: list[dict[str, Any]] = []
        for row in by_case[case_id]:
            target_ipae = _finite(row.get("target_max_i_pae"))
            ratio = _finite(row.get("offtarget_target_i_pae_ratio"))
            sequence = str(row.get("sequence") or "")
            if not row.get("contexts_complete"):
                row["candidate_status"] = "incomplete_contexts"
            elif not sequence:
                row["candidate_status"] = "missing_sequence"
            elif target_ipae is None:
                row["candidate_status"] = "missing_target_i_pae"
            elif has_offtargets and ratio is None:
                row["candidate_status"] = "missing_specificity_ratio"
            elif has_offtargets and ratio < SPECIFICITY_RATIO_THRESHOLD:
                row["candidate_status"] = "below_specificity_ratio"
            else:
                eligible.append(row)

        eligible.sort(key=lambda row: (
            float(row["target_max_i_pae"]),
            int(row["design_index"]),
            str(row.get("design_id") or ""),
        ))
        seen_sequences: dict[str, dict[str, Any]] = {}
        for row in eligible:
            sequence = str(row["sequence"])
            previous = seen_sequences.get(sequence)
            if previous is not None:
                row["candidate_status"] = "duplicate_sequence"
                row["duplicate_of_design_id"] = previous.get("design_id")
                continue
            row["candidate_status"] = "ranked"
            row["candidate_rank"] = len(seen_sequences) + 1
            seen_sequences[sequence] = row
            ranked.append(row)
    return ranked


def _candidate_fasta(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        lines.extend([
            (
                f">{row['case_id']}|rank={row['candidate_rank']}|"
                f"design={row['design_id']}|index={row['design_index']}"
            ),
            str(row["sequence"]),
        ])
    return "\n".join(lines) + ("\n" if lines else "")


def _export_candidate_structures(
    layout: RunLayout,
    cases: list[Case],
    candidates: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Copy the lowest-interface-PAE prediction for each candidate/context."""
    by_case = {case.case_id: case for case in cases}
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        case = by_case[str(candidate["case_id"])]
        design_index = int(candidate["design_index"])
        rank = int(candidate["candidate_rank"])
        for context in manifest["contexts"]:
            matches = []
            for row in case.rows:
                if (
                    int(row["design_index"]) != design_index
                    or row.get("context") != context["name"]
                    or _finite(row.get("i_pae")) is None
                    or not row.get("structure")
                ):
                    continue
                source = _run_path(layout.root, str(row["structure"]))
                if source.is_file():
                    matches.append((row, source))

            base = {
                "case_id": case.case_id,
                "candidate_rank": rank,
                "design_index": design_index,
                "design_id": candidate["design_id"],
                "context": context["name"],
                "role": context["role"],
            }
            if not matches:
                output.append({**base, "status": "unavailable"})
                continue

            def ordering(item: tuple[dict[str, Any], Path]) -> tuple[Any, ...]:
                row, _ = item
                ipsae = _finite(row.get("ipsae_min"))
                iptm = _finite(row.get("i_ptm"))
                return (
                    float(row["i_pae"]),
                    -ipsae if ipsae is not None else math.inf,
                    -iptm if iptm is not None else math.inf,
                    str(row.get("replicate") or ""),
                )

            selected, source = min(matches, key=ordering)
            suffix = source.suffix.lower() if source.suffix else ".structure"
            destination = (
                layout.candidate_structures / case.case_id
                / f"rank_{rank:03d}_t{design_index:05d}"
                / f"{_slug(context['name'])}{suffix}"
            )
            _copy_atomic(source, destination)
            output.append({
                **base,
                "status": "exported",
                "model": selected.get("model"),
                "model_name": selected.get("model_name"),
                "seed": selected.get("seed"),
                "sample": selected.get("sample"),
                "i_pae": selected.get("i_pae"),
                "ipsae_min": selected.get("ipsae_min"),
                "i_ptm": selected.get("i_ptm"),
                "source_structure": str(source.relative_to(layout.root)),
                "exported_structure": str(destination.relative_to(layout.root)),
            })
    return output


def _stable_rng(*parts: str) -> np.random.Generator:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    derived = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, derived]))


def _curve_rows(
    case: Case,
    metric: str,
    curve_kind: str,
    curve_label: str,
    values: np.ndarray,
    thresholds: np.ndarray,
    *,
    curve_role: str = "",
    higher_is_better: bool,
    second_values: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if values.size == 0:
        return []
    events = (
        values[:, None] > thresholds[None, :]
        if higher_is_better
        else values[:, None] < thresholds[None, :]
    )
    if second_values is not None:
        events &= (
            second_values[:, None] <= thresholds[None, :]
            if higher_is_better
            else second_values[:, None] >= thresholds[None, :]
        )
    point = events.mean(axis=0)
    rng = _stable_rng(case.case_id, metric, curve_kind, curve_label)
    bootstrap = rng.binomial(
        values.size,
        point[None, :],
        size=(BOOTSTRAP_REPLICATES, len(thresholds)),
    ) / values.size
    lower, upper = np.quantile(bootstrap, (0.025, 0.975), axis=0)
    return [
        {
            "case_id": case.case_id,
            "case_label": case.label,
            "source": case.source,
            "evaluator": case.evaluator,
            "evaluation_name": case.evaluation_name,
            "selection": case.selection_name,
            "metric": metric,
            "curve_kind": curve_kind,
            "curve_label": curve_label,
            "curve_role": curve_role,
            "threshold": float(threshold),
            "fraction": float(fraction),
            "ci_lower": float(low),
            "ci_upper": float(high),
            "n_designs": int(values.size),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        }
        for threshold, fraction, low, high in zip(thresholds, point, lower, upper)
    ]


def _build_curves(
    cases: list[Case],
    context_rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    contexts = [(item["name"], item["role"]) for item in manifest["contexts"]]
    target_count = sum(role == "target" for _, role in contexts)
    has_offtargets = any(role == "offtarget" for _, role in contexts)
    for case in cases:
        case_contexts = [row for row in context_rows if row["case_id"] == case.case_id]
        case_designs = [row for row in design_rows if row["case_id"] == case.case_id]
        for metric, thresholds, higher in (
            ("i_ptm", IPTM_THRESHOLDS, True),
            ("i_pae", IPAE_THRESHOLDS, False),
            ("ipsae_min", IPSAE_THRESHOLDS, True),
        ):
            for context, role in contexts:
                values = np.asarray([
                    value
                    for value in (
                        _finite(row.get(metric))
                        for row in case_contexts
                        if row["context"] == context
                    )
                    if value is not None
                ], dtype=float)
                readable = "target" if str(role) == "target" else "off-target"
                output.extend(_curve_rows(
                    case,
                    metric,
                    "context",
                    f"{context} \u00b7 {readable}",
                    values,
                    thresholds,
                    curve_role=str(role),
                    higher_is_better=higher,
                ))

            target_field = {
                "i_ptm": "target_min_i_ptm",
                "i_pae": "target_max_i_pae",
                "ipsae_min": "target_min_ipsae_min",
            }[metric]
            if target_count > 1:
                target_values = np.asarray([
                    float(row[target_field])
                    for row in case_designs
                    if row["targets_complete"] and _finite(row.get(target_field)) is not None
                ], dtype=float)
                output.extend(_curve_rows(
                    case,
                    metric,
                    "all_targets",
                    "all targets",
                    target_values,
                    thresholds,
                    higher_is_better=higher,
                ))
            if has_offtargets:
                off_field = {
                    "i_ptm": "offtarget_max_i_ptm",
                    "i_pae": "offtarget_min_i_pae",
                    "ipsae_min": "offtarget_max_ipsae_min",
                }[metric]
                complete = [
                    row for row in case_designs
                    if row["contexts_complete"]
                    and _finite(row.get(target_field)) is not None
                    and _finite(row.get(off_field)) is not None
                ]
                targets = np.asarray([float(row[target_field]) for row in complete])
                offtargets = np.asarray([float(row[off_field]) for row in complete])
                cutoff = REFERENCE_CUTOFFS[metric]
                if higher:
                    # iPTM: want the target high and the off-target low.
                    qualified = targets > cutoff
                    ratio = targets / np.maximum(offtargets, 1e-8)
                else:
                    # iPAE: want the target low and the off-target high, so the
                    # ratio inverts to keep "higher is better".
                    qualified = targets < cutoff
                    ratio = offtargets / np.maximum(targets, 1e-8)
                # Designs failing the target gate stay in the denominator but
                # can never clear a ratio threshold.
                masked = np.where(qualified, ratio, -np.inf)
                output.extend(_curve_rows(
                    case,
                    metric,
                    "specificity",
                    f"target passes {cutoff:g}, by ratio",
                    masked,
                    RATIO_THRESHOLDS,
                    higher_is_better=True,
                ))
    if has_offtargets:
        # Specificity is communicated by its target/off-target landscape and
        # ratio-yield plot. Generic iPTM/iPAE threshold sweeps add clutter;
        # iPSAE_min remains available as the dedicated interface-confidence
        # threshold view.
        output = [
            row for row in output
            if row["metric"] == "ipsae_min"
            or row["curve_kind"] == "specificity"
        ]
    return output


def _context_lookup(
    context_rows: list[dict[str, Any]], case_id: str
) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in context_rows:
        if row["case_id"] == case_id:
            result[int(row["design_index"])][str(row["context"])] = row
    return result


def _build_scatter(
    cases: list[Case],
    context_rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    contexts = manifest["contexts"]
    targets = [item["name"] for item in contexts if item["role"] == "target"]
    offtargets = [item["name"] for item in contexts if item["role"] == "offtarget"]
    output: list[dict[str, Any]] = []
    for case in cases:
        by_index = _context_lookup(context_rows, case.case_id)
        case_designs = {
            int(row["design_index"]): row
            for row in design_rows
            if row["case_id"] == case.case_id
        }
        for metric in ("i_ptm", "i_pae", "ipsae_min"):
            if not any(
                row["case_id"] == case.case_id
                and _finite(row.get(metric)) is not None
                for row in context_rows
            ):
                continue
            for index, design in case_designs.items():
                if not design["targets_complete"]:
                    continue
                if offtargets:
                    if not design["contexts_complete"]:
                        continue
                    if metric == "i_ptm":
                        x_name, y_name = "weakest target iPTM", "strongest off-target iPTM"
                        x = _finite(design["target_min_i_ptm"])
                        y = _finite(design["offtarget_max_i_ptm"])
                    elif metric == "i_pae":
                        x_name, y_name = "worst target iPAE", "strongest off-target iPAE"
                        x = _finite(design["target_max_i_pae"])
                        y = _finite(design["offtarget_min_i_pae"])
                    else:
                        x_name, y_name = "weakest target iPSAE_min", "strongest off-target iPSAE_min"
                        x = _finite(design["target_min_ipsae_min"])
                        y = _finite(design["offtarget_max_ipsae_min"])
                    scatter_kind = "specificity_landscape"
                elif len(targets) == 2:
                    x_name, y_name = targets
                    x = _finite(by_index[index][targets[0]].get(metric))
                    y = _finite(by_index[index][targets[1]].get(metric))
                    scatter_kind = "two_target_landscape"
                elif len(targets) >= 3:
                    if metric == "i_ptm":
                        x_name, y_name = "mean target iPTM", "weakest target iPTM"
                        x = _finite(design["target_mean_i_ptm"])
                        y = _finite(design["target_min_i_ptm"])
                    elif metric == "i_pae":
                        x_name, y_name = "mean target iPAE", "worst target iPAE"
                        x = _finite(design["target_mean_i_pae"])
                        y = _finite(design["target_max_i_pae"])
                    else:
                        x_name, y_name = "mean target iPSAE_min", "weakest target iPSAE_min"
                        x = _finite(design["target_mean_ipsae_min"])
                        y = _finite(design["target_min_ipsae_min"])
                    scatter_kind = "multitarget_landscape"
                else:
                    target = targets[0]
                    x_name = "binder pLDDT"
                    y_name = {
                        "i_ptm": "target iPTM",
                        "i_pae": "target iPAE",
                        "ipsae_min": "target iPSAE_min",
                    }[metric]
                    x = _finite(by_index[index][target].get("binder_plddt"))
                    y = _finite(by_index[index][target].get(metric))
                    scatter_kind = "single_target_confidence"
                if x is None or y is None:
                    continue
                passes_both_targets = None
                if scatter_kind == "two_target_landscape":
                    cutoff = REFERENCE_CUTOFFS[metric]
                    passes_both_targets = (
                        x < cutoff and y < cutoff
                        if metric == "i_pae"
                        else x > cutoff and y > cutoff
                    )
                output.append({
                    "case_id": case.case_id,
                    "case_label": case.label,
                    "source": case.source,
                    "evaluator": case.evaluator,
                    "evaluation_name": case.evaluation_name,
                    "selection": case.selection_name,
                    "design_index": index,
                    "design_id": design["design_id"],
                    "metric": metric,
                    "scatter_kind": scatter_kind,
                    "x_name": x_name,
                    "y_name": y_name,
                    "x": x,
                    "y": y,
                    "passes_both_targets": passes_both_targets,
                })
    return output


def _save_figure(fig: Any, path: Path, *, dpi: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(fd)
    try:
        fig.savefig(
            temporary,
            format=path.suffix.lstrip("."),
            dpi=dpi,
            facecolor="white",
            bbox_inches="tight",
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


_STYLE_READY = False


def _configure_style() -> Any:
    """Install the benchmark figure style once, and return pyplot.

    Matplotlib is imported lazily so a summarize that writes only CSVs, and any
    installation without a display, never pays for it. Agg is set explicitly
    rather than relying on the $DISPLAY-absent fallback.
    """
    global _STYLE_READY
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if _STYLE_READY:
        return plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.0,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.0,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 5.3,
        "legend.frameon": False,
        "axes.edgecolor": CHARCOAL,
        "axes.linewidth": 0.7,
        "axes.titlepad": 3.0,
        "lines.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "figure.dpi": 300,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    _STYLE_READY = True
    return plt


def _clean_axis(axis: Any) -> None:
    """The shared Prism-like axis contract used by the benchmark figures."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(CHARCOAL)
        axis.spines[side].set_linewidth(0.7)
    axis.tick_params(
        direction="out", pad=1.5, colors=CHARCOAL, width=0.65, length=3.0
    )
    axis.grid(False)


def _series_style(kind: str, role: str) -> tuple[str, str, float]:
    """Colour, linestyle and width for one curve, keyed by meaning not order.

    Positional colouring made a series change colour whenever the number of
    contexts changed, so the same context differed between the design and the
    reevaluation figure of one run.
    """
    if kind == "specificity":
        return CHARCOAL, "-", 1.0
    if kind == "all_targets":
        return DARK_TEAL, "-", 1.0
    if str(role) == "target":
        return TEAL, "-", 1.0
    return MID_GREY, "--", 0.8


def _figure_stem(layout: RunLayout, case_id: str, filename: str) -> Path:
    """Keep v2 figures flat; retain the legacy case-directory contract."""
    if layout.legacy:
        return layout.figures / case_id / filename
    return layout.figures / f"{case_id}_{filename}"


def _metric_stem(metric: str) -> str:
    return {"i_ptm": "iptm", "i_pae": "ipae"}.get(metric, metric)


def _display_title(row: dict[str, Any]) -> str:
    """Keep plot titles compact when a disambiguating name is present."""
    label = str(row.get("case_label") or "").strip()
    head, sep, _ = label.partition(" (")
    return head if sep else label


def _subtitle(row: dict[str, Any], n_designs: int) -> str:
    parts = [str(row.get("selection") or "").strip(), f"n={n_designs}"]
    return "  \u00b7  ".join(part for part in parts if part)


def _draw_curve_panel(
    plt: Any,
    rows: list[dict[str, Any]],
    *,
    cutoff: float | None,
    xlim: tuple[float, float],
    xticks: list[float],
    xlabel: str,
    ylabel: str,
) -> Any:
    """One sweep panel: curves, CI bands, reference line and cutoff markers."""
    fig, axis = plt.subplots(figsize=(3.5, 2.5))
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(
            str(row["curve_kind"]),
            str(row["curve_label"]),
            str(row.get("curve_role") or ""),
        )].append(row)

    # Reference line first, so curves and markers draw over it.
    if cutoff is not None:
        axis.axvline(cutoff, color=LIGHT_GREY, ls=":", lw=0.65, zorder=0)

    n_designs = 0
    for (kind, label, role), values in sorted(groups.items()):
        values.sort(key=lambda item: float(item["threshold"]))
        x = np.asarray([float(item["threshold"]) for item in values])
        y = 100.0 * np.asarray([float(item["fraction"]) for item in values])
        low = 100.0 * np.asarray([float(item["ci_lower"]) for item in values])
        high = 100.0 * np.asarray([float(item["ci_upper"]) for item in values])
        colour, linestyle, width = _series_style(kind, role)
        n_designs = max(n_designs, int(values[0]["n_designs"]))

        legend_label = label
        if cutoff is not None and x.size:
            at = float(np.interp(cutoff, x, y))
            legend_label = f"{label} ({at:.0f}% at {cutoff:g})"
        axis.plot(
            x, y, color=colour, ls=linestyle, lw=width,
            label=legend_label, zorder=3,
        )
        axis.fill_between(x, low, high, color=colour, alpha=0.10, lw=0, zorder=1)
        if cutoff is not None and x.size:
            axis.scatter(
                [cutoff], [float(np.interp(cutoff, x, y))],
                s=8, color=colour, edgecolors="white",
                linewidths=0.3, zorder=5,
            )

    axis.set_ylim(0.0, 102.0)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_xlim(*xlim)
    axis.set_xticks(xticks)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    _clean_axis(axis)
    axis.legend(loc="best", handlelength=1.35, handletextpad=0.35)
    return fig, axis, n_designs


def _plot_curves(layout: RunLayout, rows: list[dict[str, Any]]) -> int:
    plt = _configure_style()

    written = 0
    cases = sorted({str(row["case_id"]) for row in rows})
    for case_id in cases:
        for metric in ("i_ptm", "i_pae", "ipsae_min"):
            selected = [
                row for row in rows
                if row["case_id"] == case_id and row["metric"] == metric
            ]
            if not selected:
                continue
            higher = metric != "i_pae"
            direction = "\u2265" if higher else "<"

            # Panel 1: the metric sweep, one curve per context.
            metric_rows = [r for r in selected if r["curve_kind"] != "specificity"]
            if metric_rows:
                fig, axis, n = _draw_curve_panel(
                    plt, metric_rows,
                    cutoff=REFERENCE_CUTOFFS[metric],
                    xlim=AXIS_LIMITS[metric],
                    xticks=AXIS_TICKS[metric],
                    xlabel={
                        "i_ptm": "iPTM threshold, t",
                        "i_pae": "interface PAE threshold, t (\u00c5)",
                        "ipsae_min": "iPSAE_min threshold, t",
                    }[metric],
                    ylabel=f"Designs with metric {direction} t (%)",
                )
                axis.set_title(_display_title(metric_rows[0]), loc="left", pad=11)
                axis.text(
                    0.0, 1.012, _subtitle(metric_rows[0], n),
                    transform=axis.transAxes, color=MID_GREY, fontsize=5.0,
                    va="bottom", ha="left",
                )
                fig.tight_layout()
                filename = (
                    f"{_metric_stem(metric)}_curves"
                    if layout.legacy else f"{_metric_stem(metric)}_thresholds"
                )
                stem = _figure_stem(layout, case_id, filename)
                _save_figure(fig, stem.with_suffix(".svg"))
                _save_figure(fig, stem.with_suffix(".png"), dpi=300)
                plt.close(fig)
                written += 2

            # Panel 2: specificity yield. Designs are gated on the target cutoff,
            # then swept over the ratio - a different x axis, so its own figure.
            spec_rows = [r for r in selected if r["curve_kind"] == "specificity"]
            if spec_rows:
                cutoff = REFERENCE_CUTOFFS[metric]
                ratio_name = {
                    "i_ptm": "target / off-target iPTM",
                    "i_pae": "off-target / target iPAE",
                    "ipsae_min": "target / off-target iPSAE_min",
                }[metric]
                fig, axis, n = _draw_curve_panel(
                    plt, spec_rows,
                    cutoff=SPECIFICITY_RATIO_THRESHOLD,
                    xlim=(1.0, 5.0),
                    xticks=[1, 2, 3, 4, 5],
                    xlabel=f"{ratio_name} ratio threshold, r",
                    ylabel=f"Designs with target {direction} {cutoff:g}\nand ratio > r (%)",
                )
                axis.set_title(_display_title(spec_rows[0]), loc="left", pad=11)
                axis.text(
                    0.0, 1.012, _subtitle(spec_rows[0], n),
                    transform=axis.transAxes, color=MID_GREY, fontsize=5.0,
                    va="bottom", ha="left",
                )
                fig.tight_layout()
                stem = _figure_stem(
                    layout, case_id, f"{_metric_stem(metric)}_specificity"
                )
                _save_figure(fig, stem.with_suffix(".svg"))
                _save_figure(fig, stem.with_suffix(".png"), dpi=300)
                plt.close(fig)
                written += 2
    return written


def _shade_specificity(axis: Any, metric: str, limit: float) -> None:
    """Shade the region satisfying both the cutoff and the specificity ratio.

    Geometry mirrors with the metric. For iPTM higher is better, so a good
    design sits right of the cutoff with target well above off-target: the band
    lies *below* the y = x / ratio line. For iPAE lower is better, so a good
    design sits left of the cutoff with off-target well above target: the band
    lies *above* the y = ratio * x line.
    """
    ratio = SPECIFICITY_RATIO_THRESHOLD
    cutoff = REFERENCE_CUTOFFS[metric]
    if metric == "i_pae":
        x_end = min(cutoff, limit / ratio)
        x = np.linspace(0.0, x_end, 200)
        axis.fill_between(x, ratio * x, limit, color=TEAL, alpha=0.10, lw=0, zorder=0)
        axis.plot([0.0, limit / ratio], [0.0, limit], color=CHARCOAL, lw=0.7, zorder=1)
    else:
        x = np.linspace(cutoff, limit, 200)
        axis.fill_between(x, 0.0, x / ratio, color=TEAL, alpha=0.10, lw=0, zorder=0)
        axis.plot([0.0, limit], [0.0, limit / ratio], color=CHARCOAL, lw=0.7, zorder=1)
    axis.axvline(cutoff, color=CHARCOAL, ls="--", lw=0.7, zorder=1)


def _shade_both_targets(axis: Any, metric: str, low: float, high: float) -> None:
    """Mark the region where both target values clear the metric cutoff."""
    cutoff = REFERENCE_CUTOFFS[metric]
    if metric == "i_pae":
        axis.fill_between(
            [low, cutoff], low, cutoff, color=TEAL, alpha=0.10, lw=0, zorder=0
        )
    else:
        axis.fill_between(
            [cutoff, high], cutoff, high, color=TEAL, alpha=0.10, lw=0, zorder=0
        )
    axis.axvline(cutoff, color=CHARCOAL, ls="--", lw=0.7, zorder=1)
    axis.axhline(cutoff, color=CHARCOAL, ls="--", lw=0.7, zorder=1)


def _plot_scatter(layout: RunLayout, rows: list[dict[str, Any]]) -> int:
    plt = _configure_style()

    written = 0
    cases = sorted({str(row["case_id"]) for row in rows})
    for case_id in cases:
        for metric in ("i_ptm", "i_pae", "ipsae_min"):
            selected = [
                row for row in rows
                if row["case_id"] == case_id and row["metric"] == metric
            ]
            if not selected:
                continue
            x = np.asarray([float(row["x"]) for row in selected])
            y = np.asarray([float(row["y"]) for row in selected])
            fig, axis = plt.subplots(figsize=(2.8, 2.6))

            kind = str(selected[0].get("scatter_kind") or "")
            on_metric_axes = str(selected[0]["x_name"]) != "binder pLDDT"
            low, high = AXIS_LIMITS[metric]
            # The shaded region only means anything when both axes carry the
            # same metric for a target and an off-target.
            if on_metric_axes and kind == "specificity_landscape":
                _shade_specificity(axis, metric, high)
            elif kind == "two_target_landscape":
                _shade_both_targets(axis, metric, low, high)

            if kind == "two_target_landscape":
                passed = np.asarray([
                    bool(row.get("passes_both_targets")) for row in selected
                ])
                if np.any(~passed):
                    axis.scatter(
                        x[~passed], y[~passed], s=18, alpha=0.50,
                        facecolors="white", edgecolors=MID_GREY,
                        linewidths=0.7, label="Other designs", zorder=3,
                    )
                if np.any(passed):
                    axis.scatter(
                        x[passed], y[passed], s=20, alpha=0.75, color=TEAL,
                        edgecolors="white", linewidths=0.3,
                        label="Both targets pass", zorder=4,
                    )
                axis.text(
                    1.0, 1.012, f"Both pass: {int(passed.sum())}/{len(passed)}",
                    transform=axis.transAxes, color=CHARCOAL, fontsize=5.0,
                    va="bottom", ha="right",
                )
                axis.legend(loc="best", handletextpad=0.35)
            else:
                axis.scatter(
                    x, y, s=18, alpha=0.55, color=TEAL,
                    edgecolors="white", linewidths=0.3, zorder=4,
                )
            axis.set_xlabel(str(selected[0]["x_name"]))
            axis.set_ylabel(str(selected[0]["y_name"]))
            axis.set_title(_display_title(selected[0]), loc="left", pad=11)
            axis.text(
                0.0, 1.012, _subtitle(selected[0], len(selected)),
                transform=axis.transAxes, color=MID_GREY, fontsize=5.0,
                va="bottom", ha="left",
            )
            if on_metric_axes:
                axis.set_xlim(low, high)
                axis.set_xticks(AXIS_TICKS[metric])
            axis.set_ylim(low, high)
            axis.set_yticks(AXIS_TICKS[metric])
            _clean_axis(axis)
            fig.tight_layout()
            if layout.legacy:
                filename = f"{_metric_stem(metric)}_scatter"
            else:
                suffix = {
                    "specificity_landscape": "target_vs_offtarget",
                    "two_target_landscape": "target_comparison",
                    "multitarget_landscape": "target_summary",
                    "single_target_confidence": "confidence",
                }[kind]
                filename = f"{_metric_stem(metric)}_{suffix}"
            stem = _figure_stem(layout, case_id, filename)
            _save_figure(fig, stem.with_suffix(".svg"))
            _save_figure(fig, stem.with_suffix(".png"), dpi=300)
            plt.close(fig)
            written += 2
    return written


def _audit_markdown(
    layout: RunLayout,
    cases: list[Case],
    context_rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
    structure_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    designs_name = (
        "summary_by_design.csv" if layout.legacy else "data/designs.csv"
    )
    replicates_name = (
        "summary_source_data.csv" if layout.legacy else "data/replicates.csv"
    )
    lines = [
        "# Odin-Multi result summary",
        "",
        "- Replicate aggregation: arithmetic mean within each completed design/context job.",
        f"- Curve uncertainty: 95% binomial percentile intervals at each threshold, "
        f"{BOOTSTRAP_REPLICATES:,} draws of Binomial(n, p) / n, seed {BOOTSTRAP_SEED}. "
        "This is a parametric interval, not a resample of the underlying designs; "
        "it collapses to zero width where the observed fraction is exactly 0 or 1.",
        "- Cases use their available completed designs independently; no cross-source cohort intersection is imposed.",
        f"- Figures mark suggested cutoffs (iPTM {TARGET_IPTM_THRESHOLD:g}, "
        f"iPAE {TARGET_IPAE_THRESHOLD:g} \u00c5, iPSAE_min {TARGET_IPSAE_THRESHOLD:g}, specificity ratio "
        f"{SPECIFICITY_RATIO_THRESHOLD:g}); the specificity ratio is also the explicit candidate-qualification rule.",
        "- Candidate rankings are independent per case, preserve the full audit table, and leave final experimental selection to the user.",
        "",
    ]
    if not layout.legacy:
        lines.extend([
            "## Files",
            "",
            "- [Ranked candidates](candidates.csv) and [FASTA sequences](candidates.fasta)",
            "- [Candidate structures](candidate_structures/) and [structure provenance](data/candidate_structures.csv)",
            "- [Complete data tables](data/) and [figures](figures/)",
            "",
        ])
    lines.extend([
        "## Completion audit",
        "",
        "| case | selection | selected designs | expected context jobs | completed context jobs | missing context jobs | failed jobs | raw replicates |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for case in cases:
        selected = len(case.selected_rows)
        expected = selected * len(manifest["contexts"])
        completed = sum(row["case_id"] == case.case_id for row in context_rows)
        lines.append(
            f"| {case.label} | {case.selection_name} | {selected} | {expected} | "
            f"{completed} | {max(0, expected - completed)} | {case.failures} | {len(case.rows)} |"
        )

    lines.extend([
        "",
        "## Per-context medians",
        "",
        "| case | context | role | designs | binder pLDDT | iPTM | iPSAE_min | interface PAE |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for case in cases:
        for context in manifest["contexts"]:
            rows = [
                row for row in context_rows
                if row["case_id"] == case.case_id
                and row["context"] == context["name"]
            ]
            if not rows:
                continue
            lines.append(
                f"| {case.label} | {context['name']} | {context['role']} | {len(rows)} | "
                f"{_format_metric(_median(row['binder_plddt'] for row in rows))} | "
                f"{_format_metric(_median(row['i_ptm'] for row in rows))} | "
                f"{_format_metric(_median(row['ipsae_min'] for row in rows))} | "
                f"{_format_metric(_median(row['i_pae'] for row in rows))} |"
            )

    lines.extend([
        "",
        "## Complete-design role aggregates",
        "",
        "| case | complete designs | weakest target iPTM | weakest target iPSAE_min | worst target iPAE | strongest off-target iPTM | strongest off-target iPSAE_min | strongest off-target iPAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for case in cases:
        rows = [
            row for row in design_rows
            if row["case_id"] == case.case_id and row["contexts_complete"]
        ]
        lines.append(
            f"| {case.label} | {len(rows)} | "
            f"{_format_metric(_median(row['target_min_i_ptm'] for row in rows))} | "
            f"{_format_metric(_median(row['target_min_ipsae_min'] for row in rows))} | "
            f"{_format_metric(_median(row['target_max_i_pae'] for row in rows))} | "
            f"{_format_metric(_median(row['offtarget_max_i_ptm'] for row in rows))} | "
            f"{_format_metric(_median(row['offtarget_max_ipsae_min'] for row in rows))} | "
            f"{_format_metric(_median(row['offtarget_min_i_pae'] for row in rows))} |"
        )

    lines.extend([
        "",
        "## Candidate ranking",
        "",
        "Complete target-only and cross-reactivity designs are ranked by worst-target iPAE ascending. "
        f"Specificity designs must additionally have strongest-off-target / worst-target iPAE >= {SPECIFICITY_RATIO_THRESHOLD:g}. "
        f"Exact duplicate sequences keep only their best-ranked instance; all rows and their status remain in `{designs_name}`.",
        "",
        "| case | mode | ranked unique candidates | unranked rows | top design | worst target iPAE | specificity ratio |",
        "|---|---|---:|---:|---|---:|---:|",
    ])
    for case in cases:
        rows = [
            row for row in design_rows if row["case_id"] == case.case_id
        ]
        candidates = sorted(
            (row for row in rows if row["candidate_status"] == "ranked"),
            key=lambda row: int(row["candidate_rank"]),
        )
        top = candidates[0] if candidates else None
        lines.append(
            f"| {case.label} | {rows[0]['candidate_mode'] if rows else '—'} | "
            f"{len(candidates)} | {len(rows) - len(candidates)} | "
            f"{top['design_id'] if top else '—'} | "
            f"{_format_metric(_finite(top.get('target_max_i_pae')) if top else None)} | "
            f"{_format_metric(_finite(top.get('offtarget_target_i_pae_ratio')) if top else None)} |"
        )
    lines.extend([
        "",
        f"`{replicates_name}` preserves individual model, seed, and sample values. "
        f"Candidate structure exports: {sum(row.get('status') == 'exported' for row in structure_rows)}. "
        "The remaining CSVs preserve replicate means, role aggregates, statuses, and exact plotted values.",
        "",
    ])
    return "\n".join(lines)


def summarize_run(
    run_dir: Path,
    *,
    evaluator: str | None = None,
    evaluation_name: str | None = None,
) -> dict[str, int]:
    """Discover and summarize all available result cases in one run directory."""
    run_dir, manifest = load_run(run_dir)
    layout = layout_for_run(run_dir)
    lock_folder = "Summary" if layout.legacy else "summary"
    lock_path = layout.locks / lock_folder / "summarize.lock"
    with file_lock(lock_path, blocking=False) as acquired:
        if not acquired:
            raise RuntimeError("Run summary is busy")
        roots = _discover_evaluations(run_dir, evaluator, evaluation_name)
        _collect_evaluations(run_dir, roots)
        names = _selection_names(run_dir, roots, evaluator is None)
        if not names:
            raise FileNotFoundError(f"No completed selections found under {run_dir}")

        cases = [_design_case(run_dir, manifest, name) for name in names]
        evaluation_types = [
            str(
                load_json(root / "evaluation.json").get("evaluator")
                or root.parent.name
            )
            for root in roots
        ]
        evaluation_counts = Counter(evaluation_types)
        cases.extend(
            _evaluation_case(
                run_dir,
                root,
                evaluation_count=evaluation_counts[evaluation_type],
            )
            for root, evaluation_type in zip(roots, evaluation_types)
        )
        source_rows = [row for case in cases for row in case.rows]
        source_rows.sort(key=lambda row: (
            str(row["case_id"]),
            int(row["design_index"]),
            str(row["context"]),
            str(row["replicate"]),
        ))
        context_rows = _aggregate_contexts(cases)
        design_rows = _aggregate_designs(cases, context_rows, manifest)
        candidate_rows = _rank_candidates(design_rows, manifest)
        curve_rows = _build_curves(cases, context_rows, design_rows, manifest)
        scatter_rows = _build_scatter(cases, context_rows, design_rows, manifest)

        summary_dir = layout.summary
        if summary_dir.is_dir():
            shutil.rmtree(summary_dir)
        structure_rows = _export_candidate_structures(
            layout, cases, candidate_rows, manifest
        )
        atomic_write_text(
            layout.summary_table("replicates"),
            _csv_text(source_rows, list(SOURCE_FIELDS)),
        )
        context_fields = [
            "case_id", "case_label", "source", "evaluator", "evaluation_name",
            "selection", "design_index", "design_id", "selected_iteration",
            "selected_stage", "sequence", "context", "role", "replicate_count",
            "replicate_count_expected", "replicates_complete", *NUMERIC_METRICS,
        ]
        atomic_write_text(
            layout.summary_table("contexts"),
            _csv_text(context_rows, context_fields),
        )
        design_fields = [
            "case_id", "case_label", "source", "evaluator", "evaluation_name",
            "selection", "design_index", "design_id", "selected_iteration",
            "selected_stage", "sequence", "candidate_mode", "candidate_status",
            "candidate_rank", "duplicate_of_design_id", "contexts_expected", "contexts_observed",
            "contexts_complete", "target_contexts_expected",
            "target_contexts_observed", "targets_complete",
            "offtarget_contexts_expected", "offtarget_contexts_observed",
            "offtargets_complete", "target_min_i_ptm", "target_mean_i_ptm",
            "target_min_ipsae_min", "target_mean_ipsae_min",
            "target_max_i_pae", "target_mean_i_pae", "target_min_binder_plddt",
            "target_mean_binder_plddt", "offtarget_max_i_ptm",
            "offtarget_mean_i_ptm", "offtarget_min_i_pae",
            "offtarget_max_ipsae_min", "offtarget_mean_ipsae_min",
            "offtarget_mean_i_pae", "target_offtarget_i_ptm_gap",
            "target_offtarget_ipsae_min_gap",
            "offtarget_target_i_pae_ratio",
        ]
        atomic_write_text(
            layout.summary_table("designs"),
            _csv_text(design_rows, design_fields),
        )
        atomic_write_text(
            layout.candidates_csv,
            _csv_text(candidate_rows, design_fields),
        )
        atomic_write_text(
            layout.candidates_fasta,
            _candidate_fasta(candidate_rows),
        )
        curve_fields = [
            "case_id", "case_label", "source", "evaluator", "evaluation_name",
            "selection", "metric", "curve_kind", "curve_label", "curve_role",
            "threshold",
            "fraction", "ci_lower", "ci_upper", "n_designs",
            "bootstrap_replicates", "bootstrap_seed",
        ]
        atomic_write_text(
            layout.summary_table("curves"),
            _csv_text(curve_rows, curve_fields),
        )
        scatter_fields = [
            "case_id", "case_label", "source", "evaluator", "evaluation_name",
            "selection", "design_index", "design_id", "metric", "scatter_kind",
            "x_name", "y_name", "x", "y", "passes_both_targets",
        ]
        atomic_write_text(
            layout.summary_table("scatters"),
            _csv_text(scatter_rows, scatter_fields),
        )
        structure_fields = [
            "case_id", "candidate_rank", "design_index", "design_id",
            "context", "role", "status", "model", "model_name", "seed",
            "sample", "i_pae", "ipsae_min", "i_ptm", "source_structure",
            "exported_structure",
        ]
        atomic_write_text(
            layout.summary_table("candidate_structures"),
            _csv_text(structure_rows, structure_fields),
        )
        atomic_write_text(
            layout.summary_report,
            _audit_markdown(
                layout, cases, context_rows, design_rows, structure_rows,
                manifest,
            ),
        )
        figures = _plot_curves(layout, curve_rows)
        figures += _plot_scatter(layout, scatter_rows)
        return {
            "cases": len(cases),
            "source_rows": len(source_rows),
            "context_rows": len(context_rows),
            "design_rows": len(design_rows),
            "candidates": len(candidate_rows),
            "candidate_structures": sum(
                row.get("status") == "exported" for row in structure_rows
            ),
            "figures": figures,
        }
