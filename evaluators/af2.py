"""AF2 re-evaluation for selected Odin-Multi sequences."""

from __future__ import annotations

import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from odin_multi import (
    _csv_text,
    _interface_pae,
    _slug,
    atomic_write_json,
    atomic_write_text,
    file_lock,
    load_json,
    load_run,
)
from run_layout import layout_for_run


_CONFIG_KEYS = {
    "params_dir", "models", "seeds", "num_recycles", "use_multimer",
    "rm_target_seq", "rm_target_sc",
}
# Optional, so configurations written before interface scoring existed still
# load. Defaults keep the previous behaviour: no PyRosetta, no extra runtime.
_OPTIONAL_CONFIG_KEYS = {"interface_metrics": False, "interface_relax": True}
_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


from .interface import INTERFACE_EXTRA_FIELDS, INTERFACE_FIELDS, score_structure
from .ipsae import compute_ipsae_min


def _int_list(
    value: Any, name: str, minimum: int, maximum: int | None = None
) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"AF2 {name} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"AF2 {name} must contain integers")
    if any(item < minimum or maximum is not None and item > maximum for item in value):
        allowed = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
        raise ValueError(f"AF2 {name} values must be {allowed}")
    return sorted(set(value))


def normalize_af2_config(config_path: Path | None) -> dict[str, Any]:
    """Load one complete, explicit AF2 evaluator configuration."""
    if config_path is None:
        raise ValueError("AF2 requires --evaluator-config")
    config_path = config_path.resolve()
    raw = load_json(config_path)
    if not isinstance(raw, dict):
        raise ValueError("AF2 evaluator configuration must be a JSON object")
    unknown = sorted(set(raw) - _CONFIG_KEYS - set(_OPTIONAL_CONFIG_KEYS))
    if unknown:
        raise ValueError(f"Unknown AF2 evaluator setting(s): {', '.join(unknown)}")
    missing = sorted(_CONFIG_KEYS - set(raw))
    if missing:
        raise ValueError(
            f"Missing AF2 evaluator setting(s): {', '.join(missing)}"
        )

    use_multimer = raw["use_multimer"]
    if not isinstance(use_multimer, bool):
        raise ValueError("AF2 use_multimer must be true or false")
    model_limit = 4 if use_multimer else 1
    models = _int_list(raw["models"], "models", 0, model_limit)
    seeds = _int_list(raw["seeds"], "seeds", 0)

    recycles = raw["num_recycles"]
    if isinstance(recycles, bool) or not isinstance(recycles, int) or recycles < 0:
        raise ValueError("AF2 num_recycles must be a non-negative integer")

    base = config_path.parent
    params_dir = Path(str(raw["params_dir"])).expanduser()
    params_dir = (
        (base / params_dir).resolve() if not params_dir.is_absolute()
        else params_dir.resolve()
    )
    if not params_dir.is_dir():
        raise FileNotFoundError(f"AF2 parameter directory not found: {params_dir}")

    flags: dict[str, bool] = {}
    for key in ("rm_target_seq", "rm_target_sc"):
        value = raw[key]
        if not isinstance(value, bool):
            raise ValueError(f"AF2 {key} must be true or false")
        flags[key] = value
    for key, default in _OPTIONAL_CONFIG_KEYS.items():
        value = raw.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"AF2 {key} must be true or false")
        flags[key] = value
    return {
        "params_dir": str(params_dir),
        "models": models,
        "seeds": seeds,
        "num_recycles": recycles,
        "use_multimer": use_multimer,
        **flags,
    }


def _selection(run_dir: Path, name: str) -> dict[str, Any]:
    if _slug(name) != name:
        raise ValueError(f"Unsafe selection name: {name!r}")
    path = layout_for_run(run_dir).selection_dir(name) / "selection.json"
    if not path.is_file():
        raise FileNotFoundError(f"Selection not found: {path}")
    selection = load_json(path)
    if selection.get("status") != "complete":
        raise ValueError(f"Selection {name!r} is not complete")
    return selection


def _evaluation_root(
    run_dir: Path, name: str, selection_name: str, config: dict[str, Any]
) -> Path:
    if _slug(name) != name:
        raise ValueError(f"Unsafe evaluation name: {name!r}")
    layout = layout_for_run(run_dir)
    root = layout.evaluation_dir("af2", name)
    metadata_path = root / "evaluation.json"
    metadata = {
        "evaluator": "af2",
        "evaluation_name": name,
        "selection_name": selection_name,
        "config": config,
    }
    lock_root = layout.locks / ("Evaluations" if layout.legacy else "evaluations")
    lock = lock_root / "af2" / name / "configure.lock"
    with file_lock(lock):
        if metadata_path.exists() and load_json(metadata_path) != metadata:
            raise ValueError(
                f"Evaluation {name!r} uses different settings; "
                "use another --evaluation-name"
            )
        if not metadata_path.exists():
            atomic_write_json(metadata_path, metadata)
    return root


def _sequence(value: Any) -> str:
    sequence = str(value).strip().upper()
    if not sequence or any(aa not in _AMINO_ACIDS for aa in sequence):
        raise ValueError("Selected sequence must use the 20 canonical amino acids")
    return sequence


def _result_dir(
    jobs_root: Path,
    row: dict[str, Any],
    context: dict[str, Any],
    model: int,
    seed: int,
) -> Path:
    return (
        jobs_root / f"t{int(row['design_index']):05d}"
        / _slug(context["name"]) / f"model_{model + 1}_seed_{seed}"
    )


def _replicate_lock(
    run_dir: Path,
    evaluation_name: str,
    row: dict[str, Any],
    context: dict[str, Any],
    model: int,
    seed: int,
) -> Path:
    filename = (
        f"t{int(row['design_index']):05d}_{_slug(context['name'])}_"
        f"model{model + 1}_seed{seed}.lock"
    )
    layout = layout_for_run(run_dir)
    lock_root = layout.locks / ("Evaluations" if layout.legacy else "evaluations")
    return lock_root / "af2" / evaluation_name / filename


def _is_complete(path: Path, sequence: str) -> bool:
    if not path.is_file():
        return False
    try:
        result = load_json(path)
    except (OSError, ValueError):
        return False
    structure = path.parent / str(result.get("structure", "prediction.pdb"))
    metrics = result.get("metrics")
    return (
        result.get("status") == "complete"
        and result.get("sequence") == sequence
        and isinstance(metrics, dict)
        and _finite(metrics.get("ipsae_min")) is not None
        and structure.is_file()
    )


def _save_pdb(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".pdb", dir=path.parent
    )
    os.close(fd)
    try:
        model.save_pdb(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metrics(model: Any, binder_length: int) -> dict[str, float]:
    aux = model.aux
    per_residue = np.asarray(aux.get("plddt")).squeeze()
    if per_residue.ndim != 1 or per_residue.size < binder_length:
        raise ValueError("AF2 did not return binder per-residue pLDDT")
    plddt = _finite(per_residue[-binder_length:].mean())
    if plddt is None:
        raise ValueError("AF2 returned non-finite binder pLDDT")
    metrics = {"plddt": plddt}
    for key in ("ptm", "i_ptm"):
        value = _finite(aux.get("log", {}).get(key, aux.get(key)))
        if value is None:
            raise ValueError(f"AF2 did not return finite {key}")
        metrics[key] = value
    # aux["pae"] is get_pae(outputs): the expected error in Angstrom, not the
    # /31-normalised value the design loss uses. Reusing _interface_pae keeps
    # this identical to the selection stage and comparable to the AF3 column.
    i_pae = _interface_pae(aux.get("pae"), binder_length)
    if i_pae is None:
        raise ValueError("AF2 did not return a usable interface PAE")
    metrics["i_pae"] = i_pae
    pae = np.asarray(aux.get("pae"), dtype=float).squeeze()
    target_length = pae.shape[0] - binder_length
    binder_mask = np.arange(pae.shape[0]) >= target_length
    metrics["ipsae_min"] = compute_ipsae_min(pae, binder_mask, ~binder_mask)
    return metrics


def _prepare_model(
    make_model: Any,
    run_dir: Path,
    context: dict[str, Any],
    binder_length: int,
    config: dict[str, Any],
) -> Any:
    settings = load_json(run_dir / context["settings"])
    chains = settings.get("chains")
    if isinstance(chains, list):
        chains = ",".join(str(chain) for chain in chains)
    if not isinstance(chains, str) or not chains.strip():
        raise ValueError(f"Missing chains in {context['settings']}")
    model = make_model(
        protocol="binder",
        num_recycles=config["num_recycles"],
        data_dir=config["params_dir"],
        use_multimer=config["use_multimer"],
    )
    model.prep_inputs(
        pdb_filename=str(run_dir / context["pdb"]),
        chain=chains,
        binder_len=binder_length,
        rm_target_seq=config["rm_target_seq"],
        rm_target_sc=config["rm_target_sc"],
    )
    return model


def _failure(
    path: Path,
    row: dict[str, Any],
    context: dict[str, Any],
    model: int,
    seed: int,
    error: Exception,
) -> None:
    atomic_write_json(path, {
        "status": "failed",
        "design_index": int(row["design_index"]),
        "design_id": row["design_id"],
        "context": context["name"],
        "model_index": model,
        "seed": seed,
        "error": f"{type(error).__name__}: {error}",
    })


def _predict(
    af_model: Any,
    result_path: Path,
    row: dict[str, Any],
    context: dict[str, Any],
    sequence: str,
    model: int,
    seed: int,
    config: dict[str, Any],
    evaluation_name: str,
    selection_name: str,
) -> None:
    af_model.predict(
        seq=sequence,
        models=[model],
        num_recycles=config["num_recycles"],
        seed=seed,
        sample_models=False,
        dropout=False,
        verbose=False,
    )
    prediction = result_path.parent / "prediction.pdb"
    _save_pdb(af_model, prediction)
    metrics = _metrics(af_model, len(sequence))
    if config["interface_metrics"]:
        # ColabDesign's binder protocol emits exactly two chains: the whole
        # target merged into A, the hallucinated binder as B (prep.py sets
        # _lengths = [target_len, binder_len]).
        metrics.update(score_structure(
            prediction, "B", ["A"],
            relax=config["interface_relax"],
            workdir=result_path.parent,
        ))
    model_names = list(getattr(af_model, "_model_names", []))
    atomic_write_json(result_path, {
        "status": "complete",
        "evaluation_name": evaluation_name,
        "selection_name": selection_name,
        "design_index": int(row["design_index"]),
        "design_id": row["design_id"],
        "selected_iteration": int(row["iteration"]),
        "selected_stage": row["stage"],
        "sequence": sequence,
        "context": {"name": context["name"], "role": context["role"]},
        "model_index": model,
        "model_number": model + 1,
        "model_name": model_names[model] if model < len(model_names) else None,
        "seed": seed,
        "num_recycles": config["num_recycles"],
        "metrics": metrics,
        "structure": "prediction.pdb",
    })


def run_evaluation(
    run_dir: Path,
    selection_name: str,
    evaluation_name: str,
    *,
    config_path: Path | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, int]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    run_dir, manifest = load_run(run_dir)
    selection = _selection(run_dir, selection_name)
    config = normalize_af2_config(config_path)
    root = _evaluation_root(run_dir, evaluation_name, selection_name, config)
    jobs_root = layout_for_run(run_dir).evaluation_jobs("af2", evaluation_name)
    rows = [
        row for row in selection.get("rows", [])
        if row.get("selection_status") == "selected"
        and int(row["design_index"]) % num_shards == shard_index
    ]

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for context_index in range(len(manifest["contexts"])):
        for row in rows:
            grouped[(context_index, len(_sequence(row["sequence"])))].append(row)

    try:
        from colabdesign import clear_mem, mk_afdesign_model
    except ModuleNotFoundError as error:
        if error.name == "colabdesign":
            raise RuntimeError(
                "ColabDesign is not installed; run install_odin_multi.sh or "
                "pip install -e ./ColabDesign --no-deps"
            ) from error
        raise

    summary = {
        "designs": len(rows), "completed": 0, "skipped": 0,
        "busy": 0, "failed": 0,
    }
    failures: list[str] = []
    for (context_index, binder_length), group in sorted(grouped.items()):
        context = manifest["contexts"][context_index]
        af_model = None
        prepared = False
        preparation_error = None
        try:
            for row in group:
                sequence = _sequence(row["sequence"])
                for model in config["models"]:
                    for seed in config["seeds"]:
                        result_path = (
                            _result_dir(jobs_root, row, context, model, seed)
                            / "result.json"
                        )
                        if _is_complete(result_path, sequence):
                            summary["skipped"] += 1
                            continue
                        lock = _replicate_lock(
                            run_dir, evaluation_name, row, context, model, seed
                        )
                        with file_lock(lock, blocking=False) as acquired:
                            if not acquired:
                                summary["busy"] += 1
                                continue
                            if _is_complete(result_path, sequence):
                                summary["skipped"] += 1
                                continue
                            if not prepared:
                                prepared = True
                                try:
                                    clear_mem()
                                    af_model = _prepare_model(
                                        mk_afdesign_model, run_dir, context,
                                        binder_length, config,
                                    )
                                except Exception as error:
                                    preparation_error = error
                            label = (
                                f"t{int(row['design_index']):05d}/"
                                f"{context['name']}/m{model + 1}/s{seed}"
                            )
                            if preparation_error is not None:
                                _failure(
                                    result_path.parent / "failure.json", row,
                                    context, model, seed, preparation_error,
                                )
                                summary["failed"] += 1
                                failures.append(label)
                                continue
                            try:
                                _predict(
                                    af_model, result_path, row, context, sequence,
                                    model, seed, config, evaluation_name,
                                    selection_name,
                                )
                                summary["completed"] += 1
                            except Exception as error:
                                _failure(
                                    result_path.parent / "failure.json", row,
                                    context, model, seed, error,
                                )
                                summary["failed"] += 1
                                failures.append(label)
        finally:
            if af_model is not None:
                del af_model

    if summary["failed"]:
        raise RuntimeError(
            f"{summary['failed']} AF2 prediction(s) failed: "
            f"{', '.join(failures[:5])}"
        )
    return summary


def collect_evaluation(run_dir: Path, evaluation_name: str) -> dict[str, int]:
    run_dir, _ = load_run(run_dir)
    if _slug(evaluation_name) != evaluation_name:
        raise ValueError(f"Unsafe evaluation name: {evaluation_name!r}")
    layout = layout_for_run(run_dir)
    root = layout.evaluation_dir("af2", evaluation_name)
    if not (root / "evaluation.json").is_file():
        raise FileNotFoundError(f"Evaluation not found: {root}")
    lock_root = layout.locks / ("Evaluations" if layout.legacy else "evaluations")
    lock = lock_root / "af2" / evaluation_name / "summarize.lock"
    with file_lock(lock, blocking=False) as acquired:
        if not acquired:
            raise RuntimeError(f"Evaluation summary {evaluation_name!r} is busy")
        return _collect(run_dir, root)


def _collect(run_dir: Path, root: Path) -> dict[str, int]:
    layout = layout_for_run(run_dir)
    jobs_root = layout.evaluation_jobs("af2", root.name)
    results: list[dict[str, Any]] = []
    completed: set[Path] = set()
    for path in jobs_root.glob("t*/**/result.json"):
        result = load_json(path)
        if result.get("status") != "complete":
            continue
        metrics = result.get("metrics", {})
        context = result.get("context", {})
        results.append({
            "design_index": result.get("design_index"),
            "design_id": result.get("design_id"),
            "selection": result.get("selection_name"),
            "selected_iteration": result.get("selected_iteration"),
            "selected_stage": result.get("selected_stage"),
            "sequence": result.get("sequence"),
            "context": context.get("name"),
            "role": context.get("role"),
            "model": result.get("model_number"),
            "model_name": result.get("model_name"),
            "seed": result.get("seed"),
            "num_recycles": result.get("num_recycles"),
            "plddt": metrics.get("plddt"),
            "ptm": metrics.get("ptm"),
            "i_ptm": metrics.get("i_ptm"),
            "i_pae": metrics.get("i_pae"),
            "ipsae_min": metrics.get("ipsae_min"),
            **{key: metrics.get(key) for key in INTERFACE_FIELDS},
            **{key: metrics.get(key) for key in INTERFACE_EXTRA_FIELDS},
            "structure": str(
                (path.parent / result.get("structure", "prediction.pdb"))
                .relative_to(run_dir)
            ),
        })
        completed.add(path.parent.resolve())

    failures = []
    for path in jobs_root.glob("t*/**/failure.json"):
        if path.parent.resolve() in completed:
            continue
        failure = load_json(path)
        failures.append({
            "design_index": failure.get("design_index"),
            "design_id": failure.get("design_id"),
            "context": failure.get("context"),
            "model": int(failure.get("model_index", -1)) + 1,
            "seed": failure.get("seed"),
            "error": failure.get("error"),
        })
    order = lambda row: (
        int(row["design_index"]), str(row["context"]),
        int(row["model"]), int(row["seed"]),
    )
    results.sort(key=order)
    failures.sort(key=order)
    result_fields = [
        "design_index", "design_id", "selection", "selected_iteration",
        "selected_stage", "sequence", "context", "role", "model",
        "model_name", "seed", "num_recycles", "plddt", "ptm", "i_ptm", "i_pae",
        "ipsae_min",
        *INTERFACE_FIELDS, *INTERFACE_EXTRA_FIELDS,
        "structure",
    ]
    failure_fields = [
        "design_index", "design_id", "context", "model", "seed", "error",
    ]
    atomic_write_text(
        layout.evaluation_metrics("af2", root.name),
        _csv_text(results, result_fields),
    )
    atomic_write_text(
        layout.evaluation_failures("af2", root.name),
        _csv_text(failures, failure_fields),
    )
    return {"completed": len(results), "failed": len(failures)}
