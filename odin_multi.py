#!/usr/bin/env python3
"""Simple, resumable Odin-Multi design, selection, and AF2/AF3 evaluation."""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
import os
import pickle
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from run_layout import (
    CURRENT_LAYOUT_VERSION,
    RUN_FILE,
    RunLayout,
    layout_for_run,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
# Stages whose saved frames carry a discrete (one-hot) sequence, and are
# therefore selectable. "logits" and "soft" are continuous and never eligible.
# The name varies by design_algorithm: 3stage -> "hard", 3stage_ste -> "ste"
# (ColabDesign af/design.py:1571,1576), greedy -> "greedy", mcmc -> "mcmc".
HARD_STAGES = frozenset({"hard", "ste", "greedy", "mcmc"})
OFFTARGET_I_PAE_CAP = 15.0
SELECTION_METHODS = (
    "last",
    "best_i_ptm",
    "best_i_pae",
    "best_clipped_i_pae_ratio",
)
EVALUATORS = ("af2", "af3")
DEFAULT_EVALUATOR = "af3"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_.")
    if not result:
        raise ValueError(f"Cannot make a safe name from {value!r}")
    return result


@contextmanager
def file_lock(path: Path, blocking: bool = True) -> Iterator[bool]:
    """Hold a POSIX advisory lock; nonblocking callers receive False when busy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _resolve_file(value: Any, owner: Path) -> Path:
    candidate = Path(str(value)).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [
        Path.cwd() / candidate,
        owner.parent / candidate,
    ]
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"Referenced file not found: {value!r}")


def _resolve_directory(value: Any, owner: Path) -> Path:
    if value in (None, ""):
        return REPOSITORY_ROOT
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    for path in (Path.cwd() / candidate, owner.parent / candidate):
        if path.is_dir():
            return path.resolve()
    return (owner.parent / candidate).resolve()


def init_run(
    run_dir: Path,
    settings_paths: list[Path],
    general_path: Path,
    loss_paths: list[Path],
    base_seed: int,
) -> dict[str, Any]:
    """Create the small saved run configuration used by later commands."""
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    layout = layout_for_run(run_dir, create=True)
    with file_lock(layout.locks / "run-setup.lock"):
        if layout.manifest.is_file():
            return load_run(run_dir)[1]
        for stage in (
            layout.inputs,
            layout.designs,
            layout.selections,
            layout.evaluations,
            layout.summary,
        ):
            stage.mkdir(parents=True, exist_ok=True)
        if not settings_paths or len(settings_paths) != len(loss_paths):
            raise ValueError("The first run needs at least one --context SETTINGS LOSS")
        if isinstance(base_seed, bool) or base_seed < 0:
            raise ValueError("--base-seed must be a non-negative integer")
        sources = [*settings_paths, general_path, *loss_paths]
        for source in sources:
            if not source.resolve().is_file():
                raise FileNotFoundError(source)

        general = load_json(general_path)
        if not isinstance(general, dict):
            raise ValueError("--advanced must contain a JSON object")
        _copy_atomic(general_path.resolve(), layout.inputs / "general.json")
        contexts: list[dict[str, Any]] = []
        names: set[str] = set()
        for index, (settings_path, loss_path) in enumerate(
            zip(settings_paths, loss_paths)
        ):
            settings_path = settings_path.resolve()
            loss_path = loss_path.resolve()
            settings = load_json(settings_path)
            if not isinstance(settings, dict):
                raise ValueError(f"Settings must contain a JSON object: {settings_path}")
            if "role" in settings:
                raise ValueError(
                    f"{settings_path}: role belongs in the paired "
                    "--context loss file, not in its target settings"
                )
            if "lengths" in settings:
                raise ValueError(
                    f"{settings_path}: lengths belongs in the general "
                    "--advanced file, not in target settings"
                )
            loss = load_json(loss_path)
            if not isinstance(loss, dict):
                raise ValueError(f"Loss file must contain a JSON object: {loss_path}")
            name = str(settings.get("binder_name", "")).strip()
            role = str(loss.get("role", "")).strip().lower()
            if not name or _slug(name) != name or name in names:
                raise ValueError(f"binder_name must be unique and filesystem-safe: {name!r}")
            if role not in {"target", "offtarget"}:
                raise ValueError(f"Unsupported context role {role!r} for {name}")
            if index == 0 and role != "target":
                raise ValueError("The first context must have role 'target'")
            names.add(name)
            pdb = _resolve_file(settings.get("starting_pdb"), settings_path)
            folder = Path(layout.inputs.name) / "contexts" / f"{index:02d}_{name}"
            saved = {
                "name": name,
                "role": role,
                "settings": str(folder / "settings.json"),
                "loss": str(folder / "loss.json"),
                "pdb": str(folder / "target.pdb"),
            }
            _copy_atomic(settings_path, run_dir / saved["settings"])
            _copy_atomic(loss_path, run_dir / saved["loss"])
            _copy_atomic(pdb, run_dir / saved["pdb"])
            contexts.append(saved)

        manifest = {
            "layout_version": CURRENT_LAYOUT_VERSION,
            "base_seed": int(base_seed),
            "requested_designs": 0,
            "advanced": str(Path(layout.inputs.name) / "general.json"),
            "contexts": contexts,
            "af_params_dir": str(
                _resolve_directory(general.get("af_params_dir"), general_path)
            ),
        }
        atomic_write_json(layout.manifest, manifest)
        return manifest


def load_run(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    run_dir = run_dir.resolve()
    try:
        layout = layout_for_run(run_dir)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{error}. Supply --context and --advanced on the first design command."
        ) from error
    value = load_json(layout.manifest)
    if not isinstance(value, dict) or not value.get("contexts"):
        raise ValueError(f"Invalid run configuration: {layout.manifest}")
    if not layout.legacy and value.get("layout_version") != CURRENT_LAYOUT_VERSION:
        raise ValueError(
            f"Unsupported run layout version in {layout.manifest}: "
            f"{value.get('layout_version')!r}"
        )
    return run_dir, value


def ensure_run(
    run_dir: Path,
    settings_paths: list[Path] | None = None,
    general_path: Path | None = None,
    loss_paths: list[Path] | None = None,
    base_seed: int = 0,
) -> tuple[Path, dict[str, Any]]:
    run_dir = run_dir.resolve()
    try:
        layout_for_run(run_dir)
    except FileNotFoundError:
        pass
    else:
        return load_run(run_dir)
    if not settings_paths or general_path is None or not loss_paths:
        raise ValueError(
            "The first design command needs --context SETTINGS LOSS and --advanced"
        )
    init_run(run_dir, settings_paths, general_path, loss_paths, base_seed)
    return load_run(run_dir)


def update_design_target(
    run_dir: Path, manifest: dict[str, Any], requested: int
) -> int:
    if requested <= 0:
        raise ValueError("--num-designs must be a positive integer")
    run_dir = run_dir.resolve()
    layout = layout_for_run(run_dir)
    with file_lock(layout.locks / "run-total.lock"):
        current = load_json(layout.manifest)
        effective = max(int(current.get("requested_designs", 0)), requested)
        if effective != current.get("requested_designs"):
            current["requested_designs"] = effective
            atomic_write_json(layout.manifest, current)
    manifest["requested_designs"] = effective
    return effective


def requested_designs(run_dir: Path, manifest: dict[str, Any] | None = None) -> int:
    layout = layout_for_run(run_dir)
    return int(load_json(layout.manifest).get("requested_designs", 0))


def trajectory_parameters(
    base_seed: int, index: int, minimum: int, maximum: int
) -> tuple[int, int]:
    generator = np.random.default_rng(np.random.SeedSequence([base_seed, index]))
    return (
        int(generator.integers(0, 999999)),
        int(generator.integers(minimum, maximum + 1)),
    )


def trajectory_helicity(
    advanced: dict[str, Any], base_seed: int, index: int
) -> float:
    if advanced.get("random_helicity") is True:
        generator = np.random.default_rng(np.random.SeedSequence([base_seed, index, 17]))
        return round(float(generator.uniform(-3.0, 1.0)), 2)
    return float(advanced.get("weights_helicity") or 0.0)


def _run_path(run_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def _binder_length_bounds(
    run_dir: Path, manifest: dict[str, Any]
) -> tuple[int, int]:
    general = load_json(_run_path(run_dir, manifest["advanced"]))
    lengths = general.get("lengths")
    if not isinstance(lengths, list) or not lengths:
        raise ValueError("General advanced settings needs a non-empty lengths list")
    values = [int(value) for value in lengths]
    if min(values) <= 0:
        raise ValueError("Binder lengths must be positive")
    return min(values), max(values)


def _design_directories(
    layout: RunLayout, index: int | None = None
) -> dict[str, str]:
    if not layout.legacy:
        if index is None:
            raise ValueError("A design index is required for layout version 2")
        base = layout.design_dir(index)
        work = base / ".work"
        paths = {
            "Trajectory": work / "structure",
            "Trajectory/Animation": work / "animation",
            "Trajectory/Clashing": work / "clashing",
            "Trajectory/LowConfidence": work / "low_confidence",
            "Trajectory/Pickle": work / "pickle",
            "Trajectory/Plots": work / "plots",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return {name: str(path) for name, path in paths.items()}

    result: dict[str, str] = {}
    for name in (
        "Trajectory",
        "Trajectory/Animation",
        "Trajectory/Clashing",
        "Trajectory/LowConfidence",
        "Trajectory/Pickle",
        "Trajectory/Plots",
    ):
        path = layout.root / name
        path.mkdir(parents=True, exist_ok=True)
        result[name] = str(path)
    return result


def _status_path(run_dir: Path, index: int) -> Path:
    return layout_for_run(run_dir).design_status(index)


def _design_id(
    manifest: dict[str, Any],
    index: int,
    minimum: int,
    maximum: int,
) -> tuple[str, int, int]:
    seed, length = trajectory_parameters(
        int(manifest["base_seed"]), index, minimum, maximum
    )
    name = manifest["contexts"][0]["name"]
    return f"{name}_t{index:05d}_l{length}_s{seed}", seed, length


def _trajectory_file(run_dir: Path, design_id: str, index: int) -> Path:
    return layout_for_run(run_dir).design_trajectory(index, design_id)


def _completed_design(run_dir: Path, index: int) -> dict[str, Any] | None:
    path = _status_path(run_dir, index)
    if not path.is_file():
        return None
    try:
        status = load_json(path)
        artifact = _run_path(run_dir, status["trajectory_pickle"])
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return status if status.get("status") == "complete" and artifact.is_file() else None


def _failure_csv(path: Path) -> None:
    atomic_write_text(
        path,
        "Trajectory_logits_pLDDT,Trajectory_softmax_pLDDT,"
        "Trajectory_one-hot_pLDDT,Trajectory_ste_pLDDT,Trajectory_final_pLDDT,"
        "Trajectory_Contacts,Trajectory_Clashes,Trajectory_WrongHotspot\n"
        "0,0,0,0,0,0,0,0\n",
    )


def _clear_design_outputs(
    run_dir: Path,
    design_id: str,
    index: int,
) -> None:
    layout = layout_for_run(run_dir)
    if not layout.legacy:
        directory = layout.design_dir(index)
        if directory.is_dir():
            shutil.rmtree(directory)
        return

    design_paths = _design_directories(layout)
    files = [
        _status_path(run_dir, index),
        Path(design_paths["Trajectory/Pickle"]) / f"{design_id}.pickle",
        _trajectory_file(run_dir, design_id, index),
        Path(design_paths["Trajectory/Animation"]) / f"{design_id}.html",
        layout.design_checks(index),
    ]
    for key in ("Trajectory", "Trajectory/LowConfidence", "Trajectory/Clashing"):
        files.append(Path(design_paths[key]) / f"{design_id}.pdb")
    for path in files:
        path.unlink(missing_ok=True)
    plots = Path(design_paths["Trajectory/Plots"])
    for path in plots.glob(f"{design_id}_*.png"):
        path.unlink()


def _trajectory_pdb(
    design_paths: dict[str, str], design_id: str
) -> Path | None:
    for key in ("Trajectory", "Trajectory/LowConfidence", "Trajectory/Clashing"):
        path = Path(design_paths[key]) / f"{design_id}.pdb"
        if path.is_file():
            return path
    return None


def _normalize_design_outputs(
    layout: RunLayout,
    index: int,
    design_id: str,
    design_paths: dict[str, str],
) -> tuple[Path, Path | None]:
    """Give version-2 design artifacts short names inside their design folder."""
    if layout.legacy:
        return (
            layout.design_trajectory(index, design_id),
            _trajectory_pdb(design_paths, design_id),
        )

    base = layout.design_dir(index)
    moves = [
        (
            Path(design_paths["Trajectory/Pickle"]) / f"{design_id}_trajectory.pickle",
            layout.design_trajectory(index, design_id),
        ),
        (
            Path(design_paths["Trajectory/Pickle"]) / f"{design_id}.pickle",
            layout.design_auxiliary(index, design_id),
        ),
        (
            Path(design_paths["Trajectory/Animation"]) / f"{design_id}.html",
            base / "animation.html",
        ),
    ]
    pdb = _trajectory_pdb(design_paths, design_id)
    if pdb is not None:
        moves.append((pdb, base / "structure.pdb"))
    plots = Path(design_paths["Trajectory/Plots"])
    for source in plots.glob(f"{design_id}_*.png"):
        moves.append((source, base / "plots" / source.name[len(design_id) + 1:]))
    for source, destination in moves:
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
    work = base / ".work"
    if work.is_dir():
        shutil.rmtree(work)
    structure = base / "structure.pdb"
    return layout.design_trajectory(index, design_id), (
        structure if structure.is_file() else None
    )


def _relative(path: Path | None, root: Path) -> str | None:
    return str(path.relative_to(root)) if path is not None else None


def run_design(
    run_dir: Path,
    num_designs: int,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    settings_paths: list[Path] | None = None,
    general_path: Path | None = None,
    loss_paths: list[Path] | None = None,
    base_seed: int = 0,
) -> dict[str, int]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    run_dir, manifest = ensure_run(
        run_dir, settings_paths, general_path, loss_paths, base_seed
    )
    total = update_design_target(run_dir, manifest, num_designs)
    minimum, maximum = _binder_length_bounds(run_dir, manifest)
    assigned = [index for index in range(total) if index % num_shards == shard_index]
    summary = {
        "requested": total,
        "assigned": len(assigned),
        "completed": 0,
        "skipped": 0,
        "busy": 0,
        "failed": 0,
    }
    pending = [index for index in assigned if _completed_design(run_dir, index) is None]
    summary["skipped"] = len(assigned) - len(pending)
    if not pending:
        return summary

    # GPU dependencies are imported only when unfinished work exists.
    try:
        import functions.colabdesign_utils as colabdesign_utils
        from functions.generic_utils import (
            check_jax_gpu,
            load_af2_models,
            load_json_settings,
            perform_advanced_settings_check,
        )
    except ModuleNotFoundError as error:
        if error.name == "colabdesign":
            raise RuntimeError(
                "ColabDesign is not installed; run install_odin_multi.sh or "
                "pip install -e ./ColabDesign --no-deps"
            ) from error
        raise

    settings_files = [_run_path(run_dir, item["settings"]) for item in manifest["contexts"]]
    loss_files = [_run_path(run_dir, item["loss"]) for item in manifest["contexts"]]
    settings_list, advanced_list, _ = load_json_settings(
        [str(path) for path in settings_files],
        str(_run_path(run_dir, manifest["advanced"])),
        [str(path) for path in loss_files],
    )
    for settings, context in zip(settings_list, manifest["contexts"]):
        settings["starting_pdb"] = str(_run_path(run_dir, context["pdb"]))
    settings_list[0]["design_path"] = str(run_dir)
    advanced_list = [
        perform_advanced_settings_check(copy.deepcopy(value), str(REPOSITORY_ROOT))
        for value in advanced_list
    ]
    params_dir = manifest.get("af_params_dir")
    if params_dir is None:
        params_dir = manifest.get("af2_defaults", {}).get("params_dir")
    if params_dir is None:
        raise ValueError("Run manifest is missing the AF2 design parameter directory")
    for advanced in advanced_list:
        advanced["af_params_dir"] = params_dir
    check_jax_gpu()
    design_models, _, _ = load_af2_models(advanced_list[0]["use_multimer_design"])
    layout = layout_for_run(run_dir)
    failures: list[str] = []

    for index in pending:
        lock_folder = "Design" if layout.legacy else "design"
        lock_path = layout.locks / lock_folder / f"t{index:05d}.lock"
        with file_lock(lock_path, blocking=False) as acquired:
            if not acquired:
                summary["busy"] += 1
                continue
            if _completed_design(run_dir, index) is not None:
                summary["skipped"] += 1
                continue
            design_id, seed, length = _design_id(manifest, index, minimum, maximum)
            _clear_design_outputs(run_dir, design_id, index)
            design_paths = _design_directories(layout, index)
            failure_path = layout.design_checks(index)
            _failure_csv(failure_path)
            status = {
                "status": "running",
                "design_index": index,
                "design_id": design_id,
                "seed": seed,
                "length": length,
                "helicity": trajectory_helicity(
                    advanced_list[0], int(manifest["base_seed"]), index
                ),
            }
            atomic_write_json(_status_path(run_dir, index), status)
            try:
                model = colabdesign_utils.binder_hallucination(
                    design_id,
                    copy.deepcopy(settings_list[0]),
                    copy.deepcopy(advanced_list[0]),
                    copy.deepcopy(settings_list[1:]),
                    copy.deepcopy(advanced_list[1:]),
                    length,
                    seed,
                    status["helicity"],
                    design_paths,
                    str(failure_path),
                    design_models,
                )
                artifact, pdb = _normalize_design_outputs(
                    layout, index, design_id, design_paths
                )
                if not artifact.is_file():
                    raise FileNotFoundError(f"Trajectory pickle was not saved: {artifact}")
                if advanced_list[0].get("remove_unrelaxed_trajectory", False):
                    if pdb is not None:
                        pdb.unlink()
                    pdb = None
                atomic_write_json(_status_path(run_dir, index), {
                    **status,
                    "status": "complete",
                    "termination": str(model.aux.get("log", {}).get("terminate", "")),
                    "trajectory_pickle": _relative(artifact, run_dir),
                    "trajectory_pdb": _relative(pdb, run_dir),
                })
                summary["completed"] += 1
            except Exception as error:
                atomic_write_json(_status_path(run_dir, index), {
                    **status,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                })
                summary["failed"] += 1
                failures.append(design_id)
    if failures:
        raise RuntimeError(
            f"{len(failures)} trajectory attempt(s) failed: {', '.join(failures[:5])}"
        )
    return summary


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sequence(frame: Any) -> str:
    array = np.asarray(frame)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2 and array.shape[1] == len(AA_ORDER):
        indices = array.argmax(-1)
    elif array.ndim == 1 and np.issubdtype(array.dtype, np.integer):
        indices = array
    else:
        raise ValueError(f"Unsupported saved sequence shape {array.shape}")
    if np.any(indices < 0) or np.any(indices >= len(AA_ORDER)):
        raise ValueError("Saved sequence has an invalid amino-acid index")
    return "".join(AA_ORDER[int(index)] for index in indices)


def _trajectory_frames(value: Any) -> dict[str, list[Any]]:
    if not isinstance(value, dict):
        raise ValueError("Trajectory context is missing")
    keys = ("seq", "xyz", "plddt", "pae", "ptm", "i_ptm", "iteration", "stage")
    if any(not isinstance(value.get(key), list) for key in keys):
        raise ValueError("Trajectory is missing a saved frame list")
    lengths = {len(value[key]) for key in keys}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("Trajectory frame lists are empty or unaligned")
    return value


def _interface_pae(frame: Any, binder_length: int) -> float | None:
    array = np.asarray(frame).squeeze()
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        return None
    target_length = array.shape[0] - binder_length
    if target_length <= 0:
        return None
    value = (
        float(array[:target_length, target_length:].mean())
        + float(array[target_length:, :target_length].mean())
    ) / 2.0
    return value if math.isfinite(value) else None


def _aligned_frames(
    reference: dict[str, list[Any]],
    trajectories: list[tuple[str, dict[str, list[Any]]]],
) -> None:
    reference_alignment = list(zip(reference["iteration"], reference["stage"]))
    for name, trajectory in trajectories:
        alignment = list(zip(trajectory["iteration"], trajectory["stage"]))
        if alignment != reference_alignment:
            raise ValueError(f"Context {name!r} saved frames are not aligned")


def _hard_frames(trajectory: dict[str, list[Any]]) -> list[int]:
    indices = [
        index
        for index, stage in enumerate(trajectory["stage"])
        if str(stage) in HARD_STAGES
    ]
    if not indices:
        saved = sorted({str(stage) for stage in trajectory["stage"]})
        raise ValueError(
            "No saved frame with a discrete sequence; saved stages were "
            f"{saved or ['none']}, expected one of {sorted(HARD_STAGES)}"
        )
    return indices


def _frame_iteration(trajectory: dict[str, list[Any]], index: int) -> int:
    value = trajectory["iteration"][index]
    return index if value is None else int(value)


def _choose_target_frame(
    targets: list[tuple[str, dict[str, list[Any]]]],
    method: str,
    binder_length: int,
) -> tuple[int, int, str, float | int, float | None, str | None]:
    """Select a hard frame using the worst-performing target context."""
    reference = targets[0][1]
    _aligned_frames(reference, targets[1:])
    eligible = _hard_frames(reference)

    def iteration(index: int) -> int:
        return _frame_iteration(reference, index)

    if method == "last":
        selected = max(eligible, key=lambda index: (iteration(index), index))
        score: float | int = iteration(selected)
        worst_value = None
        worst_name = None
    elif method == "best_i_pae":
        scored: list[tuple[float, int, str]] = []
        for index in eligible:
            values = [
                (_interface_pae(trajectory["pae"][index], binder_length), name)
                for name, trajectory in targets
            ]
            if any(value is None for value, _ in values):
                continue
            worst_value, worst_name = max(
                ((float(value), name) for value, name in values),
                key=lambda item: item[0],
            )
            scored.append((worst_value, index, worst_name))
        if not scored:
            raise ValueError("No hard frame has interface PAE for every target")
        score, selected, worst_name = min(
            scored,
            key=lambda item: (item[0], -iteration(item[1]), -item[1]),
        )
        worst_value = score
    elif method == "best_i_ptm":
        scored = []
        for index in eligible:
            values = [
                (_finite(trajectory["i_ptm"][index]), name)
                for name, trajectory in targets
            ]
            if any(value is None for value, _ in values):
                continue
            worst_value, worst_name = min(
                ((float(value), name) for value, name in values),
                key=lambda item: item[0],
            )
            scored.append((worst_value, index, worst_name))
        if not scored:
            raise ValueError("No hard frame has interface pTM for every target")
        score, selected, worst_name = max(
            scored,
            key=lambda item: (item[0], iteration(item[1]), item[1]),
        )
        worst_value = score
    else:
        raise ValueError(f"Unsupported target selection method {method!r}")
    return (
        selected,
        iteration(selected),
        str(reference["stage"][selected]),
        score,
        worst_value,
        worst_name,
    )


def _choose_specificity_frame(
    targets: list[tuple[str, dict[str, list[Any]]]],
    offtargets: list[tuple[str, dict[str, list[Any]]]],
    binder_length: int,
) -> tuple[int, int, str, float, float, str, float, float, str]:
    """Select the hard frame with the best clipped off-target/target ratio."""
    reference = targets[0][1]
    _aligned_frames(reference, targets[1:] + offtargets)
    eligible = _hard_frames(reference)

    def iteration(index: int) -> int:
        return _frame_iteration(reference, index)

    candidates: list[tuple[float, int, float, str, float, float, str]] = []
    for index in eligible:
        target_values = [
            (_interface_pae(trajectory["pae"][index], binder_length), name)
            for name, trajectory in targets
        ]
        offtarget_values = [
            (_interface_pae(trajectory["pae"][index], binder_length), name)
            for name, trajectory in offtargets
        ]
        if (
            any(value is None for value, _ in target_values)
            or any(value is None for value, _ in offtarget_values)
        ):
            continue
        worst_target_i_pae, worst_target_name = max(
            ((float(value), name) for value, name in target_values),
            key=lambda item: item[0],
        )
        strongest_offtarget_i_pae, strongest_offtarget_name = min(
            ((float(value), name) for value, name in offtarget_values),
            key=lambda item: item[0],
        )
        if worst_target_i_pae <= 0:
            continue
        clipped_offtarget_i_pae = min(
            strongest_offtarget_i_pae, OFFTARGET_I_PAE_CAP
        )
        score = clipped_offtarget_i_pae / worst_target_i_pae
        if math.isfinite(score):
            candidates.append((
                score,
                index,
                worst_target_i_pae,
                worst_target_name,
                strongest_offtarget_i_pae,
                clipped_offtarget_i_pae,
                strongest_offtarget_name,
            ))
    if not candidates:
        raise ValueError(
            "No hard frame has valid target/off-target interface PAE"
        )

    chosen = max(
        candidates,
        key=lambda item: (item[0], iteration(item[1]), item[1]),
    )
    (
        score,
        selected,
        worst_target_i_pae,
        worst_target_name,
        strongest_offtarget_i_pae,
        clipped_offtarget_i_pae,
        strongest_offtarget_name,
    ) = chosen
    return (
        selected,
        iteration(selected),
        str(reference["stage"][selected]),
        score,
        worst_target_i_pae,
        worst_target_name,
        strongest_offtarget_i_pae,
        clipped_offtarget_i_pae,
        strongest_offtarget_name,
    )


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(
        {key: "" if value is None else value for key, value in row.items()}
        for row in rows
    )
    return buffer.getvalue()


def _save_selection_figure(fig: Any, path: Path, *, dpi: int | None = None) -> None:
    """Atomically save one selection figure."""
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


def _plot_selection_overview(
    output: Path, method: str, rows: list[dict[str, Any]]
) -> int:
    """Plot the metric that caused one frame to be selected per design."""
    figures = output / "figures"
    if figures.is_dir():
        shutil.rmtree(figures)
    selected = [row for row in rows if row["selection_status"] == "selected"]
    if not selected:
        return 0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    teal = "#2BBAAC"
    charcoal = "#42403F"
    mid_grey = "#828080"
    light_grey = "#C2C0BF"
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.0,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.0,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 5.3,
        "legend.frameon": False,
        "axes.edgecolor": charcoal,
        "axes.linewidth": 0.7,
        "axes.titlepad": 3.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "svg.fonttype": "none",
    }

    def clean_axis(axis: Any) -> None:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(
            direction="out", pad=1.5, colors=charcoal,
            width=0.65, length=3.0,
        )
        axis.grid(False)

    with plt.rc_context(style):
        if method == "best_clipped_i_pae_ratio":
            fig, axis = plt.subplots(figsize=(3.5, 2.8))
            x = np.asarray(
                [float(row["worst_target_i_pae"]) for row in selected]
            )
            y = np.asarray(
                [float(row["strongest_offtarget_i_pae"]) for row in selected]
            )
            ratio_cutoff = 1.5
            target_cutoff = 7.5
            passed = (x < target_cutoff) & (
                y / np.maximum(x, 1e-8) >= ratio_cutoff
            )
            shade_x = np.linspace(0.0, target_cutoff, 100)
            axis.fill_between(
                shade_x, ratio_cutoff * shade_x, 30.0,
                color=teal, alpha=0.10, lw=0, zorder=0,
            )
            line_x = np.linspace(0.0, 20.0, 100)
            axis.plot(
                line_x, ratio_cutoff * line_x,
                color=mid_grey, ls="--", lw=0.7, zorder=1,
            )
            axis.axvline(
                target_cutoff, color=mid_grey, ls=":", lw=0.7, zorder=1
            )
            axis.axhline(
                OFFTARGET_I_PAE_CAP, color=light_grey,
                ls=":", lw=0.7, zorder=1,
            )
            axis.scatter(
                x[~passed], y[~passed], s=18, facecolors="white",
                edgecolors=charcoal, linewidths=0.65,
                label="Outside reporting gate", zorder=3,
            )
            axis.scatter(
                x[passed], y[passed], s=18, color=teal,
                edgecolors="white", linewidths=0.35,
                label="Passes reporting gate", zorder=4,
            )
            axis.set_xlim(0.0, 25.0)
            axis.set_ylim(0.0, 30.0)
            axis.set_xticks([0, 5, 10, 15, 20, 25])
            axis.set_yticks([0, 5, 10, 15, 20, 25, 30])
            axis.set_xlabel("Worst target interface PAE (Å)")
            axis.set_ylabel("Strongest off-target interface PAE (Å)")
            axis.set_title("Specificity frame selection", loc="left", pad=11)
            axis.text(
                0.98, 0.035,
                f"{int(passed.sum())}/{len(selected)} pass target < 7.5 Å\n"
                "and off-target / target ≥ 1.5",
                transform=axis.transAxes, color=charcoal, fontsize=5.1,
                va="bottom", ha="right",
            )
            axis.text(
                12.0, 18.25, "ratio = 1.5", color=mid_grey,
                fontsize=4.8, rotation=46,
            )
            axis.text(
                15.2, 15.45, "selection clip = 15 Å",
                color=mid_grey, fontsize=4.8,
            )
            axis.legend(loc="upper left", handletextpad=0.35)
        else:
            fig, axis = plt.subplots(figsize=(3.5, 2.5))
            x = np.asarray([int(row["design_index"]) for row in selected])
            if method == "best_i_ptm":
                y = np.asarray(
                    [float(row["worst_target_i_ptm"]) for row in selected]
                )
                cutoff = 0.5
                passed = y > cutoff
                title = "Target-confidence frame selection"
                ylabel = "Selected worst-target iPTM"
                annotation = f"{int(passed.sum())}/{len(selected)} above 0.5"
            elif method == "best_i_pae":
                y = np.asarray(
                    [float(row["worst_target_i_pae"]) for row in selected]
                )
                cutoff = 7.5
                passed = y < cutoff
                title = "Target-interface frame selection"
                ylabel = "Selected worst-target interface PAE (Å)"
                annotation = f"{int(passed.sum())}/{len(selected)} below 7.5 Å"
            else:
                y = np.asarray([int(row["iteration"]) for row in selected])
                cutoff = None
                passed = np.ones(len(selected), dtype=bool)
                title = "Final hard-frame selection"
                ylabel = "Selected iteration"
                annotation = f"{len(selected)} selected designs"
            if cutoff is not None:
                axis.axhline(
                    cutoff, color=mid_grey, ls=":", lw=0.7, zorder=1
                )
                axis.scatter(
                    x[~passed], y[~passed], s=18, facecolors="white",
                    edgecolors=charcoal, linewidths=0.65, zorder=3,
                )
            axis.scatter(
                x[passed], y[passed], s=18, color=teal,
                edgecolors="white", linewidths=0.35, zorder=4,
            )
            axis.set_xlabel("Design index")
            axis.set_ylabel(ylabel)
            axis.set_title(title, loc="left", pad=11)
            axis.text(
                0.98, 0.96, annotation, transform=axis.transAxes,
                color=charcoal, fontsize=5.1, va="top", ha="right",
            )

        axis.text(
            0.0, 1.012, f"{method}  ·  n={len(selected)}",
            transform=axis.transAxes, color=mid_grey, fontsize=5.0,
            va="bottom", ha="left",
        )
        clean_axis(axis)
        fig.tight_layout()
        stem = figures / "selection_overview"
        _save_selection_figure(fig, stem.with_suffix(".svg"))
        _save_selection_figure(fig, stem.with_suffix(".png"), dpi=300)
        plt.close(fig)
    return 2


def run_selection(run_dir: Path, method: str) -> dict[str, Any]:
    if method not in SELECTION_METHODS:
        raise ValueError(f"Unknown selection method {method!r}")
    run_dir, manifest = load_run(run_dir)
    total = requested_designs(run_dir, manifest)
    if total <= 0:
        raise ValueError("No designs have been requested")
    target_contexts = [
        item for item in manifest["contexts"] if item["role"] == "target"
    ]
    offtarget_contexts = [
        item for item in manifest["contexts"] if item["role"] == "offtarget"
    ]
    if not target_contexts:
        raise ValueError("Selection requires at least one target context")
    if method == "best_clipped_i_pae_ratio" and not offtarget_contexts:
        raise ValueError(f"{method} requires at least one off-target context")
    selection_name = method
    layout = layout_for_run(run_dir)
    lock_folder = "Selections" if layout.legacy else "selections"
    lock_path = layout.locks / lock_folder / f"{selection_name}.lock"
    with file_lock(lock_path, blocking=False) as acquired:
        if not acquired:
            raise RuntimeError(f"Selection {selection_name!r} is busy")
        rows: list[dict[str, Any]] = []
        for index in range(total):
            row: dict[str, Any] = {
                "design_index": index,
                "design_id": None,
                "selection_status": "not_run",
                "termination": None,
                "iteration": None,
                "frame_index": None,
                "stage": None,
                "score": None,
                "sequence": None,
                "length": None,
                "seed": None,
                "worst_target_i_pae": None,
                "worst_target_i_ptm": None,
                "worst_target_context": None,
                "strongest_offtarget_i_pae": None,
                "strongest_offtarget_i_pae_clipped": None,
                "strongest_offtarget_context": None,
                "offtarget_i_pae_cap": None,
                "error": None,
            }
            try:
                status_path = _status_path(run_dir, index)
                if not status_path.is_file():
                    raise FileNotFoundError("Design has not run")
                status = load_json(status_path)
                row.update({
                    "design_id": status.get("design_id"),
                    "termination": status.get("termination"),
                    "length": status.get("length"),
                    "seed": status.get("seed"),
                })
                if status.get("status") != "complete":
                    row["selection_status"] = f"design_{status.get('status', 'unknown')}"
                    raise ValueError(status.get("error") or "Design is incomplete")
                row["selection_status"] = "artifact_error"
                artifact = _run_path(run_dir, status["trajectory_pickle"])
                with artifact.open("rb") as handle:
                    payload = pickle.load(handle)
                targets = [
                    (item["name"], _trajectory_frames(payload.get(item["name"])))
                    for item in target_contexts
                ]
                trajectory = targets[0][1]
                binder_length = int(status["length"])
                if method == "best_clipped_i_pae_ratio":
                    offtargets = [
                        (item["name"], _trajectory_frames(payload.get(item["name"])))
                        for item in offtarget_contexts
                    ]
                    (
                        selected,
                        iteration,
                        stage,
                        score,
                        worst_target_i_pae,
                        worst_target_name,
                        strongest_offtarget_i_pae,
                        clipped_offtarget_i_pae,
                        strongest_offtarget_name,
                    ) = _choose_specificity_frame(
                        targets, offtargets, binder_length
                    )
                    row.update({
                        "worst_target_i_pae": worst_target_i_pae,
                        "worst_target_context": worst_target_name,
                        "strongest_offtarget_i_pae": strongest_offtarget_i_pae,
                        "strongest_offtarget_i_pae_clipped": clipped_offtarget_i_pae,
                        "strongest_offtarget_context": strongest_offtarget_name,
                        "offtarget_i_pae_cap": OFFTARGET_I_PAE_CAP,
                    })
                else:
                    (
                        selected,
                        iteration,
                        stage,
                        score,
                        worst_target_value,
                        worst_target_name,
                    ) = _choose_target_frame(targets, method, binder_length)
                    row["worst_target_context"] = worst_target_name
                    if method == "best_i_pae":
                        row["worst_target_i_pae"] = worst_target_value
                    elif method == "best_i_ptm":
                        row["worst_target_i_ptm"] = worst_target_value
                sequence = _sequence(trajectory["seq"][selected])
                if len(sequence) != binder_length:
                    raise ValueError("Selected sequence has the wrong binder length")
                row.update({
                    "selection_status": "selected",
                    "iteration": iteration,
                    "frame_index": selected,
                    "stage": stage,
                    "score": score,
                    "sequence": sequence,
                })
            except Exception as error:
                row["error"] = str(error)
            rows.append(row)

        output = layout.selection_dir(selection_name)
        selected_rows = [
            row for row in rows if row["selection_status"] == "selected"
        ]
        fields = [
            "design_index",
            "design_id",
            "selection_status",
            "termination",
            "iteration",
            "frame_index",
            "stage",
            "score",
            "sequence",
            "length",
            "seed",
            "worst_target_i_pae",
            "worst_target_i_ptm",
            "worst_target_context",
            "strongest_offtarget_i_pae",
            "strongest_offtarget_i_pae_clipped",
            "strongest_offtarget_context",
            "offtarget_i_pae_cap",
            "error",
        ]
        atomic_write_text(
            layout.selection_csv(selection_name), _csv_text(rows, fields)
        )
        atomic_write_text(
            output / "sequences.fasta",
            "".join(
                f">{row['design_id']} iteration={row['iteration']} "
                f"stage={row['stage']}\n{row['sequence']}\n"
                for row in selected_rows
            ),
        )
        atomic_write_json(output / "selection.json", {
            "status": "complete",
            "selection_name": selection_name,
            "method": method,
            "targets": [item["name"] for item in target_contexts],
            "offtargets": [item["name"] for item in offtarget_contexts],
            "requested_designs": total,
            "selected": len(selected_rows),
            "rows": rows,
        })
        figures = _plot_selection_overview(output, method, rows)
    return {
        "selection_name": selection_name,
        "requested": total,
        "selected": len(selected_rows),
        "figures": figures,
        "figures_dir": str(output / "figures"),
    }


def _emit(name: str, result: dict[str, Any]) -> None:
    print(f"{name}: " + ", ".join(f"{key}={value}" for key, value in result.items()))


def _evaluator_functions(name: str) -> tuple[Any, Any]:
    if name == "af2":
        from evaluators.af2 import collect_evaluation, run_evaluation
    elif name == "af3":
        from evaluators.af3 import collect_evaluation, run_evaluation
    else:
        raise ValueError(f"Unknown evaluator {name!r}")
    return run_evaluation, collect_evaluation


def _add_sharding(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)


def _add_setup(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--context",
        action="append",
        nargs=2,
        type=Path,
        metavar=("SETTINGS", "LOSS"),
        help="Target settings and paired context loss file; repeat per context",
    )
    parser.add_argument("--advanced", "-a", action="append", type=Path)
    parser.add_argument("--base-seed", type=int, default=0)


def _setup_values(
    args: argparse.Namespace,
) -> tuple[list[Path] | None, Path | None, list[Path] | None]:
    advanced = args.advanced or []
    if len(advanced) > 1:
        raise ValueError("--advanced takes exactly one file")
    contexts = args.context or []
    settings = [item[0] for item in contexts] or None
    loss_files = [item[1] for item in contexts] or None
    return settings, advanced[0] if advanced else None, loss_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate, select, and reevaluate Odin-Multi binder designs."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    design = commands.add_parser("design", help="Create or continue trajectories")
    design.add_argument("--run-dir", required=True, type=Path)
    design.add_argument("--num-designs", required=True, type=int)
    _add_setup(design)
    _add_sharding(design)

    select = commands.add_parser("select", help="Select one saved iteration")
    select.add_argument("--run-dir", required=True, type=Path)
    select.add_argument("--method", required=True, choices=SELECTION_METHODS)

    preprocess_af3 = commands.add_parser(
        "preprocess-af3", help="Cache AF3 target MSAs and templates"
    )
    preprocess_af3.add_argument(
        "--settings", "-s", action="append", required=True, type=Path
    )
    preprocess_af3.add_argument("--evaluator-config", required=True, type=Path)

    evaluate = commands.add_parser("evaluate", help="Run AF2 or AF3 reevaluation")
    evaluate.add_argument("--run-dir", required=True, type=Path)
    evaluate.add_argument("--selection", required=True)
    evaluate.add_argument("--evaluation-name", required=True)
    evaluate.add_argument(
        "--evaluator", choices=EVALUATORS, default=DEFAULT_EVALUATOR
    )
    evaluate.add_argument("--evaluator-config", required=True, type=Path)
    _add_sharding(evaluate)

    summarize = commands.add_parser(
        "summarize", help="Summarize original AF2 and available reevaluations"
    )
    summarize.add_argument("--run-dir", required=True, type=Path)
    summarize.add_argument("--evaluation-name")
    summarize.add_argument("--evaluator", choices=EVALUATORS)

    run = commands.add_parser("run", help="Design, select, and evaluate")
    run.add_argument("--run-dir", required=True, type=Path)
    run.add_argument("--num-designs", required=True, type=int)
    run.add_argument("--method", required=True, choices=SELECTION_METHODS)
    run.add_argument("--evaluation-name")
    run.add_argument(
        "--evaluator", choices=EVALUATORS, default=DEFAULT_EVALUATOR
    )
    run.add_argument("--evaluator-config", required=True, type=Path)
    _add_setup(run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "design":
            settings, advanced, loss_files = _setup_values(args)
            result = run_design(
                args.run_dir,
                args.num_designs,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                settings_paths=settings,
                general_path=advanced,
                loss_paths=loss_files,
                base_seed=args.base_seed,
            )
            result["output"] = str(layout_for_run(args.run_dir).designs)
            _emit("design", result)
        elif args.command == "select":
            result = run_selection(args.run_dir, args.method)
            result["output"] = str(
                layout_for_run(args.run_dir).selection_dir(result["selection_name"])
            )
            _emit("select", result)
        elif args.command == "preprocess-af3":
            from evaluators.af3 import preprocess_targets

            _emit(
                "preprocess-af3",
                preprocess_targets(
                    args.settings, config_path=args.evaluator_config
                ),
            )
        elif args.command == "evaluate":
            run_evaluation, _ = _evaluator_functions(args.evaluator)

            result = run_evaluation(
                args.run_dir,
                args.selection,
                args.evaluation_name,
                config_path=args.evaluator_config,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            result["output"] = str(
                layout_for_run(args.run_dir).evaluation_dir(
                    args.evaluator, args.evaluation_name
                )
            )
            _emit("evaluate", result)
        elif args.command == "summarize":
            from evaluators.summary import summarize_run

            result = summarize_run(
                args.run_dir,
                evaluator=args.evaluator,
                evaluation_name=args.evaluation_name,
            )
            layout = layout_for_run(args.run_dir)
            result.update({
                "report": str(layout.summary_report),
                "candidates_file": str(layout.candidates_csv),
                "structures": str(layout.candidate_structures),
                "figures_dir": str(layout.figures),
            })
            _emit("summarize", result)
        elif args.command == "run":
            settings, advanced, loss_files = _setup_values(args)
            design_result = run_design(
                args.run_dir,
                args.num_designs,
                settings_paths=settings,
                general_path=advanced,
                loss_paths=loss_files,
                base_seed=args.base_seed,
            )
            design_result["output"] = str(layout_for_run(args.run_dir).designs)
            _emit("design", design_result)
            selection = run_selection(args.run_dir, args.method)
            selection["output"] = str(
                layout_for_run(args.run_dir).selection_dir(
                    selection["selection_name"]
                )
            )
            _emit("select", selection)
            run_evaluation, _ = _evaluator_functions(args.evaluator)

            evaluation_name = args.evaluation_name or f"{args.evaluator}_default"
            evaluation_result = run_evaluation(
                args.run_dir,
                selection["selection_name"],
                evaluation_name,
                config_path=args.evaluator_config,
            )
            evaluation_result["output"] = str(
                layout_for_run(args.run_dir).evaluation_dir(
                    args.evaluator, evaluation_name
                )
            )
            _emit("evaluate", evaluation_result)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"odin-multi: error: {error}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
