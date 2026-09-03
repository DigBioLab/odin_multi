"""AlphaFold 3 target preprocessing and selected-sequence reevaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from odin_multi import (
    _csv_text,
    _resolve_file,
    _slug,
    atomic_write_json,
    atomic_write_text,
    file_lock,
    load_json,
    load_run,
)
from run_layout import layout_for_run


_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K",
}
_CONFIG_KEYS = {
    "python", "run_alphafold", "model_dir", "db_dir", "seeds",
    "target_cache_dir", "data_pipeline_flags", "extra_flags",
}
# Optional, so configurations written before interface scoring existed still
# load. Defaults keep the previous behaviour: no PyRosetta, no extra runtime.
_OPTIONAL_CONFIG_KEYS = {"interface_metrics": False, "interface_relax": True}
_CACHE_SCHEMA = 1
_RESERVED_FLAGS = {
    "input_dir", "json_path", "output_dir", "model_dir", "db_dir",
    "run_data_pipeline", "run_inference",
}


from .interface import INTERFACE_EXTRA_FIELDS, INTERFACE_FIELDS, score_structure
from .ipsae import compute_ipsae_min


def _configured_path(value: Any, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"AF3 {label} must be a path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _resolve(value: Any, base: Path, label: str, directory: bool = False) -> Path:
    path = _configured_path(value, base, label)
    if not (path.is_dir() if directory else path.is_file()):
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"AF3 {label} {kind} not found: {path}")
    return path


def normalize_af3_config(
    config_path: Path | None, *, require_db: bool = False,
    require_model: bool = True,
) -> dict[str, Any]:
    if config_path is None:
        raise ValueError("AF3 requires --evaluator-config")
    config_path = config_path.resolve()
    raw = load_json(config_path)
    if not isinstance(raw, dict):
        raise ValueError("AF3 evaluator configuration must be a JSON object")
    if "run_data_pipeline" in raw:
        raise ValueError(
            "AF3 run_data_pipeline is no longer configurable; run "
            "preprocess-af3 before evaluate"
        )
    unknown = sorted(set(raw) - _CONFIG_KEYS - set(_OPTIONAL_CONFIG_KEYS))
    if unknown:
        raise ValueError(f"Unknown AF3 evaluator setting(s): {', '.join(unknown)}")

    base = config_path.parent
    python = str(raw.get("python") or sys.executable)
    if os.sep in python:
        python = str(_resolve(python, base, "python"))
    else:
        found = shutil.which(python)
        if found is None:
            raise FileNotFoundError(f"AF3 python not found on PATH: {python}")
        python = found

    seeds = raw.get("seeds", [1])
    if (
        not isinstance(seeds, list) or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("AF3 seeds must be a non-empty list of integers")
    seeds = list(dict.fromkeys(seeds))

    data_pipeline_flags = raw.get("data_pipeline_flags", {})
    if not isinstance(data_pipeline_flags, dict):
        raise ValueError("AF3 data_pipeline_flags must be a JSON object")
    extra_flags = raw.get("extra_flags", {})
    if not isinstance(extra_flags, dict):
        raise ValueError("AF3 extra_flags must be a JSON object")
    cache_value = raw.get("target_cache_dir")
    if not isinstance(cache_value, str) or not cache_value.strip():
        raise ValueError("AF3 target_cache_dir must be a path")
    cache_dir = Path(cache_value).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = (base / cache_dir).resolve()
    interface_flags: dict[str, bool] = {}
    for key, default in _OPTIONAL_CONFIG_KEYS.items():
        value = raw.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"AF3 {key} must be true or false")
        interface_flags[key] = value
    return {
        "python": python,
        "run_alphafold": str(_resolve(raw.get("run_alphafold"), base, "run_alphafold")),
        "model_dir": str(
            _resolve(raw.get("model_dir"), base, "model_dir", directory=True)
            if require_model else _configured_path(raw.get("model_dir"), base, "model_dir")
        ),
        "db_dir": str(
            _resolve(raw.get("db_dir"), base, "db_dir", directory=True)
            if require_db else _configured_path(raw.get("db_dir"), base, "db_dir")
        ),
        "target_cache_dir": str(cache_dir),
        "seeds": seeds,
        "data_pipeline_flags": data_pipeline_flags,
        "extra_flags": extra_flags,
        **interface_flags,
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


def _sequence(value: Any) -> str:
    sequence = str(value).strip().upper()
    if not sequence or any(amino_acid not in _AMINO_ACIDS for amino_acid in sequence):
        raise ValueError("Selected sequence must use the 20 canonical amino acids")
    return sequence


def _chain_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        chains = [str(item).strip() for item in value]
    else:
        text = str(value or "").strip()
        if "," in text:
            chains = [item.strip() for item in text.split(",")]
        elif " " in text:
            chains = text.split()
        else:
            chains = list(text)
    if not chains or any(len(chain) != 1 for chain in chains):
        raise ValueError("AF3 target chains must be one-character PDB chain IDs")
    if len(chains) != len(set(chains)):
        raise ValueError("AF3 target chains contain duplicates")
    return chains


def _pdb_chains(pdb_path: Path, chains: list[str]) -> list[dict[str, str]]:
    residues: dict[str, list[str]] = {chain: [] for chain in chains}
    seen: set[tuple[str, str, str]] = set()
    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or len(line) < 27:
                continue
            chain = line[21].strip()
            key = (chain, line[22:26].strip(), line[26].strip())
            if line[12:16].strip() != "CA" or chain not in residues or key in seen:
                continue
            seen.add(key)
            residues[chain].append(_THREE_TO_ONE.get(line[17:20].strip().upper(), "X"))
    missing = [chain for chain, sequence in residues.items() if not sequence]
    if missing:
        raise ValueError(f"No residues found for chain(s) {missing} in {pdb_path}")
    return [
        {"chain_id": chain, "sequence": "".join(residues[chain])}
        for chain in chains
    ]


def _target_chains(run_dir: Path, context: dict[str, Any]) -> list[dict[str, str]]:
    settings = load_json(run_dir / context["settings"])
    return _pdb_chains(
        run_dir / context["pdb"], _chain_ids(settings.get("chains"))
    )


def _settings_target(settings_path: Path) -> tuple[str, list[dict[str, str]]]:
    settings_path = settings_path.resolve()
    settings = load_json(settings_path)
    if not isinstance(settings, dict):
        raise ValueError(f"Target settings must contain a JSON object: {settings_path}")
    name = str(settings.get("binder_name") or settings_path.stem).strip()
    return name, _pdb_chains(
        _resolve_file(settings.get("starting_pdb"), settings_path),
        _chain_ids(settings.get("chains")),
    )


def build_af3_input(
    name: str,
    sequence: str,
    target_chains: list[dict[str, Any]],
    seeds: list[int],
) -> tuple[dict[str, Any], str]:
    used = {chain["chain_id"] for chain in target_chains}
    binder_id = next(
        (candidate for candidate in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if candidate not in used),
        None,
    )
    if binder_id is None:
        raise ValueError("Could not assign an AF3 binder chain ID")
    proteins = []
    for chain in target_chains:
        protein = {
            key: value for key, value in chain.items() if key != "chain_id"
        }
        protein["id"] = chain["chain_id"]
        proteins.append({"protein": protein})
    proteins.append({"protein": {
        "id": binder_id, "sequence": sequence,
        "unpairedMsa": "", "pairedMsa": "", "templates": [],
    }})
    return ({
        "name": name,
        "modelSeeds": seeds,
        "sequences": proteins,
        "dialect": "alphafold3",
        "version": 2,
    }, binder_id)


def target_cache_key(target_chains: list[dict[str, str]]) -> str:
    payload = {
        "schema_version": _CACHE_SCHEMA,
        "sequences": [chain["sequence"] for chain in target_chains],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cache_contract(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": _CACHE_SCHEMA,
        "preprocessing": {
            "python": config["python"],
            "run_alphafold": config["run_alphafold"],
            "db_dir": config["db_dir"],
            "data_pipeline_flags": config["data_pipeline_flags"],
        },
    }


def _ensure_cache(config: dict[str, Any], *, create: bool) -> Path:
    root = Path(config["target_cache_dir"])
    if create:
        root.mkdir(parents=True, exist_ok=True)
    elif not root.is_dir():
        raise FileNotFoundError(
            f"AF3 target cache not found: {root}; run preprocess-af3 first"
        )
    metadata_path = root / "cache.json"
    expected = _cache_contract(config)
    with file_lock(root / ".locks" / "configure.lock"):
        if metadata_path.is_file():
            if load_json(metadata_path) != expected:
                raise ValueError(
                    f"AF3 target cache {root} uses different preprocessing settings; "
                    "use another target_cache_dir"
                )
        elif create:
            atomic_write_json(metadata_path, expected)
        else:
            raise FileNotFoundError(
                f"AF3 target cache metadata not found: {metadata_path}; "
                "run preprocess-af3 first"
            )
    return root


def _cache_entry(root: Path, key: str) -> Path:
    return root / "targets" / key


def _feature_files_exist(features: list[dict[str, Any]]) -> bool:
    for feature in features:
        for key in ("unpairedMsaPath", "pairedMsaPath"):
            value = feature.get(key)
            if value and not Path(value).is_file():
                return False
        for template in feature.get("templates", []):
            value = template.get("mmcifPath")
            if not value or not Path(value).is_file():
                return False
    return True


def _complete_cache(path: Path, sequences: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        metadata = load_json(path)
        features = metadata.get("features", [])
        if (
            metadata.get("status") == "complete"
            and metadata.get("sequences") == sequences
            and len(features) == len(sequences)
            and _feature_files_exist(features)
        ):
            return metadata
    except (OSError, TypeError, ValueError):
        pass
    return None


def _processed_json(output_dir: Path, name: str) -> Path:
    matches = sorted(output_dir.rglob(f"{name}_data.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one AF3 processed JSON for {name}, found {len(matches)}"
        )
    return matches[0]


def _materialize_features(
    processed_path: Path,
    target_chains: list[dict[str, str]],
    features_dir: Path,
) -> list[dict[str, Any]]:
    processed = load_json(processed_path)
    proteins = [
        item["protein"] for item in processed.get("sequences", [])
        if isinstance(item, dict) and isinstance(item.get("protein"), dict)
    ]
    sequences = [chain["sequence"] for chain in target_chains]
    if len(proteins) != len(sequences):
        raise ValueError("AF3 processed target has a different number of protein chains")

    features_dir.mkdir(parents=True, exist_ok=True)
    features: list[dict[str, Any]] = []
    for index, (protein, sequence) in enumerate(zip(proteins, sequences)):
        if protein.get("sequence") != sequence:
            raise ValueError("AF3 processed target sequence does not match its input")
        feature: dict[str, Any] = {"sequence": sequence}
        prefix = f"chain_{index:02d}"
        for field, path_field, suffix in (
            ("unpairedMsa", "unpairedMsaPath", "unpaired.a3m"),
            ("pairedMsa", "pairedMsaPath", "paired.a3m"),
        ):
            value = protein.get(field)
            if not isinstance(value, str):
                raise ValueError(f"AF3 processed target is missing {field}")
            if value:
                msa_path = (features_dir / f"{prefix}.{suffix}").resolve()
                atomic_write_text(msa_path, value)
                feature[path_field] = str(msa_path)
            else:
                feature[field] = ""

        templates = protein.get("templates")
        if not isinstance(templates, list):
            raise ValueError("AF3 processed target is missing templates")
        saved_templates = []
        for template_index, template in enumerate(templates):
            if not isinstance(template, dict) or not isinstance(template.get("mmcif"), str):
                raise ValueError("AF3 processed target contains an invalid template")
            template_path = (
                features_dir / f"{prefix}.template_{template_index:03d}.cif"
            ).resolve()
            atomic_write_text(template_path, template["mmcif"])
            saved_templates.append({
                "mmcifPath": str(template_path),
                "queryIndices": template.get("queryIndices"),
                "templateIndices": template.get("templateIndices"),
            })
        feature["templates"] = saved_templates
        features.append(feature)
    return features


def _preprocessing_input(
    key: str, target_chains: list[dict[str, str]]
) -> tuple[str, dict[str, Any]]:
    if len(target_chains) > 26:
        raise ValueError("AF3 preprocessing supports at most 26 target chains")
    name = f"af3_target_{key[:16]}"
    proteins = [
        {"protein": {
            "id": chr(ord("A") + index),
            "sequence": chain["sequence"],
        }}
        for index, chain in enumerate(target_chains)
    ]
    return name, {
        "name": name,
        "modelSeeds": [1],
        "sequences": proteins,
        "dialect": "alphafold3",
        "version": 1,
    }


def _preprocess_target(
    root: Path,
    config: dict[str, Any],
    settings_path: Path,
    source_name: str,
    target_chains: list[dict[str, str]],
    af3_env: dict[str, str],
) -> str:
    sequences = [chain["sequence"] for chain in target_chains]
    key = target_cache_key(target_chains)
    entry = _cache_entry(root, key)
    metadata_path = entry / "metadata.json"
    if _complete_cache(metadata_path, sequences):
        return "skipped"

    lock = root / ".locks" / "targets" / f"{key}.lock"
    with file_lock(lock, blocking=False) as acquired:
        if not acquired:
            return "busy"
        if _complete_cache(metadata_path, sequences):
            return "skipped"

        entry.mkdir(parents=True, exist_ok=True)
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=entry))
        input_dir = attempt / "input"
        output_dir = attempt / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        job_name, input_data = _preprocessing_input(key, target_chains)
        input_path = input_dir / "fold_input.json"
        atomic_write_json(input_path, input_data)
        command = _command(
            config, input_dir, output_dir,
            run_data_pipeline=True, run_inference=False,
        )
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, env=af3_env
        )
        stdout_path = attempt / "stdout.log"
        stderr_path = attempt / "stderr.log"
        atomic_write_text(stdout_path, completed.stdout)
        atomic_write_text(stderr_path, completed.stderr)
        if completed.returncode:
            atomic_write_json(entry / "failure.json", {
                "status": "failed",
                "source_settings": str(settings_path.resolve()),
                "error": f"AF3 preprocessing exited with code {completed.returncode}",
                "stdout": str(stdout_path.resolve()),
                "stderr": str(stderr_path.resolve()),
                "command": command,
            })
            raise RuntimeError(
                f"AF3 preprocessing exited with code {completed.returncode}"
            )

        processed_path = _processed_json(output_dir, job_name)
        features = _materialize_features(
            processed_path, target_chains, entry / "features"
        )
        atomic_write_json(metadata_path, {
            "status": "complete",
            "cache_key": key,
            "source_name": source_name,
            "source_settings": str(settings_path.resolve()),
            "source_chain_ids": [chain["chain_id"] for chain in target_chains],
            "sequences": sequences,
            "features": features,
            "processed_json": str(processed_path.resolve()),
            "stdout": str(stdout_path.resolve()),
            "stderr": str(stderr_path.resolve()),
            "runner_returncode": completed.returncode,
            "command": command,
        })
        return "completed"


def preprocess_targets(
    settings_paths: list[Path] | None, *, config_path: Path | None = None
) -> dict[str, int]:
    if not settings_paths:
        raise ValueError("preprocess-af3 requires at least one --settings file")
    config = normalize_af3_config(
        config_path, require_db=True, require_model=False
    )
    af3_env = _pinned_env()
    root = _ensure_cache(config, create=True)
    summary = {
        "targets": len(settings_paths), "completed": 0,
        "skipped": 0, "busy": 0, "failed": 0,
    }
    failures = []
    for settings_path in settings_paths:
        try:
            source_name, target_chains = _settings_target(settings_path)
            status = _preprocess_target(
                root, config, settings_path, source_name, target_chains,
                af3_env,
            )
            summary[status] += 1
        except Exception as error:
            summary["failed"] += 1
            failures.append(
                f"{settings_path}: {type(error).__name__}: {error}"
            )
    if failures:
        raise RuntimeError(
            f"{len(failures)} AF3 target preprocessing job(s) failed: "
            + "; ".join(failures[:3])
        )
    return summary


def _cached_target(
    root: Path, target_chains: list[dict[str, str]]
) -> tuple[str, list[dict[str, Any]]]:
    sequences = [chain["sequence"] for chain in target_chains]
    key = target_cache_key(target_chains)
    entry = _cache_entry(root, key)
    metadata = _complete_cache(entry / "metadata.json", sequences)
    if metadata is None:
        raise FileNotFoundError(
            f"AF3 target cache entry {key[:16]} is missing; run preprocess-af3 "
            "for this target settings file"
        )
    cached = []
    for chain, feature in zip(target_chains, metadata["features"]):
        cached.append({
            "chain_id": chain["chain_id"],
            **feature,
            "sequence": chain["sequence"],
        })
    return key, cached


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _binder_iptm(
    summary: dict[str, Any], token_chains: list[str], binder_id: str
) -> float | None:
    """Return AF3's mean cross-chain ipTM for the binder chain.

    AF3's scalar iptm covers every interface in the complex. For a multichain
    target that score can therefore be dominated by target-target interfaces.
    chain_iptm has one entry per chain, in the order the chains first occur in
    the token data, and isolates the binder's interfaces with the target
    complex.
    """
    values = summary.get("chain_iptm")
    if not isinstance(values, (list, tuple)):
        return None
    chain_ids = list(dict.fromkeys(token_chains))
    if len(values) != len(chain_ids) or binder_id not in chain_ids:
        return None
    value = values[chain_ids.index(binder_id)]
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def compute_af3_metrics(
    summary: dict[str, Any],
    confidences: dict[str, Any],
    binder_id: str,
    target_ids: Iterable[str] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for source, destination in (
        ("ptm", "ptm"), ("iptm", "global_i_ptm"),
        ("ranking_score", "ranking_score")
    ):
        value = summary.get(source)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metrics[destination] = float(value)

    plddt = confidences.get("atom_plddts") or []
    atom_chains = [str(item) for item in confidences.get("atom_chain_ids") or []]
    mean = _mean([float(value) for value in plddt]) if plddt else None
    if mean is not None:
        metrics["plddt"] = mean / 100.0 if mean > 1.5 else mean
    if len(atom_chains) == len(plddt):
        binder_values = [
            float(value) for value, chain in zip(plddt, atom_chains)
            if chain == binder_id
        ]
        mean = _mean(binder_values)
        if mean is not None:
            metrics["binder_plddt"] = mean / 100.0 if mean > 1.5 else mean

    pae = confidences.get("pae") or []
    token_chains = [str(item) for item in confidences.get("token_chain_ids") or []]
    binder_iptm = _binder_iptm(summary, token_chains, binder_id)
    if binder_iptm is not None:
        metrics["i_ptm"] = binder_iptm
    elif "chain_iptm" in summary:
        raise ValueError(
            "AF3 chain_iptm does not match token chain IDs or binder chain"
        )
    elif "global_i_ptm" in metrics:
        # Older AF3 summaries may not contain per-chain scores. Retain the
        # previous scalar behaviour as a compatibility fallback.
        metrics["i_ptm"] = metrics["global_i_ptm"]
    if pae and len(pae) == len(token_chains) and all(len(row) == len(pae) for row in pae):
        pae_array = np.asarray(pae, dtype=float)
        chains = np.asarray(token_chains)
        binder_mask = chains == binder_id
        target_mask = (
            chains != binder_id
            if target_ids is None
            else np.isin(chains, tuple(str(item) for item in target_ids))
        )
        binder = np.flatnonzero(binder_mask)
        target = np.flatnonzero(target_mask)
        values = [float(pae[i][j]) for i in binder for j in target]
        values += [float(pae[i][j]) for i in target for j in binder]
        mean = _mean(values)
        if mean is not None:
            metrics["i_pae"] = mean
            metrics["min_i_pae"] = min(values)
            metrics["ipsae_min"] = compute_ipsae_min(
                pae_array, binder_mask, target_mask
            )
    return metrics


def _sample_paths(output_dir: Path) -> list[tuple[Path, Path, Path]]:
    paths = []
    for summary in sorted(output_dir.rglob("summary_confidences.json")):
        confidences = summary.parent / "confidences.json"
        structure = summary.parent / "model.cif"
        if confidences.is_file() and structure.is_file():
            paths.append((summary, confidences, structure))
    if paths:
        return paths
    for summary in sorted(output_dir.rglob("*_summary_confidences.json")):
        stem = summary.name.removesuffix("_summary_confidences.json")
        confidences = summary.with_name(f"{stem}_confidences.json")
        structure = summary.with_name(f"{stem}_model.cif")
        if confidences.is_file() and structure.is_file():
            paths.append((summary, confidences, structure))
    return paths


def collect_af3_output(
    output_dir: Path,
    binder_id: str,
    requested_seeds: list[int],
    target_ids: Iterable[str] | None = None,
    expected_samples: int = 5,
) -> list[dict[str, Any]]:
    paths = _sample_paths(output_dir)
    if not paths:
        raise FileNotFoundError(f"No complete AF3 output found under {output_dir}")
    samples = []
    for fallback_sample, (summary_path, confidence_path, structure_path) in enumerate(paths):
        match = re.search(r"seed-(-?\d+)_sample-(\d+)", str(summary_path))
        seed = int(match.group(1)) if match else (
            requested_seeds[0] if len(requested_seeds) == 1 else None
        )
        sample = int(match.group(2)) if match else fallback_sample
        samples.append({
            "seed": seed,
            "sample": sample,
            "metrics": compute_af3_metrics(
                load_json(summary_path), load_json(confidence_path), binder_id,
                target_ids,
            ),
            "structure": str(structure_path),
            "summary_confidences": str(summary_path),
            "confidences": str(confidence_path),
        })
    samples = _deduplicate_samples(output_dir, samples)
    expected = {
        (seed, sample)
        for seed in requested_seeds
        for sample in range(expected_samples)
    }
    observed = {(sample.get("seed"), sample.get("sample")) for sample in samples}
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            "AF3 output has incomplete seed/sample identities; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return samples


def _sample_recency(base: Path, sample: dict[str, Any]) -> int:
    """Return the newest artifact timestamp available for one AF3 sample."""
    timestamps = []
    for key in ("summary_confidences", "confidences", "structure"):
        value = sample.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = base / path
        try:
            timestamps.append(path.stat().st_mtime_ns)
        except OSError:
            pass
    return max(timestamps, default=-1)


def _sample_attempt(base: Path, sample: dict[str, Any]) -> Path:
    """Return the AF3 output-attempt directory containing a sample."""
    value = sample.get("summary_confidences")
    if not value:
        return base.resolve()
    path = Path(str(value))
    if not path.is_absolute():
        path = base / path
    parent = path.parent
    if re.fullmatch(r"seed--?\d+_sample-\d+", parent.name):
        parent = parent.parent
    return parent.resolve()


def _deduplicate_samples(
    base: Path, samples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep one coherent attempt, preferring completeness and then recency."""
    attempts: dict[
        Path, dict[tuple[Any, ...], tuple[int, int, dict[str, Any]]]
    ] = {}
    for order, sample in enumerate(samples):
        seed = sample.get("seed")
        sample_index = sample.get("sample")
        if seed is None or sample_index is None:
            identity: tuple[Any, ...] = ("unidentified", order)
        else:
            identity = ("prediction", seed, sample_index)
        candidate = (_sample_recency(base, sample), order, sample)
        selected = attempts.setdefault(_sample_attempt(base, sample), {})
        if identity not in selected or candidate[:2] > selected[identity][:2]:
            selected[identity] = candidate
    if not attempts:
        return []
    selected = max(
        attempts.values(),
        key=lambda attempt: (
            len(attempt),
            max((item[0] for item in attempt.values()), default=-1),
            max((item[1] for item in attempt.values()), default=-1),
        ),
    )
    return [
        candidate[2]
        for candidate in sorted(selected.values(), key=lambda item: item[1])
    ]


def _pinned_env() -> dict[str, str]:
    """Snapshot the environment AF3 subprocesses should run under.

    AF3 on pre-Ampere GPUs needs the ``XLA_FLAGS`` exported by the job script.
    Optional interface scoring imports ColabDesign, whose package initialiser
    *assigns* to ``os.environ["XLA_FLAGS"]`` rather than appending, destroying
    that flag for every later prediction in the same worker. Capturing the
    environment once, before any such import can run, and passing it explicitly
    to each subprocess keeps every prediction seeing what the job script set.
    """
    return os.environ.copy()


def _command(
    config: dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    *,
    run_data_pipeline: bool = False,
    run_inference: bool = True,
) -> list[str]:
    command = [
        config["python"], config["run_alphafold"],
        f"--input_dir={input_dir}", f"--output_dir={output_dir}",
    ]
    if run_data_pipeline:
        command.append(f"--db_dir={config['db_dir']}")
    if run_inference:
        command.append(f"--model_dir={config['model_dir']}")
    command.extend([
        f"--run_data_pipeline={str(run_data_pipeline).lower()}",
        f"--run_inference={str(run_inference).lower()}",
    ])
    flags = (
        config["data_pipeline_flags"]
        if run_data_pipeline else config["extra_flags"]
    )
    for key, value in sorted(flags.items()):
        if key in _RESERVED_FLAGS or value is None:
            continue
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        command.append(f"--{key}={rendered}")
    return command


def _evaluation_root(
    run_dir: Path,
    name: str,
    selection_name: str,
    config: dict[str, Any],
    target_cache_keys: dict[str, str],
) -> Path:
    if _slug(name) != name:
        raise ValueError(f"Unsafe evaluation name: {name!r}")
    layout = layout_for_run(run_dir)
    root = layout.evaluation_dir("af3", name)
    metadata_path = root / "evaluation.json"
    metadata = {
        "evaluator": "af3", "evaluation_name": name,
        "selection_name": selection_name, "config": config,
        "target_cache": {
            "path": config["target_cache_dir"],
            "contexts": target_cache_keys,
        },
    }
    lock_root = layout.locks / ("Evaluations" if layout.legacy else "evaluations")
    lock = lock_root / "af3" / name / "configure.lock"
    with file_lock(lock):
        if metadata_path.exists() and load_json(metadata_path) != metadata:
            raise ValueError(
                f"Evaluation {name!r} uses different settings; "
                "use another --evaluation-name"
            )
        if not metadata_path.exists():
            atomic_write_json(metadata_path, metadata)
    return root


def _job_dir(jobs_root: Path, row: dict[str, Any], context: dict[str, Any]) -> Path:
    return jobs_root / f"t{int(row['design_index']):05d}" / _slug(context["name"])


def _complete(
    result_path: Path,
    run_dir: Path,
    sequence: str,
    requested_seeds: list[int],
    expected_samples: int,
) -> bool:
    if not result_path.is_file():
        return False
    try:
        result = load_json(result_path)
        samples = result.get("samples", [])
        expected = {
            (seed, sample)
            for seed in requested_seeds
            for sample in range(expected_samples)
        }
        observed = {
            (sample.get("seed"), sample.get("sample")) for sample in samples
        }
        return (
            result.get("status") == "complete"
            and result.get("sequence") == sequence
            and observed == expected
            and all((run_dir / sample["structure"]).is_file() for sample in samples)
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _relative(path: str | Path, run_dir: Path) -> str:
    return str(Path(path).resolve().relative_to(run_dir))


def _expected_diffusion_samples(config: dict[str, Any]) -> int:
    value = config["extra_flags"].get("num_diffusion_samples", 5)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("AF3 num_diffusion_samples must be a positive integer")
    return value


def _run_job(
    run_dir: Path,
    job_dir: Path,
    row: dict[str, Any],
    context: dict[str, Any],
    target_chains: list[dict[str, Any]],
    target_cache_key: str,
    sequence: str,
    config: dict[str, Any],
    evaluation_name: str,
    selection_name: str,
    af3_env: dict[str, str],
    expected_samples: int,
) -> None:
    input_dir = job_dir / "input"
    attempts_dir = job_dir / "attempts"
    input_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="af3_", dir=attempts_dir))
    input_data, binder_id = build_af3_input(
        f"t{int(row['design_index']):05d}_{_slug(context['name'])}",
        sequence,
        target_chains,
        config["seeds"],
    )
    target_ids = [str(chain["chain_id"]) for chain in target_chains]
    input_path = input_dir / "fold_input.json"
    atomic_write_json(input_path, input_data)
    command = _command(config, input_dir, output_dir)
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False, env=af3_env
    )
    atomic_write_text(job_dir / "stdout.log", completed.stdout)
    atomic_write_text(job_dir / "stderr.log", completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"AF3 exited with code {completed.returncode}")
    samples = collect_af3_output(
        output_dir, binder_id, config["seeds"], target_ids, expected_samples
    )
    if config["interface_metrics"]:
        for sample in samples:
            sample["metrics"].update(score_structure(
                Path(sample["structure"]), binder_id, target_ids,
                relax=config["interface_relax"],
                workdir=Path(sample["structure"]).parent,
            ))
    for sample in samples:
        for key in ("structure", "summary_confidences", "confidences"):
            sample[key] = _relative(sample[key], run_dir)
    atomic_write_json(job_dir / "result.json", {
        "status": "complete",
        "evaluator": "af3",
        "evaluation_name": evaluation_name,
        "selection_name": selection_name,
        "design_index": int(row["design_index"]),
        "design_id": row["design_id"],
        "selected_iteration": int(row["iteration"]),
        "selected_stage": row["stage"],
        "sequence": sequence,
        "context": {"name": context["name"], "role": context["role"]},
        "target_cache": {
            "key": target_cache_key,
            "path": config["target_cache_dir"],
        },
        "binder_chain_id": binder_id,
        "samples": samples,
        "target_chain_ids": target_ids,
        "input_json": _relative(input_path, run_dir),
        "stdout": _relative(job_dir / "stdout.log", run_dir),
        "stderr": _relative(job_dir / "stderr.log", run_dir),
        "runner_returncode": completed.returncode,
        "command": command,
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
    config = normalize_af3_config(config_path)
    expected_samples = _expected_diffusion_samples(config)
    af3_env = _pinned_env()
    cache_root = _ensure_cache(config, create=False)
    target_cache_keys: dict[str, str] = {}
    targets: dict[str, list[dict[str, Any]]] = {}
    for context in manifest["contexts"]:
        key, cached = _cached_target(
            cache_root, _target_chains(run_dir, context)
        )
        target_cache_keys[context["name"]] = key
        targets[context["name"]] = cached
    root = _evaluation_root(
        run_dir, evaluation_name, selection_name, config, target_cache_keys
    )
    layout = layout_for_run(run_dir)
    jobs_root = layout.evaluation_jobs("af3", evaluation_name)
    rows = [
        row for row in selection.get("rows", [])
        if row.get("selection_status") == "selected"
        and int(row["design_index"]) % num_shards == shard_index
    ]
    summary = {
        "designs": len(rows), "completed": 0, "skipped": 0,
        "busy": 0, "failed": 0,
    }
    failures = []
    for context in manifest["contexts"]:
        for row in rows:
            sequence = _sequence(row["sequence"])
            job_dir = _job_dir(jobs_root, row, context)
            result_path = job_dir / "result.json"
            if _complete(
                result_path, run_dir, sequence, config["seeds"],
                expected_samples,
            ):
                summary["skipped"] += 1
                continue
            lock_root = layout.locks / (
                "Evaluations" if layout.legacy else "evaluations"
            )
            lock = (
                lock_root / "af3" / evaluation_name
                / f"t{int(row['design_index']):05d}_{_slug(context['name'])}.lock"
            )
            with file_lock(lock, blocking=False) as acquired:
                if not acquired:
                    summary["busy"] += 1
                    continue
                if _complete(
                    result_path, run_dir, sequence, config["seeds"],
                    expected_samples,
                ):
                    summary["skipped"] += 1
                    continue
                try:
                    _run_job(
                        run_dir, job_dir, row, context, targets[context["name"]],
                        target_cache_keys[context["name"]], sequence, config,
                        evaluation_name, selection_name, af3_env,
                        expected_samples,
                    )
                    summary["completed"] += 1
                except Exception as error:
                    atomic_write_json(job_dir / "failure.json", {
                        "status": "failed",
                        "design_index": int(row["design_index"]),
                        "design_id": row["design_id"],
                        "context": context["name"],
                        "error": f"{type(error).__name__}: {error}",
                    })
                    summary["failed"] += 1
                    failures.append(
                        f"t{int(row['design_index']):05d}/{context['name']}"
                    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} AF3 prediction(s) failed: {', '.join(failures[:5])}"
        )
    return summary


def _backfill_sample_ipsae(
    run_dir: Path, result: dict[str, Any], sample: dict[str, Any]
) -> bool:
    """Fill iPSAE from a saved AF3 confidence file without rerunning AF3."""
    metrics = sample.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
        sample["metrics"] = metrics
    value = metrics.get("ipsae_min")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return False

    binder_id = str(result.get("binder_chain_id") or "")
    confidence_value = sample.get("confidences")
    if not binder_id or not confidence_value:
        return False
    confidence_path = Path(str(confidence_value))
    if not confidence_path.is_absolute():
        confidence_path = run_dir / confidence_path
    target_ids = result.get("target_chain_ids")
    if not isinstance(target_ids, list):
        target_ids = None
    try:
        calculated = compute_af3_metrics(
            {}, load_json(confidence_path), binder_id, target_ids
        ).get("ipsae_min")
    except (OSError, TypeError, ValueError):
        return False
    if calculated is None:
        return False
    metrics["ipsae_min"] = calculated
    return True


def _refresh_sample_iptm(
    run_dir: Path, result: dict[str, Any], sample: dict[str, Any]
) -> bool:
    """Refresh binder-specific and global iPTM from saved AF3 confidence data."""
    metrics = sample.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
        sample["metrics"] = metrics

    binder_id = str(result.get("binder_chain_id") or "")
    summary_value = sample.get("summary_confidences")
    confidence_value = sample.get("confidences")
    if not binder_id or not summary_value or not confidence_value:
        return False
    summary_path = Path(str(summary_value))
    confidence_path = Path(str(confidence_value))
    if not summary_path.is_absolute():
        summary_path = run_dir / summary_path
    if not confidence_path.is_absolute():
        confidence_path = run_dir / confidence_path
    target_ids = result.get("target_chain_ids")
    if not isinstance(target_ids, list):
        target_ids = None
    try:
        calculated = compute_af3_metrics(
            load_json(summary_path), load_json(confidence_path), binder_id,
            target_ids,
        )
    except (OSError, TypeError, ValueError):
        return False

    updated = False
    for key in ("i_ptm", "global_i_ptm"):
        value = calculated.get(key)
        if value is None:
            continue
        current = metrics.get(key)
        if (
            not isinstance(current, (int, float))
            or not math.isfinite(float(current))
            or not math.isclose(float(current), value, rel_tol=0.0, abs_tol=1e-12)
        ):
            metrics[key] = value
            updated = True
    return updated


def collect_evaluation(run_dir: Path, evaluation_name: str) -> dict[str, int]:
    run_dir, _ = load_run(run_dir)
    if _slug(evaluation_name) != evaluation_name:
        raise ValueError(f"Unsafe evaluation name: {evaluation_name!r}")
    layout = layout_for_run(run_dir)
    root = layout.evaluation_dir("af3", evaluation_name)
    if not (root / "evaluation.json").is_file():
        raise FileNotFoundError(f"Evaluation not found: {root}")
    lock_root = layout.locks / ("Evaluations" if layout.legacy else "evaluations")
    lock = lock_root / "af3" / evaluation_name / "summarize.lock"
    with file_lock(lock, blocking=False) as acquired:
        if not acquired:
            raise RuntimeError(f"Evaluation summary {evaluation_name!r} is busy")
        results = []
        complete_jobs: set[Path] = set()
        jobs_root = layout.evaluation_jobs("af3", evaluation_name)
        for path in jobs_root.glob("t*/**/result.json"):
            result = load_json(path)
            if result.get("status") != "complete":
                continue
            complete_jobs.add(path.parent.resolve())
            context = result.get("context", {})
            saved_samples = result.get("samples", [])
            if not isinstance(saved_samples, list):
                saved_samples = []
            samples = _deduplicate_samples(run_dir, saved_samples)
            updated = samples != saved_samples
            if updated:
                result["samples"] = samples
            for sample in samples:
                updated = _refresh_sample_iptm(run_dir, result, sample) or updated
                updated = _backfill_sample_ipsae(run_dir, result, sample) or updated
                metrics = sample.get("metrics", {})
                results.append({
                    "design_index": result.get("design_index"),
                    "design_id": result.get("design_id"),
                    "selection": result.get("selection_name"),
                    "selected_iteration": result.get("selected_iteration"),
                    "selected_stage": result.get("selected_stage"),
                    "sequence": result.get("sequence"),
                    "context": context.get("name"),
                    "role": context.get("role"),
                    "seed": sample.get("seed"),
                    "sample": sample.get("sample"),
                    "plddt": metrics.get("plddt"),
                    "binder_plddt": metrics.get("binder_plddt"),
                    "ptm": metrics.get("ptm"),
                    "i_ptm": metrics.get("i_ptm"),
                    "global_i_ptm": metrics.get("global_i_ptm"),
                    "i_pae": metrics.get("i_pae"),
                    "min_i_pae": metrics.get("min_i_pae"),
                    "ipsae_min": metrics.get("ipsae_min"),
                    "ranking_score": metrics.get("ranking_score"),
                    **{key: metrics.get(key) for key in INTERFACE_FIELDS},
                    **{key: metrics.get(key) for key in INTERFACE_EXTRA_FIELDS},
                    "structure": sample.get("structure"),
                })
            if updated:
                atomic_write_json(path, result)
        failures = []
        for path in jobs_root.glob("t*/**/failure.json"):
            if path.parent.resolve() in complete_jobs:
                continue
            failure = load_json(path)
            failures.append({
                "design_index": failure.get("design_index"),
                "design_id": failure.get("design_id"),
                "context": failure.get("context"),
                "error": failure.get("error"),
            })
        results.sort(key=lambda row: (
            int(row["design_index"]), str(row["context"]),
            int(row["seed"]), int(row["sample"]),
        ))
        failures.sort(key=lambda row: (int(row["design_index"]), str(row["context"])))
        fields = [
            "design_index", "design_id", "selection", "selected_iteration",
            "selected_stage", "sequence", "context", "role", "seed", "sample",
            "plddt", "binder_plddt", "ptm", "i_ptm", "global_i_ptm",
            "i_pae", "min_i_pae", "ipsae_min", "ranking_score",
            *INTERFACE_FIELDS, *INTERFACE_EXTRA_FIELDS,
            "structure",
        ]
        atomic_write_text(
            layout.evaluation_metrics("af3", evaluation_name),
            _csv_text(results, fields),
        )
        atomic_write_text(
            layout.evaluation_failures("af3", evaluation_name),
            _csv_text(
                failures, ["design_index", "design_id", "context", "error"]
            ),
        )
        return {"completed": len(results), "failed": len(failures)}
