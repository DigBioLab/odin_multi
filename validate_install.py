#!/usr/bin/env python3
"""Validate an Odin-Multi installation without downloading data or running inference."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parent
AF2_WEIGHT_FILES = tuple(f"params_model_{index}_ptm.npz" for index in range(1, 6))
REQUIRED_IMPORTS = {
    "NumPy": "numpy",
    "JAX": "jax",
    "Biopython": "Bio",
    "ColabDesign": "colabdesign",
    "Odin-Multi design utilities": "functions.colabdesign_utils",
}
MAXIMUM_JAX_VERSION = (0, 6, 0)


@dataclass(frozen=True)
class Check:
    level: str
    label: str
    detail: str


class Reporter:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, level: str, label: str, detail: str) -> None:
        check = Check(level, label, detail)
        self.checks.append(check)
        print(f"[{level}] {label}: {detail}")

    def passed(self, label: str, detail: str) -> None:
        self.add("PASS", label, detail)

    def warning(self, label: str, detail: str) -> None:
        self.add("WARN", label, detail)

    def failed(self, label: str, detail: str) -> None:
        self.add("FAIL", label, detail)

    @property
    def failures(self) -> int:
        return sum(check.level == "FAIL" for check in self.checks)

    @property
    def warnings(self) -> int:
        return sum(check.level == "WARN" for check in self.checks)


def _within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def check_imports(reporter: Reporter) -> dict[str, ModuleType]:
    imported: dict[str, ModuleType] = {}
    failures: list[str] = []
    for label, module_name in REQUIRED_IMPORTS.items():
        try:
            imported[module_name] = importlib.import_module(module_name)
        except Exception as error:  # imports can fail for binary/runtime reasons
            failures.append(f"{label} ({type(error).__name__}: {error})")
    if failures:
        reporter.failed("Python imports", "; ".join(failures))
    else:
        reporter.passed("Python imports", ", ".join(REQUIRED_IMPORTS))
    return imported


def check_colabdesign(
    reporter: Reporter, repository: Path, imported: dict[str, ModuleType]
) -> None:
    checkout = repository / "ColabDesign"
    module = imported.get("colabdesign")
    module_file = Path(str(getattr(module, "__file__", ""))) if module else None
    if not checkout.is_dir() or not (checkout / ".git").exists():
        reporter.failed(
            "ColabDesign submodule",
            f"not initialized: {checkout}; run git submodule update --init --recursive",
        )
        return
    try:
        expected = _git_output(repository, "rev-parse", "HEAD:ColabDesign")
        head = _git_output(checkout, "rev-parse", "HEAD")
        status = _git_output(
            checkout, "status", "--porcelain", "--untracked-files=all"
        )
    except (OSError, subprocess.CalledProcessError) as error:
        reporter.failed("ColabDesign submodule", str(error))
        return

    reporter.passed("ColabDesign submodule", str(checkout.resolve()))
    if head != expected:
        reporter.failed(
            "ColabDesign revision", f"found {head}; expected Gitlink {expected}"
        )
    else:
        reporter.passed("ColabDesign revision", head)
    if status:
        reporter.failed("ColabDesign working tree", "modified files are present")
    else:
        reporter.passed("ColabDesign working tree", "clean")

    if (
        module_file is None
        or not module_file.is_file()
        or not _within(module_file, checkout)
    ):
        reporter.failed(
            "ColabDesign import",
            f"Python must import the repository-local checkout at {checkout}",
        )
    else:
        reporter.passed("ColabDesign import", str(module_file.resolve()))


def check_af2_weights(reporter: Reporter, repository: Path) -> None:
    params = repository / "params"
    missing = [name for name in AF2_WEIGHT_FILES if not (params / name).is_file()]
    if missing:
        reporter.failed("AlphaFold 2 weights", "missing from params/: " + ", ".join(missing))
    else:
        reporter.passed("AlphaFold 2 weights", f"{len(AF2_WEIGHT_FILES)} pTM models found")


def check_dssp(reporter: Reporter, repository: Path) -> None:
    dssp = repository / "functions" / "dssp"
    if not dssp.is_file():
        reporter.failed("DSSP", f"executable not found: {dssp}")
    elif not os.access(dssp, os.X_OK):
        reporter.failed("DSSP", f"not executable; run chmod +x {dssp}")
    else:
        reporter.passed("DSSP", str(dssp))


def check_pyrosetta(reporter: Reporter, repository: Path) -> None:
    """PyRosetta plus the bundled DAlphaBall binary, used by interface metrics."""
    dalphaball = repository / "functions" / "DAlphaBall.gcc"
    if not dalphaball.is_file():
        reporter.failed("DAlphaBall", f"binary not found: {dalphaball}")
    elif not os.access(dalphaball, os.X_OK):
        reporter.failed("DAlphaBall", f"not executable; run chmod +x {dalphaball}")
    else:
        reporter.passed("DAlphaBall", str(dalphaball))
    try:
        import pyrosetta  # noqa: F401
    except Exception as error:
        reporter.failed(
            "PyRosetta",
            f"import failed ({type(error).__name__}); evaluator "
            f"interface_metrics will not run: {error}",
        )
    else:
        reporter.passed("PyRosetta", "import ok")


def check_jax_gpu(
    reporter: Reporter, imported: dict[str, ModuleType], allow_cpu: bool
) -> None:
    jax = imported.get("jax")
    if jax is None:
        return
    try:
        devices = list(jax.devices())
    except Exception as error:
        reporter.failed("JAX devices", f"{type(error).__name__}: {error}")
        return
    gpu_devices = [device for device in devices if str(getattr(device, "platform", "")).lower() == "gpu"]
    if gpu_devices:
        reporter.passed("JAX GPU", ", ".join(str(device) for device in gpu_devices))
        return
    detail = "no JAX GPU device found; detected " + (", ".join(str(device) for device in devices) or "no devices")
    if allow_cpu:
        reporter.warning("JAX GPU", detail + " (--allow-cpu enabled)")
    else:
        reporter.failed("JAX GPU", detail)


def check_jax_compatibility(
    reporter: Reporter, imported: dict[str, ModuleType]
) -> None:
    jax = imported.get("jax")
    if jax is None:
        return
    version = str(getattr(jax, "__version__", ""))
    fields = version.split(".")
    try:
        parsed = tuple(int(field.split("+", 1)[0]) for field in fields[:3])
    except ValueError:
        reporter.failed("JAX version", f"could not parse installed version {version!r}")
        return
    parsed += (0,) * (3 - len(parsed))
    if parsed >= MAXIMUM_JAX_VERSION:
        reporter.failed(
            "JAX version",
            f"{version} is incompatible with the pinned ColabDesign checkout; "
            "install jax<0.6.0 and a matching jaxlib build",
        )
    else:
        reporter.passed("JAX version", version)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def check_af3(reporter: Reporter, config_path: Path) -> None:
    try:
        from evaluators.af3 import normalize_af3_config

        config = normalize_af3_config(
            config_path, require_db=True, require_model=True
        )
    except Exception as error:
        reporter.failed("AlphaFold 3 configuration", f"{type(error).__name__}: {error}")
        return
    reporter.passed("AlphaFold 3 configuration", str(config_path.resolve()))

    python = Path(config["python"])
    if not os.access(python, os.X_OK):
        reporter.failed("AlphaFold 3 Python", f"not executable: {python}")
    else:
        reporter.passed("AlphaFold 3 Python", str(python))

    cache = Path(config["target_cache_dir"])
    if cache.exists() and not cache.is_dir():
        reporter.failed("AlphaFold 3 cache", f"not a directory: {cache}")
        return
    parent = _nearest_existing_parent(cache)
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        reporter.failed("AlphaFold 3 cache", f"location is not writable: {cache}")
    else:
        reporter.passed("AlphaFold 3 cache", str(cache))


def run_validation(
    *, allow_cpu: bool = False, af3_config: Path | None = None,
    repository: Path | None = None,
) -> int:
    root = (repository or REPOSITORY_ROOT).resolve()
    reporter = Reporter()
    imported = check_imports(reporter)
    check_jax_compatibility(reporter, imported)
    check_colabdesign(reporter, root, imported)
    check_af2_weights(reporter, root)
    check_dssp(reporter, root)
    check_pyrosetta(reporter, root)
    check_jax_gpu(reporter, imported, allow_cpu)
    if af3_config is not None:
        check_af3(reporter, af3_config)
    print(
        f"Validation complete: {reporter.failures} failure(s), "
        f"{reporter.warnings} warning(s)."
    )
    return 1 if reporter.failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an Odin-Multi installation without running inference."
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Warn instead of failing when JAX cannot see a GPU",
    )
    parser.add_argument(
        "--af3-config",
        type=Path,
        help="Also validate an AlphaFold 3 evaluator configuration",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_validation(
        allow_cpu=args.allow_cpu,
        af3_config=args.af3_config,
    )


if __name__ == "__main__":
    sys.exit(main())
