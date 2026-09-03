from __future__ import annotations

import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from validate_install import Reporter, check_colabdesign, check_jax_compatibility


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _commit(repository: Path, message: str) -> None:
    _git(repository, "add", "-A")
    _git(
        repository,
        "-c", "user.name=Odin Test",
        "-c", "user.email=odin@example.invalid",
        "commit", "-m", message,
    )


class ColabDesignValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "odin"
        self.checkout = self.root / "ColabDesign"
        self.root.mkdir()
        _git(self.root, "init")
        (self.root / "README.md").write_text("test\n", encoding="utf-8")
        _commit(self.root, "initial Odin")

        self.checkout.mkdir()
        _git(self.checkout, "init")
        package = self.checkout / "colabdesign"
        package.mkdir()
        self.module_file = package / "__init__.py"
        self.module_file.write_text("\n", encoding="utf-8")
        _commit(self.checkout, "initial ColabDesign")
        revision = _git(self.checkout, "rev-parse", "HEAD")

        _git(
            self.root,
            "update-index", "--add", "--cacheinfo",
            f"160000,{revision},ColabDesign",
        )
        _commit(self.root, "pin ColabDesign")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _check(self, module_file: Path | None = None) -> Reporter:
        module = ModuleType("colabdesign")
        module.__file__ = str(module_file or self.module_file)
        reporter = Reporter()
        with contextlib.redirect_stdout(io.StringIO()):
            check_colabdesign(reporter, self.root, {"colabdesign": module})
        return reporter

    def test_accepts_clean_checkout_at_gitlink_revision(self) -> None:
        reporter = self._check()
        self.assertEqual(reporter.failures, 0)
        self.assertEqual(
            [check.label for check in reporter.checks],
            [
                "ColabDesign submodule",
                "ColabDesign revision",
                "ColabDesign working tree",
                "ColabDesign import",
            ],
        )

    def test_rejects_dirty_checkout(self) -> None:
        (self.checkout / "dirty.txt").write_text("dirty\n", encoding="utf-8")
        reporter = self._check()
        self.assertTrue(any(
            check.level == "FAIL" and check.label == "ColabDesign working tree"
            for check in reporter.checks
        ))

    def test_rejects_revision_different_from_gitlink(self) -> None:
        self.module_file.write_text("changed\n", encoding="utf-8")
        _commit(self.checkout, "different revision")
        reporter = self._check()
        self.assertTrue(any(
            check.level == "FAIL" and check.label == "ColabDesign revision"
            for check in reporter.checks
        ))

    def test_rejects_import_from_another_installation(self) -> None:
        outside = Path(self.temporary.name) / "other_colabdesign.py"
        outside.write_text("\n", encoding="utf-8")
        reporter = self._check(outside)
        self.assertTrue(any(
            check.level == "FAIL" and check.label == "ColabDesign import"
            for check in reporter.checks
        ))

    def test_reports_uninitialized_submodule(self) -> None:
        self.checkout.rename(self.root / "uninitialized")
        self.checkout.mkdir()
        reporter = self._check()
        self.assertEqual(reporter.failures, 1)
        self.assertEqual(reporter.checks[0].label, "ColabDesign submodule")


class JaxCompatibilityValidationTest(unittest.TestCase):
    @staticmethod
    def _check(version: str) -> Reporter:
        module = ModuleType("jax")
        module.__version__ = version
        reporter = Reporter()
        with contextlib.redirect_stdout(io.StringIO()):
            check_jax_compatibility(reporter, {"jax": module})
        return reporter

    def test_accepts_last_pre_removal_release(self) -> None:
        reporter = self._check("0.5.3")
        self.assertEqual(reporter.failures, 0)
        self.assertEqual(reporter.checks[0].label, "JAX version")

    def test_rejects_release_that_removed_tree_map(self) -> None:
        reporter = self._check("0.6.0")
        self.assertEqual(reporter.failures, 1)
        self.assertIn("jax<0.6.0", reporter.checks[0].detail)


if __name__ == "__main__":
    unittest.main()
