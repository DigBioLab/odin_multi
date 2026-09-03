"""AF3 subprocesses must run under the environment captured at start-up.

ColabDesign's package initialiser assigns to ``os.environ["XLA_FLAGS"]``
instead of appending to it, which used to strip the pre-Ampere AF3 flag from
every prediction after the first in a worker.
"""

from __future__ import annotations

import os
import subprocess
import unittest
from unittest import mock

from evaluators.af3 import _pinned_env


V100_FLAGS = "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"


class PinnedEnvTest(unittest.TestCase):
    def test_carries_the_current_xla_flags(self):
        with mock.patch.dict(os.environ, {"XLA_FLAGS": V100_FLAGS}):
            self.assertEqual(_pinned_env()["XLA_FLAGS"], V100_FLAGS)

    def test_survives_a_later_clobber(self):
        """A ColabDesign-style overwrite must not reach an already-pinned env."""
        with mock.patch.dict(os.environ, {"XLA_FLAGS": V100_FLAGS}):
            pinned = _pinned_env()
            os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
            self.assertEqual(pinned["XLA_FLAGS"], V100_FLAGS)

    def test_is_a_copy_not_a_live_view(self):
        with mock.patch.dict(os.environ, {"XLA_FLAGS": V100_FLAGS}):
            pinned = _pinned_env()
        self.assertNotIsInstance(pinned, type(os.environ))
        self.assertEqual(pinned["XLA_FLAGS"], V100_FLAGS)


class SubprocessEnvTest(unittest.TestCase):
    """Both AF3 call sites must pass ``env=``, not inherit os.environ."""

    def _run_two_jobs(self):
        calls = []

        def record(command, **kwargs):
            calls.append(kwargs.get("env"))
            # Emulate the clobber that interface scoring used to cause.
            os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.dict(os.environ, {"XLA_FLAGS": V100_FLAGS}):
            af3_env = _pinned_env()
            for _ in range(2):
                record(["af3"], env=af3_env)
        return calls

    def test_both_jobs_see_the_same_flags(self):
        first, second = self._run_two_jobs()
        self.assertIsNotNone(first)
        self.assertEqual(first["XLA_FLAGS"], V100_FLAGS)
        self.assertEqual(second["XLA_FLAGS"], V100_FLAGS)


class CallSiteTest(unittest.TestCase):
    def test_every_af3_subprocess_passes_env(self):
        import inspect

        import evaluators.af3 as af3

        source = inspect.getsource(af3)
        runs = source.count("subprocess.run(")
        self.assertEqual(runs, source.count("env=af3_env"), source)
        self.assertEqual(runs, 2)


if __name__ == "__main__":
    unittest.main()
