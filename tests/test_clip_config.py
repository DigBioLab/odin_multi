from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from unittest import mock

import jax.numpy as jnp

from functions.colabdesign_utils import (
    _configure_offtarget_core_weights,
    _finalize_offtarget_optional_weights,
    add_clipped_losses,
    add_contact_max_loss,
)
from functions.generic_utils import (
    GENERAL_KEYS,
    LOSS_KEYS,
    _check_clip_config,
    _validate_advanced_keys,
)


class _Model:
    def __init__(self) -> None:
        self._callbacks = {"model": {"loss": []}}
        self.opt = {"weights": {}}


def _valid_loss() -> dict:
    return {
        "role": "offtarget",
        "weights_pae_inter": -0.1,
        "weights_con_inter": -1.0,
        "weights_iptm": -0.05,
        "weights_ptm_energy_craft": -0.1,
        "weights_contact_max": 0.05,
        "use_ptm_energy_craft_loss": True,
        "clip": {
            "i_pae": {"below": 0.35},
            "i_con": {"below": 3.5},
            "i_ptm": {"above": 0.3},
            "ptm_energy": {"below": 0.3},
            "contact_max": {"above": 0.3},
        },
    }


class ClipConfigTests(unittest.TestCase):
    def test_all_supported_terms_validate(self) -> None:
        _check_clip_config(_valid_loss(), "loss.json")

    def test_unknown_term_lists_valid_weight(self) -> None:
        loss = _valid_loss()
        loss["clip"] = {"unknown": {"below": 1.0}}
        with self.assertRaisesRegex(
            ValueError, r"unknown clipped loss.*i_pae \(weights_pae_inter\)"
        ):
            _check_clip_config(loss, "loss.json")

    def test_each_term_requires_its_fixed_direction(self) -> None:
        for term, wrong in (
            ("i_pae", {"above": 0.35}),
            ("i_con", {"above": 3.5}),
            ("i_ptm", {"below": 0.3}),
            ("ptm_energy", {"above": 0.3}),
            ("contact_max", {"below": 0.3}),
        ):
            with self.subTest(term=term):
                loss = _valid_loss()
                loss["clip"] = {term: wrong}
                with self.assertRaisesRegex(ValueError, rf"clip\.{term} must"):
                    _check_clip_config(loss, "loss.json")

    def test_multiple_directions_are_rejected(self) -> None:
        loss = _valid_loss()
        loss["clip"] = {"i_pae": {"below": 0.35, "above": 0.1}}
        with self.assertRaisesRegex(ValueError, r"clip\.i_pae must contain exactly"):
            _check_clip_config(loss, "loss.json")

    def test_threshold_must_be_finite_number(self) -> None:
        for threshold in (True, "0.3", math.inf, math.nan):
            with self.subTest(threshold=threshold):
                loss = _valid_loss()
                loss["clip"] = {"i_pae": {"below": threshold}}
                with self.assertRaisesRegex(ValueError, r"must be a finite number"):
                    _check_clip_config(loss, "loss.json")

    def test_clipped_term_requires_nonzero_weight(self) -> None:
        loss = _valid_loss()
        loss["weights_pae_inter"] = 0.0
        loss["clip"] = {"i_pae": {"below": 0.35}}
        with self.assertRaisesRegex(ValueError, r"requires nonzero weights_pae_inter"):
            _check_clip_config(loss, "loss.json")

    def test_clip_is_offtarget_only(self) -> None:
        loss = _valid_loss()
        loss["role"] = "target"
        with self.assertRaisesRegex(ValueError, r"only valid for role 'offtarget'"):
            _check_clip_config(loss, "loss.json")

    def test_ptm_energy_keeps_existing_enable_requirement(self) -> None:
        loss = _valid_loss()
        loss["use_ptm_energy_craft_loss"] = False
        loss["weights_contact_max"] = 0.0
        loss["clip"] = {"ptm_energy": {"below": 0.3}}
        with self.assertRaisesRegex(
            ValueError, r"requires use_ptm_energy_craft_loss=true"
        ):
            _check_clip_config(loss, "loss.json")

    def test_contact_max_weight_requires_its_clip(self) -> None:
        loss = _valid_loss()
        loss.pop("clip")
        with self.assertRaisesRegex(
            ValueError, r"weights_contact_max requires clip.contact_max.above"
        ):
            _check_clip_config(loss, "loss.json")

    def test_old_keys_have_migration_errors(self) -> None:
        old_keys = {
            "ipae_threshold": "clip.i_pae.below",
            "con_inter_threshold": "clip.i_con.below",
            "iptm_threshold": "clip.i_ptm.above",
            "ptm_energy_threshold": "clip.ptm_energy.below",
            "use_contact_max_loss": "clip.contact_max.above",
            "contact_max": "clip.contact_max.above",
        }
        for key, replacement in old_keys.items():
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, replacement.replace(".", r"\.")):
                    _validate_advanced_keys(
                        {key: None}, LOSS_KEYS, GENERAL_KEYS, "loss.json", "loss"
                    )

    def test_shipped_offtarget_config_uses_new_schema(self) -> None:
        loss = json.loads(
            (Path(__file__).parents[1] / "settings_loss" / "offtarget.json")
            .read_text(encoding="utf-8")
        )
        _check_clip_config(loss, "settings_loss/offtarget.json")
        self.assertEqual(
            loss["clip"],
            {
                "i_pae": {"below": 0.35},
                "i_con": {"below": 3.5},
                "i_ptm": {"above": 0.3},
            },
        )


class ClipRuntimeTests(unittest.TestCase):
    def test_weights_stay_on_original_loss_keys(self) -> None:
        advanced = _valid_loss()
        advanced.update({
            "weights_pae_intra": 0.0,
            "weights_plddt": 0.0,
            "weights_con_intra": 0.0,
            "weights_rg": 0.0,
            "use_rg_loss": False,
            "weights_termini_loss": 0.0,
            "use_termini_distance_loss": False,
            "weights_helicity": 0.0,
        })
        off_opt = {"weights": {}}

        _configure_offtarget_core_weights(off_opt, advanced)
        _finalize_offtarget_optional_weights(off_opt, advanced)

        self.assertEqual(off_opt["weights"]["i_pae"], -0.1)
        self.assertEqual(off_opt["weights"]["i_con"], -1.0)
        self.assertEqual(off_opt["weights"]["i_ptm"], -0.05)
        self.assertEqual(off_opt["weights"]["ptm_energy"], -0.1)
        self.assertEqual(off_opt["weights"]["contact_max"], 0.05)
        self.assertFalse(
            any(key.endswith("_threshold") for key in off_opt["weights"])
        )

    def test_hard_gates_replace_the_original_losses(self) -> None:
        model = _Model()
        add_clipped_losses(model)
        callback = model._callbacks["model"]["loss"][0]
        clip = {
            "i_pae": {"below": 0.35},
            "i_con": {"below": 3.5},
            "i_ptm": {"above": 0.3},
            "ptm_energy": {"below": 0.3},
        }

        active = callback(
            {"offtarget_len": 10},
            {},
            opt={"clip": clip},
            aux={"losses": {
                "i_pae": jnp.array(0.2),
                "i_con": jnp.array(3.0),
                "i_ptm": jnp.array(0.6),
                "ptm_energy": jnp.array(0.2),
            }},
        )
        self.assertEqual(set(active), {"i_pae", "i_con", "i_ptm", "ptm_energy"})
        self.assertAlmostEqual(float(active["i_pae"]), 0.2)
        self.assertAlmostEqual(float(active["i_con"]), 3.0)
        self.assertAlmostEqual(float(active["i_ptm"]), 0.6)
        self.assertAlmostEqual(float(active["ptm_energy"]), 0.2)

        inactive = callback(
            {"offtarget_len": 10},
            {},
            opt={"clip": clip},
            aux={"losses": {
                "i_pae": jnp.array(0.4),
                "i_con": jnp.array(4.0),
                "i_ptm": jnp.array(0.8),
                "ptm_energy": jnp.array(0.4),
            }},
        )
        self.assertTrue(all(float(value) == 0.0 for value in inactive.values()))

    def test_contact_max_keeps_published_hinge(self) -> None:
        model = _Model()
        add_contact_max_loss(model)
        callback = model._callbacks["model"]["loss"][0]
        contacts = jnp.zeros((4, 4), dtype=jnp.float32)
        contacts = contacts.at[2, 0].set(0.5)
        contacts = contacts.at[3, 0].set(0.4)

        with mock.patch(
            "functions.colabdesign_utils.get_contact_map", return_value=contacts
        ) as get_contact_map:
            result = callback(
                {
                    "seq_mask": jnp.ones(4),
                    "offtarget_len": 2,
                    "binder_len_offtarget": 2,
                },
                {},
                opt={
                    "clip": {"contact_max": {"above": 0.3}},
                    "hotspot": jnp.array([0]),
                    "i_con": {"cutoff": 20.0},
                },
            )

        get_contact_map.assert_called_once_with({}, dist=20.0)
        self.assertAlmostEqual(float(result["contact_max"]), 0.3, places=6)
        self.assertEqual(float(result["contact_max_n_viol"]), 2.0)


if __name__ == "__main__":
    unittest.main()
