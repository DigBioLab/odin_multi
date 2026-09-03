####################################
############## ColabDesign functions
####################################
### Import dependencies
import os, shutil, math, pickle
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Sequence

from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss, get_contact_map

try:
    from colabdesign.mpnn import mk_mpnn_model
except Exception:
    mk_mpnn_model = None

try:
    from colabdesign.mpnn.ligand import mk_ligand_mpnn_model
except Exception:
    mk_ligand_mpnn_model = None

from .proteinmpnn_backprop import ProteinMPNNARConfig, _mpnn_score_call, compute_ar_cce_loss
from .offtarget_thresholds import hard_gate_above, hard_gate_below, contact_rowmax_hinge

from .biopython_utils import hotspot_residues, calculate_clash_score, calc_ss_percentage, calculate_percentages
from .generic_utils import update_failures
from Bio.PDB import PDBParser, PPBuilder
from Bio.Seq import Seq

DEFAULT_LIGAND_MPNN_SETTINGS = {
    "model_name": "v_32_010",
    "cutoff": 8.0,
}

EARLY_STOPPING_PLDDT_THRESHOLD = 0.65


def _strip_prev_state_for_pickle(value):
    """Return a container copy without previous-recycle state entries."""
    if isinstance(value, dict):
        return {
            key: _strip_prev_state_for_pickle(item)
            for key, item in value.items()
            if not (isinstance(key, str) and "prev" in key.lower())
        }
    if isinstance(value, list):
        return [_strip_prev_state_for_pickle(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_prev_state_for_pickle(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            return np.asarray(value)
        except (TypeError, ValueError):
            pass
    return value


def _atomic_pickle_dump(path, value):
    """Write a pickle atomically so interrupted jobs never look complete."""
    temporary_path = f"{path}.tmp-{os.getpid()}"
    try:
        with open(temporary_path, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _get_discrete_stage_model_kwargs(main_advanced, design_models, stage_name):
    """Return model-selection kwargs for late discrete stages."""
    if main_advanced.get("ensemble_discrete_stages", False):
        print(
            f"{stage_name}: running ensemble mode over "
            f"{len(design_models)} design models"
        )
        return {
            "models": design_models,
            "num_models": len(design_models),
            "sample_models": False,
        }

    return {
        "models": design_models,
        "num_models": 1,
        "sample_models": main_advanced["sample_models"],
    }


def _has_clip(config, term):
    return term in config.get("clip", {})


def _has_nonzero_weight(config, key):
    try:
        return float(config.get(key, 0.0)) != 0.0
    except (TypeError, ValueError):
        return False


def _needs_offtarget_i_ptm_callback(config):
    return (
        bool(config.get("use_i_ptm_loss", False))
        or _has_nonzero_weight(config, "weights_iptm")
        or _has_clip(config, "i_ptm")
    )


def _needs_offtarget_ptm_energy_craft_callback(config):
    return (
        bool(config.get("use_ptm_energy_craft_loss", False))
        or _has_nonzero_weight(config, "weights_ptm_energy_craft")
        or _has_clip(config, "ptm_energy")
    )


def _needs_offtarget_clip_callback(config):
    return any(
        _has_clip(config, term)
        for term in ("i_ptm", "i_pae", "ptm_energy", "i_con")
    )




def _configure_offtarget_core_weights(off_opt, off_advanced):
    """Wire always-available off-target metrics and clipping config."""
    weights = off_opt["weights"]

    weights.update({
        "pae": off_advanced["weights_pae_intra"],
        "plddt": off_advanced["weights_plddt"],
        "i_pae": off_advanced["weights_pae_inter"],
        "con": off_advanced["weights_con_intra"],
        "i_con": off_advanced["weights_con_inter"],
        "contact_max": 0.0,
        "contact_max_n_viol": 0.0,
    })

    clip = off_advanced.get("clip", {})
    if clip:
        off_opt["clip"] = clip
    else:
        off_opt.pop("clip", None)


def _finalize_offtarget_optional_weights(off_opt, off_advanced):
    """
    Restore per-context optional weights after callbacks are registered.

    Callback helpers overwrite shared off-target weights, so this pass restores
    each off-target's configured values.
    """
    weights = off_opt["weights"]
    clip = off_advanced.get("clip", {})

    if "contact_max" in clip:
        weights["contact_max"] = off_advanced.get("weights_contact_max", 0.0)
    else:
        weights["contact_max"] = 0.0
    weights["contact_max_n_viol"] = 0.0

    weights["i_ptm"] = off_advanced.get("weights_iptm", 0.0)

    # rg, termini and helix share a single callback across all off-targets, so
    # each off-target's own weight has to be restored here. The enable flag is
    # honoured per context: a target that did not ask for the loss gets 0.0
    # rather than inheriting it from whichever target switched the callback on.
    weights["rg"] = (
        off_advanced.get("weights_rg", 0.0)
        if off_advanced.get("use_rg_loss", False) else 0.0
    )
    weights["NC"] = (
        off_advanced.get("weights_termini_loss", 0.0)
        if off_advanced.get("use_termini_distance_loss", False) else 0.0
    )
    # helix has no enable flag; it is always registered
    weights["helix"] = off_advanced.get("weights_helicity", 0.0) or 0.0

    weights["ptm_energy"] = off_advanced.get("weights_ptm_energy_craft", 0.0)


def _run_pssm_semigreedy_stage(
    af_model,
    *,
    soft_iters,
    hard_iters,
    tries,
    design_models,
    main_advanced,
    stage_name,
    save_best=True,
):
    """Keep the preparatory logits stage sampled and ensemble the discrete stage."""
    if (
        main_advanced.get("ensemble_discrete_stages", False)
        and soft_iters > 0
        and hard_iters > 0
    ):
        af_model.design_3stage(
            soft_iters=soft_iters,
            temp_iters=0,
            hard_iters=0,
            models=design_models,
            num_models=1,
            sample_models=main_advanced["sample_models"],
            save_best=False,
        )
        af_model._tmp["seq_logits"] = af_model.aux["seq"]["logits"]
        af_model.clear_best()
        af_model.design_pssm_semigreedy(
            soft_iters=0,
            hard_iters=hard_iters,
            tries=tries,
            seq_logits=af_model._tmp["seq_logits"],
            ramp_models=False,
            save_best=save_best,
            **_get_discrete_stage_model_kwargs(main_advanced, design_models, stage_name),
        )
        return

    af_model.design_pssm_semigreedy(
        soft_iters=soft_iters,
        hard_iters=hard_iters,
        tries=tries,
        ramp_models=False,
        save_best=save_best,
        **_get_discrete_stage_model_kwargs(main_advanced, design_models, stage_name),
    )

####################################
# The updated binder_hallucination function
####################################
def binder_hallucination(
    design_name,
    main_settings,
    main_advanced,
    aux_settings_list,
    aux_advanced_list,
    length,
    seed,
    helicity_value,
    design_paths,
    failure_csv,
    design_models
):
    """
    Main binder design function, supporting multiple off-targets,
    each with its own advanced settings.

    Args:
      design_name: A string identifier for this design.
      main_settings: The "main" (primary) target settings dict.
      main_advanced: The advanced settings dict for the main target.
      aux_settings_list: A list of "auxiliary" (off-target or additional target) setting dicts.
      aux_advanced_list: A list of advanced settings for each auxiliary (same length as aux_settings_list).
      length: Binder design length (int).
      seed: Random seed used in input initialization.
      helicity_value: Helicity loss weight or offset (float).
      design_paths: Dictionary of directory paths where outputs are stored.
      failure_csv: Path to a CSV file for logging design failures.
      design_models: A list of AF2 model indices to use.

    Returns:
      The designed model instance (af_model) after input preparation and loss setup.
    """
    if not main_settings:
        raise ValueError("No main settings provided to binder_hallucination.")

    clear_mem()

    # The "main target"
    pdb_filename = main_settings.get("starting_pdb")
    chain_id = main_settings.get("chains")
    hotspot = main_settings.get("target_hotspot_residues", None) or None

    # Instantiate the design model
    af_model = mk_afdesign_model(
        protocol='binder',
        debug=False,
        data_dir=main_advanced['af_params_dir'],
        use_multimer=main_advanced['use_multimer_design'],
        num_recycles=main_advanced['num_recycles_design'],
        best_metric='loss'
    )

    if not hasattr(af_model, "enable_mpnn_backprop"):
        af_model.enable_mpnn_backprop = enable_mpnn_backprop.__get__(af_model, type(af_model))
    # Prepare main target inputs
    main_name = main_settings['binder_name']

    af_model.prep_inputs(
        pdb_filename=pdb_filename,
        chain=chain_id,
        binder_len=length,
        hotspot=hotspot,
        seed=seed,
        rm_aa=main_advanced['omit_AAs'],
        rm_target_seq=main_advanced['rm_template_seq_design'],
        rm_target_sc=main_advanced['rm_template_sc_design'],
        role=main_advanced['role'],
        name=main_name
    )

    # Prepare off-target inputs if any
    aux_params_list = []
    used_names = {main_name}
    for i, auxiliary in enumerate(aux_settings_list):
        off_advanced = aux_advanced_list[i] if i < len(aux_advanced_list) else main_advanced
        starting_pdb_offtarget = auxiliary.get("starting_pdb")
        chain_offtarget = auxiliary.get("chains")
        hotspots_offtarget = auxiliary.get("target_hotspot_residues", None) or None
        off_name = auxiliary['binder_name']
        if off_name in used_names:
            raise ValueError(
                f"Off-target binder_name '{off_name}' duplicates an existing target/off-target name. "
                "Provide unique binder_name values to avoid logging and state conflicts."
            )
        used_names.add(off_name)
        aux_params = {
            "pdb_filename": starting_pdb_offtarget,
            "offtarget_chain": chain_offtarget,
            "binder_len": length,
            "hotspot": hotspots_offtarget,
            "seed": seed,
            "rm_aa": off_advanced["omit_AAs"],
            "rm_target_seq": off_advanced["rm_template_seq_design"],
            "rm_target_sc": off_advanced["rm_template_sc_design"],
            "role": off_advanced['role'],
            "name": off_name
        }
        aux_params_list.append(aux_params)

    if aux_params_list:
        af_model.prep_offtarget_inputs(aux_params_list)

    af_model._args["gradient_weight"] = float(main_advanced["gradient_weight"])
    if hasattr(af_model, "_offtargets"):
        for index, offdict in enumerate(af_model._offtargets):
            advanced = (
                aux_advanced_list[index]
                if index < len(aux_advanced_list)
                else main_advanced
            )
            offdict["gradient_weight"] = float(advanced["gradient_weight"])

    # set up main weights
    af_model.opt["weights"].update({
        "pae": main_advanced["weights_pae_intra"],
        "plddt": main_advanced["weights_plddt"],
        "i_pae": main_advanced["weights_pae_inter"],
        "con": main_advanced["weights_con_intra"],
        "i_con": main_advanced["weights_con_inter"],
    })
    af_model.opt["con"].update({
        "num": main_advanced["intra_contact_number"],
        "cutoff": main_advanced["intra_contact_distance"],
        "num_pos": main_advanced.get("intra_contact_num_pos", float("inf")),
        "binary": False,
        "seqsep": 9,
    })
    af_model.opt["i_con"].update({
        "num": main_advanced["inter_contact_number"],
        "cutoff": main_advanced["inter_contact_distance"],
        "num_pos": main_advanced.get("inter_contact_num_pos", float("inf")),
        "binary": False,
    })

    af_model.opt["total_iters"] = (
        main_advanced["soft_iterations"]
        + main_advanced["temporary_iterations"]
        + main_advanced["hard_iterations"]
        + main_advanced.get("greedy_iterations", 0)
    )

    # Gradient combination is fixed inside the custom ColabDesign fork.

    # handle off-target weights
    if hasattr(af_model, "_offtargets"):
        for i, offdict in enumerate(af_model._offtargets):
            off_advanced = aux_advanced_list[i] if i < len(aux_advanced_list) else main_advanced
            _configure_offtarget_core_weights(offdict["opt_offtarget"], off_advanced)
            offdict["opt_offtarget"]["con"].update({
                "num": off_advanced["intra_contact_number"],
                "cutoff": off_advanced["intra_contact_distance"],
                "num_pos": off_advanced.get("intra_contact_num_pos", float("inf")),
                "binary": False,
                "seqsep": 9,
            })
            offdict["opt_offtarget"]["i_con"].update({
                "num": off_advanced["inter_contact_number"],
                "cutoff": off_advanced["inter_contact_distance"],
                "num_pos": off_advanced.get("inter_contact_num_pos", float("inf")),
                "binary": False,
            })
            # store per-offtarget base weights for later scaling
            try:
                from copy import deepcopy as _dc
                offdict["_base_weights"] = _dc(offdict["opt_offtarget"]["weights"])
            except Exception:
                pass
    elif hasattr(af_model, '_offtarget_opt'):
        # fallback single off-target
        off_advanced = aux_advanced_list[0] if aux_advanced_list else main_advanced
        _configure_offtarget_core_weights(af_model._offtarget_opt, off_advanced)
        af_model._offtarget_opt["con"].update({
            "num": off_advanced["intra_contact_number"],
            "cutoff": off_advanced["intra_contact_distance"],
            "num_pos": off_advanced.get("intra_contact_num_pos", float("inf")),
            "binary": False,
            "seqsep": 9,
        })
        af_model._offtarget_opt["i_con"].update({
            "num": off_advanced["inter_contact_number"],
            "cutoff": off_advanced["inter_contact_distance"],
            "num_pos": off_advanced.get("inter_contact_num_pos", float("inf")),
            "binary": False,
        })

    # add optional losses
    needs_offtarget_rg = any(
        aux_advanced.get("use_rg_loss", False) for aux_advanced in aux_advanced_list
    )
    if main_advanced.get("use_rg_loss", False) or needs_offtarget_rg:
        add_rg_loss(
            af_model,
            weight=(main_advanced["weights_rg"] if main_advanced.get("use_rg_loss", False) else 0.0),
            offtarget_weight=0.0,
        )

    needs_offtarget_i_ptm = any(
        _needs_offtarget_i_ptm_callback(aux_advanced)
        for aux_advanced in aux_advanced_list
    )
    if main_advanced["use_i_ptm_loss"] or needs_offtarget_i_ptm:
        add_i_ptm_loss(
            af_model,
            weight=(main_advanced["weights_iptm"] if main_advanced["use_i_ptm_loss"] else 0.0),
            offtarget_weight=0.0,
        )

    needs_offtarget_ptm_energy_craft = any(
        _needs_offtarget_ptm_energy_craft_callback(aux_advanced)
        for aux_advanced in aux_advanced_list
    )
    if main_advanced.get("use_ptm_energy_craft_loss", False) or needs_offtarget_ptm_energy_craft:
        add_ptm_energy_craft_loss(
            af_model,
            weight=(
                main_advanced.get("weights_ptm_energy_craft", 0.05)
                if main_advanced.get("use_ptm_energy_craft_loss", False)
                else 0.0
            ),
            offtarget_weight=0.0,
        )

    if any(_has_clip(aux_advanced, "contact_max") for aux_advanced in aux_advanced_list):
        add_contact_max_loss(af_model, weight=0.0, offtarget_weight=0.0)

    if any(
        _needs_offtarget_clip_callback(aux_advanced)
        for aux_advanced in aux_advanced_list
    ):
        add_clipped_losses(af_model)

    needs_offtarget_termini = any(
        aux_advanced.get("use_termini_distance_loss", False) for aux_advanced in aux_advanced_list
    )
    if main_advanced.get("use_termini_distance_loss", False) or needs_offtarget_termini:
        add_termini_distance_loss(
            af_model,
            weight=(
                main_advanced["weights_termini_loss"]
                if main_advanced.get("use_termini_distance_loss", False)
                else 0.0
            ),
            offtarget_weight=0.0,
        )

    if main_advanced.get("use_ptm_energy_loss", False):
        add_ptm_energy_loss(
            af_model,
            weight=main_advanced["weights_ptm_energy"],
            offtarget_weight=(aux_advanced_list[0]["weights_ptm_energy"] if aux_advanced_list else main_advanced["weights_ptm_energy"]),
            threshold=main_advanced.get("ptm_energy_threshold", 0.8),
            mode=main_advanced.get("ptm_energy_mode", "penalty")
        )

    add_helix_loss(
        af_model,
        weight=helicity_value,
        offtarget_weight=0.0,
    )

    # Restore per-offtarget optional routing after shared callback registration.
    if hasattr(af_model, "_offtargets"):
        from copy import deepcopy as _dc

        for i, offdict in enumerate(af_model._offtargets):
            off_advanced = aux_advanced_list[i] if i < len(aux_advanced_list) else main_advanced
            _finalize_offtarget_optional_weights(offdict["opt_offtarget"], off_advanced)
            offdict["_base_weights"] = _dc(offdict["opt_offtarget"]["weights"])
    elif hasattr(af_model, "_offtarget_opt"):
        from copy import deepcopy as _dc

        off_advanced = aux_advanced_list[0] if aux_advanced_list else main_advanced
        _finalize_offtarget_optional_weights(af_model._offtarget_opt, off_advanced)
        af_model._offtarget_base_weights = _dc(af_model._offtarget_opt["weights"])

    mpnn_weight_keys = (
        "weights_mpnn_structure_confidence",
        "weights_mpnn_sequence_kl",
        "weights_mpnn_autoregressive_ce",
    )
    num_offtargets = len(getattr(af_model, "_offtargets", []))
    if num_offtargets == 0 and hasattr(af_model, "_offtarget_opt"):
        num_offtargets = 1
    offtarget_advanced = [
        aux_advanced_list[index] if index < len(aux_advanced_list) else {}
        for index in range(num_offtargets)
    ]
    context_advanced = [main_advanced, *offtarget_advanced]
    mpnn_active = any(
        float(advanced.get(key, 0.0)) != 0.0
        for advanced in context_advanced
        for key in mpnn_weight_keys
    )

    if mpnn_active:
        use_soluble_flag = (
            str(main_advanced.get("mpnn_weights", "original")).strip().lower()
            == "soluble"
        )
        use_ligand_backend = bool(main_advanced.get("mpnn_use_ligand_context", False))
        af_model._mpnn_backend_kind = "ligand" if use_ligand_backend else "protein"
        mpnn_model_default = (
            DEFAULT_LIGAND_MPNN_SETTINGS["model_name"]
            if use_ligand_backend else "v_48_020"
        )
        mpnn_model_name = str(main_advanced.get("model_path", mpnn_model_default))
        af_model._mpnn_ligand_cfg = (
            {
                "cutoff": float(main_advanced.get(
                    "mpnn_ligand_mask_cutoff",
                    main_advanced.get(
                        "ligand_mask_cutoff",
                        DEFAULT_LIGAND_MPNN_SETTINGS["cutoff"],
                    ),
                )),
            }
            if use_ligand_backend else None
        )

        if use_ligand_backend:
            print(
                f"[MPNN-BP] Using LigandMPNN backend model={mpnn_model_name} "
                "with fixed-target side-chain context enabled."
            )

        af_model.enable_mpnn_backprop(
            use_solubleMPNN=use_soluble_flag,
            mpnn_model_name=mpnn_model_name,
            mpnn_struct_weight=float(main_advanced["weights_mpnn_structure_confidence"]),
            mpnn_seq_weight=float(main_advanced["weights_mpnn_sequence_kl"]),
            mpnn_ar_cce_weight=float(main_advanced["weights_mpnn_autoregressive_ce"]),
            mpnn_struct_offtarget_weights=[
                float(advanced["weights_mpnn_structure_confidence"])
                for advanced in offtarget_advanced
            ],
            mpnn_seq_offtarget_weights=[
                float(advanced["weights_mpnn_sequence_kl"])
                for advanced in offtarget_advanced
            ],
            mpnn_ar_cce_offtarget_weights=[
                float(advanced["weights_mpnn_autoregressive_ce"])
                for advanced in offtarget_advanced
            ],
            mpnn_interface_only=bool(main_advanced["mpnn_interface_only"]),
            mpnn_offtarget_interface_only=[
                bool(advanced["mpnn_interface_only"])
                for advanced in offtarget_advanced
            ],
            mpnn_num_samples=int(main_advanced["mpnn_backprop_num_samples"]),
            mpnn_offtarget_num_samples=[
                int(advanced["mpnn_backprop_num_samples"])
                for advanced in offtarget_advanced
            ],
        )

    # Log loss weights at job start
    print("\n[CONFIG] === Loss Weights ===")
    print(
        f"[CONFIG] Main target ({main_settings.get('binder_name', 'main')}): "
        f"gradient_weight={main_advanced['gradient_weight']} "
        f"losses={af_model.opt.get('weights', {})}"
    )
    if hasattr(af_model, "_offtargets") and af_model._offtargets:
        for i, offdict in enumerate(af_model._offtargets):
            off_name = aux_settings_list[i].get('binder_name', f'off_{i}') if i < len(aux_settings_list) else f'off_{i}'
            off_weights = offdict.get("opt_offtarget", {}).get("weights", {})
            advanced = aux_advanced_list[i] if i < len(aux_advanced_list) else main_advanced
            print(
                f"[CONFIG] Offtarget {i} ({off_name}): "
                f"gradient_weight={advanced['gradient_weight']} losses={off_weights}"
            )
    print("[CONFIG] === End Loss Weights ===\n")

    # whether to skip remaining stages on low intermediate pLDDT
    use_early_stopping = main_advanced.get("use_early_stopping", False)

    # compute number of random mutations for greedy approach
    greedy_tries = math.ceil(length * (main_advanced["greedy_percentage"] / 100))

    # pick the design algorithm
    if main_advanced["design_algorithm"] == '2stage':
        _run_pssm_semigreedy_stage(
            af_model,
            soft_iters=main_advanced["soft_iterations"],
            hard_iters=main_advanced["greedy_iterations"],
            tries=greedy_tries,
            design_models=design_models,
            main_advanced=main_advanced,
            stage_name="Stage 2: PSSM Semigreedy Optimisation",
            save_best=True,
        )

    elif main_advanced["design_algorithm"] == '3stage':
        af_model.design_3stage(
            soft_iters=main_advanced["soft_iterations"],
            temp_iters=main_advanced["temporary_iterations"],
            hard_iters=main_advanced["hard_iterations"],
            num_models=1,
            models=design_models,
            sample_models=main_advanced["sample_models"],
            save_best=True
        )

    elif main_advanced["design_algorithm"] == 'greedy':
        af_model.design_semigreedy(
            main_advanced["greedy_iterations"],
            tries=greedy_tries,
            **_get_discrete_stage_model_kwargs(
                main_advanced, design_models, "Greedy Semigreedy Optimisation"
            ),
            save_best=True
        )

    elif main_advanced["design_algorithm"] == 'mcmc':
        half_life = round(main_advanced["greedy_iterations"] / 5, 0)
        t_mcmc = 0.01
        af_model._design_mcmc(
            main_advanced["greedy_iterations"],
            half_life=half_life,
            T_init=t_mcmc,
            mutation_rate=greedy_tries,
            num_models=1,
            models=design_models,
            sample_models=main_advanced["sample_models"],
            save_best=True
        )

    elif main_advanced["design_algorithm"] == '4stage':
        # see your existing multi-stage code
        print("Stage 1: Test Logits")
        af_model.design_logits(
            iters=50,
            e_soft=0.9,
            models=design_models,
            num_models=1,
            sample_models=main_advanced["sample_models"],
            save_best=True
        )


        if _early_stopping_checkpoint(
            af_model, length, "initial logits", use_early_stopping
        ):
            if main_advanced["optimise_beta"]:
                model_pdb_path = os.path.join(design_paths["Trajectory"], design_name+".pdb")
                af_model.save_pdb(model_pdb_path)
                _, beta, *_ = calc_ss_percentage(model_pdb_path, main_advanced, 'B')
                os.remove(model_pdb_path)

                if float(beta) > 15:
                    main_advanced["soft_iterations"] += main_advanced["optimise_beta_extra_soft"]
                    main_advanced["temporary_iterations"] += main_advanced["optimise_beta_extra_temp"]
                    af_model.set_opt(num_recycles=main_advanced["optimise_beta_recycles_design"])
                    print("Beta sheeted trajectory detected, optimising settings")

            logits_iter = main_advanced["soft_iterations"] - 50
            if logits_iter > 0:
                print("Stage 1: Additional Logits Optimisation")
                af_model.clear_best()
                af_model.design_logits(
                    iters=logits_iter,
                    e_soft=1,
                    models=design_models,
                    num_models=1,
                    sample_models=main_advanced["sample_models"],
                    ramp_recycles=False,
                    save_best=True
                )
                af_model._tmp["seq_logits"] = af_model.aux["seq"]["logits"]
                logit_plddt = get_best_plddt(af_model, length)
                print("Optimised logit trajectory pLDDT: "+str(logit_plddt))

            if main_advanced["temporary_iterations"] > 0:
                print("Stage 2: Softmax Optimisation")
                af_model.clear_best()
                af_model.design_soft(
                    main_advanced["temporary_iterations"],
                    e_temp=1e-2,
                    models=design_models,
                    num_models=1,
                    sample_models=main_advanced["sample_models"],
                    ramp_recycles=False,
                    save_best=True
                )

            if _early_stopping_checkpoint(
                af_model, length, "softmax", use_early_stopping
            ):
                if main_advanced["hard_iterations"] > 0:
                    af_model.clear_best()
                    print("Stage 3: One-hot Optimisation")
                    af_model.design_hard(
                        main_advanced["hard_iterations"],
                        temp=1e-2,
                        **_get_discrete_stage_model_kwargs(
                            main_advanced, design_models, "Stage 3: One-hot Optimisation"
                        ),
                        dropout=False,
                        ramp_recycles=False,
                        save_best=True
                    )

                if _early_stopping_checkpoint(
                    af_model, length, "one-hot", use_early_stopping
                ):
                    if main_advanced["greedy_iterations"] > 0:
                        print("Stage 4: PSSM Semigreedy Optimisation")
                        af_model.clear_best()
                        _run_pssm_semigreedy_stage(
                            af_model,
                            soft_iters=0,
                            hard_iters=main_advanced["greedy_iterations"],
                            tries=greedy_tries,
                            design_models=design_models,
                            main_advanced=main_advanced,
                            stage_name="Stage 4: PSSM Semigreedy Optimisation",
                            save_best=True,
                        )

                else:
                    update_failures(failure_csv, 'Trajectory_one-hot_pLDDT')
                    print("One-hot checkpoint failed; skipping greedy optimisation")
            else:
                update_failures(failure_csv, 'Trajectory_softmax_pLDDT')
                print("Softmax checkpoint failed; skipping later optimisation stages")
        else:
            update_failures(failure_csv, 'Trajectory_logits_pLDDT')
            print("Initial-logits checkpoint failed; skipping later optimisation stages")

    elif main_advanced["design_algorithm"] == '3stage_ste':
        print("Stage 1: Test Logits")
        af_model.design_logits(
            iters=50,
            e_soft=0.9,
            models=design_models,
            num_models=1,
            sample_models=main_advanced["sample_models"],
            save_best=True
        )
        if _early_stopping_checkpoint(
            af_model, length, "initial logits", use_early_stopping
        ):
            logits_iter = main_advanced["soft_iterations"] - 50
            if logits_iter > 0:
                print("Stage 1: Additional Logits Optimisation")
                af_model.clear_best()
                af_model.design_logits(
                    iters=logits_iter,
                    e_soft=1,
                    models=design_models,
                    num_models=1,
                    sample_models=main_advanced["sample_models"],
                    ramp_recycles=False,
                    save_best=True
                )
                af_model._tmp["seq_logits"] = af_model.aux["seq"]["logits"]
                logit_plddt = get_best_plddt(af_model, length)
                print("Optimised logit trajectory pLDDT: " + str(logit_plddt))

            if main_advanced["temporary_iterations"] > 0:
                print("Stage 2: Softmax Optimisation")
                af_model.clear_best()
                af_model.design_soft(
                    main_advanced["temporary_iterations"],
                    e_temp=0.03,
                    models=design_models,
                    num_models=1,
                    sample_models=main_advanced["sample_models"],
                    ramp_recycles=False,
                    save_best=True
                )

            if _early_stopping_checkpoint(
                af_model, length, "softmax", use_early_stopping
            ):
                if main_advanced["hard_iterations"] > 0:
                    print("Stage 3: Sampled STE One-Hot Optimisation")
                    af_model.clear_best()
                    af_model.design_gumbel(
                        main_advanced["hard_iterations"],
                        temp=0.03,
                        **_get_discrete_stage_model_kwargs(
                            main_advanced,
                            design_models,
                            "Stage 3: Sampled STE One-Hot Optimisation",
                        ),
                        dropout=False,
                        ramp_recycles=False,
                        save_best=True
                    )
                    if not _early_stopping_checkpoint(
                        af_model, length, "sampled STE diagnostic", use_early_stopping
                    ):
                        update_failures(failure_csv, 'Trajectory_ste_pLDDT')
            else:
                update_failures(failure_csv, 'Trajectory_softmax_pLDDT')
                print("Softmax checkpoint failed; skipping sampled STE optimisation")
        else:
            update_failures(failure_csv, 'Trajectory_logits_pLDDT')
            print("Initial-logits checkpoint failed; skipping later optimisation stages")
    else:
        print("ERROR: No valid design model selected")
        exit()
        return

    # final PDB
    model_pdb_path = os.path.join(design_paths["Trajectory"], design_name + ".pdb")
    final_plddt = get_best_plddt(af_model, length)
    af_model.save_pdb(model_pdb_path)
    af_model.aux["log"]["terminate"] = ""

    # check clashes
    ca_clashes = calculate_clash_score(model_pdb_path, 2.5, only_ca=True)
    if ca_clashes > 0:
        af_model.aux["log"]["terminate"] = "Clashing"
        update_failures(failure_csv, 'Trajectory_Clashes')
        print("Severe clashes detected, skipping analysis and MPNN optimisation\n")
    else:
        # check confidence
        if final_plddt < 0.7:
            af_model.aux["log"]["terminate"] = "LowConfidence"
            update_failures(failure_csv, 'Trajectory_final_pLDDT')
            print("Trajectory starting confidence low, skipping analysis and MPNN optimisation\n")
        else:
            # check interface
            binder_contacts = hotspot_residues(model_pdb_path)
            binder_contacts_n = len(binder_contacts.items())
            if binder_contacts_n < 3:
                af_model.aux["log"]["terminate"] = "LowConfidence"
                update_failures(failure_csv, 'Trajectory_Contacts')
                print("Too few contacts at the interface, skipping analysis and MPNN optimisation\n")
            else:
                af_model.aux["log"]["terminate"] = ""
                print("Trajectory successful, final pLDDT: "+str(final_plddt))

    if af_model.aux["log"]["terminate"] != "":
        shutil.move(model_pdb_path, design_paths[f"Trajectory/{af_model.aux['log']['terminate']}"])

    # get the sampled sequence for plotting
    af_model.get_seqs()
    if main_advanced["save_design_trajectory_plots"]:
        plot_trajectory(af_model, design_name, design_paths)

    if main_advanced["save_design_animations"]:
        plots = af_model.animate(dpi=150)
        with open(os.path.join(design_paths["Trajectory/Animation"], design_name+".html"), 'w') as f:
            f.write(plots)
        plt.close('all')

    # Frame history is a required Odin-Multi artifact and is always saved.

    # --- File 1: Aux Data and Metadata (.pickle) ---
    aux_data_to_save = {}

    # 1a. Add final auxiliary data
    aux_data_to_save["final_aux_all"] = _strip_prev_state_for_pickle(af_model.aux.get('all', None))

    # 1b. Add Metadata (reading from actual storage locations)
    # Build task_names and task_roles from where they're actually stored
    task_names = [af_model.name] if hasattr(af_model, 'name') else []
    task_roles = [af_model.role] if hasattr(af_model, 'role') else []

    # Append off-target names and roles
    if hasattr(af_model, "_offtargets") and af_model._offtargets:
        for offdict in af_model._offtargets:
            task_names.append(offdict.get("name", f"offtarget_{len(task_names)}"))
            task_roles.append(offdict.get("role", "unknown"))

    aux_data_to_save["metadata"] = {
        "task_names": task_names,
        "task_roles": task_roles, # Storing the full roles ("target", "offtarget", etc.)
    }

    # 1c. Define filename and save File 1
    aux_pickle_filename = os.path.join(design_paths["Trajectory/Pickle"], design_name + ".pickle")
    try:
        _atomic_pickle_dump(
            aux_pickle_filename,
            _strip_prev_state_for_pickle(aux_data_to_save),
        )
        print(f"Saved aux data and metadata to: {aux_pickle_filename}")
    except Exception as e:
        print(f"Error saving aux/metadata pickle file '{aux_pickle_filename}': {e}")


    # --- File 2: Trajectories Only (_trajectory.pickle) ---

    trajectory_data_to_save = {
        "__schema_version__": 2,
    } # Context names remain direct keys for backward compatibility.

    if task_names:
        main_traj = {
            key: value for key, value in af_model._tmp.get("traj", {}).items()
            if "prev" not in key
        }
        main_traj["role"] = task_roles[0]
        trajectory_data_to_save[task_names[0]] = main_traj
        for index, offdict in enumerate(getattr(af_model, "_offtargets", []), 1):
            if index >= len(task_names):
                break
            off_traj = {
                key: value
                for key, value in offdict.get("_tmp", {}).get("traj", {}).items()
                if "prev" not in key
            }
            off_traj["role"] = task_roles[index]
            trajectory_data_to_save[task_names[index]] = off_traj
    else:
        print("Warning: Task metadata (name/role) not found in af_model. Cannot save named trajectories in the trajectory file.")

    traj_pickle_filename = os.path.join(design_paths["Trajectory/Pickle"], design_name + "_trajectory.pickle")
    try:
        _atomic_pickle_dump(
            traj_pickle_filename,
            _strip_prev_state_for_pickle(trajectory_data_to_save),
        )
        print(f"Saved trajectories to: {traj_pickle_filename}")
    except Exception as e:
        print(f"Error saving trajectory pickle file '{traj_pickle_filename}': {e}")

    return af_model


def _compute_kl_loss(
    seq_logits_slice: Optional[jnp.ndarray], log_q_slice: jnp.ndarray, logits_slice: jnp.ndarray,
    is_contact: jnp.ndarray, p_tau: float, q_tau: float, use_interface_mask: bool
) -> jnp.ndarray:
    """Computes the KL divergence loss between input sequence and MPNN predictions."""
    if seq_logits_slice is None:
        return jnp.array(0.0, dtype=jnp.float32)

    log_p = jax.nn.log_softmax(seq_logits_slice / max(p_tau, 1e-6), axis=-1)
    log_q_teacher = log_q_slice if abs(q_tau - 1.0) < 1e-6 else jax.nn.log_softmax(logits_slice / max(q_tau, 1e-6), axis=-1)
    kl_per_pos = jnp.sum(jnp.exp(log_p) * (log_p - jax.lax.stop_gradient(log_q_teacher)), axis=-1)

    # Behavior decision: Apply loss only to the interface if specified (for repulsion).
    # The 'if' statement is replaced with jnp.where to avoid TracerBoolConversionError.
    interface_loss = jnp.sum(is_contact * kl_per_pos) / (jnp.sum(is_contact) + 1e-8)
    full_loss = jnp.mean(kl_per_pos)
    
    return jnp.where(use_interface_mask, interface_loss, full_loss)


def _get_all_offtarget_opts(model_instance):
    """Yield mutable optimization dictionaries for every off-target."""
    if hasattr(model_instance, "_offtargets"):
        for offdict in model_instance._offtargets:
            yield offdict.setdefault("opt_offtarget", {})
    elif hasattr(model_instance, "_offtarget_opt"):
        yield model_instance._offtarget_opt



def _ensure_mpnn_backend(model_instance, *, use_solubleMPNN: bool, model_name: str):
    backend_kind = getattr(model_instance, "_mpnn_backend_kind", "protein")
    current = getattr(model_instance, "_mpnn_struct", None)
    current_kind = getattr(current, "_bindcraft_backend", None)
    current_model_name = getattr(current, "_bindcraft_model_name", None)
    current_weights_kind = getattr(current, "_bindcraft_weights_kind", None)
    requested_weights_kind = "soluble" if use_solubleMPNN else "original"

    if (
        current is not None
        and current_kind == backend_kind
        and current_model_name == model_name
        and (backend_kind != "protein" or current_weights_kind == requested_weights_kind)
    ):
        return current

    if backend_kind == "ligand":
        if mk_ligand_mpnn_model is None:
            raise RuntimeError("ColabDesign JAX LigandMPNN is unavailable")
        if use_solubleMPNN:
            raise ValueError("LigandMPNN does not support soluble ProteinMPNN weights.")
        lig_cfg = getattr(model_instance, "_mpnn_ligand_cfg", None)
        if lig_cfg is None:
            raise RuntimeError("LigandMPNN backend requested without cached context config.")
        model = mk_ligand_mpnn_model(model_name=model_name)
        model._bindcraft_ligand_cfg = dict(lig_cfg)
    else:
        if mk_mpnn_model is None:
            raise RuntimeError("ColabDesign JAX ProteinMPNN is unavailable")
        model = mk_mpnn_model(
            model_name=model_name,
            weights=requested_weights_kind,
        )
        model._bindcraft_weights_kind = requested_weights_kind

    model._bindcraft_backend = backend_kind
    model._bindcraft_model_name = model_name
    model_instance._mpnn_struct = model
    return model


def _enable_struct_seq_mpnn_backprop(
    self,
    *,
    use_solubleMPNN: bool,
    model_name: str,
    struct_weight: float,
    seq_weight: float,
    struct_offtarget_weights: Sequence[float],
    seq_offtarget_weights: Sequence[float],
):
    """Register the structural NLL and sequence KL MPNN objectives."""
    off_opts = list(_get_all_offtarget_opts(self))
    self.opt["weights"]["mpnn_struct"] = struct_weight
    self.opt["weights"]["mpnn_seq"] = seq_weight
    for index, opt in enumerate(off_opts):
        opt["weights"]["mpnn_struct"] = (
            struct_offtarget_weights[index] if index < len(struct_offtarget_weights) else 0.0
        )
        opt["weights"]["mpnn_seq"] = (
            seq_offtarget_weights[index] if index < len(seq_offtarget_weights) else 0.0
        )

    configured_weights = [
        struct_weight,
        seq_weight,
        *struct_offtarget_weights,
        *seq_offtarget_weights,
    ]
    if not any(weight != 0.0 for weight in configured_weights):
        return

    _ensure_mpnn_backend(self, use_solubleMPNN=use_solubleMPNN, model_name=model_name)
    from colabdesign.af.alphafold.common import residue_constants as _rc

    if not hasattr(self, "mpnn_seq_p_tau"):
        self.mpnn_seq_p_tau = 1.0
    if not hasattr(self, "mpnn_seq_q_tau"):
        self.mpnn_seq_q_tau = 1.0

    def _mpnn_loss_fn(inputs, outputs, aux, key):
        opt = inputs.get("opt", {})
        interface_only = jnp.asarray(opt.get("mpnn_interface_only", False), dtype=bool)

        final_positions = outputs["structure_module"]["final_atom_positions"]
        final_mask = outputs["structure_module"]["final_atom_mask"]
        length = final_positions.shape[0]
        binder_length = (
            int(getattr(self, "_binder_len", length))
            if self.protocol == "binder" else length
        )
        binder_start = length - binder_length if self.protocol == "binder" else 0

        mpnn_inputs = {
            "S": inputs["aatype"],
            "residue_idx": inputs["residue_index"],
            "chain_idx": inputs.get("asym_id", jnp.zeros(length, dtype=jnp.int32)),
        }
        if "offset" in inputs:
            mpnn_inputs["offset"] = inputs["offset"]

        if self.protocol == "binder":
            mpnn_inputs["ar_mask"] = (
                (1 - jnp.eye(length))
                .at[-binder_length:, -binder_length:]
                .set(0)
            )
        else:
            mpnn_inputs["ar_mask"] = jnp.zeros((length, length))

        mpnn_out = _mpnn_score_call(
            self._mpnn_struct,
            atom_positions=final_positions,
            atom_mask=final_mask,
            residue_idx=mpnn_inputs["residue_idx"],
            chain_idx=mpnn_inputs["chain_idx"],
            key=key,
            binder_start=binder_start,
            S=mpnn_inputs["S"],
            ar_mask=mpnn_inputs["ar_mask"],
            **({"offset": mpnn_inputs["offset"]} if "offset" in mpnn_inputs else {}),
        )
        logits_full = mpnn_out["logits"][:, :20]
        log_q_full = jax.nn.log_softmax(logits_full, axis=-1)
        log_q_slice = log_q_full[binder_start:]
        logits_slice = logits_full[binder_start:]
        binder_length = log_q_slice.shape[0]

        seq_logits_slice = inputs.get("seq", {}).get("logits")
        if seq_logits_slice is not None:
            if seq_logits_slice.ndim == 3:
                seq_logits_slice = seq_logits_slice[0]
            if seq_logits_slice.shape[-1] > 20:
                seq_logits_slice = seq_logits_slice[..., :20]
            if seq_logits_slice.shape[0] >= binder_length:
                seq_logits_slice = seq_logits_slice[-binder_length:]
            else:
                seq_logits_slice = None

        receptor_length = length - binder_length
        is_contact = jnp.zeros((binder_length,), dtype=jnp.float32)
        d_min = None
        if binder_length > 0 and receptor_length > 0:
            ca = final_positions[:, _rc.atom_order["CA"], :]
            binder_ca = ca[receptor_length:]
            receptor_ca = ca[:receptor_length]
            distances = jnp.linalg.norm(
                binder_ca[:, None, :] - receptor_ca[None, :, :],
                axis=-1,
            )
            d_min = jnp.min(distances, axis=1)
            is_contact = (
                d_min <= float(getattr(self, "mpnn_mask_cutoff", 8.0))
            ).astype(jnp.float32)

        per_position_nll = -jnp.log(
            jnp.clip(jnp.max(jnp.exp(log_q_slice), axis=-1), 1e-8, 1.0)
        )
        struct_full = per_position_nll.mean()
        tau = float(getattr(self, "mpnn_soft_tau", 1.0))
        interface_weights = (
            jax.nn.sigmoid(
                (float(getattr(self, "mpnn_mask_cutoff", 8.0)) - d_min) / tau
            )
            if d_min is not None
            else jnp.zeros((binder_length,), dtype=jnp.float32)
        )
        struct_interface = (
            jnp.sum(interface_weights * per_position_nll)
            / (jnp.sum(interface_weights) + 1e-8)
        )
        struct_loss = jnp.where(interface_only, struct_interface, struct_full)

        seq_loss = _compute_kl_loss(
            seq_logits_slice,
            log_q_slice,
            logits_slice,
            is_contact,
            float(getattr(self, "mpnn_seq_p_tau", 1.0)),
            float(getattr(self, "mpnn_seq_q_tau", 1.0)),
            use_interface_mask=interface_only,
        )
        return {"mpnn_struct": struct_loss, "mpnn_seq": seq_loss}

    if not getattr(self, "_mpnn_struct_seq_registered", False):
        self._callbacks["model"]["loss"].append(_mpnn_loss_fn)
        self._mpnn_struct_seq_registered = True


def _enable_ar_cce_mpnn_backprop(
    self,
    *,
    use_solubleMPNN: bool,
    model_name: str,
    weight: float,
    offtarget_weights: Sequence[float],
    num_samples: int,
    offtarget_num_samples: Sequence[int],
):
    """Register autoregressive loss branches for each configured sample count."""
    off_opts = list(_get_all_offtarget_opts(self))
    off_counts = list(offtarget_num_samples)
    while len(off_counts) < len(off_opts):
        off_counts.append(num_samples)
    counts = [int(num_samples), *[int(value) for value in off_counts[:len(off_opts)]]]
    unique_counts = tuple(sorted(set(counts)))
    count_indices = {value: index for index, value in enumerate(unique_counts)}

    self.opt["weights"]["mpnn_ar_cce"] = weight
    self.opt["mpnn_num_samples_index"] = count_indices[counts[0]]
    for index, opt in enumerate(off_opts):
        opt["weights"]["mpnn_ar_cce"] = (
            offtarget_weights[index] if index < len(offtarget_weights) else 0.0
        )
        opt["mpnn_num_samples_index"] = count_indices[counts[index + 1]]

    if not any(value != 0.0 for value in [weight, *offtarget_weights]):
        return

    _ensure_mpnn_backend(self, use_solubleMPNN=use_solubleMPNN, model_name=model_name)
    self._mpnn_ar_cce_configs = tuple(
        ProteinMPNNARConfig(
            num_samples=value,
            mask_cutoff=float(getattr(self, "mpnn_mask_cutoff", 8.0)),
            mask_tau=float(getattr(self, "mpnn_soft_tau", 1.0)),
        )
        for value in unique_counts
    )

    def _mpnn_ar_cce_loss_fn(inputs, outputs, aux, key):
        binder_length = int(getattr(
            self,
            "_binder_len",
            outputs["structure_module"]["final_atom_positions"].shape[0],
        ))
        branches = tuple(
            (
                lambda _, config=config: compute_ar_cce_loss(
                    self._mpnn_struct,
                    config,
                    inputs=inputs,
                    outputs=outputs,
                    binder_len=binder_length,
                    key=key,
                )
            )
            for config in self._mpnn_ar_cce_configs
        )
        sample_index = jnp.asarray(
            inputs.get("opt", {}).get("mpnn_num_samples_index", 0),
            dtype=jnp.int32,
        )
        loss = jax.lax.switch(sample_index, branches, operand=None)
        return {"mpnn_ar_cce": loss}

    if not getattr(self, "_mpnn_ar_cce_registered", False):
        self._callbacks["model"]["loss"].append(_mpnn_ar_cce_loss_fn)
        self._mpnn_ar_cce_registered = True


def enable_mpnn_backprop(
    self,
    *,
    use_solubleMPNN: bool = False,
    mpnn_model_name: str = "v_48_020",
    mpnn_struct_weight: float = 0.0,
    mpnn_seq_weight: float = 0.0,
    mpnn_ar_cce_weight: float = 0.0,
    mpnn_struct_offtarget_weights: Optional[Sequence[float]] = None,
    mpnn_seq_offtarget_weights: Optional[Sequence[float]] = None,
    mpnn_ar_cce_offtarget_weights: Optional[Sequence[float]] = None,
    mpnn_interface_only: bool = False,
    mpnn_offtarget_interface_only: Optional[Sequence[bool]] = None,
    mpnn_num_samples: int = 16,
    mpnn_offtarget_num_samples: Optional[Sequence[int]] = None,
):
    """Enable the three publication MPNN objectives."""
    struct_off = list(mpnn_struct_offtarget_weights or [])
    seq_off = list(mpnn_seq_offtarget_weights or [])
    ar_off = list(mpnn_ar_cce_offtarget_weights or [])
    interface_off = list(mpnn_offtarget_interface_only or [])

    self.opt["mpnn_interface_only"] = bool(mpnn_interface_only)
    off_opts = list(_get_all_offtarget_opts(self))
    for index, opt in enumerate(off_opts):
        opt["mpnn_interface_only"] = (
            bool(interface_off[index]) if index < len(interface_off) else False
        )

    _enable_struct_seq_mpnn_backprop(
        self,
        use_solubleMPNN=use_solubleMPNN,
        model_name=mpnn_model_name,
        struct_weight=mpnn_struct_weight,
        seq_weight=mpnn_seq_weight,
        struct_offtarget_weights=struct_off,
        seq_offtarget_weights=seq_off,
    )
    _enable_ar_cce_mpnn_backprop(
        self,
        use_solubleMPNN=use_solubleMPNN,
        model_name=mpnn_model_name,
        weight=mpnn_ar_cce_weight,
        offtarget_weights=ar_off,
        num_samples=mpnn_num_samples,
        offtarget_num_samples=list(mpnn_offtarget_num_samples or []),
    )

    target_scope = "interface" if mpnn_interface_only else "all"
    off_summary = []
    for index, opt in enumerate(off_opts):
        off_summary.append(
            f"offtarget_{index}(struct={opt['weights'].get('mpnn_struct', 0.0)}, "
            f"seq={opt['weights'].get('mpnn_seq', 0.0)}, "
            f"ar_cce={opt['weights'].get('mpnn_ar_cce', 0.0)}, "
            f"interface_only={opt.get('mpnn_interface_only', False)})"
        )
    print(
        f"[MPNN-BP] target(struct={mpnn_struct_weight}, seq={mpnn_seq_weight}, "
        f"ar_cce={mpnn_ar_cce_weight}, scope={target_scope})"
        + (f" | {', '.join(off_summary)}" if off_summary else "")
    )

# Get pLDDT of best model
def get_best_plddt(af_model, length):
    return round(np.mean(af_model._tmp["best"]["aux"]["plddt"][-length:]),2)


def _mean_binder_plddt(plddt, binder_length, context_name):
    """Return the unrounded mean binder pLDDT for one context."""
    if isinstance(binder_length, bool) or not isinstance(binder_length, int) or binder_length < 1:
        raise ValueError("Early stopping requires a positive integer binder length.")

    values = np.asarray(plddt).squeeze()
    if values.ndim != 1 or values.size < binder_length:
        raise ValueError(
            f"Early-stopping pLDDT for context {context_name!r} must be a "
            f"one-dimensional array with at least {binder_length} residues."
        )
    score = float(np.mean(values[-binder_length:]))
    if not math.isfinite(score):
        raise ValueError(
            f"Early-stopping pLDDT for context {context_name!r} is not finite."
        )
    return score


def _get_active_target_plddts(af_model, binder_length):
    """Read synchronized stage-best binder pLDDTs for active target contexts."""
    try:
        best_aux = af_model._tmp["best"]["aux"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("Early stopping requires a saved stage-best model snapshot.") from exc

    if not isinstance(best_aux, dict):
        raise ValueError("Early-stopping stage-best auxiliary data must be a dictionary.")

    scores = {}
    main_name = str(getattr(af_model, "name", "target"))
    main_role = str(getattr(af_model, "role", "target")).strip().lower()
    main_args = getattr(af_model, "_args", {})
    main_weight = float(main_args.get("gradient_weight", 1.0))
    if main_role == "target" and main_weight > 0.0:
        if "plddt" not in best_aux:
            raise ValueError(
                f"Early-stopping pLDDT is missing for active target {main_name!r}."
            )
        scores[main_name] = _mean_binder_plddt(
            best_aux["plddt"], binder_length, main_name
        )

    auxiliary_contexts = list(getattr(af_model, "_offtargets", []) or [])
    auxiliary_outputs = best_aux.get("offtargets", [])
    if not isinstance(auxiliary_outputs, (list, tuple)):
        raise ValueError("Early-stopping auxiliary context outputs must be a list.")
    if len(auxiliary_outputs) != len(auxiliary_contexts):
        raise ValueError(
            "Early-stopping context metadata and synchronized outputs are misaligned: "
            f"{len(auxiliary_contexts)} context(s), {len(auxiliary_outputs)} output(s)."
        )

    for index, context in enumerate(auxiliary_contexts):
        role = str(context.get("role", "")).strip().lower()
        weight = float(context.get("gradient_weight", 1.0))
        if role != "target" or weight <= 0.0:
            continue
        name = str(context.get("name", f"target_{index + 2}"))
        output = auxiliary_outputs[index]
        if not isinstance(output, dict) or output.get("plddt") is None:
            raise ValueError(
                f"Early-stopping pLDDT is missing for active target {name!r}."
            )
        scores[name] = _mean_binder_plddt(
            output["plddt"], binder_length, name
        )

    if not scores:
        raise ValueError("Early stopping found no active target contexts.")
    return scores


def _passes_early_stopping(af_model, binder_length):
    """Return whether every active target clears the fixed pLDDT threshold."""
    scores = _get_active_target_plddts(af_model, binder_length)
    passed = all(
        value > EARLY_STOPPING_PLDDT_THRESHOLD for value in scores.values()
    )
    return passed, scores


def _early_stopping_checkpoint(af_model, binder_length, stage_name, enabled):
    """Evaluate and report one optional multi-target early-stopping checkpoint."""
    if not enabled:
        print(f"[EARLY STOP] {stage_name}: disabled")
        return True

    passed, scores = _passes_early_stopping(af_model, binder_length)
    score_text = ", ".join(f"{name}={value:.3f}" for name, value in scores.items())
    failed = [
        name
        for name, value in scores.items()
        if value <= EARLY_STOPPING_PLDDT_THRESHOLD
    ]
    if passed:
        print(
            f"[EARLY STOP] {stage_name}: {score_text}; all active targets "
            f"> {EARLY_STOPPING_PLDDT_THRESHOLD:.2f}; passed"
        )
    else:
        print(
            f"[EARLY STOP] {stage_name}: {score_text}; failed active target(s): "
            f"{', '.join(failed)} (required > {EARLY_STOPPING_PLDDT_THRESHOLD:.2f})"
        )
    return passed

# define radius of gyration loss for colabdesign
def add_rg_loss(self, weight=0.1, offtarget_weight=0.1):
    """
    Add radius-of-gyration (Rg) loss on the binder portion only.

    For the main 'binder', we store the weight in self.opt['weights']['rg'].
    For off-target(s), we set 'rg' weight in each offdict["opt_offtarget"]["weights"].
    """

    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        # final_atom_positions shape = [L, 37, 3]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        # Only operate on the binder portion
        ca = ca[-self._binder_len:]  # or wherever your "binder" starts
        # compute Rg
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        # approximate theoretical Rg for proteins (Da ~ #res * 110)
        rg_th = 2.38 * ca.shape[0] ** 0.365
        # penalize Rg away from threshold
        rg = jax.nn.elu(rg - rg_th)
        return {"rg": rg}

    # Add the callback
    self._callbacks["model"]["loss"].append(loss_fn)
    # Set the weight for the main binder
    self.opt["weights"]["rg"] = weight

    # For multiple off-targets
    if hasattr(self, "_offtargets"):
        for offdict in self._offtargets:
            offdict["opt_offtarget"]["weights"]["rg"] = offtarget_weight
    elif hasattr(self, '_offtarget_opt'):
        self._offtarget_opt["weights"]["rg"] = offtarget_weight


# define interface pTM loss for colabdesign
def add_i_ptm_loss(self, weight=0.1, offtarget_weight=0.1):
    """
    Add interface pTM loss. 
    For the main binder we set self.opt["weights"]["i_ptm"].
    For off-target(s), we set each offdict["opt_offtarget"]["weights"]["i_ptm"].
    """

    def loss_iptm(inputs, outputs):
        # pTM at interface
        p = 1.0 - get_ptm(inputs, outputs, interface=True)
        i_ptm = mask_loss(p)
        return {"i_ptm": i_ptm}

    self._callbacks["model"]["loss"].append(loss_iptm)
    self.opt["weights"]["i_ptm"] = weight

    if hasattr(self, "_offtargets"):
        for offdict in self._offtargets:
            offdict["opt_offtarget"]["weights"]["i_ptm"] = offtarget_weight
    elif hasattr(self, '_offtarget_opt'):
        self._offtarget_opt["weights"]["i_ptm"] = offtarget_weight


def add_contact_max_loss(self, weight=0.0, offtarget_weight=0.0):
    """
    Add the off-target row-wise hard-max contact hinge.

    V_con = sum_i relu(max_j c_ij - kappa)
    """

    def loss_fn(inputs, outputs, opt=None):
        opt = opt or {}
        cfg = opt.get("clip", {}).get("contact_max")
        if not cfg or "offtarget_len" not in inputs:
            return {}

        mask = inputs["seq_mask"]
        zeros = jnp.zeros_like(mask)

        try:
            otL = int(inputs.get("offtarget_len", getattr(self, "offtarget_len", 0)))
        except Exception:
            otL = getattr(self, "offtarget_len", 0)

        try:
            bL = int(inputs.get("binder_len_offtarget", getattr(self, "binder_len_offtarget", 0)))
        except Exception:
            bL = getattr(self, "binder_len_offtarget", 0)

        if otL <= 0 or bL <= 0:
            zero = jnp.array(0.0, dtype=jnp.float32)
            return {"contact_max": zero, "contact_max_n_viol": zero}

        binder_id = zeros.at[-bL:].set(mask[-bL:])
        hotspot = opt.get("hotspot")
        if hotspot is not None:
            offtarget_id = zeros.at[hotspot].set(mask[hotspot])
        else:
            offtarget_id = zeros.at[:otL].set(mask[:otL])

        contact_prob = get_contact_map(outputs, dist=opt["i_con"]["cutoff"])
        pair_mask = (binder_id[:, None] * offtarget_id[None, :]) > 0
        contact_max, _, n_viol = contact_rowmax_hinge(
            contact_prob,
            pair_mask,
            cfg["above"],
        )
        return {"contact_max": contact_max, "contact_max_n_viol": n_viol}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["contact_max"] = weight
    self.opt["weights"]["contact_max_n_viol"] = 0.0

    if hasattr(self, "_offtargets"):
        for offdict in self._offtargets:
            offdict["opt_offtarget"]["weights"]["contact_max"] = offtarget_weight
            offdict["opt_offtarget"]["weights"]["contact_max_n_viol"] = 0.0
    elif hasattr(self, "_offtarget_opt"):
        self._offtarget_opt["weights"]["contact_max"] = offtarget_weight
        self._offtarget_opt["weights"]["contact_max_n_viol"] = 0.0


def add_clipped_losses(self):
    """
    Apply configured off-target hard gates to existing scalar losses in place.

    ``above`` and ``below`` describe where a loss remains active. Returning the
    original loss key replaces its raw value before ColabDesign applies the
    unchanged weight.
    """

    def loss_fn(inputs, outputs, opt=None, aux=None):
        del outputs  # unused; thresholding reuses already-computed losses
        opt = opt or {}
        aux = aux or {}

        if "offtarget_len" not in inputs:
            return {}

        base_losses = aux.get("losses", {})
        clip = opt.get("clip", {})
        out = {}

        if "i_ptm" in clip and "i_ptm" in base_losses:
            raw_iptm = 1.0 - base_losses["i_ptm"]
            out["i_ptm"] = hard_gate_above(
                raw_iptm,
                clip["i_ptm"]["above"],
                base_losses["i_ptm"],
            )

        if "i_pae" in clip and "i_pae" in base_losses:
            out["i_pae"] = hard_gate_below(
                base_losses["i_pae"],
                clip["i_pae"]["below"],
                base_losses["i_pae"],
            )

        if "ptm_energy" in clip and "ptm_energy" in base_losses:
            out["ptm_energy"] = hard_gate_below(
                base_losses["ptm_energy"],
                clip["ptm_energy"]["below"],
                base_losses["ptm_energy"],
            )

        if "i_con" in clip and "i_con" in base_losses:
            out["i_con"] = hard_gate_below(
                base_losses["i_con"],
                clip["i_con"]["below"],
                base_losses["i_con"],
            )

        return out

    self._callbacks["model"]["loss"].append(loss_fn)


# add helicity loss
def add_helix_loss(self, weight=0.0, offtarget_weight=0.0):
    """
    Add a 'helix' contact pattern loss at offset=3 for binder portion.
    """

    def binder_helicity(inputs, outputs):
        if "offset" in inputs:
            offset = inputs["offset"]
        else:
            idx = inputs["residue_index"].flatten()
            offset = idx[:, None] - idx[None, :]

        # get the distogram from outputs
        dgram = outputs["distogram"]["logits"]
        dgram_bins = get_dgram_bins(outputs)

        # Compute contact matrix and derive current sequence length
        x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=True)
        L = int(x.shape[0])
        bL = int(getattr(self, "_binder_len", 0))
        start = max(L - bL, 0)
        # Binder mask for the last bL residues in the current batch
        binder_mask_1d = jnp.arange(L) >= start
        mask_2d = jnp.logical_and(binder_mask_1d[:, None], binder_mask_1d[None, :])
        # handle offset or mask
        if offset is None:
            if mask_2d is None:
                helix_loss = jnp.diagonal(x, 3).mean()
            else:
                diag_vals = jnp.diagonal(x * mask_2d, 3)
                diag_mask = jnp.diagonal(mask_2d, 3)
                helix_loss = diag_vals.sum() / (diag_mask.sum() + 1e-8)
        else:
            mask = (offset == 3)
            if mask_2d is not None:
                mask = jnp.where(mask_2d, mask, 0)
            helix_loss = jnp.where(mask, x, 0.0).sum() / (mask.sum() + 1e-8)

        return {"helix": helix_loss}

    self._callbacks["model"]["loss"].append(binder_helicity)
    self.opt["weights"]["helix"] = weight

    if hasattr(self, "_offtargets"):
        for offdict in self._offtargets:
            offdict["opt_offtarget"]["weights"]["helix"] = offtarget_weight
    elif hasattr(self, '_offtarget_opt'):
        self._offtarget_opt["weights"]["helix"] = offtarget_weight


# add N- and C-terminus distance loss
def add_termini_distance_loss(self, weight=0.1, offtarget_weight=0.1, threshold_distance=7.0):
    """
    Add a penalty if the binder's N-terminus and C-terminus are 
    too far or too close from a given threshold_distance (7A).
    """

    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        # Only the binder portion
        ca = ca[-self._binder_len:]
        n_terminus = ca[0]
        c_terminus = ca[-1]
        termini_distance = jnp.linalg.norm(n_terminus - c_terminus)

        deviation = jax.nn.elu(termini_distance - threshold_distance)
        termini_distance_loss = jax.nn.relu(deviation)
        return {"NC": termini_distance_loss}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["NC"] = weight

    if hasattr(self, "_offtargets"):
        for offdict in self._offtargets:
            offdict["opt_offtarget"]["weights"]["NC"] = offtarget_weight
    elif hasattr(self, '_offtarget_opt'):
        self._offtarget_opt["weights"]["NC"] = offtarget_weight


# add pTM energy loss for surface penalization
def add_ptm_energy_loss(self, weight=0.1, offtarget_weight=0.1, threshold=0.8, mode="penalty"):
    """
    Add pTM energy loss to penalize or reward specific pTM score ranges.
    
    This loss function uses pTM scores to implement surface penalty - 
    high pTM scores often indicate buried/non-surface binding which may be undesirable.
    
    Args:
        self: The AF design model instance
        weight: Weight for main target pTM energy loss
        offtarget_weight: Weight for off-target pTM energy loss  
        threshold: pTM threshold value (default 0.8)
        mode: Either "penalty" (penalize high pTM) or "reward" (reward optimal range)
    """
    
    def loss_fn(inputs, outputs):
        # Calculate pTM score using the existing get_ptm function
        ptm_score = get_ptm(inputs, outputs, interface=False)
        
        if mode == "penalty":
            # Penalize high pTM scores (surface penalty for buried binding)
            # Use smooth penalty that increases as pTM exceeds threshold
            ptm_energy = jax.nn.relu(ptm_score - threshold)
        elif mode == "reward":
            # Reward pTM scores near the threshold (optimal surface binding)
            # Penalize both too high and too low pTM scores
            deviation = jnp.abs(ptm_score - threshold)
            ptm_energy = jax.nn.elu(deviation - 0.1)  # Allow small deviation around threshold
        else:
            raise ValueError(f"Unknown pTM energy mode: {mode}. Use 'penalty' or 'reward'.")
            
        return {"ptm_energy": ptm_energy}

    # Add the callback to the model loss functions
    self._callbacks["model"]["loss"].append(loss_fn)
    # Set the weight for the main target
    self.opt["weights"]["ptm_energy"] = weight
    
    # Set weights for off-targets
    if hasattr(self, "_offtargets"):
        for offdict in self._offtargets:
            offdict["opt_offtarget"]["weights"]["ptm_energy"] = offtarget_weight
    elif hasattr(self, '_offtarget_opt'):
        self._offtarget_opt["weights"]["ptm_energy"] = offtarget_weight


def plot_trajectory(af_model, design_name, design_paths, topk_list=(1, 3, 5)):
    import os
    import matplotlib.pyplot as plt
    from collections import defaultdict

    metrics_to_plot = [
        'loss', 'plddt', 'ptm', 'i_ptm', 'con',
        'i_con', 'pae', 'i_pae', 'rg', 'ptm_energy',
        'contact_max',
        'contact_max_n_viol',
        'mpnn_struct',
        'mpnn_seq',
        'mpnn_ar_cce',
    ]

    # Curated palette of distinct, aesthetically pleasing colors
    # Based on ColorBrewer Set1, Tableau, and seaborn palettes
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf',  # blue-teal
        '#aec7e8',  # light blue
        '#ffbb78',  # light orange
        '#98df8a',  # light green
        '#ff9896',  # light red
        '#c5b0d5',  # light purple
        '#c49c94',  # light brown
        '#f7b6d2',  # light pink
        '#c7c7c7',  # light gray
        '#dbdb8d',  # light yellow-green
        '#9edae5',  # light teal
    ]

    logs = af_model._tmp["log"]  # List of per-iteration logs

    for metric in metrics_to_plot:
        # 1) Collect data for THIS metric across iterations
        task_series = defaultdict(list)
        for iteration_log in logs:
            tasks_dict = iteration_log.get("tasks", {})
            for task_name, task_data in tasks_dict.items():
                role = task_data.get("_role", "unknown")
                if metric in task_data:
                    task_series[(task_name, role)].append(task_data[metric])

        # If no tasks reported this metric, skip
        if not task_series:
            continue

        # 2) Plot the time-series for this metric
        plt.figure()
        for i, ((task_name, role), values) in enumerate(task_series.items()):
            x = range(1, len(values) + 1)
            color = colors[i % len(colors)]
            
            # Determine role label for each task individually
            if role == 'target':
                role_shorter = 't'
            else:
                role_shorter = 'o'
                
            if role == "target":
                plt.plot(x, values, label=f'{role_shorter} {task_name}', color=color)
            else:  # off-target
                plt.plot(x, values, label=f'{role_shorter} {task_name}', linestyle='--', color=color)

        plt.xlabel('Iterations')
        plt.ylabel(metric)
        plt.title(design_name)
        plt.legend()
        plt.grid(True)

        # 3) Save a PNG for each metric
        out_path = os.path.join(
            design_paths["Trajectory/Plots"], 
            f"{design_name}_{metric}.png"
        )
        plt.savefig(out_path, dpi=150)
        plt.close()


# BindEnergyCraft – pTMEnergy loss (Eq. 7–8, Nori et al. 2025)
# Implementation of the pTMEnergy loss function that replaces ipTM for better binding optimization

def ptm_energy_craft_loss(pae_logits: jnp.ndarray, breaks: jnp.ndarray, 
                         residue_weights=None, asym_id=None) -> jnp.ndarray:
    """
    BindEnergyCraft pTMEnergy loss function following ColabDesign interface patterns.
    
    This reuses the same d0 calculation and interface masking as the existing 
    get_ptm function in ColabDesign for consistency.
    
    Args:
        pae_logits: (L,L,B) raw logits from AF‑Multimer
        breaks: error bin edges from predicted_aligned_error output
        residue_weights: per-residue weights (same as get_ptm)
        asym_id: chain IDs per residue (same as get_ptm interface calculation)
    
    Returns:
        scalar pTMEnergy loss (lower = better binding)
    """
    if residue_weights is None:
        residue_weights = jnp.ones(pae_logits.shape[0])
    
    num_res = residue_weights.shape[0]
    clipped_num_res = jnp.maximum(residue_weights.sum(), 19)
    
    # Use the same d0 calculation as AlphaFold TM-score
    d0 = 1.24 * (clipped_num_res - 15) ** (1./3) - 1.8
    
    # Calculate bin centers (same as AlphaFold confidence.py)
    step = breaks[1] - breaks[0]
    bin_centers = breaks + step / 2
    bin_centers = jnp.append(bin_centers, bin_centers[-1] + step)
    
    # TM-score weighting kernel (same as predicted_tm_score)
    g = 1. / (1 + jnp.square(bin_centers) / jnp.square(d0))
    
    # Compute log partition function with TM weighting
    log_Z = jax.nn.logsumexp(pae_logits + jnp.log(g), axis=-1)  # (L,L)
    
    # Interface mask (same logic as predicted_tm_score)
    if asym_id is None:
        pair_mask = jnp.ones((num_res, num_res), dtype=bool)
    else:
        pair_mask = asym_id[:, None] != asym_id[None, :]  # interface only
    
    # Apply interface mask to log_Z
    masked_log_Z = jnp.where(pair_mask, log_Z, 0.0)
    
    # Weight by residue weights and normalize
    pair_residue_weights = pair_mask * (residue_weights[None, :] * residue_weights[:, None])
    total_weight = pair_residue_weights.sum() + 1e-8
    
    # Return negative average (lower = better binding)
    return -jnp.sum(masked_log_Z * pair_residue_weights) / total_weight


# Add pTMEnergy loss function following existing ColabDesign patterns
def add_ptm_energy_craft_loss(self, weight=0.05, offtarget_weight=0.05):
    """
    Add BindEnergyCraft pTMEnergy loss to replace/supplement ipTM.
    
    This loss uses the pTMEnergy formulation from Nori et al. 2025 (Eq. 7-8)
    which provides dense gradients across the interface for better optimization.
    
    Uses the same interface masking approach as get_ptm(interface=True) for consistency.
    
    Args:
        self: The AF design model instance
        weight: Weight for main target pTMEnergy loss
        offtarget_weight: Weight for off-target pTMEnergy loss
    """
    
    def ptm_energy_loss_fn(inputs, outputs):
        """Loss function callback for ColabDesign - reuses get_ptm interface logic"""
        # Use the same PAE data extraction as get_ptm
        pae_output = outputs.get("predicted_aligned_error")
        if pae_output is None:
            return {"pTMEnergy": 0.0}
        
        pae_logits = pae_output.get("logits")
        breaks = pae_output.get("breaks")
        if pae_logits is None or breaks is None:
            return {"pTMEnergy": 0.0}
        
        # Use the same residue weighting as get_ptm
        residue_weights = inputs.get("seq_mask")
        if residue_weights is None:
            residue_weights = jnp.ones(pae_logits.shape[0])
        
        # Use the same interface identification as get_ptm(interface=True)
        asym_id = inputs.get("asym_id")
        
        # Compute pTMEnergy using the consistent interface
        energy = ptm_energy_craft_loss(pae_logits, breaks, residue_weights, asym_id)
        return {"ptm_energy": energy}

    # Add the loss callback to the model
    self._callbacks["model"]["loss"].append(ptm_energy_loss_fn)
    
    # Set the weight for the main target
    self.opt["weights"]["ptm_energy"] = weight
    
    # Set weights for off-targets
    if hasattr(self, "_offtargets"):
        for offdict in self._offtargets:
            offdict["opt_offtarget"]["weights"]["ptm_energy"] = offtarget_weight
    else:
        # single off-target fallback
        if hasattr(self, '_offtarget_opt'):
            self._offtarget_opt["weights"]["ptm_energy"] = offtarget_weight
