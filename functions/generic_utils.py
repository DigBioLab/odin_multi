####################################
################## General functions
####################################
### Import dependencies
import copy
import os
import json
import jax
import random
import math
import pandas as pd
import numpy as np

# Define labels for dataframes
def generate_dataframe_labels():
    # labels for trajectory
    trajectory_labels = ['Design', 'Protocol', 'Length', 'Seed', 'Helicity', 'Target_Hotspot', 'Sequence', 'InterfaceResidues', 'pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes',
                        'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity', 'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues',
                        'n_InterfaceHbonds', 'InterfaceHbondsPercentage', 'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%',
                        'Binder_Helix%', 'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Target_RMSD', 'TrajectoryTime', 'Notes', 'TargetSettings', 'Filters', 'AdvancedSettings']

    # labels for mpnn designs
    core_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                    'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                    'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%', 
                    'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD', 'Binder_pLDDT', 'Binder_pTM', 'Binder_pAE', 'Binder_RMSD']

    design_labels = ['Design', 'Protocol', 'Length', 'Seed', 'Helicity', 'Target_Hotspot', 'Sequence', 'InterfaceResidues', 'MPNN_score', 'MPNN_seq_recovery']

    for label in core_labels:
        design_labels += ['Average_' + label] + [f'{i}_{label}' for i in range(1, 6)]

    design_labels += ['DesignTime', 'Notes', 'TargetSettings', 'Filters', 'AdvancedSettings']

    final_labels = ['Rank'] + design_labels

    return trajectory_labels, design_labels, final_labels

# Create base directions of the project
def generate_directories(design_path):
    design_path_names = ["Accepted", "Accepted/Ranked", "Accepted/Animation", "Accepted/Plots", "Accepted/Pickle", "Trajectory",
                        "Trajectory/Relaxed", "Trajectory/Plots", "Trajectory/Clashing", "Trajectory/LowConfidence", "Trajectory/Animation", "Trajectory/Pickle",
                        "MPNN", "MPNN/Binder", "MPNN/Sequences", "MPNN/Relaxed", "Rejected"]
    design_paths = {}

    # make directories and set design_paths[FOLDER_NAME] variable
    for name in design_path_names:
        path = os.path.join(design_path, name)
        os.makedirs(path, exist_ok=True)
        design_paths[name] = path

    return design_paths

# update failure rates from trajectories and early predictions
def update_failures(failure_csv, failure_column_or_dict):
    failure_df = pd.read_csv(failure_csv)
    
    def strip_model_prefix(name):
        # Strips the model-specific prefix if it exists
        parts = name.split('_')
        if parts[0].isdigit():
            return '_'.join(parts[1:])
        return name
    
    # update dictionary coming from complex prediction
    if isinstance(failure_column_or_dict, dict):
        # Update using a dictionary of failures
        for filter_name, count in failure_column_or_dict.items():
            stripped_name = strip_model_prefix(filter_name)
            if stripped_name in failure_df.columns:
                failure_df[stripped_name] += count
            else:
                failure_df[stripped_name] = count
    else:
        # Update a single column from trajectory generation
        failure_column = strip_model_prefix(failure_column_or_dict)
        if failure_column in failure_df.columns:
            failure_df[failure_column] += 1
        else:
            failure_df[failure_column] = 1
    
    failure_df.to_csv(failure_csv, index=False)

# Load required helicity value
def load_helicity(advanced_settings):
    if advanced_settings["random_helicity"] is True:
        # will sample a random bias towards helicity
        helicity_value = round(np.random.uniform(-3, 1),2)
    elif advanced_settings["weights_helicity"] != 0:
        # using a preset helicity bias
        helicity_value = advanced_settings["weights_helicity"]
    else:
        # no bias towards helicity
        helicity_value = 0
    return helicity_value

# Report JAX-capable devices
def check_jax_gpu():
    devices = jax.devices()

    has_gpu = any(device.platform == 'gpu' for device in devices)

    if not has_gpu:
        print("No GPU device found, terminating.")
        exit(1)
    else:
        print("Available GPUs:")
        for i, device in enumerate(devices):
            print(f"{device.device_kind}{i + 1}: {device.platform}")

# ---------------------------------------------------------------------------
# Advanced settings are split into two files:
#   --advanced               one general file, run-level settings shared by every context
#   --context SETTINGS LOSS  one loss file per context, paired with its target settings
#
# Every key belongs to exactly one of the two sets, so merging is a union and a
# key appearing in both files is an error. Membership was determined by where
# the code actually reads each key: GENERAL_KEYS are read only from the main
# advanced dict, LOSS_KEYS are read per-context from an auxiliary dict.
# ---------------------------------------------------------------------------
GENERAL_KEYS = frozenset("""
lengths use_multimer_design af_params_dir num_recycles_design sample_models
ensemble_discrete_stages
design_algorithm soft_iterations temporary_iterations hard_iterations greedy_iterations
greedy_percentage use_early_stopping optimise_beta optimise_beta_extra_soft optimise_beta_extra_temp
optimise_beta_recycles_design
mpnn_weights mpnn_use_ligand_context
random_helicity dssp_path
save_design_animations save_design_trajectory_plots remove_unrelaxed_trajectory
omit_AAs
""".split())

LOSS_KEYS = frozenset("""
role gradient_weight
weights_plddt weights_pae_intra weights_pae_inter weights_con_intra weights_con_inter
weights_iptm weights_ptm_energy_craft weights_contact_max weights_rg
weights_termini_loss weights_helicity
use_i_ptm_loss use_ptm_energy_craft_loss
use_rg_loss use_termini_distance_loss
intra_contact_number intra_contact_distance intra_contact_num_pos
inter_contact_number inter_contact_distance inter_contact_num_pos
weights_mpnn_structure_confidence weights_mpnn_sequence_kl weights_mpnn_autoregressive_ce
mpnn_interface_only mpnn_backprop_num_samples
rm_template_seq_design rm_template_sc_design
clip
""".split())

_INTERNAL_KEYS = frozenset({"_file_path", "_general_file_path"})
_CLIP_TERMS = {
    "i_pae": ("below", "weights_pae_inter"),
    "i_con": ("below", "weights_con_inter"),
    "i_ptm": ("above", "weights_iptm"),
    "ptm_energy": ("below", "weights_ptm_energy_craft"),
    "contact_max": ("above", "weights_contact_max"),
}
_REMOVED_ADVANCED_KEYS = {
    "enable_mpnn_backprop": (
        "remove it; MPNN backpropagation is inferred from nonzero MPNN loss weights"
    ),
    "mpnn_backprop_weights": "use the general mpnn_weights setting",
    "weight_initial": "use the context loss file gradient_weight setting",
    "weight_final": "use the context loss file gradient_weight setting",
    "scheduler_type": "remove it; context gradient weights are constant",
    "scheduler_params": "remove it; context gradient weights are constant",
    "num_recycles_validation": "put num_recycles in the AF2 evaluator JSON",
    "rm_template_seq_predict": "put rm_target_seq in the AF2 evaluator JSON",
    "rm_template_sc_predict": "put rm_target_sc in the AF2 evaluator JSON",
    "ipae_threshold": "use clip.i_pae.below",
    "con_inter_threshold": "use clip.i_con.below",
    "iptm_threshold": "use clip.i_ptm.above",
    "ptm_energy_threshold": "use clip.ptm_energy.below",
    "use_contact_max_loss": (
        "use clip.contact_max.above to enable contact_max; omit it to disable"
    ),
    "contact_max": "use clip.contact_max.above",
}


def _validate_advanced_keys(data, allowed, other, path, kind):
    """Reject keys that are unknown, or that belong in the other file."""
    removed = [key for key in data if key in _REMOVED_ADVANCED_KEYS]
    if removed:
        details = "; ".join(
            f"{key}: {_REMOVED_ADVANCED_KEYS[key]}" for key in sorted(removed)
        )
        raise ValueError(f"{path}: removed advanced setting(s): {details}")
    misplaced = sorted(k for k in data if k in other)
    if misplaced:
        where = "a context loss file" if kind == "general" else "the general advanced file"
        raise ValueError(
            f"{path}: {len(misplaced)} key(s) belong in {where}, not here: {', '.join(misplaced)}"
        )
    unknown = sorted(k for k in data if k not in allowed and k not in _INTERNAL_KEYS)
    if unknown:
        raise ValueError(
            f"{path}: unknown advanced setting(s): {', '.join(unknown)}. "
            f"Check for typos; every key must be listed in GENERAL_KEYS or LOSS_KEYS."
        )


def _check_clip_config(loss, path):
    """Validate the optional off-target loss clipping interface."""
    if "clip" not in loss:
        clip = {}
    else:
        if loss.get("role") != "offtarget":
            raise ValueError(f"{path}: clip is only valid for role 'offtarget'.")
        clip = loss["clip"]
        if not isinstance(clip, dict):
            raise ValueError(f"{path}: clip must be a JSON object.")

    unknown = sorted(set(clip) - set(_CLIP_TERMS))
    if unknown:
        valid = ", ".join(
            f"{term} ({weight_key})"
            for term, (_, weight_key) in _CLIP_TERMS.items()
        )
        raise ValueError(
            f"{path}: unknown clipped loss term(s): {', '.join(unknown)}. "
            f"Valid terms and weights are: {valid}."
        )

    for term, rule in clip.items():
        direction, weight_key = _CLIP_TERMS[term]
        if not isinstance(rule, dict) or set(rule) != {direction}:
            raise ValueError(
                f"{path}: clip.{term} must contain exactly "
                f'{{"{direction}": NUMBER}}.'
            )
        threshold = rule[direction]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
        ):
            raise ValueError(
                f"{path}: clip.{term}.{direction} must be a finite number."
            )

        weight = loss.get(weight_key, 0.0)
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
        ):
            raise ValueError(f"{path}: {weight_key} must be a finite number.")
        if float(weight) == 0.0:
            raise ValueError(
                f"{path}: clip.{term} requires nonzero {weight_key}."
            )

    contact_weight = loss.get("weights_contact_max", 0.0)
    if (
        isinstance(contact_weight, bool)
        or not isinstance(contact_weight, (int, float))
        or not math.isfinite(float(contact_weight))
    ):
        raise ValueError(f"{path}: weights_contact_max must be a finite number.")
    if float(contact_weight) != 0.0 and "contact_max" not in clip:
        raise ValueError(
            f"{path}: nonzero weights_contact_max requires "
            "clip.contact_max.above."
        )

    if (
        "ptm_energy" in clip
        and not loss.get("use_ptm_energy_craft_loss", False)
    ):
        raise ValueError(
            f"{path}: clip.ptm_energy requires "
            "use_ptm_energy_craft_loss=true."
        )


def _check_context_loss(loss, path):
    """Validate one complete, constant-weight context loss file."""
    gradient_weight = loss.get("gradient_weight")
    if (
        isinstance(gradient_weight, bool)
        or not isinstance(gradient_weight, (int, float))
        or not math.isfinite(float(gradient_weight))
        or float(gradient_weight) < 0.0
    ):
        raise ValueError(
            f"{path}: gradient_weight must be a finite non-negative number."
        )

    weight_keys = (
        "weights_mpnn_structure_confidence",
        "weights_mpnn_sequence_kl",
        "weights_mpnn_autoregressive_ce",
    )
    missing = [key for key in weight_keys if key not in loss]
    if missing:
        raise ValueError(
            f"{path}: every loss file must explicitly set "
            f"{', '.join(missing)}. Use 0.0 to disable an objective."
        )
    if not isinstance(loss.get("mpnn_interface_only"), bool):
        raise ValueError(f"{path}: mpnn_interface_only must be true or false.")
    samples = loss.get("mpnn_backprop_num_samples")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError(
            f"{path}: mpnn_backprop_num_samples must be a positive integer."
        )

    try:
        weights = [float(loss[key]) for key in weight_keys]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: all MPNN backprop weights must be numeric.") from exc
    if any(not math.isfinite(weight) for weight in weights):
        raise ValueError(f"{path}: all MPNN backprop weights must be finite.")

    role = loss["role"]
    if role == "target" and any(weight < 0 for weight in weights):
        raise ValueError(f"{path}: target MPNN backprop weights must be non-negative.")
    if role == "offtarget" and any(weight > 0 for weight in weights):
        raise ValueError(f"{path}: off-target MPNN backprop weights must be non-positive.")

    _check_clip_config(loss, path)


def merge_advanced(general, loss):
    """Union the general settings with one context loss file.

    The two key sets are disjoint by construction, so a key present in both is a
    bug rather than an override and is reported as such. The general dict is
    deep-copied per target so that in-place mutation for one target cannot leak
    into another.
    """
    clash = sorted(set(general) & set(loss) - _INTERNAL_KEYS)
    if clash:
        raise ValueError(
            f"Key(s) present in both the general file and {loss.get('_file_path', 'a loss file')}: "
            f"{', '.join(clash)}"
        )
    merged = copy.deepcopy(general)
    merged.update(loss)
    merged["_general_file_path"] = general.get("_file_path")
    merged["_file_path"] = loss.get("_file_path")
    return merged


# check all input files being passed
def perform_input_check(args):
    """
    Validate one --advanced (general) and one --target-advanced per --settings.
    """
    # check settings
    settings_paths = []
    if not args.settings:
        raise FileNotFoundError("No --settings provided.")
    for path in args.settings:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Settings file not found: {path}")
        settings_paths.append(path)

    # --advanced is collected with action='append' so that repeating it fails
    # loudly instead of silently keeping the last value, as a plain option would.
    if not args.advanced:
        raise FileNotFoundError("No --advanced provided.")
    if len(args.advanced) > 1:
        raise ValueError(
            f"--advanced takes a single general settings file, but {len(args.advanced)} were given: "
            f"{', '.join(args.advanced)}. Per-target settings now go in --target-advanced."
        )
    general_path = args.advanced[0]
    if not os.path.exists(general_path):
        raise FileNotFoundError(f"General advanced settings file not found: {general_path}")

    # one loss file per target, paired with --settings by occurrence order
    loss_paths = list(args.target_advanced or [])
    if len(loss_paths) != len(settings_paths):
        raise ValueError(
            f"Each --settings needs one --target-advanced: got {len(settings_paths)} settings file(s) "
            f"and {len(loss_paths)} loss file(s). They are paired by occurrence order."
        )
    for path in loss_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Target advanced settings file not found: {path}")

    # check filters
    filters_path = args.filters
    if not os.path.exists(filters_path):
        raise FileNotFoundError(f"Filters file not found: {filters_path}")

    return settings_paths, general_path, loss_paths, filters_path

# check specific advanced settings
def perform_advanced_settings_check(advanced_settings, bindcraft_folder):
    ensemble_discrete_stages = advanced_settings.get("ensemble_discrete_stages", False)
    if not isinstance(ensemble_discrete_stages, bool):
        raise ValueError("ensemble_discrete_stages must be a boolean.")
    advanced_settings["ensemble_discrete_stages"] = ensemble_discrete_stages

    use_early_stopping = advanced_settings.get("use_early_stopping", False)
    if not isinstance(use_early_stopping, bool):
        raise ValueError("use_early_stopping must be a boolean.")
    advanced_settings["use_early_stopping"] = use_early_stopping

    # set paths to model weights and executables
    if bindcraft_folder == "colab":
        advanced_settings["af_params_dir"] = '/content/bindcraft/params/'
        advanced_settings["dssp_path"] = '/content/bindcraft/functions/dssp'
    else:
        # Set paths individually if they are not already set
        if not advanced_settings["af_params_dir"]:
            advanced_settings["af_params_dir"] = bindcraft_folder
        if not advanced_settings["dssp_path"]:
            advanced_settings["dssp_path"] = os.path.join(bindcraft_folder, 'functions', 'dssp')

    # check formatting of omit_AAs setting
    if advanced_settings["omit_AAs"] in [None, False, '']:
        advanced_settings["omit_AAs"] = None
    elif isinstance(advanced_settings["omit_AAs"], str):
        advanced_settings["omit_AAs"] = advanced_settings["omit_AAs"].strip()

    return advanced_settings

# Load settings from JSONs
def load_json_settings(settings_paths, general_path, loss_paths, filters_path=None):
    """Load settings, and resolve one complete advanced dict per target.

    Each returned advanced dict is the union of the general file and that
    target's loss file, so everything downstream keeps seeing a single complete
    dict and needs no knowledge of the split.
    """
    # load settings
    settings_list = []
    for sp in settings_paths:
        with open(sp, "r") as f:
            sdata = json.load(f)
            if "role" in sdata:
                raise ValueError(
                    f"{sp}: role belongs in the paired --context loss file, "
                    "not in the target settings."
                )
            sdata["_file_path"] = sp
            if "lengths" in sdata:
                raise ValueError(
                    f"{sp}: lengths belongs in the general --advanced file, "
                    "not in --settings."
                )
            settings_list.append(sdata)

    # load the general advanced file
    with open(general_path, "r") as f:
        general = json.load(f)
    _validate_advanced_keys(general, GENERAL_KEYS, LOSS_KEYS, general_path, "general")
    general["_file_path"] = general_path

    # load each context loss file and merge it onto the general settings
    advanced_list = []
    for pp in loss_paths:
        with open(pp, "r") as f:
            loss = json.load(f)
        _validate_advanced_keys(loss, LOSS_KEYS, GENERAL_KEYS, pp, "loss")
        role = str(loss.get("role", "")).strip().lower()
        if role not in {"target", "offtarget", "internal"}:
            raise ValueError(
                f"{pp}: role must be 'target', 'offtarget', or 'internal'."
            )
        loss["role"] = role
        _check_context_loss(loss, pp)
        loss["_file_path"] = pp
        advanced_list.append(merge_advanced(general, loss))

    if not any(
        advanced.get("role") == "target"
        and float(advanced.get("gradient_weight", 0.0)) > 0.0
        for advanced in advanced_list
    ):
        raise ValueError("At least one target loss file must have gradient_weight > 0.")

    filters = {}
    if filters_path is not None:
        with open(filters_path, 'r') as f:
            filters = json.load(f)

    return settings_list, advanced_list, filters


# AF2 model settings, make sure non-overlapping models with template option are being used for design and re-prediction
def load_af2_models(af_multimer_setting):
    if af_multimer_setting:
        design_models = [0,1,2,3,4]
        prediction_models = [0,1]
        multimer_validation = False
    else:
        design_models = [0,1]
        prediction_models = [0,1,2,3,4]
        multimer_validation = True

    return design_models, prediction_models, multimer_validation

# create csv for insertion of data
def create_dataframe(csv_file, columns):
    if not os.path.exists(csv_file):
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_file, index=False)

# insert row of statistics into csv
def insert_data(csv_file, data_array):
    df = pd.DataFrame([data_array])
    df.to_csv(csv_file, mode='a', header=False, index=False)

# save generated sequence
def save_fasta(design_name, sequence, design_paths):
    fasta_path = os.path.join(design_paths["MPNN/Sequences"], design_name+".fasta")
    with open(fasta_path,"w") as fasta:
        line = f'>{design_name}\n{sequence}'
        fasta.write(line+"\n")

# clean unnecessary rosetta information from PDB
def clean_pdb(pdb_file):
    # Read the pdb file and filter relevant lines
    with open(pdb_file, 'r') as f_in:
        relevant_lines = [line for line in f_in if line.startswith(('ATOM', 'HETATM', 'MODEL', 'TER', 'END'))]

    # Write the cleaned lines back to the original pdb file
    with open(pdb_file, 'w') as f_out:
        f_out.writelines(relevant_lines)

# calculate averages for statistics
def calculate_averages(statistics, handle_aa=False):
    # Initialize a dictionary to hold the sums of each statistic
    sums = {}
    # Initialize a dictionary to hold the sums of each amino acid count
    aa_sums = {}

    # Iterate over the model numbers
    for model_num in range(1, 6):  # assumes models are numbered 1 through 5
        # Check if the model's data exists
        if model_num in statistics:
            # Get the model's statistics
            model_stats = statistics[model_num]
            # For each statistic, add its value to the sum
            for stat, value in model_stats.items():
                # If this is the first time we've seen this statistic, initialize its sum to 0
                if stat not in sums:
                    sums[stat] = 0

                if value is None:
                    value = 0

                # If the statistic is mpnn_interface_AA and we're supposed to handle it separately, do so
                if handle_aa and stat == 'InterfaceAAs':
                    for aa, count in value.items():
                        # If this is the first time we've seen this amino acid, initialize its sum to 0
                        if aa not in aa_sums:
                            aa_sums[aa] = 0
                        aa_sums[aa] += count
                else:
                    sums[stat] += value

    # Now that we have the sums, we can calculate the averages
    averages = {stat: round(total / len(statistics), 2) for stat, total in sums.items()}

    # If we're handling aa counts, calculate their averages
    if handle_aa:
        aa_averages = {aa: round(total / len(statistics),2) for aa, total in aa_sums.items()}
        averages['InterfaceAAs'] = aa_averages

    return averages
