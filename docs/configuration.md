# Configuration reference

[Back to the README](../README.md)

Odin-Multi separates the fixed protein context, the objective applied to that
context, the optimization schedule shared by all contexts, and the independent
reevaluation runtime.

| Layer | Purpose | Example |
| --- | --- | --- |
| Target settings | PDB, chains, hotspots, unique context name | `settings_target/specificity_target.json` |
| Context loss | role, gradient scale, structural losses, MPNN losses | `settings_loss/target.json` |
| General settings | binder length and shared optimization schedule | `settings_advanced/general.json` |
| Evaluator settings | AF3 or AF2 reevaluation runtime | `settings_reevaluation/af3.example.json` |

Repeat `--context SETTINGS LOSS` in the desired context order. The first
context must use a loss file with role `target`. Supply exactly one
`--advanced` file when creating a run.

## Target settings

A minimal target settings file is:

```json
{
  "binder_name": "my_target",
  "starting_pdb": "/absolute/path/to/target.pdb",
  "chains": "A",
  "target_hotspot_residues": "A10,A14-18"
}
```

- `binder_name` is the unique, filesystem-safe name used in plots and result
  tables.
- `starting_pdb` may be absolute or relative to the working directory or the
  settings file.
- `chains` is a comma-separated list of target chains in the PDB.
- `target_hotspot_residues` is a comma-separated set of residues and ranges,
  or `null` when no hotspot is used.

The target settings file does not contain `role` or binder `lengths`; those
belong to the paired context loss file and shared general settings,
respectively.

## Context loss files and roles

Start from the complete shipped loss files:

- `settings_loss/target.json`
- `settings_loss/offtarget.json`

The central target fields include:

```json
{
  "role": "target",
  "gradient_weight": 1.0,
  "weights_pae_inter": 0.1,
  "weights_con_inter": 1.0,
  "weights_iptm": 0.05,
  "weights_mpnn_structure_confidence": 0.0,
  "weights_mpnn_sequence_kl": 0.0,
  "weights_mpnn_autoregressive_ce": 1.0
}
```

The corresponding off-target loss file uses explicit repulsive signs and
clipped loss thresholds:

```json
{
  "role": "offtarget",
  "gradient_weight": 0.3,
  "weights_pae_inter": -0.1,
  "weights_con_inter": -1.0,
  "weights_iptm": -0.05,
  "clip": {
    "i_pae": {"below": 0.35},
    "i_con": {"below": 3.5},
    "i_ptm": {"above": 0.3}
  },
  "weights_mpnn_structure_confidence": 0.0,
  "weights_mpnn_sequence_kl": 0.0,
  "weights_mpnn_autoregressive_ce": -1.0,
  "mpnn_interface_only": true
}
```

These are excerpts for orientation. Use the complete shipped JSON files as the
starting point. Their weights and thresholds are initial settings to tune for a
particular target set, not universal cutoffs.

### Clipping scales

`clip` is optional and valid only for off-target contexts. `above` or
`below` describes where the loss remains active; omitting a term leaves its
ordinary, unclipped loss unchanged.

The `i_pae` clipping value is in the normalized ColabDesign optimization
scale, where interface PAE is divided by 31. Thus `i_pae.below: 0.35`
corresponds to an interface PAE of approximately 10.85 Å. Reevaluation,
selection output, plots, and summary tables report interface PAE directly in
ångströms. Do not copy an ångström cutoff such as `7.5` into `clip.i_pae`;
divide it by 31 first.

| Clip term | Required direction | Scale and behavior |
| --- | --- | --- |
| `i_pae` | `below` | normalized interface-PAE loss; hard gate |
| `i_con` | `below` | ColabDesign interface-contact loss; hard gate |
| `i_ptm` | `above` | iPTM confidence on a 0–1 scale; hard gate |
| `ptm_energy` | `below` | pTM-energy loss; hard gate; requires `use_ptm_energy_craft_loss: true` |
| `contact_max` | `above` | contact probability on a 0–1 scale; row-wise hinge |

The published hotspot-contact objective is configured with a positive weight
and a `contact_max` clip:

```json
{
  "inter_contact_distance": 20.0,
  "weights_contact_max": 0.05,
  "clip": {
    "contact_max": {"above": 0.3}
  }
}
```

It uses the configured off-target hotspots when present and inherits its
distance cutoff from `inter_contact_distance`. Legacy `*_threshold`,
`use_contact_max_loss`, and nested `contact_max` settings are rejected with
a migration message.

## Roles, signs, and gradient combination

`role` classifies a context as `"target"` or `"offtarget"`. Scaled
gradients from all target contexts are summed to form the target consensus.
Each off-target gradient is then PCGrad-projected against that consensus before
being added. `gradient_weight` controls the magnitude of each context's
contribution.

The role does not reverse loss signs. Configure off-target repulsion explicitly
through loss weights and, where desired, `clip` settings. Inverse confidence
objectives normally use negative weights; direct penalties such as
`contact_max` use positive weights. Odin-Multi also uses the role to validate
MPNN weight signs, include only targets in early stopping, distinguish targets
from off-targets during iteration selection, and label trajectory plots and
evaluation results.

The canonical presets enable only autoregressive cross-entropy: `+1.0` for
targets and `-1.0` for off-targets. Structure-conditioned ProteinMPNN NLL and
sequence KL divergence remain available as publication or ablation objectives
but are disabled with zero weights. Every enabled MPNN objective must agree
with the context role: positive for targets and non-positive for off-targets.
`mpnn_interface_only` can restrict enabled off-target MPNN objectives to
interface positions.

For cross-reactive design, pair every context with a target loss file. For
specificity design, place at least one target first, followed by one or more
off-target loss files.

## General design settings

`settings_advanced/general.json` is the canonical shared configuration. The
most important fields are:

| Field | Meaning |
| --- | --- |
| `lengths` | inclusive binder-length bounds; `[80, 80]` fixes length 80 |
| `af_params_dir` | AF2 parameter directory; an empty value uses the repository root, where the installer creates `params/` |
| `use_multimer_design` | selects the AF2 design model family |
| `num_recycles_design` | recycles in ordinary design steps |
| `design_algorithm` | staged optimization algorithm |
| `soft_iterations`, `temporary_iterations`, `hard_iterations`, `greedy_iterations` | optimization schedule |
| `use_early_stopping` | enables target-only early-stopping checks |
| `omit_AAs` | amino acids excluded during design |

The run copies every input configuration and PDB into its `00_inputs/`
directory at creation time. Later edits to the source JSON files do not change
that run; start a new run directory to change immutable inputs.

## Evaluator settings

AlphaFold 3 and AlphaFold 2 reevaluation settings are independent from the
design configuration. Copy an example file before adding local paths:

- `settings_reevaluation/af3.example.json` configures an external AF3 Python,
  runner, model directory, database directory, reusable target cache, seeds,
  and extra runner flags.
- `settings_reevaluation/af2.example.json` configures the installed AF2
  parameters, model indices, seeds, recycles, and target masking.

See the [installation and reevaluation instructions](../README.md#install-and-configure-alphafold-3)
for the corresponding commands.
