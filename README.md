# Odin-Multi: multi-target protein binder design

Odin-Multi extends AlphaFold2-based binder hallucination from a single-complex
objective to a unified multi-complex optimisation framework. Within each design
trajectory, the same binder sequence is evaluated independently against all
specified on-target and off-target structures, producing separate predictions
and objectives for each complex. These objectives jointly update the shared
sequence, allowing multiple desired and undesired interactions to influence
optimisation within the same trajectory.

Depending on the configured contexts, Odin-Multi can be used for cross-reactive
design across multiple on-targets, specificity design through explicit
off-target counter-selection, or combinations of the two.

## Documentation map

- [Install and validate](#installation)
- [Run the target/off-target example](#quick-start-target-versus-off-target)
- [Run the cross-reactivity example](#quick-start-cross-reactivity)
- [Configure targets, losses, and optimization](docs/configuration.md)
- [Design, select, and reevaluate](#design-trajectories)
- [Interpret summaries, rankings, and figures](docs/reporting.md)
- [Understand the run directory](#outputs)
- [Troubleshoot common failures](#troubleshooting)

## Method overview

![Odin-Multi workflow: fixed protein contexts feed a shared binder sequence that is predicted against every target and off-target complex, scored with attractive and repulsive losses, and updated through merged gradients before selection, re-evaluation, and filtering.](odin_multi_workflow.svg)

*(**a**–**b**) Fixed on-target and off-target contexts drive one shared binder
sequence. (**c**–**e**) The sequence is predicted against each context with a
frozen AlphaFold2; attractive and repulsive losses are merged into a single
gradient update. (**f**) Optimisation runs soft, annealed, and discrete stages.
(**g**–**i**) Converged sequences are re-evaluated, filtered, and grouped into
specific, cross-reactive, or selectively cross-reactive sets.*

## Workflow

Each stage is a subcommand of `odin_multi.py`:

- **`design`** — optimise one shared binder sequence across all configured
  target and off-target contexts using AlphaFold2/ColabDesign.
- **`select`** — choose a discrete sequence from each completed design
  trajectory according to the desired interaction profile.
- **`preprocess-af3`** — cache AlphaFold3 MSAs and templates for the fixed
  targets, once per target set.
- **`evaluate`** — independently predict selected sequences against every
  context with AlphaFold3 by default, or AlphaFold2 as an additional evaluator.
- **`summarize`** — compare per-context predictions and generate candidate
  rankings, plots, and CSV/FASTA outputs.

`run` chains design, selection, and evaluation in a single call; the staged
subcommands are preferable on a cluster, where the stages have different
resource requirements.

Design and re-evaluation are kept separate so that the intended interaction
profile can be assessed outside the optimisation trajectory.

## Installation

### Requirements

Odin-Multi uses AlphaFold 2 through ColabDesign for binder design. AlphaFold 3
is the default reevaluation backend; AF2 reevaluation remains available as an
additional local option.

The core installation requires:

- Linux, Git, and Conda or Mamba
- an NVIDIA GPU supported by JAX for design and AF2 reevaluation
- enough local storage for the AlphaFold 2 parameters

AF3 is external to Odin-Multi. Its data pipeline needs access to the AF3 genetic
databases, and inference needs the AF3 model parameters and a supported GPU.

### Install Odin-Multi and AlphaFold 2

Clone with the pinned custom ColabDesign submodule, then run the shell installer
from the repository root:

```bash
git clone --recurse-submodules \
  https://github.com/DigBioLab/odin_multi.git
cd odin_multi
```

```bash
bash install_odin_multi.sh --cuda 12.4 --pkg-manager conda
conda activate Odin-Multi
```

For an existing clone, `git submodule update --init --recursive` initializes
the same pinned checkout. The installer also runs this command automatically,
so forgetting `--recurse-submodules` during clone is recoverable.

Replace `12.4` with the CUDA version exposed by the machine. To resolve
packages with Mamba instead, use `--pkg-manager mamba`; activate the resulting
environment with `conda activate Odin-Multi` in either case.

The installer creates the `Odin-Multi` environment, initializes and verifies
the repository's pinned custom ColabDesign submodule, installs it as an editable
Python package, downloads the AF2 parameters, and makes the bundled DSSP and
DAlphaBall executables runnable. It does not install AlphaFold 3.

Validate the core installation before starting a run:

```bash
python validate_install.py
```

This checks the important imports, ColabDesign Gitlink revision, clean submodule
state and import location, AF2 pTM weights, DSSP executable, and JAX GPU visibility.
On a login or CPU-only node, use `python validate_install.py --allow-cpu` to
turn only the missing-GPU failure into a warning.

`environment.yml` records the core environment dependencies, but
`install_odin_multi.sh` is the supported complete setup because it also pins and
verifies ColabDesign and installs the AF2 parameters.

### Install and configure AlphaFold 3

Install AF3 separately using the
[official AlphaFold 3 installation guide](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md),
including its model parameters and genetic databases. The AF3 license and model
parameter terms apply independently of Odin-Multi.

Copy `settings_reevaluation/af3.example.json` to a local configuration and
replace every placeholder path:

```bash
cp settings_reevaluation/af3.example.json \
  settings_reevaluation/af3.local.json
```

```json
{
  "python": "/path/to/af3/bin/python",
  "run_alphafold": "/path/to/alphafold3/run_alphafold.py",
  "model_dir": "/path/to/alphafold3/models",
  "db_dir": "/path/to/alphafold3/databases",
  "target_cache_dir": "/shared/path/odin_multi_af3_target_cache",
  "seeds": [1],
  "data_pipeline_flags": {},
  "extra_flags": {}
}
```

Here AF3 target cache directory will save the precomputed MSAs that would be reused among the runs.

Then validate both the core and AF3 paths without downloading data or running
inference:

```bash
python validate_install.py \
  --af3-config settings_reevaluation/af3.local.json
```

Odin-Multi invokes the configured `run_alphafold.py` with the configured Python
executable. Target preprocessing is a separate, reusable CPU/database step;
AF3 inference adds the designed binder with an empty MSA and no templates.

Every AF3 subprocess runs under the environment captured when preprocessing or
evaluation starts, so anything the job script exports — `XLA_FLAGS` in
particular — reaches all predictions in a worker regardless of what later
imports do to `os.environ`. On pre-Ampere GPUs, export
`XLA_FLAGS=--xla_disable_hlo_passes=custom-kernel-fusion-rewriter` in the job
script and set `"flash_attention_implementation": "xla"` in `extra_flags`.

## Quick start: target versus off-target

This smoke run uses the shipped structures
`example/target_a_trunc_101.pdb` and `example/target_d_trunc_101.pdb`. It creates
one design so that configuration and execution can be checked; production runs
normally request more trajectories.

The two 229-residue, three-chain contexts share chains A and B and differ by one
residue in the nine-residue chain C: `SLLMWITQC` in the target and `SLLAWITQC`
in the off-target. All nine chain-C residues are hotspots. This deliberately
close pair demonstrates how to ask for binding to one presented sequence while
counter-selecting against another. The shipped general settings design an
80-residue binder.

First create the run and design one binder:

```bash
python -u odin_multi.py design \
  --run-dir outputs/specificity_smoke \
  --context settings_target/specificity_target.json \
            settings_loss/target.json \
  --context settings_target/specificity_offtarget.json \
            settings_loss/offtarget.json \
  --advanced settings_advanced/general.json \
  --base-seed 42 \
  --num-designs 1
```

Select the best saved hard iteration for specificity:

```bash
python -u odin_multi.py select \
  --run-dir outputs/specificity_smoke \
  --method best_clipped_i_pae_ratio
```

Preprocess the two fixed targets once for AF3:

```bash
python -u odin_multi.py preprocess-af3 \
  --settings settings_target/specificity_target.json \
  --settings settings_target/specificity_offtarget.json \
  --evaluator-config settings_reevaluation/af3.local.json
```

Run and summarize the default AF3 reevaluation. Because AF3 is the default,
`--evaluator af3` is optional and omitted here:

```bash
python -u odin_multi.py evaluate \
  --run-dir outputs/specificity_smoke \
  --selection best_clipped_i_pae_ratio \
  --evaluation-name af3_standard \
  --evaluator-config settings_reevaluation/af3.local.json

python -u odin_multi.py summarize \
  --run-dir outputs/specificity_smoke
```

The first place to inspect is
`outputs/specificity_smoke/04_summary/README.md`. It gives the completion
audit and candidate overview, while `candidates.csv`, `data/`, and
`figures/` contain the ranked table, exact plotted values, and publication
figures.

To finish the smoke test without an AlphaFold 3 installation, replace the
preprocessing and AF3 evaluation above with the bundled AF2 reevaluator:

```bash
python -u odin_multi.py evaluate \
  --run-dir outputs/specificity_smoke \
  --selection best_clipped_i_pae_ratio \
  --evaluation-name af2_standard \
  --evaluator af2 \
  --evaluator-config settings_reevaluation/af2.example.json

python -u odin_multi.py summarize \
  --run-dir outputs/specificity_smoke
```

## Quick start: cross-reactivity

The second shipped example designs one binder against both snake-toxin targets.
Both contexts use `settings_loss/target.json`, so they contribute to the shared
target objective. Each input is a single toxin chain with no explicit hotspot
restriction, making this the simplest example of requiring the same sequence to
work against two targets:

```bash
python -u odin_multi.py design \
  --run-dir outputs/crossreactivity_smoke \
  --context settings_target/toxin_erabutoxin_a.json \
            settings_loss/target.json \
  --context settings_target/toxin_short_neurotoxin_alpha_nk.json \
            settings_loss/target.json \
  --advanced settings_advanced/general.json \
  --base-seed 42 \
  --num-designs 1

python -u odin_multi.py select \
  --run-dir outputs/crossreactivity_smoke \
  --method best_i_ptm
```

Preprocess both target settings, then evaluate and summarize with `best_i_ptm`
as the selection name:

```bash
python -u odin_multi.py preprocess-af3 \
  --settings settings_target/toxin_erabutoxin_a.json \
  --settings settings_target/toxin_short_neurotoxin_alpha_nk.json \
  --evaluator-config settings_reevaluation/af3.local.json

python -u odin_multi.py evaluate \
  --run-dir outputs/crossreactivity_smoke \
  --selection best_i_ptm \
  --evaluation-name af3_standard \
  --evaluator-config settings_reevaluation/af3.local.json

python -u odin_multi.py summarize \
  --run-dir outputs/crossreactivity_smoke
```

Open `outputs/crossreactivity_smoke/04_summary/README.md` and its `figures/`
directory to inspect completion and the two-target pass regions.

## Configuration

A run combines target settings, a paired loss file for each context, one
shared design configuration, and an independent evaluator configuration.
Repeat `--context SETTINGS LOSS` in the desired order; the first context must
have role `target`. Supply exactly one `--advanced` file when creating a run.

Start from the shipped JSON files, then read the
[configuration reference](docs/configuration.md) for field definitions,
target and off-target role semantics, loss signs, clipping scales, and
immutable run inputs. In particular, optimization clips interface PAE on a
normalized 0–1 scale, while selection and reevaluation report it in ångströms.

## Design trajectories

The first design command creates the immutable run manifest and copies its
inputs. `--num-designs` is the desired total number of indexed trajectories,
not the number to add.

```bash
python -u odin_multi.py design \
  --run-dir outputs/my_run \
  --context settings_target/specificity_target.json \
            settings_loss/target.json \
  --context settings_target/specificity_offtarget.json \
            settings_loss/offtarget.json \
  --advanced settings_advanced/general.json \
  --base-seed 42 \
  --num-designs 100
```

The base seed and trajectory index deterministically assign the design seed and
binder length. File locks prevent two workers from claiming the same index.
Completed work is skipped; failed or incomplete work can be retried.

### Continue or extend a run

Do not repeat setup arguments after the run exists:

```bash
# Resume unfinished trajectories among indices 0-99.
python -u odin_multi.py design \
  --run-dir outputs/my_run \
  --num-designs 100

# Preserve indices 0-99 and extend the requested total to 150.
python -u odin_multi.py design \
  --run-dir outputs/my_run \
  --num-designs 150
```

Start a new run directory to change contexts, loss files, general settings, input
PDBs, or the base seed. After extending a run, rerun selection and the desired
named evaluations so the new trajectories are included.

## Select a saved iteration

Selection considers saved hard-stage frames and chooses one sequence from each
completed trajectory.

| Method | Behavior | Appropriate use |
| --- | --- | --- |
| `last` | final saved hard iteration | deterministic endpoint |
| `best_i_ptm` | maximize the lowest iPTM over target contexts | target-only or cross-reactive runs |
| `best_i_pae` | minimize the highest interface PAE over target contexts | target-only or cross-reactive runs |
| `best_clipped_i_pae_ratio` | maximize off-target separation relative to the weakest target | specificity runs with off-targets |

For specificity runs:

```bash
python -u odin_multi.py select \
  --run-dir outputs/my_run \
  --method best_clipped_i_pae_ratio
```

The specificity score is:

```text
min_offtarget(min(interface_PAE, 15 A)) / max_target(interface_PAE)
```

Higher is better. The numerator uses the strongest predicted off-target
interaction and caps its interface PAE at 15 A, while the denominator uses the
weakest predicted target interaction. The method requires at least one
off-target; use `best_i_pae` or `best_i_ptm` otherwise.

The selection name is the method name. Results are written to
`02_selections/METHOD/selection.csv`, `sequences.fasta`, and
`selection.json`. The `figures/selection_overview.{png,svg}` pair shows the
metric that selected each frame. Specificity selections plot target against
off-target interface PAE with the ratio boundary, target cutoff, and clipping
cap; the other methods show their selected metric by design.

## Reevaluate with AlphaFold 3 by default

AF3 reevaluation predicts every selected sequence separately with every saved
target and off-target context. A fixed target's database-derived MSA and
templates are cached once and can be reused across designs and runs when the
target sequence and preprocessing configuration match.

### Preprocess fixed targets

Run preprocessing before `evaluate` or the combined `run` command:

```bash
python -u odin_multi.py preprocess-af3 \
  --settings settings_target/specificity_target.json \
  --settings settings_target/specificity_offtarget.json \
  --evaluator-config settings_reevaluation/af3.local.json
```

Preprocessing needs `db_dir` but does not run AF3 inference. The configured
`target_cache_dir` must be writable from the preprocessing job and readable
from every inference job. Repeating the command reuses compatible complete
entries.

### Run a named AF3 evaluation

```bash
python -u odin_multi.py evaluate \
  --run-dir outputs/my_run \
  --selection best_clipped_i_pae_ratio \
  --evaluation-name af3_standard \
  --evaluator-config settings_reevaluation/af3.local.json
```

AF3 is the default evaluator. Supplying `--evaluator af3` is valid but not
necessary. Named evaluations are independently resumable; completed jobs are
skipped when the same command is submitted again.

Summarize after all evaluation workers have finished:

```bash
python -u odin_multi.py summarize \
  --run-dir outputs/my_run
```

`03_evaluations/af3/NAME/metrics.csv` contains one row per design, context, seed, and AF3
sample. It reports pLDDT, binder pLDDT, pTM, binder-specific iPTM, iPSAE_min,
interface PAE, minimum interface PAE, ranking score, and the predicted structure
path. For multichain targets, `i_ptm` is AF3's mean cross-chain iPTM for the
binder chain, so confidence in target-target interfaces cannot inflate the
reported binder interaction. AF3's whole-complex scalar is retained separately
as `global_i_ptm` for auditing. Failures are collected separately in
`03_evaluations/af3/NAME/failures.csv`.

## Additional AlphaFold 2 reevaluation

AF2 reevaluation is a local BindCraft-style reprediction of the selected fixed
sequences. It is additional to the default AF3 path and uses the AF2 parameters
installed for design. Copy and adjust `settings_reevaluation/af2.example.json`:

```json
{
  "params_dir": "../params",
  "models": [0, 1],
  "seeds": [0],
  "num_recycles": 3,
  "use_multimer": false,
  "rm_target_seq": false,
  "rm_target_sc": false
}
```

Model values are zero-based indices. Run AF2 explicitly:

```bash
python -u odin_multi.py evaluate \
  --run-dir outputs/my_run \
  --selection best_clipped_i_pae_ratio \
  --evaluation-name af2_standard \
  --evaluator af2 \
  --evaluator-config settings_reevaluation/af2.example.json
```

The AF2 evaluator writes `03_evaluations/af2/NAME/metrics.csv` with one raw row per design,
context, model, and seed, including pLDDT, pTM, iPTM, iPSAE_min, interface PAE,
and the predicted structure path.

Both reevaluators calculate only the conservative `ipsae_min` variant, using
the minimum of binder-to-target and target-to-binder residue-family scores at
the standard strict `PAE < 10 Å` cutoff. All configured target chains are
scored together as one target group. Higher values are better; summaries mark
`0.60` as a reference cutoff without filtering designs. Existing AF3 results
can be backfilled from their saved confidence JSON during summarization.
Existing AF2 results require reevaluation because their PAE matrices were not
previously retained.

## Summarize a run

After evaluation workers finish, regenerate all available result cases with:

```bash
python -u odin_multi.py summarize --run-dir outputs/my_run
```

Start with `outputs/my_run/04_summary/README.md` for the completion audit and
candidate overview. Ranked sequences, exact plot data, structures, and
publication figures sit beside it in `candidates.csv`, `data/`,
`candidate_structures/`, and `figures/`.

The [reporting reference](docs/reporting.md) defines replicate completeness,
candidate statuses and ranking, scatter axes, and the direction of each
specificity ratio. Confidence metrics use target/off-target ratios; interface
PAE uses off-target/target because lower PAE indicates stronger binding.

## Combined local command

`run` performs design, selection, and reevaluation sequentially. Its evaluator
default is AF3, but it deliberately does not run AF3 preprocessing. Populate
the target cache first, then run:

```bash
python -u odin_multi.py run \
  --run-dir outputs/combined_run \
  --context settings_target/specificity_target.json \
            settings_loss/target.json \
  --context settings_target/specificity_offtarget.json \
            settings_loss/offtarget.json \
  --advanced settings_advanced/general.json \
  --base-seed 42 \
  --num-designs 1 \
  --method best_clipped_i_pae_ratio \
  --evaluation-name af3_standard \
  --evaluator-config settings_reevaluation/af3.local.json
```

The staged commands are preferable on a cluster because preprocessing,
design, inference, and summarization have different resource requirements.

## Slurm

The provided wrappers keep the same CLI while assigning independent design or
evaluation indices across a contiguous, zero-based job array.

Run target preprocessing on a CPU/database node:

```bash
sbatch odin_multi_cpu.slurm preprocess-af3 \
  --settings settings_target/specificity_target.json \
  --settings settings_target/specificity_offtarget.json \
  --evaluator-config settings_reevaluation/af3.local.json
```

Run 100 design trajectories across eight GPU workers:

```bash
sbatch --array=0-7 odin_multi_gpu.slurm design outputs/my_run \
  --num-designs 100 \
  --context settings_target/specificity_target.json \
            settings_loss/target.json \
  --context settings_target/specificity_offtarget.json \
            settings_loss/offtarget.json \
  --advanced settings_advanced/general.json \
  --base-seed 42
```

Select on CPU, reevaluate on the GPU array, and summarize on CPU:

```bash
sbatch odin_multi_cpu.slurm select outputs/my_run \
  --method best_clipped_i_pae_ratio

sbatch --array=0-7 odin_multi_gpu.slurm evaluate outputs/my_run \
  --selection best_clipped_i_pae_ratio \
  --evaluation-name af3_standard \
  --evaluator-config settings_reevaluation/af3.local.json

sbatch odin_multi_cpu.slurm summarize outputs/my_run
```

Submit dependent stages only after the preceding jobs complete. Resubmission is
safe for completed items, and file locks prevent duplicate active work.

## Outputs

A run has the following high-level layout:

```text
outputs/my_run/
├── run.json
├── 00_inputs/
│   ├── general.json
│   └── contexts/
├── 01_designs/
│   └── t00000/
│       ├── design.json
│       ├── trajectory.pickle
│       └── plots/
├── 02_selections/
│   └── METHOD/
│       ├── selection.csv
│       ├── sequences.fasta
│       ├── selection.json
│       └── figures/
│           ├── selection_overview.png
│           └── selection_overview.svg
├── 03_evaluations/
│   ├── af3/EVALUATION_NAME/
│   │   ├── metrics.csv
│   │   ├── failures.csv
│   │   └── jobs/
│   └── af2/EVALUATION_NAME/
│       ├── metrics.csv
│       ├── failures.csv
│       └── jobs/
├── 04_summary/
│   ├── README.md
│   ├── candidates.csv
│   ├── candidates.fasta
│   ├── candidate_structures/
│   ├── data/
│   └── figures/
└── .pipeline/
    └── locks/
```

`run.json` and `00_inputs/` preserve run provenance. Per-job JSON,
structures, stdout, and stderr remain under each evaluation's `jobs/`
directory. `summarize` regenerates the evaluator CSVs from those job artifacts,
then refreshes the publication-facing report under `04_summary/`. Legacy runs
remain readable and retain their existing directory layout.

## Troubleshooting

- **The first design command asks for setup options:** supply at least one
  `--context SETTINGS LOSS` pair and exactly one `--advanced` file. The first
  loss file must have role `target`.
- **An existing run ignores edited JSON files:** inputs are copied at run
  creation. Use a new `--run-dir` for changed settings.
- **AF3 reports that its target cache is missing:** run `preprocess-af3` for
  every target settings file with the same AF3 configuration used for
  evaluation.
- **AF3 reports different preprocessing settings:** do not mix cache entries
  prepared with incompatible AF3 runner, database, or data-pipeline flags;
  preprocess again into an appropriate cache.
- **JAX sees only a CPU:** run `python validate_install.py`. On a GPU node,
  confirm the NVIDIA driver, CUDA compatibility, environment activation, and
  JAX build. `--allow-cpu` is intended only for validation on CPU-only nodes.
- **Specificity selection rejects a target-only run:**
  `best_clipped_i_pae_ratio` requires an off-target. Use `best_i_pae` or
  `best_i_ptm` for target-only or cross-reactive runs.
- **Some evaluations are absent from the CSV:** inspect
  `03_evaluations/EVALUATOR/NAME/failures.csv` and the corresponding job
  directory, rerun the
  named evaluation, then summarize again.
- **A design/context is missing from means or plots:** its raw replicate count
  is lower than configured. Check `04_summary/README.md` and
  `04_summary/data/replicates.csv`,
  finish or rerun that evaluation job, and summarize again.
- **A run failed partway through:** rerun the same command. Completed indexed
  work is skipped, while incomplete or failed items are retried.

## Current scope

- AF3 is the default reevaluation backend but remains an external installation.
- AF2 is required for design and is available as an additional reevaluator.
- Loss weights, clip settings, and selection scores are computational
  heuristics and should be validated for each campaign.
- More contexts increase design and reevaluation cost because each sequence is
  modeled against every configured context.
- Odin-Multi produces transparent per-case candidate rankings, but it does not
  impose a cross-reactivity quality cutoff or make the final experimental
  selection.

## References and license

Odin-Multi builds on the AlphaFold2/ColabDesign binder-design foundation of
[BindCraft](https://github.com/martinpacesa/BindCraft), diverging after commit
[`8f8c0dc`](https://github.com/martinpacesa/BindCraft/commit/8f8c0dc93328c42e3bcbbaf31a7cba312968835c),
and adds multi-context optimisation, staged sequence selection, independent
AlphaFold2/AlphaFold3 reevaluation, and publication-oriented summaries. See
[NOTICE.md](NOTICE.md) for provenance and third-party notices.

- [BindCraft source](https://github.com/martinpacesa/BindCraft)
- [BindCraft publication](https://doi.org/10.1038/s41586-025-09429-6)
- [ColabDesign](https://github.com/sokrypton/ColabDesign)
- [Odin-Multi ColabDesign fork](https://github.com/DigBioLab/colabdesign-odin-multi)
- [AlphaFold 3](https://github.com/google-deepmind/alphafold3)
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- [PCGrad paper](https://arxiv.org/abs/2001.06782)

Odin-Multi is distributed under the repository's MIT license. External
components, model parameters, and databases retain their own licenses and terms.
