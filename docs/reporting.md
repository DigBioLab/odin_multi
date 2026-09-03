# Reporting reference

[Back to the README](../README.md)

The `summarize` command combines the selected AlphaFold 2 design frame with
every available named AF2 and AF3 reevaluation:

```bash
python -u odin_multi.py summarize --run-dir outputs/my_run
```

It refreshes evaluator result and failure CSVs from their job artifacts, then
writes the publication-facing report to `outputs/my_run/04_summary/`. To
inspect one named reevaluation alongside its original selected AF2 frames,
narrow both fields:

```bash
python -u odin_multi.py summarize \
  --run-dir outputs/my_run \
  --evaluator af3 \
  --evaluation-name af3_standard
```

`--evaluator` and `--evaluation-name` must either both be present or both be
omitted.

## Replicates and completeness

`04_summary/data/replicates.csv` preserves every individual AF2 model/seed
and AF3 seed/sample value. Headline summaries in
`04_summary/data/contexts.csv` use the arithmetic mean across configured
replicates for each design and context.

A design/context job with fewer than its expected replicates remains in the raw
CSV but is omitted from means and plots until complete. The generated
`04_summary/README.md` reports expected, completed, missing, and failed job
counts for each source case.

## Report contents

| File | Contents |
| --- | --- |
| `README.md` | starting point, completion audit, medians, and candidate overview |
| `candidates.csv` | ranked, sequence-unique candidates from every available result case |
| `candidates.fasta` | the same ranked candidate sequences in FASTA format |
| `candidate_structures/` | lowest-interface-PAE reevaluation structure for each candidate/context |
| `data/` | replicate, context, design, curve, scatter, and structure-provenance CSVs |
| `figures/` | flat publication figure set in SVG and PNG formats |

Figure filenames begin with a compact source identifier such as
`design_specificity_`, `design_best_i_ptm_`, or `af3_`. A single
reevaluation uses only its evaluator name in publication outputs. Its internal
evaluation name remains in the CSV `evaluation_name` column for provenance.
When multiple evaluations from the same evaluator are summarized together,
their names are appended only where needed to keep filenames distinct.

## Candidate ranking

Rankings are computed independently for every source/evaluation case:

- Target-only and cross-reactivity runs rank every complete, unique sequence by
  worst-target interface PAE ascending; no quality threshold is applied.
- Specificity runs first require
  `min(off-target interface PAE) / max(target interface PAE) >= 1.5`, then
  rank qualifying unique sequences by worst-target interface PAE ascending.

Possible statuses are `ranked`, `incomplete_contexts`, `missing_sequence`,
`missing_target_i_pae`, `missing_specificity_ratio`,
`below_specificity_ratio`, and `duplicate_sequence`. For duplicates,
`duplicate_of_design_id` identifies the retained instance.

## Figures and ratio direction

Specificity reports use target-versus-off-target scatters and ratio-yield plots
for iPTM and interface PAE, without redundant generic threshold curves.
iPSAE_min additionally retains its threshold curves. Cross-reactivity reports
retain threshold curves for all available metrics, and their two-target
scatters mark the region where both targets pass and report the passing count.

For specificity, a design first passes a fixed target cutoff: iPTM > 0.5,
iPSAE_min > 0.60, or interface PAE < 7.5 Å. The curve then sweeps a separation
ratio from 1 to 5. The ratio direction depends on whether higher or lower
values indicate better binding:

| Metric | Specificity ratio | Desired direction |
| --- | --- | --- |
| iPTM | `min(target) / max(off-target)` | larger |
| iPSAE_min | `min(target) / max(off-target)` | larger |
| interface PAE | `min(off-target) / max(target)` | larger |

Shaded bands are deterministic pointwise 95% binomial percentile intervals
from 2,000 draws with seed 0.

Scatter plots adapt to the configured contexts:

- With any off-targets, they plot the weakest target against the strongest
  off-target: minimum target iPTM versus maximum off-target iPTM, minimum
  target iPSAE_min versus maximum off-target iPSAE_min, or maximum target
  interface PAE versus minimum off-target interface PAE.
- With exactly two targets and no off-targets, they plot target A directly
  against target B.
- With three or more targets and no off-targets, they plot the target mean
  against the weakest target iPTM/iPSAE_min or worst target interface PAE.
- With one target and no off-targets, they plot binder pLDDT against target
  iPTM, iPSAE_min, or interface PAE.

## Illustrative specificity result

Suppose two designs are reevaluated against one target and one off-target with
five AF3 samples configured per context:

| Design | Context | Role | Raw samples | Complete mean |
| --- | --- | --- | ---: | ---: |
| `d000` | target | target | 5/5 | iPTM 0.80, interface PAE 6.0 Å |
| `d000` | off-target | offtarget | 5/5 | iPTM 0.20, interface PAE 20.0 Å |
| `d001` | target | target | 5/5 | iPTM 0.62, interface PAE 9.5 Å |
| `d001` | off-target | offtarget | 4/5 | omitted until complete |

The AF3 audit reports two selected designs, four expected context jobs, three
completed jobs, one missing job, one failure record, and 19 raw samples. The
incomplete four-sample group remains in the raw CSV but does not bias the mean.

The iPTM specificity scatter contains `d000` at `(0.80, 0.20)`: minimum
target iPTM on the x-axis and maximum off-target iPTM on the y-axis. Its
interface-PAE scatter contains `d000` at `(6.0, 20.0)`: maximum target
interface PAE versus minimum off-target interface PAE. `d001` appears after
its fifth off-target sample completes.

At the fixed target iPTM cutoff of 0.5 and the annotated specificity threshold
of 1.5, the iPTM curve uses the one complete design (`n=1`) and reports 100%
because `0.80 / 0.20 = 4`. Exact plotted values remain in
`04_summary/data/curves.csv` and `scatters.csv`.

For candidate ranking, `d000` qualifies because its interface-PAE ratio is
`20.0 / 6.0 = 3.33`; it receives candidate rank 1. The incomplete `d001`
remains in `04_summary/data/designs.csv` with status
`incomplete_contexts` and no rank.

Each source case uses its own available complete results. Candidate rankings do
not force a cross-source cohort intersection or make the final experimental
selection.
