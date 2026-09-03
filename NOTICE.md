# Project lineage and third-party notices

## BindCraft

Odin-Multi is built on and derived from BindCraft:

- Source: https://github.com/martinpacesa/BindCraft
- Published work: https://doi.org/10.1038/s41586-025-09429-6
- Upstream lineage point: 8f8c0dc93328c42e3bcbbaf31a7cba312968835c

Odin-Multi retains and adapts BindCraft's AlphaFold2/ColabDesign binder-design
foundation while adding multi-context optimisation, explicit on-target and
off-target objectives, staged selection, independent AlphaFold2/AlphaFold3
reevaluation, and publication-oriented result summaries.

The public Odin-Multi repository begins with a clean release snapshot. Its
relationship to BindCraft is therefore recorded here rather than represented
by shared Git ancestry. BindCraft's MIT license notice is retained in LICENSE.
No endorsement by the BindCraft authors is implied.

## ColabDesign

ColabDesign is included as a pinned Git submodule:

- Upstream: https://github.com/sokrypton/ColabDesign
- Odin-Multi fork: https://github.com/DigBioLab/colabdesign-odin-multi
- Upstream lineage point: 4c0bc6d67f8f967135ecccc135a26b3bfded25e8

The Odin-Multi fork is an official ColabDesign fork. Its Odin-specific release
commit consolidates the required extensions while preserving the exact tested
source tree. ColabDesign remains governed by the license included in that
repository.

## Other components

AlphaFold model parameters, AlphaFold 3, ProteinMPNN, PyRosetta, and other
external components retain their own licenses and usage terms.
