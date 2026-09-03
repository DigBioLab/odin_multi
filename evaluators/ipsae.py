"""Shared iPSAE_min calculation for AF2 and AF3 reevaluations.

Implements the residue-family/d0res method from:
https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


IPSAE_PAE_CUTOFF = 10.0


def _d0(residue_count: int) -> float:
    """Return the protein d0 value used by residue-family iPSAE."""
    length = max(float(residue_count), 27.0)
    return max(1.0, 1.24 * (length - 15.0) ** (1.0 / 3.0) - 1.8)


def _directional_ipsae(
    pae: np.ndarray,
    source_mask: np.ndarray,
    partner_mask: np.ndarray,
    pae_cutoff: float,
) -> float:
    """Return max-over-source-residues iPSAE for one group direction."""
    source = np.flatnonzero(source_mask)
    partner = np.flatnonzero(partner_mask)
    if source.size == 0 or partner.size == 0:
        return 0.0

    best = 0.0
    for index in source:
        values = pae[index, partner]
        valid = np.isfinite(values) & (values < pae_cutoff)
        count = int(valid.sum())
        if count == 0:
            continue
        d0 = _d0(count)
        score = float(np.mean(1.0 / (1.0 + (values[valid] / d0) ** 2.0)))
        best = max(best, score)
    return best


def compute_ipsae_min(
    pae_matrix: Any,
    binder_mask: Any,
    target_mask: Any,
    *,
    pae_cutoff: float = IPSAE_PAE_CUTOFF,
) -> float:
    """Calculate conservative residue-family iPSAE for two chain groups.

    ``binder_mask`` and ``target_mask`` may each select residues from one or
    several chains. The public score is the minimum of the two directional
    max-over-residue scores.
    """
    pae = np.asarray(pae_matrix, dtype=float).squeeze()
    binder = np.asarray(binder_mask, dtype=bool).reshape(-1)
    target = np.asarray(target_mask, dtype=bool).reshape(-1)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError("iPSAE requires a square PAE matrix")
    if binder.size != pae.shape[0] or target.size != pae.shape[0]:
        raise ValueError("iPSAE chain masks must match the PAE matrix")
    if not math.isfinite(float(pae_cutoff)) or pae_cutoff <= 0.0:
        raise ValueError("iPSAE PAE cutoff must be a positive finite number")

    binder_to_target = _directional_ipsae(
        pae, binder, target, float(pae_cutoff)
    )
    target_to_binder = _directional_ipsae(
        pae, target, binder, float(pae_cutoff)
    )
    return min(binder_to_target, target_to_binder)
