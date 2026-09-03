"""Pure helpers for thresholded off-target losses."""

from __future__ import annotations

import jax.numpy as jnp


def hard_gate_above(metric, threshold, loss_value=None):
    """Return ``loss_value`` only when ``metric`` is above the threshold."""
    metric = jnp.asarray(metric, dtype=jnp.float32)
    threshold = jnp.asarray(threshold, dtype=metric.dtype)
    if loss_value is None:
        loss_value = metric
    loss_value = jnp.asarray(loss_value, dtype=metric.dtype)
    return jnp.where(metric > threshold, loss_value, jnp.zeros_like(loss_value))


def hard_gate_below(metric, threshold, loss_value=None):
    """Return ``loss_value`` only when ``metric`` is below the threshold."""
    metric = jnp.asarray(metric, dtype=jnp.float32)
    threshold = jnp.asarray(threshold, dtype=metric.dtype)
    if loss_value is None:
        loss_value = metric
    loss_value = jnp.asarray(loss_value, dtype=metric.dtype)
    return jnp.where(metric < threshold, loss_value, jnp.zeros_like(loss_value))


def contact_rowmax_hinge(contact_prob, pair_mask, kappa):
    """
    Apply the row-wise hard-max contact hinge used for off-target pruning.

    Returns the hinge loss, per-row maxima, and the number of violating rows.
    """
    contact_prob = jnp.asarray(contact_prob, dtype=jnp.float32)
    pair_mask = jnp.asarray(pair_mask, dtype=bool)
    kappa = jnp.asarray(kappa, dtype=contact_prob.dtype)

    masked_contacts = jnp.where(pair_mask, contact_prob, jnp.array(-1e6, dtype=contact_prob.dtype))
    row_max = jnp.max(masked_contacts, axis=1)
    has_partner = jnp.any(pair_mask, axis=1)
    row_max = jnp.where(has_partner, row_max, jnp.zeros_like(row_max))
    violation = jnp.maximum(row_max - kappa, 0.0)
    n_viol = (row_max > kappa).astype(contact_prob.dtype).sum()
    return violation.sum(), row_max, n_viol
