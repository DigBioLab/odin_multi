"""Focused differentiable MPNN support for BindCraft design.

The publication pipeline retains three objectives: structural NLL and sequence
KL (registered in colabdesign_utils.py), plus the sampled autoregressive
cross-entropy implemented here. ProteinMPNN and LigandMPNN share this adapter.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from colabdesign.af.alphafold.common import residue_constants as _rc


ATOM_ORDER = tuple(_rc.atom_order[name] for name in ("N", "CA", "C", "O"))
LOG_PROB_MIN = -10.0
CROSS_ENTROPY_CAP = 10.0


@dataclass(frozen=True)
class ProteinMPNNARConfig:
    num_samples: int = 16
    mask_cutoff: float = 8.0
    mask_tau: float = 1.0


def _extract_binder_logits(inputs: dict, binder_len: int) -> jnp.ndarray | None:
    seq_logits = inputs.get("seq", {}).get("logits")
    if seq_logits is not None:
        if seq_logits.ndim == 3:
            seq_logits = seq_logits[0]
        if seq_logits.shape[-1] > 20:
            seq_logits = seq_logits[..., :20]
        if seq_logits.shape[0] >= binder_len:
            seq_logits = seq_logits[-binder_len:]
        else:
            seq_logits = None
    return seq_logits


def _binder_soft(inputs: dict, binder_len: int) -> jnp.ndarray:
    seq_logits = _extract_binder_logits(inputs, binder_len)
    if seq_logits is not None:
        return jax.nn.softmax(seq_logits, axis=-1)
    return jax.nn.one_hot(
        inputs["aatype"][-binder_len:],
        20,
        dtype=jnp.float32,
    )


def _binder_last_decoding_order(key, total_length: int, binder_start: int) -> jnp.ndarray:
    order = jax.random.uniform(key, shape=(total_length,))
    order = order.at[binder_start:].add(2.0)
    return jnp.argsort(order)


def _mpnn_backend_kind(mpnn_model) -> str:
    return getattr(mpnn_model, "_bindcraft_backend", "protein")


def _empty_atom_context(binder_start: int):
    if binder_start <= 0:
        raise RuntimeError(
            "LigandMPNN protein-context backend requires a non-empty target prefix."
        )
    return (
        jnp.zeros((1, 3), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.float32),
    )


def _mpnn_context_kwargs(
    mpnn_model,
    atom_positions: jnp.ndarray,
    atom_mask: jnp.ndarray,
    binder_start: int,
) -> dict:
    if _mpnn_backend_kind(mpnn_model) != "ligand":
        return {}

    ligand_config = getattr(mpnn_model, "_bindcraft_ligand_cfg", None)
    if ligand_config is None:
        raise RuntimeError("LigandMPNN backend is missing cached context config.")

    total_length = atom_positions.shape[0]
    y, y_type, y_mask = _empty_atom_context(binder_start)
    chain_mask = jnp.concatenate(
        [
            jnp.zeros((binder_start,), dtype=jnp.float32),
            jnp.ones((total_length - binder_start,), dtype=jnp.float32),
        ],
        axis=0,
    )
    return {
        "Y": y,
        "Y_t": y_type,
        "Y_m": y_mask,
        "cutoff_for_score": float(ligand_config["cutoff"]),
        "use_atom_context": True,
        "chain_mask": chain_mask,
        "xyz_37": atom_positions,
        "xyz_37_m": atom_mask,
        "ligand_mpnn_use_side_chain_context": True,
    }


def _mpnn_score_call(
    mpnn_model,
    *,
    atom_positions: jnp.ndarray,
    atom_mask: jnp.ndarray,
    residue_idx: jnp.ndarray,
    chain_idx: jnp.ndarray,
    key,
    binder_start: int,
    **kwargs,
):
    """Score a structure while preserving gradients to its coordinates.

    ProteinMPNN parameters are closed-over constants here, so allowing the
    coordinate gradient does not train ProteinMPNN. It supplies the intended
    structural objective gradient to the AF2 design trajectory.
    """
    atom_mask = jax.lax.stop_gradient(atom_mask)
    score_kwargs = {
        "X": atom_positions[:, ATOM_ORDER],
        "mask": atom_mask[:, 1].astype(jnp.float32),
        "residue_idx": residue_idx,
        "chain_idx": chain_idx,
        "key": key,
    }
    score_kwargs.update(
        _mpnn_context_kwargs(mpnn_model, atom_positions, atom_mask, binder_start)
    )
    score_kwargs.update(kwargs)
    return mpnn_model._score(**score_kwargs)


def _mpnn_sample_call(
    mpnn_model,
    *,
    atom_positions: jnp.ndarray,
    atom_mask: jnp.ndarray,
    residue_idx: jnp.ndarray,
    chain_idx: jnp.ndarray,
    key,
    binder_start: int,
    **kwargs,
):
    """Call a backend sampler against a frozen structure snapshot."""
    atom_positions = jax.lax.stop_gradient(atom_positions)
    atom_mask = jax.lax.stop_gradient(atom_mask)
    sample_kwargs = {
        "X": atom_positions[:, ATOM_ORDER],
        "mask": atom_mask[:, 1].astype(jnp.float32),
        "residue_idx": residue_idx,
        "chain_idx": chain_idx,
        "key": key,
    }
    sample_kwargs.update(
        _mpnn_context_kwargs(mpnn_model, atom_positions, atom_mask, binder_start)
    )
    sample_kwargs.update(kwargs)
    return mpnn_model._sample(**sample_kwargs)


def _interface_weights(
    final_positions: jnp.ndarray,
    binder_start: int,
    cutoff: float,
    tau: float,
) -> jnp.ndarray:
    """Return the existing soft C-alpha interface weights for binder residues."""
    binder_ca = final_positions[binder_start:, _rc.atom_order["CA"], :]
    target_ca = final_positions[:binder_start, _rc.atom_order["CA"], :]
    binder_len = binder_ca.shape[0]
    if binder_len == 0 or target_ca.shape[0] == 0:
        return jnp.ones(binder_len, dtype=jnp.float32)
    distances = jnp.linalg.norm(
        binder_ca[:, None, :] - target_ca[None, :, :],
        axis=-1,
    )
    minimum_distance = jnp.min(distances, axis=1)
    return jax.nn.sigmoid(
        (cutoff - minimum_distance) / max(tau, 1e-6)
    )


def _autoregressive_teacher(
    mpnn_model,
    *,
    target_onehot: jnp.ndarray,
    atom_positions: jnp.ndarray,
    atom_mask: jnp.ndarray,
    residue_idx: jnp.ndarray,
    chain_idx: jnp.ndarray,
    binder_len: int,
    binder_start: int,
    total_length: int,
    num_samples: int,
    key,
) -> jnp.ndarray:
    bias = jnp.concatenate(
        [
            1e7 * target_onehot,
            jnp.zeros((binder_len, 20), dtype=jnp.float32),
        ],
        axis=0,
    )

    def one_sample(sample_key):
        order_key, sequence_key = jax.random.split(sample_key, 2)
        decoding_order = _binder_last_decoding_order(
            order_key,
            total_length,
            binder_start,
        )
        sampled = _mpnn_sample_call(
            mpnn_model,
            atom_positions=atom_positions,
            atom_mask=atom_mask,
            residue_idx=residue_idx,
            chain_idx=chain_idx,
            key=sequence_key,
            binder_start=binder_start,
            temperature=0.1,
            bias=bias,
            decoding_order=decoding_order,
        )
        return sampled["S"][binder_start:, :20]

    teachers = jax.vmap(one_sample)(jax.random.split(key, num_samples))
    return teachers.mean(axis=0)


def compute_ar_cce_loss(
    mpnn_model,
    config: ProteinMPNNARConfig,
    *,
    inputs: dict,
    outputs: dict,
    binder_len: int,
    key,
) -> jnp.ndarray:
    """Compute sampled AR teacher cross-entropy for one design context."""
    final_positions = outputs["structure_module"]["final_atom_positions"]
    final_mask = outputs["structure_module"]["final_atom_mask"]
    total_length = final_positions.shape[0]
    binder_start = total_length - binder_len
    residue_idx = inputs["residue_index"]
    chain_idx = inputs.get(
        "asym_id",
        jnp.zeros(total_length, dtype=jnp.int32),
    )
    binder_soft = _binder_soft(inputs, binder_len)
    target_onehot = jax.nn.one_hot(
        inputs["aatype"][:binder_start],
        20,
        dtype=jnp.float32,
    )
    signed_weight = jnp.asarray(
        inputs.get("opt", {}).get("weights", {}).get("mpnn_ar_cce", 0.0),
        dtype=jnp.float32,
    )

    def active_teacher(_):
        teacher = _autoregressive_teacher(
            mpnn_model,
            target_onehot=target_onehot,
            atom_positions=final_positions,
            atom_mask=final_mask,
            residue_idx=residue_idx,
            chain_idx=chain_idx,
            binder_len=binder_len,
            binder_start=binder_start,
            total_length=total_length,
            num_samples=config.num_samples,
            key=key,
        )
        return jax.lax.stop_gradient(teacher)

    teacher = jax.lax.cond(
        signed_weight != 0.0,
        active_teacher,
        lambda _: jnp.zeros((binder_len, 20), dtype=jnp.float32),
        operand=None,
    )

    seq_logits = _extract_binder_logits(inputs, binder_len)
    if seq_logits is not None:
        log_probabilities = jax.nn.log_softmax(seq_logits, axis=-1)
    else:
        log_probabilities = jnp.log(
            jnp.clip(binder_soft, jnp.exp(LOG_PROB_MIN), 1.0)
        )
    log_probabilities = jnp.nan_to_num(
        log_probabilities,
        nan=LOG_PROB_MIN,
        neginf=LOG_PROB_MIN,
        posinf=0.0,
    )
    log_probabilities = jnp.clip(log_probabilities, LOG_PROB_MIN, 0.0)
    per_position = jnp.clip(
        -(teacher * log_probabilities).sum(-1),
        0.0,
        CROSS_ENTROPY_CAP,
    )

    full_loss = per_position.mean()
    interface_weights = _interface_weights(
        final_positions,
        binder_start,
        cutoff=config.mask_cutoff,
        tau=config.mask_tau,
    )
    interface_loss = (
        jnp.sum(interface_weights * per_position)
        / (jnp.sum(interface_weights) + 1e-8)
    )
    interface_only = jnp.asarray(
        inputs.get("opt", {}).get("mpnn_interface_only", False),
        dtype=bool,
    )
    return jnp.where(interface_only, interface_loss, full_loss)
