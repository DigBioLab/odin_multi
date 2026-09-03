#!/usr/bin/env python3
"""Convert an official LigandMPNN PyTorch checkpoint into the local Haiku tree."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

try:
  import joblib
except Exception:  # pragma: no cover - joblib is optional in the converter runtime
  joblib = None
  import pickle


def _sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


def _lin(sd, prefix):
  return {
    "w": sd[f"{prefix}.weight"].detach().cpu().numpy().T.astype(np.float32),
    "b": sd[f"{prefix}.bias"].detach().cpu().numpy().astype(np.float32),
  }


def _lin_no_bias(sd, prefix):
  return {
    "w": sd[f"{prefix}.weight"].detach().cpu().numpy().T.astype(np.float32),
  }


def _ln(sd, prefix):
  return {
    "scale": sd[f"{prefix}.weight"].detach().cpu().numpy().astype(np.float32),
    "offset": sd[f"{prefix}.bias"].detach().cpu().numpy().astype(np.float32),
  }


def _embed(sd, prefix):
  return {"W_s": sd[f"{prefix}.weight"].detach().cpu().numpy().astype(np.float32)}


def _dense(sd, prefix, hk_prefix):
  return {
    f"{hk_prefix}/~/position_wise_feed_forward/~/{hk_prefix.split('/')[-1].replace('enc_layer', 'enc').replace('dec_layer', 'dec').replace('context_encoder_layer', 'context').replace('y_context_encoder_layer', 'yctx')}_dense_W_in": _lin(sd, f"{prefix}.dense.W_in"),
    f"{hk_prefix}/~/position_wise_feed_forward/~/{hk_prefix.split('/')[-1].replace('enc_layer', 'enc').replace('dec_layer', 'dec').replace('context_encoder_layer', 'context').replace('y_context_encoder_layer', 'yctx')}_dense_W_out": _lin(sd, f"{prefix}.dense.W_out"),
  }


def _merge(*parts):
  out = {}
  for part in parts:
    out.update(part)
  return out


def _enc_layer(sd, idx):
  hk_prefix = "ligand_mpnn/~/enc_layer" if idx == 0 else f"ligand_mpnn/~/enc_layer_{idx}"
  pt_prefix = f"encoder_layers.{idx}"
  inner = f"enc{idx}"
  out = {
    f"{hk_prefix}/~/{inner}_W1": _lin(sd, f"{pt_prefix}.W1"),
    f"{hk_prefix}/~/{inner}_W2": _lin(sd, f"{pt_prefix}.W2"),
    f"{hk_prefix}/~/{inner}_W3": _lin(sd, f"{pt_prefix}.W3"),
    f"{hk_prefix}/~/{inner}_W11": _lin(sd, f"{pt_prefix}.W11"),
    f"{hk_prefix}/~/{inner}_W12": _lin(sd, f"{pt_prefix}.W12"),
    f"{hk_prefix}/~/{inner}_W13": _lin(sd, f"{pt_prefix}.W13"),
    f"{hk_prefix}/~/{inner}_norm1": _ln(sd, f"{pt_prefix}.norm1"),
    f"{hk_prefix}/~/{inner}_norm2": _ln(sd, f"{pt_prefix}.norm2"),
    f"{hk_prefix}/~/{inner}_norm3": _ln(sd, f"{pt_prefix}.norm3"),
    f"{hk_prefix}/~/position_wise_feed_forward/~/{inner}_dense_W_in": _lin(sd, f"{pt_prefix}.dense.W_in"),
    f"{hk_prefix}/~/position_wise_feed_forward/~/{inner}_dense_W_out": _lin(sd, f"{pt_prefix}.dense.W_out"),
  }
  return out


def _dec_layer(sd, idx):
  hk_prefix = "ligand_mpnn/~/dec_layer" if idx == 0 else f"ligand_mpnn/~/dec_layer_{idx}"
  pt_prefix = f"decoder_layers.{idx}"
  inner = f"dec{idx}"
  return {
    f"{hk_prefix}/~/{inner}_W1": _lin(sd, f"{pt_prefix}.W1"),
    f"{hk_prefix}/~/{inner}_W2": _lin(sd, f"{pt_prefix}.W2"),
    f"{hk_prefix}/~/{inner}_W3": _lin(sd, f"{pt_prefix}.W3"),
    f"{hk_prefix}/~/{inner}_norm1": _ln(sd, f"{pt_prefix}.norm1"),
    f"{hk_prefix}/~/{inner}_norm2": _ln(sd, f"{pt_prefix}.norm2"),
    f"{hk_prefix}/~/position_wise_feed_forward/~/{inner}_dense_W_in": _lin(sd, f"{pt_prefix}.dense.W_in"),
    f"{hk_prefix}/~/position_wise_feed_forward/~/{inner}_dense_W_out": _lin(sd, f"{pt_prefix}.dense.W_out"),
  }


def _context_layer(sd, idx):
  hk_prefix = "ligand_mpnn/~/context_encoder_layer" if idx == 0 else f"ligand_mpnn/~/context_encoder_layer_{idx}"
  pt_prefix = f"context_encoder_layers.{idx}"
  inner = f"context{idx}"
  return {
    f"{hk_prefix}/~/{inner}_W1": _lin(sd, f"{pt_prefix}.W1"),
    f"{hk_prefix}/~/{inner}_W2": _lin(sd, f"{pt_prefix}.W2"),
    f"{hk_prefix}/~/{inner}_W3": _lin(sd, f"{pt_prefix}.W3"),
    f"{hk_prefix}/~/{inner}_norm1": _ln(sd, f"{pt_prefix}.norm1"),
    f"{hk_prefix}/~/{inner}_norm2": _ln(sd, f"{pt_prefix}.norm2"),
    f"{hk_prefix}/~/position_wise_feed_forward/~/{inner}_dense_W_in": _lin(sd, f"{pt_prefix}.dense.W_in"),
    f"{hk_prefix}/~/position_wise_feed_forward/~/{inner}_dense_W_out": _lin(sd, f"{pt_prefix}.dense.W_out"),
  }


def _y_context_layer(sd, idx):
  hk_prefix = "ligand_mpnn/~/y_context_encoder_layer" if idx == 0 else f"ligand_mpnn/~/y_context_encoder_layer_{idx}"
  pt_prefix = f"y_context_encoder_layers.{idx}"
  inner = f"yctx{idx}"
  return {
    f"{hk_prefix}/~/{inner}_W1": _lin(sd, f"{pt_prefix}.W1"),
    f"{hk_prefix}/~/{inner}_W2": _lin(sd, f"{pt_prefix}.W2"),
    f"{hk_prefix}/~/{inner}_W3": _lin(sd, f"{pt_prefix}.W3"),
    f"{hk_prefix}/~/{inner}_norm1": _ln(sd, f"{pt_prefix}.norm1"),
    f"{hk_prefix}/~/{inner}_norm2": _ln(sd, f"{pt_prefix}.norm2"),
    f"{hk_prefix}/~/position_wise_feed_forward/~/{inner}_dense_W_in": _lin(sd, f"{pt_prefix}.dense.W_in"),
    f"{hk_prefix}/~/position_wise_feed_forward/~/{inner}_dense_W_out": _lin(sd, f"{pt_prefix}.dense.W_out"),
  }


def convert_checkpoint(checkpoint_path: Path, output_path: Path, source_commit: str | None):
  checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
  sd = checkpoint["model_state_dict"]

  params = {
    "ligand_mpnn/~/protein_features_ligand/~/positional_encodings/~/embedding_linear": _lin(sd, "features.embeddings.linear"),
    "ligand_mpnn/~/protein_features_ligand/~/edge_embedding": _lin_no_bias(sd, "features.edge_embedding"),
    "ligand_mpnn/~/protein_features_ligand/~/norm_edges": _ln(sd, "features.norm_edges"),
    "ligand_mpnn/~/protein_features_ligand/~/node_project_down": _lin(sd, "features.node_project_down"),
    "ligand_mpnn/~/protein_features_ligand/~/norm_nodes": _ln(sd, "features.norm_nodes"),
    "ligand_mpnn/~/protein_features_ligand/~/type_linear": _lin(sd, "features.type_linear"),
    "ligand_mpnn/~/protein_features_ligand/~/y_nodes": _lin_no_bias(sd, "features.y_nodes"),
    "ligand_mpnn/~/protein_features_ligand/~/y_edges": _lin_no_bias(sd, "features.y_edges"),
    "ligand_mpnn/~/protein_features_ligand/~/norm_y_edges": _ln(sd, "features.norm_y_edges"),
    "ligand_mpnn/~/protein_features_ligand/~/norm_y_nodes": _ln(sd, "features.norm_y_nodes"),
    "ligand_mpnn/~/W_e": _lin(sd, "W_e"),
    "ligand_mpnn/~/embed_token": _embed(sd, "W_s"),
    "ligand_mpnn/~/W_v": _lin(sd, "W_v"),
    "ligand_mpnn/~/W_c": _lin(sd, "W_c"),
    "ligand_mpnn/~/W_nodes_y": _lin(sd, "W_nodes_y"),
    "ligand_mpnn/~/W_edges_y": _lin(sd, "W_edges_y"),
    "ligand_mpnn/~/V_C": _lin_no_bias(sd, "V_C"),
    "ligand_mpnn/~/V_C_norm": _ln(sd, "V_C_norm"),
    "ligand_mpnn/~/W_out": _lin(sd, "W_out"),
  }
  for idx in range(3):
    params.update(_enc_layer(sd, idx))
    params.update(_dec_layer(sd, idx))
  for idx in range(2):
    params.update(_context_layer(sd, idx))
    params.update(_y_context_layer(sd, idx))

  payload = {
    "model_state_dict": params,
    "num_edges": int(checkpoint["num_edges"]),
    "atom_context_num": int(checkpoint["atom_context_num"]),
    "noise_level": float(checkpoint.get("noise_level", 0.0)),
    "ligand_mpnn_use_side_chain_context": True,
    "source_commit": source_commit,
  }

  output_path.parent.mkdir(parents=True, exist_ok=True)
  if joblib is not None:
    joblib.dump(payload, output_path)
  else:  # pragma: no cover
    with output_path.open("wb") as fh:
      pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

  manifest = {
    "source_checkpoint": str(checkpoint_path),
    "source_checkpoint_sha256": _sha256(checkpoint_path),
    "source_commit": source_commit,
    "target_checkpoint": str(output_path),
    "target_checkpoint_sha256": _sha256(output_path),
    "num_edges": payload["num_edges"],
    "atom_context_num": payload["atom_context_num"],
    "noise_level": payload["noise_level"],
  }
  with output_path.with_suffix(".manifest.json").open("w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True, help="Path to ligandmpnn_v_32_010_25.pt")
  parser.add_argument("--output", required=True, help="Output .pkl path")
  parser.add_argument("--source-commit", default=None, help="Optional LigandMPNN git commit used for conversion")
  args = parser.parse_args()
  convert_checkpoint(Path(args.checkpoint), Path(args.output), args.source_commit)


if __name__ == "__main__":
  main()
