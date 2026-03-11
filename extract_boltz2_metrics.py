"""Extract confidence metrics from Boltz2 prediction outputs.

Replaces 04_get_metrics.py. Parses Boltz2 JSON confidence files and NPZ arrays
to produce per-chain and inter-chain metrics, including ipSAE and PDE.
"""

import os
import json

import argparse
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm

from ipsae_calculator import calculate_ipsae, load_boltz2_pae_and_chains


def get_chains_from_pdb(pdb_path):
    """Extract sorted unique chain IDs from a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    chains = [chain.id for chain in structure[0].get_chains()]
    return sorted(set(chains))


def get_chain_residue_mapping(pdb_path):
    """Build per-token chain ID and residue type arrays from PDB.

    Returns:
        tuple: (token_chain_ids, token_res_ids, residue_types) as numpy arrays
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)

    token_chain_ids = []
    token_res_ids = []
    residue_types = []

    for chain in structure[0]:
        for residue in chain:
            if "CA" in residue:
                token_chain_ids.append(chain.id)
                token_res_ids.append(residue.id[1])
                residue_types.append(residue.get_resname().upper())

    return (
        np.array(token_chain_ids),
        np.array(token_res_ids),
        np.array(residue_types),
    )


def get_interface_residues(pdb_path, chain1, chain2, dist_cutoff=10.0):
    """Find interface residues between two chains based on CA-CA distance."""
    chain_coords = defaultdict(dict)

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                chain_id = line[21].strip()
                res_id = int(line[22:26])
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                chain_coords[chain_id][res_id] = np.array([x, y, z])

    res1 = sorted(chain_coords.get(chain1, {}).keys())
    res2 = sorted(chain_coords.get(chain2, {}).keys())

    if not res1 or not res2:
        return [], []

    coords1 = np.array([chain_coords[chain1][r] for r in res1])
    coords2 = np.array([chain_coords[chain2][r] for r in res2])

    dist = np.sqrt(np.sum((coords1[:, None, :] - coords2[None, :, :]) ** 2, axis=2))
    contacts = np.where(dist < dist_cutoff)

    iface1 = sorted({res1[i] for i in contacts[0]})
    iface2 = sorted({res2[i] for i in contacts[1]})
    return iface1, iface2


def _find_npz_file(pred_dir, prefix, stem):
    """Find NPZ file with fallback pattern matching."""
    primary = pred_dir / f"{prefix}_{stem}_model_0.npz"
    if primary.exists():
        return primary
    matches = list(pred_dir.glob(f"{prefix}_*.npz"))
    return matches[0] if matches else None


def load_boltz2_pae(pred_dir, stem):
    """Load PAE matrix from Boltz2 NPZ output."""
    path = _find_npz_file(pred_dir, "pae", stem)
    if path is None:
        return None
    return np.load(path)["pae"]


def load_boltz2_plddt(pred_dir, stem):
    """Load per-token pLDDT from Boltz2 NPZ output (values 0-1)."""
    path = _find_npz_file(pred_dir, "plddt", stem)
    if path is None:
        return None
    return np.load(path)["plddt"]


def load_boltz2_pde(pred_dir, stem):
    """Load PDE matrix from Boltz2 NPZ output."""
    path = _find_npz_file(pred_dir, "pde", stem)
    if path is None:
        return None
    return np.load(path)["pde"]


def load_boltz2_confidence(pred_dir, stem):
    """Load confidence JSON from Boltz2 output."""
    conf_path = pred_dir / f"confidence_{stem}_model_0.json"
    if not conf_path.exists():
        conf_files = list(pred_dir.glob("confidence_*.json"))
        if conf_files:
            conf_path = conf_files[0]
        else:
            return None

    with open(conf_path) as f:
        return json.load(f)


def validate_token_dimensions(pdb_path, pae_matrix, plddt_array, stem, validation_log=None):
    """Validate that PDB token count matches PAE/pLDDT dimensions.

    Raises ValueError on critical mismatch (>20% difference).
    Logs minor mismatches (5-20%) to validation_log file if provided.
    Returns trimmed dimensions for minor mismatches.
    """
    token_chain_ids, _, _ = get_chain_residue_mapping(pdb_path)
    n_pdb_tokens = len(token_chain_ids)

    issues = []
    if pae_matrix is not None:
        n_pae = pae_matrix.shape[0]
        if n_pdb_tokens != n_pae:
            pct_diff = abs(n_pdb_tokens - n_pae) / max(n_pdb_tokens, n_pae) * 100
            if pct_diff > 20:
                raise ValueError(
                    f"{stem}: PDB tokens ({n_pdb_tokens}) vs PAE dim ({n_pae}) "
                    f"differ by {pct_diff:.0f}% — likely mismatched files"
                )
            issues.append(f"PAE dim={n_pae} ({pct_diff:.1f}%)")

    if plddt_array is not None:
        n_plddt = len(plddt_array)
        if n_pdb_tokens != n_plddt:
            pct_diff = abs(n_pdb_tokens - n_plddt) / max(n_pdb_tokens, n_plddt) * 100
            if pct_diff > 20:
                raise ValueError(
                    f"{stem}: PDB tokens ({n_pdb_tokens}) vs pLDDT len ({n_plddt}) "
                    f"differ by {pct_diff:.0f}% — likely mismatched files"
                )
            issues.append(f"pLDDT len={n_plddt} ({pct_diff:.1f}%)")

    if issues:
        msg = f"{stem}: PDB tokens={n_pdb_tokens}, {', '.join(issues)}. Cropping to min."
        print(f"[Warning] {msg}")
        if validation_log is not None:
            validation_log.append(msg)

    return n_pdb_tokens


def compute_chain_pae(pae_matrix, token_chain_ids, chains):
    """Compute intra-chain PAE (average PAE within each chain)."""
    chain_pae = {}
    for ch in chains:
        idxs = np.where(token_chain_ids == ch)[0]
        if len(idxs) > 0:
            chain_pae[ch] = float(np.mean(pae_matrix[np.ix_(idxs, idxs)]))
    return chain_pae


def compute_chain_pde(pde_matrix, token_chain_ids, chains):
    """Compute intra-chain PDE (average PDE within each chain)."""
    chain_pde = {}
    for ch in chains:
        idxs = np.where(token_chain_ids == ch)[0]
        if len(idxs) > 0:
            chain_pde[ch] = float(np.mean(pde_matrix[np.ix_(idxs, idxs)]))
    return chain_pde


def compute_interface_pae(pae_matrix, token_chain_ids, token_res_ids, pdb_path, chains):
    """Compute interface PAE and inter-chain PAE for all chain pairs."""
    ipae = {}
    inter_pae = {}

    for ch1, ch2 in combinations(chains, 2):
        try:
            iface1, iface2 = get_interface_residues(pdb_path, ch1, ch2)
            idx1 = [i for i, (r, c) in enumerate(zip(token_res_ids, token_chain_ids))
                    if c == ch1 and r in iface1]
            idx2 = [i for i, (r, c) in enumerate(zip(token_res_ids, token_chain_ids))
                    if c == ch2 and r in iface2]

            pair_key = f"{ch1}_{ch2}"
            if idx1 and idx2:
                ipae[pair_key] = float(np.mean([
                    np.mean(pae_matrix[np.ix_(idx1, idx2)]),
                    np.mean(pae_matrix[np.ix_(idx2, idx1)]),
                ]))

            c1_idx = np.where(token_chain_ids == ch1)[0]
            c2_idx = np.where(token_chain_ids == ch2)[0]
            if len(c1_idx) > 0 and len(c2_idx) > 0:
                inter_pae[pair_key] = float(np.mean([
                    np.mean(pae_matrix[np.ix_(c1_idx, c2_idx)]),
                    np.mean(pae_matrix[np.ix_(c2_idx, c1_idx)]),
                ]))
        except Exception as e:
            print(f"Warning: Failed pair ({ch1}, {ch2}): {e}")

    return ipae, inter_pae


def compute_interface_pde(pde_matrix, token_chain_ids, chains):
    """Compute inter-chain PDE for all chain pairs."""
    inter_pde = {}
    for ch1, ch2 in combinations(chains, 2):
        c1_idx = np.where(token_chain_ids == ch1)[0]
        c2_idx = np.where(token_chain_ids == ch2)[0]
        if len(c1_idx) > 0 and len(c2_idx) > 0:
            pair_key = f"{ch1}_{ch2}"
            inter_pde[pair_key] = float(np.mean([
                np.mean(pde_matrix[np.ix_(c1_idx, c2_idx)]),
                np.mean(pde_matrix[np.ix_(c2_idx, c1_idx)]),
            ]))
    return inter_pde


def map_boltz_chains_to_pdb(confidence_data, pdb_chains):
    """Map Boltz2 0-indexed chain keys to PDB chain letter IDs."""
    return {str(i): chain_id for i, chain_id in enumerate(pdb_chains)}


def process_single_prediction(args):
    """Process metrics for a single Boltz2 prediction."""
    stem, pred_dir, input_pdb_dir = args
    pred_dir = Path(pred_dir)
    validation_warnings = []

    try:
        pdb_path = Path(input_pdb_dir) / f"{stem}.pdb"
        if not pdb_path.exists():
            return None, f"{stem}: missing PDB file at {pdb_path}", []

        # Load Boltz2 outputs
        confidence = load_boltz2_confidence(pred_dir, stem)
        if confidence is None:
            return None, f"{stem}: missing confidence JSON in {pred_dir}", []

        pae_matrix = load_boltz2_pae(pred_dir, stem)
        plddt_array = load_boltz2_plddt(pred_dir, stem)

        # PDE loading with graceful fallback (C3)
        pde_matrix = None
        try:
            pde_matrix = load_boltz2_pde(pred_dir, stem)
        except Exception as e:
            print(f"[Warning] {stem}: PDE load failed ({e}), skipping PDE metrics")

        # Strict token validation (#20), with log collection (C1)
        validate_token_dimensions(str(pdb_path), pae_matrix, plddt_array, stem,
                                  validation_log=validation_warnings)

        # Get chain info from PDB
        chains = get_chains_from_pdb(str(pdb_path))
        token_chain_ids, token_res_ids, residue_types = get_chain_residue_mapping(str(pdb_path))
        chain_map = map_boltz_chains_to_pdb(confidence, chains)

        # Base metrics from confidence JSON
        result = {
            "description": stem,
            "ptm": confidence.get("ptm", 0.0),
            "iptm": confidence.get("iptm", 0.0),
            "confidence_score": confidence.get("confidence_score", 0.0),
            "protein_iptm": confidence.get("protein_iptm", 0.0),
            "ligand_iptm": confidence.get("ligand_iptm", 0.0),
            "complex_plddt": confidence.get("complex_plddt", 0.0),
            "complex_iplddt": confidence.get("complex_iplddt", 0.0),
            "complex_pde": confidence.get("complex_pde", 0.0),
            "complex_ipde": confidence.get("complex_ipde", 0.0),
        }

        # Per-chain pTM from Boltz2 JSON (0-indexed)
        chains_ptm = confidence.get("chains_ptm", {})
        for boltz_idx, chain_letter in chain_map.items():
            result[f"chain_{chain_letter}_ptm"] = chains_ptm.get(boltz_idx, np.nan)

        # Per-chain ipTM from pair_chains_iptm
        pair_iptm = confidence.get("pair_chains_iptm", {})
        for boltz_idx, chain_letter in chain_map.items():
            chain_iptm_values = []
            for other_idx in pair_iptm.get(boltz_idx, {}):
                if other_idx != boltz_idx:
                    chain_iptm_values.append(pair_iptm[boltz_idx][other_idx])
            result[f"chain_{chain_letter}_iptm"] = (
                float(np.mean(chain_iptm_values)) if chain_iptm_values else np.nan
            )

        # Inter-chain pair ipTM
        for i, ch1 in enumerate(chains):
            for j, ch2 in enumerate(chains):
                if i < j:
                    val = pair_iptm.get(str(i), {}).get(str(j), np.nan)
                    result[f"iptm_{ch1}_{ch2}"] = val

        # Per-token pLDDT → per-chain pLDDT
        if plddt_array is not None and len(token_chain_ids) > 0:
            n_tokens = min(len(plddt_array), len(token_chain_ids))
            for ch in chains:
                ch_mask = token_chain_ids[:n_tokens] == ch
                if ch_mask.any():
                    # Boltz2 pLDDT is 0-1, convert to 0-100 for compatibility
                    result[f"chain_{ch}_plddt"] = float(np.mean(plddt_array[:n_tokens][ch_mask]) * 100)
                else:
                    result[f"chain_{ch}_plddt"] = np.nan

        # PAE-based metrics
        if pae_matrix is not None and len(token_chain_ids) > 0:
            n_tokens = min(pae_matrix.shape[0], len(token_chain_ids))
            pae_trimmed = pae_matrix[:n_tokens, :n_tokens]
            tc_trimmed = token_chain_ids[:n_tokens]
            tr_trimmed = token_res_ids[:n_tokens]
            rt_trimmed = residue_types[:n_tokens]

            # Intra-chain PAE
            chain_pae = compute_chain_pae(pae_trimmed, tc_trimmed, chains)
            for ch in chains:
                result[f"chain_{ch}_pae"] = chain_pae.get(ch, np.nan)

            # Interface & inter-chain PAE
            ipae, inter_pae = compute_interface_pae(
                pae_trimmed, tc_trimmed, tr_trimmed, str(pdb_path), chains
            )
            for key, val in ipae.items():
                result[f"ipae_{key}"] = val
            for key, val in inter_pae.items():
                result[f"inter_pae_{key}"] = val

            # ipSAE calculation
            ipsae_scores = calculate_ipsae(pae_trimmed, tc_trimmed, rt_trimmed, pae_cutoff=10)
            for key, val in ipsae_scores.items():
                result[f"ipsae_{key}"] = val

        # PDE-based metrics (#12)
        if pde_matrix is not None and len(token_chain_ids) > 0:
            n_tokens = min(pde_matrix.shape[0], len(token_chain_ids))
            pde_trimmed = pde_matrix[:n_tokens, :n_tokens]
            tc_trimmed = token_chain_ids[:n_tokens]

            # Intra-chain PDE
            chain_pde = compute_chain_pde(pde_trimmed, tc_trimmed, chains)
            for ch in chains:
                result[f"chain_{ch}_pde"] = chain_pde.get(ch, np.nan)

            # Inter-chain PDE
            inter_pde = compute_interface_pde(pde_trimmed, tc_trimmed, chains)
            for key, val in inter_pde.items():
                result[f"inter_pde_{key}"] = val

        return result, None, validation_warnings

    except ValueError as e:
        # Strict validation errors bubble up
        return None, str(e), validation_warnings
    except Exception as e:
        return None, f"{stem}: {str(e)}", validation_warnings


def find_boltz2_predictions(output_dir):
    """Find all Boltz2 prediction directories.

    Searches multiple directory patterns for robustness (#21):
    1. out_dir/boltz_results_*/predictions/*/
    2. out_dir/predictions/*/
    3. out_dir/*/ (if contains confidence JSON files)
    """
    results = []
    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        print("  Expected Boltz2 output structure: out_dir/boltz_results_*/predictions/*/")
        return results

    # Pattern 1: Standard Boltz2 output
    for results_dir in sorted(output_path.glob("boltz_results_*")):
        predictions_dir = results_dir / "predictions"
        if predictions_dir.is_dir():
            for pred_dir in sorted(predictions_dir.iterdir()):
                if pred_dir.is_dir():
                    results.append((pred_dir.name, str(pred_dir)))

    # Pattern 2: Direct predictions/ subdirectory
    if not results:
        predictions_dir = output_path / "predictions"
        if predictions_dir.is_dir():
            for pred_dir in sorted(predictions_dir.iterdir()):
                if pred_dir.is_dir():
                    results.append((pred_dir.name, str(pred_dir)))

    # Pattern 3: Subdirs containing confidence JSON
    if not results:
        for subdir in sorted(output_path.iterdir()):
            if subdir.is_dir() and list(subdir.glob("confidence_*.json")):
                results.append((subdir.name, str(subdir)))

    if not results:
        print(f"Error: No Boltz2 predictions found in {output_dir}")
        print("  Searched patterns:")
        print(f"    {output_dir}/boltz_results_*/predictions/*/")
        print(f"    {output_dir}/predictions/*/")
        print(f"    {output_dir}/*/ (with confidence_*.json)")
        print("  Run `boltz predict` first, or check --boltz_output_dir path.")

    return results


def extract_all_metrics(input_pdb_dir, boltz_output_dir, num_workers=None):
    """Extract metrics from all Boltz2 predictions in parallel."""
    pred_dirs = find_boltz2_predictions(boltz_output_dir)

    if not pred_dirs:
        return pd.DataFrame(), []

    args_list = [(stem, pred_dir, input_pdb_dir) for stem, pred_dir in pred_dirs]

    results = []
    failed = []
    all_validation_warnings = []
    n_proc = num_workers or max(1, mp.cpu_count() - 1)

    with mp.Pool(processes=n_proc) as pool:
        for res, err, warnings in tqdm(
            pool.imap_unordered(process_single_prediction, args_list),
            total=len(args_list),
            desc="Extracting Boltz2 metrics",
        ):
            if err:
                failed.append(err)
            elif res:
                results.append(res)
            all_validation_warnings.extend(warnings)

    # Write validation report if there were dimension mismatches (C1)
    if all_validation_warnings:
        report_path = Path(boltz_output_dir) / "validation_report.txt"
        with open(report_path, "w") as f:
            f.write("# Dimension mismatch warnings (PDB vs NPZ)\n")
            for w in all_validation_warnings:
                f.write(f"{w}\n")
        print(f"Validation report: {report_path} ({len(all_validation_warnings)} warnings)")

    return pd.DataFrame(results), failed


def main():
    parser = argparse.ArgumentParser(
        description="Extract metrics from Boltz2 prediction outputs"
    )
    parser.add_argument("--input_pdb_dir", type=str, required=True,
                        help="Directory containing original input PDB files")
    parser.add_argument("--boltz_output_dir", type=str, required=True,
                        help="Boltz2 output directory (containing boltz_results_*/)")
    parser.add_argument("--save_csv", type=str, default="boltz2_metrics.csv",
                        help="Path to save extracted metrics CSV")
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    df, failed = extract_all_metrics(
        args.input_pdb_dir, args.boltz_output_dir, args.num_workers
    )

    if not df.empty:
        df.to_csv(args.save_csv, index=False)
        print(f"Successfully processed {len(df)} predictions, saved to {args.save_csv}")
    else:
        print("No results extracted.")

    if failed:
        print(f"Failed: {len(failed)} predictions")
        failed_log = Path(args.boltz_output_dir) / "failed_records.txt"
        with open(failed_log, "w") as f:
            f.write("\n".join(failed))
        print(f"Failure log: {failed_log}")


if __name__ == "__main__":
    main()
