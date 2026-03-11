"""Convert PDB files to Boltz2 YAML input format.

Replaces 01_prepare_get_json.py (AF3 JSON generation) with Boltz2 YAML generation.
Extracts chain sequences from PDB structures and generates Boltz2-compatible YAML files.
"""

import os
import argparse
import math
import multiprocessing as mp
from pathlib import Path

import yaml
import pandas as pd
from Bio import PDB
from Bio.PDB import PDBParser
from tqdm import tqdm

# Three-letter to one-letter amino acid code mapping
PROTEIN_LETTERS_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M",
}


def get_sequence_from_chain(chain):
    """Convert BioPython chain object to single-letter amino acid sequence."""
    sequence = ""
    for residue in chain:
        if residue.id[0] == " ":
            resname = residue.get_resname().upper()
            sequence += PROTEIN_LETTERS_3TO1.get(resname, "X")
    return sequence


def validate_pdb_file(pdb_path):
    """Validate PDB file integrity before processing.

    Checks:
    - File is not empty
    - Contains ATOM records
    - Has at least one model

    Returns:
        tuple: (is_valid, error_message)
    """
    if os.path.getsize(pdb_path) == 0:
        return False, "empty file"

    has_atom = False
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                has_atom = True
                break
    if not has_atom:
        return False, "no ATOM/HETATM records"

    return True, None


def validate_chain_sequences(chain_sequences, base_name):
    """Validate extracted chain sequences for integrity.

    Checks:
    - At least 2 chains for complex scoring (warn if single)
    - No duplicate sequences across chains (warn only)
    - Sequences are not all-X (unknown residues)
    - Minimum sequence length per chain

    Returns:
        dict: Validated chain_sequences (may have chains removed)
    """
    validated = {}
    for chain_id, seq in chain_sequences.items():
        if len(seq) < 5:
            print(f"Warning: {base_name} chain {chain_id} too short ({len(seq)} residues), skipping")
            continue
        if all(c == "X" for c in seq):
            print(f"Warning: {base_name} chain {chain_id} is all unknown residues, skipping")
            continue
        validated[chain_id] = seq

    if len(validated) == 1:
        print(f"Note: {base_name} has only 1 chain — inter-chain metrics will be empty")

    # Warn on duplicate sequences (homomers are valid but worth noting)
    seqs = list(validated.values())
    if len(seqs) != len(set(seqs)):
        print(f"Note: {base_name} has duplicate chain sequences (homomer)")

    return validated


def process_single_pdb(args):
    """Extract chain sequences from a single PDB file.

    Returns:
        tuple: (base_name, chain_sequences_dict, total_length) or (None, None, None) on error
    """
    input_pdb, = args
    try:
        # Validate PDB file integrity (#16)
        is_valid, err_msg = validate_pdb_file(input_pdb)
        if not is_valid:
            print(f"Warning: Skipping {input_pdb}: {err_msg}")
            return None, None, None

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("structure", input_pdb)
        base_name = os.path.splitext(os.path.basename(input_pdb))[0]

        chain_sequences = {}
        total_length = 0

        for chain in structure[0]:
            chain_id = chain.id
            sequence = get_sequence_from_chain(chain)
            if sequence:
                chain_sequences[chain_id] = sequence
                total_length += len(sequence)

        if not chain_sequences:
            print(f"Warning: No valid protein chains in {input_pdb}")
            return None, None, None

        # Validate chain sequences (#16)
        chain_sequences = validate_chain_sequences(chain_sequences, base_name)
        if not chain_sequences:
            print(f"Warning: No valid chains after validation in {input_pdb}")
            return None, None, None

        total_length = sum(len(s) for s in chain_sequences.values())
        return base_name, chain_sequences, total_length

    except Exception as e:
        print(f"Error processing {input_pdb}: {e}")
        return None, None, None


def generate_boltz_yaml(complex_name, chain_sequences, output_dir, template_pdb=None):
    """Generate a Boltz2 YAML input file from chain sequences.

    Args:
        complex_name: Name identifier for the complex
        chain_sequences: Dict mapping chain_id -> sequence
        output_dir: Directory to write YAML files
        template_pdb: Optional path to PDB file to use as template

    Returns:
        str: Output filename or None on failure
    """
    if not chain_sequences:
        print(f"Warning: No valid chain sequences for {complex_name}")
        return None

    sequences = []
    for chain_id, sequence in sorted(chain_sequences.items()):
        if not sequence or len(sequence) < 5:
            print(f"Warning: Skipping chain {chain_id} in {complex_name} (too short: {len(sequence)} residues)")
            continue

        # Validate sequence contains only standard amino acids
        valid_aas = set("ACDEFGHIKLMNPQRSTVWYX")
        cleaned_seq = "".join(c if c in valid_aas else "X" for c in sequence)

        protein_entry = {
            "protein": {
                "id": chain_id,
                "sequence": cleaned_seq,
            }
        }
        sequences.append(protein_entry)

    if not sequences:
        print(f"Warning: No valid sequences for {complex_name}")
        return None

    yaml_data = {
        "version": 1,
        "sequences": sequences,
    }

    # Add template if provided
    if template_pdb and os.path.exists(template_pdb):
        chain_ids = [list(s.values())[0]["id"] for s in sequences]
        yaml_data["templates"] = [
            {
                "pdb": str(template_pdb),
                "chain_id": chain_ids,
            }
        ]

    output_filename = f"{complex_name}.yaml"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    return output_filename


def split_by_total_length(df, num_jobs):
    """Split dataframe into balanced batches by total sequence length."""
    df = df.sort_values("total_length", ascending=False).reset_index(drop=True)
    total_sum = df["total_length"].sum()
    target_sum = total_sum / num_jobs

    groups = []
    current_group = []
    current_sum = 0

    for _, row in df.iterrows():
        val = int(row["total_length"])
        if current_sum + val > target_sum and len(groups) < num_jobs - 1:
            groups.append(pd.DataFrame(current_group))
            current_group = [row]
            current_sum = val
        else:
            current_group.append(row)
            current_sum += val

    if current_group:
        groups.append(pd.DataFrame(current_group))

    return groups


def run_prepare(input_dir, output_dir, save_csv="seq.csv", use_templates=False,
                num_workers=None, batch_dir=None, num_jobs=None):
    """Prepare Boltz2 YAML inputs from PDB files.

    Args:
        input_dir: Directory containing input PDB files
        output_dir: Directory to write YAML files
        save_csv: Path to save sequence metadata CSV
        use_templates: Use input PDBs as structural templates
        num_workers: Parallel workers (default: CPU-4)
        batch_dir: If set, create batch directories with symlinks
        num_jobs: Number of batch groups (required with batch_dir)
    """
    num_workers = num_workers or max(1, mp.cpu_count() - 4)
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: Extract sequences from PDB files
    pdb_files = sorted(Path(input_dir).glob("*.pdb"))
    if not pdb_files:
        print(f"No PDB files found in {input_dir}")
        return

    print(f"Found {len(pdb_files)} PDB files in {input_dir}")
    process_args = [(str(f),) for f in pdb_files]

    sequences_dict = {}
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_pdb, process_args),
            total=len(pdb_files),
            desc="Extracting sequences",
        ))

    for base_name, chain_seqs, length in results:
        if base_name is not None:
            sequences_dict[base_name] = {"sequences": chain_seqs, "length": length}

    # Build metadata DataFrame
    all_chain_ids = sorted({
        cid for entry in sequences_dict.values()
        for cid in entry["sequences"]
    })

    rows = []
    for complex_name, entry in sequences_dict.items():
        row = {"complex": complex_name, "total_length": entry["length"]}
        for chain_id in all_chain_ids:
            row[f"chain_{chain_id}_seq"] = entry["sequences"].get(chain_id, "")
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["complex", "total_length"] + [c for c in df.columns if c not in ["complex", "total_length"]]
    df = df[cols]
    df.to_csv(save_csv, index=False)
    print(f"Sequence info saved to {save_csv}")

    # Phase 2: Generate Boltz2 YAML files
    success_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating YAMLs"):
        complex_name = row["complex"]
        chain_seqs = {}
        for col in row.index:
            if col.startswith("chain_") and col.endswith("_seq") and pd.notna(row[col]) and row[col]:
                chain_id = col.split("_")[1]
                chain_seqs[chain_id] = row[col]

        template_pdb = None
        if use_templates:
            template_pdb = os.path.join(input_dir, f"{complex_name}.pdb")

        result = generate_boltz_yaml(complex_name, chain_seqs, output_dir, template_pdb)
        if result:
            success_count += 1

    print(f"YAML generation complete: {success_count}/{len(df)} files created")

    # Phase 3: Optional batch partitioning
    if batch_dir and num_jobs:
        batch_yaml_root = os.path.join(batch_dir, "yaml")
        batch_pdb_root = os.path.join(batch_dir, "pdb")

        subs = split_by_total_length(df.sample(frac=1).reset_index(drop=True), num_jobs)
        print(f"Splitting {len(df)} samples into {num_jobs} batches")

        for i, sub in enumerate(subs):
            if sub.empty:
                continue
            max_len = int(sub["total_length"].max())
            batch_name = f"batch_{i}_{max_len}"
            bd_yaml = os.path.join(batch_yaml_root, batch_name)
            bd_pdb = os.path.join(batch_pdb_root, batch_name)
            os.makedirs(bd_yaml, exist_ok=True)
            os.makedirs(bd_pdb, exist_ok=True)

            count = 0
            for _, r in sub.iterrows():
                cid = r["complex"]
                for ext, src_dir, dest_dir in [
                    (".pdb", input_dir, bd_pdb),
                    (".yaml", output_dir, bd_yaml),
                ]:
                    src = os.path.join(src_dir, f"{cid}{ext}")
                    if os.path.exists(src):
                        dest = os.path.join(dest_dir, os.path.basename(src))
                        if os.path.exists(dest):
                            os.remove(dest)
                        os.symlink(os.path.abspath(src), dest)
                count += 1
            print(f"  {batch_name}: {count} complexes")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDB files to Boltz2 YAML input format"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input PDB files")
    parser.add_argument("--output_dir", type=str, default="boltz_yaml",
                        help="Directory to write YAML files")
    parser.add_argument("--save_csv", type=str, default="seq.csv",
                        help="Path to save sequence metadata CSV")
    parser.add_argument("--use_templates", action="store_true",
                        help="Use input PDBs as templates for Boltz2")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU-4)")
    parser.add_argument("--batch_dir", type=str, default=None,
                        help="If set, create batch directories with symlinks")
    parser.add_argument("--num_jobs", type=int, default=None,
                        help="Number of batch groups (required with --batch_dir)")
    args = parser.parse_args()
    run_prepare(**vars(args))


if __name__ == "__main__":
    main()
