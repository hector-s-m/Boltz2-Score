# Boltz2 Research Summary

## 1. YAML Input Format Specification

### Full YAML Schema
The complete schema from the docs:

```yaml
version: 1  # Optional, defaults to 1

sequences:
  - protein:
      id: A  # or [A, B] for multiple identical chains
      sequence: "MKVL..."  # Protein sequence
      msa: ./path/to/msa.a3m  # MSA file (required unless --use_msa_server)
      modifications:
        - position: 1  # 1-indexed residue position
          ccd: "CCD_CODE"  # Modified residue CCD code
      cyclic: false  # Whether chain is cyclic
  
  - dna:
      id: B
      sequence: "ATCG..."
      msa: "./path.a3m"  # Optional for DNA
  
  - rna:
      id: C
      sequence: "AUGC..."
  
  - ligand:
      id: [D, E]  # Multiple copies
      smiles: "N[C@@H](Cc1ccc(O)cc1)C(=O)O"  # SMILES string
      # OR
      # ccd: "SAH"  # CCD code (mutually exclusive with smiles)

constraints:  # Optional
  - bond:
      atom1: [A, 1, CA]  # [CHAIN_ID, RES_IDX (1-indexed), ATOM_NAME]
      atom2: [A, 2, N]
  
  - pocket:
      binder: D  # Chain that binds to pocket
      contacts: [[A, 10], [A, 20]]  # Residues forming pocket
      max_distance: 6  # Angstroms (4-20A range, default 6)
      force: false  # Apply potential to enforce
  
  - contact:
      token1: [A, 1]  # Residue position
      token2: [D, 1]  # Atom name for ligands
      max_distance: 8
      force: false

templates:  # Optional
  - cif: "./path/to/template.cif"
    chain_id: [A]  # Optional, specify which chains to template
    force: false
    threshold: 1.5  # Max deviation in Angstroms (if force=true)
  
  - pdb: "./path/to/template.pdb"
    chain_id: [A, B]
    template_id: [A1, A2]  # Explicit chain mapping

properties:  # Optional
  - affinity:
      binder: D  # Ligand chain for affinity prediction
```

### Example YAML Files from Repository

**Single Protein (prot.yaml):**
```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG
```

**Multimer with Ligands (multimer.yaml):**
```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MAHHHHHHVAVDAVSFTLLQDQLQSVLDTLSEREAGVVRLRFGLTDGQPRTLDEIGQVYGVTRERIRQIESKTMSKLRHPSRSQVLRDYLDGSSGSGTPEERLLRAIFGEKA
      msa: ./examples/msa/seq1.a3m
  - protein:
      id: B
      sequence: MRYAFAAEATTCNAFWRNVDMTVTALYEVPLGVCTQDPDRWTTTPDDEAKTLCRACPRRWLCARDAVESAGAEGLWAGVVIPESGRARAFALGQLRSLAERNGYPVRDHRVSAQSA
      msa: ./examples/msa/seq1.a3m
  - ligand:
      id: [C, D]
      ccd: SAH
  - ligand:
      id: [E, F]
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
```

**Affinity Prediction (affinity.yaml):**
```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVGVTEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ
  - ligand:
      id: B
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
properties:
  - affinity:
      binder: B
```

---

## 2. Running Boltz2 Programmatically from Python

### Method 1: CLI Invocation
The `predict` function is decorated with `@click` and exposed as a CLI command, but can be called directly:

```python
from boltz.main import predict
from pathlib import Path

# Direct function call (bypassing click decorator)
predict(
    data="/path/to/input.yaml",
    out_dir="./output",
    cache="~/.boltz",
    checkpoint=None,  # Uses default model
    devices=1,
    accelerator="gpu",  # or "cpu", "tpu"
    recycling_steps=3,
    sampling_steps=200,
    diffusion_samples=1,
    max_parallel_samples=5,
    step_scale=None,  # 1.5 for Boltz2, 1.638 for Boltz1
    write_full_pae=False,
    write_full_pde=False,
    output_format="mmcif",  # or "pdb"
    num_workers=2,
    override=False,
    seed=None,
    use_msa_server=True,  # Auto-generate MSA via mmseqs2
    msa_server_url="https://api.colabfold.com",
    msa_pairing_strategy="greedy",  # or "complete"
    model="boltz2",  # or "boltz1"
    use_potentials=False,
    affinity_mw_correction=False,
    preprocessing_threads=4,
    max_msa_seqs=8192,
    subsample_msa=True,
    num_subsampled_msa=1024,
    no_kernels=False,
    write_embeddings=False,
)
```

### Method 2: Build Input YAML Programmatically

```python
import yaml
from pathlib import Path

def create_boltz_input_yaml(
    protein_sequence: str,
    chain_id: str = "A",
    ligand_smiles: str = None,
    ligand_id: str = "B",
    msa_path: str = None,
    use_msa_server: bool = True,
    output_path: str = "input.yaml"
) -> Path:
    """Generate Boltz2 YAML input from protein sequence and optional ligand."""
    
    config = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": chain_id,
                    "sequence": protein_sequence,
                }
            }
        ]
    }
    
    # Add MSA if provided or using server
    if msa_path:
        config["sequences"][0]["protein"]["msa"] = msa_path
    elif not use_msa_server:
        # For single-sequence mode (not recommended)
        config["sequences"][0]["protein"]["msa"] = "empty"
    
    # Add ligand if provided
    if ligand_smiles:
        config["sequences"].append({
            "ligand": {
                "id": ligand_id,
                "smiles": ligand_smiles
            }
        })
        
        # Add affinity property
        config["properties"] = [
            {
                "affinity": {
                    "binder": ligand_id
                }
            }
        ]
    
    # Write YAML
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return output_path
```

### Method 3: Parse Existing PDB/CIF Structures

```python
from boltz.data.parse.pdb import parse_pdb
from boltz.data.parse.mmcif import parse_mmcif
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

def extract_sequence_from_pdb(pdb_path: str, chain_id: str = "A") -> str:
    """Extract protein sequence from PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_path)
    
    ppb = PPBuilder()
    sequences = ppb.build_peptides(structure[0][chain_id])
    
    if sequences:
        return str(sequences[0].get_sequence())
    return ""

def create_boltz_yaml_from_pdb(
    pdb_path: str,
    output_yaml: str,
    use_msa_server: bool = True
) -> Path:
    """Create Boltz YAML from existing PDB structure."""
    from Bio.PDB import PDBParser, PPBuilder
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_path)
    ppb = PPBuilder()
    
    sequences_config = []
    for chain in structure[0]:
        peptides = ppb.build_peptides(chain)
        if peptides:
            seq = str(peptides[0].get_sequence())
            sequences_config.append({
                "protein": {
                    "id": chain.id,
                    "sequence": seq,
                    "msa": "empty" if not use_msa_server else None
                }
            })
    
    config = {
        "version": 1,
        "sequences": sequences_config
    }
    
    # Remove 'msa: empty' if using server
    if use_msa_server:
        for seq in config["sequences"]:
            if "protein" in seq and seq["protein"].get("msa") is None:
                del seq["protein"]["msa"]
    
    with open(output_yaml, "w") as f:
        yaml.dump(config, f)
    
    return Path(output_yaml)
```

---

## 3. Boltz2 Output Format Specification

### Output Directory Structure

```
out_dir/
├── predictions/
│   ├── [input_stem]/                      # Folder for each input file
│   │   ├── [input_stem]_model_0.cif       # Predicted structure (top-ranked)
│   │   ├── [input_stem]_model_1.cif       # Alternative model
│   │   ├── confidence_[input_stem]_model_0.json
│   │   ├── pae_[input_stem]_model_0.npz
│   │   ├── pde_[input_stem]_model_0.npz
│   │   ├── plddt_[input_stem]_model_0.npz
│   │   ├── affinity_[input_stem].json     # If affinity predicted
│   │   └── embeddings_[input_stem].npz    # If --write_embeddings
│   └── [input_stem2]/
│       └── ...
└── processed/
    ├── manifest.json
    ├── structures/
    ├── msa/
    ├── constraints/   # If constraints provided
    └── templates/     # If templates provided
```

### Confidence JSON Schema

**File:** `confidence_[input_stem]_model_0.json`

```json
{
    "confidence_score": 0.8367,
    "ptm": 0.8425,
    "iptm": 0.8225,
    "ligand_iptm": 0.0,
    "protein_iptm": 0.8225,
    "complex_plddt": 0.8402,
    "complex_iplddt": 0.8241,
    "complex_pde": 0.8912,
    "complex_ipde": 5.1650,
    "chains_ptm": {
        "0": 0.8533,
        "1": 0.8330
    },
    "pair_chains_iptm": {
        "0": {
            "0": 0.8533,
            "1": 0.8090
        },
        "1": {
            "0": 0.8225,
            "1": 0.8330
        }
    }
}
```

**Key Definitions:**
- `confidence_score`: Aggregated score (0.8 * complex_plddt + 0.2 * iptm/ptm)
- `ptm`: Predicted TM score for the complex (single chain only)
- `iptm`: Predicted interface TM score (multi-chain)
- `ligand_iptm`: Interface TM for protein-ligand interactions
- `protein_iptm`: Interface TM for protein-protein interactions
- `complex_plddt`: Average per-residue confidence (0-1)
- `complex_iplddt`: Average interface confidence
- `complex_pde`: Average predicted distance error (in Angstroms, lower is better)
- `complex_ipde`: Average interface distance error

### Affinity JSON Schema

**File:** `affinity_[input_stem].json`

```json
{
    "affinity_pred_value": 0.8367,
    "affinity_probability_binary": 0.8425,
    "affinity_pred_value1": 0.8225,
    "affinity_probability_binary1": 0.0,
    "affinity_pred_value2": 0.8225,
    "affinity_probability_binary2": 0.8402
}
```

**Key Definitions:**
- `affinity_pred_value`: Ensemble binding affinity prediction (log10(IC50) in μM)
  - -3 = strong binder (IC50 = 10^-9 M)
  - 0 = moderate binder (IC50 = 10^-6 M)
  - 2 = weak binder (IC50 = 10^-4 M)
- `affinity_probability_binary`: Probability ligand is a binder (0-1, use for screening)
- `*1`, `*2`: Individual model outputs from ensemble

**Conversion to pIC50 (kcal/mol):** `(6 - affinity_pred_value) * 1.364`

### NPZ File Schemas

All NPZ files load via `np.load(path)` and contain:

**pLDDT file:** `plddt_[input_stem]_model_0.npz`
```python
data = np.load("plddt_input_model_0.npz")
plddt = data["plddt"]  # Shape: (num_tokens,), dtype: float32
# Values 0-1, per-residue/token confidence
```

**PAE file:** `pae_[input_stem]_model_0.npz`
```python
data = np.load("pae_input_model_0.npz")
pae = data["pae"]  # Shape: (num_tokens, num_tokens), dtype: float32
# PAE in Angstroms, symmetric matrix
# pae[i,j] = predicted distance error between residues i and j
```

**PDE file:** `pde_[input_stem]_model_0.npz`
```python
data = np.load("pde_input_model_0.npz")
pde = data["pde"]  # Shape: (num_tokens, num_tokens), dtype: float32
# Predicted distance error, symmetric matrix
```

**Embeddings file:** `embeddings_[input_stem].npz` (if `--write_embeddings`)
```python
data = np.load("embeddings_input.npz")
s = data["s"]  # Token embeddings, shape: (seq_len, hidden_dim)
z = data["z"]  # Pairwise embeddings, shape: (seq_len, seq_len, hidden_dim)
```

---

## 4. Code Example: Complete Workflow

```python
import json
import yaml
import numpy as np
from pathlib import Path
from boltz.main import predict

# Step 1: Create YAML input
input_yaml = {
    "version": 1,
    "sequences": [
        {
            "protein": {
                "id": "A",
                "sequence": "MVHLTPEEKS..."
            }
        },
        {
            "ligand": {
                "id": "B",
                "smiles": "CC(=O)Nc1ccc(O)cc1"
            }
        }
    ],
    "properties": [
        {
            "affinity": {
                "binder": "B"
            }
        }
    ]
}

yaml_path = Path("input.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(input_yaml, f)

# Step 2: Run prediction
output_dir = Path("./results")
predict(
    data=str(yaml_path),
    out_dir=str(output_dir),
    use_msa_server=True,
    model="boltz2",
    recycling_steps=3,
    diffusion_samples=1,
    write_full_pae=True,
    write_embeddings=False,
)

# Step 3: Parse results
pred_dir = output_dir / "boltz_results_input" / "predictions" / "input"

# Load confidence scores
with open(pred_dir / "confidence_input_model_0.json") as f:
    confidence = json.load(f)
    print(f"pLDDT: {confidence['complex_plddt']:.3f}")
    print(f"iPTM: {confidence['iptm']:.3f}")

# Load affinity
with open(pred_dir / "affinity_input.json") as f:
    affinity = json.load(f)
    ic50_um = 10 ** affinity["affinity_pred_value"]
    print(f"Predicted IC50: {ic50_um:.2e} μM")
    print(f"Binder probability: {affinity['affinity_probability_binary']:.3f}")

# Load pLDDT array
plddt_data = np.load(pred_dir / "plddt_input_model_0.npz")
plddt = plddt_data["plddt"]
print(f"pLDDT shape: {plddt.shape}")

# Load PAE matrix
pae_data = np.load(pred_dir / "pae_input_model_0.npz")
pae = pae_data["pae"]
print(f"PAE shape: {pae.shape}, min: {pae.min():.2f}, max: {pae.max():.2f}")

# Read structure
structure_path = pred_dir / "input_model_0.cif"
print(f"Structure saved to: {structure_path}")
```

---

## 5. Key Implementation Details

### YAML Parsing (Source: `/tmp/boltz/src/boltz/data/parse/yaml.py`)
```python
from boltz.data.parse.yaml import parse_yaml

# Used internally by Boltz
# parse_yaml(path: Path, ccd: dict, mol_dir: Path, boltz2: bool) -> Target
```

### CLI Entry Point (Source: `/tmp/boltz/src/boltz/main.py`, line 817)
The `predict` function uses `@click` decorators but is a regular Python function that can be imported and called directly.

### Data Types
- Sequences section uses `entity_type` keys: `protein`, `dna`, `rna`, `ligand`
- Chain IDs can be scalar (`id: A`) or list (`id: [A, B]`) for multiple copies
- Residue/atom indices are 1-indexed (starting from 1, not 0)
- MSA paths are relative to the YAML file location or absolute paths

### MSA Handling
- If `msa` field omitted and NOT using `--use_msa_server`: ERROR
- If `msa: empty`: Forces single-sequence mode (not recommended)
- If `msa: path/to/file.a3m`: Uses provided MSA file
- If `msa: path/to/file.csv`: Custom format with columns `sequence` and `key`

### Output Ranking
- Models are ranked by `confidence_score`
- Filenames include rank index: `model_0` = highest confidence, `model_1` = next, etc.
- Number of models controlled by `--diffusion_samples`

### Affinity Notes
- Only ligand chains (not protein/DNA/RNA) can be binders
- Ligand must have ≤128 heavy atoms (recommended ≤56)
- Output is log10(IC50) in μM

