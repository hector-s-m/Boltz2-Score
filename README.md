# Boltz2-Score

Protein complex scoring pipeline using [Boltz2](https://github.com/jwohlwend/boltz) (MIT license). Drop-in replacement for AF3Score, using Boltz2's PyTorch engine instead of AlphaFold3.

## Installation

```bash
conda create -n boltz2score python=3.11
conda activate boltz2score
pip install -r requirements.txt
```

Requirements: `boltz[cuda]>=0.5.0`, `biopython`, `numpy`, `pandas`, `pyyaml`, `tqdm`.

## Quick Start

```bash
# Full pipeline: PDB files → Boltz2 prediction → metrics CSV
python boltz2_score.py --input_dir input_pdbs/ --output_dir results/
```

Output: `results/boltz2_metrics.csv`

## Pipeline Steps

The pipeline has three stages, each runnable independently:

### 1. Prepare Boltz2 inputs (PDB → YAML)

```bash
python prepare_boltz_input.py --input_dir input_pdbs/ --output_dir boltz_yaml/
```

Extracts chain sequences from PDB files and generates Boltz2-compatible YAML files. Validates PDB integrity and chain sequences before conversion.

### 2. Run Boltz2 inference

```bash
python run_boltz2score.py --input boltz_yaml/ --output_dir boltz_predictions/
```

Runs `boltz predict` via subprocess. Generates structure predictions with confidence metrics (PAE, pLDDT, PDE as NPZ files + confidence JSON).

### 3. Extract metrics

```bash
python extract_boltz2_metrics.py --input_pdb_dir input_pdbs/ --boltz_output_dir boltz_predictions/
```

Parses Boltz2 outputs into a CSV with per-chain and inter-chain metrics.

## Unified CLI Options

```bash
python boltz2_score.py --input_dir input_pdbs/ --output_dir results/ \
    --use_templates \          # Use input PDBs as structural templates
    --use_potentials \         # Enable physical potentials
    --no_msa_server \          # Disable MSA server (use pre-computed MSAs)
    --diffusion_samples 5 \    # Number of structure samples per input
    --recycling_steps 3 \      # Recycling iterations
    --sampling_steps 200 \     # Diffusion sampling steps
    --devices 1 \              # Number of GPU devices
    --accelerator gpu \        # gpu, cpu, or tpu
    --num_workers 4            # Parallel workers for data processing
```

Skip individual stages with `--skip_prepare`, `--skip_inference`, or `--skip_metrics`.

### Batch mode

```bash
# Step 1: Prepare with batch partitioning
python prepare_boltz_input.py --input_dir pdbs/ --output_dir yaml/ \
    --batch_dir batches/ --num_jobs 4

# Step 2: Run with batch directory
python boltz2_score.py --input_dir pdbs/ --output_dir results/ \
    --skip_prepare --batch_dir batches/
```

## Output Metrics

| Metric | Level | Description |
|--------|-------|-------------|
| **pTM** | Global / Per-chain | Predicted TM-score. Overall topological accuracy. |
| **ipTM** | Global / Inter-chain | Interface pTM. Accuracy of chain-chain interfaces. |
| **pLDDT** | Per-chain | Predicted Local Distance Difference Test (0-100). Per-residue confidence. |
| **PAE** | Per-chain / Inter-chain | Predicted Aligned Error (Angstroms). Lower = higher confidence. |
| **PDE** | Per-chain / Inter-chain | Predicted Distance Error. Boltz2-specific distance confidence. |
| **ipSAE** | Inter-chain | Interaction prediction Score from Aligned Errors. Interface binding quality. |
| **confidence_score** | Global | Boltz2 composite confidence score. |

## ipSAE Calculator (standalone)

```bash
python ipsae_calculator.py --pdb structure.pdb --pae pae_model_0.npz --format boltz2
```

Supports both Boltz2 NPZ and AF3 JSON PAE formats.

## Reference

- Boltz2: [github.com/jwohlwend/boltz](https://github.com/jwohlwend/boltz)
- AF3Score (original): [github.com/Mingchenchen/AF3Score](https://github.com/Mingchenchen/AF3Score)
