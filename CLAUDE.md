# Boltz2-Score

## Project Overview
Boltz2-Score is a protein complex scoring tool that uses Boltz2 (MIT-licensed) to predict structure confidence metrics. Migration from AF3Score (AlphaFold3, CC BY-NC-SA 4.0) to Boltz2 (PyTorch, open-source).

## Architecture
**Pipeline**: PDB input → Boltz2 YAML generation → Boltz2 inference → Metrics extraction → CSV output

### Key Components
- `prepare_boltz_input.py` — Converts PDB files to Boltz2 YAML format
- `run_boltz2score.py` — Orchestrates Boltz2 inference (programmatic or CLI)
- `extract_boltz2_metrics.py` — Parses Boltz2 JSON/NPZ outputs into per-chain metrics
- `ipsae_calculator.py` — Computes ipSAE from PAE matrix (framework-agnostic, reusable)
- `boltz2_score.py` — Unified CLI entry point

### Boltz2 Integration
- Install: `pip install boltz[cuda]`
- Input: YAML with protein sequences, optional MSA/templates
- Output: mmCIF/PDB structures + JSON confidence + NPZ arrays (PAE, pLDDT, PDE)
- Metrics: pTM, ipTM, pLDDT, PAE, PDE, affinity, confidence_score

## Tech Stack
- Python 3.10+
- PyTorch 2.2+ (via Boltz2)
- BioPython, NumPy, Pandas
- Multiprocessing for batch parallelism

## Conventions
- argparse for CLI arguments
- Multiprocessing pool for parallel PDB processing
- CSV output with per-chain and inter-chain columns
- Snake_case for files and functions
