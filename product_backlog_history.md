# Product Backlog History — Boltz2-Score

## Completed Items

### Sprint 1 (2026-03-10) — Core Migration
- [x] `prepare_boltz_input.py` — PDB → Boltz2 YAML converter
- [x] `run_boltz2score.py` — Boltz2 inference wrapper
- [x] `extract_boltz2_metrics.py` — Metrics extraction from Boltz2 outputs
- [x] `ipsae_calculator.py` — Added Boltz2 NPZ loader (`load_boltz2_pae_and_chains`)
- [x] `boltz2_score.py` — Unified end-to-end CLI
- [x] `requirements.txt` — Updated for Boltz2 dependencies

### Sprint 2 (2026-03-10) — Quality & Batch Processing
- [x] **#7** Removed AF3 source tree and legacy files
- [x] **#19** Consolidated PAE loading helpers
- [x] **#20** Strict token validation (>20% mismatch = ValueError)
- [x] **#21** Fallback path handling (3-pattern search)
- [x] **#12** PDE metrics added to CSV
- [x] **#16** Input validation (PDB file + chain sequence checks)
- [x] **#9** Batch processing (run_boltz_batch, --batch_dir, multi-batch aggregation)
- [x] Drop-in parity audit confirmed all AF3Score columns present

## Dropped Items

### Sprint 1
- **Boltz2 model manager** — DROPPED: Not needed. Boltz2 handles model loading internally via its CLI/API. No singleton pattern required.

### Brainstormer S1
- *(No ideas dropped — all 3 accepted to backlog)*

### Brainstormer S2
- *(All ideas overlapped existing backlog — boldness increased to 20%)*

---
*Last updated: 2026-03-10 Sprint 2 Complete*
