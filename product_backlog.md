# Product Backlog — Boltz2-Score

## Sprint 1 — COMPLETED (2026-03-10)

### P0 — Core Migration (Must-Have)
1. [x] **Create `prepare_boltz_input.py`** — PDB → Boltz2 YAML. Chain extraction, sequence validation, batch partitioning. [core]
2. [x] **Create `run_boltz2score.py`** — Boltz2 inference wrapper via subprocess CLI. [core]
3. [x] **Create `extract_boltz2_metrics.py`** — Parse Boltz2 JSON + NPZ. Per-chain pLDDT/PAE/pTM/ipTM + ipSAE. [core]
4. [x] **Adapt `ipsae_calculator.py`** — Added `load_boltz2_pae_and_chains()` for NPZ support. [core]
5. [x] **Create unified `boltz2_score.py` CLI** — End-to-end pipeline with --skip flags. [core]

### P1 — Infrastructure
6. [x] **Update `requirements.txt`** — Boltz2[cuda], BioPython, NumPy, Pandas, PyYAML, tqdm. [core]
8. [ ] **Add Boltz2 model manager** — Not needed: Boltz2 handles model loading internally. [DROPPED]

---

## Sprint 2 — COMPLETED (2026-03-10)

### P0 — Core Quality & Cleanup
7. [x] **Remove AF3 source tree** — Deleted `src/alphafold3/` and legacy AF3 files. [core]
19. [x] **Consolidate PAE loading** — Use `load_boltz2_pae_and_chains` from ipsae_calculator in extract_boltz2_metrics. [core]
20. [x] **Strict token validation** — Fail loudly on PDB/PAE dimension mismatch (>20% = ValueError). [core]
21. [x] **Fallback path handling** — 3-pattern search for boltz_results_* dirs with better error messages. [core]

### P1 — Feature Enhancements
12. [x] **PDE metrics** — Added PDE (Predicted Distance Error) to output CSV alongside PAE. [diagnostic]
16. [x] **Input validation** — PDB file + chain sequence checks before YAML generation. [core]

### P2 — Batch Processing
9. [x] **Batch processing support** — `run_boltz_batch()`, `--batch_dir`, multi-batch aggregation with parity audit. [core]

### Backlog
10. [ ] **Template support** — Pass PDB chain templates to Boltz2 YAML (structural guidance). [diagnostic]
11. [ ] **Binding affinity integration** — Leverage Boltz2's native affinity prediction. [diagnostic]
13. [ ] **MSA server integration** — Already supported via `--use_msa_server` flag. Needs testing. [diagnostic]
14. [ ] **Constraint support** — Pass distance/contact constraints to Boltz2. [diagnostic]
15. [ ] **README & documentation** — Usage guide for Boltz2-Score. [diagnostic]
17. [ ] **Incremental CSV append** — Resume batch jobs without reprocessing. [diagnostic]
18. [ ] **Per-chain aggregation strategy** — Explicit weighted/min/max aggregation methods. [diagnostic]

---
*Last updated: 2026-03-10 Sprint 2 — Complete. All deliverables done. Drop-in parity audit confirmed.*
