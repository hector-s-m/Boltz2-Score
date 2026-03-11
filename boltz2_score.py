"""Boltz2-Score: End-to-end protein complex scoring using Boltz2.

Unified CLI that orchestrates the full pipeline:
  1. PDB → Boltz2 YAML input preparation
  2. Boltz2 inference (structure prediction + confidence)
  3. Metrics extraction → CSV output
"""

import os
import sys
import time
import argparse
from pathlib import Path

import pandas as pd

from prepare_boltz_input import run_prepare
from run_boltz2score import run_boltz_predict, run_boltz_batch, find_prediction_dirs
from extract_boltz2_metrics import extract_all_metrics


def run_pipeline(args):
    """Execute the full Boltz2-Score pipeline."""
    total_start = time.time()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    yaml_dir = output_dir / "boltz_yaml"
    boltz_output = output_dir / "boltz_predictions"
    csv_path = output_dir / "boltz2_metrics.csv"
    seq_csv = output_dir / "seq.csv"

    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Prepare Boltz2 YAML inputs ---
    if not args.skip_prepare:
        print("=" * 60)
        print("Step 1: Preparing Boltz2 YAML inputs from PDB files")
        print("=" * 60)

        run_prepare(
            input_dir=str(input_dir),
            output_dir=str(yaml_dir),
            save_csv=str(seq_csv),
            use_templates=args.use_templates,
            num_workers=args.num_workers,
        )
    else:
        print("Skipping input preparation (--skip_prepare)")

    # --- Step 2: Run Boltz2 inference ---
    if not args.skip_inference:
        print("\n" + "=" * 60)
        print("Step 2: Running Boltz2 inference")
        print("=" * 60)

        boltz_kwargs = dict(
            use_msa_server=not args.no_msa_server,
            use_potentials=args.use_potentials,
            recycling_steps=args.recycling_steps,
            sampling_steps=args.sampling_steps,
            diffusion_samples=args.diffusion_samples,
            output_format=args.output_format,
            devices=args.devices,
            accelerator=args.accelerator,
            seed=args.seed,
            write_full_pae=True,
            write_full_pde=True,
            override=args.override,
            num_workers=args.num_workers or 2,
            max_msa_seqs=args.max_msa_seqs,
        )

        # Detect batch directories (created by prepare_boltz_input --batch_dir)
        batch_dirs = None
        if args.batch_dir:
            batch_yaml = Path(args.batch_dir) / "yaml"
            if batch_yaml.is_dir():
                batch_dirs = [d.name for d in sorted(batch_yaml.iterdir()) if d.is_dir()]
                yaml_dir = batch_yaml

        rc, batch_results = run_boltz_batch(
            yaml_dir=str(yaml_dir),
            output_dir=str(boltz_output),
            batch_dirs=batch_dirs,
            **boltz_kwargs,
        )

        if rc != 0:
            print(f"Error: Boltz2 inference failed (exit code {rc})")
            sys.exit(rc)

        total_preds = sum(len(preds) for _, preds in batch_results)
        print(f"Completed {total_preds} predictions across {len(batch_results)} batch(es)")
    else:
        print("Skipping inference (--skip_inference)")

    # --- Step 3: Extract metrics ---
    if not args.skip_metrics:
        print("\n" + "=" * 60)
        print("Step 3: Extracting metrics from Boltz2 outputs")
        print("=" * 60)

        # Collect metrics from all output dirs (supports batch mode)
        output_dirs_to_scan = [str(boltz_output)]
        if args.batch_dir:
            batch_subdirs = [
                str(boltz_output / d.name)
                for d in sorted(boltz_output.iterdir())
                if d.is_dir()
            ] if boltz_output.is_dir() else []
            if batch_subdirs:
                output_dirs_to_scan = batch_subdirs

        all_dfs = []
        all_failed = []
        for scan_dir in output_dirs_to_scan:
            df_part, failed_part = extract_all_metrics(
                str(input_dir),
                scan_dir,
                num_workers=args.num_workers,
            )
            if not df_part.empty:
                all_dfs.append(df_part)
            all_failed.extend(failed_part)

        df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        failed = all_failed

        # Deduplicate by description in case batches overlap (C2)
        if not df.empty and "description" in df.columns:
            n_before = len(df)
            df = df.drop_duplicates(subset=["description"], keep="first")
            n_dropped = n_before - len(df)
            if n_dropped:
                print(f"Deduplicated: removed {n_dropped} duplicate predictions")

        if not df.empty:
            df.to_csv(str(csv_path), index=False)
            print(f"Metrics saved to {csv_path}")
            print(f"Successfully processed: {len(df)}")
        else:
            print("No metrics extracted.")

        if failed:
            print(f"Failed: {len(failed)}")
    else:
        print("Skipping metrics extraction (--skip_metrics)")

    total_time = time.time() - total_start
    print(f"\nTotal pipeline time: {total_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Boltz2-Score: Protein complex scoring using Boltz2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python boltz2_score.py --input_dir input_pdbs/ --output_dir results/

  # Skip inference (already ran), just extract metrics
  python boltz2_score.py --input_dir input_pdbs/ --output_dir results/ --skip_prepare --skip_inference

  # Use structural templates and potentials
  python boltz2_score.py --input_dir input_pdbs/ --output_dir results/ --use_templates --use_potentials
        """,
    )

    # Required
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input PDB files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for all outputs")

    # Pipeline control
    parser.add_argument("--skip_prepare", action="store_true",
                        help="Skip YAML input preparation")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip Boltz2 inference")
    parser.add_argument("--skip_metrics", action="store_true",
                        help="Skip metrics extraction")
    parser.add_argument("--batch_dir", type=str, default=None,
                        help="Batch directory (from prepare_boltz_input --batch_dir) for multi-batch inference")

    # Boltz2 inference options
    parser.add_argument("--no_msa_server", action="store_true",
                        help="Disable MSA server (requires pre-computed MSAs)")
    parser.add_argument("--use_potentials", action="store_true",
                        help="Enable physical potentials")
    parser.add_argument("--use_templates", action="store_true",
                        help="Use input PDBs as structural templates")
    parser.add_argument("--recycling_steps", type=int, default=3)
    parser.add_argument("--sampling_steps", type=int, default=200)
    parser.add_argument("--diffusion_samples", type=int, default=1)
    parser.add_argument("--output_format", choices=["mmcif", "pdb"], default="mmcif")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", choices=["gpu", "cpu", "tpu"], default="gpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--max_msa_seqs", type=int, default=4096)

    # General
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Parallel workers for data processing")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
