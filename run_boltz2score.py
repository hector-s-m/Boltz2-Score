"""Run Boltz2 inference for structure prediction and scoring.

Replaces run_af3score.py. Uses Boltz2's prediction API to generate
structures with confidence metrics (pTM, ipTM, pLDDT, PAE, PDE).
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path


def run_boltz_predict(
    input_path,
    output_dir,
    use_msa_server=True,
    use_templates=False,
    use_potentials=False,
    recycling_steps=3,
    sampling_steps=200,
    diffusion_samples=1,
    output_format="mmcif",
    devices=1,
    accelerator="gpu",
    seed=None,
    write_full_pae=True,
    write_full_pde=True,
    override=False,
    cache=None,
    num_workers=2,
    max_msa_seqs=4096,
    extra_args=None,
):
    """Run Boltz2 prediction via CLI subprocess.

    Uses subprocess to invoke `boltz predict` — more robust than calling
    the click-decorated function directly, and provides better logging.

    Args:
        input_path: Path to YAML file or directory of YAML files
        output_dir: Directory for prediction output
        use_msa_server: Auto-generate MSA via mmseqs2
        use_templates: Include structural templates
        use_potentials: Enable physical potentials
        recycling_steps: Number of recycling iterations
        sampling_steps: Diffusion sampling steps
        diffusion_samples: Number of structure samples per input
        output_format: "mmcif" or "pdb"
        devices: Number of GPU devices
        accelerator: "gpu", "cpu", or "tpu"
        seed: Random seed (None for random)
        write_full_pae: Write full PAE matrix as NPZ
        write_full_pde: Write full PDE matrix as NPZ
        override: Re-run predictions even if output exists
        cache: Cache directory for model weights
        num_workers: Data loading workers
        max_msa_seqs: Maximum MSA sequences
        extra_args: List of additional CLI arguments

    Returns:
        int: Return code (0 for success)
    """
    cmd = ["boltz", "predict", str(input_path)]

    cmd.extend(["--out_dir", str(output_dir)])
    cmd.extend(["--recycling_steps", str(recycling_steps)])
    cmd.extend(["--sampling_steps", str(sampling_steps)])
    cmd.extend(["--diffusion_samples", str(diffusion_samples)])
    cmd.extend(["--output_format", output_format])
    cmd.extend(["--devices", str(devices)])
    cmd.extend(["--accelerator", accelerator])
    cmd.extend(["--num_workers", str(num_workers)])
    cmd.extend(["--max_msa_seqs", str(max_msa_seqs)])

    if use_msa_server:
        cmd.append("--use_msa_server")
    if use_potentials:
        cmd.append("--use_potentials")
    if write_full_pae:
        cmd.append("--write_full_pae")
    if write_full_pde:
        cmd.append("--write_full_pde")
    if override:
        cmd.append("--override")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if cache:
        cmd.extend(["--cache", str(cache)])
    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()

    result = subprocess.run(cmd, capture_output=False)

    elapsed = time.time() - start_time
    print(f"Boltz2 prediction completed in {elapsed:.1f}s (exit code: {result.returncode})")

    return result.returncode


def run_boltz_batch(
    yaml_dir,
    output_dir,
    batch_dirs=None,
    **kwargs,
):
    """Run Boltz2 predictions in batch mode.

    If batch_dirs is provided (list of subdirectories containing YAML files),
    run each batch sequentially with separate output dirs.
    Otherwise, run the entire yaml_dir as a single batch.

    Args:
        yaml_dir: Directory containing YAML files (or parent of batch dirs)
        output_dir: Base output directory
        batch_dirs: Optional list of batch subdirectory names
        **kwargs: Forwarded to run_boltz_predict

    Returns:
        tuple: (total_return_code, list of (batch_name, pred_dirs))
    """
    if batch_dirs is None:
        # Single batch mode — run entire directory
        rc = run_boltz_predict(input_path=yaml_dir, output_dir=output_dir, **kwargs)
        preds = find_prediction_dirs(output_dir)
        return rc, [("all", preds)]

    # Multi-batch mode — run each batch sequentially
    all_results = []
    overall_rc = 0

    for batch_name in sorted(batch_dirs):
        batch_input = os.path.join(yaml_dir, batch_name)
        if not os.path.isdir(batch_input):
            print(f"Warning: Batch dir not found: {batch_input}, skipping")
            continue

        batch_output = os.path.join(output_dir, batch_name)
        print(f"\n--- Batch: {batch_name} ---")

        rc = run_boltz_predict(
            input_path=batch_input,
            output_dir=batch_output,
            **kwargs,
        )

        if rc != 0:
            print(f"Warning: Batch {batch_name} failed (exit code {rc})")
            overall_rc = rc
        else:
            preds = find_prediction_dirs(batch_output)
            all_results.append((batch_name, preds))
            print(f"Batch {batch_name}: {len(preds)} predictions completed")

    return overall_rc, all_results


def find_prediction_dirs(output_dir):
    """Find all prediction output directories under the Boltz2 output tree.

    Boltz2 output structure: out_dir/boltz_results_*/predictions/*/

    Returns:
        list: List of (stem_name, prediction_dir) tuples
    """
    results = []
    output_path = Path(output_dir)

    # Look for boltz_results_* directories
    for results_dir in sorted(output_path.glob("boltz_results_*")):
        predictions_dir = results_dir / "predictions"
        if predictions_dir.is_dir():
            for pred_dir in sorted(predictions_dir.iterdir()):
                if pred_dir.is_dir():
                    results.append((pred_dir.name, pred_dir))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Boltz2 structure prediction and scoring"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to YAML file or directory of YAML files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for predictions")
    parser.add_argument("--use_msa_server", action="store_true", default=True,
                        help="Use MMseqs2 MSA server (default: True)")
    parser.add_argument("--no_msa_server", action="store_true",
                        help="Disable MSA server")
    parser.add_argument("--use_potentials", action="store_true",
                        help="Enable physical potentials for better structures")
    parser.add_argument("--recycling_steps", type=int, default=3)
    parser.add_argument("--sampling_steps", type=int, default=200)
    parser.add_argument("--diffusion_samples", type=int, default=1,
                        help="Number of structure samples per input")
    parser.add_argument("--output_format", choices=["mmcif", "pdb"], default="mmcif")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", choices=["gpu", "cpu", "tpu"], default="gpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--override", action="store_true",
                        help="Re-run predictions even if output exists")
    parser.add_argument("--cache", type=str, default=None,
                        help="Cache directory for model weights")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_msa_seqs", type=int, default=4096)
    args = parser.parse_args()

    use_msa = not args.no_msa_server

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)

    return_code = run_boltz_predict(
        input_path=args.input,
        output_dir=args.output_dir,
        use_msa_server=use_msa,
        use_potentials=args.use_potentials,
        recycling_steps=args.recycling_steps,
        sampling_steps=args.sampling_steps,
        diffusion_samples=args.diffusion_samples,
        output_format=args.output_format,
        devices=args.devices,
        accelerator=args.accelerator,
        seed=args.seed,
        override=args.override,
        cache=args.cache,
        num_workers=args.num_workers,
        max_msa_seqs=args.max_msa_seqs,
    )

    if return_code != 0:
        print(f"Error: Boltz2 prediction failed with exit code {return_code}")
        sys.exit(return_code)

    # Report results
    pred_dirs = find_prediction_dirs(args.output_dir)
    print(f"\nCompleted {len(pred_dirs)} predictions:")
    for name, path in pred_dirs:
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
