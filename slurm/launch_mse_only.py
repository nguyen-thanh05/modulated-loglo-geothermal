#!/usr/bin/env python3
"""Submit the MSE-only architecture sweep."""

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slurm.launch import MODELS, default_jobs_per_chain, submit_chain


DEFAULT_SEED = 42
CONFIG_DIR = Path("configs/mse_only")


def get_config_path(model_key):
    return CONFIG_DIR / f"{model_key}_mse_only.yml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Training seed for every architecture (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--jobs-per-chain",
        type=int,
        default=None,
        help="Override the number of chained SLURM segments per experiment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting jobs",
    )
    args = parser.parse_args()

    if args.jobs_per_chain is not None and args.jobs_per_chain < 1:
        parser.error("--jobs-per-chain must be at least 1")

    os.chdir(REPO_ROOT)

    experiments = []
    for model_key in MODELS:
        config = get_config_path(model_key)
        segment_count = (
            args.jobs_per_chain
            if args.jobs_per_chain is not None
            else default_jobs_per_chain(model_key)
        )
        experiments.append((model_key, config, segment_count))

    missing = [config for _, config, _ in experiments if not config.is_file()]
    if missing:
        for config in missing:
            print(f"ERROR: missing config: {config}")
        sys.exit(1)

    total_jobs = sum(segment_count for _, _, segment_count in experiments)
    mode = "Dry run" if args.dry_run else "Submitting"
    print(
        f"{mode}: {len(experiments)} MSE-only experiments = "
        f"{total_jobs} jobs (seed {args.seed})\n"
    )

    all_job_ids = []
    for model_key, config, segment_count in experiments:
        job_ids = submit_chain(
            str(config),
            args.seed,
            segment_count,
            dry_run=args.dry_run,
        )
        all_job_ids.extend(job_ids)

        display = MODELS[model_key]["display"]
        print(f"{display} / seed{args.seed}:")
        for index, job_id in enumerate(job_ids):
            dependency = f" (depends on {job_ids[index - 1]})" if index else ""
            print(f"  Segment {index + 1}: {job_id}{dependency}")
        print()

    if not args.dry_run:
        print("Monitor: squeue -u $USER")
        print(f'Cancel all: scancel {" ".join(all_job_ids)}')


if __name__ == "__main__":
    main()
