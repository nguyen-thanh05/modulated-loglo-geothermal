#!/usr/bin/env python3
"""HPC job submission launcher for geothermal surrogate models."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    'modulated_loglo':   {'display': 'Modulated LOGLO-FNO', 'config': 'configs/modulated_loglo.yml'},
    'modulated_loglo_aux': {'display': 'Modulated LOGLO-FNO Aux', 'config': 'configs/modulated_loglo_aux.yml'},
    'vanilla_loglo':     {'display': 'Vanilla LOGLO-FNO', 'config': 'configs/vanilla_loglo.yml'},
    'fno_m4x16x8_h128':  {'display': 'FNO m4x16x8 h128', 'config': 'configs/fno_m4x16x8_h128.yml'},
    'fno_m8x32x16_h64':  {'display': 'FNO m8x32x16 h64', 'config': 'configs/fno_m8x32x16_h64.yml'},
    'ufno':              {'display': 'U-FNO', 'config': 'configs/ufno.yml'},
    'unet':              {'display': 'UNet3D', 'config': 'configs/unet.yml'},
}

DEFAULT_JOBS_PER_CHAIN = 4
LOGLO_JOBS_PER_CHAIN = 4
SLURM_TEMPLATE = str(REPO_ROOT / 'slurm' / 'train.sh')


def default_jobs_per_chain(model_key):
    if 'loglo' in model_key:
        return LOGLO_JOBS_PER_CHAIN
    return DEFAULT_JOBS_PER_CHAIN


def resolve_jobs_per_chain(model_key, jobs_per_chain):
    if jobs_per_chain is not None:
        return jobs_per_chain
    return default_jobs_per_chain(model_key)


def submit_chain(config_path, seed, jobs_per_chain, dry_run=False):
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    job_name = f'{config_name}_s{seed}'

    job_ids = []
    for i in range(jobs_per_chain):
        cmd = [
            'sbatch',
            f'--job-name={job_name}',
            f'--chdir={REPO_ROOT}',
            '--exclude=fc10713',
            f'--export=ALL,CONFIG={config_path},SEED={seed}',
        ]
        if job_ids:
            cmd.append(f'--dependency=afterok:{job_ids[-1]}')
        cmd.append(SLURM_TEMPLATE)

        if dry_run:
            print(f'  [dry-run] {" ".join(cmd)}')
            job_ids.append(f'DRY{i}')
            continue

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f'ERROR: sbatch failed: {result.stderr.strip()}')
            sys.exit(1)
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)

    return job_ids


def pick_numbered(prompt, options, allow_all=True):
    for i, opt in enumerate(options, 1):
        print(f'  {i}) {opt}')
    hint = "comma-separated numbers, or 'all'" if allow_all else 'comma-separated numbers'
    default = 'all' if allow_all else '1'
    raw = input(f'{prompt} [{default}]: ').strip() or default
    if raw.lower() == 'all' and allow_all:
        return list(options)
    try:
        indices = [int(x.strip()) for x in raw.split(',')]
        return [options[i - 1] for i in indices if 1 <= i <= len(options)]
    except (ValueError, IndexError):
        print('Invalid selection.')
        sys.exit(1)


def interactive_mode(dry_run=False, jobs_per_chain=None):
    print('\n=== HPC Job Launcher ===\n')

    model_keys = list(MODELS.keys())
    model_displays = [MODELS[k]['display'] for k in model_keys]
    print('Available models:')
    selected_displays = pick_numbered('Select models', model_displays)
    selected_models = [model_keys[model_displays.index(d)] for d in selected_displays]

    seeds_per_model = {}
    print()
    for mk in selected_models:
        display = MODELS[mk]['display']
        raw = input(f'Seeds for {display} (comma-separated) [42]: ').strip() or '42'
        try:
            seeds_per_model[mk] = [int(s.strip()) for s in raw.split(',')]
        except ValueError:
            print('Seeds must be integers.')
            sys.exit(1)

    default_label = (
        f'auto (LOGLO {LOGLO_JOBS_PER_CHAIN}, others {DEFAULT_JOBS_PER_CHAIN})'
        if jobs_per_chain is None else str(jobs_per_chain)
    )
    raw = input(f'\nSegments per experiment [{default_label}]: ').strip()
    if raw:
        try:
            jobs_per_chain = int(raw)
        except ValueError:
            print('Segments must be an integer.')
            sys.exit(1)

    experiments = []
    for mk in selected_models:
        config = MODELS[mk]['config']
        if not os.path.isfile(config):
            print(f'WARNING: {config} not found, skipping.')
            continue
        for seed in seeds_per_model[mk]:
            experiments.append((mk, seed, config))

    print_summary(experiments, jobs_per_chain)

    confirm = input('Submit? [y/N]: ').strip().lower()
    if confirm != 'y':
        print('Aborted.')
        sys.exit(0)

    submit_experiments(experiments, jobs_per_chain, dry_run)


def cli_mode(args, dry_run=False):
    selected_models = [m.strip() for m in args.models.split(',')]
    for m in selected_models:
        if m not in MODELS:
            print(f"Unknown model: '{m}'. Available: {', '.join(MODELS.keys())}")
            sys.exit(1)

    seeds_map = json.loads(args.seeds)
    seeds_per_model = {}
    for mk in selected_models:
        raw = seeds_map.get(mk, '42')
        seeds_per_model[mk] = [int(s.strip()) for s in str(raw).split(',')]

    experiments = []
    for mk in selected_models:
        config = MODELS[mk]['config']
        if not os.path.isfile(config):
            print(f'WARNING: {config} not found, skipping.')
            continue
        for seed in seeds_per_model[mk]:
            experiments.append((mk, seed, config))

    print_summary(experiments, args.jobs_per_chain)
    submit_experiments(experiments, args.jobs_per_chain, dry_run)


def print_summary(experiments, jobs_per_chain):
    rows = []
    total_jobs = 0
    for mk, seed, config in experiments:
        segment_count = resolve_jobs_per_chain(mk, jobs_per_chain)
        rows.append((MODELS[mk]['display'], seed, segment_count, config))
        total_jobs += segment_count

    print(f'\n{"Model":<22} {"Seed":<12} {"Segments":<8} {"Config"}')
    print('-' * 72)
    for display, seed, segment_count, config in rows:
        print(f'{display:<22} {seed:<12} {segment_count:<8} {config}')
    print(f'\nTotal: {len(experiments)} experiments = {total_jobs} SLURM jobs\n')


def submit_experiments(experiments, jobs_per_chain, dry_run=False):
    print('Submitting...\n')
    all_job_ids = []

    for mk, seed, config in experiments:
        display = MODELS[mk]['display']
        segment_count = resolve_jobs_per_chain(mk, jobs_per_chain)
        job_ids = submit_chain(config, seed, segment_count, dry_run=dry_run)
        all_job_ids.extend(job_ids)

        print(f'{display} / seed{seed}:')
        for i, jid in enumerate(job_ids):
            dep = f' (depends on {job_ids[i-1]})' if i > 0 else ''
            print(f'  Segment {i+1}: {jid}{dep}')
        print()

    if not dry_run:
        print(f'Monitor: squeue -u $USER')
        print(f'Cancel all: scancel {" ".join(all_job_ids)}')


def main():
    parser = argparse.ArgumentParser(
        description='HPC job launcher for geothermal surrogate models')
    parser.add_argument('--models', type=str, default=None,
                        help=f'Comma-separated model keys: {",".join(MODELS.keys())}')
    parser.add_argument('--seeds', type=str, default=None,
                        help='JSON dict mapping model key to comma-separated seeds, '
                             'e.g. \'{"modulated_loglo":"42,123"}\'')
    parser.add_argument('--jobs-per-chain', type=int, default=None,
                        help='Override chained segments per experiment. '
                             f'Default: LOGLO={LOGLO_JOBS_PER_CHAIN}, '
                             f'others={DEFAULT_JOBS_PER_CHAIN}.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print sbatch commands without submitting')
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    (REPO_ROOT / 'logs').mkdir(exist_ok=True)

    if args.models is not None:
        if args.seeds is None:
            print('CLI mode requires --models and --seeds.')
            sys.exit(1)
        cli_mode(args, args.dry_run)
    else:
        interactive_mode(args.dry_run, args.jobs_per_chain)


if __name__ == '__main__':
    main()
