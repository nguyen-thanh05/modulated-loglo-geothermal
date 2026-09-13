# Modulated LOGLO-FNO geothermal surrogate

Standalone training and evaluation code for autoregressive geothermal reservoir surrogates on the heterogeneous dataset. The proposed model is **Modulated LOGLO-FNO**, trained with a rollout-aware composite objective (weighted MSE + H1 + mass-balance + radial spectral loss, with pushforward). Vanilla LOGLO-FNO, FNO, U-FNO, and U-Net are included as baselines.

This directory is self-contained. Move it anywhere and run `git init` when you want a repository.

## Install

Use the existing Anaconda environment named `torch`:

```bash
conda activate torch
pip install -e . -r requirements.txt
```

## Data

Point `data.path` in each YAML config at the heterogeneous `.npy` stack. The default `../dataset_v3_hetero/` works while this folder sits next to that dataset. After you move the code, edit that one field.

Required files:

- `all_temp_formation.npy`, `all_temp_frac.npy`
- `all_pres_formation.npy`, `all_pres_frac.npy`
- `all_action.npy`
- `all_por_matrix.npy`, `all_por_frac.npy`
- `all_perm_matrix.npy`, `all_perm_frac.npy`

Train trajectories are indices `0–299`. Held-out test trajectories are `350–399`.

The special `modulated_loglo_aux` config also reads `all_energyrate_bhp.npy` (9 BHPs + 7 producer energy rates).

## Train

```bash
python training/train.py --config configs/modulated_loglo.yml --seed 42
```

Auxiliary-head variant (fields + well BHP/energy):

```bash
python training/train.py --config configs/modulated_loglo_aux.yml --seed 42
```

Configs under `configs/` are the full composite objective. `configs/mse_only/` disables H1, MBE, spectral loss, and pushforward. `configs/loss_ablation/` leaves one term out of Modulated LOGLO-FNO.

## HPC

From the repo root on the cluster:

```bash
python slurm/launch.py --models modulated_loglo --seeds '{"modulated_loglo":"42"}' --dry-run
python slurm/launch.py --models modulated_loglo_aux --seeds '{"modulated_loglo_aux":"42"}' --dry-run
python slurm/launch_mse_only.py --dry-run
python slurm/launch_loss_ablation.py --dry-run
```

`slurm/train.sh` is the sbatch template. Edit account, modules, and log paths for your cluster if they differ.

## Evaluate

Autoregressive rollouts (relative L2, RMSE, absolute error):

```bash
python eval/rollout.py --checkpoint-root /path/to/checkpoints/new_lr
python eval/plot_rollouts.py
```

`--checkpoint-root` looks for `<root>/seed<seed>/<basename(final_path)>`, matching the existing `new_lr` layout. Weights default to the `ema_model` key.

Auxiliary-head rollouts (fields plus 9 BHP and 7 energy-rate timeseries):

```bash
python eval/rollout_aux.py --checkpoint-root /path/to/checkpoints --seed 42
```
