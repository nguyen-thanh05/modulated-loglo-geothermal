"""Autoregressive field + aux rollouts for Modulated LOGLO-FNO Aux."""

from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.constants import (
    BHP_NAMES,
    CHANNEL_NAMES,
    ENERGY_NAMES,
    N_PRODUCER_WELLS,
    TEST_INDEX,
    WELL_COORDS,
    WELL_NAMES,
)
from training.dataset import ARDataset
from training.model_adapters import create_adapter, split_model_output
from training.model_factory import create_model


DEFAULT_CONFIG = REPO_ROOT / "configs" / "modulated_loglo_aux.yml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "evaluation_results" / "rollouts_aux"
CHANNEL_UNITS_PHYSICAL = ("degC", "degC", "kPa", "kPa")
FIELD_METRIC_NAMES = (
    "relative_l2",
    "rmse_normalized",
    "rmse_physical",
)


def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def read_config(config_path):
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key in ("data", "model", "checkpoints"):
        if key not in config:
            raise KeyError(f"{config_path} is missing '{key}'")
    return config


def inject_seed_into_paths(config, seed):
    seeded = copy.deepcopy(config)
    checkpoints = seeded["checkpoints"]
    seed_dir = f"seed{seed}"
    if "running_dir" in checkpoints:
        checkpoints["running_dir"] = os.path.join(
            checkpoints["running_dir"], seed_dir, ""
        )
    final_dir, final_name = os.path.split(checkpoints["final_path"])
    checkpoints["final_path"] = os.path.join(final_dir, seed_dir, final_name)
    return seeded


def resolve_checkpoint_path(config, seed, checkpoint_root):
    final_name = os.path.basename(config["checkpoints"]["final_path"])
    if checkpoint_root is not None:
        seeded_path = checkpoint_root / f"seed{seed}" / final_name
        if seeded_path.is_file():
            return seeded_path.resolve()
        return (checkpoint_root / final_name).resolve()
    return resolve_path(
        inject_seed_into_paths(config, seed)["checkpoints"]["final_path"]
    )


def load_array(data_path, filename):
    path = data_path / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset array: {path}")
    return np.load(path, mmap_mode="r")


def load_dataset(config):
    data_path = resolve_path(config["data"]["path"])
    k_max = config.get("training", {}).get("pushforward_k_max", 1)
    dataset = ARDataset(
        load_array(data_path, "all_temp_formation.npy"),
        load_array(data_path, "all_temp_frac.npy"),
        load_array(data_path, "all_pres_formation.npy"),
        load_array(data_path, "all_pres_frac.npy"),
        load_array(data_path, "all_action.npy"),
        load_array(data_path, "all_por_matrix.npy"),
        load_array(data_path, "all_por_frac.npy"),
        load_array(data_path, "all_perm_matrix.npy"),
        load_array(data_path, "all_perm_frac.npy"),
        load_array(data_path, "all_energyrate_bhp.npy"),
        k_max=k_max,
    )
    return dataset, data_path


def load_model(config, checkpoint_path, device, weight_key):
    model_type = config["model"]["type"]
    model = create_model(config["model"], model_type)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )
    if not isinstance(checkpoint, dict) or weight_key not in checkpoint:
        raise KeyError(
            f"Checkpoint has no '{weight_key}' weights: {checkpoint_path}"
        )
    state = checkpoint[weight_key]
    if "_metadata" in state:
        state = dict(state)
        del state["_metadata"]
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, create_adapter(model_type)


def mask_wells(action):
    for h_index, w_index in WELL_COORDS:
        action[:, 0:2, h_index, w_index] = 0.0


def stack_field(dataset, getter, indices, device, *extra):
    return torch.stack(
        [getter(index, *extra) for index in indices], dim=0,
    ).to(device)


def physical_scales(dataset, device):
    return torch.tensor(
        [
            dataset._temp_range,
            dataset._temp_range,
            dataset._pres_range,
            dataset._pres_range,
        ],
        dtype=torch.float32,
        device=device,
    )


def compute_field_metrics(prediction, target, scales):
    difference = prediction - target
    flat_diff = difference.flatten(start_dim=2)
    flat_target = target.flatten(start_dim=2)
    relative_l2 = torch.linalg.vector_norm(flat_diff, ord=2, dim=2) / (
        torch.linalg.vector_norm(flat_target, ord=2, dim=2)
    )
    rmse_normalized = torch.sqrt(difference.square().mean(dim=(2, 3, 4)))
    return {
        "relative_l2": relative_l2,
        "rmse_normalized": rmse_normalized,
        "rmse_physical": rmse_normalized * scales,
    }


def aux_rmse(pred, target):
    return torch.sqrt((pred - target).square())


@torch.inference_mode()
def evaluate_checkpoint(model, adapter, dataset, trajectory_indices,
                        batch_size, device):
    n_trajectories = len(trajectory_indices)
    n_timesteps = dataset.n_timesteps
    n_channels = len(CHANNEL_NAMES)
    field_metrics = {
        name: np.empty((n_trajectories, n_timesteps, n_channels), dtype=np.float32)
        for name in FIELD_METRIC_NAMES
    }
    pred_aux = np.empty((n_trajectories, n_timesteps, 16), dtype=np.float32)
    gt_aux = np.empty((n_trajectories, n_timesteps, 16), dtype=np.float32)
    scales = physical_scales(dataset, device)

    for batch_start in range(0, n_trajectories, batch_size):
        batch_stop = min(batch_start + batch_size, n_trajectories)
        batch_indices = trajectory_indices[batch_start:batch_stop]
        state = stack_field(dataset, dataset._state_at, batch_indices, device, 0)
        static = stack_field(dataset, dataset._static_at, batch_indices, device)

        for timestep in range(n_timesteps):
            action = stack_field(
                dataset, dataset._action_at, batch_indices, device, timestep,
            )
            mask_wells(action)
            prediction, prediction_aux = split_model_output(
                adapter.forward(
                    model, adapter.build_model_input(state, action, static),
                )
            )
            if prediction_aux is None:
                raise RuntimeError(
                    "Model did not return aux; expected modulated_loglo_aux"
                )
            target = stack_field(
                dataset, dataset._state_at, batch_indices, device, timestep + 1,
            )
            aux_target = stack_field(
                dataset, dataset._aux_at, batch_indices, device, timestep + 1,
            )
            step_metrics = compute_field_metrics(prediction, target, scales)
            for name, values in step_metrics.items():
                field_metrics[name][batch_start:batch_stop, timestep] = (
                    values.cpu().numpy()
                )
            pred_aux[batch_start:batch_stop, timestep] = (
                prediction_aux.cpu().numpy()
            )
            gt_aux[batch_start:batch_stop, timestep] = aux_target.cpu().numpy()
            state = prediction

        print(f"    completed {batch_stop}/{n_trajectories} trajectories", flush=True)

    pred_aux_t = torch.from_numpy(pred_aux)
    gt_aux_t = torch.from_numpy(gt_aux)
    bhp_rmse_norm = aux_rmse(pred_aux_t[..., :9], gt_aux_t[..., :9]).numpy()
    energy_rmse_norm = aux_rmse(pred_aux_t[..., 9:], gt_aux_t[..., 9:]).numpy()
    pred_bhp, pred_energy = dataset.denormalize_aux(pred_aux_t)
    gt_bhp, gt_energy = dataset.denormalize_aux(gt_aux_t)
    aux_metrics = {
        "bhp_rmse_normalized": bhp_rmse_norm,
        "energy_rmse_normalized": energy_rmse_norm,
        "bhp_rmse_physical": aux_rmse(pred_bhp, gt_bhp).numpy(),
        "energy_rmse_physical": aux_rmse(pred_energy, gt_energy).numpy(),
        "pred_aux_normalized": pred_aux,
        "gt_aux_normalized": gt_aux,
        "pred_bhp_physical": pred_bhp.numpy(),
        "gt_bhp_physical": gt_bhp.numpy(),
        "pred_energy_physical": pred_energy.numpy(),
        "gt_energy_physical": gt_energy.numpy(),
    }
    return field_metrics, aux_metrics


def plot_aux_timeseries(aux_metrics, trajectory_indices, output_dir, max_plots):
    output_dir.mkdir(parents=True, exist_ok=True)
    n_plot = min(max_plots, len(trajectory_indices))
    timesteps = np.arange(aux_metrics["pred_bhp_physical"].shape[1])

    for plot_index in range(n_plot):
        traj = trajectory_indices[plot_index]
        fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
        for well, ax in enumerate(axes.ravel()):
            ax.plot(
                timesteps, aux_metrics["gt_bhp_physical"][plot_index, :, well],
                color="black", linestyle="--", label="GT",
            )
            ax.plot(
                timesteps, aux_metrics["pred_bhp_physical"][plot_index, :, well],
                color="tab:blue", label="Pred",
            )
            ax.set_title(f"BHP — {WELL_NAMES[well]}")
            ax.set_ylabel("kPa")
            if well == 0:
                ax.legend(frameon=False, fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("week")
        fig.suptitle(f"Trajectory {traj} bottomhole pressure")
        fig.tight_layout()
        fig.savefig(output_dir / f"traj{traj}_bhp.png", dpi=120)
        plt.close(fig)

        fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
        for well, ax in enumerate(axes.ravel()):
            if well >= N_PRODUCER_WELLS:
                ax.axis("off")
                continue
            ax.plot(
                timesteps,
                aux_metrics["gt_energy_physical"][plot_index, :, well],
                color="black", linestyle="--", label="GT",
            )
            ax.plot(
                timesteps,
                aux_metrics["pred_energy_physical"][plot_index, :, well],
                color="tab:orange", label="Pred",
            )
            ax.set_title(f"Energy — {WELL_NAMES[well]}")
            ax.set_ylabel("J/day")
            if well == 0:
                ax.legend(frameon=False, fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("week")
        fig.suptitle(f"Trajectory {traj} producer energy rate")
        fig.tight_layout()
        fig.savefig(output_dir / f"traj{traj}_energy.png", dpi=120)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    axes[0].plot(
        timesteps,
        aux_metrics["bhp_rmse_physical"].mean(axis=(0, 2)),
        color="tab:blue",
    )
    axes[0].set_title("Mean BHP RMSE")
    axes[0].set_ylabel("kPa")
    axes[0].set_xlabel("week")
    axes[1].plot(
        timesteps,
        aux_metrics["energy_rmse_physical"].mean(axis=(0, 2)),
        color="tab:orange",
    )
    axes[1].set_title("Mean energy RMSE")
    axes[1].set_ylabel("J/day")
    axes[1].set_xlabel("week")
    fig.tight_layout()
    fig.savefig(output_dir / "mean_aux_rmse_vs_time.png", dpi=120)
    plt.close(fig)


def save_metrics(output_path, field_metrics, aux_metrics, config_path, config,
                 checkpoint_path, seed, trajectory_indices, data_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"field_{name}": values for name, values in field_metrics.items()}
    payload.update(aux_metrics)
    payload.update(
        {
            "seed": np.asarray(seed, dtype=np.int64),
            "trajectory_indices": np.asarray(trajectory_indices, dtype=np.int64),
            "channel_names": np.asarray(CHANNEL_NAMES),
            "bhp_names": np.asarray(BHP_NAMES),
            "energy_names": np.asarray(ENERGY_NAMES),
            "channel_units_physical": np.asarray(CHANNEL_UNITS_PHYSICAL),
            "config_path": np.asarray(str(config_path)),
            "data_path": np.asarray(str(data_path)),
            "checkpoint_path": np.asarray(str(checkpoint_path)),
            "model_type": np.asarray(config["model"]["type"]),
        }
    )
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp.npz")
    np.savez_compressed(temporary_path, **payload)
    os.replace(temporary_path, output_path)


def release_device_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate field + aux autoregressive rollouts "
                    "for modulated_loglo_aux.",
    )
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG),
        help="YAML config for modulated_loglo_aux.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=TEST_INDEX[0])
    parser.add_argument("--end-index", type=int, default=TEST_INDEX[-1])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=None,
        help="Prefers <root>/seed<seed>/<basename(final_path)>.",
    )
    parser.add_argument(
        "--weight-key",
        type=str,
        choices=("ema_model", "model"),
        default="ema_model",
    )
    parser.add_argument(
        "--plot-max", type=int, default=4,
        help="Number of trajectories to plot (metrics are saved for all).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size <= 0 or args.end_index < args.start_index:
        raise ValueError("Invalid --batch-size or trajectory index range")

    config_path = resolve_path(args.config)
    config = read_config(config_path)
    if config["model"]["type"] != "modulated_loglo_aux":
        raise ValueError(
            f"{config_path} has model.type={config['model']['type']}; "
            "expected modulated_loglo_aux"
        )

    checkpoint_root = (
        resolve_path(args.checkpoint_root) if args.checkpoint_root else None
    )
    checkpoint_path = resolve_checkpoint_path(
        config, args.seed, checkpoint_root,
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = torch.device(
        args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    output_root = resolve_path(args.output_root) / config_path.stem
    trajectory_indices = list(range(args.start_index, args.end_index + 1))

    print(f"Device: {device}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Trajectories: {args.start_index}-{args.end_index}")

    dataset, data_path = load_dataset(config)
    model = None
    try:
        model, adapter = load_model(
            config, checkpoint_path, device, args.weight_key,
        )
        field_metrics, aux_metrics = evaluate_checkpoint(
            model, adapter, dataset, trajectory_indices,
            args.batch_size, device,
        )
    finally:
        del model
        release_device_memory(device)

    save_metrics(
        output_root / "metrics.npz",
        field_metrics, aux_metrics, config_path, config,
        checkpoint_path, args.seed, trajectory_indices, data_path,
    )
    plot_aux_timeseries(
        aux_metrics, trajectory_indices, output_root / "plots", args.plot_max,
    )
    print(f"Saved {output_root / 'metrics.npz'}")
    print(f"Plots in {output_root / 'plots'}")
    mean_bhp = aux_metrics["bhp_rmse_physical"].mean()
    mean_energy = aux_metrics["energy_rmse_physical"].mean()
    mean_field = field_metrics["relative_l2"].mean()
    print(f"Mean field rel-L2: {mean_field:.5f}")
    print(f"Mean BHP RMSE: {mean_bhp:.2f} kPa")
    print(f"Mean energy RMSE: {mean_energy:.4e} J/day")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
