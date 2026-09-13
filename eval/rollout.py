"""Autoregressive rollouts for one or more model configs."""

from __future__ import annotations

import argparse
import copy
import gc
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.constants import CHANNEL_NAMES, WELL_COORDS
from training.dataset import ARDataset
from training.model_adapters import create_adapter
from training.model_factory import create_model


DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "evaluation_results" / "rollouts"
DEFAULT_SEEDS = (5, 42, 2026)
CHANNEL_UNITS_PHYSICAL = ("degC", "degC", "kPa", "kPa")
METRIC_NAMES = (
    "relative_l2",
    "rmse_normalized",
    "rmse_physical",
    "absolute_error_max_normalized",
    "absolute_error_min_normalized",
    "absolute_error_max_physical",
    "absolute_error_min_physical",
)


def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def expand_config_paths(raw_paths):
    if raw_paths is None:
        paths = sorted(DEFAULT_CONFIG_DIR.glob("*.yml"))
        if not paths:
            raise FileNotFoundError(f"No YAML configs in {DEFAULT_CONFIG_DIR}")
        return [path.resolve() for path in paths]

    expanded = []
    for raw_path in raw_paths:
        if glob.has_magic(raw_path):
            pattern = Path(raw_path)
            if not pattern.is_absolute():
                pattern = REPO_ROOT / pattern
            matches = sorted(Path(match).resolve() for match in glob.glob(str(pattern)))
            if not matches:
                raise FileNotFoundError(f"Config pattern matched nothing: {raw_path}")
            expanded.extend(matches)
        else:
            expanded.append(resolve_path(raw_path))
    return expanded


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
    return resolve_path(inject_seed_into_paths(config, seed)["checkpoints"]["final_path"])


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
        raise KeyError(f"Checkpoint has no '{weight_key}' weights: {checkpoint_path}")
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


def compute_metrics(prediction, target, scales):
    difference = prediction - target
    flat_diff = difference.flatten(start_dim=2)
    flat_target = target.flatten(start_dim=2)
    relative_l2 = torch.linalg.vector_norm(flat_diff, ord=2, dim=2) / (
        torch.linalg.vector_norm(flat_target, ord=2, dim=2)
    )
    rmse_normalized = torch.sqrt(difference.square().mean(dim=(2, 3, 4)))
    abs_error = difference.abs()
    abs_max = abs_error.amax(dim=(2, 3, 4))
    abs_min = abs_error.amin(dim=(2, 3, 4))
    return {
        "relative_l2": relative_l2,
        "rmse_normalized": rmse_normalized,
        "rmse_physical": rmse_normalized * scales,
        "absolute_error_max_normalized": abs_max,
        "absolute_error_min_normalized": abs_min,
        "absolute_error_max_physical": abs_max * scales,
        "absolute_error_min_physical": abs_min * scales,
    }


@torch.inference_mode()
def evaluate_checkpoint(model, adapter, dataset, trajectory_indices, batch_size, device):
    n_trajectories = len(trajectory_indices)
    n_timesteps = dataset.n_timesteps
    n_channels = len(CHANNEL_NAMES)
    metrics = {
        name: np.empty((n_trajectories, n_timesteps, n_channels), dtype=np.float32)
        for name in METRIC_NAMES
    }
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
            prediction = adapter.forward(
                model, adapter.build_model_input(state, action, static),
            )
            target = stack_field(
                dataset, dataset._state_at, batch_indices, device, timestep + 1,
            )
            step_metrics = compute_metrics(prediction, target, scales)
            for name, values in step_metrics.items():
                if not torch.isfinite(values).all():
                    raise FloatingPointError(
                        f"Non-finite {name} at trajectories {batch_indices}, "
                        f"timestep={timestep}"
                    )
                metrics[name][batch_start:batch_stop, timestep] = values.cpu().numpy()
            state = prediction

        print(f"    completed {batch_stop}/{n_trajectories} trajectories", flush=True)
    return metrics


def save_metrics(output_path, metrics, config_path, config, checkpoint_paths,
                 seeds, trajectory_indices, data_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_timesteps = metrics["relative_l2"].shape[2]
    payload = dict(metrics)
    payload.update(
        {
            "seeds": np.asarray(seeds, dtype=np.int64),
            "trajectory_indices": np.asarray(trajectory_indices, dtype=np.int64),
            "rollout_timesteps": np.arange(n_timesteps, dtype=np.int64),
            "target_state_indices": np.arange(1, n_timesteps + 1, dtype=np.int64),
            "channel_names": np.asarray(CHANNEL_NAMES),
            "channel_units_physical": np.asarray(CHANNEL_UNITS_PHYSICAL),
            "config_path": np.asarray(str(config_path)),
            "data_path": np.asarray(str(data_path)),
            "checkpoint_paths": np.asarray([str(path) for path in checkpoint_paths]),
            "model_type": np.asarray(config["model"]["type"]),
            "relative_l2_definition": np.asarray(
                "||prediction-target||_2 / ||target||_2 over spatial axes"
            ),
            "rmse_normalized_definition": np.asarray(
                "sqrt(mean((prediction-target)^2)) over spatial axes"
            ),
            "physical_error_units": np.asarray(
                "Temperature errors are degC; pressure errors are kPa"
            ),
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
        description="Evaluate full autoregressive rollouts for one or more configs.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="YAML paths or globs. Default: every top-level configs/*.yml.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help="Checkpoint seeds (default: 5 42 2026).",
    )
    parser.add_argument("--start-index", type=int, default=300)
    parser.add_argument("--end-index", type=int, default=399)
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size <= 0 or args.end_index < args.start_index:
        raise ValueError("Invalid --batch-size or trajectory index range")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"--seeds contains duplicates: {args.seeds}")

    config_paths = expand_config_paths(args.configs)
    checkpoint_root = (
        resolve_path(args.checkpoint_root) if args.checkpoint_root else None
    )
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_root = resolve_path(args.output_root)
    trajectory_indices = list(range(args.start_index, args.end_index + 1))

    print(f"Device: {device}")
    print(f"Trajectories: {args.start_index}-{args.end_index}")
    print(f"Seeds: {args.seeds}")
    print(f"Configs: {len(config_paths)}")
    if checkpoint_root is not None:
        print(f"Checkpoint root: {checkpoint_root}")

    for config_number, config_path in enumerate(config_paths, start=1):
        config = read_config(config_path)
        output_name = config_path.stem
        checkpoint_paths = [
            resolve_checkpoint_path(config, seed, checkpoint_root)
            for seed in args.seeds
        ]
        missing = [path for path in checkpoint_paths if not path.is_file()]
        if missing:
            details = "\n".join(f"  {path}" for path in missing)
            raise FileNotFoundError(f"Missing checkpoints:\n{details}")

        print(f"\n[{config_number}/{len(config_paths)}] {output_name} "
              f"({config['model']['type']})")
        dataset, data_path = load_dataset(config)
        if args.end_index >= dataset.n_trajectories:
            raise IndexError(
                f"{config_path}: end index {args.end_index} outside "
                f"{dataset.n_trajectories} trajectories"
            )

        config_metrics = {
            name: np.empty(
                (len(args.seeds), len(trajectory_indices), dataset.n_timesteps,
                 len(CHANNEL_NAMES)),
                dtype=np.float32,
            )
            for name in METRIC_NAMES
        }

        for seed_index, (seed, checkpoint_path) in enumerate(
            zip(args.seeds, checkpoint_paths)
        ):
            print(f"  Seed {seed}: {checkpoint_path}", flush=True)
            model = None
            try:
                model, adapter = load_model(
                    config, checkpoint_path, device, args.weight_key,
                )
                seed_metrics = evaluate_checkpoint(
                    model, adapter, dataset, trajectory_indices,
                    args.batch_size, device,
                )
                for name in METRIC_NAMES:
                    config_metrics[name][seed_index] = seed_metrics[name]
            finally:
                del model
                release_device_memory(device)

        output_path = output_root / output_name / "metrics.npz"
        save_metrics(
            output_path, config_metrics, config_path, config, checkpoint_paths,
            args.seeds, trajectory_indices, data_path,
        )
        print(f"  Saved {output_path}")
        del config_metrics
        del dataset
        release_device_memory(device)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
