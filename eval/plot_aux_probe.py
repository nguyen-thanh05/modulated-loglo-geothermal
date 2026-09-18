"""Probe saved aux-rollout metrics: field L2 envelopes and aux vs ground truth.

Reads ``evaluation_results/rollouts_aux/<run>/metrics.npz`` (no GPU / checkpoint).
Default output: ``<run>/probe/`` so the full 100-trajectory archive stays the
source of truth and later probing can point at the same file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = (
    REPO_ROOT / "evaluation_results" / "rollouts_aux"
    / "modulated_loglo_aux" / "metrics.npz"
)
CHANNEL_COLORS = ("tab:red", "tab:orange", "tab:blue", "tab:green")
CHANNEL_UNITS_PHYSICAL = ("degC", "degC", "kPa", "kPa")
N_PRODUCER_WELLS = 7
WELL_COORDS = (
    (31, 15), (45, 4), (56, 15), (45, 27), (18, 27),
    (4, 15), (18, 4), (18, 15), (45, 15),
)


def resolve_path(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_metrics(metrics_path):
    with np.load(metrics_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["channel_names"] = np.asarray(payload["channel_names"]).astype(str)
    payload["bhp_names"] = np.asarray(payload["bhp_names"]).astype(str)
    payload["energy_names"] = np.asarray(payload["energy_names"]).astype(str)
    payload["trajectory_indices"] = np.asarray(
        payload["trajectory_indices"], dtype=np.int64,
    )
    return payload


def trajectory_scores(payload):
    relative_l2 = payload["field_relative_l2"]
    return {
        "mean_relative_l2": relative_l2.mean(axis=(1, 2)),
        "mean_temperature_l2": relative_l2[..., :2].mean(axis=(1, 2)),
        "mean_pressure_l2": relative_l2[..., 2:].mean(axis=(1, 2)),
        "peak_pressure_l2": relative_l2[..., 2:].max(axis=(1, 2)),
        "final_pressure_l2": relative_l2[:, -1, 2:].mean(axis=1),
        "mean_bhp_rmse": payload["bhp_rmse_physical"].mean(axis=(1, 2)),
        "mean_energy_rmse": payload["energy_rmse_physical"].mean(axis=(1, 2)),
    }


def select_probe_indices(payload, n_select=8):
    """Spread along field L2, then add pressure / aux outliers."""
    scores = trajectory_scores(payload)
    mean_l2 = scores["mean_relative_l2"]
    n_trajectories = mean_l2.shape[0]
    order = np.argsort(mean_l2)

    def percentile_index(percent):
        return int(order[int(round(percent / 100.0 * (n_trajectories - 1)))])

    selected = []
    reasons = {}

    def add(index, reason):
        index = int(index)
        if index not in selected:
            selected.append(index)
            reasons[index] = reason

    add(percentile_index(0), "best mean field relative L2")
    add(percentile_index(25), "25th-percentile mean field relative L2")
    add(percentile_index(50), "median mean field relative L2")
    add(percentile_index(75), "75th-percentile mean field relative L2")
    add(percentile_index(100), "worst mean field relative L2")

    for index in np.argsort(scores["peak_pressure_l2"])[::-1]:
        if int(index) not in selected:
            add(int(index), "pressure relative-L2 spike")
            break

    add(int(np.argmax(scores["mean_energy_rmse"])), "worst mean energy RMSE")

    z_l2 = (mean_l2 - mean_l2.mean()) / (mean_l2.std() + 1e-12)
    z_bhp = (
        (scores["mean_bhp_rmse"] - scores["mean_bhp_rmse"].mean())
        / (scores["mean_bhp_rmse"].std() + 1e-12)
    )
    add(int(np.argmax(z_bhp - z_l2)), "high BHP RMSE relative to field L2")

    for percent, label in (
        (10, "10th-percentile mean field relative L2"),
        (90, "90th-percentile mean field relative L2"),
    ):
        if len(selected) >= n_select:
            break
        add(percentile_index(percent), label)

    selected = selected[:n_select]
    return selected, {index: reasons[index] for index in selected}


def resolve_requested_indices(payload, trajectory_ids):
    lookup = {
        int(traj): position
        for position, traj in enumerate(payload["trajectory_indices"])
    }
    missing = [int(traj) for traj in trajectory_ids if int(traj) not in lookup]
    if missing:
        raise KeyError(f"Trajectories not in metrics.npz: {missing}")
    indices = [lookup[int(traj)] for traj in trajectory_ids]
    reasons = {
        index: "user-specified" for index in indices
    }
    return indices, reasons


def envelope(values, axis=0):
    return (
        values.mean(axis=axis),
        values.min(axis=axis),
        values.max(axis=axis),
        np.quantile(values, 0.25, axis=axis),
        np.quantile(values, 0.75, axis=axis),
    )


def plot_channel_envelopes(relative_l2, channel_names, title, output_path,
                           overlay=None, overlay_ids=None, show_iqr=True):
    timesteps = np.arange(relative_l2.shape[1])
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for channel, axis in enumerate(axes.ravel()):
        mean, lower, upper, q25, q75 = envelope(relative_l2[:, :, channel])
        color = CHANNEL_COLORS[channel]
        axis.fill_between(
            timesteps, lower, upper, color=color, alpha=0.12, label="min–max",
        )
        if show_iqr:
            axis.fill_between(
                timesteps, q25, q75, color=color, alpha=0.28, label="IQR",
            )
        axis.plot(timesteps, mean, color=color, linewidth=2.2, label="mean")
        if overlay is not None:
            cmap = plt.get_cmap("tab10")
            for row, traj in enumerate(overlay_ids):
                axis.plot(
                    timesteps,
                    overlay[row, :, channel],
                    color=cmap(row % cmap.N),
                    linewidth=1.1,
                    alpha=0.85,
                    label=f"traj {int(traj)}",
                )
        axis.set_title(channel_names[channel])
        axis.set_ylabel("Relative L2")
        axis.set_xlabel("week")
        axis.set_ylim(bottom=0.0)
        axis.grid(True, alpha=0.25)
    figure.suptitle(title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if overlay is None:
        axes[0, 0].legend(handles, labels, frameon=False, fontsize=8)
        figure.tight_layout()
    else:
        figure.legend(
            handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8,
        )
        figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_per_traj_channel_envelope(relative_l2, trajectory_ids, output_path):
    n_plot = len(trajectory_ids)
    n_cols = 2
    n_rows = int(np.ceil(n_plot / n_cols))
    timesteps = np.arange(relative_l2.shape[1])
    figure, axes = plt.subplots(
        n_rows, n_cols, figsize=(12, 3.4 * n_rows), sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for row, (axis, traj) in enumerate(zip(axes, trajectory_ids)):
        values = relative_l2[row]
        mean = values.mean(axis=1)
        lower = values.min(axis=1)
        upper = values.max(axis=1)
        axis.fill_between(
            timesteps, lower, upper, color="tab:purple", alpha=0.18,
            label="min–max across channels",
        )
        axis.plot(
            timesteps, mean, color="tab:purple", linewidth=2.0,
            label="mean across channels",
        )
        axis.set_title(f"Trajectory {int(traj)}")
        axis.set_ylabel("Relative L2")
        axis.set_xlabel("week")
        axis.set_ylim(bottom=0.0)
        axis.grid(True, alpha=0.25)
        if row == 0:
            axis.legend(frameon=False, fontsize=8)
    for axis in axes[n_plot:]:
        axis.axis("off")
    figure.suptitle(
        "Per-trajectory relative L2: channel mean with min–max band"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_traj_l2(traj_l2, channel_names, traj_id, output_path, spatial=None):
    timesteps = np.arange(traj_l2.shape[0])
    if spatial is None:
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for channel, axis in enumerate(axes.ravel()):
            axis.plot(
                timesteps, traj_l2[:, channel], color=CHANNEL_COLORS[channel],
                linewidth=2.2,
            )
            axis.set_title(channel_names[channel])
            axis.set_ylabel("Relative L2")
            axis.set_xlabel("week")
            axis.set_ylim(bottom=0.0)
            axis.grid(True, alpha=0.25)
        figure.suptitle(f"Trajectory {int(traj_id)} field relative L2")
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return

    mae = spatial["mae_physical"]
    max_abs = spatial["max_abs_physical"]
    maps = spatial["maps_at_peak_physical"]
    peak_weeks = spatial["peak_weeks"]
    units = spatial["channel_units_physical"]
    figure, axes = plt.subplots(4, 2, figsize=(13, 16), constrained_layout=True)
    for channel in range(4):
        curve_ax = axes[channel, 0]
        map_ax = axes[channel, 1]
        color = CHANNEL_COLORS[channel]
        peak_week = int(peak_weeks[channel])
        ratio = float(max_abs[peak_week, channel] / max(mae[peak_week, channel], 1e-12))

        curve_ax.plot(
            timesteps, traj_l2[:, channel], color=color, linewidth=2.2,
            label="relative L2",
        )
        curve_ax.set_ylabel("Relative L2")
        curve_ax.set_xlabel("week")
        curve_ax.set_ylim(bottom=0.0)
        curve_ax.grid(True, alpha=0.25)
        twin = curve_ax.twinx()
        twin.plot(
            timesteps, mae[:, channel], color="0.25", linestyle="--",
            linewidth=1.6, label="spatial MAE",
        )
        twin.plot(
            timesteps, max_abs[:, channel], color="0.05", linestyle=":",
            linewidth=2.4, label="spatial max |e|",
        )
        twin.axvline(peak_week, color="0.4", linewidth=0.8, alpha=0.7)
        twin.set_ylabel(f"|e| ({units[channel]})")
        twin.set_yscale("log")
        ymin = float(np.nanmin(mae[:, channel]))
        ymax = float(np.nanmax(max_abs[:, channel]))
        twin.set_ylim(bottom=max(ymin * 0.8, 1e-6), top=ymax * 1.3)
        curve_ax.set_title(
            f"{channel_names[channel]}  ·  max/MAE={ratio:.1f} at week {peak_week}"
        )
        handles_left, labels_left = curve_ax.get_legend_handles_labels()
        handles_right, labels_right = twin.get_legend_handles_labels()
        if channel == 0:
            curve_ax.legend(
                handles_left + handles_right,
                labels_left + labels_right,
                frameon=False,
                fontsize=8,
                loc="lower right",
            )

        vmax = float(np.max(maps[channel])) or 1.0
        image = map_ax.imshow(
            maps[channel], origin="upper", cmap="magma", aspect="equal",
            vmin=0.0, vmax=vmax,
        )
        well_handle = None
        for height_index, width_index in WELL_COORDS:
            well_handle = map_ax.scatter(
                width_index, height_index, marker="x", c="cyan", s=22,
                linewidths=0.9,
            )
        map_ax.set_title(f"|e| map at week {peak_week} (max over depth)")
        map_ax.set_xlabel("x")
        map_ax.set_ylabel("y")
        colorbar = figure.colorbar(image, ax=map_ax, fraction=0.046, pad=0.04)
        colorbar.set_label(units[channel])
        if channel == 0 and well_handle is not None:
            map_ax.legend(
                [well_handle], ["wells"], frameon=False, fontsize=8, loc="upper right",
            )

    figure.suptitle(
        f"Trajectory {int(traj_id)} field error: relative L2 vs spatial MAE / max",
        fontsize=13,
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_traj_l2_channel_envelope(traj_l2, channel_names, traj_id, output_path):
    timesteps = np.arange(traj_l2.shape[0])
    figure, axis = plt.subplots(figsize=(8, 4.5))
    mean = traj_l2.mean(axis=1)
    axis.fill_between(
        timesteps, traj_l2.min(axis=1), traj_l2.max(axis=1),
        color="tab:purple", alpha=0.18, label="min–max across channels",
    )
    axis.plot(
        timesteps, mean, color="tab:purple", linewidth=2.2,
        label="mean across channels",
    )
    for channel, name in enumerate(channel_names):
        axis.plot(
            timesteps, traj_l2[:, channel], color=CHANNEL_COLORS[channel],
            linewidth=1.1, alpha=0.85, label=name,
        )
    axis.set_title(f"Trajectory {int(traj_id)} relative L2")
    axis.set_ylabel("Relative L2")
    axis.set_xlabel("week")
    axis.set_ylim(bottom=0.0)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_traj_l2_against_population(traj_l2, population_l2, channel_names,
                                    traj_id, output_path):
    timesteps = np.arange(traj_l2.shape[0])
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for channel, axis in enumerate(axes.ravel()):
        mean, lower, upper, q25, q75 = envelope(population_l2[:, :, channel])
        color = CHANNEL_COLORS[channel]
        axis.fill_between(
            timesteps, lower, upper, color="0.75", alpha=0.35, label="all min–max",
        )
        axis.fill_between(
            timesteps, q25, q75, color="0.55", alpha=0.25, label="all IQR",
        )
        axis.plot(
            timesteps, mean, color="0.35", linewidth=1.4, linestyle="--",
            label="all mean",
        )
        axis.plot(
            timesteps, traj_l2[:, channel], color=color, linewidth=2.2,
            label=f"traj {int(traj_id)}",
        )
        axis.set_title(channel_names[channel])
        axis.set_ylabel("Relative L2")
        axis.set_xlabel("week")
        axis.set_ylim(bottom=0.0)
        axis.grid(True, alpha=0.25)
        if channel == 0:
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle(f"Trajectory {int(traj_id)} field relative L2")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_bhp(gt, pred, well_names, traj_id, output_path):
    timesteps = np.arange(gt.shape[0])
    figure, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    for well, axis in enumerate(axes.ravel()):
        axis.plot(timesteps, gt[:, well], color="black", linestyle="--", label="GT")
        axis.plot(timesteps, pred[:, well], color="tab:blue", label="Pred")
        axis.set_title(f"BHP — {well_names[well]}")
        axis.set_ylabel("kPa")
        if well == 0:
            axis.legend(frameon=False, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("week")
    figure.suptitle(f"Trajectory {int(traj_id)} bottomhole pressure")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_energy(gt, pred, well_names, traj_id, output_path):
    timesteps = np.arange(gt.shape[0])
    figure, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    for well, axis in enumerate(axes.ravel()):
        if well >= N_PRODUCER_WELLS:
            axis.axis("off")
            continue
        axis.plot(timesteps, gt[:, well], color="black", linestyle="--", label="GT")
        axis.plot(timesteps, pred[:, well], color="tab:orange", label="Pred")
        axis.set_title(f"Energy — {well_names[well]}")
        axis.set_ylabel("J/day")
        if well == 0:
            axis.legend(frameon=False, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("week")
    figure.suptitle(f"Trajectory {int(traj_id)} producer energy rate")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_aux_rmse_envelope(bhp_rmse, energy_rmse, title, output_path,
                           overlay_bhp=None, overlay_energy=None,
                           overlay_ids=None):
    timesteps = np.arange(bhp_rmse.shape[1])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    series = (
        (bhp_rmse, "BHP RMSE (kPa)", "tab:blue", overlay_bhp),
        (energy_rmse, "Energy RMSE (J/day)", "tab:orange", overlay_energy),
    )
    for axis, (values, ylabel, color, overlay) in zip(axes, series):
        mean, lower, upper, q25, q75 = envelope(values.mean(axis=2))
        axis.fill_between(timesteps, lower, upper, color=color, alpha=0.12)
        axis.fill_between(timesteps, q25, q75, color=color, alpha=0.28)
        axis.plot(timesteps, mean, color=color, linewidth=2.0, label="mean")
        if overlay is not None:
            cmap = plt.get_cmap("tab10")
            for row, traj in enumerate(overlay_ids):
                axis.plot(
                    timesteps,
                    overlay[row].mean(axis=1),
                    color=cmap(row % cmap.N),
                    linewidth=1.1,
                    alpha=0.85,
                    label=f"traj {int(traj)}",
                )
        axis.set_title(ylabel.split(" (")[0])
        axis.set_ylabel(ylabel)
        axis.set_xlabel("week")
        axis.set_ylim(bottom=0.0)
        axis.grid(True, alpha=0.25)
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overlay_ids is None:
        figure.tight_layout()
    else:
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8,
        )
        figure.tight_layout(rect=(0.0, 0.14, 1.0, 0.95))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def well_name_from_aux(aux_name):
    return aux_name.split("_", 1)[1]


def write_summary(path, payload, selected, reasons):
    scores = trajectory_scores(payload)
    trajs = payload["trajectory_indices"]
    field = payload["field_relative_l2"]
    rows = []
    for index in selected:
        channel_means = field[index].mean(axis=0)
        rows.append({
            "trajectory": int(trajs[index]),
            "reason": reasons[index],
            "mean_relative_l2": float(scores["mean_relative_l2"][index]),
            "mean_temperature_l2": float(scores["mean_temperature_l2"][index]),
            "mean_pressure_l2": float(scores["mean_pressure_l2"][index]),
            "peak_pressure_l2": float(scores["peak_pressure_l2"][index]),
            "final_pressure_l2": float(scores["final_pressure_l2"][index]),
            "mean_bhp_rmse_kPa": float(scores["mean_bhp_rmse"][index]),
            "mean_energy_rmse_Jday": float(scores["mean_energy_rmse"][index]),
            "mean_l2_T_form": float(channel_means[0]),
            "mean_l2_T_frac": float(channel_means[1]),
            "mean_l2_P_form": float(channel_means[2]),
            "mean_l2_P_frac": float(channel_means[3]),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    selection = {
        "seed": int(np.asarray(payload["seed"]).item()),
        "metrics_path": str(payload.get("_metrics_path", "")),
        "checkpoint_path": str(payload["checkpoint_path"]),
        "n_trajectories_in_archive": int(len(trajs)),
        "trajectory_range": [int(trajs[0]), int(trajs[-1])],
        "selected": rows,
    }
    json_path = path.with_name("selection.json")
    json_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return rows


def save_selected_metrics(path, payload, selected):
    keep = {}
    n_all = payload["field_relative_l2"].shape[0]
    for key, value in payload.items():
        if key.startswith("_"):
            continue
        array = np.asarray(value)
        if array.shape[:1] == (n_all,):
            keep[key] = array[selected]
        else:
            keep[key] = array
    keep["probe_source_indices"] = np.asarray(selected, dtype=np.int64)
    np.savez_compressed(path, **keep)


def load_rollout_aux_module():
    import importlib.util

    path = REPO_ROOT / "eval" / "rollout_aux.py"
    spec = importlib.util.spec_from_file_location("rollout_aux_mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_spatial_cache(path, trajectory_ids):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        cached_ids = np.asarray(archive["trajectory_indices"], dtype=np.int64)
        if not np.array_equal(cached_ids, np.asarray(trajectory_ids, dtype=np.int64)):
            return None
        return {key: archive[key] for key in archive.files}


def collect_spatial_errors(payload, trajectory_ids, cache_path, refresh, device_name):
    cached = None if refresh else load_spatial_cache(cache_path, trajectory_ids)
    if cached is not None:
        print(f"Loaded spatial diagnostics {cache_path}")
        return cached

    import torch

    rollout_aux = load_rollout_aux_module()
    config_path = rollout_aux.resolve_path(str(payload["config_path"]))
    config = rollout_aux.read_config(config_path)
    checkpoint_path = Path(str(payload["checkpoint_path"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    device = torch.device(
        device_name if device_name else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    print(f"Computing spatial MAE / max |e| on {device} for {trajectory_ids}")
    dataset, _data_path = rollout_aux.load_dataset(config)
    model = None
    try:
        model, adapter = rollout_aux.load_model(
            config, checkpoint_path, device, "ema_model",
        )
        spatial = rollout_aux.evaluate_error_maps(
            model, adapter, dataset, list(trajectory_ids), device,
        )
    finally:
        del model
        rollout_aux.release_device_memory(device)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **spatial)
    print(f"Saved {cache_path}")
    return spatial


def spatial_for_trajectory(spatial, row):
    return {
        "mae_physical": spatial["mae_physical"][row],
        "max_abs_physical": spatial["max_abs_physical"][row],
        "maps_at_peak_physical": spatial["maps_at_peak_physical"][row],
        "peak_weeks": spatial["peak_weeks"][row],
        "channel_units_physical": np.asarray(
            spatial["channel_units_physical"]
        ).astype(str),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot field L2 envelopes and aux vs GT from saved "
                    "aux-rollout metrics.npz.",
    )
    parser.add_argument("--metrics", type=str, default=str(DEFAULT_METRICS))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Default: <metrics-parent>/probe",
    )
    parser.add_argument(
        "--trajectories",
        nargs="+",
        type=int,
        default=None,
        help="Trajectory IDs to plot. Default: auto-select a diverse subset.",
    )
    parser.add_argument("--n-select", type=int, default=8)
    parser.add_argument(
        "--refresh-spatial",
        action="store_true",
        help="Recompute spatial MAE / max |e| even if spatial_errors.npz exists.",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    metrics_path = resolve_path(args.metrics)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics: {metrics_path}")
    payload = load_metrics(metrics_path)
    payload["_metrics_path"] = str(metrics_path)

    output_dir = (
        resolve_path(args.output_dir) if args.output_dir
        else metrics_path.parent / "probe"
    )
    ensemble_dir = output_dir / "ensemble"

    if args.trajectories:
        selected, reasons = resolve_requested_indices(payload, args.trajectories)
    else:
        selected, reasons = select_probe_indices(payload, args.n_select)

    trajs = payload["trajectory_indices"]
    selected_ids = [int(trajs[index]) for index in selected]
    relative_l2 = payload["field_relative_l2"]
    channel_names = payload["channel_names"]
    bhp_names = [well_name_from_aux(name) for name in payload["bhp_names"]]
    energy_names = [well_name_from_aux(name) for name in payload["energy_names"]]

    write_summary(output_dir / "summary.csv", payload, selected, reasons)
    save_selected_metrics(output_dir / "selected_metrics.npz", payload, selected)
    spatial = collect_spatial_errors(
        payload,
        selected_ids,
        output_dir / "spatial_errors.npz",
        args.refresh_spatial,
        args.device,
    )

    plot_channel_envelopes(
        relative_l2,
        channel_names,
        "All test trajectories — field relative L2 (mean, IQR, min–max)",
        ensemble_dir / "field_relative_l2_all.png",
    )
    plot_channel_envelopes(
        relative_l2[selected],
        channel_names,
        "Selected trajectories — field relative L2 (mean, IQR, min–max)",
        ensemble_dir / "field_relative_l2_selected.png",
        overlay=relative_l2[selected],
        overlay_ids=selected_ids,
        show_iqr=True,
    )
    plot_per_traj_channel_envelope(
        relative_l2[selected],
        selected_ids,
        ensemble_dir / "per_traj_l2_channel_envelope.png",
    )
    plot_aux_rmse_envelope(
        payload["bhp_rmse_physical"],
        payload["energy_rmse_physical"],
        "All test trajectories — aux RMSE (mean, IQR, min–max across trajs)",
        ensemble_dir / "aux_rmse_all.png",
    )
    plot_aux_rmse_envelope(
        payload["bhp_rmse_physical"][selected],
        payload["energy_rmse_physical"][selected],
        "Selected trajectories — aux RMSE",
        ensemble_dir / "aux_rmse_selected.png",
        overlay_bhp=payload["bhp_rmse_physical"][selected],
        overlay_energy=payload["energy_rmse_physical"][selected],
        overlay_ids=selected_ids,
    )

    for row, (index, traj) in enumerate(zip(selected, selected_ids)):
        traj_dir = output_dir / f"traj{traj}"
        traj_dir.mkdir(parents=True, exist_ok=True)
        plot_traj_l2(
            relative_l2[index],
            channel_names,
            traj,
            traj_dir / "field_relative_l2.png",
            spatial=spatial_for_trajectory(spatial, row),
        )
        plot_traj_l2_channel_envelope(
            relative_l2[index],
            channel_names,
            traj,
            traj_dir / "field_relative_l2_envelope.png",
        )
        plot_traj_l2_against_population(
            relative_l2[index],
            relative_l2,
            channel_names,
            traj,
            traj_dir / "field_relative_l2_vs_all.png",
        )
        plot_bhp(
            payload["gt_bhp_physical"][index],
            payload["pred_bhp_physical"][index],
            bhp_names,
            traj,
            traj_dir / "bhp.png",
        )
        plot_energy(
            payload["gt_energy_physical"][index],
            payload["pred_energy_physical"][index],
            energy_names,
            traj,
            traj_dir / "energy.png",
        )

    print(f"Metrics: {metrics_path}")
    print(f"Selected trajectories: {selected_ids}")
    for index, traj in zip(selected, selected_ids):
        print(f"  {traj}: {reasons[index]}")
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
