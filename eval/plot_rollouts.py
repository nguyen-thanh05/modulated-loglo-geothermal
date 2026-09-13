"""Plot rollout errors from saved evaluation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "evaluation_results" / "rollouts"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation_results" / "plots"

ARCHITECTURE_LABELS = {
    "modulated_loglo": "Modulated LOGLO-FNO",
    "vanilla_loglo": "Vanilla LOGLO-FNO",
    "fno_m4x16x8_h128": "FNO (m4x16x8, h128)",
    "fno_m8x32x16_h64": "FNO (m8x32x16, h64)",
    "ufno": "U-FNO",
    "unet": "U-Net",
}
ARCHITECTURE_ORDER = tuple(ARCHITECTURE_LABELS)


def resolve_path(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def architecture_sort_key(path):
    name = path.parent.name
    try:
        return ARCHITECTURE_ORDER.index(name), name
    except ValueError:
        return len(ARCHITECTURE_ORDER), name


def load_results(results_root):
    metric_paths = sorted(
        results_root.glob("*/metrics.npz"),
        key=architecture_sort_key,
    )
    if not metric_paths:
        raise FileNotFoundError(
            f"No architecture metrics under {results_root}; expected "
            "<architecture>/metrics.npz"
        )

    results = []
    for path in metric_paths:
        with np.load(path, allow_pickle=False) as archive:
            name = path.parent.name
            results.append({
                "name": name,
                "label": ARCHITECTURE_LABELS.get(name, name.replace("_", " ")),
                "path": path,
                "seeds": np.asarray(archive["seeds"]),
                "trajectory_indices": np.asarray(archive["trajectory_indices"]),
                "rollout_timesteps": np.asarray(archive["rollout_timesteps"]),
                "channel_names": np.asarray(archive["channel_names"]).astype(str),
                "channel_units_physical": np.asarray(
                    archive["channel_units_physical"]
                ).astype(str),
                "relative_l2": np.asarray(archive["relative_l2"]),
            })
    return results


def seed_mean_curve(values):
    """Mean over trajectories per seed, then mean/min/max across seeds."""
    per_seed = values.mean(axis=1, dtype=np.float64)
    return per_seed.mean(axis=0), per_seed.min(axis=0), per_seed.max(axis=0)


def plot_vs_timestep(results, output_dir):
    reference = results[0]
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    for channel_index, axis in enumerate(axes.ravel()):
        for architecture_index, result in enumerate(results):
            mean, lower, upper = seed_mean_curve(result["relative_l2"])
            color = colors(architecture_index % colors.N)
            highlighted = result["name"] == "modulated_loglo"
            axis.plot(
                reference["rollout_timesteps"],
                mean[:, channel_index],
                color=color,
                linewidth=2.8 if highlighted else 1.7,
                label=result["label"],
            )
            axis.fill_between(
                reference["rollout_timesteps"],
                lower[:, channel_index],
                upper[:, channel_index],
                color=color,
                alpha=0.20 if highlighted else 0.12,
                linewidth=0.0,
            )
        axis.set_title(reference["channel_names"][channel_index])
        axis.set_ylabel("Relative L2 error")
        axis.set_xlabel("Rollout timestep")
        axis.grid(True, alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Autoregressive relative L2 error", fontsize=15)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))

    output_path = output_dir / "relative_l2_vs_timestep.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_final_box(results, output_dir):
    reference = results[0]
    last_timestep = int(reference["rollout_timesteps"][-1])
    colors = plt.get_cmap("tab10")
    labels = [result["label"] for result in results]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for channel_index, axis in enumerate(axes.ravel()):
        samples = [
            result["relative_l2"][:, :, -1, channel_index].ravel()
            for result in results
        ]
        box = axis.boxplot(
            samples, tick_labels=labels, patch_artist=True, showfliers=False,
        )
        for architecture_index, patch in enumerate(box["boxes"]):
            color = colors(architecture_index % colors.N)
            highlighted = results[architecture_index]["name"] == "modulated_loglo"
            patch.set_facecolor(color)
            patch.set_alpha(0.70 if highlighted else 0.55)
        for median in box["medians"]:
            median.set_color("black")
        axis.set_title(reference["channel_names"][channel_index])
        axis.set_ylabel("Relative L2 error")
        axis.set_ylim(bottom=0.0)
        axis.tick_params(axis="x", labelrotation=25)
        axis.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"Relative L2 at final timestep t={last_timestep}", fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    output_path = output_dir / "relative_l2_final_timestep_box.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot architecture comparison from rollout metrics.npz files.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_root = resolve_path(args.results_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(results_root)
    plot_vs_timestep(results, output_dir)
    plot_final_box(results, output_dir)


if __name__ == "__main__":
    main()
