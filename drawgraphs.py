import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from config import Config

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"
FIGURE_BACKGROUND = "#101923"
AXIS_BACKGROUND = "#172534"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#78909c"


def latest_results_file() -> Path:
    csv_files = sorted(OUTPUT_DIR.glob("run_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    for csv_path in csv_files:
        if csv_path.stat().st_size > 0:
            return csv_path
    raise FileNotFoundError(f"No non-empty run CSV files found in {OUTPUT_DIR}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a 3D blur benchmark chart.")
    parser.add_argument(
        "-csv",
        type=Path,
        help="Benchmark CSV file. Defaults to the latest output/run_*.csv file.",
    )
    parser.add_argument(
        "-output",
        type=Path,
        help="Output image path. Defaults to output/<csv-name>_runtime_heatmap_3d.png.",
    )
    return parser.parse_args()


def resolution_labels() -> tuple[list[str], dict[tuple[int, int], str]]:
    labels = []
    resolution_by_size = {}
    for width, height, label in Config.RESOLUTIONS:
        labels.append(label)
        resolution_by_size[(width, height)] = label
    return labels, resolution_by_size


def main() -> None:
    arguments = parse_arguments()
    csv_path = arguments.csv or latest_results_file()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    results = pd.read_csv(csv_path)
    labels, resolution_by_size = resolution_labels()
    results["resolution"] = [resolution_by_size.get((width, height)) for width, height in zip(results["width"], results["height"], strict=True)]
    results = results.dropna(subset=["resolution"])
    if results.empty:
        raise ValueError("The results file does not contain resolutions from Config.RESOLUTIONS")

    results["milliseconds"] = results["seconds"] * 1_000
    averages = results.groupby(["resolution", "radius", "type"], as_index=False)["milliseconds"].mean()
    method_order = ("pt_gpu", "pt_gpu_opt", "cuda", "cuda_opt")
    method_title = ("pyTorch GPU 2D", "pyTorch GPU 2 pass H/V", "CUDA native 2D", "CUDA native 2 pass H/V")
    methods = [method for method in method_order if method in averages["type"].unique()]
    if not methods:
        raise ValueError("The results file does not contain supported GPU methods")

    radii = sorted(averages["radius"].unique())
    scale_min = averages["milliseconds"].min()
    scale_max = averages["milliseconds"].max()
    color_normalization = Normalize(vmin=scale_min, vmax=scale_max)
    fig = plt.figure(figsize=(14, 12), facecolor=FIGURE_BACKGROUND)

    for index, method in enumerate(methods, start=1):
        axis = fig.add_subplot(2, 2, index, projection="3d")
        axis.set_facecolor(AXIS_BACKGROUND)
        axis.xaxis.pane.set_facecolor(AXIS_BACKGROUND)
        axis.yaxis.pane.set_facecolor(AXIS_BACKGROUND)
        axis.zaxis.pane.set_facecolor(AXIS_BACKGROUND)
        axis.xaxis.pane.set_edgecolor(GRID_COLOR)
        axis.yaxis.pane.set_edgecolor(GRID_COLOR)
        axis.zaxis.pane.set_edgecolor(GRID_COLOR)
        axis.tick_params(colors=TEXT_COLOR)
        axis.xaxis.label.set_color(TEXT_COLOR)
        axis.yaxis.label.set_color(TEXT_COLOR)
        axis.zaxis.label.set_color(TEXT_COLOR)
        method_results = averages[averages["type"] == method]
        heatmap_data = method_results.pivot(index="resolution", columns="radius", values="milliseconds")
        heatmap_data = heatmap_data.reindex(index=labels, columns=radii)
        x, y = np.meshgrid(np.arange(len(labels)), np.arange(len(radii)))
        z = heatmap_data.to_numpy(dtype=float).T
        axis.plot_surface(
            x,
            y,
            z,
            cmap="viridis",
            norm=color_normalization,
            edgecolor="none",
        )
        axis.scatter(x, y, z, color="#dbe9f4", s=8, depthshade=False)
        axis.set_title(method_title[index - 1], color=TEXT_COLOR, fontsize=18)
        axis.set_xlabel("Resolution")
        axis.set_ylabel("Radius px")
        axis.set_zlim(scale_min, scale_max)
        axis.set_xticks(range(len(labels)), labels)
        axis.set_yticks(range(len(radii)), radii)
        axis.tick_params(axis="x", pad=0)
        for tick_label in axis.get_xticklabels():
            tick_label.set_horizontalalignment("left")
            tick_label.set_color(TEXT_COLOR)
        for tick_label in axis.get_yticklabels() + axis.get_zticklabels():
            tick_label.set_color(TEXT_COLOR)
        axis.set_proj_type("ortho")
        axis.set_box_aspect((len(labels), len(radii), 4.0))
        axis.view_init(elev=30, azim=-130)

    colorbar_axis = fig.add_axes((0.48, 0.2, 0.014, 0.6))
    colorbar = fig.colorbar(
        ScalarMappable(norm=color_normalization, cmap="viridis"),
        cax=colorbar_axis,
    )
    colorbar.ax.set_title("ms", color=TEXT_COLOR, pad=10)
    colorbar.ax.tick_params(colors=TEXT_COLOR)
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_rotation(45)
        tick_label.set_rotation_mode("anchor")
        tick_label.set_verticalalignment("center")

    colorbar.outline.set_edgecolor(GRID_COLOR)
    fig.suptitle("Average runtime by Resolution & Radius", y=1.0, color=TEXT_COLOR, fontsize=24)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.1, top=0.9, wspace=0.0, hspace=0.0)
    output_path = arguments.output or OUTPUT_DIR / f"{csv_path.stem}_runtime_heatmap_3d.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"3D heatmap saved: {output_path}")


if __name__ == "__main__":
    main()
