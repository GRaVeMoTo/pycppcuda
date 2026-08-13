from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"


def latest_results_file() -> Path:
    csv_files = sorted(OUTPUT_DIR.glob("run_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    for csv_path in csv_files:
        if csv_path.stat().st_size > 0:
            return csv_path
    raise FileNotFoundError(f"No non-empty run CSV files found in {OUTPUT_DIR}")


def main() -> None:
    csv_path = latest_results_file()
    results = pd.read_csv(csv_path)
    results["image_size"] = results["width"].astype(str) + "x" + results["height"].astype(str)

    averages = results.groupby(["image_size", "radius", "type"], as_index=False)["seconds"].mean()
    methods = sorted(averages["type"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 6), squeeze=False)

    for axis, method in zip(axes[0], methods, strict=True):
        method_results = averages[averages["type"] == method]
        heatmap_data = method_results.pivot(index="image_size", columns="radius", values="seconds")
        sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlOrRd", cbar_kws={"label": "Average runtime (s)"}, ax=axis)
        axis.set_title(f"{method}: average runtime")
        axis.set_xlabel("Blur radius")
        axis.set_ylabel("Image size (width x height)")

    fig.suptitle(f"Average blur runtime by image size - {csv_path.stem}", y=1.02)
    fig.tight_layout()
    output_path = OUTPUT_DIR / f"{csv_path.stem}_runtime_heatmap.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Heatmap saved: {output_path}")


if __name__ == "__main__":
    main()
