import csv
from datetime import UTC, datetime
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from blurs import CudaBlur, CudaOptimizedBlur, CpuPyTorchBlur, GpuPyTorchBlur, GpuPyTorchOptimizedBlur
from config import BlurInterface, Config

ROOT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = ROOT_DIR / "images"
OUTPUT_DIR = ROOT_DIR / "output"


def image_paths() -> list[Path]:
    return sorted(path for path in IMAGES_DIR.iterdir() if path.suffix.lower() in Config.IMAGE_SUFFIXES)


def write_stat_header(csv_file) -> csv.DictWriter:
    if not csv_file:
        raise ValueError("csv_file required!")

    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "image",
            "width",
            "height",
            "radius",
            "iteration",
            "type",
            "seconds",
        ],
    )
    writer.writeheader()
    return writer


def write_stat(
    writer: csv.DictWriter,
    name: str,
    width: int,
    height: int,
    radius: int,
    type: str,
    result: list[float],
) -> None:
    for iteration in range(len(result)):
        writer.writerow(
            {
                "image": name,
                "width": width,
                "height": height,
                "radius": radius,
                "iteration": iteration,
                "type": type,
                "seconds": f"{result[iteration]:.6f}",
            }
        )


def run(
    blur: BlurInterface,
    type: str,
    image_path: Path,
    img: torch.Tensor,
    writer: csv.DictWriter,
    save_output: bool,
) -> None:
    height, width = img.shape[-2:]

    for radius in Config.Blur.Radii:
        print(f"Process: {image_path.name} ({width}x{height}), radius={radius}")
        path = None
        if save_output:
            image_output_dir = OUTPUT_DIR / image_path.stem.split("_", maxsplit=1)[0]
            image_output_dir.mkdir(exist_ok=True)
            path = image_output_dir / f"{image_path.stem}_r{radius}_{type}.jpg"

        results = blur.Apply(img, radius, Config.Blur.RUNS_COUNT, path)
        write_stat(writer, image_path.name, width, height, radius, type, results)

        print(f"\t{type}: AVG {sum(results) / len(results) * 1_000:.3f} ms =< {min(results) * 1_000:.3f} .. {max(results) * 1_000:.3f} >=")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"run_{run_timestamp}.csv"
    paths = image_paths()

    if not paths:
        print(f"No test images in {IMAGES_DIR}!")
        return

    methods = [
        (GpuPyTorchBlur(), "pt_gpu"),
        (GpuPyTorchOptimizedBlur(), "pt_gpu_opt"),
        (CudaBlur(), "cuda"),
        (CudaOptimizedBlur(), "cuda_opt"),
    ]
    largest_resolution = max(Config.RESOLUTIONS, key=lambda resolution: resolution[0] * resolution[1])

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = write_stat_header(csv_file)

        for blur, m_name in methods:
            for image_path in paths:
                with Image.open(image_path) as source:
                    source_rgb = source.convert("RGB")

                for width, height, _ in Config.RESOLUTIONS:
                    resized = source_rgb.resize((width, height), Image.Resampling.LANCZOS)
                    img = transforms.ToTensor()(resized)
                    save_output = (width, height) == largest_resolution[:2]

                    run(blur, m_name, image_path, img, writer, save_output)

    print(f"Running times: {csv_path}")


if __name__ == "__main__":
    main()
