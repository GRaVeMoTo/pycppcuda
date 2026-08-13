import csv
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import pycuda_extension

ROOT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = ROOT_DIR / "images"
OUTPUT_DIR = ROOT_DIR / "output"
BLUR_RADII = [1, 2, 3, 5, 7, 9]
RUNS_COUNT = 6
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def cpu_pytorch_blur(
    img_tensor: torch.Tensor,
    radius: int,
    cicle: int = 1,
    output_path: Path | None = None,
) -> list[float]:
    if cicle < 1:
        raise ValueError("repetitions must be at least 1")

    filter_size = 2 * radius + 1
    conv_filter = torch.ones(3, 1, filter_size, filter_size) / (filter_size**2)

    results = []
    cpu_input = img_tensor.unsqueeze(0)
    cpu_output = None
    for _ in range(cicle):
        start = time.perf_counter()
        cpu_output = torch.nn.functional.conv2d(cpu_input, conv_filter, padding=radius, groups=3)
        results.append(time.perf_counter() - start)

    if output_path is not None and cpu_output is not None:
        cpu_img = transforms.ToPILImage()(cpu_output.squeeze(0))
        cpu_img.save(output_path)

    return results


def gpu_blur(
    img_tensor: torch.Tensor,
    radius: int,
    cicle: int,
    output_path: Path | None = None,
) -> list[float]:
    if cicle < 1:
        raise ValueError("repetitions must be at least 1")

    gpu_input = img_tensor.cuda()
    torch.cuda.synchronize()
    results = []

    gpu_output = None
    for _ in range(cicle):
        start = time.perf_counter()
        gpu_output = pycuda_extension.blur(gpu_input, radius)
        torch.cuda.synchronize()
        results.append(time.perf_counter() - start)

    if gpu_output is None:
        raise RuntimeError("GPU blur did not produce output")
    if output_path is not None:
        gpu_img = transforms.ToPILImage()(gpu_output.cpu())
        gpu_img.save(output_path)

    return results


def image_paths() -> list[Path]:
    return sorted(path for path in IMAGES_DIR.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


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
    fn: Callable,
    type: str,
    image_path: Path,
    img: torch.Tensor,
    radii: list[int],
    cicle: int,
    writer: csv.DictWriter,
) -> None:
    height, width = img.shape[-2:]

    for radius in radii:
        print(f"Process: {image_path.name} ({width}x{height}), radius={radius}")
        path = OUTPUT_DIR / f"{image_path.stem}_r{radius}_{type}.png"

        results = fn(img, radius, cicle, path)
        write_stat(writer, image_path.name, width, height, radius, type, results)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"run_{run_timestamp}.csv"
    paths = image_paths()

    if not paths:
        print(f"No test images in {IMAGES_DIR}!")
        return

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = write_stat_header(csv_file)

        for image_path in paths:
            with Image.open(image_path) as source:
                img = source.convert("RGB")
                imageTensor = transforms.ToTensor()(img)

            run(
                cpu_pytorch_blur,
                "ptcpu",
                image_path,
                imageTensor,
                BLUR_RADII,
                RUNS_COUNT,
                writer,
            )
            run(
                gpu_blur,
                "cuda",
                image_path,
                imageTensor,
                BLUR_RADII,
                RUNS_COUNT,
                writer,
            )

    print(f"Running times: {csv_path}")


if __name__ == "__main__":
    main()
