import csv
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import pycuda_extension

ROOT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = ROOT_DIR / "images"
OUTPUT_DIR = ROOT_DIR / "output"
BLUR_RADIUS = 5
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def cpu_pytorch_blur(img: Image.Image, radius: int, output_path: Path) -> float:
    start = time.time()

    filter_size = 2 * radius + 1
    conv_filter = torch.ones(3, 1, filter_size, filter_size) / (filter_size**2)

    cpu_input = transforms.ToTensor()(img).unsqueeze(0)
    cpu_output = torch.nn.functional.conv2d(
        cpu_input, conv_filter, padding=radius, groups=3
    )
    runtime = time.time() - start

    cpu_img = transforms.ToPILImage()(cpu_output.squeeze(0))
    cpu_img.save(output_path)
    return runtime


def gpu_blur(img: Image.Image, radius: int, output_path: Path) -> float:
    gpu_input = transforms.ToTensor()(img).cuda()

    # (Warm-up)
    warm_start = time.time()
    _ = pycuda_extension.blur(gpu_input, radius)
    torch.cuda.synchronize()
    warm_runtime = time.time() - warm_start

    start = time.time()
    gpu_output = pycuda_extension.blur(gpu_input, radius)
    torch.cuda.synchronize()
    runtime = time.time() - start
    gpu_img = transforms.ToPILImage()(gpu_output.cpu())
    gpu_img.save(output_path)

    print(f"gpu: {warm_runtime} {runtime} dT: {runtime-warm_runtime}")

    return runtime


def image_paths() -> list[Path]:
    return sorted(
        path for path in IMAGES_DIR.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"run_{run_timestamp}.csv"
    paths = image_paths()

    if not paths:
        print(f"No test images in {IMAGES_DIR}!")
        return

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image",
                "width",
                "height",
                "cpu_seconds",
                "gpu_seconds",
                "speedup",
            ],
        )
        writer.writeheader()

        for image_path in paths:
            with Image.open(image_path) as source:
                img = source.convert("RGB")

            print(f"Process: {image_path.name} ({img.width}x{img.height})")
            cpu_path = OUTPUT_DIR / f"{image_path.stem}_cpu.png"
            gpu_path = OUTPUT_DIR / f"{image_path.stem}_gpu.png"
            cpu_seconds = cpu_pytorch_blur(img, BLUR_RADIUS, cpu_path)
            gpu_seconds = gpu_blur(img, BLUR_RADIUS, gpu_path)
            speedup = cpu_seconds / gpu_seconds
            writer.writerow(
                {
                    "image": image_path.name,
                    "width": img.width,
                    "height": img.height,
                    "cpu_seconds": f"{cpu_seconds:.6f}",
                    "gpu_seconds": f"{gpu_seconds:.6f}",
                    "speedup": f"{speedup:.2f}",
                }
            )

    print(f"Running times: {csv_path}")

if __name__ == "__main__":
    main()
