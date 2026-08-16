import time
from pathlib import Path

import torch
from torchvision import transforms

import pycuda_extension
from config import BlurInterface


class CpuPyTorchBlur(BlurInterface):
    def Apply(
        self,
        img: torch.Tensor,
        radius: int,
        cicle: int,
        output_path: Path | None = None,
    ) -> list[float]:
        if cicle < 1:
            raise ValueError("repetitions must be at least 1")
        else:
            cicle = min(3, cicle)

        filter_size = 2 * radius + 1
        conv_filter = torch.ones(3, 1, filter_size, filter_size) / (filter_size**2)
        cpu_input = img.unsqueeze(0)
        results = []
        cpu_output = None

        for _ in range(cicle):
            start = time.perf_counter()
            cpu_output = torch.nn.functional.conv2d(cpu_input, conv_filter, padding=radius, groups=3)
            results.append(time.perf_counter() - start)

        if output_path is not None and cpu_output is not None:
            transforms.ToPILImage()(cpu_output.squeeze(0)).save(output_path)

        return results


class GpuPyTorchBlur(BlurInterface):
    def Apply(
        self,
        img: torch.Tensor,
        radius: int,
        cicle: int,
        output_path: Path | None = None,
    ) -> list[float]:
        if cicle < 1:
            raise ValueError("repetitions must be at least 1")

        filter_size = 2 * radius + 1
        gpu_input = img.cuda()
        conv_filter = torch.ones(3, 1, filter_size, filter_size, device="cuda") / (filter_size**2)
        torch.cuda.synchronize()
        results = []
        gpu_output = None

        for _ in range(cicle):
            start = time.perf_counter()
            gpu_output = torch.nn.functional.conv2d(gpu_input.unsqueeze(0), conv_filter, padding=radius, groups=3)
            torch.cuda.synchronize()
            results.append(time.perf_counter() - start)

        if gpu_output is None:
            raise RuntimeError("PyTorch GPU blur did not produce output")
        if output_path is not None:
            transforms.ToPILImage()(gpu_output.squeeze(0).cpu()).save(output_path)

        return results


class GpuPyTorchOptimizedBlur(BlurInterface):
    def __init__(self) -> None:
        torch.backends.cudnn.benchmark = True

    def Apply(
        self,
        img: torch.Tensor,
        radius: int,
        cicle: int,
        output_path: Path | None = None,
    ) -> list[float]:
        if cicle < 1:
            raise ValueError("repetitions must be at least 1")

        filter_size = 2 * radius + 1
        gpu_input = img.cuda().unsqueeze(0)
        horizontal_filter = torch.ones(3, 1, 1, filter_size, device="cuda") / filter_size
        vertical_filter = torch.ones(3, 1, filter_size, 1, device="cuda") / filter_size
        torch.cuda.synchronize()
        results = []
        gpu_output = None

        with torch.inference_mode():
            for _ in range(cicle):
                start = time.perf_counter()
                horizontal = torch.nn.functional.conv2d(
                    gpu_input,
                    horizontal_filter,
                    padding=(0, radius),
                    groups=3,
                )
                gpu_output = torch.nn.functional.conv2d(
                    horizontal,
                    vertical_filter,
                    padding=(radius, 0),
                    groups=3,
                )
                torch.cuda.synchronize()
                results.append(time.perf_counter() - start)

        if gpu_output is None:
            raise RuntimeError("Optimized PyTorch GPU blur did not produce output")
        if output_path is not None:
            transforms.ToPILImage()(gpu_output.squeeze(0).cpu()).save(output_path)

        return results


class CudaBlur(BlurInterface):
    def Apply(
        self,
        img: torch.Tensor,
        radius: int,
        cicle: int,
        output_path: Path | None = None,
    ) -> list[float]:
        if cicle < 1:
            raise ValueError("repetitions must be at least 1")

        gpu_input = img.cuda()
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
            transforms.ToPILImage()(gpu_output.cpu()).save(output_path)

        return results


class CudaOptimizedBlur(CudaBlur):
    def Apply(
        self,
        img: torch.Tensor,
        radius: int,
        cicle: int,
        output_path: Path | None = None,
    ) -> list[float]:
        if cicle < 1:
            raise ValueError("repetitions must be at least 1")

        gpu_input = img.cuda()
        torch.cuda.synchronize()
        results = []
        gpu_output = None

        for _ in range(cicle):
            start = time.perf_counter()
            gpu_output = pycuda_extension.blur_opt(gpu_input, radius)
            torch.cuda.synchronize()
            results.append(time.perf_counter() - start)

        if gpu_output is None:
            raise RuntimeError("Optimized GPU blur did not produce output")
        if output_path is not None:
            transforms.ToPILImage()(gpu_output.cpu()).save(output_path)

        return results
