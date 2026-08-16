from abc import ABC, abstractmethod
from pathlib import Path

import torch


class Config:
    IMAGE_SUFFIXES = (".bmp", ".jpeg", ".jpg", ".png", ".webp")
    RESOLUTIONS = (
        (640, 480, "VGA"),
        (1024, 768, "QVGA"),
        (1280, 720, "HD"),
        (1920, 1080, "FHD"),
        (2048, 1200, "2K"),
        (2560, 1440, "QHD"),
        (3840, 2160, "UHD"),
        (4032, 3024, "12MP"),
    )

    class Blur:
        Radii = (1, 2, 3, 5, 7, 9, 11)
        RUNS_COUNT = 5


class BlurInterface(ABC):
    @abstractmethod
    def Apply(self, img: torch.Tensor, radius: int, cicle: int, output_path: Path | None = None) -> list[float]: ...
