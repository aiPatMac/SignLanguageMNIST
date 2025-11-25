"""Dataset definitions for Sign Language MNIST."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class SignLanguageMNISTDataset(Dataset):
    """PyTorch dataset wrapping the Sign Language MNIST CSV files."""

    def __init__(
        self,
        csv_path: Path | str,
        transform: Optional[Callable] = None,
        remap_labels: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        frame = pd.read_csv(self.csv_path)
        if "label" not in frame.columns:
            raise ValueError("CSV is expected to contain a 'label' column")

        self._original_labels = frame["label"].to_numpy(dtype=np.int64)
        pixels = frame.drop(columns=["label"]).to_numpy(dtype=np.uint8)
        self.images = pixels.reshape(-1, 28, 28)

        if remap_labels:
            uniques = sorted(set(int(lbl) for lbl in self._original_labels))
            self.label_mapping = {label: idx for idx, label in enumerate(uniques)}
            self.labels = np.array([self.label_mapping[int(lbl)] for lbl in self._original_labels])
        else:
            self.label_mapping = None
            self.labels = self._original_labels

        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        image_array = self.images[idx]
        image = Image.fromarray(image_array, mode="L")  # grayscale
        label = int(self.labels[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @property
    def num_classes(self) -> int:
        return len(set(self.labels.tolist()))

    @property
    def original_label_mapping(self) -> Optional[dict[int, int]]:
        return self.label_mapping
