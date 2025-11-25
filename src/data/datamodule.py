"""LightningDataModule for Sign Language MNIST."""
from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.config import DataConfig
from src.data.dataset import SignLanguageMNISTDataset


class SignLanguageDataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig, seed: int = 42) -> None:
        super().__init__()
        self.config = config
        self.seed = seed
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self._num_classes: Optional[int] = None

    def _train_transform(self):
        aug = []
        if self.config.augment:
            aug.extend(
                [
                    transforms.RandomRotation(6),
                    transforms.RandomResizedCrop(28, scale=(0.9, 1.1)),
                ]
            )
        aug.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        return transforms.Compose(aug)

    def _eval_transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        if self.train_set is not None and self.val_set is not None and self.test_set is not None:
            return

        full_train = SignLanguageMNISTDataset(
            self.config.train_csv_path(), transform=self._train_transform()
        )
        val_size = int(len(full_train) * self.config.val_split)
        train_size = len(full_train) - val_size
        generator = torch.Generator().manual_seed(self.seed)
        self.train_set, self.val_set = random_split(full_train, [train_size, val_size], generator=generator)

        self.test_set = SignLanguageMNISTDataset(
            self.config.test_csv_path(), transform=self._eval_transform()
        )
        self._num_classes = full_train.num_classes

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    @property
    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("Setup has not been run yet")
        return self._num_classes
