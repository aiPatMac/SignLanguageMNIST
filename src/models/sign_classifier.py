"""LightningModule defining the CNN classifier."""
from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MulticlassAccuracy

from src.config import ModelConfig, OptimConfig


class SignClassifier(pl.LightningModule):
    def __init__(
        self,
        model_cfg: ModelConfig,
        optim_cfg: OptimConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model_cfg", "optim_cfg"])
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg

        c1, c2, c3 = model_cfg.hidden_channels
        self.features = nn.Sequential(
            nn.Conv2d(model_cfg.in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(c3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(128, model_cfg.num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=model_cfg.num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=model_cfg.num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=model_cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple wrapper
        return self.classifier(self.features(x))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, labels)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        if self.optim_cfg.optimizer.lower() == "adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.optim_cfg.lr,
                weight_decay=self.optim_cfg.weight_decay,
            )
        elif self.optim_cfg.optimizer.lower() == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=self.optim_cfg.lr,
                momentum=0.9,
                weight_decay=self.optim_cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optim_cfg.optimizer}")

        if self.optim_cfg.scheduler == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.optim_cfg.scheduler_factor,
                patience=self.optim_cfg.plateu_patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        return optimizer
