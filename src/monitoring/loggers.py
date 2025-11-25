"""Utility helpers for experiment logging and callbacks."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.config import ExperimentConfig

try:  # pragma: no cover - dependency optional
    from pytorch_lightning.loggers import WandbLogger
except ImportError:  # pragma: no cover
    WandbLogger = None  # type: ignore[assignment]


def setup_logging(
    config: ExperimentConfig,
    save_dir: str | Path = "outputs",
    enable_wandb: bool = True,
) -> Tuple[CSVLogger, List[ModelCheckpoint]]:
    """Create logger + default callbacks, preferring W&B when available."""

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            filename="sign-mnist-{epoch:02d}-{val_acc:.3f}",
            save_top_k=1,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if enable_wandb and WandbLogger is not None:
        logger = WandbLogger(
            project=config.monitoring.project,
            entity=config.monitoring.entity,
            mode=config.monitoring.mode,
            name=config.monitoring.run_name,
            save_dir=str(save_path),
            tags=config.monitoring.tags,
        )
        logger.experiment.config.update(config.as_dict())
    else:
        logger = CSVLogger(save_dir=str(save_path), name="sign-mnist")

    return logger, callbacks
