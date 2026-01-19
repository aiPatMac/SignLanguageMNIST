"""Central configuration dataclasses for the Sign Language MNIST project."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class DataConfig:
    """Settings for locating data and preparing loaders."""

    data_root: Path = Path("data")
    train_csv: str = "sign_mnist_train.csv"
    test_csv: str = "sign_mnist_test.csv"
    val_split: float = 0.1
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    augment: bool = True

    def train_csv_path(self) -> Path:
        return self.data_root / self.train_csv

    def test_csv_path(self) -> Path:
        return self.data_root / self.test_csv


@dataclass(slots=True)
class ModelConfig:
    """Hyper-parameters describing the classifier architecture."""

    input_size: int = 28
    in_channels: int = 1
    num_classes: int = 24
    dropout: float = 0.2
    hidden_channels: tuple[int, int, int] = (8, 16, 32)


@dataclass(slots=True)
class OptimConfig:
    """Optimizer and scheduler parameters."""

    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: Optional[str] = "reduce_on_plateau"
    plateu_patience: int = 3
    scheduler_factor: float = 0.5


@dataclass(slots=True)
class TrainerConfig:
    """PyTorch Lightning Trainer configuration."""

    max_epochs: int = 15
    accelerator: str = "auto"
    devices: Optional[int | list[int] | str] = None
    precision: str | int = "16-mixed"
    log_every_n_steps: int = 50
    limit_train_batches: Optional[float] = None
    limit_val_batches: Optional[float] = None
    deterministic: bool = True
    fast_dev_run: bool = False


@dataclass(slots=True)
class MonitoringConfig:
    """Experiment tracking configuration (W&B)."""

    project: str = "sign-mnist-lightning"
    entity: Optional[str] = None
    mode: str = "online"  # set to "offline" for local logging
    run_name: Optional[str] = None
    tags: list[str] = field(default_factory=lambda: ["baseline"])


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    def as_dict(self) -> dict[str, object]:
        """Serialize nested config for loggers."""
        return {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "optim": asdict(self.optim),
            "trainer": asdict(self.trainer),
            "monitoring": asdict(self.monitoring),
        }
