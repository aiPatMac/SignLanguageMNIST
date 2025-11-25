"""Hyper-parameter optimization using Optuna."""
from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping

from src.config import ExperimentConfig
from src.data.datamodule import SignLanguageDataModule
from src.models.sign_classifier import SignClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna search for Sign MNIST model")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--study-name", type=str, default="sign-mnist-optuna")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URI")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hpo"),
        help="Dir to persist best params",
    )
    return parser.parse_args()


class OptunaPruningCallback(Callback):
    """Lightweight pruning callback compatible with Lightning 2.x."""

    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss") -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        value = float(metric.detach().cpu()) if hasattr(metric, "detach") else float(metric)
        self.trial.report(value, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial pruned at epoch {trainer.current_epoch}")


def build_config(trial: optuna.Trial, base_cfg: ExperimentConfig) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.data.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    cfg.data.augment = trial.suggest_categorical("augment", [True, False])

    cfg.model.dropout = trial.suggest_float("dropout", 0.1, 0.5)
    width_scale = trial.suggest_categorical("width_scale", [0.75, 1.0, 1.25])
    cfg.model.hidden_channels = tuple(int(ch * width_scale) for ch in base_cfg.model.hidden_channels)

    cfg.optim.lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    cfg.optim.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    cfg.trainer.max_epochs = base_cfg.trainer.max_epochs
    cfg.trainer.fast_dev_run = False
    cfg.trainer.limit_train_batches = 0.2
    cfg.trainer.limit_val_batches = 0.2
    cfg.monitoring.mode = "disabled"
    return cfg


def objective(trial: optuna.Trial, base_cfg: ExperimentConfig, args: argparse.Namespace) -> float:
    cfg = build_config(trial, base_cfg)
    seed_everything(args.seed, workers=True)

    data_module = SignLanguageDataModule(cfg.data, seed=args.seed)
    model = SignClassifier(cfg.model, cfg.optim)

    early_stop = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    trainer = pl.Trainer(
        max_epochs=min(cfg.trainer.max_epochs, args.max_epochs),
        accelerator="auto",
        devices="auto",
        logger=False,
        callbacks=[OptunaPruningCallback(trial, monitor="val_loss"), early_stop],
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
    )

    trainer.fit(model, datamodule=data_module)
    val_acc = trainer.callback_metrics.get("val_acc")
    if val_acc is None:
        raise RuntimeError("Validation accuracy not available")
    return float(val_acc.cpu())


def main() -> None:
    args = parse_args()
    base_cfg = ExperimentConfig()

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=bool(args.storage),
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    study.optimize(lambda trial: objective(trial, base_cfg, args), n_trials=args.trials)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_params.txt"
    best_path.write_text(str(study.best_params))
    print(f"Best value: {study.best_value:.4f}\nBest params saved to {best_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
