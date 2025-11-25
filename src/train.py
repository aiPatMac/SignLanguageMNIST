"""Training entrypoint for Sign Language MNIST using Lightning."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from src.config import ExperimentConfig
from src.data.datamodule import SignLanguageDataModule
from src.models.sign_classifier import SignClassifier
from src.monitoring.loggers import setup_logging


def _str2bool(value: str) -> bool:
    value_lower = value.lower()
    if value_lower in {"true", "t", "1", "yes", "y"}:
        return True
    if value_lower in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret '{value}' as boolean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sign Language classifier")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay override")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout override")
    parser.add_argument(
        "--width-scale",
        type=float,
        default=None,
        help="Scale the base conv channels (e.g., 0.75)",
    )
    parser.add_argument(
        "--augment",
        type=_str2bool,
        default=None,
        help="Force-enable/disable data augmentation (true/false)",
    )
    parser.add_argument("--fast-dev-run", action="store_true", help="Quick pipeline smoke test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--monitoring-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Control W&B logging",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for checkpoints/logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig()

    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.augment is not None:
        config.data.augment = args.augment
    if args.max_epochs:
        config.trainer.max_epochs = args.max_epochs
    if args.lr:
        config.optim.lr = args.lr
    if args.weight_decay is not None:
        config.optim.weight_decay = args.weight_decay
    if args.dropout is not None:
        config.model.dropout = args.dropout
    if args.width_scale is not None:
        scale = args.width_scale
        if scale <= 0:
            raise ValueError("width_scale must be positive")
        config.model.hidden_channels = tuple(
            max(1, int(ch * scale)) for ch in config.model.hidden_channels
        )
    if args.fast_dev_run:
        config.trainer.fast_dev_run = True

    requested_mode = args.monitoring_mode or config.monitoring.mode
    use_wandb = requested_mode != "disabled"
    config.monitoring.mode = requested_mode if use_wandb else "offline"

    seed_everything(args.seed, workers=True)

    data_module = SignLanguageDataModule(config.data, seed=args.seed)
    model = SignClassifier(config.model, config.optim)

    logger, callbacks = setup_logging(
        config, save_dir=args.output_dir, enable_wandb=use_wandb
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices or "auto",
        precision=config.trainer.precision,
        log_every_n_steps=config.trainer.log_every_n_steps,
        deterministic=config.trainer.deterministic,
        fast_dev_run=config.trainer.fast_dev_run,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":  # pragma: no cover
    main()
