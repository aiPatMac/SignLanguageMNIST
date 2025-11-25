# Sign Language MNIST

Modular training stack for the Sign Language MNIST dataset covering Lightning training, Weights & Biases monitoring, and Optuna-based hyper-parameter optimization.

## Environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you plan to log to Weights & Biases, set `WANDB_API_KEY` in your shell or create a `.env` file that exports it before running `train.py`.

## Training

```powershell
python -m src.train --batch-size 128 --max-epochs 15 --monitoring-mode online
```

Helpful flags:
- `--monitoring-mode {online,offline,disabled}`: control W&B usage (falls back to CSV logs when disabled/offline).
- `--fast-dev-run`: single-batch smoke test for debugging.
- `--output-dir`: where checkpoints/metrics are stored.

## Hyper-parameter optimization

```powershell
python -m src.hpo_optuna --trials 20 --max-epochs 8 --output-dir outputs/hpo
```

The script searches key knobs (learning rate, dropout, width, batch size, augmentation flag) with Optuna + Lightning’s pruning callback. Final parameters are written to `outputs/hpo/best_params.txt`.
