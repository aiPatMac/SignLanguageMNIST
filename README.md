# Sign Language MNIST – Lightning + LitServe

Compact training + serving workflow for Sign Language MNIST built with PyTorch Lightning, W&B logging, Optuna tuning, and LitServe for deployment.

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

Helpful overrides:
- `--monitoring-mode {online,offline,disabled}` toggles W&B.
- `--fast-dev-run` runs 1 batch for sanity checks.
- `--lr`, `--weight-decay`, `--dropout`, `--width-scale`, `--augment` let you replay Optuna findings.

## Hyper-parameter optimization

```powershell
python -m src.hpo_optuna --trials 20 --max-epochs 8 --output-dir outputs/hpo
```

## Serving (LitServe REST API)

Start the API (defaults to the best checkpoint path, port 8000):

```powershell
python -m src.service.lit_server --host 0.0.0.0 --port 8000
```

Send a sample request taken from the dataset:

```powershell
python -m src.service.send_example --csv data/sign_mnist_test.csv --index 0 --url http://127.0.0.1:8000/predict
```

The service accepts base64-encoded images in JSON (`{"image_base64": "..."}`) and returns `{class_index, letter, confidence}`. Use curl/requests with webcam captures or saved images to integrate with other clients.

### Webcam streaming client

Run the webcam overlay (press `q` to exit):

```powershell
python -m src.service.webcam_client --camera 0 --url http://127.0.0.1:8000/predict
```

Each frame is resized to 28×28, sent to the REST endpoint, and the returned letter/confidence is rendered on the video feed. Point `--url` to the EC2 endpoint when the service runs remotely.
