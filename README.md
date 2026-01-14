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

## Requests-only smoke test

`client.py` is a tiny script that uses nothing but the Python standard library and the `requests` package to exercise the deployed endpoint. A sample image extracted from the dataset is stored in `assets/sample_request.png`.

```powershell
python client.py --url http://127.0.0.1:8000/predict --image assets/sample_request.png
```

Override `--url` with your VM's public address (e.g., `http://<ip>:8000/predict`). The payload matches the format used for grading, so you can hand over the same command with the production URL.

## Containerized & cloud deployment guide

1. **Build the Docker image locally**

	```powershell
	docker build -t sign-mnist-serve:latest .
	```

2. **Smoke-test the container on your machine**

	```powershell
	docker run --rm -p 8000:8000 sign-mnist-serve:latest
	python client.py --url http://127.0.0.1:8000/predict --image assets/sample_request.png
	```

3. **Provision a VM** (AWS EC2 t3.small/t3.medium, GCP e2-standard-2, Azure B2s all work). Open ports 22 (SSH) and 8000 (HTTP) in the security group/firewall. Assign an Elastic/Public IP.

4. **Install Docker on the VM** after SSH-ing in:

	```bash
	curl -fsSL https://get.docker.com | sudo sh
	sudo usermod -aG docker $USER  # log out/in afterwards
	```

5. **Transfer code or image**

	- *Option A (registry)*: `docker tag sign-mnist-serve <registry>/sign-mnist-serve:latest` then push to ECR/GCR/ACR, login on the VM, and `docker pull`.
	- *Option B (recommended for homework)*: `scp -r ./proj ubuntu@<ip>:/home/ubuntu/proj` (or use Git to clone the repo) and run `docker build` directly on the VM.

6. **Run the container on the VM**

	```bash
	docker run -d --name sign-mnist \
	  -p 8000:8000 \
	  -e CHECKPOINT_PATH=/app/outputs/sign-mnist-lightning/tmggzfkj/checkpoints/sign-mnist-12-0.077.ckpt \
	  sign-mnist-serve:latest
	```

	The LitServe process exposes `POST /predict` and loads the checkpoint baked into the image. If you upload a different `.ckpt`, update `CHECKPOINT_PATH` or pass `--checkpoint` to the container command.

7. **Harden networking** by restricting the security group / firewall rule to only the graders' IPs if possible. Keeping the instance in a private subnet behind a load balancer is even better, but not required for this assignment.

8. **Validate from your laptop** using the same requests-only client, this time pointing at the public endpoint:

	```powershell
	python client.py --url http://<public-ip>:8000/predict --image assets/sample_request.png
	```

### Troubleshooting tips

- LitServe logs appear via `docker logs -f sign-mnist`.
- If the VM reboots, enable `docker run --restart unless-stopped ...` or create a `systemd` unit to keep the service alive.
- Handle OS-level firewalls (UFW on Ubuntu) with `sudo ufw allow 8000/tcp` if needed.
- When using another model checkpoint, keep the file inside the repo (or mount it with `-v /path:/app/checkpoints`) and re-run the container with the new path.
