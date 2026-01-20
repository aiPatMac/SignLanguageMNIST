# Sign Language MNIST – End-to-End MLOps Pipeline

This project implements a robust machine learning pipeline for classifying hand signs from the MNIST Sign Language dataset. It transitions from a research-oriented PyTorch Lightning implementation to a production-ready system utilizing modern MLOps tools.

## Core Features

* **Training**: PyTorch Lightning implementation with custom DataModules and LightningModules.
* **Monitoring**: Real-time experiment tracking via Weights & Biases (W&B).
* **Optimization**: Hyper-parameter tuning using Optuna.
* **Deployment 1**: REST API using LitServe.
* **Deployment 2**: Containerized cloud hosting on AWS EC2.
* **Deployment 3**: Serverless event-driven inference using AWS Lambda, S3, and AWS CDK (Infrastructure as Code).

---

## 1. Environment Setup

The project requires Python 3.11+. Standardize the environment using the following commands:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

```

---

## 2. Training and Optimization

### Model Architecture

The model is a Convolutional Neural Network (CNN) defined in `src/models/sign_classifier.py`. It utilizes three convolutional layers followed by batch normalization, max pooling, and a fully connected classifier head with dropout for regularization.

### Experiment Tracking

Weights & Biases is used for monitoring training and validation curves.

```powershell
python -m src.train --batch-size 128 --max-epochs 15 --monitoring-mode online

```

### Hyper-parameter Optimization (HPO)

Optuna automates the search for optimal parameters (learning rate, dropout, width scaling).

```powershell
python -m src.hpo_optuna --trials 20 --max-epochs 8 --output-dir outputs/hpo

```

---

## 3. Deployment Strategy I: Local LitServe API

LitServe provides a high-performance wrapper around the model to expose it as a REST endpoint.

**Start the server:**

```powershell
python -m src.service.lit_server --host 0.0.0.0 --port 8000

```

**Client verification:**
The service expects base64-encoded images. Use the provided webcam client for real-time testing:

```powershell
python -m src.service.webcam_client --camera 0 --url http://127.0.0.1:8000/predict

```

---

## 4. Deployment Strategy II: Containerized AWS EC2

This strategy focuses on "lift and shift" containerization, running the LitServe API inside a Docker container on a persistent virtual machine.

### Local Build and Smoke Test

```powershell
docker build -t sign-mnist-serve:latest -f Dockerfile.litserve .
docker run --rm -p 8000:8000 sign-mnist-serve:latest

```

### EC2 Provisioning

1. Launch an AWS EC2 instance (t3.medium recommended).
2. Configure Security Groups to allow inbound traffic on port **22 (SSH)** and **8000 (HTTP)**.
3. Install Docker on the VM and deploy the container:
```bash
docker run -d --name sign-mnist -p 8000:8000 sign-mnist-serve:latest

```



---

## 5. Deployment Strategy III: Serverless (AWS Lambda & CDK)

This is a fully event-driven architecture where an image upload to S3 triggers a Lambda function to perform inference.

### Technical Implementation

* **Infrastructure as Code (IaC)**: Built using **AWS CDK** in Python.
* **Compute**: AWS Lambda running a Docker image to handle heavy dependencies like PyTorch.
* **Storage**: AWS S3 for input images and output JSON results.
* **Trigger**: S3 `ObjectCreated` events filtered by the `.jpg` suffix.

### Deployment Commands

Navigate to the `infra` directory:

```powershell
cd infra
cdk bootstrap
cdk deploy

```

### Testing the Pipeline

1. Upload a hand sign image (e.g., `test.jpg`) to the created S3 bucket.
2. The S3 trigger invokes the Lambda function.
3. The Lambda downloads the image, runs inference, and uploads `test.json` back to the bucket.

**Manual Verification via AWS CLI:**

```powershell
aws s3 ls s3://infrastack-mnistinferencebucket/

```

---

## Technical Challenges & Solutions

### Environment Parity (EC2 vs. Lambda)

A significant challenge involved Python import paths. The project uses absolute imports (`from src.models...`) compatible with EC2. However, AWS Lambda's default container structure flattens the directory.

* **Solution**: The `Dockerfile.lambda` was configured to copy the `src` directory as a named package and set the `PYTHONPATH` to `/var/task`, ensuring that the same code used on EC2 runs on Lambda without modification.

### Pytorch Lightning Initialization

Loading models from checkpoints in a restricted environment (Lambda) requires manual instantiation of configuration classes.

* **Solution**: The `lambda_handler` was updated to explicitly instantiate `ModelConfig` and `OptimConfig` before calling `load_from_checkpoint`, preventing initialization errors related to missing positional arguments.

---

## Repository Structure

* `src/`: Core logic (Training, DataModules, Models).
* `src/service/`: Deployment handlers for LitServe and Lambda.
* `infra/`: AWS CDK code for serverless infrastructure.
* `models/`: Best model checkpoints (ignored by git, must be provided).
* `Dockerfile.litserve`: Container config for EC2.
* `Dockerfile.lambda`: Container config for AWS Lambda.