# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data/data_description.md ./data/data_description.md
RUN mkdir -p /app/outputs/sign-mnist-lightning/tmggzfkj/checkpoints
COPY outputs/sign-mnist-lightning/tmggzfkj/checkpoints/sign-mnist-12-0.077.ckpt \
    /app/outputs/sign-mnist-lightning/tmggzfkj/checkpoints/

EXPOSE 8000

ENV CHECKPOINT_PATH=/app/outputs/sign-mnist-lightning/tmggzfkj/checkpoints/sign-mnist-12-0.077.ckpt
CMD ["python", "-m", "src.service.lit_server", "--host", "0.0.0.0", "--port", "8000", "--checkpoint", "/app/outputs/sign-mnist-lightning/tmggzfkj/checkpoints/sign-mnist-12-0.077.ckpt"]