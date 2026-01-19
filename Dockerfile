# syntax=docker/dockerfile:1.6

# Builder
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Runtime
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 appuser && useradd -r -u 1000 -g appuser appuser

COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser data/data_description.md ./data/

# COPY --chown=appuser:appuser outputs/toy_model/sign-mnist-lightning/0hbdp350/checkpoints/sign-mnist-02-0.549.ckpt ./models/model.ckpt
COPY --chown=appuser:appuser models/toy_model.ckpt ./models/model.ckpt

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV CHECKPOINT_PATH=/app/models/model.ckpt

CMD ["sh", "-c", "python -m src.service.lit_server --host 0.0.0.0 --port 8000 --checkpoint ${CHECKPOINT_PATH} --width-scale 1.0"]