"""Send a sample Sign Language MNIST image to the LitServe endpoint."""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import requests

from src.config import DataConfig
from src.data.dataset import SignLanguageMNISTDataset


def encode_sample(csv_path: Path, index: int) -> tuple[str, int]:
    dataset = SignLanguageMNISTDataset(csv_path, transform=None, remap_labels=True)
    if index >= len(dataset):
        raise IndexError(f"Index {index} out of range for dataset of size {len(dataset)}")
    image, label = dataset[index]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send sample inference request to LitServe API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000/predict",
        help="LitServe predict endpoint",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DataConfig().test_csv_path(),
        help="Path to Sign Language MNIST CSV (train or test)",
    )
    parser.add_argument("--index", type=int, default=0, help="Row index to sample")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_base64, label = encode_sample(args.csv, args.index)
    payload = {"image_base64": image_base64}
    response = requests.post(args.url, json=payload, timeout=args.timeout)
    response.raise_for_status()
    print("Ground truth label:", label)
    print("Server response:", response.json())


if __name__ == "__main__":  # pragma: no cover
    main()
