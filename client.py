"""Utility script that sends a Sign Language MNIST sample to the REST API."""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

import requests


def _encode_image(image_path: Path) -> str:
	if not image_path.exists():
		raise FileNotFoundError(f"Image not found: {image_path}")
	image_bytes = image_path.read_bytes()
	return base64.b64encode(image_bytes).decode("utf-8")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Send a base64-encoded Sign Language MNIST image to the LitServe endpoint"
	)
	parser.add_argument(
		"--url",
		type=str,
		default="http://127.0.0.1:8000/predict",
		help="HTTP endpoint exposed by the VM/container",
	)
	parser.add_argument(
		"--image",
		type=Path,
		default=Path("assets/sample_request.png"),
		help="Path to a 28x28 grayscale PNG file",
	)
	parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	payload = {"image_base64": _encode_image(args.image)}
	response = requests.post(args.url, json=payload, timeout=args.timeout)
	response.raise_for_status()
	print("Server response:", response.json())


if __name__ == "__main__":
	main()
