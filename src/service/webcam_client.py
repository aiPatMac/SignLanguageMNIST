"""Stream webcam frames to the LitServe API and display predictions."""
from __future__ import annotations

import argparse
import base64
import io
from typing import Tuple

import cv2
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"


def encode_frame(frame) -> Tuple[str, Image.Image]:
    """Convert BGR frame to base64 PNG expected by the service."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pil_image = Image.fromarray(gray)
    pil_image = pil_image.resize((28, 28))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded, pil_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send webcam frames to Sign MNIST API")
    parser.add_argument("--url", type=str, default=API_URL, help="Predict endpoint URL")
    parser.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture index")
    parser.add_argument("--timeout", type=float, default=3.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera}")

    last_status = "waiting for response"
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            payload_b64, _ = encode_frame(frame)
            payload = {"image_base64": payload_b64}
            try:
                response = requests.post(args.url, json=payload, timeout=args.timeout)
                response.raise_for_status()
                data = response.json()
                last_status = f"Pred: {data.get('letter', '?')} ({data.get('confidence', 0):.2f})"
            except requests.RequestException as exc:
                last_status = f"Request error: {exc}"[:80]

            cv2.putText(
                frame,
                last_status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Sign Language MNIST", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":  
    main()
