"""LitServe REST API for Sign Language MNIST classifier."""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from typing import Any, Dict

import litserve as ls
import torch
from PIL import Image
from torchvision import transforms

from src.config import ExperimentConfig
from src.models.sign_classifier import SignClassifier

DEFAULT_CKPT = Path(
    "outputs/toy_model/sign-mnist-lightning/0hbdp350/checkpoints/sign-mnist-02-0.549.ckpt"
)
DEFAULT_WIDTH_SCALE = 1.0
DEFAULT_DROPOUT = 0.31539834816417345


def _apply_width_scale(hidden_channels: tuple[int, ...], scale: float) -> tuple[int, ...]:
    if scale <= 0:
        raise ValueError("width scale must be positive")
    return tuple(max(1, int(round(ch * scale))) for ch in hidden_channels)


class SignLanguageAPI(ls.LitAPI):
    """LitServe API wrapper that loads a Lightning checkpoint and serves predictions."""

    def __init__(
        self,
        checkpoint: Path,
        dropout: float,
        width_scale: float,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.dropout = dropout
        self.width_scale = width_scale
        self.device = device
        self._label_map = [chr(ord("A") + i) for i in range(26) if i not in {9, 25}]
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def setup(self, device: torch.device) -> None:
        cfg = ExperimentConfig()
        cfg.model.dropout = self.dropout
        cfg.model.hidden_channels = _apply_width_scale(cfg.model.hidden_channels, self.width_scale)

        self.model = SignClassifier.load_from_checkpoint(
            self.checkpoint,
            model_cfg=cfg.model,
            optim_cfg=cfg.optim,
        )
        self.model.eval()
        self.model.to(device)
        self.device = str(device)

    def _decode_request(self, request: Dict[str, Any]) -> torch.Tensor:
        if "image_base64" not in request:
            raise ValueError("Request payload must include 'image_base64'")
        raw_bytes = base64.b64decode(request["image_base64"])
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("L")
        tensor = self.transform(pil_image).unsqueeze(0)
        return tensor

    @torch.inference_mode()
    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        model_device = next(self.model.parameters()).device
        image_tensor = self._decode_request(request).to(model_device)
        logits = self.model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, class_idx = torch.max(probabilities, dim=1)
        idx = int(class_idx.item())
        result = {
            "class_index": idx,
            "confidence": float(confidence.item()),
            "letter": self._label_map[idx] if idx < len(self._label_map) else idx,
        }
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Sign Language MNIST model via LitServe")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CKPT,
        help="Path to Lightning checkpoint (.ckpt)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--width-scale", type=float, default=DEFAULT_WIDTH_SCALE)
    parser.add_argument("--accelerator", type=str, default="auto", help="LitServe accelerator")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    api = SignLanguageAPI(
        checkpoint=args.checkpoint,
        dropout=args.dropout,
        width_scale=args.width_scale,
    )
    server = ls.LitServer(api, accelerator=args.accelerator)
    server.run(host=args.host, port=args.port)


if __name__ == "__main__": 
    main()
