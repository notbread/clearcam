from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def export_model(model: str, int8: bool = False, device: str = "cpu") -> str:
    """Export a PyTorch YOLO model to an OpenVINO IR directory."""
    if int8:
        try:
            import nncf  # noqa: F401
        except ImportError:
            raise SystemExit(
                "INT8 quantization requires 'nncf'. Install it with: uv pip install nncf"
            )
    logger.info("Loading baseline model weights: %s", model)
    yolo = YOLO(model)
    export_path = yolo.export(format="openvino", quantize=8 if int8 else 16, device=device)
    logger.info("Model exported to: %s", export_path)
    return str(export_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a YOLO model to OpenVINO format.")
    parser.add_argument("--model", required=True, help="Source .pt model (e.g. yolo11n.pt)")
    parser.add_argument("--int8", action="store_true", help="Use INT8 quantization (default: FP16)")
    parser.add_argument("--device", default="cpu", help="Device used during export (default: cpu)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not Path(args.model).is_file():
        logger.error("Model file '%s' does not exist.", args.model)
        sys.exit(1)

    export_model(args.model, int8=args.int8, device=args.device)


if __name__ == "__main__":
    main()
