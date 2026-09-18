from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from ..config import DEFAULT_TARGET_CLASSES

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    labels: set[str] = field(default_factory=set)
    first_frame: int | None = None
    last_frame: int | None = None


@dataclass
class FrameDetection:
    xyxy: tuple[int, int, int, int]
    cls_id: int
    confidence: float


def is_openvino_model(model: str) -> bool:
    """Ultralytics saves exported OpenVINO models as a '<stem>_openvino_model' directory."""
    return Path(model).is_dir()


def resolve_target_classes(model: YOLO, class_names: tuple[str, ...]) -> set[int]:
    """Map requested class names to COCO ids via model.names; default to the
    built-in person+animal set when no names are given."""
    if not class_names:
        return set(DEFAULT_TARGET_CLASSES)

    resolved: set[int] = set()
    for name in class_names:
        matches = [cid for cid, cname in model.names.items() if cname == name]
        if matches:
            resolved.update(matches)
        else:
            logger.warning("Unknown class name '%s' for this model, ignoring", name)
    if not resolved:
        raise ValueError(f"None of the requested classes {class_names} exist in the model")
    return resolved


class UltralyticsDetector:
    """Wraps ultralytics YOLO. Works transparently for .pt weights and
    exported OpenVINO IR directories."""

    def __init__(self, model_path: str, device: str | None, class_names: tuple[str, ...] = ()):
        self.device = self._resolve_device(model_path, device)
        self._model = self._load_with_fallback(model_path)
        self.target_classes = resolve_target_classes(self._model, class_names)

    @property
    def class_names(self) -> dict[int, str]:
        return self._model.names

    @staticmethod
    def _resolve_device(model_path: str, device: str | None) -> str | None:
        if device is not None:
            return device
        if is_openvino_model(model_path):
            return "intel:gpu"  # auto-prefer iGPU; falls back to CPU on failure
        return None  # ultralytics default (cpu) for .pt models

    @staticmethod
    def _load(model_path: str, device: str | None) -> YOLO:
        model = YOLO(model_path, task="detect")
        if device is not None:
            # A trivial predict forces backend initialization so an unusable
            # device fails here, during construction.
            model.predict(
                source=np.zeros((64, 64, 3), dtype=np.uint8),
                device=device,
                verbose=False,
            )
        return model

    def _load_with_fallback(self, model_path: str) -> YOLO:
        try:
            return self._load(model_path, self.device)
        except Exception:
            if self.device == "intel:gpu":
                logger.warning("intel:gpu unavailable, falling back to cpu")
                self.device = "cpu"
                return self._load(model_path, self.device)
            raise

    def detect(
        self,
        video: Path,
        conf: float,
        vid_stride: int,
        stop_on_first: bool,
    ) -> DetectionResult:
        result = DetectionResult()
        kwargs: dict = {"source": str(video), "stream": True, "conf": conf, "vid_stride": vid_stride, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device

        frame_idx = 0
        for res in self._model.predict(**kwargs):
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    cls = int(box.cls[0])
                    if cls in self.target_classes:
                        label = self._model.names[cls]
                        if label not in result.labels:
                            logger.info("  [+] Found %s with confidence %.2f", label, float(box.conf[0]))
                        result.labels.add(label)
                        if result.first_frame is None:
                            result.first_frame = frame_idx
                        result.last_frame = frame_idx
            frame_idx += vid_stride
            if result.labels and stop_on_first:
                break
        return result

    def predict_frame(self, frame: np.ndarray, conf: float) -> list[FrameDetection]:
        kwargs: dict = {"source": frame, "conf": conf, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        res = self._model.predict(**kwargs)[0]
        detections: list[FrameDetection] = []
        if res.boxes is not None:
            for box in res.boxes:
                cls = int(box.cls[0])
                if cls in self.target_classes:
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    detections.append(FrameDetection(xyxy=(x1, y1, x2, y2), cls_id=cls, confidence=float(box.conf[0])))
        return detections
