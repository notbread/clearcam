from __future__ import annotations

from ..config import Settings
from .yolo import UltralyticsDetector


def create_detector(settings: Settings) -> UltralyticsDetector:
    return UltralyticsDetector(settings.model, settings.device, settings.classes)
