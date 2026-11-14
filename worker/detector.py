from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from worker.geometry import Box

VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}
MIN_CONFIDENCE = 0.1  
MODEL_NAME = "yolo26s.pt"
WEIGHTS_DIR = Path("models")  


@dataclass
class Detection:
  box: Box
  label: str
  confidence: float


def keep_vehicles(
  boxes: list[list[float]],
  class_ids: list[int],
  confidences: list[float],
  min_confidence: float = MIN_CONFIDENCE,
) -> list[Detection]:
  """Keep only vehicle boxes the model is confident about."""
  return [
    Detection(box=tuple(box), label=VEHICLE_CLASSES[class_id], confidence=confidence)
    for box, class_id, confidence in zip(boxes, class_ids, confidences)
    if class_id in VEHICLE_CLASSES and confidence >= min_confidence
  ]


class CarDetector:
  """Loads YOLO once, then finds vehicles in as many images as you give it."""

  def __init__(self, model_name: str = MODEL_NAME, image_size: int = 1280, min_confidence: float = MIN_CONFIDENCE):
    WEIGHTS_DIR.mkdir(exist_ok=True)
    self.model = YOLO(str(WEIGHTS_DIR / model_name))  # downloads the weights the first time
    self.image_size = image_size
    self.min_confidence = min_confidence
    self.device = "mps" if torch.backends.mps.is_available() else "cpu"

  def detect(self, image: np.ndarray) -> list[Detection]:
    result = self.model.predict(
      image,
      imgsz=self.image_size,
      conf=self.min_confidence,
      device=self.device,
      verbose=False,
    )[0]
    return keep_vehicles(
      result.boxes.xyxy.tolist(),
      [int(c) for c in result.boxes.cls.tolist()],
      result.boxes.conf.tolist(),
      self.min_confidence,
    )
