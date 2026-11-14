from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2


@dataclass
class Spot:
  id: int
  occupied: bool | None  # None when the label file does not say
  polygon: list[tuple[int, int]]


def _corners_from_rotated_rect(space: ET.Element) -> list[tuple[int, int]]:
  rect = space.find("rotatedRect")
  if rect is None:
    return []
  center, size, angle = rect.find("center"), rect.find("size"), rect.find("angle")
  box = cv2.boxPoints((
    (float(center.get("x")), float(center.get("y"))),
    (float(size.get("w")), float(size.get("h"))),
    float(angle.get("d")),
  ))
  return [(int(round(x)), int(round(y))) for x, y in box]


def parse_spots(xml_text: str) -> list[Spot]:
  root = ET.fromstring(xml_text)
  spots = []
  for space in root.findall("space"):
    polygon = [(int(p.get("x")), int(p.get("y"))) for p in space.findall("contour/point")]
    if not polygon:
      polygon = _corners_from_rotated_rect(space)
    occupied_text = space.get("occupied")
    occupied = None if occupied_text is None else occupied_text == "1"
    spots.append(Spot(id=int(space.get("id")), occupied=occupied, polygon=polygon))
  return spots


def load_spots(xml_path: str | Path) -> list[Spot]:
  return parse_spots(Path(xml_path).read_text())
