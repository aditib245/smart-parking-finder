import sys
from pathlib import Path

import cv2
import numpy as np

from eval.pklot import Spot, load_spots
from eval.preview import GREEN, GREY, OUT_DIR, RED
from worker.detector import CarDetector, Detection
from worker.geometry import is_taken

ORANGE = (0, 165, 255)  # car boxes (BGR)
YELLOW = (0, 230, 255)  # spots where our answer disagrees with the label


def draw(image: np.ndarray, spots: list[Spot], cars: list[Detection], verdicts: dict[int, bool]) -> np.ndarray:
  canvas = image.copy()
  for car in cars:
    x1, y1, x2, y2 = (int(v) for v in car.box)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), ORANGE, 1)
  for spot in spots:
    if not spot.polygon:
      continue
    taken = verdicts[spot.id]
    wrong = spot.occupied is not None and taken != spot.occupied
    if wrong:
      colour, thickness = YELLOW, 4
    elif taken:
      colour, thickness = RED, 2
    else:
      colour, thickness = GREEN, 2
    points = np.array(spot.polygon, dtype=np.int32)
    cv2.polylines(canvas, [points], isClosed=True, color=colour, thickness=thickness)
  return canvas


def main(image_path: str) -> None:
  image_file = Path(image_path)
  image = cv2.imread(str(image_file))
  if image is None:
    sys.exit(f"Could not read image: {image_file}")
  spots = load_spots(image_file.with_suffix(".xml"))

  cars = CarDetector().detect(image)
  boxes = [car.box for car in cars]
  verdicts = {spot.id: is_taken(spot.polygon, boxes) for spot in spots}

  labelled = [s for s in spots if s.occupied is not None]
  correct = sum(verdicts[s.id] == s.occupied for s in labelled)
  missed = sum(s.occupied and not verdicts[s.id] for s in labelled)       # car there, we said free
  phantom = sum(not s.occupied and verdicts[s.id] for s in labelled)      # spot empty, we said taken

  OUT_DIR.mkdir(exist_ok=True)
  out_file = OUT_DIR / f"detect_{image_file.stem}.jpg"
  cv2.imwrite(str(out_file), draw(image, spots, cars, verdicts))

  print(f"{len(cars)} vehicles found. {correct}/{len(labelled)} spots match the labels "
        f"({correct / len(labelled):.0%}); {missed} taken spots called free, "
        f"{phantom} free spots called taken -> {out_file}")


if __name__ == "__main__":
  if len(sys.argv) != 2:
    sys.exit("Usage: python -m eval.detect_preview <path/to/image.jpg>")
  main(sys.argv[1])
