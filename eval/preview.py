import sys
from pathlib import Path

import cv2
import numpy as np

from eval.pklot import Spot, load_spots

GREEN = (0, 200, 0)
RED = (0, 0, 220)
GREY = (160, 160, 160)

OUT_DIR = Path("out")


def colour_for(spot: Spot) -> tuple[int, int, int]:
  if spot.occupied is None:
    return GREY
  return RED if spot.occupied else GREEN


def draw_spots(image: np.ndarray, spots: list[Spot]) -> np.ndarray:
  canvas = image.copy()
  for spot in spots:
    if not spot.polygon:
      continue
    points = np.array(spot.polygon, dtype=np.int32)
    cv2.polylines(canvas, [points], isClosed=True, color=colour_for(spot), thickness=2)
    x, y = points.mean(axis=0).astype(int)
    label_pos = (int(x) - 8, int(y) + 5)
    cv2.putText(canvas, str(spot.id), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, str(spot.id), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
  return canvas


def main(image_path: str) -> None:
  image_file = Path(image_path)
  image = cv2.imread(str(image_file))
  if image is None:
    sys.exit(f"Could not read image: {image_file}")

  spots = load_spots(image_file.with_suffix(".xml"))
  canvas = draw_spots(image, spots)

  OUT_DIR.mkdir(exist_ok=True)
  out_file = OUT_DIR / f"preview_{image_file.stem}.jpg"
  cv2.imwrite(str(out_file), canvas)

  free = sum(1 for s in spots if s.occupied is False)
  taken = sum(1 for s in spots if s.occupied is True)
  unknown = len(spots) - free - taken
  print(f"{len(spots)} spots: {free} free, {taken} taken, {unknown} unlabelled -> {out_file}")


if __name__ == "__main__":
  if len(sys.argv) != 2:
    sys.exit("Usage: python -m eval.preview <path/to/image.jpg>")
  main(sys.argv[1])