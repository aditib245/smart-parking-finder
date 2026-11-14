from shapely.geometry import Polygon, box

Box = tuple[float, float, float, float]
Corners = list[tuple[int, int]]

TAKEN_THRESHOLD = 0.5


def coverage(spot_corners: Corners, car_box: Box) -> float:
  """Fraction of the spot's area that the car box covers (0.0 to 1.0)."""
  spot = Polygon(spot_corners)
  if spot.area == 0:
    return 0.0
  car = box(*car_box)
  return spot.intersection(car).area / spot.area


def iou(spot_corners: Corners, car_box: Box) -> float:
  """Overlap area divided by the combined area of both shapes (0.0 to 1.0)."""
  spot = Polygon(spot_corners)
  car = box(*car_box)
  union_area = spot.union(car).area
  if union_area == 0:
    return 0.0
  return spot.intersection(car).area / union_area


def best_coverage(spot_corners: Corners, car_boxes: list[Box]) -> float:
  """The highest coverage any single car gives this spot, or 0.0 if there are no cars."""
  return max((coverage(spot_corners, b) for b in car_boxes), default=0.0)


def is_taken(spot_corners: Corners, car_boxes: list[Box], threshold: float = TAKEN_THRESHOLD) -> bool:
  """A spot is taken when some car covers at least `threshold` of it."""
  return best_coverage(spot_corners, car_boxes) >= threshold
