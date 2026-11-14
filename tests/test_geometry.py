import pytest

from worker.geometry import best_coverage, coverage, iou, is_taken

# A 10 x 10 square spot from (0, 0) to (10, 10). Its area is 100.
SPOT = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_car_exactly_on_spot():
  assert coverage(SPOT, (0, 0, 10, 10)) == pytest.approx(1.0)
  assert iou(SPOT, (0, 0, 10, 10)) == pytest.approx(1.0)


def test_car_far_away():
  assert coverage(SPOT, (50, 50, 60, 60)) == 0.0
  assert iou(SPOT, (50, 50, 60, 60)) == 0.0


def test_car_covers_left_half():
  # Box from x=0 to x=5 covers 5 x 10 = 50 of the spot's 100.
  assert coverage(SPOT, (0, 0, 5, 10)) == pytest.approx(0.5)
  # Union is just the spot (100), so IoU is also 50 / 100.
  assert iou(SPOT, (0, 0, 5, 10)) == pytest.approx(0.5)


def test_big_box_has_full_coverage_but_low_iou():
  # A 20 x 20 box (area 400) fully contains the spot.
  # Coverage: 100 / 100 = 1.0. IoU: 100 / 400 = 0.25.
  assert coverage(SPOT, (-5, -5, 15, 15)) == pytest.approx(1.0)
  assert iou(SPOT, (-5, -5, 15, 15)) == pytest.approx(0.25)


def test_best_coverage_picks_the_biggest_overlap():
  boxes = [(0, 0, 2, 10), (0, 0, 8, 10)]  # 20% and 80%
  assert best_coverage(SPOT, boxes) == pytest.approx(0.8)
  assert best_coverage(SPOT, []) == 0.0


def test_is_taken_uses_threshold():
  assert is_taken(SPOT, [(0, 0, 6, 10)]) is True       # 60% >= 50%
  assert is_taken(SPOT, [(0, 0, 4, 10)]) is False      # 40% < 50%
  assert is_taken(SPOT, [(0, 0, 4, 10)], threshold=0.3) is True


def test_zero_area_spot_is_never_taken():
  flat = [(0, 0), (10, 0), (10, 0), (0, 0)]
  assert coverage(flat, (0, 0, 10, 10)) == 0.0
