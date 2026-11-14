from worker.detector import keep_vehicles

BOXES = [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50], [60, 60, 70, 70]]


def test_keeps_cars_trucks_and_buses():
  # COCO ids: 2 = car, 7 = truck, 5 = bus, 0 = person
  found = keep_vehicles(BOXES, [2, 7, 5, 0], [0.9, 0.9, 0.9, 0.9])
  assert [d.label for d in found] == ["car", "truck", "bus"]


def test_drops_low_confidence():
  found = keep_vehicles(BOXES[:2], [2, 2], [0.9, 0.05])
  assert len(found) == 1
  assert found[0].box == (0, 0, 10, 10)
  assert found[0].confidence == 0.9


def test_custom_confidence_cut_off():
  assert len(keep_vehicles(BOXES[:2], [2, 2], [0.9, 0.05], min_confidence=0.01)) == 2


def test_nothing_in_nothing_out():
  assert keep_vehicles([], [], []) == []
