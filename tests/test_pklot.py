from eval.pklot import parse_spots

SAMPLE = """
<parking id="test">
  <space id="1" occupied="1">
    <contour>
      <point x="10" y="20" />
      <point x="30" y="20" />
      <point x="30" y="40" />
      <point x="10" y="40" />
    </contour>
  </space>
  <space id="2" occupied="0">
    <contour>
      <point x="50" y="20" />
      <point x="70" y="20" />
      <point x="70" y="40" />
      <point x="50" y="40" />
    </contour>
  </space>
  <space id="3">
    <rotatedRect>
      <center x="50" y="50" />
      <size w="20" h="10" />
      <angle d="0" />
    </rotatedRect>
    <contour />
  </space>
</parking>
"""


def test_reads_id_occupied_and_corners():
  spots = parse_spots(SAMPLE)
  assert len(spots) == 3
  assert spots[0].id == 1
  assert spots[0].occupied is True
  assert spots[0].polygon == [(10, 20), (30, 20), (30, 40), (10, 40)]
  assert spots[1].occupied is False


def test_missing_occupied_is_none():
  assert parse_spots(SAMPLE)[2].occupied is None


def test_empty_contour_falls_back_to_rotated_rect():
  corners = parse_spots(SAMPLE)[2].polygon
  assert sorted(corners) == [(40, 45), (40, 55), (60, 45), (60, 55)]
