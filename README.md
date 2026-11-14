# Smart Parking Finder

Real-time parking availability for a large campus lot: cameras decide which spots are taken, gate counts and crowd answers cover what cameras cannot see.

**Status:** Week 1, session 1 done. Repo set up and PKLot downloaded and extracted. Next: session 2 (read the XML labels, draw the spots).

Plan: see `plan.html` (interactive) in this folder.

## Setup

Dependencies are listed in `pyproject.toml` and pinned in `uv.lock`. Install them with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                       # creates .venv with exactly what pyproject.toml lists
uv run pytest -q              # run the tests
uv add <package>              # add a new dependency (updates pyproject.toml and uv.lock)
```

## Layout

| Folder | Purpose |
|---|---|
| `data/` | Datasets and spot files. Not committed. |
| `worker/` | Detector and geometry code. |
| `eval/` | Evaluation scripts and `report.md`. |
| `tests/` | Unit tests. |

## Datasets

### PKLot

- Source: https://web.inf.ufpr.br/vri/databases/parking-lot-database/
- Downloaded: the official `PKLot.tar.gz` (4,898,276,304 bytes, SHA-256 `e89bbc1dc735298c478688d50c7a682fb3b0076a87b6634923132709f2d2fa9b`, computed here; the site publishes no checksum). Extracted to `data/pklot/PKLot/` (not committed).
- `PKLot/PKLot/`: 12,417 full frames (1280x720) with one XML file each. Layout: `<lot>/<weather>/<date>/<date>_<hh>_<mm>_<ss>.jpg` and `.xml`.
  - Lots: `PUCPR`, `UFPR04`, `UFPR05`. Weather: `Sunny`, `Cloudy`, `Rainy`. So lot and weather come from the folder path, and the capture time from the file name.
  - Spaces per image are fixed per lot: PUCPR 100, UFPR04 28, UFPR05 40 (checked on the first 300 files of each).
  - One PUCPR/Sunny image has no XML (2,315 images, 2,314 XML files).
- XML: one `<space id=".." occupied="0|1">` per spot, holding a `<rotatedRect>` (center, size, angle) and a `<contour>` of four `<point x y>` corners. The contour is the spot polygon, so no trigonometry is needed.
- `PKLot/PKLotSegmented/`: about 695,900 single-spot crops sorted into `Empty` and `Occupied`, for the optional classifier (F12).
- Licence: CC BY 4.0 (the archive's `licence` file). The dataset web page also shows "Copyright. All rights reserved." in its footer. Cite the paper.
- Citation: Almeida, P., Oliveira, L. S., Silva Jr, E., Britto Jr, A., Koerich, A., "PKLot - A robust dataset for parking lot classification", Expert Systems with Applications, 42(11):4937-4949, 2015.
- A Roboflow re-export (640x640 stretched, COCO boxes) was tried first and discarded: its boxes are looser than the original polygons and the stretch distorts the cars.

### CNRPark+EXT

- Source files named in the author's code repository (https://github.com/fabiocarrara/deep-parking): `CNRPark-Patches-150x150.zip` (36.6 MB) and `CNR-EXT-Patches-150x150.zip` (449.5 MB), hosted on cnrpark.it.
- These are 150x150 **patches of single spots**, not full camera frames, as far as the README shows. Useful for the optional classifier (F12), not for replaying a camera.
- The cnrpark.it page could not be opened (certificate error), so its licence and terms are unknown. Cite: Amato et al., "Deep learning for decentralized parking lot occupancy detection", Expert Systems with Applications, 72:327-334, 2017.

## Results

_None yet. The first accuracy table comes from `eval/run.py` in session 4._

## Known limits

_To be written as they are found._
