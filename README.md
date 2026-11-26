# Single-View Metrology from Orthogonal Vanishing Points

Estimate the metric height of a vertical object in one photograph using perspective geometry and a reference object of known height. The primary goal is fully automatic measurement in a prepared, marker-equipped scene. The planned tool will show its geometry and quality decision, and export a measurement record that another person can replay.

**Status:** The repository currently contains the Python module skeleton and project documentation. The measurement pipeline, demo, and experimental results are planned.

This is a self-defined final project for CS460 / EIE460 / SE460 Computer Vision at Macau University of Science and Technology.

## What the project does

**Inputs.** One photograph of a prepared scene and a saved setup profile with tag roles and the known reference height. Four flat tags define a rectangle on the support plane. Tags at the reference and target endpoints identify the points to measure. The reference and targets are vertical and share that support plane. Manual and assisted modes can supply the same points and lines for unmarked images.

**Method.** Detect ArUco IDs and corners, recover the two directions of the support plane and the vertical direction, then transfer the reference scale to the target position. The target's physical height is used only for evaluation. Tag IDs and image locations provide geometry, and only the reference height supplies the metric scale. Level and pitched cameras share the same homogeneous estimator, including valid vanishing points at infinity. Focal length remains an optional diagnostic for suitable finite orthogonal points. [OpenCV ArUco detector](https://docs.opencv.org/4.12.0/d2/d1a/classcv_1_1aruco_1_1ArucoDetector.html) · [Single-view height relation](https://www.robots.ox.ac.uk/~vgg/publications/2000/Criminisi00a/)

**Outputs.** The planned result includes the height in centimeters or meters, an annotated image, and a quality status with reasons. Batch evaluation will record estimates, errors, and failed measurements. A sensitivity analysis will show how small changes in annotations affect the estimate.

| Mode | User interaction | Purpose |
| --- | --- | --- |
| Automatic | Select a photo and reuse the saved setup profile. | Measure a valid prepared scene with zero per-photo annotations. Return actionable failure reasons when the inputs cannot support a reliable estimate. |
| Manual | Mark scene lines and object endpoints. | Establish a reproducible geometric baseline and give the reviewer direct control. |
| Assisted | Confirm or correct candidate directions and mark object endpoints. | Reduce line annotation effort while keeping the geometry visible and editable. |

## Capture equipment and setup

The team has selected the following setup for the automatic measurement experiments. Measure the assembled reference object before capture and record the first usable photographs during testing.

| Item | Selected setup | Status |
| --- | --- | --- |
| Camera | Xiaomi 13 rear **1× main camera**. Use normal 4:3 still photos and retain the originals. | On hand |
| Tripod and lighting | AT005 desktop tripod with ball head, landscape phone clamp, and fill-light kit. Use room lighting with fill where shadows obscure the markers. | Selected for purchase |
| Common support plane | One **60 × 80 cm, 5 mm** matte white rigid PVC foam board. Place both boxes and all six flat markers on it. | Selected for purchase |
| Reference and target | Rigid, flat-bottom cardboard boxes, each with a **20 × 20 cm** footprint. The nominal heights are **20 cm** for the reference and **30 cm** for the initial target. | Selected for purchase |
| Physical measurements and mounting | Tape measure for the actual heights. Removable tape or a flat scrap of card can secure and support the paper markers when needed. | Tape measure on hand |
| Printable markers | [ArUco marker sheets, IDs 0–7](docs/assets/aruco-4x4-50-80mm.pdf). Print three complete sets on single-sided black-and-white A4 at **100% actual size** (12 sheets total). Each cutout is **10 × 10 cm**, with an 8 cm black square and 1 cm white margins. | PDF ready for printing |

The initial site is a **70 × 130 cm** indoor floor area, with an unobstructed corridor as an alternate. Align the board's 80 cm edge with the 130 cm direction to leave about 50 cm for the camera position. Place the boxes side by side behind the flat marker rectangle. Start with rectangle center spacings of **44 × 35 cm**. Adjust the camera position or pitch until all eight markers are sharp, visible in full, and clear of reflections and obstructions. For the perspective tests, also stagger the boxes at different distances from the camera while keeping their bases and the flat markers on the same plane. Keep the setup temporary and clear the walkway after each capture session.

| Marker IDs | Placement |
| --- | --- |
| `0`, `1`, `2`, `3` | Flat on the board at the rear left, rear right, front right, and front left, viewed from the camera. Their centers define the rectangle. |
| `4`, `5` | Reference box base and top. |
| `6`, `7` | Target box base and top. |

All eight marks use OpenCV `DICT_4X4_50`. Use each ID once in the scene. Lay each base tag flat in front of its box and mount each top tag vertically on the corresponding front face. Align the **printed upper-left corner of the black square** on each base and top tag with the actual lower and upper endpoints of the same vertical line. Confirm the box stands square and measure the reference height on this line. Record the target's measured height separately for evaluation. See the [scene format](data/README.md) for the saved setup and [OpenCV's ArUco documentation](https://docs.opencv.org/4.12.0/de/d67/group__objdetect__aruco.html) for the dictionary.

## Experiments

- **Geometry checks.** Use known projections to verify the estimator for level and pitched cameras, finite and infinite vanishing points, and degenerate inputs.
- **Public development examples.** Use documented photographs with known reference dimensions and traceable target measurements. [Oxford library example, Figure 21](https://www.robots.ox.ac.uk/~vgg/publications/2000/Criminisi00a/criminisi00a.pdf)
- **Independent captured examples.** Use a tripod for consistent framing and independently measure the targets. Reuse each object's measured height across photographs. Freeze the method and parameters before evaluating the reserved scenes.
- **Controlled comparisons.** Compare the direct pixel-height ratio, manually annotated geometry, and fully automatic marker-based geometry on the same prepared photographs. Hold endpoints fixed in a separate experiment to isolate line estimation. Report height errors, zero-click success rate, unsuccessful cases with reasons, and annotation effort. Evaluate unmarked images separately with the manual or assisted modes.

## Team and presentation

| Team member | Student ID | GitHub | Planned responsibilities |
| --- | --- | --- | --- |
| 刘军威 | 1230025933 | [Ricin-brony](https://github.com/Ricin-brony) | Projective geometry, vanishing-point estimation, and the measurement pipeline. |
| 许彦熙 | 1230023419 | [cdwfb](https://github.com/cdwfb) | Scene preparation, independent measurements, the pixel-ratio baseline, and evaluation. |

Both members review the automatic demo, reproduce the evaluation, and contribute to the presentation and report.

The **December 10 Pre** introduces the project. The presentation covers the task, one example of the input and expected output, the method, the experiment design, and the two members' responsibilities. [Pre content outline](docs/oral_presentation.md)

## Requirements: software

The planned implementation targets Python 3.10 or later with NumPy, OpenCV-Python, and Matplotlib. See [requirements.txt](requirements.txt) for the dependency ranges. The geometry pipeline runs on a CPU.

Create an environment with `python -m venv .venv`. Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS and Linux. Then run `python -m pip install -r requirements.txt`.

## Pretrained models

The project uses classical vision and projective geometry. It does not train a model or load checkpoints. [Model and weight notes](weights/README.md)

## Preparation for testing

1. Prepare a scene with a known reference height. Save the marker profile for automatic runs or mark its geometry for a manual run. Record target ground truth separately if available. [Scene format](data/README.md)
2. The planned reproducibility check will use a versioned synthetic example and a documented configuration. A clean environment should be able to regenerate the same overlay and measurement table.
3. These commands describe the planned interfaces. They will become the supported quick start when the implementation and examples are available.

```bash
python -m src.metrology --input data/example --output results/example --save-debug results/example/debug
python -m src.metrology --mode auto --input data/rigged-example --output results/rigged-example --save-debug results/rigged-example/debug
```

The planned outputs are `overlay.png`, `heights.csv`, and `run.json`. The run record will preserve the input annotations, configuration, line source, and quality reasons used to obtain the result.

## Acceptance criteria

- With a saved setup profile and a valid photograph of a prepared scene, the automatic mode produces a height and the three output files with zero per-photo annotations.
- A missing or ambiguous required marker returns a clear failure reason. The measured height is empty for a rejected target.
- The automatic and manual modes share the same geometry and retain the inputs needed to replay a result. The held-out evaluation reports both height error and zero-click success rate, including failed cases.
- The provisional team targets for valid held-out scenes are at least **90% zero-click success** and a **median absolute relative height error of at most 5%** among successful cases. Confirm and freeze the test conditions and targets using development captures before running the held-out set.

## Development and report

The measurement goals, data contract, and acceptance criteria guide the implementation. The initial Python skeleton is a starting point. Replace individual components or rebuild the pipeline where that better serves these requirements. Record interface changes and verify the relevant examples against the acceptance criteria.

- [Development order and acceptance criteria](docs/development_plan.md)
- [Data and annotation contract](data/README.md)
- [Final report map](docs/report_map.md)
