# Scene data and annotation contract

This is the planned contract for the measurement and replay commands. Prepare publishable synthetic examples, use documented public images to develop and debug, and reserve independently captured marker scenes for acceptance.

## Manual and assisted scene files

```text
data/example/
  image.jpg
  reference.txt
  objects.txt
  lines.txt
  truth.txt
  scene.json
```

| File | Planned format |
| --- | --- |
| `image.jpg` | The image in the same orientation and dimensions used for annotation. Store point coordinates against this version. |
| `reference.txt` | One line with `object_id height_m`. The height must be finite and positive, and the ID must match exactly one row in `objects.txt`. |
| `objects.txt` | One row per physical object with `object_id base_x base_y top_x top_y`. The reference object appears once. Use unique IDs and full-resolution pixel coordinates. |
| `lines.txt` | Manual mode uses one row per segment with `x1 y1 x2 y2 label`, where the label is `h1`, `h2`, or `v`. In assisted mode, the user-confirmed line groups can be exported in the same format for replay. |
| `truth.txt` | Optional held-out target heights, one row with `object_id measured_height_m` per target. Evaluation reads this file after the estimates are produced. |
| `scene.json` | Source, attribution or capture record, image dimensions, resize and orientation history, and split label (`synthetic`, `development`, or `acceptance`). Record the known origin of each reference and truth value. |

Coordinate origin is the top-left pixel. Record meters in input and machine-readable outputs. Use centimeters in presentation figures and tables as needed. Each group requires distinct projected lines. Collect redundant lines from different physical edges when assessing stability.

## Automatic scene profile

An automatic run reads `image.jpg`, `reference.txt`, and a reusable `rig.json` profile. It derives object endpoints and line groups without reading manual `objects.txt` or `lines.txt` for detection. Export the derived annotations in the same scene format for inspection and replay.

- `rig.json` records `dictionary`, `plane_tag_ids` in physical clockwise order around the rectangle, and `objects`, a list mapping each `object_id` to its `base` and `top` tag IDs and designated canonical corner indices. Extra vertical support lines may be configured separately. `reference.txt` holds the reference ID and its known height. The profile contains no target heights or physical tag sizes.
- Put the four plane tags flat on the shared support plane, with their physical centers at the corners of a rectangle. Recover each image center as the intersection of the tag's diagonals, then connect opposite sides to find the two vanishing directions. Exact rectangle dimensions are unnecessary.
- Align each selected base-tag corner to the object's actual base on that plane and each top-tag corner to the actual height endpoint. The base tag can lie flat next to the foot with its selected corner aligned to the vertical measurement axis. Check the alignment with the object upright and leave enough visible border to detect each tag. The configured corner is defined in the printed tag's orientation so image rotation leaves its role unchanged. Record any rig whose placement uses different endpoints as a separate measured segment.
- Missing or duplicate role IDs, an invalid plane quadrilateral, and a weak vertical estimate get explicit failure reasons. Keep the detected corners, the quality decision, and the derived points in the run record. A failed automatic run can be replayed in manual mode with corrected annotations.

Capture acceptance scenes with a phone tripod and record camera pitch and roll. Keep the markers in frame and add controlled case variations for distance, mild tilt, and partial occlusion. The reference and each target use the same support plane. Reuse an independently measured dimension for repeat photographs of the same object.

Keep large or private images under the ignored `data/raw/` directory. Commit the team's shareable synthetic inputs and annotated examples with suitable reuse terms. For examples distributed elsewhere, document the original source, image version, and how to obtain it.
