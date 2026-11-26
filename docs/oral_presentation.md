# Five-minute oral presentation

One member presents. Stay within 5 minutes. Upload the slide file to Moodle. The four headings below are the four required parts of the oral presentation guidelines.

## 1. Team members and division of work

- 刘君威 Liu Junwei, student ID 1230025933: proposal, implementation, experiments, English report, and the GitHub page.
- Add every other member’s name, student ID, and task on this slide before the presentation. Do not leave a required member off the slide or off the later PDF file name.

## 2. What the final project plans to do

**Title:** Single-View Metrology from Orthogonal Vanishing Points

**Task:** from one photograph, estimate the height of a vertical object standing on the ground. The scale comes from a second object whose height was measured with a ruler.

This is a self-defined topic under Topic 4. The scene has two horizontal directions and a vertical direction, such as a room corner. The method uses vanishing points and a projective height formula.

Project page: https://github.com/Ricin-brony/single-view-metrology

## 3. General research methodology

**Dataset.** The team will photograph the scenes and measure the objects. An easy scene is an indoor corner with clear edges, one reference object, and two other vertical objects. A harder scene has weak lines, a large depth change, or an object that is not vertical. Base and top image positions are annotated. Heights used in the error table come from the ruler, not from the image.

**Method used as the basis.**

1. Smooth with a Gaussian and compute the gradient magnitude.
2. Fit line segments with homogeneous least squares inside RANSAC.
3. Estimate two horizontal vanishing points and one vertical vanishing point.
4. Take the ground vanishing line as the cross product of the two horizontal vanishing points.
5. Check the focal length from the orthogonal pair, with the principal point at the image center. Record the check even when the value is not meaningful.
6. Apply the Criminisi height ratio to each vertical object. Run that same ratio on hand-marked lines and on detected lines.

OpenCV reads and writes images. Line fitting, vanishing points, and the height equation are written in the project so the geometry is visible.

**Experiments.** For each object, record the ruler height, the estimated height, and the relative error. Repeat with hand-marked lines and with RANSAC lines. Run both the indoor scene and the harder scene. Report the focal-length sanity value beside the table.

There is no training loop and no learning rate. The hyperparameters are the Gaussian sigma, the gradient threshold, the line RANSAC threshold, the minimum segment length, and the vanishing-point RANSAC threshold. They will be fixed before the comparison runs.

## 4. Expected outcomes

- an overlay of lines, vanishing points, and vertical segments for each scene
- a table of ruler height, estimated height, and relative error
- the same table for both line sources
- the focal-length check, including a case that shows a failed orthogonal assumption
- a visual example of a tilted object or a poor line set, together with which assumption failed

Performance is evaluated by relative height error against the ruler. Results are presented as the overlay, the two error tables, and the focal-length check.

## Timing

| Minute | Content |
| --- | --- |
| 0:00–0:30 | Members and tasks |
| 0:30–1:30 | Title and the single-image measuring task |
| 1:30–3:30 | Data, six pipeline steps, experiment table |
| 3:30–5:00 | Figures and numbers the team expects to show |

Speak from this plan. Do not quote a height or an error until the experiments have been run.
