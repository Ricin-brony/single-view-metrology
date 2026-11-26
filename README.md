# Single-View Metrology from Orthogonal Vanishing Points

Self-defined final-project topic under Topic 4 of the course assignment.

From one photograph of a scene that contains two horizontal directions and a vertical direction, estimate the vanishing points, the vanishing line of the ground, and the height of an object standing on that ground. A second object of known height supplies the scale. The one-week version implements this classical projective pipeline and checks it against ruler measurements. It does not train a network.

This repository is the project webpage required by the final assignment. It records the plan for the oral presentation and every item the assignment asks the webpage to contain. The measuring program itself is the work of the coming week. No height, focal length, or error in this repository is invented.

Course: 2609 CS460 / EIE460 / SE460, Computer Vision, Macau University of Science and Technology.

## Team members and division of work

| Name | Student ID | Responsibility in the current plan |
| --- | --- | --- |
| 刘君威 Liu Junwei | 1230025933 | Project proposal, pipeline design, line and vanishing-point estimation, height formula, experiments, English report, and this webpage |

If the team has more members, add each name and student ID to this table before the oral presentation and before the PDF is submitted. The PDF file name must contain the Chinese names and student IDs of every member. Only one member submits that PDF.

Planned split if another member joins: one person owns capture, the ruler measurements, and the experiment table; the other owns the report figures and the webpage check. The split above is the plan while the team has one confirmed member.

## Research title and topic

**Title:** Single-View Metrology from Orthogonal Vanishing Points

**Topic:** Recover the height of a vertical object from a single image, using vanishing points of an orthogonal scene and one reference object whose height is measured with a ruler.

The scene is approximately a ground plane plus vertical objects, such as a room corner or a building facade. The camera intrinsic matrix is not calibrated beforehand. The principal point is taken as the image center, and the pixels are assumed square. Those assumptions are checked by the sign of the focal-length estimate. The height formula itself uses the vertical vanishing point, the ground vanishing line, and the reference height.

## Requirements: software

- Python 3.10 or newer
- Packages in `requirements.txt`: NumPy, OpenCV-Python, Matplotlib
- A browser for reading this page
- A ruler, or any object whose height can be measured in meters before the photo is taken
- No GPU is required

Create the environment and install the packages:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Pretrained models

This project does not train or download a neural network. There is no weight file. Edges, lines, vanishing points, and heights are classical geometry with the settings listed under Method.

`weights/` is reserved and currently empty. Testing does not load a checkpoint.

## Preparation for testing

1. Install the software above.
2. Measure the reference object with a ruler and write the height in meters. Measure every object that will appear in the error table the same way, before trusting the image estimate.
3. Put one scene in `data/scene1/` as described in `data/README.md`: the photograph, the reference height, and the base and top image points of each vertical object.
4. Create `results/` if it does not exist. The test command creates it when the program is in place.
5. Run the test command below from the repository root.

Photographs are not in this preliminary commit. The team will capture them in week one. Small images used in the report will be stored under `data/`. Large private originals can stay out of the repository.

## Command for testing

Planned command, to be implemented on day 2–5 of the one-week schedule:

```bash
python -m src.metrology --input data/scene1 --output results/scene1 --save-debug results/debug
```

The reference height is read from `data/scene1/reference.txt`. It is not hard-coded.

Expected behavior once the program exists:

- read the scene image and the annotation files
- write an overlay with the detected lines, the vanishing points, and the measured segments
- write a text table of estimated height against the ruler height for each annotated object
- print the vertical vanishing point, the ground vanishing line, the focal-length sanity check, and the relative height error of each object

This command is not runnable in the preliminary commit. `src/metrology.py` is the module layout and the comments for the key steps, not a finished measuring program.

## Dataset

The team will photograph 2 scenes with the same camera:

- one easy indoor corner, or a corridor, with clear horizontal edges and vertical edges, a reference object, and two other vertical objects whose heights are known from the ruler
- one harder scene with weak edges, a large depth range, or a tilted object, so the report can show where the vertical-object assumption fails

Each object used in the table stands on the same floor. The reference object is vertical. Image coordinates of each base point and top point are stored in the annotation file. No external labeled dataset is required.

## Method

The pipeline follows projective single-view metrology. The height equation is the one derived by Criminisi, Reid, and Zisserman (IJCV 2000), written below with the normalization the implementation will use. The report will derive it in the text and cite that paper.

1. **Gaussian smoothing**, then a gradient magnitude, so edge pixels are less sensitive to noise.
2. **Line hypotheses** from edge pixels. A line is the homogeneous least-squares fit \(ax+by+c=0\). RANSAC keeps a segment when inlier edge pixels lie within a pixel distance of the line.
3. **Two horizontal vanishing points.** Parallel scene lines in one horizontal direction meet at one vanishing point. That point is the null-space vector of the stacked line equations, again inside RANSAC so a stray line does not pull the intersection. The same step is repeated for the second horizontal direction and for the vertical direction, giving \(v_1\), \(v_2\), and \(v_z\).
4. **Ground vanishing line** \(l = v_1 \times v_2\). It is scaled so that \(l_1^2+l_2^2=1\). With image points written \((x,y,1)\), \(l \cdot p\) is then the signed pixel distance from the point to the line.
5. **Focal-length check**, not a required input of the height formula. With the principal point \(p\) at the image center and square pixels,
   \[
   f=\sqrt{-(v_1-p)^\top(v_2-p)}
   \]
   when the quantity under the square root is positive. A non-positive value means the orthogonal-direction assumption or the principal-point assumption failed on that photo. The program records the value and still computes heights.
6. **Reference scale and object height.** For a vertical reference with base \(b_r\), top \(t_r\), and known height \(H_r\), and for a vertical object with base \(b\), top \(t\), and unknown height \(H\),
   \[
   H = H_r \frac{\lVert t \times b \rVert \; \lVert v_z \times t_r \rVert \; (l \cdot b_r)}{\lVert t_r \times b_r \rVert \; \lVert v_z \times t \rVert \; (l \cdot b)}.
   \]
   All points are homogeneous \((x,y,1)\), \(l\) uses the scaling in step 4, and \(\lVert\cdot\rVert\) is the Euclidean norm. The same equation is applied to hand-marked lines and to automatically detected lines, so the experiment can separate an error in the formula from an error in the line detector.

OpenCV is used to read and write images and for the Gaussian convolution. Line fitting, vanishing points, and the height equation are project code.

## Experiments

| Comparison | What changes | What is recorded |
| --- | --- | --- |
| Line source | hand-marked segments versus RANSAC segments | vanishing-point location, height of each object |
| Object | each ruler-measured vertical object | absolute error in meters and relative error |
| Scene | indoor orthogonal corner versus the harder scene | whether \(f^2\) is positive and whether the height error stays small |

Relative error for an object is \(|H_{\mathrm{est}}-H_{\mathrm{ruler}}|/H_{\mathrm{ruler}}\).

Fixed settings to write down before running, and to keep unchanged across the table: Gaussian \(\sigma\), gradient threshold, RANSAC pixel threshold for a line, minimum segment length, and RANSAC threshold for a vanishing point. These are the hyperparameters. There is no learning rate.

Hand-marked endpoints are a second input to the same equation. They are not a hidden correction applied after the automatic result.

## Expected outcomes and how they will be shown

- one overlay per scene showing lines, vanishing points, and the vertical segments
- a table of ruler height, estimated height, and relative error for each object
- the same table for hand-marked lines and for detected lines
- the focal-length sanity value, including a scene where it is not meaningful
- a short statement of which assumption failed when the harder scene is wrong: lines, orthogonality, or a non-vertical object

The oral presentation uses these as the expected outcomes. Measured numbers will be added only after the experiments run.

## How the rubric is addressed

- **Difficulty:** one photograph, an implemented vanishing-point estimator, a projective height formula, and a scene that should break the assumptions.
- **Innovation:** the comparison asks how much of the height error comes from the line detector and how much remains when the lines are marked by hand. The focal length is a check on the orthogonal assumption, separate from the height number.
- **Experimental completeness:** the table above is the full experiment list for the week. Every test object has a ruler measurement taken by the team.
- **Paper structure:** section headings below match the report template.
- **Code:** each key step has a commented function slot in `src/metrology.py`.
- **Webpage:** software, pretrained models, test preparation, test command, and environment are sections of this file.

## Report checklist

The English PDF uses `Final_report_template.tex`. Planned section content:

- **Title:** Single-View Metrology from Orthogonal Vanishing Points
- **Authors:** name and MUST ID of every member
- **Abstract:** task, classical vanishing-point pipeline, self-captured scenes with ruler heights, and the comparisons above. The last line of the abstract is the GitHub URL of this repository: https://github.com/Ricin-brony/single-view-metrology
- **Keywords:** single-view metrology; vanishing point; vanishing line; camera focal length; height measurement
- **Introduction:** why one image does not determine height without a scene assumption, the orthogonal ground-plane assumption, and what the experiments compare
- **Related work:** vanishing points, the Criminisi height relation, and robust line estimation, with what this pipeline follows and what it does not add
- **Technical solution:** formulas for the line fit, the vanishing point, the vanishing line, the focal-length check, and the height ratio; a pipeline diagram
- **Experiments:** scenes, every hyperparameter, the ruler protocol, the quantitative table, and the qualitative figures
- **Discussion:** strengths, sensitivity to the line detector, and the failure of a tilted object or a wrong principal point
- **Conclusion:** problem, method, and results
- **Member contributions:** one paragraph per person
- **References:** every paper or lecture source cited in the text is listed and cited at the point of use

PDF file name, in Chinese, containing every member’s name and student ID. One member uploads it. Do not submit a second copy from another member.

Optional extra material after the system runs: a screen recording of the test command, and the oral slides.

## One-week schedule

| Day | Work |
| --- | --- |
| 1 | Capture both scenes. Measure every test object with a ruler. Mark base and top points. |
| 2 | Gradient edges and RANSAC line segments. Save the edge and line figures. |
| 3 | Three vanishing points, the ground vanishing line, and the focal-length check. |
| 4 | Height formula on hand-marked lines, then on the detected lines. |
| 5 | Run the full table on both scenes. Export figures. |
| 6 | Write the English report from the template. Fill measured numbers only from day 5. |
| 7 | Check this page against the assignment list, record the test command output, and submit the single PDF. |

## Lecture material used in the pipeline

- Homogeneous coordinates, least squares, and a null-space fit for a line and for a vanishing point
- Vanishing points, the vanishing line of a plane, and RANSAC
- A pinhole camera, the principal point, and focal length from two orthogonal vanishing points
- A projective invariant that turns one known height into another height
- Gaussian smoothing and gradient magnitude before line extraction

Lectures on stereo matching, convolutional networks, detection, and sequence models are outside the implemented system. They stay in related work only if a sentence needs them. The week is reserved for a finished height estimate and a complete comparison against the ruler.

## Key module comments

See `src/metrology.py`. Each function states its input, output, and the course idea it implements. The bodies will be filled during the week.

## Repository layout

```text
README.md                 this webpage
requirements.txt          Python packages
src/metrology.py          pipeline functions
docs/oral_presentation.md five-minute oral script
docs/report_map.md        report template mapped section by section
data/                     photographs and ruler annotations, added during the week
results/                  outputs, not committed until they exist
weights/                  unused; explained above
```
