# Final report map

The report is English, uses `Final_report_template.tex`, and is submitted as one PDF. The file name is Chinese and contains the name and student ID of every team member. Only one member uploads it.

Header in the template: “The final project report of computer vision, September 2026”.

Course line in the template: Macau University of Science and Technology, CS460/ EIE460/ SE460.

## Abstract

State the single-image height task, the vanishing-point pipeline, the self-captured scenes, the ruler protocol, and the comparisons (hand-marked lines versus detected lines, easy scene versus harder scene, focal-length check).

The last line of the abstract is the GitHub URL of this repository. That line is required.

https://github.com/Ricin-brony/single-view-metrology

## Keywords

single-view metrology; vanishing point; vanishing line; camera focal length; height measurement

## Introduction

Describe a single photograph, the ambiguity of height without a reference and a ground plane, the orthogonal-direction assumption, the datasets, and the experimental outcome. State what is specific to this project: an implemented estimator, a ruler comparison, and a split between line-detection error and formula error.

## Related work

Cover vanishing-point estimation, the Criminisi, Reid, and Zisserman height relation, and robust fitting of lines. Say what this pipeline shares with that method, and what it does not claim to add.

Cite each source in the paragraph that uses it, and list the same sources in References. The height formula in the technical section must cite Criminisi, Reid, and Zisserman, IJCV 2000.

## Technical solution

Include the formulas and a pipeline figure.

- Homogeneous line \(ax+by+c=0\) from least squares
- Vanishing point as the null space of stacked lines, inside RANSAC
- Ground vanishing line \(l=v_1\times v_2\), scaled so \(l_1^2+l_2^2=1\)
- Focal length \(f=\sqrt{-(v_1-p)^\top(v_2-p)}\) as a sanity check
- Height ratio with points \((x,y,1)\) and the normalized line, as written in the README

Use a three-line table if symbols need a table. The template shows that table style.

## Experiments

Name the scenes, the camera, and the objects. State that each height in the reference column was measured with a ruler, and give those measurements only after they exist. List every hyperparameter. Give the quantitative table and the qualitative figures from the README experiment section. Do not fill cells before the runs exist.

## Discussion

Strengths: the geometry is visible, the ruler gives an external number, and the hand-marked baseline separates detector error from formula error.

Limitations: a tilted object is not vertical, a wrong principal point biases \(f\), and a missed vanishing point biases every height. Lens distortion is left uncorrected in this one-week scope.

Possible next step: a radial-distortion estimate before the line fit. Describe it as future work, not as a completed result.

## Conclusion

Restate the problem, the pipeline, and the results that were actually measured.

## Description of member contributions

One concrete paragraph per member: which modules, which scenes, which report sections.

## References

Every citation in the text appears here. The template’s instruction is to cite in the text and to list the cited works in this section.

Include the Criminisi, Reid, and Zisserman IJCV 2000 paper if the height formula is used, which it is.

## Optional extra material

A screen recording of the test command, a short demo of the overlays, and the oral slides may be submitted for extra consideration. They are optional.
