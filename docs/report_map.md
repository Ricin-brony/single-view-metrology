# Final report map

Use the course LaTeX template and write the report in English. Submit one PDF per team with a Chinese file name containing both names and student IDs.

| Section | Content |
| --- | --- |
| Abstract | State the problem, method, data, and measured results. Put the team GitHub URL on the final line. |
| Introduction | Explain scale ambiguity in a single image, the reference height, and the questions tested. |
| Related work | Cite the original single-view metrology method, fiducial detection, and the line estimation used for unmarked images. |
| Technical solution | Show the tag profile, automatic measurement pipeline, manual baseline, homogeneous coordinates, geometry checks, and quality states. Explain the source of the metric scale and document the optional focal-length check. |
| Experiments | Describe synthetic checks, public development examples, and held-out marker-equipped captures. Record shared inputs, ground truth sources, thresholds, height errors, zero-click success rate, and rejection reasons. Report unmarked images separately. |
| Discussion | Explain perspective, annotation error, weak line groups, lens distortion, and departures from the assumed geometry. Discuss which results support each claim and how rejected cases affected coverage. |
| Conclusion | Summarize the completed method, evidence, and practical limitations. |
| Member contributions | List both members' actual implementation, data, review, and writing contributions. |
| References | Cite every external method, image, and data source at its point of use and in the bibliography. |

Use the output figure, a compact comparison table, and an annotated failure case. If the report shows a perturbation interval, state the assumed noise model and its limitations. Report empirical coverage on cases with ground truth if evaluated. [Development acceptance criteria](development_plan.md)

Course template header and formatting follow `Final_report_template.tex`. An offline demo recording is optional.
