# Scene data

Photographs are added during the week. This preliminary commit has no images and no measured heights.

Each scene is one folder.

```text
data/scene1/image.jpg
data/scene1/reference.txt
data/scene1/objects.txt
data/scene1/lines.txt          optional hand-marked segments
```

`reference.txt` contains one number, the ruler height of the reference object in meters. Write that number only after the ruler measurement. Do not copy a sample value into the file.

`objects.txt` has one object per line. The columns are the name, the base pixel, the top pixel, and the ruler height in meters:

```text
name base_x base_y top_x top_y height_m
```

The reference object is included in this file so the program can resolve its base and top. `lines.txt`, when present, has hand-marked segments `x1 y1 x2 y2`, one per line, plus a direction label `h1`, `h2`, or `v` at the end of the line. The automatic detector does not read this file. The comparison run does.

Use the same camera for both scenes. Keep the original pixel size. Record the ruler values in the text files before looking at the program's estimate.

`data/raw/` is gitignored for extra captures that are not needed to reproduce the report.
