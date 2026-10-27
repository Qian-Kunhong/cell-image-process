# Cellpose Segmentation and Border-Object Removal

This package segments 2D microscopy images with Cellpose and automatically
removes cells or nuclei that touch the image border.

## Files

- `cellpose_segment_remove_border.py`: main program
- `requirements.txt`: required Python packages

## Supported systems

- macOS (Intel or Apple Silicon)
- Windows
- Linux

CPU processing is used by default and is recommended on macOS.

## Installation

Open Terminal and create a Python virtual environment:

```bash
python3 -m venv cellpose_env
source cellpose_env/bin/activate
```

Install the required packages:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Installing the standalone Cellpose GUI application does not always install the
`cellpose` package into Terminal's Python environment. Run the commands above
even if the GUI is already installed.

## Run on macOS

Open Terminal, change to this package directory, and run:

```bash
python3 cellpose_segment_remove_border.py \
  "/Users/your_name/path/to/input_images" \
  "/Users/your_name/path/to/output"
```

Replace the first path with the folder containing the original images and the
second path with the folder in which results should be saved. You can drag a
folder from Finder into Terminal to insert its full path automatically.

Example:

```bash
python3 cellpose_segment_remove_border.py \
  "/Users/amy/Desktop/cell_images" \
  "/Users/amy/Desktop/cellpose_results"
```

## Optional arguments

Overwrite existing results:

```bash
python3 cellpose_segment_remove_border.py INPUT_DIR OUTPUT_DIR --overwrite
```

Remove only objects touching the outermost image pixels:

```bash
python3 cellpose_segment_remove_border.py INPUT_DIR OUTPUT_DIR --border-margin 0
```

Remove objects within five pixels of the border:

```bash
python3 cellpose_segment_remove_border.py INPUT_DIR OUTPUT_DIR --border-margin 5
```

Use a configured CUDA GPU on Windows or Linux:

```bash
python cellpose_segment_remove_border.py INPUT_DIR OUTPUT_DIR --gpu
```

The `--gpu` option is not required on macOS. Use the default CPU mode.

## Supported input images

- TIFF (`.tif`, `.tiff`)
- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)

The input folder and all its subfolders are searched automatically. The script
processes 2D images; unprojected 3D Z-stacks are not supported.

## Output

The output directory contains:

```text
output/
├── masks_npy/   NumPy label arrays for Python analysis
├── masks_tif/   Label images for Fiji/ImageJ
└── previews/    Original images with retained boundaries shown in red
```

In each mask, `0` is background and `1, 2, 3, ...` identify separate objects.
Objects touching the selected border region are removed before saving.

## Cellpose GUI masks

This program segments the original images itself; it does not require masks
exported from the Cellpose GUI.

In the GUI, `Command + S` saves a project as `*_seg.npy`. To delete one mask
manually on a Mac, hold `Command` and left-click the mask, then save again.

## Troubleshooting

Show all available command-line options:

```bash
python3 cellpose_segment_remove_border.py --help
```

If the program reports that no images were found, check that the input path is
correct and that the files use one of the supported extensions.

