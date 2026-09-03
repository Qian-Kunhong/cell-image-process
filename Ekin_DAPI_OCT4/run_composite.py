"""PyCharm entry: composite DAPI features. Edit MAGNIFICATION or pass CLI options."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.entrypoint import configured_args
from day2_trial import parse_args, run_trial

MAGNIFICATION = "40x"  # Change to "20x" for a separate fit.

if __name__ == "__main__":
    args = configured_args(parse_args, Path(__file__).resolve().parent, "composite", MAGNIFICATION)
    run_trial(args)
