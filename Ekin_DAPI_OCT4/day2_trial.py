"""Compatibility entry for the OCT4 Model A workflow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.dapi_model_a import *

if __name__ == "__main__":
    run_trial(parse_args())
