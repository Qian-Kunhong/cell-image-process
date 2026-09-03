"""Small reusable launcher: fixed feature choice, explicit CLI overrides."""
from pathlib import Path
import sys


def configured_args(parse_args, directory, feature_set, magnification="40x", argv=None):
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    # The named entry owns feature choice; do not silently run the other model.
    if argv is None:
        argv = sys.argv[1:]
    if any(x == "--feature-set" or x.startswith("--feature-set=") for x in argv):
        if args.feature_set != feature_set:
            raise ValueError(f"This entry runs {feature_set}; use the matching entry for {args.feature_set}")
    if not any(x == "--fit-magnification" or x.startswith("--fit-magnification=") for x in argv):
        args.fit_magnification = magnification
    args.feature_set = feature_set
    if not any(x == "--output-root" or x.startswith("--output-root=") for x in argv):
        # OCT4 day/sample are part of the key so distinct experiments do not overwrite.
        scope = "all_fields"
        if hasattr(args, "culture_day"):
            import re
            sample = re.sub(r"[^A-Za-z0-9_-]+", "_", args.sample)
            replicate = re.sub(r"[^A-Za-z0-9_-]+", "_", args.replicate)
            scope = f"day{args.culture_day}_{sample}_{replicate}"
        args.output_root = Path(directory) / "outputs" / feature_set / args.fit_magnification / scope
    return args
