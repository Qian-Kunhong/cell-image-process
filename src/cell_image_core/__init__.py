"""Project-neutral kernels for cell-image analysis.

This package contains deterministic computation only. Dataset discovery,
biological labels, model choices and presentation remain in project adapters.
"""

from .contracts import FeatureTable, RunArtifact, SampleRecord, SegmentationResult

__all__ = [
    "FeatureTable",
    "RunArtifact",
    "SampleRecord",
    "SegmentationResult",
]

__version__ = "0.1.0"

