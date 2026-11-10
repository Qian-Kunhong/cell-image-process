"""Small data contracts shared by project adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SampleRecord:
    """A project-neutral image record.

    Marker meaning is deliberately kept in ``channels`` instead of hard-coded
    fields so DAPI/OCT4 and DAPI/YAP projects can use the same IO contract.
    """

    sample_id: str
    image_id: str
    dapi_path: Path
    channels: Mapping[str, Path] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id or not self.image_id:
            raise ValueError("sample_id and image_id must be nonempty")
        object.__setattr__(self, "dapi_path", Path(self.dapi_path))
        object.__setattr__(
            self,
            "channels",
            {str(name): Path(path) for name, path in self.channels.items()},
        )


@dataclass(frozen=True)
class SegmentationResult:
    image_id: str
    labels: np.ndarray
    algorithm: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must be a 2D integer array")
        if np.any(labels < 0):
            raise ValueError("labels must be nonnegative")
        object.__setattr__(self, "labels", labels)


@dataclass(frozen=True)
class FeatureTable:
    frame: pd.DataFrame
    unit: str = "cell"

    def __post_init__(self) -> None:
        required = {"image_id", "label"}
        missing = sorted(required.difference(self.frame.columns))
        if missing:
            raise ValueError(f"feature table is missing required columns: {missing}")
        if self.unit not in {"cell", "field", "well"}:
            raise ValueError(f"unsupported analysis unit: {self.unit}")


@dataclass(frozen=True)
class RunArtifact:
    path: Path
    kind: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("artifact kind must be nonempty")
        object.__setattr__(self, "path", Path(self.path))

