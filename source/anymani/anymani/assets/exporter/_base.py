"Defines common exporter protocols and output records."

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExportResult:
    "Result record with ordered geometry, provenance, and validation status."

    written: list[Path] = field(default_factory=list)
    "Whether the declared artifact was written successfully."

    skipped: list[Path] = field(default_factory=list)
    "Items omitted by an explicit geometry support policy."

    errors: list[tuple[Path, str]] = field(default_factory=list)
    "Explicit candidate rejection or validation errors."

    def merge(self, other: "ExportResult") -> "ExportResult":
        "Combines sub-export results while retaining failed-stage evidence."

        self.written.extend(other.written)
        self.skipped.extend(other.skipped)
        self.errors.extend(other.errors)
        return self

    @property
    def ok(self) -> bool:
        "Reports whether every required export stage succeeded."

        return len(self.errors) == 0

    def __bool__(self) -> bool:
        return self.ok


class ExporterBase:
    "Shared export protocol that reports success or a structured failure."

    def export(self, target: object, output_dir: Path) -> ExportResult:
        "Writes the declared hand component and its metadata to the output bundle."

        raise NotImplementedError


__all__ = ["ExportResult", "ExporterBase"]
