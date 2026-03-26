from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Tuple


BoundingBox = Tuple[float, float, float, float]


@dataclass
class TextBlockArtifact:
    block_id: str
    section_id: str
    page_number: int
    order: int
    text: str
    bbox: BoundingBox | None = None
    description: str = ""
    keep: bool = True
    role: str = "text"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TableArtifact:
    table_id: str
    section_id: str
    page_number: int
    order: int
    n_rows: int
    n_cols: int
    headers: list[str] = field(default_factory=list)
    preview_rows: list[list[str]] = field(default_factory=list)
    table_type: str = "unknown"
    bbox: BoundingBox | None = None
    caption: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FigureArtifact:
    figure_id: str
    section_id: str
    page_number: int
    order: int
    bbox: BoundingBox | None = None
    caption: str = ""
    figure_type: str = "figure"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SectionArtifact:
    section_id: str
    title: str
    level: int
    order: int
    section_number: str = ""
    page_start: int | None = None
    page_end: int | None = None
    parent_id: str | None = None
    text_block_ids: list[str] = field(default_factory=list)
    table_ids: list[str] = field(default_factory=list)
    figure_ids: list[str] = field(default_factory=list)
    description: str = ""
    keep: bool = True
    analysis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedDocument:
    source_path: str
    filename: str
    title: str
    sections: list[SectionArtifact] = field(default_factory=list)
    text_blocks: list[TextBlockArtifact] = field(default_factory=list)
    tables: list[TableArtifact] = field(default_factory=list)
    figures: list[FigureArtifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "filename": self.filename,
            "title": self.title,
            "sections": [section.to_dict() for section in self.sections],
            "text_blocks": [block.to_dict() for block in self.text_blocks],
            "tables": [table.to_dict() for table in self.tables],
            "figures": [figure.to_dict() for figure in self.figures],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParsedDocument":
        return cls(
            source_path=data["source_path"],
            filename=data["filename"],
            title=data["title"],
            sections=[SectionArtifact(**section) for section in data.get("sections", [])],
            text_blocks=[TextBlockArtifact(**block) for block in data.get("text_blocks", [])],
            tables=[TableArtifact(**table) for table in data.get("tables", [])],
            figures=[FigureArtifact(**figure) for figure in data.get("figures", [])],
            metadata=data.get("metadata", {}),
        )
