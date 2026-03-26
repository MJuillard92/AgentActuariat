from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from offline.models import (
    FigureArtifact,
    ParsedDocument,
    SectionArtifact,
    TableArtifact,
    TextBlockArtifact,
)

_SECTION_LINE_PATTERN = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)*)\.?\s+(?P<title>.+?)\s+(?P<page>\d+)\s*$"
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass
class _Anchor:
    section_id: str
    page_number: int
    y0: float
    level: int


def _normalize_text(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", (text or "").replace("\xa0", " ")).strip()


def _ensure_pdf_dependencies():
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise ImportError(
            "PyMuPDF est requis pour l'analyse PDF. Installez `PyMuPDF`."
        ) from exc
    return fitz


def _parse_toc_entries(pages_text: list[list[str]]) -> list[dict]:
    entries: list[dict] = []
    for page_lines in pages_text[: min(8, len(pages_text))]:
        for line in page_lines:
            match = _SECTION_LINE_PATTERN.match(line)
            if not match:
                continue
            number = match.group("number")
            title = match.group("title").strip(" .\t")
            page_number = int(match.group("page"))
            entries.append({
                "section_number": number,
                "title": title,
                "page_start": page_number,
                "level": number.count(".") + 1,
            })
    deduped: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    for entry in entries:
        key = (entry["section_number"], entry["title"].lower(), entry["page_start"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _build_sections_from_toc(toc_entries: list[dict], page_count: int) -> list[SectionArtifact]:
    if not toc_entries:
        return [
            SectionArtifact(
                section_id="sec_000",
                title="Préambule",
                section_number="",
                level=0,
                order=0,
                page_start=1,
                page_end=page_count,
            )
        ]

    sections: list[SectionArtifact] = []
    first_page = toc_entries[0]["page_start"]
    if first_page > 1:
        sections.append(
            SectionArtifact(
                section_id="sec_000",
                title="Préambule",
                section_number="",
                level=0,
                order=0,
                page_start=1,
                page_end=first_page - 1,
            )
        )

    stack: dict[int, str] = {}
    for index, entry in enumerate(toc_entries):
        level = entry["level"]
        parent_id = stack.get(level - 1)
        section = SectionArtifact(
            section_id=f"sec_{index + 1:03d}",
            title=entry["title"],
            section_number=entry["section_number"],
            level=level,
            order=index + 1,
            page_start=entry["page_start"],
            page_end=page_count,
            parent_id=parent_id,
        )
        sections.append(section)
        stack[level] = section.section_id
        for stale_level in list(stack):
            if stale_level > level:
                del stack[stale_level]

    for index, section in enumerate(sections):
        if section.level == 0 and section.section_number == "":
            continue
        next_page = None
        for next_section in sections[index + 1:]:
            if next_section.level <= section.level:
                next_page = next_section.page_start
                break
        section.page_end = page_count if next_page is None else max(section.page_start or 1, next_page - 1)

    return sections


def _collect_page_text_lines(fitz_doc) -> list[list[str]]:
    pages_lines: list[list[str]] = []
    for page in fitz_doc:
        text = page.get_text("text")
        lines = [_normalize_text(line) for line in text.splitlines()]
        pages_lines.append([line for line in lines if line])
    return pages_lines


def _heading_variants(section: SectionArtifact) -> list[str]:
    variants = []
    title = _normalize_text(section.title)
    if section.section_number:
        variants.append(_normalize_text(f"{section.section_number}. {section.title}"))
        variants.append(_normalize_text(f"{section.section_number} {section.title}"))
    variants.append(title)
    return [variant.lower() for variant in variants if variant]


def _find_section_anchors(fitz_doc, sections: list[SectionArtifact]) -> dict[str, _Anchor]:
    anchors: dict[str, _Anchor] = {}
    blocks_by_page: dict[int, list[tuple]] = {}

    for page_index in range(fitz_doc.page_count):
        page = fitz_doc.load_page(page_index)
        blocks = page.get_text("blocks")
        blocks_by_page[page_index + 1] = blocks

    for section in sections:
        candidate_pages = range(section.page_start or 1, min((section.page_end or fitz_doc.page_count), fitz_doc.page_count) + 1)
        heading_variants = _heading_variants(section)
        found = None
        for page_number in candidate_pages:
            page = fitz_doc.load_page(page_number - 1)
            for variant in heading_variants:
                try:
                    rects = page.search_for(variant)
                except Exception:
                    rects = []
                if rects:
                    rect = rects[0]
                    found = _Anchor(section.section_id, page_number, float(rect.y0), section.level)
                    break
            if found:
                break
            for block in blocks_by_page.get(page_number, []):
                x0, y0, x1, y1, text, *_rest = block
                normalized = _normalize_text(text).lower()
                if not normalized:
                    continue
                if any(variant in normalized for variant in heading_variants):
                    found = _Anchor(section.section_id, page_number, float(y0), section.level)
                    break
            if found:
                break
        if found is None:
            found = _Anchor(section.section_id, section.page_start or 1, 0.0, section.level)
        anchors[section.section_id] = found
    return anchors


def _section_for_position(
    page_number: int,
    y0: float,
    sections: list[SectionArtifact],
    anchors: dict[str, _Anchor],
) -> SectionArtifact:
    eligible: list[tuple[int, float, int, SectionArtifact]] = []
    for section in sections:
        if section.page_start is None or section.page_end is None:
            continue
        if not (section.page_start <= page_number <= section.page_end):
            continue
        anchor = anchors.get(section.section_id)
        if anchor is None:
            continue
        if page_number == anchor.page_number and y0 < anchor.y0:
            continue
        distance = page_number - anchor.page_number
        eligible.append((distance, -anchor.y0 if page_number == anchor.page_number else 0, section.level, section))

    if not eligible:
        return sections[0]

    eligible.sort(key=lambda item: (item[0], item[1], -item[2]))
    return eligible[0][3]


def _looks_like_noise(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    if len(normalized) <= 2:
        return True
    if re.fullmatch(r"\d+", normalized):
        return True
    return False


def _infer_table_type(headers: list[str], preview_rows: list[list[str]]) -> str:
    header_text = " ".join(headers).lower()
    first_row = " ".join(preview_rows[0]).lower() if preview_rows else ""

    if any(token in header_text for token in ("moyenne", "median", "écart", "std", "min", "max")):
        return "statistiques_descriptives"
    if any(token in header_text for token in ("date", "année", "annee", "mois", "period")):
        return "serie_temporelle"
    if any(token in header_text for token in ("auc", "precision", "recall", "f1", "rmse", "mae")):
        return "metriques_modele"
    if any(token in header_text for token in ("age", "âge", "sexe", "segment", "classe")):
        return "segmentation"
    if any(token in first_row for token in ("total", "ensemble", "global")):
        return "synthese"
    return "unknown"


def parse_pdf(pdf_path: str) -> ParsedDocument:
    fitz = _ensure_pdf_dependencies()
    path = Path(pdf_path)
    fitz_doc = fitz.open(path)
    page_count = fitz_doc.page_count

    pages_text = _collect_page_text_lines(fitz_doc)
    outline_entries = fitz_doc.get_toc() or []
    toc_entries = [
        {
            "section_number": "",
            "title": title.strip(),
            "page_start": int(page),
            "level": int(level),
        }
        for level, title, page, *rest in outline_entries
        if str(title).strip() and int(page) > 0
    ]
    if toc_entries:
        for entry in toc_entries:
            match = re.match(r"^\s*(?P<number>\d+(?:\.\d+)*)\.?\s+(?P<title>.+?)\s*$", entry["title"])
            if match:
                entry["section_number"] = match.group("number")
                entry["title"] = match.group("title")
            else:
                entry["section_number"] = ""
    else:
        toc_entries = _parse_toc_entries(pages_text)
    sections = _build_sections_from_toc(toc_entries, page_count)
    anchors = _find_section_anchors(fitz_doc, sections)

    text_blocks: list[TextBlockArtifact] = []
    figures: list[FigureArtifact] = []
    tables: list[TableArtifact] = []

    for page_index in range(page_count):
        page_number = page_index + 1
        page = fitz_doc.load_page(page_index)
        for block in page.get_text("dict").get("blocks", []):
            block_type = block.get("type", 0)
            bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
            y0 = float(bbox[1]) if bbox else 0.0
            section = _section_for_position(page_number, y0, sections, anchors)

            if block_type == 0:
                lines = block.get("lines", [])
                spans = []
                for line in lines:
                    for span in line.get("spans", []):
                        text = _normalize_text(span.get("text", ""))
                        if text:
                            spans.append(text)
                text = _normalize_text(" ".join(spans))
                if _looks_like_noise(text):
                    continue
                text_block = TextBlockArtifact(
                    block_id=f"blk_{len(text_blocks):04d}",
                    section_id=section.section_id,
                    page_number=page_number,
                    order=len(text_blocks),
                    text=text,
                    bbox=bbox,
                )
                text_blocks.append(text_block)
                section.text_block_ids.append(text_block.block_id)

            elif block_type == 1:
                figure = FigureArtifact(
                    figure_id=f"fig_{len(figures):03d}",
                    section_id=section.section_id,
                    page_number=page_number,
                    order=len(figures),
                    bbox=bbox,
                )
                figures.append(figure)
                section.figure_ids.append(figure.figure_id)

    for page_index in range(page_count):
        page_number = page_index + 1
        page = fitz_doc.load_page(page_index)
        try:
            found_tables = page.find_tables()
            table_objects = getattr(found_tables, "tables", found_tables)
        except Exception:
            table_objects = []

        for found_table in table_objects:
            rows = found_table.extract() or []
            rows = [[_normalize_text(cell or "") for cell in row] for row in rows if row and any(cell for cell in row)]
            if not rows:
                continue
            bbox = tuple(found_table.bbox) if getattr(found_table, "bbox", None) else None
            section = _section_for_position(page_number, float(bbox[1]) if bbox else 0.0, sections, anchors)
            headers = rows[0]
            preview_rows = rows[: min(4, len(rows))]
            table = TableArtifact(
                table_id=f"tbl_{len(tables):03d}",
                section_id=section.section_id,
                page_number=page_number,
                order=len(tables),
                n_rows=len(rows),
                n_cols=max(len(row) for row in rows),
                headers=headers,
                preview_rows=preview_rows,
                table_type=_infer_table_type(headers, preview_rows),
                bbox=bbox,
            )
            tables.append(table)
            section.table_ids.append(table.table_id)

    title = path.stem
    for page_lines in pages_text[:2]:
        for line in page_lines:
            if len(line) > 12 and not _SECTION_LINE_PATTERN.match(line):
                title = line
                break
        if title != path.stem:
            break

    metadata = {
        "page_count": page_count,
        "section_count": len(sections),
        "toc_entry_count": len(toc_entries),
        "text_block_count": len(text_blocks),
        "table_count": len(tables),
        "figure_count": len(figures),
        "source_format": "pdf",
    }

    return ParsedDocument(
        source_path=str(path),
        filename=path.name,
        title=title,
        sections=sections,
        text_blocks=text_blocks,
        tables=tables,
        figures=figures,
        metadata=metadata,
    )
