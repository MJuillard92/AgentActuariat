"""
_chunking.py — Groupement par section et découpage en chunks.

Pipeline :
  1. group_blocks_by_section : segmente sur les headings (level ≤ 2)
  2. chunk_blocks            : accumule les paragraphes d'une section en chunks
                               respectant target_chars/max_chars/min_chars.

Sortie chunk_blocks :
  list[dict] avec text, block_indices, section_title_raw
"""
from __future__ import annotations

import logging
from typing import Iterator

from ._extract import Block

log = logging.getLogger(__name__)


def group_blocks_by_section(
    blocks: list[Block],
    section_level_threshold: int = 2,
) -> list[tuple[str | None, list[Block]]]:
    """Partitionne les blocs sur tout heading_level ≤ threshold.

    Retourne une liste de (section_title, blocks_of_that_section).
    Le 1er groupe peut avoir section_title=None si le doc ne commence pas
    par un heading.
    """
    sections: list[tuple[str | None, list[Block]]] = []
    current_title: str | None = None
    current_blocks: list[Block] = []

    for b in blocks:
        hl = b.get("heading_level")
        if hl is not None and hl <= section_level_threshold:
            # Flush la section courante
            if current_blocks:
                sections.append((current_title, current_blocks))
            current_title = b["text"].strip()
            current_blocks = []
        else:
            current_blocks.append(b)

    if current_blocks:
        sections.append((current_title, current_blocks))
    elif not sections and current_title is not None:
        # Heading seul sans contenu → on garde quand même la section (vide)
        sections.append((current_title, []))

    return sections


def _flush(
    current_text_parts: list[str],
    current_indices: list[int],
    section_title: str | None,
    out: list[dict],
) -> None:
    if not current_text_parts:
        return
    text = "\n\n".join(current_text_parts).strip()
    out.append({
        "text": text,
        "block_indices": list(current_indices),
        "section_title_raw": section_title,
    })


def chunk_blocks(
    blocks: list[Block],
    section_title: str | None = None,
    target_chars: int = 2000,
    max_chars: int = 2500,
    min_chars: int = 400,
) -> list[dict]:
    """Découpe une liste de blocs (d'une même section) en chunks.

    Algorithme :
      - Accumulation paragraphe par paragraphe (séparateur \\n\\n)
      - Flush si l'ajout dépasse max_chars ET le courant ≥ min_chars
      - Block seul > max_chars → chunk standalone (pas de split intra-paragraphe)
      - Dernier chunk < min_chars ET un précédent existe dans cette section
        → merge avec le précédent
    """
    out: list[dict] = []
    current_text_parts: list[str] = []
    current_indices: list[int] = []
    current_len = 0
    sep_len = 2  # "\n\n"

    for b in blocks:
        text = b["text"].strip()
        if not text:
            continue
        bidx = b.get("block_index", 0)
        added_len = len(text) + (sep_len if current_text_parts else 0)

        # Block géant standalone
        if len(text) > max_chars and not current_text_parts:
            out.append({
                "text": text,
                "block_indices": [bidx],
                "section_title_raw": section_title,
            })
            continue

        # Dépasserait max_chars : flush si on a au moins min_chars
        if current_len + added_len > max_chars and current_len >= min_chars:
            _flush(current_text_parts, current_indices, section_title, out)
            current_text_parts = [text]
            current_indices = [bidx]
            current_len = len(text)
            continue

        # Sinon accumulate
        current_text_parts.append(text)
        current_indices.append(bidx)
        current_len += added_len

        # Atteint target sans dépasser max → flush opportuniste
        if current_len >= target_chars:
            _flush(current_text_parts, current_indices, section_title, out)
            current_text_parts = []
            current_indices = []
            current_len = 0

    # Reste à flush
    if current_text_parts:
        _flush(current_text_parts, current_indices, section_title, out)

    # Merge du dernier chunk si trop court ET un précédent existe
    if len(out) >= 2 and len(out[-1]["text"]) < min_chars:
        prev = out[-2]
        last = out[-1]
        prev["text"] = prev["text"] + "\n\n" + last["text"]
        prev["block_indices"] = prev["block_indices"] + last["block_indices"]
        out.pop()

    return out


def chunk_document(
    blocks: list[Block],
    target_chars: int = 2000,
    max_chars: int = 2500,
    min_chars: int = 400,
) -> list[dict]:
    """Pipeline complet : groupe par section puis chunk chaque section.

    Retourne la concaténation des chunks de toutes les sections (dans l'ordre).
    """
    sections = group_blocks_by_section(blocks)
    if not sections:
        return []
    all_chunks: list[dict] = []
    for section_title, section_blocks in sections:
        section_chunks = chunk_blocks(
            section_blocks,
            section_title=section_title,
            target_chars=target_chars,
            max_chars=max_chars,
            min_chars=min_chars,
        )
        all_chunks.extend(section_chunks)
    log.info("Chunking : %d sections → %d chunks", len(sections), len(all_chunks))
    return all_chunks
