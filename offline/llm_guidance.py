from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from offline.models import ParsedDocument, SectionArtifact

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"

try:
    import config

    DEFAULT_MODEL = getattr(config, "FORMATTER_MODEL", DEFAULT_MODEL)
except Exception:
    pass


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY manquante dans .env")
    return OpenAI(api_key=api_key)


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Réponse JSON introuvable dans la sortie du modèle.")
    return json.loads(raw_text[start:end + 1])


def _section_payload(section: SectionArtifact, parsed_document: ParsedDocument) -> dict[str, Any]:
    block_map = {block.block_id: block for block in parsed_document.text_blocks}
    table_map = {table.table_id: table for table in parsed_document.tables}
    figure_map = {figure.figure_id: figure for figure in parsed_document.figures}

    text_blocks = [
        {
            "block_id": block.block_id,
            "page_number": block.page_number,
            "bbox": block.bbox,
            "text": block.text,
        }
        for block_id in section.text_block_ids
        if block_id in block_map
        for block in [block_map[block_id]]
    ]
    tables = [
        {
            "table_id": table.table_id,
            "page_number": table.page_number,
            "bbox": table.bbox,
            "headers": table.headers,
            "preview_rows": table.preview_rows,
            "table_type": table.table_type,
            "caption": table.caption,
        }
        for table_id in section.table_ids
        if table_id in table_map
        for table in [table_map[table_id]]
    ]
    figures = [
        {
            "figure_id": figure.figure_id,
            "page_number": figure.page_number,
            "bbox": figure.bbox,
            "caption": figure.caption,
            "figure_type": figure.figure_type,
        }
        for figure_id in section.figure_ids
        if figure_id in figure_map
        for figure in [figure_map[figure_id]]
    ]
    return {
        "section_id": section.section_id,
        "section_number": section.section_number,
        "title": section.title,
        "level": section.level,
        "page_start": section.page_start,
        "page_end": section.page_end,
        "text_blocks": text_blocks,
        "tables": tables,
        "figures": figures,
    }


def describe_section(
    parsed_document: ParsedDocument,
    section: SectionArtifact,
    model: str | None = None,
) -> dict[str, Any]:
    payload = _section_payload(section, parsed_document)
    client = _get_client()
    model_name = model or DEFAULT_MODEL

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu analyses la structure d'un rapport technique ou statistique pour construire un blueprint. "
                    "Tu reçois une section complète déjà extraite d'un PDF : titre, pagination, blocs texte, tableaux, figures et positions. "
                    "Tu dois expliquer ce qui est fait dans cette section afin qu'un agent puisse reproduire une analyse similaire. "
                    "Réponds uniquement en JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Analyse cette section et retourne un objet JSON avec exactement ces clés : "
                    "`section_id`, `purpose`, `analysis_logic`, `table_roles`, `figure_roles`, "
                    "`narrative_guidance`, `expected_outputs`, `expected_tables`, `expected_figures`, `agent_guidance`.\n\n"
                    f"Section extraite :\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
        max_tokens=2200,
    )

    content = response.choices[0].message.content or "{}"
    result = _extract_json_object(content)
    result.setdefault("section_id", section.section_id)
    for key in (
        "purpose",
        "analysis_logic",
        "agent_guidance",
    ):
        result[key] = (result.get(key) or "").strip()
    for key in (
        "table_roles",
        "figure_roles",
        "narrative_guidance",
        "expected_outputs",
        "expected_tables",
        "expected_figures",
    ):
        value = result.get(key) or []
        if isinstance(value, str):
            value = [line.strip() for line in value.splitlines() if line.strip()]
        result[key] = value
    return result


def build_agent_guidance(
    parsed_document: ParsedDocument,
    model: str | None = None,
) -> dict[str, Any]:
    included_sections = [
        section
        for section in parsed_document.sections
        if section.keep and section.analysis
    ]
    sections_payload = [
        {
            "section_number": section.section_number,
            "title": section.title,
            "page_start": section.page_start,
            "page_end": section.page_end,
            "analysis": section.analysis,
        }
        for section in included_sections
    ]
    if not sections_payload:
        raise ValueError("Aucune section décrite n'est disponible pour construire le prompt agent.")

    client = _get_client()
    model_name = model or DEFAULT_MODEL
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu construis un guide de rédaction pour un agent chargé de produire une analyse similaire à un rapport de référence. "
                    "Tu dois synthétiser les descriptions de sections en un blueprint global et un prompt opérationnel pour l'agent. "
                    "Réponds uniquement en JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    "À partir des sections décrites ci-dessous, retourne un objet JSON avec exactement les clés : "
                    "`report_type`, `objective`, `audience`, `ordered_sections`, `global_narrative_rules`, `agent_prompt`.\n\n"
                    f"Sections décrites :\n{json.dumps(sections_payload, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
        max_tokens=2600,
    )
    content = response.choices[0].message.content or "{}"
    result = _extract_json_object(content)
    for key in ("report_type", "objective", "audience", "agent_prompt"):
        result[key] = (result.get(key) or "").strip()
    for key in ("ordered_sections", "global_narrative_rules"):
        value = result.get(key) or []
        if isinstance(value, str):
            value = [line.strip() for line in value.splitlines() if line.strip()]
        result[key] = value
    return result
