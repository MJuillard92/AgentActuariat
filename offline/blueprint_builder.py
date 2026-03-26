from __future__ import annotations

import json
from pathlib import Path

from offline.models import ParsedDocument, SectionArtifact, TableArtifact, TextBlockArtifact

_SECTION_PURPOSE = {
    "contexte": "poser le périmètre, les sources et l'objectif de l'analyse",
    "qualite_donnees": "contrôler la cohérence, la qualité et les anomalies éventuelles des données",
    "analyse_descriptive": "décrire la structure de la population et les répartitions principales",
    "comparaison": "comparer des populations, des références ou des niveaux observés/attendus",
    "visualisation": "mettre en avant les tendances ou relations clés à l'aide de graphiques",
    "performance_modele": "évaluer la performance et les limites du modèle construit",
    "transformation": "décrire une transformation, un ajustement ou un lissage appliqué aux données",
    "conclusion": "résumer les constats clés et formuler les conclusions ou recommandations",
    "analyse": "présenter un résultat d'analyse et son interprétation",
}


def _classify_section(section: SectionArtifact, text_blocks: list[TextBlockArtifact]) -> str:
    title = section.title.lower()
    if any(token in title for token in ("synthèse", "conclusion", "recommand")):
        return "conclusion"
    if any(token in title for token in ("introduction", "préambule", "preambule", "contrat", "données transmises")):
        return "contexte"
    if any(token in title for token in ("contrôle", "controle", "qualité", "qualite")):
        return "qualite_donnees"
    if any(token in title for token in ("comparaison", "positionnement", "smr", "commentaire")):
        return "comparaison"
    if any(token in title for token in ("modèle", "model", "performance", "construction")):
        return "performance_modele"
    if any(token in title for token in ("visualisation", "graphique", "figure")):
        return "visualisation"
    if any(token in title for token in ("descriptive", "répartition", "repartition", "segmentation", "exposition", "données initiales")):
        return "analyse_descriptive"
    if any(token in title for token in ("lissage",)):
        return "transformation"

    corpus = " ".join(
        [section.title] +
        [text_block.description or text_block.text for text_block in text_blocks]
    ).lower()

    if any(token in corpus for token in ("synthèse", "conclusion", "recommand")):
        return "conclusion"
    if any(token in corpus for token in ("contrôle", "qualité", "cohérence", "anomal")):
        return "qualite_donnees"
    if any(token in corpus for token in ("modèle", "model", "auc", "recall", "precision", "rmse", "mae")):
        return "performance_modele"
    if any(token in corpus for token in ("comparaison", "écart", "versus", "vs", "attendu")):
        return "comparaison"
    if any(token in corpus for token in ("répartition", "distribution", "descript", "moyenne", "segment")):
        return "analyse_descriptive"
    if any(token in corpus for token in ("périmètre", "source", "portefeuille", "données")):
        return "contexte"
    return "analyse"


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _extract_required_inputs(
    title: str,
    text_blocks: list[TextBlockArtifact],
    tables: list[TableArtifact],
) -> list[str]:
    corpus = " ".join(
        [title] +
        [text_block.description or text_block.text for text_block in text_blocks] +
        [" ".join(table.headers) for table in tables]
    ).lower()

    mapping = {
        "age": ("age", "âges", "âge"),
        "sexe": ("sexe", "genre"),
        "date": ("date", "année", "annee", "mois", "période", "periode"),
        "segment": ("segment", "tranche", "classe"),
        "exposition": ("exposition", "expositions"),
        "deces": ("décès", "deces"),
        "taux": ("taux", "ratio", "smr"),
        "prediction": ("prédiction", "prediction", "score", "probabilité", "probabilite"),
        "target": ("cible", "target", "observé", "observe", "attendu"),
    }

    return [
        label
        for label, tokens in mapping.items()
        if any(token in corpus for token in tokens)
    ]


def _suggest_charts(section_kind: str, text_blocks: list[TextBlockArtifact], figure_count: int) -> list[dict]:
    text = " ".join((text_block.description or text_block.text) for text_block in text_blocks).lower()
    chart_specs: list[dict] = []

    if figure_count > 0:
        chart_specs.append({
            "chart_type": "figure_reference",
            "purpose": "préserver l'idée des figures repérées dans le document source",
        })
    if section_kind == "analyse_descriptive":
        chart_specs.append({
            "chart_type": "histogramme_ou_barres",
            "purpose": "montrer une distribution ou une répartition de segment",
        })
    if section_kind == "comparaison":
        chart_specs.append({
            "chart_type": "courbe_ou_barres_comparees",
            "purpose": "comparer les écarts entre plusieurs populations ou scénarios",
        })
    if section_kind == "visualisation":
        chart_specs.append({
            "chart_type": "graphique_exploratoire",
            "purpose": "illustrer la tendance ou la relation déjà décrite dans le texte",
        })
    if section_kind == "performance_modele":
        if any(token in text for token in ("roc", "auc", "classification")):
            chart_specs.append({
                "chart_type": "roc_curve",
                "purpose": "illustrer la performance de classement du modèle",
            })
        else:
            chart_specs.append({
                "chart_type": "graphique_performance",
                "purpose": "illustrer les métriques ou résidus du modèle",
            })
    return chart_specs


def _table_specs(tables: list[TableArtifact]) -> list[dict]:
    return [
        {
            "table_id": table.table_id,
            "page_number": table.page_number,
            "table_type": table.table_type,
            "headers": table.headers,
            "size": {"rows": table.n_rows, "cols": table.n_cols},
            "preview_rows": table.preview_rows,
            "caption": table.caption,
        }
        for table in tables
    ]


def _narrative_guidelines(text_blocks: list[TextBlockArtifact]) -> list[str]:
    descriptions = [text_block.description.strip() for text_block in text_blocks if text_block.description.strip()]
    return _dedupe_preserve_order(descriptions)


def _section_description(section: SectionArtifact, section_kind: str, guidelines: list[str]) -> str:
    if section.description.strip():
        return section.description.strip()
    snippets = guidelines[:2]
    if snippets:
        return " ; ".join(snippets)
    return f"Section de type `{section_kind}` : {_SECTION_PURPOSE.get(section_kind, _SECTION_PURPOSE['analyse'])}."


def _recommended_tables(section_kind: str, tables: list[TableArtifact]) -> list[str]:
    if tables:
        return _dedupe_preserve_order([table.table_type for table in tables if table.table_type != "unknown"])
    defaults = {
        "qualite_donnees": ["tableau_anomalies_ou_controles"],
        "analyse_descriptive": ["statistiques_descriptives_ou_segmentation"],
        "comparaison": ["tableau_comparatif"],
        "performance_modele": ["tableau_metriques_modele"],
        "contexte": ["tableau_perimetre_ou_source"],
    }
    return defaults.get(section_kind, [])


def build_blueprint(parsed_document: ParsedDocument) -> dict:
    text_block_map = {block.block_id: block for block in parsed_document.text_blocks}
    table_map = {table.table_id: table for table in parsed_document.tables}
    figure_map = {figure.figure_id: figure for figure in parsed_document.figures}

    sections_payload = []
    for section in parsed_document.sections:
        if not section.keep:
            continue

        analysis = section.analysis or {}
        text_blocks = [
            text_block_map[block_id]
            for block_id in section.text_block_ids
            if block_id in text_block_map and text_block_map[block_id].keep
        ]
        tables = [table_map[table_id] for table_id in section.table_ids if table_id in table_map]
        figures = [figure_map[figure_id] for figure_id in section.figure_ids if figure_id in figure_map]

        section_kind = analysis.get("section_kind") or _classify_section(section, text_blocks)
        narrative_guidelines = analysis.get("narrative_guidance") or _narrative_guidelines(text_blocks)
        required_inputs = analysis.get("required_inputs") or _extract_required_inputs(section.title, text_blocks, tables)
        recommended_tables = analysis.get("expected_tables") or _recommended_tables(section_kind, tables)
        section_description = analysis.get("analysis_logic") or _section_description(section, section_kind, narrative_guidelines)
        purpose = analysis.get("purpose") or _SECTION_PURPOSE.get(section_kind, _SECTION_PURPOSE["analyse"])
        chart_specs = analysis.get("expected_figures")
        if chart_specs:
            chart_specs = [
                {"chart_type": item, "purpose": "figure attendue d'après la description de section"}
                if isinstance(item, str) else item
                for item in chart_specs
            ]
        else:
            chart_specs = _suggest_charts(section_kind, text_blocks, len(figures))

        sections_payload.append({
            "section_id": section.section_id,
            "section_number": section.section_number,
            "title": section.title,
            "level": section.level,
            "page_start": section.page_start,
            "page_end": section.page_end,
            "parent_id": section.parent_id,
            "section_kind": section_kind,
            "purpose": purpose,
            "section_description": section_description,
            "text_block_count": len(section.text_block_ids),
            "kept_text_block_count": len(text_blocks),
            "table_count": len(tables),
            "figure_count": len(figures),
            "narrative_guidelines": narrative_guidelines,
            "required_inputs": required_inputs,
            "recommended_tables": recommended_tables,
            "table_roles": analysis.get("table_roles", []),
            "figure_roles": analysis.get("figure_roles", []),
            "expected_outputs": analysis.get("expected_outputs", []),
            "agent_guidance": analysis.get("agent_guidance", ""),
            "source_text_blocks": [text_block.text for text_block in text_blocks],
            "table_specs": _table_specs(tables),
            "chart_specs": chart_specs,
        })

    kept_text_blocks = [text_block for text_block in parsed_document.text_blocks if text_block.keep]
    kept_sections = [section for section in parsed_document.sections if section.keep]
    described_sections = [section for section in parsed_document.sections if section.keep and section.analysis]
    global_guidance = parsed_document.metadata.get("global_guidance", {})
    return {
        "blueprint_name": f"{Path(parsed_document.filename).stem}_blueprint",
        "source_document": {
            "path": parsed_document.source_path,
            "filename": parsed_document.filename,
            "title": parsed_document.title,
        },
        "metadata": {
            **parsed_document.metadata,
            "kept_section_count": len(kept_sections),
            "described_section_count": len(described_sections),
            "kept_text_block_count": len(kept_text_blocks),
            "described_text_block_count": sum(1 for text_block in parsed_document.text_blocks if text_block.description.strip()),
        },
        "global_rules": {
            "tone": "formel",
            "evidence_based": True,
            "section_descriptions_reviewed": True,
            "section_order_should_be_preserved": True,
            "report_type": global_guidance.get("report_type", ""),
            "objective": global_guidance.get("objective", ""),
            "audience": global_guidance.get("audience", ""),
            "global_narrative_rules": global_guidance.get("global_narrative_rules", []),
        },
        "agent_guidance_prompt": global_guidance.get("agent_prompt", ""),
        "sections": sections_payload,
    }


def save_blueprint(blueprint: dict, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blueprint, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
