"""
_enrich.py — Enrichissement des métadonnées pour un chunk.

Produit le dict `metadata` strictement compatible avec le schéma de
`chunks_enriched.json` (regex_matches, tags, has_formula, formula_count,
regulatory, tables_referenced, doc_id, section_id) + 3 champs additionnels
pour l'idempotence et la traçabilité :
  - source_fingerprint  (SHA256 du fichier source, identique pour tous les chunks)
  - source_filename     (nom du fichier d'origine)
  - ingested_at         (timestamp UTC ISO 8601)
"""
from __future__ import annotations

from . import _patterns


def enrich_metadata(
    text: str,
    *,
    doc_id: str,
    section_id: str,
    source_fingerprint: str,
    source_filename: str,
    ingested_at: str,
) -> dict:
    """Construit le bloc metadata complet pour un chunk donné."""
    regex_matches = _patterns.find_regex_matches(text)
    tags = _patterns.extract_tags(regex_matches)
    regulatory = _patterns.is_regulatory(regex_matches)
    tables_referenced = _patterns.extract_tables(text)
    formula_count = _patterns.count_formulas(text)
    has_formula = formula_count > 0

    return {
        "regex_matches":      regex_matches,
        "tags":               tags,
        "has_formula":        has_formula,
        "formula_count":      formula_count,
        "regulatory":         regulatory,
        "tables_referenced":  tables_referenced,
        "doc_id":             doc_id,
        "section_id":         section_id,
        "source_fingerprint": source_fingerprint,
        "source_filename":    source_filename,
        "ingested_at":        ingested_at,
    }
