"""
TOOL CONTRACT — build_pdf.write_section
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : build_pdf.write_section
domain        : mortality_experience
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-12

DESCRIPTION
-----------
Enregistre le contenu d'une section du rapport (texte narratif + dernier
tableau rendu + dernier graphique généré) dans data_store["section_outputs"].
À appeler par le WriterAgent après avoir :
  1. Rédigé le texte narratif (dans le message LLM)
  2. Appelé build_pdf.table_renderer (si la section a des tableaux)
  3. Appelé graphs.graph_from_spec (si la section a des graphiques)

Lit depuis data_store (et les consomme) :
  - _last_table_rows : list[list[str]] stocké par table_renderer
  - _last_graph_path : str stocké par graph_from_spec

Supporte les appels multiples pour la même section (mode append).

WHEN TO USE
-----------
Appeler après chaque section rédigée, avant de passer à la section suivante.
Dernière étape avant build_pdf.assemble_sections.

INPUTS
------
params:
  section_id:
    type    : string
    note    : Identifiant de section YAML (preamble, data_submission, construction,
              analysis, conclusion, annex)
  text:
    type    : string
    note    : Texte narratif généré par GPT-4o pour cette section.
  table_caption:
    type    : string
    default : ""
    note    : Légende du tableau (si _last_table_rows présent).
  graph_caption:
    type    : string
    default : ""
    note    : Légende du graphique (si _last_graph_path présent).
  status:
    type    : string
    default : done
    note    : Statut de la section (done | skipped | partial).
  clear:
    type    : bool
    default : false
    note    : True pour réinitialiser la section (repartir de zéro).

OUTPUTS
-------
data_store_keys_written:
  - section_outputs  # dict {section_id: {text, tables, table_captions, graphs, graph_captions, status}}
return_payload:
  success    : bool
  section_id : str
  text_len   : int
  n_tables   : int
  n_graphs   : int
  status     : str

CATALOGUE METADATA
------------------
display_name      : Enregistrement section rapport
short_description : Accumule texte narratif + tableau + graphique dans section_outputs.
domain            : mortality_experience
capability_group  : reporting
depends_on        : [build_pdf.load_yaml_template, build_pdf.table_renderer, graphs.graph_from_spec]
required_by       : [build_pdf.assemble_sections]
client_visible    : false
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_VALID_SECTION_IDS = {
    "preamble", "data_submission", "construction",
    "analysis", "conclusion", "annex",
}


def run(data: dict | None = None, params: dict | None = None) -> dict:
    """
    Enregistre le contenu d'une section dans data_store["section_outputs"][section_id].
    Consomme _last_table_rows et _last_graph_path s'ils sont présents.
    """
    data   = data   or {}
    params = params or {}

    section_id    = params.get("section_id", "")
    text          = params.get("text", "")
    table_caption = params.get("table_caption", "")
    graph_caption = params.get("graph_caption", "")
    status        = params.get("status", "done")
    clear         = bool(params.get("clear", False))

    if not section_id:
        return {"erreur": "section_id requis (preamble, data_submission, construction, analysis, conclusion, annex)"}

    # Accepter aussi les section_id numeriques (ex. "4.1") pour les sous-sections
    section_outputs = data.setdefault("section_outputs", {})

    # Initialiser ou réinitialiser la section
    if clear or section_id not in section_outputs:
        section_outputs[section_id] = {
            "text":           "",
            "tables":         [],
            "table_captions": [],
            "graphs":         [],
            "graph_captions": [],
            "status":         status,
        }

    sec = section_outputs[section_id]

    # ── Texte narratif ────────────────────────────────────────────────────────
    if text:
        if sec["text"]:
            sec["text"] = sec["text"] + "\n\n" + text
        else:
            sec["text"] = text

    # ── Tableau : consommer _last_table_rows ──────────────────────────────────
    last_rows = data.pop("_last_table_rows", None)
    if last_rows:
        sec["tables"].append(last_rows)
        cap = table_caption or f"Tableau {len(sec['tables'])}"
        sec["table_captions"].append(cap)
        log.info("[write_section] %s — tableau ajouté (%d lignes)", section_id, len(last_rows))

    # ── Graphique : consommer _last_graph_path ────────────────────────────────
    last_graph = data.pop("_last_graph_path", None)
    if last_graph:
        sec["graphs"].append(last_graph)
        cap = graph_caption or f"Graphique {len(sec['graphs'])}"
        sec["graph_captions"].append(cap)
        log.info("[write_section] %s — graphique ajouté : %s", section_id, last_graph)

    sec["status"] = status

    log.info(
        "[write_section] %s — texte %d chars, %d tableaux, %d graphiques, status=%s",
        section_id, len(sec["text"]), len(sec["tables"]), len(sec["graphs"]), status,
    )

    return {
        "success":    True,
        "section_id": section_id,
        "text_len":   len(sec["text"]),
        "n_tables":   len(sec["tables"]),
        "n_graphs":   len(sec["graphs"]),
        "status":     status,
    }
