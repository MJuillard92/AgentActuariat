"""
TOOL CONTRACT — conversation.describe_capabilities
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : conversation.describe_capabilities
domain        : conversation
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-05-13

DESCRIPTION
-----------
Liste structurée de tout ce que le système actuariel sait faire +
des inputs qu'il faut fournir + des outputs producibles. Agrège trois
sources existantes : le catalogue des tools (tool_registry), le manifest
YAML (template_loader), et les règles d'activation des sections.

WHEN TO USE
-----------
Phase conversationnelle, quand l'utilisateur demande :
  - "que sais-tu faire ?"
  - "qu'as-tu besoin de moi ?"
  - "comment ça marche ?"
  - "quels rapports peux-tu produire ?"

WHEN NOT TO USE
---------------
Pas pour exécuter un calcul. Ce tool est purement descriptif (read-only
sur le code, aucune mutation de data_store).

PREREQUISITES
-------------
required_data_store_keys: []
Note: tool stateless, ne lit pas le df ni le data_store.

INPUTS
------
params:
  function_name:
    type    : string
    values  : all | capabilities | required_inputs | outputs_produced
    default : all
    note    : Filtre la sortie pour ne retourner qu'un sous-bloc.

OUTPUTS
-------
return_payload:
  capabilities       : dict — exploration / calculs_actuariels / rapports_pdf
  required_inputs    : dict — fichier / from_user / auto_detected
  outputs_produced   : dict — tables / charts par mode de rapport

AGENT GUIDANCE
--------------
reasoning_hint: >
  Appeler avec function_name="all" pour une présentation complète.
  Le LLM doit reformuler en langage naturel — pas balancer le JSON brut.

CATALOGUE METADATA
------------------
display_name      : Description des capacités du système
short_description : Liste outils, inputs requis, rapports producibles.
domain            : conversation
capability_group  : data_exploration
client_visible    : true
"""
from __future__ import annotations

from typing import Any


# Groupes attendus dans le catalogue (depuis docstrings tools)
_GROUP_TO_LABEL = {
    "data_exploration":   "exploration",
    "table_construction": "calculs_actuariels",
    "actuarial_modeling": "calculs_actuariels",
    "reporting":          "rapports_pdf",
    "preprocessing":      "calculs_actuariels",
    "validation":         "calculs_actuariels",
    "benchmarking":       "calculs_actuariels",
}


def _extract_capabilities() -> dict:
    """Walk le catalogue (source riche, pas la version nested simplifiée),
    filtre client_visible=true, groupe par capability_group."""
    try:
        from tools.catalogue import get_catalogue
    except Exception:
        return {}
    cat = get_catalogue() or {}
    tools_flat = cat.get("tools") or {}

    groups: dict[str, list] = {"exploration": [], "calculs_actuariels": [], "autres": []}
    for qualified_name, info in tools_flat.items():
        if info.get("client_visible") is False:
            continue
        group_raw = info.get("capability_group") or "autres"
        group = _GROUP_TO_LABEL.get(group_raw, "autres")
        groups.setdefault(group, []).append({
            "tool":        qualified_name,
            "display":     info.get("display_name") or qualified_name,
            "description": info.get("short_description")
                           or (info.get("description") or "")[:200],
        })
    # Trier alphabétiquement par display
    for g in groups:
        groups[g].sort(key=lambda x: x["display"].lower())
    return groups


def _extract_report_modes() -> list[dict]:
    """Liste les modes de rapport producibles depuis les sections du YAML
    et leurs règles d'activation."""
    try:
        from knowledge_base.report_template.template_loader import build_manifest
    except Exception:
        return []
    manifest = build_manifest()
    modes_seen: set[str] = set()
    genders_seen: set[str] = set()
    for section in manifest.sections:
        act = section.get("activation") or {}
        rm_vals = act.get("report_mode") or []
        if isinstance(rm_vals, str):
            rm_vals = [rm_vals]
        for v in rm_vals:
            modes_seen.add(v)
        gs_vals = act.get("gender_segmentation") or []
        if isinstance(gs_vals, str):
            gs_vals = [gs_vals]
        for v in gs_vals:
            genders_seen.add(v)

    _mode_labels = {
        "description":  "Analyse descriptive du portefeuille (sans table de mortalité)",
        "raw_rates":    "Taux bruts par âge (sans lissage)",
        "full_report":  "Rapport complet : taux bruts + lissage + validation",
    }
    out = [
        {"mode": m, "label": _mode_labels.get(m, m)}
        for m in ("description", "raw_rates", "full_report") if m in modes_seen
    ]
    if "by_sex" in genders_seen:
        out.append({"axe": "gender", "label":
                    "Variante by_sex : tables H/F séparées (sinon table unisex agrégée)"})
    return out


def _extract_required_inputs() -> dict:
    """Walk data_contract pour distinguer inputs user (confirm_with_user=True)
    vs auto-détectés depuis les données."""
    try:
        from knowledge_base.report_template.template_loader import build_manifest
    except Exception:
        return {}
    manifest = build_manifest()

    from_user: list[dict] = []
    auto: list[dict] = []
    for spec in manifest.master_from_data + manifest.master_from_modeling:
        entry = {
            "field":       spec.key,
            "description": spec.description,
        }
        if spec.allowed:
            entry["options"] = list(spec.allowed)
        if spec.confirm_with_user:
            from_user.append(entry)
        else:
            auto.append(entry)

    return {
        "fichier": (
            "CSV ou Parquet contenant au minimum : date_naissance, date_entree, "
            "date_sortie, cause_sortie (décès / vivant), sexe. Le mapping des "
            "colonnes (renommage des noms CSV vers noms canoniques) est proposé "
            "automatiquement et confirmé par l'utilisateur via l'UI ou en chat."
        ),
        "from_user":     from_user,
        "auto_detected": auto,
    }


def _extract_outputs_produced() -> dict:
    """Liste les tableaux et graphiques producibles toutes sections confondues."""
    try:
        from knowledge_base.report_template.template_loader import build_manifest
    except Exception:
        return {}
    manifest = build_manifest()

    tables: list[dict] = []
    charts: list[dict] = []
    for section in manifest.sections:
        sid = section.get("id", "?")
        act = section.get("activation") or {}
        modes = act.get("report_mode") or ["all"]
        if isinstance(modes, str):
            modes = [modes]
        for v in (section.get("visual_specs") or []):
            entry = {
                "id":             v.get("id", "?"),
                "section":        sid,
                "purpose":        v.get("purpose", ""),
                "active_in":      modes,
            }
            vtype = v.get("type", "")
            if vtype == "table":
                tables.append(entry)
            elif vtype == "chart":
                entry["chart_type"] = v.get("chart_type", "")
                charts.append(entry)
    return {"tables": tables, "charts": charts}


def run(df=None, params: dict | None = None) -> dict[str, Any]:
    """Point d'entrée tool. `df` est ignoré (stateless)."""
    fn = (params or {}).get("function_name", "all")

    if fn == "capabilities":
        return {"capabilities": {
            **_extract_capabilities(),
            "rapports_pdf": _extract_report_modes(),
        }}

    if fn == "required_inputs":
        return {"required_inputs": _extract_required_inputs()}

    if fn == "outputs_produced":
        return {"outputs_produced": _extract_outputs_produced()}

    # all
    return {
        "capabilities": {
            **_extract_capabilities(),
            "rapports_pdf": _extract_report_modes(),
        },
        "required_inputs":  _extract_required_inputs(),
        "outputs_produced": _extract_outputs_produced(),
    }
