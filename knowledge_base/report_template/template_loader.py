"""
template_loader.py — API unifiée pour lire le YAML Design 3.

Trois points d'entrée, un seul parseur :

    build_manifest(yaml_path)        → usage Master (data_contract + DAG)
    load_section(sid, yaml_path)     → usage Writer (narrative + visuals)
    resolve_placeholders(text, ds)   → utilitaire partagé ({{ key }} → str)

Ne fait aucune évaluation d'expression Python : pure substitution regex.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "mortality_template.yaml"
DEFAULT_STYLE    = Path(__file__).resolve().parent / "style.yaml"

_DEFAULT_STYLE = {
    "colors": {
        "primary":   "#1A3668",
        "secondary": "#D6E4F7",
        "light":     "#F5F8FF",
        "neutral":   "#888888",
    },
    "table": {
        "header_bg":    "#1A3668",
        "header_fg":    "#FFFFFF",
        "row_alt_bg":   "#F5F8FF",
        "border_color": "#888888",
        "padding_pt":   4,
        "font_size_pt": 9,
    },
}

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_BLOCKS = ("master_from_data", "master_from_modeling", "builder_outputs")


# ───────────────── Dataclasses publiques ─────────────────

@dataclass
class KeySpec:
    key: str
    type: str
    produced_by: dict
    description: str = ""
    unit: str | None = None
    allowed: list | None = None
    confirm_with_user: bool = False
    raw: dict = field(default_factory=dict)


@dataclass
class Aggregation:
    section_id: str
    visual_id: str
    rule: str
    params: dict
    source: str
    target: str
    weight: str | None = None


@dataclass
class Manifest:
    master_from_data: list[KeySpec]
    master_from_modeling: list[KeySpec]
    builder_outputs: list[KeySpec]
    aggregations: list[Aggregation]
    dag: list[dict]
    sections: list[dict] = field(default_factory=list)


@dataclass
class Section:
    id: str
    label: str
    required: bool
    dependencies: list[str]
    narrative: dict
    llm_directives: dict
    visual_specs: list[dict]


# ───────────────── Helpers internes ─────────────────

def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _to_keyspec(entry: dict) -> KeySpec:
    return KeySpec(
        key=entry["key"],
        type=entry.get("type", ""),
        produced_by=entry.get("produced_by") or {},
        description=entry.get("description", ""),
        unit=entry.get("unit"),
        allowed=entry.get("allowed"),
        confirm_with_user=entry.get("confirm_with_user", False),
        raw=entry,
    )


def _collect_session_input_keys(tpl: dict) -> set[str]:
    return {e["key"] for e in (tpl.get("session_inputs") or []) if "key" in e}


def _collect_produced_keys(all_entries: list[KeySpec]) -> set[str]:
    return {e.key for e in all_entries}


def _build_dag(all_entries: list[KeySpec], pre_existing: set[str]) -> list[dict]:
    """Tri topologique des produced_by par dépendances d'inputs."""
    entries_by_key = {e.key: e for e in all_entries}
    produced = set(pre_existing)
    remaining = list(all_entries)
    dag: list[dict] = []

    while remaining:
        progress = False
        for entry in list(remaining):
            pb = entry.produced_by
            inputs_ok = True
            for v in (pb.get("inputs") or {}).values():
                if not isinstance(v, str):
                    # littéral (list, dict, int, ...) → rien à résoudre
                    continue
                if v not in produced and v not in pre_existing:
                    inputs_ok = False
                    break
            if inputs_ok:
                dag.append({
                    "key": entry.key,
                    "tool": pb.get("tool"),
                    "inputs": pb.get("inputs") or {},
                    "output_mapping": pb.get("output_mapping") or {entry.key: entry.key},
                })
                produced.update((pb.get("output_mapping") or {entry.key: entry.key}).values())
                remaining.remove(entry)
                progress = True
        if not progress:
            raise ValueError(f"cycle ou clé non résoluble dans le DAG : {[e.key for e in remaining]}")

    _ = entries_by_key  # lint
    return dag


def _is_active(section: dict, context: dict) -> bool:
    """Retourne True si la section est active dans le contexte donné.

    Une section sans champ `activation` est toujours active.

    Deux formats supportés :
      1. Ancien (mono-clé, scalaire) :
         activation: {key: gender_segmentation, equals: unisex}
         → active ssi context[key] == equals.

      2. Nouveau (multi-clé, listes avec AND implicite) :
         activation: {report_mode: [full_report, raw_rates], gender_segmentation: [unisex]}
         → active ssi, pour chaque clé, context[clé] est dans la liste.
    """
    act = section.get("activation")
    if not act:
        return True

    # Format ancien : {key: ..., equals: ...}
    if "key" in act and "equals" in act:
        return context.get(act["key"]) == act["equals"]

    # Format nouveau : {field_name: [allowed_values], ...} (AND implicite).
    # Si une clé d'activation n'est pas fournie par le contexte, on considère
    # la contrainte non évaluable → on skippe (compatibilité ascendante : un
    # context partiel ne fait pas tomber des sections hors scope).
    for field_name, allowed in act.items():
        if field_name not in context:
            continue
        ctx_val = context.get(field_name)
        if isinstance(allowed, list):
            if ctx_val not in allowed:
                return False
        else:
            if ctx_val != allowed:
                return False
    return True


def _collect_aggregations(tpl: dict) -> list[Aggregation]:
    aggs: list[Aggregation] = []
    for section in tpl.get("sections") or []:
        sid = section.get("id", "")
        for v in section.get("visual_specs") or []:
            agg = v.get("aggregation")
            if not agg:
                continue
            aggs.append(Aggregation(
                section_id=sid,
                visual_id=v.get("id", ""),
                rule=agg.get("rule", ""),
                params=agg.get("params") or {},
                source=agg.get("source", ""),
                target=agg.get("target", ""),
                weight=agg.get("weight"),
            ))
    return aggs


# ───────────────── API publique ─────────────────

def build_manifest(
    yaml_path: Path = DEFAULT_TEMPLATE,
    context: dict | None = None,
) -> Manifest:
    """Usage Master : projection de data_contract en manifest + DAG.

    Si `context` est fourni, filtre les sections dont l'activation n'est pas
    satisfaite. Sans `context`, toutes les sections sont retenues (rétro-compat).
    """
    tpl = _load_yaml(Path(yaml_path))
    dc = tpl.get("data_contract") or {}

    mfd = [_to_keyspec(e) for e in (dc.get("master_from_data") or [])]
    mfm = [_to_keyspec(e) for e in (dc.get("master_from_modeling") or [])]
    bo  = [_to_keyspec(e) for e in (dc.get("builder_outputs") or [])]

    pre_existing = _collect_session_input_keys(tpl)
    dag = _build_dag(mfd + mfm + bo, pre_existing)

    raw_sections = tpl.get("sections") or []
    if context is not None:
        raw_sections = [s for s in raw_sections if _is_active(s, context)]

    return Manifest(
        master_from_data=mfd,
        master_from_modeling=mfm,
        builder_outputs=bo,
        aggregations=_collect_aggregations(tpl),
        dag=dag,
        sections=raw_sections,
    )


def load_section(
    sid: str,
    yaml_path: Path = DEFAULT_TEMPLATE,
    context: dict | None = None,
) -> Section:
    """Usage Writer : livre narrative + directives + visuals d'une section.

    Si la section définit plusieurs variantes de narrative (`text_default` et
    `text_raw_rates`), la variante est choisie en fonction de
    `context["report_mode"]`. Le champ retourné reste `narrative["text"]`.
    """
    ctx = context or {}
    tpl = _load_yaml(Path(yaml_path))
    for section in tpl.get("sections") or []:
        if section.get("id") == sid:
            narrative_raw = section.get("narrative") or {}
            narrative = _select_narrative_variant(narrative_raw, ctx)
            return Section(
                id=section["id"],
                label=section.get("label", ""),
                required=section.get("required", False),
                dependencies=section.get("dependencies") or [],
                narrative=narrative,
                llm_directives=section.get("llm_directives") or {},
                visual_specs=section.get("visual_specs") or [],
            )
    raise KeyError(f"section inconnue : {sid!r}")


def _select_narrative_variant(narrative_raw: dict, context: dict) -> dict:
    """Choisit la variante de narrative selon `context["report_mode"]`.

    Formats supportés dans le YAML :
      narrative: {text: "..."}                     → un seul text
      narrative: {text_default: "...", text_raw_rates: "..."}  → variantes
    """
    out = dict(narrative_raw)  # copie shallow, on ne mute pas la source
    if "text" in narrative_raw:
        return out
    mode = (context or {}).get("report_mode")
    if mode == "raw_rates" and "text_raw_rates" in narrative_raw:
        out["text"] = narrative_raw["text_raw_rates"]
    elif mode == "description" and "text_description" in narrative_raw:
        out["text"] = narrative_raw["text_description"]
    else:
        out["text"] = narrative_raw.get("text_default", "")
    return out


def load_enum_specs(yaml_path: Path = DEFAULT_TEMPLATE) -> dict[str, list]:
    """Extrait {column: [allowed]} depuis session_inputs.input_records.shape.

    Ne considère que les champs de type `enum` déclarant `allowed`.
    """
    tpl = _load_yaml(Path(yaml_path))
    specs: dict[str, list] = {}
    for entry in tpl.get("session_inputs") or []:
        for field_def in entry.get("shape") or []:
            if field_def.get("type") == "enum" and field_def.get("allowed"):
                specs[field_def["key"]] = list(field_def["allowed"])
    return specs


def load_style(style_path: Path = DEFAULT_STYLE) -> dict:
    """Charge style.yaml. Retourne defaults minimaux si fichier absent."""
    p = Path(style_path)
    if not p.exists():
        return {k: dict(v) for k, v in _DEFAULT_STYLE.items()}
    return _load_yaml(p)


_DEFAULT_FORMATS: dict = {
    "defaults":         {},
    "na_display":       "—",
    "number_separator": " ",
}


def load_formats(yaml_path: Path = DEFAULT_TEMPLATE) -> dict:
    """Charge la section `formats:` du YAML template.

    Retourne :
      {"defaults": {col_key: format_str}, "na_display": str, "number_separator": str}

    Si la section est absente, retourne les defaults minimaux.
    """
    try:
        doc = _load_yaml(Path(yaml_path))
    except Exception:
        return {**_DEFAULT_FORMATS}
    fmt = doc.get("formats") or {}
    merged = {**_DEFAULT_FORMATS}
    merged.update({
        "defaults":         dict(fmt.get("defaults") or {}),
        "na_display":       fmt.get("na_display") or _DEFAULT_FORMATS["na_display"],
        "number_separator": fmt.get("number_separator") or _DEFAULT_FORMATS["number_separator"],
    })
    return merged


def _format_placeholder_value(v) -> str:
    """Formate une valeur pour insertion dans une narrative : séparateurs
    de milliers, pas de notation scientifique, pas de décimales superflues
    pour les entiers. Sans ça, 6082714.05 risque d'apparaître comme
    '6.08271e+06' dans le PDF final."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "oui" if v else "non"
    if isinstance(v, int):
        return f"{v:,}".replace(",", " ")
    if isinstance(v, float):
        # NaN / Inf
        if v != v or v in (float("inf"), float("-inf")):
            return "—"
        # Entier déguisé en float
        if v.is_integer() and abs(v) < 1e15:
            return f"{int(v):,}".replace(",", " ")
        # Float « ordinaire » : 2 décimales + séparateur milliers
        return f"{v:,.2f}".replace(",", " ")
    return str(v)


def resolve_placeholders(text: str, data_store: dict) -> str:
    """Substitue {{ key }} par une représentation formatée de data_store[key].
    KeyError si clé absente."""
    def _sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in data_store:
            raise KeyError(f"placeholder non résolu : {key!r}")
        return _format_placeholder_value(data_store[key])
    return _PLACEHOLDER_RE.sub(_sub, text)
