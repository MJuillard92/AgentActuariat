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

def build_manifest(yaml_path: Path = DEFAULT_TEMPLATE) -> Manifest:
    """Usage Master : projection de data_contract en manifest + DAG."""
    tpl = _load_yaml(Path(yaml_path))
    dc = tpl.get("data_contract") or {}

    mfd = [_to_keyspec(e) for e in (dc.get("master_from_data") or [])]
    mfm = [_to_keyspec(e) for e in (dc.get("master_from_modeling") or [])]
    bo  = [_to_keyspec(e) for e in (dc.get("builder_outputs") or [])]

    pre_existing = _collect_session_input_keys(tpl)
    dag = _build_dag(mfd + mfm + bo, pre_existing)

    return Manifest(
        master_from_data=mfd,
        master_from_modeling=mfm,
        builder_outputs=bo,
        aggregations=_collect_aggregations(tpl),
        dag=dag,
    )


def load_section(sid: str, yaml_path: Path = DEFAULT_TEMPLATE) -> Section:
    """Usage Writer : livre narrative + directives + visuals d'une section."""
    tpl = _load_yaml(Path(yaml_path))
    for section in tpl.get("sections") or []:
        if section.get("id") == sid:
            return Section(
                id=section["id"],
                label=section.get("label", ""),
                required=section.get("required", False),
                dependencies=section.get("dependencies") or [],
                narrative=section.get("narrative") or {},
                llm_directives=section.get("llm_directives") or {},
                visual_specs=section.get("visual_specs") or [],
            )
    raise KeyError(f"section inconnue : {sid!r}")


def load_style(style_path: Path = DEFAULT_STYLE) -> dict:
    """Charge style.yaml. Retourne defaults minimaux si fichier absent."""
    p = Path(style_path)
    if not p.exists():
        return {k: dict(v) for k, v in _DEFAULT_STYLE.items()}
    return _load_yaml(p)


def resolve_placeholders(text: str, data_store: dict) -> str:
    """Substitue {{ key }} par str(data_store[key]). KeyError si clé absente."""
    def _sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in data_store:
            raise KeyError(f"placeholder non résolu : {key!r}")
        return str(data_store[key])
    return _PLACEHOLDER_RE.sub(_sub, text)
