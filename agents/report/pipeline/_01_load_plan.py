"""
agents/report/pipeline/_01_load_plan.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPE 1 — Déterministe, zéro LLM (Design 3)

Lit le YAML via template_loader, résout les placeholders depuis le
data_store, assemble un prompt de rédaction par section.

Interface publique :
    load_plan(data_store, study_plan=None, yaml_path=None) -> ReportPlan
    ReportPlan, SectionPlan (dataclasses)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SectionPlan:
    section_id:       str
    label:            str
    ready:            bool
    missing_inputs:   list[str]
    prompt:           str
    visual_specs:     list[dict]
    context_snapshot: dict
    # TODO US-23/24: remove compat shim — champs Design 1 conservés pour ne pas
    # casser les modules aval (_04_redaction) avant leur réécriture.
    table_specs:      list[dict] = field(default_factory=list)
    graph_specs:      list[dict] = field(default_factory=list)
    stat_specs:       list[dict] = field(default_factory=list)


@dataclass
class ReportPlan:
    sections:       list[SectionPlan]
    context:        dict
    missing_fields: list[str]
    n_ready:        int
    n_total:        int
    yaml_path:      str


_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def _extract_placeholder_keys(text: str) -> list[str]:
    return list(dict.fromkeys(_PLACEHOLDER_RE.findall(text or "")))


def _resolve_or_placeholder(text: str, context: dict) -> tuple[str, list[str]]:
    """Substitue les placeholders ; renvoie (texte résolu, clés manquantes).
    Les clés manquantes sont remplacées par '—' pour garder le prompt lisible."""
    missing: list[str] = []

    def _sub(m: re.Match) -> str:
        key = m.group(1)
        val = context.get(key)
        if val in (None, "", []):
            missing.append(key)
            return "—"
        if isinstance(val, float):
            return f"{round(val, 4):g}"
        if isinstance(val, (list, dict)):
            return str(val)[:80]
        return str(val)

    return _PLACEHOLDER_RE.sub(_sub, text or ""), missing


def _build_prompt(section, context: dict) -> tuple[str, list[str]]:
    """Assemble le prompt de rédaction d'une section Design 3.
    Retourne (prompt, clés_manquantes_dans_narrative)."""
    narrative_text = (section.narrative or {}).get("text") or ""
    narrative_resolved, missing = _resolve_or_placeholder(narrative_text, context)

    directives = section.llm_directives or {}
    tone = directives.get("tone", "neutre, descriptif")
    length = directives.get("length_words", [])
    length_str = (
        f"{length[0]}-{length[1]} mots"
        if isinstance(length, list) and len(length) == 2 else ""
    )

    lines = [
        f"# Section {section.label} — Instructions de rédaction",
        "",
        "## Rôle",
        f"Tu rédiges la section '{section.label}' d'un rapport actuariel.",
        "Tu cites UNIQUEMENT des valeurs présentes dans les données ci-dessous.",
        "",
        f"## Ton attendu : {tone}",
    ]
    if length_str:
        lines.append(f"## Longueur cible : {length_str}")
    lines += [
        "",
        "## Narrative de référence (placeholders résolus)",
        narrative_resolved,
        "",
    ]

    visuals = section.visual_specs or []
    if visuals:
        lines.append("## Visuels à produire (détails dans SectionPlan.visual_specs)")
        for v in visuals:
            lines.append(
                f"- `{v.get('id', '?')}` ({v.get('type', '?')}) — {v.get('purpose', '')}"
            )
        lines.append("")

    lines += [
        "## Règles",
        "- Ne cite QUE des chiffres présents dans la narrative ou les visual_specs",
        "- Ne dépasse pas 10% au-delà de la longueur cible",
        "- Français, style professionnel actuariel",
    ]
    return "\n".join(lines), missing


def load_plan(
    data_store: dict,
    study_plan: dict | None = None,
    yaml_path:  str | Path | None = None,
) -> ReportPlan:
    """Charge le YAML Design 3 et assemble un ReportPlan."""
    from knowledge_base.report_template.template_loader import (
        DEFAULT_TEMPLATE, load_section,
    )
    import yaml as _yaml

    yaml_path = Path(yaml_path) if yaml_path else DEFAULT_TEMPLATE

    context: dict = dict(data_store or {})
    if study_plan:
        context.update(study_plan)

    with open(yaml_path, encoding="utf-8") as f:
        tpl = _yaml.safe_load(f) or {}
    active_section_ids = [s["id"] for s in (tpl.get("sections") or []) if "id" in s]

    sections: list[SectionPlan] = []
    missing_fields_global: set[str] = set()

    for sid in active_section_ids:
        sec = load_section(sid, yaml_path)
        prompt, missing_narrative = _build_prompt(sec, context)

        # Visuals : si `source` pointe vers une clé absente du contexte, on la marque manquante.
        for v in (sec.visual_specs or []):
            source = v.get("source")
            if source and source not in context:
                missing_narrative.append(source)

        missing_fields_global.update(missing_narrative)

        narrative_text = (sec.narrative or {}).get("text") or ""
        keys_in_narrative = _extract_placeholder_keys(narrative_text)
        sections.append(SectionPlan(
            section_id       = sec.id,
            label            = sec.label,
            ready            = len(missing_narrative) == 0,
            missing_inputs   = list(dict.fromkeys(missing_narrative)),
            prompt           = prompt,
            visual_specs     = list(sec.visual_specs or []),
            context_snapshot = {k: context[k] for k in keys_in_narrative if k in context},
        ))

    n_ready = sum(1 for s in sections if s.ready)

    return ReportPlan(
        sections       = sections,
        context        = context,
        missing_fields = sorted(missing_fields_global),
        n_ready        = n_ready,
        n_total        = len(sections),
        yaml_path      = str(yaml_path),
    )
