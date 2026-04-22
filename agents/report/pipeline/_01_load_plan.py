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


from knowledge_base.report_template.template_loader import _PLACEHOLDER_RE  # noqa: E402


def _extract_placeholder_keys(text: str) -> list[str]:
    return list(dict.fromkeys(_PLACEHOLDER_RE.findall(text or "")))


def _resolve_or_placeholder(text: str, context: dict) -> tuple[str, list[str]]:
    """Substitue les placeholders ; renvoie (texte résolu, clés manquantes).
    Les clés manquantes sont remplacées par '—' pour garder le prompt lisible."""
    missing: list[str] = []

    def _sub(m) -> str:
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
    context:    dict | None = None,
) -> ReportPlan:
    """Charge le YAML Design 3 et assemble un ReportPlan.

    Args:
        data_store: données produites par le BuilderAgent.
        study_plan: paramètres complémentaires (fusionnés dans context).
        yaml_path:  chemin vers le YAML du template (défaut : DEFAULT_TEMPLATE).
        context:    dict de filtrage pour l'activation des sections (ex.
                    {"gender_segmentation": "unisex"}). Si None, extraction
                    automatique depuis data_store (compat ascendante).
    """
    from knowledge_base.report_template.template_loader import (
        DEFAULT_TEMPLATE, build_manifest, load_section,
    )

    yaml_path = Path(yaml_path) if yaml_path else DEFAULT_TEMPLATE

    context_merged: dict = dict(data_store or {})
    if study_plan:
        context_merged.update(study_plan)

    # context=None → compat ascendante : pas de filtrage (toutes les sections retournées).
    # context explicite (même vide) → filtrage des sections par activation.
    manifest = build_manifest(yaml_path, context=context)
    active_section_ids = [s["id"] for s in manifest.sections if "id" in s]

    # context local pour la résolution des placeholders = data_store + study_plan
    context = context_merged

    sections: list[SectionPlan] = []
    missing_fields_global: set[str] = set()

    for sid in active_section_ids:
        sec = load_section(sid, yaml_path)
        prompt, missing_narrative = _build_prompt(sec, context)

        # Visuals : si `source` pointe vers une clé absente du contexte, on la marque manquante.
        # `source` peut être un sub-path pointé (ex. `segmentations.sexe`) — on résout.
        for v in (sec.visual_specs or []):
            source = v.get("source")
            if not source:
                continue
            root, *parts = source.split(".")
            val = context.get(root)
            for p in parts:
                if not isinstance(val, dict):
                    val = None
                    break
                val = val.get(p)
            if val is None:
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
