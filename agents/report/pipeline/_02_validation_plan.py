"""
agents/report/pipeline/02_validation_plan.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPE 2 — LLM (GPT-4o, JSON mode)

Reçoit le ReportPlan produit par 01_load_plan.
Pour chaque section marquée ready=True, demande au LLM de juger si les
données sont non seulement présentes mais SUFFISANTES pour rédiger la
section de manière professionnelle.

load_yaml_template vérifie la présence des clés — 02_validation_plan
vérifie la qualité et la suffisance des valeurs.

Interface publique :
    validate_plan(plan, data_store) -> PlanValidation
    PlanValidation, SectionValidation (dataclasses)

Sorties :
    all_valid = True  → on passe à 03_completion_plan
    all_valid = False → PlanValidation.ko_fields contient les champs
                        à transmettre au MasterAgent pour relancer le Builder
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SectionValidation:
    section_id:              str
    valid:                   bool   # True = données suffisantes
    reason:                  str    # explication courte
    missing_or_insufficient: list[str] = field(default_factory=list)


@dataclass
class PlanValidation:
    sections:    list[SectionValidation]
    all_valid:   bool
    ko_sections: list[str]   # section_ids avec valid=False
    ko_fields:   list[str]   # tous les champs insuffisants (pour le Master)


# ── Construction du prompt ────────────────────────────────────────────────────

def _build_prompt(plan, data_store: dict) -> str:
    """
    Construit le prompt envoyé au LLM.
    Utilise plan.context (placeholders résolus par load_yaml_template) plutôt que
    le data_store brut — le LLM voit les valeurs effectives, pas la structure interne.
    """
    # plan.context contient tous les placeholders résolus par load_yaml_template
    ctx = getattr(plan, "context", {}) or {}

    lines = [
        "Tu es un contrôleur qualité pour un rapport de certification de table de mortalité.",
        "Pour chaque section ci-dessous, tu dois juger si les données disponibles sont",
        "SUFFISANTES pour rédiger la section de manière professionnelle.",
        "",
        "IMPORTANT : réponds UNIQUEMENT sur la base des données listées ci-dessous.",
        "Ne marque une section KO QUE si une donnée CRITIQUE est réellement absente",
        "ou si une série par âge contient moins de 5 entrées non-nulles.",
        "Si une donnée est présente avec des valeurs non-nulles, marque valid=true.",
        "",
        "Réponds UNIQUEMENT en JSON valide avec ce format :",
        '{"sections": [{"section_id": "...", "valid": true/false, "reason": "...",',
        '               "missing_or_insufficient": ["field1", "field2"]}]}',
        "",
        "## Valeurs résolues (résultat de load_yaml_template)",
        "",
    ]

    # Afficher les valeurs résolues du contexte — source de vérité
    for key, val in ctx.items():
        if isinstance(val, dict):
            n = len(val)
            non_null = sum(1 for v in val.values() if v is not None and v != 0)
            lines.append(f"  - `{key}` : {n} entrées ({non_null} non-nulles)")
        elif isinstance(val, list):
            lines.append(f"  - `{key}` : {len(val)} enregistrements — ex: {str(val[:2])[:60]}")
        elif val is not None:
            lines.append(f"  - `{key}` : {str(val)[:80]}")

    lines += ["", "## Sections à valider", ""]

    for sec in plan.sections:
        if not sec.ready:
            continue
        lines += [
            f"### {sec.label} (`{sec.section_id}`)",
            f"Champs requis : {', '.join(sec.missing_inputs) or 'tous présents (load_yaml_template ready=True)'}",
            "",
        ]

    return "\n".join(lines)


# ── Appel LLM ─────────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> dict:
    """Appelle GPT-4o en mode JSON et retourne le dict parsé."""
    import openai
    from agents.mortality.agents._utils import call_with_retry

    client = openai.OpenAI()
    try:
        response = call_with_retry(
            client,
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un contrôleur qualité actuariel. "
                        "Tu réponds UNIQUEMENT en JSON valide, sans markdown, sans commentaire."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1500,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    except json.JSONDecodeError as exc:
        log.warning("[02_validation_plan] JSON invalide : %s", exc)
        return {"sections": []}
    except Exception as exc:
        log.error("[02_validation_plan] Erreur LLM : %s", exc)
        return {"sections": []}


# ── Parsing de la réponse ─────────────────────────────────────────────────────

def _parse_response(raw: dict, plan) -> list[SectionValidation]:
    """
    Transforme la réponse JSON du LLM en liste de SectionValidation.
    Les sections ready=False sont automatiquement marquées valid=False.
    Les sections absentes de la réponse LLM sont marquées valid=True (bénéfice du doute).
    """
    results: list[SectionValidation] = []

    # Index des réponses LLM par section_id
    llm_map: dict[str, dict] = {}
    for item in (raw.get("sections") or []):
        sid = item.get("section_id", "")
        if sid:
            llm_map[sid] = item

    for sec in plan.sections:
        if not sec.ready:
            # Section non prête selon load_yaml_template → KO par définition
            results.append(SectionValidation(
                section_id              = sec.section_id,
                valid                   = False,
                reason                  = f"Données manquantes selon load_yaml_template : {sec.missing_inputs}",
                missing_or_insufficient = sec.missing_inputs,
            ))
            continue

        llm_item = llm_map.get(sec.section_id)
        if llm_item is None:
            # Absent de la réponse → bénéfice du doute
            results.append(SectionValidation(
                section_id = sec.section_id,
                valid      = True,
                reason     = "Non évalué par le LLM — supposé valide",
            ))
        else:
            results.append(SectionValidation(
                section_id              = sec.section_id,
                valid                   = bool(llm_item.get("valid", True)),
                reason                  = llm_item.get("reason", ""),
                missing_or_insufficient = llm_item.get("missing_or_insufficient", []),
            ))

    return results


# ── Point d'entrée public ─────────────────────────────────────────────────────

def validate_plan(plan, data_store: dict) -> PlanValidation:
    """
    Valide le ReportPlan : pour chaque section ready=True, le LLM vérifie
    que les données sont suffisantes (pas juste présentes).

    Args:
        plan       : ReportPlan produit par 01_load_plan.load_plan()
        data_store : résultats bruts du BuilderAgent

    Returns:
        PlanValidation
            .all_valid = True  → passer à 03_completion_plan
            .all_valid = False → .ko_fields contient les champs à demander au Builder
    """
    log.info("[02_validation_plan] validation de %d sections (%d ready)",
             len(plan.sections),
             sum(1 for s in plan.sections if s.ready))

    # Si aucune section n'est ready, inutile d'appeler le LLM
    if not any(s.ready for s in plan.sections):
        ko = [s.section_id for s in plan.sections]
        all_fields = [f for s in plan.sections for f in s.missing_inputs]
        log.warning("[02_validation_plan] aucune section ready — toutes KO")
        return PlanValidation(
            sections    = [SectionValidation(s.section_id, False,
                           "Aucune donnée disponible", s.missing_inputs)
                           for s in plan.sections],
            all_valid   = False,
            ko_sections = ko,
            ko_fields   = list(set(all_fields)),
        )

    prompt   = _build_prompt(plan, data_store)
    raw      = _call_llm(prompt)
    sections = _parse_response(raw, plan)

    ko_sections = [s.section_id for s in sections if not s.valid]
    ko_fields   = list(set(
        f for s in sections if not s.valid
        for f in s.missing_or_insufficient
    ))
    all_valid = len(ko_sections) == 0

    log.info("[02_validation_plan] résultat : %d/%d sections valides — KO : %s",
             len(sections) - len(ko_sections), len(sections), ko_sections or "aucune")

    return PlanValidation(
        sections    = sections,
        all_valid   = all_valid,
        ko_sections = ko_sections,
        ko_fields   = ko_fields,
    )
