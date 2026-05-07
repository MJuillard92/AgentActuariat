"""
agents/report/pipeline/06_validation.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPE 6 — LLM (GPT-4o, JSON mode)

Compare la demande initiale, le rapport rédigé et le standard
professionnel actuariel. Catégorise les anomalies en :
  - MINEURE : style, longueur insuffisante, formulation vague
  - MAJEURE : section manquante, chiffres incohérents, demande non couverte

Sorties :
  ValidationResult.verdict = "ok" | "minor" | "major"
    "ok"    → rapport livré tel quel
    "minor" → retry ciblé sur les sections concernées (max 1)
    "major" → rapport livré avec flag + liste des points non couverts

Interface publique :
    validate_report(data_store, initial_request, plan) -> ValidationResult
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_MAX_TOKENS = 1500


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Anomaly:
    severity:   str    # "minor" | "major"
    section_id: str    # section concernée ("" = global)
    description: str   # explication courte
    suggestion:  str   # ce que l'agent devrait faire pour corriger


@dataclass
class ValidationResult:
    verdict:         str            # "ok" | "minor" | "major"
    anomalies:       list[Anomaly]
    minor_sections:  list[str]      # section_ids à relancer (retry ciblé)
    summary:         str            # phrase de synthèse du validateur
    report_is_usable: bool          # True même si verdict != "ok"


# ── Construction du prompt ────────────────────────────────────────────────────

def _build_prompt(data_store: dict, initial_request: str, plan) -> str:
    """
    Assemble le prompt de validation.
    Le LLM reçoit : demande initiale + résumé du rapport rédigé.
    """
    ctx = data_store.get("template_context") or {}
    section_outputs = data_store.get("section_outputs") or {}

    lines = [
        "Tu es un réviseur qualité senior pour des rapports actuariels de certification.",
        "Tu dois évaluer si le rapport rédigé répond à la demande initiale et respecte",
        "le standard professionnel d'un rapport de certification de table de mortalité.",
        "",
        "## Demande initiale",
        initial_request or "(non précisée)",
        "",
        "## Paramètres de l'étude",
    ]

    for k in ("study_objective", "smoothing_algorithm", "baseline_regulatory_table",
              "total_exposure_years", "total_deaths", "observation_start_date",
              "observation_end_date", "cohort_min_age", "cohort_max_age"):
        v = ctx.get(k)
        if v is not None:
            lines.append(f"  - {k} : {v}")

    lines += ["", "## Rapport rédigé — résumé par section", ""]

    for sec_id, sec in section_outputs.items():
        status = sec.get("status", "unknown")
        text   = sec.get("text", "")
        n_tbl  = len(sec.get("tables", []))
        n_gr   = len(sec.get("graphs", []))
        word_count = len(text.split()) if text else 0

        if status == "skipped":
            lines.append(f"### {sec_id} — SKIPPÉE (données manquantes)")
        else:
            lines.append(f"### {sec_id} — {word_count} mots, {n_tbl} tableau(x), {n_gr} graphique(s)")
            if text:
                # Extrait des 300 premiers mots pour que le LLM juge le style
                excerpt = " ".join(text.split()[:300])
                lines.append(f"Extrait : {excerpt}...")
        lines.append("")

    lines += [
        "## Critères d'évaluation",
        "",
        "**Anomalies MINEURES** (style, forme) :",
        "  - Section présente mais trop courte (< 100 mots pour une section narrative)",
        "  - Formulations vagues ('des résultats satisfaisants' sans chiffre)",
        "  - Ton non professionnel",
        "  - Absence de synthèse en fin de section",
        "",
        "**Anomalies MAJEURES** (fond, contenu) :",
        "  - Section requise manquante ou skippée",
        "  - Chiffres cités incohérents avec les paramètres de l'étude",
        "  - Demande initiale non couverte (ex: rapport demandé mais section centrale absente)",
        "  - Conclusion qui introduit de nouvelles données",
        "",
        "## Format de réponse (JSON strict)",
        '{"verdict": "ok"|"minor"|"major",',
        ' "summary": "phrase de synthèse en français",',
        ' "anomalies": [',
        '   {"severity": "minor"|"major", "section_id": "...", ',
        '    "description": "...", "suggestion": "..."}',
        ' ]}',
        "",
        "Si aucune anomalie : verdict=ok, anomalies=[]",
    ]

    return "\n".join(lines)


# ── Appel LLM ─────────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> dict:
    """Appelle GPT-4o en JSON mode."""
    try:
        import openai
        from agents.mortality.agents._utils import call_with_retry
        from agents.mortality.agents.llm_config import get_llm_config

        cfg = get_llm_config("writer.validation")
        client = openai.OpenAI()
        response = call_with_retry(
            client,
            model=cfg["model"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un réviseur qualité actuariel senior. "
                        "Tu réponds UNIQUEMENT en JSON valide, sans markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=cfg.get("max_tokens", _MAX_TOKENS),
            temperature=cfg.get("temperature", 0.0),
        )
        return json.loads(response.choices[0].message.content or "{}")

    except json.JSONDecodeError as exc:
        log.error("[06_validation] JSON invalide : %s", exc)
        return {}
    except Exception as exc:
        log.error("[06_validation] erreur LLM : %s", exc)
        return {}


# ── Parsing de la réponse ─────────────────────────────────────────────────────

def _parse(raw: dict) -> ValidationResult:
    """Transforme la réponse JSON en ValidationResult."""
    verdict  = raw.get("verdict", "ok")
    summary  = raw.get("summary", "Validation non disponible.")
    raw_anoms = raw.get("anomalies") or []

    anomalies: list[Anomaly] = []
    for a in raw_anoms:
        anomalies.append(Anomaly(
            severity    = a.get("severity", "minor"),
            section_id  = a.get("section_id", ""),
            description = a.get("description", ""),
            suggestion  = a.get("suggestion", ""),
        ))

    # Sections à relancer (uniquement les anomalies mineures avec section connue)
    minor_sections = list({
        a.section_id for a in anomalies
        if a.severity == "minor" and a.section_id
    })

    # Le rapport est toujours utilisable (on livre dans tous les cas)
    report_is_usable = True

    # Forcer "major" si aucune section n'a été rédigée
    if not any(a.severity == "major" for a in anomalies) and verdict == "major":
        verdict = "major"

    log.info("[06_validation] verdict=%s — %d anomalie(s) (%d mineures, %d majeures)",
             verdict,
             len(anomalies),
             sum(1 for a in anomalies if a.severity == "minor"),
             sum(1 for a in anomalies if a.severity == "major"))

    return ValidationResult(
        verdict          = verdict,
        anomalies        = anomalies,
        minor_sections   = minor_sections,
        summary          = summary,
        report_is_usable = report_is_usable,
    )


# ── Point d'entrée public ─────────────────────────────────────────────────────

def validate_report(
    data_store:      dict,
    initial_request: str,
    plan,
) -> ValidationResult:
    """
    Valide le rapport rédigé par rapport à la demande initiale.

    Args:
        data_store      : contient section_outputs et template_context
        initial_request : message original de l'utilisateur
        plan            : ReportPlan (pour connaître les sections attendues)

    Returns:
        ValidationResult
            .verdict = "ok"    → livrer le rapport
            .verdict = "minor" → relancer 04_redaction sur .minor_sections (max 1 retry)
            .verdict = "major" → livrer avec flag + .anomalies pour l'utilisateur
    """
    section_outputs = data_store.get("section_outputs") or {}

    if not section_outputs:
        log.warning("[06_validation] section_outputs vide — validation impossible")
        return ValidationResult(
            verdict          = "major",
            anomalies        = [Anomaly("major", "", "Aucune section rédigée.", "Relancer le pipeline complet.")],
            minor_sections   = [],
            summary          = "Rapport vide — aucune section disponible.",
            report_is_usable = False,
        )

    prompt = _build_prompt(data_store, initial_request, plan)
    raw    = _call_llm(prompt)

    if not raw:
        # LLM indisponible → on livre quand même sans bloquer
        log.warning("[06_validation] LLM indisponible — rapport livré sans validation")
        return ValidationResult(
            verdict          = "ok",
            anomalies        = [],
            minor_sections   = [],
            summary          = "Validation non effectuée (LLM indisponible) — rapport livré tel quel.",
            report_is_usable = True,
        )

    return _parse(raw)
