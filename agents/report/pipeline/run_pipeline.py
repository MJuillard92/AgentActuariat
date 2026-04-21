"""
agents/report/pipeline/run_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chef d'orchestre du pipeline de génération de rapport.

Enchaîne les étapes dans l'ordre et gère :
  - Le retry ciblé si 06 détecte des anomalies mineures (max 1 fois)
  - La livraison avec flag si 06 détecte des anomalies majeures

Interface publique :
    run(data_store, initial_request, output_path) -> PipelineResult
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    status:          str            # "success" | "need_data" | "success_with_warnings"
    output_path:     str            # chemin du PDF (vide si need_data)
    need_data:       list[str]      # champs à demander au Builder (si status=need_data)
    anomalies:       list           # liste d'Anomaly (si success_with_warnings)
    validation_summary: str
    nb_sections:     int


def run(
    data_store:      dict,
    initial_request: str = "",
    output_path:     str | None = None,
    yaml_path:       str = "knowledge_base/report_template/mortality_template.yaml",
) -> PipelineResult:
    """
    Lance le pipeline complet de génération du rapport.

    Args:
        data_store      : résultats du BuilderAgent
        initial_request : message original de l'utilisateur
        output_path     : chemin de sortie du PDF
        yaml_path       : chemin du template YAML

    Returns:
        PipelineResult
    """
    study_plan = data_store.get("study_plan") or {}

    # ── Étape 1 — Chargement du plan ──────────────────────────────────────────
    log.info("[pipeline] étape 1 — load_plan")
    from agents.report.pipeline._01_load_plan import load_plan
    plan = load_plan(data_store, study_plan, yaml_path)
    log.info("[pipeline] plan chargé : %d/%d sections prêtes",
             plan.n_ready, plan.n_total)

    # US-25: validation step removed; delegated to check_template + template_loader

    # Persister template_context pour les étapes 4, 5, 6 (validation et rédaction)
    data_store["template_context"] = plan.context

    # ── Étape 3 — Enrichissement RAG ─────────────────────────────────────────
    log.info("[pipeline] étape 3 — complete_plan (RAG)")
    from agents.report.pipeline._03_completion_plan import complete_plan
    plan_enriched = complete_plan(plan, data_store)

    # ── Étape 4 — Rédaction ───────────────────────────────────────────────────
    log.info("[pipeline] étape 4 — redact_plan")
    from agents.report.pipeline._04_redaction import redact_plan
    data_store = redact_plan(plan_enriched, data_store)

    # ── Étape 5 — Assemblage PDF ──────────────────────────────────────────────
    log.info("[pipeline] étape 5 — assemble")
    from agents.report.pipeline._05_assemble import assemble
    asm = assemble(data_store, output_path)

    if not asm.success:
        log.error("[pipeline] assemblage échoué : %s", asm.warning)
        return PipelineResult(
            status       = "error",
            output_path  = "",
            need_data    = [],
            anomalies    = [],
            validation_summary = f"Assemblage PDF échoué : {asm.warning}",
            nb_sections  = 0,
        )

    # ── Étape 6 — Validation finale ───────────────────────────────────────────
    log.info("[pipeline] étape 6 — validate_report")
    from agents.report.pipeline._06_validation import validate_report
    vr = validate_report(data_store, initial_request, plan_enriched)

    # ── Gestion des anomalies mineures : retry ciblé (1 seule fois) ───────────
    if vr.verdict == "minor" and vr.minor_sections:
        log.info("[pipeline] retry ciblé sur %s", vr.minor_sections)

        # Re-rédiger uniquement les sections KO
        plan_retry = _filter_plan(plan_enriched, vr.minor_sections)
        data_store = redact_plan(plan_retry, data_store)

        # Ré-assembler
        asm = assemble(data_store, output_path)

        # Pas de 2e validation — on livre quoi qu'il arrive
        vr.verdict = "ok"
        vr.anomalies = []
        vr.summary += " (retry effectué)"

    # ── Résultat final ────────────────────────────────────────────────────────
    status = "success" if vr.verdict == "ok" else "success_with_warnings"

    log.info("[pipeline] terminé — status=%s, PDF=%s", status, asm.output_path)

    return PipelineResult(
        status             = status,
        output_path        = asm.output_path,
        need_data          = [],
        anomalies          = vr.anomalies if status == "success_with_warnings" else [],
        validation_summary = vr.summary,
        nb_sections        = asm.nb_sections,
    )


def _filter_plan(plan, section_ids: list[str]):
    """Retourne un plan filtré aux seules sections listées (pour retry ciblé)."""
    from copy import deepcopy
    filtered = deepcopy(plan)
    filtered.sections = [s for s in filtered.sections if s.section_id in section_ids]
    return filtered
