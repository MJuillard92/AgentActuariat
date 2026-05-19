"""Tests pour Bug 8 — WriterAgent stages WRITER.0 → WRITER.4.

Avant ce fix, l'UI n'affichait que "WriterAgent actif" pendant les 1-2min
de génération PDF. Maintenant 4-5 stages encadrent l'activité (selon que
le pipeline réussit, manque des données, ou plante).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage


def _fake_pipeline_result(status="success", nb_sections=10,
                           output_path="/tmp/rapport_test.pdf",
                           validation_summary="OK",
                           need_data=None, anomalies=None):
    r = MagicMock()
    r.status = status
    r.nb_sections = nb_sections
    r.output_path = output_path
    r.validation_summary = validation_summary
    r.need_data = need_data or []
    r.anomalies = anomalies or []
    return r


def _state(data_store=None):
    return {
        "messages":   [HumanMessage(content="génère le rapport")],
        "data_store": data_store or {"report_mode": "full_report"},
    }


def _extract_stage_events(events):
    return [e for e in (events or [])
            if e.get("type") == "master_stage"
            and (e.get("stage") or "").startswith("WRITER.")]


# ──────────────────────────────────────────────────────────────────────
# Succès nominal — 4 stages présents
# ──────────────────────────────────────────────────────────────────────

def test_writer_emits_stages_on_nominal_path():
    """Pipeline success → WRITER.0/1/2/4 émis."""
    from agents.mortality.agents.writer_node import writer_node

    fake_result = _fake_pipeline_result(status="success")
    with patch("agents.report.pipeline.run_pipeline.run",
               return_value=fake_result):
        result = writer_node(_state())

    stages = _extract_stage_events(result.get("events"))
    stage_ids = [s["stage"] for s in stages]
    assert "WRITER.0" in stage_ids
    assert "WRITER.1" in stage_ids
    assert "WRITER.2" in stage_ids
    assert "WRITER.4" in stage_ids
    # WRITER.3 absent sur le path success (réservé erreur/need_data)
    assert "WRITER.3" not in stage_ids


def test_writer_stage_4_includes_output_path_and_sections():
    """WRITER.4 doit mentionner le chemin du PDF et le nombre de sections."""
    from agents.mortality.agents.writer_node import writer_node

    fake_result = _fake_pipeline_result(
        status="success", nb_sections=15, output_path="/tmp/test.pdf",
    )
    with patch("agents.report.pipeline.run_pipeline.run",
               return_value=fake_result):
        result = writer_node(_state())

    stages = _extract_stage_events(result.get("events"))
    w4 = next((s for s in stages if s["stage"] == "WRITER.4"), None)
    assert w4 is not None
    # Nouveau label user-friendly (Bug 13) : "PDF produit (15 sections)"
    # Le chemin n'apparaît plus dans le label stage (visible via event report_ready).
    assert "PDF" in w4["label"]
    assert "15 sections" in w4["label"]


# ──────────────────────────────────────────────────────────────────────
# Path need_data — WRITER.3 mentionne les champs
# ──────────────────────────────────────────────────────────────────────

def test_writer_emits_stage_3_on_need_data():
    """status=need_data → WRITER.3 avec label des champs manquants."""
    from agents.mortality.agents.writer_node import writer_node

    fake_result = _fake_pipeline_result(
        status="need_data",
        need_data=["smr_par_decile", "abatement_table"],
    )
    with patch("agents.report.pipeline.run_pipeline.run",
               return_value=fake_result):
        result = writer_node(_state())

    stages = _extract_stage_events(result.get("events"))
    w3 = next((s for s in stages if s["stage"] == "WRITER.3"), None)
    assert w3 is not None
    # Nouveau label user-friendly (Bug 13) : "Données insuffisantes (N champ(s) manquant(s))"
    # Les noms de champs ne sont plus dans le label stage (visibles via le message).
    assert "Données insuffisantes" in w3["label"] or "insuffisantes" in w3["label"]
    assert "2" in w3["label"], f"Nombre de champs manquants attendu : {w3['label']}"


# ──────────────────────────────────────────────────────────────────────
# Path error — WRITER.3 sur erreur pipeline
# ──────────────────────────────────────────────────────────────────────

def test_writer_emits_stage_3_on_error():
    """status=error → WRITER.3 avec extrait du message d'erreur."""
    from agents.mortality.agents.writer_node import writer_node

    fake_result = _fake_pipeline_result(
        status="error",
        validation_summary="reportlab: missing TTF font",
    )
    with patch("agents.report.pipeline.run_pipeline.run",
               return_value=fake_result):
        result = writer_node(_state())

    stages = _extract_stage_events(result.get("events"))
    w3 = next((s for s in stages if s["stage"] == "WRITER.3"), None)
    assert w3 is not None
    # Nouveau label user-friendly (Bug 13) : "Erreur lors de l'assemblage PDF"
    # Le détail technique (validation_summary) est dans le message, pas le stage.
    assert "Erreur" in w3["label"]
