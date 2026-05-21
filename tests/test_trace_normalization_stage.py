"""
HOTFIX-pre-refacto-2026-05 — Bug 16 (A2) : master_node émet un stage
'0.norm' APRÈS la normalisation, reflétant ce qui s'est réellement passé
(construction / cache / différée), plus le stage trompeur '0.b' émis avant.
"""
from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import HumanMessage


def _extract_stage(events, stage_id):
    return next((e for e in (events or [])
                 if e.get("type") == "master_stage" and e.get("stage") == stage_id), None)


def _base_state():
    return {
        "messages":    [HumanMessage(content="calcule l'exposition")],
        "data_store":  {},          # _disambiguation_done absent → entre dans le bloc
        "dataset_ref": None,
    }


def test_stage_0norm_emitted_after_normalization() -> None:
    """Quand maybe_normalize_records renvoie des updates, le stage 0.norm
    affiche le détail de la construction."""
    from agents.mortality.agents import master_node as mn

    norm_updates = {
        "records_normalized":     True,
        "dataset_ref_normalized": "/tmp/x_normalized.parquet",
        "_audit": {"normalization": {"rows_in": 530345, "rows_out": 530345}},
    }
    with patch("openai.OpenAI"), \
         patch("agents.master.disambiguation.run_disambiguation",
               return_value={"status": "ready"}), \
         patch("agents.master.disambiguation.maybe_normalize_records",
               return_value=norm_updates), \
         patch.object(mn, "_classify_intent",
                      return_value={"intent": "question", "kind": "question",
                                    "write": "ask", "report_mode": "full_report",
                                    "confidence": 0.9, "reply": ""}):
        result = mn.master_node(_base_state())

    s = _extract_stage(result.get("events"), "0.norm")
    assert s is not None, f"stage 0.norm manquant : {result.get('events')}"
    assert "Construction base de données synthétique" in s["label"]
    assert "530 345 lignes" in s["label"]


def test_stage_0norm_reused_cache() -> None:
    """Si déjà normalisé et pas d'updates, le stage indique la réutilisation."""
    from agents.mortality.agents import master_node as mn

    state = _base_state()
    state["data_store"]["records_normalized"] = True

    with patch("openai.OpenAI"), \
         patch("agents.master.disambiguation.run_disambiguation",
               return_value={"status": "ready"}), \
         patch("agents.master.disambiguation.maybe_normalize_records",
               return_value=None), \
         patch.object(mn, "_classify_intent",
                      return_value={"intent": "question", "kind": "question",
                                    "write": "ask", "report_mode": "full_report",
                                    "confidence": 0.9, "reply": ""}):
        result = mn.master_node(state)

    s = _extract_stage(result.get("events"), "0.norm")
    assert s is not None
    assert "réutilisée" in s["label"].lower()


def test_stage_0norm_deferred_when_no_updates() -> None:
    """Pas d'updates + pas déjà normalisé → stage 'normalisation différée'."""
    from agents.mortality.agents import master_node as mn

    with patch("openai.OpenAI"), \
         patch("agents.master.disambiguation.run_disambiguation",
               return_value={"status": "ready"}), \
         patch("agents.master.disambiguation.maybe_normalize_records",
               return_value=None), \
         patch.object(mn, "_classify_intent",
                      return_value={"intent": "question", "kind": "question",
                                    "write": "ask", "report_mode": "full_report",
                                    "confidence": 0.9, "reply": ""}):
        result = mn.master_node(_base_state())

    s = _extract_stage(result.get("events"), "0.norm")
    assert s is not None
    assert "différée" in s["label"].lower()
