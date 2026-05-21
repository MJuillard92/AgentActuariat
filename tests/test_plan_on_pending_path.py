"""
HOTFIX-pre-refacto-2026-05 — Bug 19 : l'étape « récupérer le plan » doit
s'exécuter sur les DEUX chemins de routing vers le Builder, dont le
court-circuit pending (réponse à une question en attente).

Avant : répondre à la question gender_segmentation routait vers le Builder
SANS dériver ni transmettre le plan → Builder lancé à l'aveugle.
"""
from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import HumanMessage

from agents.mortality.agents.master_node import (
    _derive_calculation_plan,
    _plan_stage_label,
)


# ── Helper pur ──────────────────────────────────────────────────────────────

def test_derive_plan_returns_required_and_missing() -> None:
    """_derive_calculation_plan dérive les clés depuis report_mode + gender."""
    plan = _derive_calculation_plan({}, "full_report", "by_sex")
    assert isinstance(plan["required_keys"], list)
    assert isinstance(plan["missing_keys"], list)
    assert plan["report_mode"] == "full_report"
    assert plan["gender"] == "by_sex"
    # data_store vide → rien déjà produit
    assert plan["already_done"] == []
    assert plan["missing_keys"] == plan["required_keys"]


def test_derive_plan_marks_already_done() -> None:
    """Les clés déjà dans data_store sont classées already_done."""
    plan_all = _derive_calculation_plan({}, "full_report", "unisex")
    if not plan_all["required_keys"]:
        return  # rien à tester si le YAML ne produit aucune clé pour ce mode
    first_key = plan_all["required_keys"][0]
    plan = _derive_calculation_plan({first_key: "deja_la"}, "full_report", "unisex")
    assert first_key in plan["already_done"]
    assert first_key not in plan["missing_keys"]


def test_plan_stage_label_mentions_yaml_and_mode() -> None:
    plan = _derive_calculation_plan({}, "full_report", "by_sex")
    label = _plan_stage_label(plan)
    assert "Plan de calcul" in label
    assert "YAML" in label
    assert "full_report" in label


# ── Chemin pending ──────────────────────────────────────────────────────────

def _state_pending_gender():
    """État : une question gender_segmentation est en attente, l'utilisateur
    vient de répondre 'par sexe'."""
    return {
        "messages":   [HumanMessage(content="par sexe")],
        "data_store": {
            "_pending_need": {
                "context_key": "gender_segmentation",
                "question":    "unisex ou by_sex ?",
                "options":     ["unisex", "by_sex"],
            },
            "report_mode": "full_report",
        },
        "dataset_ref": "test_session",
    }


def test_pending_path_emits_plan_stage() -> None:
    """Répondre à la question pending doit émettre le stage 0.plan."""
    from agents.mortality.agents import master_node as mn

    with patch("openai.OpenAI"):
        result = mn.master_node(_state_pending_gender())

    stages = [e for e in result.get("events", [])
              if e.get("type") == "master_stage" and e.get("stage") == "0.plan"]
    assert len(stages) == 1, f"stage 0.plan manquant sur le chemin pending : {result.get('events')}"
    assert "Plan de calcul" in stages[0]["label"]


def test_pending_path_transmits_checklist_to_builder() -> None:
    """L'instruction envoyée au Builder via le chemin pending doit contenir
    la checklist 'Reste à produire'."""
    from agents.mortality.agents import master_node as mn

    with patch("openai.OpenAI"):
        result = mn.master_node(_state_pending_gender())

    assert result.get("active_agent") == "builder"
    msgs = result.get("messages") or []
    assert msgs, "aucune instruction Builder émise"
    instr = msgs[0].content
    assert "Reste à produire" in instr, f"checklist absente : {instr[:300]}"
    assert "gender_segmentation" in instr  # contexte de la réponse préservé


def test_pending_path_records_gender_answer() -> None:
    """La réponse gender est bien enregistrée dans study_plan."""
    from agents.mortality.agents import master_node as mn

    with patch("openai.OpenAI"):
        result = mn.master_node(_state_pending_gender())

    sp = result["data_store"].get("study_plan") or {}
    assert sp.get("gender_segmentation") == "by_sex"
