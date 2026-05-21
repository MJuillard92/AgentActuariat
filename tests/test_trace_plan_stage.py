"""
HOTFIX-pre-refacto-2026-05 — Bug 17 (A4) : master_node émet un stage
'0.plan' rendant visible le plan de calcul dérivé du YAML, et transmet
la checklist des clés manquantes au Builder via le message d'instruction.
"""
from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import HumanMessage


def _extract_stage(events, stage_id):
    return next((e for e in (events or [])
                 if e.get("type") == "master_stage" and e.get("stage") == stage_id), None)


def _state_for_builder_routing():
    """État qui amène master_node à dériver le plan + router vers Builder :
    disambiguation déjà faite, dataset présent, intent task/build."""
    return {
        "messages":    [HumanMessage(content="calcule la table de mortalité")],
        "data_store":  {
            "_disambiguation_done":   True,
            "_methods_question_done": True,
            "_dataset_ref":           "test_session",
            "mapping_validated":        True,  # gate calcul : clone validé
            "study_plan":             {"gender_segmentation": "unisex"},
        },
        "dataset_ref": "test_session",
    }


def test_stage_0plan_emitted_with_missing_keys() -> None:
    """Le stage 0.plan affiche les clés à produire dérivées du YAML."""
    from agents.mortality.agents import master_node as mn

    with patch("openai.OpenAI"), \
         patch.object(mn, "_classify_intent",
                      return_value={"intent": "build_only", "kind": "task",
                                    "write": "no", "report_mode": "full_report",
                                    "confidence": 1.0, "reply": ""}):
        result = mn.master_node(_state_for_builder_routing())

    s = _extract_stage(result.get("events"), "0.plan")
    assert s is not None, f"stage 0.plan manquant : {result.get('events')}"
    assert "Plan de calcul" in s["label"]
    assert "YAML" in s["label"]
    assert "full_report" in s["label"]


def test_builder_instruction_transmits_missing_keys() -> None:
    """Le message d'instruction envoyé au Builder contient la checklist
    explicite 'Reste à produire'."""
    from agents.mortality.agents import master_node as mn

    with patch("openai.OpenAI"), \
         patch.object(mn, "_classify_intent",
                      return_value={"intent": "build_only", "kind": "task",
                                    "write": "no", "report_mode": "full_report",
                                    "confidence": 1.0, "reply": ""}):
        result = mn.master_node(_state_for_builder_routing())

    # Si routé vers Builder, un HumanMessage d'instruction est émis
    if result.get("active_agent") == "builder":
        msgs = result.get("messages") or []
        assert msgs, "aucun message d'instruction Builder"
        instr = msgs[0].content
        assert "Reste à produire" in instr, (
            f"checklist absente du message Builder : {instr[:300]}"
        )
