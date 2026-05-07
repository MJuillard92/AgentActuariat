"""Tests d'intégration : master_node détecte le marqueur need_user_input
émis par le Builder et applique la résolution 3-niveaux."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _builder_emits_need(question="Lambda 100, 200 ou 500 ?",
                        context_key="smoothing_lambda",
                        options=None,
                        default=None):
    return AIMessage(
        content="J'ai besoin d'une précision pour le lissage.",
        additional_kwargs={
            "need_user_input": {
                "context_key": context_key,
                "question":    question,
                "options":     options or [100, 200, 500],
                **({"default": default} if default is not None else {}),
            }
        }
    )


def test_master_resolves_via_study_plan_and_routes_back_to_builder():
    """study_plan contient déjà la réponse → Master injecte et route Builder."""
    from agents.mortality.agents import master_node as mn

    state = {
        "messages":    [
            HumanMessage(content="construit la table"),
            _builder_emits_need(),
        ],
        "data_store":  {
            "_disambiguation_done":   True,
            "_master_builder_cycles": 1,
            "study_plan":             {"smoothing_lambda": 200, "gender_segmentation": "unisex"},
            "_user_messages":         ["construit la table"],
        },
        "dataset_ref": None,
        "active_agent": "master",
    }
    out = mn.master_node(state)

    assert out.get("active_agent") == "builder"
    msgs = out.get("messages") or []
    injection = next((m for m in msgs if isinstance(m, HumanMessage)), None)
    assert injection is not None
    assert "200" in injection.content
    src = (injection.additional_kwargs or {}).get("source")
    assert src == "master_synthetic"


def test_master_forwards_to_user_when_no_signal(monkeypatch):
    """study_plan vide + LLM ne trouve rien → Master pose la question à l'user."""
    from agents.mortality.agents import master_node as mn
    from agents.master import question_filter as qf

    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": False, "confidence": 0.1})

    state = {
        "messages":    [
            HumanMessage(content="construit la table"),
            _builder_emits_need(),
        ],
        "data_store":  {
            "_disambiguation_done":   True,
            "_master_builder_cycles": 1,
            "study_plan":             {"gender_segmentation": "unisex"},
            "_user_messages":         ["construit la table"],
        },
        "dataset_ref": None,
        "active_agent": "master",
    }
    out = mn.master_node(state)

    assert out.get("active_agent") != "builder"
    msgs = out.get("messages") or []
    assert any(isinstance(m, AIMessage) and "lambda" in (m.content or "").lower() for m in msgs)
    assert out["data_store"].get("_pending_need") is not None


def test_master_extracts_user_response_and_routes_back(monkeypatch):
    """Quand _pending_need existe et user répond, Master extrait et inject Builder."""
    from agents.mortality.agents import master_node as mn
    from agents.master import question_filter as qf

    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": True, "value": 200, "confidence": 0.95})

    state = {
        "messages":    [HumanMessage(content="200 ça me va")],
        "data_store":  {
            "_disambiguation_done":   True,
            "_master_builder_cycles": 1,
            "study_plan":             {"gender_segmentation": "unisex"},
            "_user_messages":         ["construit la table", "200 ça me va"],
            "_pending_need":          {
                "context_key": "smoothing_lambda",
                "question":    "Lambda 100, 200 ou 500 ?",
                "options":     [100, 200, 500],
            },
        },
        "dataset_ref": None,
        "active_agent": "master",
    }
    out = mn.master_node(state)

    assert out["data_store"]["study_plan"]["smoothing_lambda"] == 200
    assert "_pending_need" not in out["data_store"]
    assert out.get("active_agent") == "builder"


def test_master_reprompts_when_extract_fails(monkeypatch):
    """Quand _pending_need est set mais extract_user_answer échoue (typo),
    Master doit RE-POSER la question avec un hint au lieu de tomber dans
    classify_intent (sinon le mot isolé est classé 'question hors calculs')."""
    from agents.mortality.agents import master_node as mn
    from agents.master import question_filter as qf

    # extract_user_answer retourne None → simule l'échec sur typo "unisexe"
    monkeypatch.setattr(qf, "extract_user_answer",
                        lambda text, need: None)

    # Patch classify_intent pour vérifier qu'il N'EST PAS appelé
    classify_called = []
    def _track_classify(*a, **kw):
        classify_called.append(True)
        return {"kind": "question", "write": "ask", "report_mode": "full_report",
                "intent": "question", "reply": ""}
    monkeypatch.setattr(mn, "_classify_intent", _track_classify)

    state = {
        "messages":    [HumanMessage(content="unisexe")],
        "data_store":  {
            "_disambiguation_done": True,
            "_pending_need":        {
                "context_key": "gender_segmentation",
                "question":    "Voulez-vous unisex ou by_sex ?",
                "options":     ["unisex", "by_sex"],
                "default":     "unisex",
            },
            "study_plan":           {},
        },
        "dataset_ref": None,
        "active_agent": "master",
    }
    out = mn.master_node(state)

    # classify_intent n'a PAS été appelé (pending_need a court-circuité)
    assert not classify_called, "Master a appelé classify alors que _pending_need était set"

    # Master a re-émis un AIMessage avec le hint
    msgs = out.get("messages") or []
    assert any(isinstance(m, AIMessage) and "unisex" in (m.content or "").lower()
               and "by_sex" in (m.content or "").lower()
               for m in msgs), (
        f"Master n'a pas re-posé la question avec hint. Messages : "
        f"{[(type(m).__name__, m.content[:100]) for m in msgs]}"
    )

    # _pending_need toujours actif (pas consommé tant que la réponse est pas comprise)
    assert out["data_store"].get("_pending_need") is not None


def test_master_uses_default_after_max_questions_in_cycle(monkeypatch):
    """Au-delà de 3 questions dans un cycle, Master force use_default."""
    from agents.mortality.agents import master_node as mn
    from agents.master import question_filter as qf

    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": False, "confidence": 0.1})

    builder_msg = _builder_emits_need(
        question="Quel paramètre ?",
        context_key="lambda_3rd",
        options=[100, 200],
        default=100,
    )
    state = {
        "messages":    [
            HumanMessage(content="bonjour"),
            builder_msg,
        ],
        "data_store":  {
            "_disambiguation_done":             True,
            "_master_builder_cycles":           1,
            "_questions_asked_this_cycle":      3,
            "study_plan":                       {"gender_segmentation": "unisex"},
            "_user_messages":                   ["bonjour"],
        },
        "dataset_ref": None,
        "active_agent": "master",
    }
    out = mn.master_node(state)

    assert out.get("active_agent") == "builder"
    sp = out["data_store"].get("study_plan", {})
    assert sp.get("lambda_3rd") == 100
    assert "_pending_need" not in out["data_store"]
