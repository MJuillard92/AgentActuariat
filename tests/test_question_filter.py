"""Tests pour agents.master.question_filter — résolution des questions Builder."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Dataclass + Niveau 1 ────────────────────────────────────────────────

def test_resolution_dataclass_has_required_fields():
    from agents.master.question_filter import QuestionResolution
    r = QuestionResolution(decision="answered", value=200, source="study_plan", confidence=1.0)
    assert r.decision == "answered"
    assert r.value == 200
    assert r.source == "study_plan"
    assert r.confidence == 1.0


def test_level1_finds_in_study_plan():
    from agents.master.question_filter import _try_resolve_from_data_store
    need = {"context_key": "smoothing_lambda"}
    data_store = {"study_plan": {"smoothing_lambda": 200}}
    val, source = _try_resolve_from_data_store(need, data_store)
    assert val == 200
    assert source == "study_plan"


def test_level1_finds_at_top_level_data_store():
    from agents.master.question_filter import _try_resolve_from_data_store
    need = {"context_key": "report_mode"}
    data_store = {"report_mode": "raw_rates"}
    val, source = _try_resolve_from_data_store(need, data_store)
    assert val == "raw_rates"
    assert source == "data_store"


def test_level1_returns_none_when_not_found():
    from agents.master.question_filter import _try_resolve_from_data_store
    need = {"context_key": "lambda_inconnu"}
    val, source = _try_resolve_from_data_store(need, {"study_plan": {}})
    assert val is None
    assert source is None


# ── detect_need_in_message ──────────────────────────────────────────────

def test_detect_need_in_message_returns_dict():
    from agents.master.question_filter import detect_need_in_message
    from langchain_core.messages import AIMessage
    need = {"context_key": "smoothing_lambda", "question": "?", "options": [100, 200, 500]}
    msg = AIMessage(content="...", additional_kwargs={"need_user_input": need})
    assert detect_need_in_message(msg) == need


def test_detect_need_in_message_returns_none_when_absent():
    from agents.master.question_filter import detect_need_in_message
    from langchain_core.messages import AIMessage
    msg = AIMessage(content="Plan d'analyse...")
    assert detect_need_in_message(msg) is None


def test_detect_need_in_message_returns_none_for_non_ai_message():
    from agents.master.question_filter import detect_need_in_message
    from langchain_core.messages import HumanMessage
    msg = HumanMessage(content="ok")
    assert detect_need_in_message(msg) is None


# ── Niveau 2 ─────────────────────────────────────────────────────────────

def test_level2_llm_infers_lambda_from_smooth_keyword(monkeypatch):
    from agents.master import question_filter as qf
    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": True, "value": 100, "confidence": 0.85,
                                   "reasoning": "user dit 'doux'"})
    need = {"context_key": "smoothing_lambda", "question": "Lambda ?", "options": [100, 200, 500]}
    inf = qf._llm_infer_from_history(need, ["Construis avec un lissage doux"])
    assert inf["answered"] is True
    assert inf["value"] == 100
    assert inf["confidence"] >= 0.7


def test_level2_returns_no_answer_when_user_silent(monkeypatch):
    from agents.master import question_filter as qf
    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": False, "value": None, "confidence": 0.2})
    inf = qf._llm_infer_from_history(
        {"context_key": "smoothing_lambda", "question": "?", "options": [100, 200, 500]},
        ["Bonjour"],
    )
    assert inf["answered"] is False


def test_level2_handles_empty_user_messages(monkeypatch):
    from agents.master import question_filter as qf
    called = []
    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: called.append(p) or {"answered": False})
    inf = qf._llm_infer_from_history({"context_key": "x", "question": "y"}, [])
    assert inf["answered"] is False
    assert called == []  # pas d'appel LLM si rien à inférer


# ── Orchestrateur ────────────────────────────────────────────────────────

def test_resolve_uses_level1_when_study_plan_match():
    from agents.master.question_filter import resolve_builder_question
    need = {"context_key": "smoothing_lambda", "question": "?", "options": [100, 200]}
    data_store = {"study_plan": {"smoothing_lambda": 200}}
    res = resolve_builder_question(need, data_store, ["bonjour"])
    assert res.decision == "answered"
    assert res.value == 200
    assert res.source == "study_plan"
    assert res.confidence == 1.0


def test_resolve_uses_level2_when_level1_misses(monkeypatch):
    from agents.master import question_filter as qf
    from agents.master.question_filter import resolve_builder_question
    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": True, "value": 100, "confidence": 0.85})
    need = {"context_key": "smoothing_lambda", "question": "?", "options": [100, 200]}
    res = resolve_builder_question(need, {}, ["lissage doux"])
    assert res.decision == "answered"
    assert res.value == 100
    assert res.source == "llm_inference"
    assert res.confidence == 0.85


def test_resolve_forwards_when_no_level_matches(monkeypatch):
    from agents.master import question_filter as qf
    from agents.master.question_filter import resolve_builder_question
    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": False, "confidence": 0.2})
    need = {"context_key": "smoothing_lambda", "question": "?", "options": [100, 200]}
    res = resolve_builder_question(need, {}, ["bonjour"])
    assert res.decision == "forward"
    assert res.value is None


def test_resolve_forwards_when_confidence_below_threshold(monkeypatch):
    from agents.master import question_filter as qf
    from agents.master.question_filter import resolve_builder_question
    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": True, "value": 100, "confidence": 0.5})
    need = {"context_key": "smoothing_lambda", "question": "?", "options": [100, 200]}
    res = resolve_builder_question(need, {}, ["lissage"], confidence_threshold=0.7)
    assert res.decision == "forward"


# ── extract_user_answer ─────────────────────────────────────────────────

def test_extract_user_answer_uses_default_when_options_unspecified():
    from agents.master.question_filter import extract_user_answer
    need = {"context_key": "objectif", "question": "Quel objectif ?"}
    val = extract_user_answer("certifier la table", need)
    assert val == "certifier la table"


def test_extract_user_answer_matches_option_when_explicit(monkeypatch):
    from agents.master import question_filter as qf
    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": True, "value": 200, "confidence": 0.95})
    need = {"context_key": "lambda", "question": "?", "options": [100, 200, 500]}
    val = qf.extract_user_answer("200 ça me va", need)
    assert val == 200


def test_extract_user_answer_returns_none_when_unparseable(monkeypatch):
    from agents.master import question_filter as qf
    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": False, "confidence": 0.1})
    need = {"context_key": "lambda", "question": "?", "options": [100, 200]}
    val = qf.extract_user_answer("euh je sais pas", need)
    assert val is None
