"""
HOTFIX-pre-refacto-2026-05 — Bug 2 : émettre des sous-stages
(0.d.1 / 0.d.2 / 0.d.3) pendant classify_intent pour rendre la
qualification Master visible côté canvas, comme le fait déjà RAGAgent.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _fake_llm_response(payload: dict) -> MagicMock:
    """Construit un faux objet OpenAI ChatCompletion contenant `payload` en JSON."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(payload)
    return resp


@pytest.fixture
def stub_classification_payload() -> dict:
    return {
        "kind":                "task",
        "write":               "yes",
        "report_mode":         "full_report",
        "gender_segmentation": "unknown",
        "confidence":          0.9,
        "reasoning":           "calcul d'exposition demandé",
        "reply":               "OK, je lance.",
    }


def test_classify_intent_emits_three_substages(stub_classification_payload) -> None:
    """Avec un callback _stage fourni, classify_intent doit émettre au moins
    3 sous-stages couvrant : préparation contexte / appel LLM / réception."""
    from agents.master import classify_intent as ci

    stages_emitted: list[tuple[str, str]] = []
    def _stage(sid: str, label: str) -> None:
        stages_emitted.append((sid, label))

    with patch("openai.OpenAI") as _client_mock, \
         patch("agents.mortality.agents._utils.call_with_retry") as mock_call:
        mock_call.return_value = _fake_llm_response(stub_classification_payload)
        ci.classify_intent(
            "calcule l'exposition",
            has_data=True,
            has_calcs=False,
            _stage=_stage,
        )

    stage_ids = [s for s, _ in stages_emitted]
    assert "0.d.1" in stage_ids, f"Stage 0.d.1 manquant. Émis : {stage_ids}"
    assert "0.d.2" in stage_ids, f"Stage 0.d.2 manquant. Émis : {stage_ids}"
    assert "0.d.3" in stage_ids, f"Stage 0.d.3 manquant. Émis : {stage_ids}"


def test_classify_intent_substage_03_includes_kind(stub_classification_payload) -> None:
    """Le stage 0.d.3 doit mentionner l'intention reçue (kind) pour traçabilité."""
    from agents.master import classify_intent as ci

    stages_emitted: list[tuple[str, str]] = []
    def _stage(sid: str, label: str) -> None:
        stages_emitted.append((sid, label))

    with patch("openai.OpenAI") as _client_mock, \
         patch("agents.mortality.agents._utils.call_with_retry") as mock_call:
        mock_call.return_value = _fake_llm_response(stub_classification_payload)
        ci.classify_intent("calcule l'exposition", has_data=True, _stage=_stage)

    label_03 = next((lbl for sid, lbl in stages_emitted if sid == "0.d.3"), None)
    assert label_03 is not None
    assert "task" in label_03, f"0.d.3 doit mentionner kind='task'. Reçu : '{label_03}'"


def test_classify_intent_without_stage_callback_works(stub_classification_payload) -> None:
    """Rétro-compat : sans callback _stage, classify_intent doit fonctionner."""
    from agents.master import classify_intent as ci

    with patch("openai.OpenAI") as _client_mock, \
         patch("agents.mortality.agents._utils.call_with_retry") as mock_call:
        mock_call.return_value = _fake_llm_response(stub_classification_payload)
        result = ci.classify_intent("calcule l'exposition", has_data=True)

    assert result["kind"] == "task"
    assert result["write"] == "yes"


def test_classify_intent_no_stages_on_llm_error() -> None:
    """En cas d'erreur LLM, classify_intent retourne le fallback sans crash
    si _stage est fourni (peut émettre 0.d.1, 0.d.2, mais pas 0.d.3)."""
    from agents.master import classify_intent as ci

    stages_emitted: list[tuple[str, str]] = []
    def _stage(sid: str, label: str) -> None:
        stages_emitted.append((sid, label))

    with patch("openai.OpenAI") as _client_mock, \
         patch("agents.mortality.agents._utils.call_with_retry") as mock_call:
        mock_call.side_effect = RuntimeError("LLM down")
        result = ci.classify_intent("demande", has_data=False, _stage=_stage)

    assert result["intent"] == "unclear"
    # 0.d.3 ne doit PAS être émis (l'appel LLM a planté avant)
    stage_ids = [s for s, _ in stages_emitted]
    assert "0.d.3" not in stage_ids, (
        f"0.d.3 ne doit pas être émis si le LLM a planté. Émis : {stage_ids}"
    )
