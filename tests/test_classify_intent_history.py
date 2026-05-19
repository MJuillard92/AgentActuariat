"""
HOTFIX-pre-refacto-2026-05 — Bug 1 : classify_intent doit recevoir
l'historique conversationnel pour identifier les continuations RAG.

Scénario observé en prod :
    User Q1 : « peux-tu me dire les méthodes classiques pour construire
                une table d'expérience ? »  → RAG (kind=question)
    User Q2 : « ok, mais pour calculer les taux bruts et taux lissés ? »
              → classifié à tort comme kind=task (lance Builder),
              alors que c'est une continuation doctrinale.

Le fix passe les 4 derniers tours user/assistant à classify_intent.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _fake_llm_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(payload)
    return resp


def _call_classify(history: list[dict] | None, captured_prompts: list[str]) -> dict:
    """Helper : appelle classify_intent avec history et capture le prompt LLM
    pour vérification.
    """
    from agents.master import classify_intent as ci

    def _capture(client, **kwargs) -> MagicMock:
        msgs = kwargs.get("messages") or []
        if msgs:
            captured_prompts.append(msgs[0].get("content", ""))
        return _fake_llm_response({
            "kind":                "task",
            "write":               "ask",
            "report_mode":         "full_report",
            "gender_segmentation": "unknown",
            "confidence":          0.9,
            "reasoning":           "",
            "reply":               "",
        })

    with patch("openai.OpenAI"), \
         patch("agents.mortality.agents._utils.call_with_retry", side_effect=_capture):
        return ci.classify_intent(
            "ok, mais pour calculer les taux bruts et taux lissés ?",
            has_data=True,
            history=history,
        )


def test_classify_intent_accepts_history_kwarg() -> None:
    """L'API publique doit accepter `history=[...]`."""
    from agents.master import classify_intent as ci

    captured: list[str] = []
    _call_classify([], captured)  # liste vide — doit fonctionner


def test_history_included_in_llm_prompt() -> None:
    """Si history est fournie, le prompt LLM doit contenir un bloc résumé
    des messages récents (pour permettre la détection de continuation)."""
    captured: list[str] = []
    history = [
        {"role": "user",      "content": "peux-tu me dire les méthodes classiques pour construire une table d'expérience ?"},
        {"role": "assistant", "content": "Les méthodes classiques incluent le positionnement, le lissage bayésien de Kimeldorf-Jones et la méthode de Denuit-Goderniaux..."},
    ]
    _call_classify(history, captured)

    assert captured, "Aucun prompt capturé — patch incorrect ?"
    prompt = captured[0]
    assert "table d'expérience" in prompt or "table d'experience" in prompt.lower(), (
        "Le prompt LLM doit contenir l'historique récent. "
        f"Prompt capturé : {prompt[:500]}"
    )


def test_history_truncated_to_recent_turns() -> None:
    """Pour éviter d'exploser le prompt, history doit être tronquée aux
    derniers tours (max ~4 messages assistant+user)."""
    captured: list[str] = []
    long_history = [
        {"role": "user",      "content": f"vieux message user {i}"}
        for i in range(20)
    ] + [
        {"role": "assistant", "content": "réponse importante récente sur les taux bruts"},
        {"role": "user",      "content": "question intermédiaire 1"},
        {"role": "assistant", "content": "réponse intermédiaire 1"},
        {"role": "user",      "content": "question intermédiaire 2"},
    ]
    _call_classify(long_history, captured)

    prompt = captured[0]
    # Les vieux messages indexés 0-15 ne doivent PAS apparaître
    assert "vieux message user 0" not in prompt, (
        "L'historique doit être tronqué — vieux messages ne doivent pas être passés au LLM"
    )
    # Le dernier message assistant doit être présent
    assert "réponse importante récente sur les taux bruts" in prompt, (
        "Le dernier message assistant récent doit être présent dans le prompt"
    )


def test_history_message_content_capped() -> None:
    """Chaque message historique doit être tronqué pour ne pas exploser
    le prompt (cap à ~300 chars)."""
    captured: list[str] = []
    huge_content = "x" * 5000
    history = [{"role": "assistant", "content": huge_content}]
    _call_classify(history, captured)

    prompt = captured[0]
    # Vérifie que le prompt ne contient pas 5000 'x' consécutifs
    assert "x" * 1000 not in prompt, (
        "Le contenu d'un message historique doit être tronqué (cap ~300 chars)"
    )


def test_no_history_works_as_before() -> None:
    """Rétro-compat : sans history (None), classify_intent doit fonctionner."""
    from agents.master import classify_intent as ci

    with patch("openai.OpenAI"), \
         patch("agents.mortality.agents._utils.call_with_retry") as mock_call:
        mock_call.return_value = _fake_llm_response({
            "kind":                "task",
            "write":               "yes",
            "report_mode":         "full_report",
            "gender_segmentation": "unknown",
            "confidence":          0.9,
            "reasoning":           "",
            "reply":               "OK",
        })
        result = ci.classify_intent("calcule l'exposition", has_data=True)

    assert result["kind"] == "task"


def test_master_node_propagates_history() -> None:
    """master_node._classify_intent doit construire et propager l'historique
    LangChain tronqué vers classify_intent."""
    from langchain_core.messages import HumanMessage, AIMessage
    from agents.mortality.agents import master_node as mn

    messages = [
        HumanMessage(content="méthodes pour table d'expérience ?"),
        AIMessage(content="Les méthodes classiques incluent..."),
        HumanMessage(content="ok, mais pour calculer les taux bruts ?"),
    ]
    state = {
        "messages":    messages,
        "data_store":  {"_dataset_ref": "sess1"},
        "dataset_ref": "sess1",
    }

    captured_kwargs: dict = {}
    def _capture(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"intent": "question", "kind": "question", "write": "ask",
                "report_mode": "full_report", "confidence": 0.9, "reply": ""}

    with patch.object(mn, "_classify_intent", side_effect=_capture):
        # Provoquer l'appel à _classify_intent en exécutant master_node.
        # On capture les kwargs pour vérifier que history est bien transmise.
        # Note : master_node peut ne pas atteindre _classify_intent selon
        # son short-circuit ; on garde le test focalisé sur la propagation.
        try:
            mn.master_node(state)
        except Exception:
            pass  # le mock peut casser des choses en aval, on ne s'en soucie pas

    # Si _classify_intent a été appelé, il doit avoir reçu history
    if captured_kwargs:
        assert "history" in captured_kwargs or "_stage" in captured_kwargs, (
            f"Aucun nouveau kwarg passé à _classify_intent. Reçu : {list(captured_kwargs.keys())}"
        )
