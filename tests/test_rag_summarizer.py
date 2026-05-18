"""Tests du summarizer LLM nano (mocké)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def _mock_llm_json(payload: dict) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = json.dumps(payload)
    response.choices = [choice]
    return response


def _sample_old_turns():
    from agents.rag.memory.schemas import RAGTurn
    return [
        RAGTurn(user_q="qu'est-ce que Whittaker ?",
                rag_answer="Le lissage Whittaker-Henderson pénalise [D03.02].",
                sources=[]),
        RAGTurn(user_q="comment choisir h ?",
                rag_answer="Le paramètre h s'optimise par validation croisée [D03.04].",
                sources=[]),
    ]


def test_summarize_returns_pydantic_summary():
    from agents.rag.memory import summarizer
    fake_payload = {
        "topics_covered":     ["Whittaker-Henderson"],
        "user_focus":         "méthodes de lissage",
        "key_facts_stated":   ["h optimisé par CV"],
        "citations_used":     ["D03.02", "D03.04"],
        "n_turns_summarized": 2,
    }
    fake_resp = _mock_llm_json(fake_payload)
    with patch("agents.rag.memory.summarizer.openai.OpenAI"), \
         patch("agents.rag.memory.summarizer.call_with_retry", return_value=fake_resp):
        out = summarizer.summarize_old_turns(_sample_old_turns(), existing=None)
    assert out.topics_covered == ["Whittaker-Henderson"]
    assert out.user_focus == "méthodes de lissage"
    assert out.citations_used == ["D03.02", "D03.04"]
    assert out.n_turns_summarized == 2


def test_summarize_merges_existing_summary():
    """Si un summary existe, le LLM doit recevoir l'existant + nouveaux tours."""
    from agents.rag.memory import summarizer
    from agents.rag.memory.schemas import RAGSummary
    existing = RAGSummary(
        topics_covered=["TH 00-02"],
        user_focus="tables réglementaires",
        n_turns_summarized=4,
    )
    fake_payload = {
        "topics_covered":     ["TH 00-02", "Whittaker-Henderson"],
        "user_focus":         "tables et lissage",
        "key_facts_stated":   [],
        "citations_used":     [],
        "n_turns_summarized": 6,
    }
    fake_resp = _mock_llm_json(fake_payload)
    with patch("agents.rag.memory.summarizer.openai.OpenAI"), \
         patch("agents.rag.memory.summarizer.call_with_retry",
               return_value=fake_resp) as mock_call:
        out = summarizer.summarize_old_turns(_sample_old_turns(), existing=existing)
    assert "TH 00-02" in out.topics_covered
    assert out.n_turns_summarized == 6
    # Vérifier que l'existant a été envoyé dans le prompt
    messages = mock_call.call_args.kwargs["messages"]
    payload = "\n".join((m.get("content") or "") for m in messages)
    assert "TH 00-02" in payload  # vu l'existant


def test_summarize_falls_back_to_existing_on_llm_error():
    """Si le LLM échoue, on garde le summary existant (graceful)."""
    from agents.rag.memory import summarizer
    from agents.rag.memory.schemas import RAGSummary
    existing = RAGSummary(topics_covered=["déjà là"], n_turns_summarized=4)
    with patch("agents.rag.memory.summarizer.openai.OpenAI"), \
         patch("agents.rag.memory.summarizer.call_with_retry",
               side_effect=RuntimeError("openai 500")):
        out = summarizer.summarize_old_turns(_sample_old_turns(), existing=existing)
    assert out is existing


def test_summarize_returns_empty_summary_when_no_existing_and_llm_fails():
    from agents.rag.memory import summarizer
    with patch("agents.rag.memory.summarizer.openai.OpenAI"), \
         patch("agents.rag.memory.summarizer.call_with_retry",
               side_effect=RuntimeError("openai 500")):
        out = summarizer.summarize_old_turns(_sample_old_turns(), existing=None)
    from agents.rag.memory.schemas import RAGSummary
    assert isinstance(out, RAGSummary)
    assert out.topics_covered == []


def test_summarize_uses_nano_role_config():
    from agents.rag.memory import summarizer
    fake_resp = _mock_llm_json({
        "topics_covered": [], "user_focus": "", "key_facts_stated": [],
        "citations_used": [], "n_turns_summarized": 0,
    })
    with patch("agents.rag.memory.summarizer.openai.OpenAI"), \
         patch("agents.rag.memory.summarizer.call_with_retry", return_value=fake_resp), \
         patch("agents.rag.memory.summarizer.get_llm_config",
               return_value={"model": "gpt-5.4-nano", "temperature": 0.0,
                             "max_tokens": 500}) as mock_cfg:
        summarizer.summarize_old_turns(_sample_old_turns(), existing=None)
    mock_cfg.assert_called_with("rag.summarizer")
