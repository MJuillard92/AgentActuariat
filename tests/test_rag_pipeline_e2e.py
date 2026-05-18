"""Tests E2E du pipeline RAG (orchestrateur run_pipeline.run()).

Retriever + LLMs mockés — on vérifie le flux complet : extraction query,
normalisation, rewriting conditionnel, retrieval, génération.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage


def _mock_llm_response(text: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = text
    response.choices = [choice]
    return response


def _sample_hits() -> dict:
    return {
        "query_used":  "whittaker henderson",
        "n_returned":  2,
        "results":     [
            {
                "chunk_id":      "ch_001",
                "doc_id":        "D03",
                "section_id":    "D03.02",
                "section_title": "Whittaker-Henderson 1D",
                "text":          "Le lissage de Whittaker-Henderson pénalise "
                                 "les différences finies.",
                "score":         0.91,
            },
            {
                "chunk_id":      "ch_002",
                "doc_id":        "D03",
                "section_id":    "D03.04",
                "section_title": "Sélection du paramètre h",
                "text":          "Le paramètre h s'optimise par validation croisée.",
                "score":         0.87,
            },
        ],
    }


_OLD_TEST_COUNTER = 0


def _state_with_question(text: str) -> dict:
    global _OLD_TEST_COUNTER
    _OLD_TEST_COUNTER += 1
    return {
        "messages": [HumanMessage(content=text)],
        "data_store": {},
        "session_id": f"test_old_{_OLD_TEST_COUNTER:03d}",
    }


# ──────────────────────────────────────────────────────────────────────
# Flux nominal
# ──────────────────────────────────────────────────────────────────────

def test_run_returns_answer_sources_and_stage_events():
    from agents.rag.pipeline.run_pipeline import run

    state = _state_with_question("c'est quoi le wittaker ?")
    fake_answer = _mock_llm_response(
        "Le lissage de Whittaker pénalise les différences finies [D03.02].\n\n"
        "Sources :\n- D03.02 — Whittaker-Henderson 1D"
    )
    with patch("tools.conversation.search_doctrine.run",
               return_value=_sample_hits()), \
         patch("agents.rag.pipeline.answer_generator.openai.OpenAI"), \
         patch("agents.rag.pipeline.answer_generator.call_with_retry",
               return_value=fake_answer), \
         patch("agents.rag.pipeline.run_pipeline.is_in_scope", return_value=True):
        result = run(state)

    assert "answer" in result
    assert "sources" in result
    assert "stage_events" in result
    assert "[D03.02]" in result["answer"]
    assert len(result["sources"]) == 2
    # Stage events : extract, normalize, retrieve, generate au minimum (4)
    # + rewrite si query éligible (la query "c'est quoi le wittaker ?" est éligible)
    stage_ids = [s[0] for s in result["stage_events"]]
    assert "RAG.1" in stage_ids  # extract
    assert "RAG.2" in stage_ids  # normalize
    assert "RAG.4" in stage_ids  # retrieve
    assert "RAG.5" in stage_ids  # generate


def test_run_normalizes_typo_before_retrieval():
    """La query envoyée au retriever doit avoir été normalisée (wittaker → whittaker)."""
    from agents.rag.pipeline.run_pipeline import run

    state = _state_with_question("c'est quoi le wittaker ?")
    fake_answer = _mock_llm_response("ok")
    with patch("tools.conversation.search_doctrine.run",
               return_value=_sample_hits()) as mock_search, \
         patch("agents.rag.pipeline.answer_generator.openai.OpenAI"), \
         patch("agents.rag.pipeline.answer_generator.call_with_retry",
               return_value=fake_answer), \
         patch("agents.rag.pipeline.query_rewriter.rewrite",
               side_effect=lambda q, **kw: q), \
         patch("agents.rag.pipeline.run_pipeline.is_in_scope", return_value=True):  # no rewrite (pour isoler le test)
        run(state)

    # Le query passé à search_doctrine doit contenir "whittaker" (forme canonique)
    sent_query = mock_search.call_args.args[1]["query"] if len(mock_search.call_args.args) > 1 \
        else mock_search.call_args.kwargs["params"]["query"]
    assert "whittaker" in sent_query.lower()


def test_run_skips_rewrite_for_short_technical_query():
    """Query courte + terme technique → pas d'appel rewriter."""
    from agents.rag.pipeline.run_pipeline import run

    state = _state_with_question("whittaker-henderson")  # 19 chars + technique
    fake_answer = _mock_llm_response("ok")
    with patch("tools.conversation.search_doctrine.run",
               return_value=_sample_hits()), \
         patch("agents.rag.pipeline.answer_generator.openai.OpenAI"), \
         patch("agents.rag.pipeline.answer_generator.call_with_retry",
               return_value=fake_answer), \
         patch("agents.rag.pipeline.query_rewriter.rewrite") as mock_rewrite:
        result = run(state)

    mock_rewrite.assert_not_called()
    # Stage RAG.3 doit apparaître mais en mode "Skip" (skip, pas de reformulation LLM)
    stage_events_dict = {s[0]: s[1] for s in result["stage_events"]}
    assert "RAG.3" in stage_events_dict
    assert "Skip" in stage_events_dict["RAG.3"]


# ──────────────────────────────────────────────────────────────────────
# Cas dégénérés
# ──────────────────────────────────────────────────────────────────────

def test_run_handles_no_human_message():
    """Si aucun HumanMessage dans state, on retourne une réponse vide propre."""
    from agents.rag.pipeline.run_pipeline import run

    state = {"messages": [AIMessage(content="hello")], "data_store": {}}
    result = run(state)

    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)
    assert isinstance(result["stage_events"], list)


def test_run_handles_zero_chunks_retrieved():
    """Retriever retourne 0 chunk → réponse 'corpus ne couvre pas' sans appel LLM."""
    from agents.rag.pipeline.run_pipeline import run

    state = _state_with_question("question hors corpus exotique blabla")
    empty_hits = {"query_used": "x", "n_returned": 0, "results": []}
    with patch("tools.conversation.search_doctrine.run", return_value=empty_hits), \
         patch("agents.rag.pipeline.answer_generator.call_with_retry") as mock_llm, \
         patch("agents.rag.pipeline.query_rewriter.call_with_retry",
               return_value=_mock_llm_response("ok")), \
         patch("agents.rag.pipeline.run_pipeline.is_in_scope", return_value=True):
        result = run(state)

    assert "corpus" in result["answer"].lower()
    # Aucun appel au LLM answer_generator (économie)
    mock_llm.assert_not_called()


def test_run_with_verify_true_runs_grounding_check():
    """Avec verify=True, un stage RAG.6 doit apparaître."""
    from agents.rag.pipeline.run_pipeline import run

    state = _state_with_question("whittaker-henderson")
    fake_answer = _mock_llm_response(
        "Le lissage Whittaker pénalise [D03.02].\n\nSources :\n- D03.02 — WH 1D"
    )
    with patch("tools.conversation.search_doctrine.run",
               return_value=_sample_hits()), \
         patch("agents.rag.pipeline.answer_generator.openai.OpenAI"), \
         patch("agents.rag.pipeline.answer_generator.call_with_retry",
               return_value=fake_answer), \
         patch("agents.rag.pipeline.grounding_check.openai.OpenAI"), \
         patch("agents.rag.pipeline.grounding_check.call_with_retry",
               return_value=_mock_llm_response("OK")):
        result = run(state, verify=True)

    stage_ids = [s[0] for s in result["stage_events"]]
    assert "RAG.6" in stage_ids
