"""Tests E2E au niveau du graph LangGraph complet — multi-turn."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage


def _mock_llm(text: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = text
    response.choices = [choice]
    return response


def _sample_hits() -> dict:
    return {
        "query_used":  "x",
        "n_returned":  1,
        "results":     [{
            "chunk_id": "ch1", "doc_id": "D03", "section_id": "D03.02",
            "section_title": "Whittaker-Henderson 1D",
            "text": "Le lissage pénalise.", "score": 0.9,
        }],
    }


def test_graph_multiturn_three_questions_share_memory():
    """T1 Whittaker → la mémoire est dans le store partagé via session_id."""
    from agents.mortality.agents.graph import build_graph
    from agents.rag.memory.rag_memory_store import RAGMemoryStore
    RAGMemoryStore._cache.clear()

    def _classify_question(*args, **kwargs):
        return {"kind": "question", "write": "no", "report_mode": None,
                "confidence": 0.95}

    graph = build_graph()
    config = {"configurable": {"thread_id": "test_e2e_mt_001"}, "recursion_limit": 25}
    state_t1 = {
        "messages":     [HumanMessage(content="c'est quoi Whittaker-Henderson ?")],
        "data_store":   {"_session_id": "test_e2e_mt_001",
                          "_history":    [HumanMessage(content="c'est quoi Whittaker-Henderson ?")]},
        "active_agent": "master",
        "events":       [],
        "plan_established": False,
    }
    with patch("agents.master.classify_intent.classify_intent",
               side_effect=_classify_question), \
         patch("tools.conversation.search_doctrine.run",
               return_value=_sample_hits()), \
         patch("agents.rag.pipeline.answer_generator.openai.OpenAI"), \
         patch("agents.rag.pipeline.answer_generator.call_with_retry",
               return_value=_mock_llm("Whittaker lisse [D03.02].")):
        list(graph.stream(state_t1, config=config, stream_mode="updates"))

    # Le store doit avoir 1 tour
    store = RAGMemoryStore._cache["test_e2e_mt_001"]
    assert len(store.get_buffer()) == 1
    assert "Whittaker" in store.get_buffer()[0].user_q


def test_graph_multiturn_does_not_loop_with_rag_done():
    """Régression : le short-circuit <RAG_DONE> dans master_node évite
    la boucle Master ↔ RAG (déjà testé pour single-turn — re-validation
    avec multi-turn actif)."""
    from agents.mortality.agents.graph import build_graph
    from agents.rag.memory.rag_memory_store import RAGMemoryStore
    RAGMemoryStore._cache.clear()

    def _classify_question(*args, **kwargs):
        return {"kind": "question", "write": "no", "report_mode": None,
                "confidence": 0.95}

    graph = build_graph()
    config = {"configurable": {"thread_id": "test_e2e_loop_001"}, "recursion_limit": 25}
    state = {
        "messages":     [HumanMessage(content="qu'est-ce que Whittaker ?")],
        "data_store":   {"_session_id": "test_e2e_loop_001",
                          "_history":    [HumanMessage(content="qu'est-ce que Whittaker ?")]},
        "active_agent": "master",
        "events":       [],
        "plan_established": False,
    }
    nodes_visited: list[str] = []
    with patch("agents.master.classify_intent.classify_intent",
               side_effect=_classify_question), \
         patch("tools.conversation.search_doctrine.run",
               return_value=_sample_hits()), \
         patch("agents.rag.pipeline.answer_generator.openai.OpenAI"), \
         patch("agents.rag.pipeline.answer_generator.call_with_retry",
               return_value=_mock_llm("Réponse [D03.02].")):
        for chunk in graph.stream(state, config=config, stream_mode="updates"):
            nodes_visited.extend(chunk.keys())
    # Doit s'arrêter en ≤ 5 nodes (master → rag → master → END)
    assert len(nodes_visited) <= 5
    assert "rag" in nodes_visited


def test_graph_multiturn_stage_events_include_rag_0_and_7():
    """Vérifie que les stages RAG.0 (hydrate) et RAG.7 (append) sont
    bien émis dans le stream events (visibilité UI internal agent)."""
    from agents.mortality.agents.graph import build_graph
    from agents.rag.memory.rag_memory_store import RAGMemoryStore
    RAGMemoryStore._cache.clear()

    def _classify_question(*args, **kwargs):
        return {"kind": "question", "write": "no", "report_mode": None,
                "confidence": 0.95}

    graph = build_graph()
    config = {"configurable": {"thread_id": "test_e2e_stages_001"}, "recursion_limit": 25}
    state = {
        "messages":     [HumanMessage(content="explique-moi Whittaker-Henderson ?")],
        "data_store":   {"_session_id": "test_e2e_stages_001",
                          "_history":    [HumanMessage(content="explique-moi Whittaker-Henderson ?")]},
        "active_agent": "master",
        "events":       [],
        "plan_established": False,
    }
    all_events: list[dict] = []
    with patch("agents.master.classify_intent.classify_intent",
               side_effect=_classify_question), \
         patch("tools.conversation.search_doctrine.run",
               return_value=_sample_hits()), \
         patch("agents.rag.pipeline.answer_generator.openai.OpenAI"), \
         patch("agents.rag.pipeline.answer_generator.call_with_retry",
               return_value=_mock_llm("Whittaker [D03.02].")):
        for chunk in graph.stream(state, config=config, stream_mode="updates"):
            for update in chunk.values():
                all_events.extend(update.get("events") or [])
    rag_stages = {e.get("stage") for e in all_events
                  if e.get("type") == "master_stage"
                  and (e.get("stage") or "").startswith("RAG.")}
    assert "RAG.0" in rag_stages
    assert "RAG.7" in rag_stages
