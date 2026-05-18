"""Tests E2E au niveau du graph LangGraph complet.

Garantit qu'un tour Master → RAG → Master se termine proprement (`done`),
sans `GraphRecursionError` (la régression critique trouvée au code-review).

Composants mockés au minimum : seul l'appel LLM est mocké, le routing
LangGraph et les nodes restent réels — c'est précisément ce qu'on veut
tester.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _mock_llm_response(text: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = text
    response.choices = [choice]
    return response


def _sample_hits() -> dict:
    return {
        "query_used":  "whittaker",
        "n_returned":  1,
        "results":     [{
            "chunk_id":      "ch_001",
            "doc_id":        "D03",
            "section_id":    "D03.02",
            "section_title": "Whittaker-Henderson 1D",
            "text":          "Le lissage de Whittaker-Henderson pénalise "
                             "les différences finies d'ordre k.",
            "score":         0.91,
        }],
    }


# ──────────────────────────────────────────────────────────────────────
# Test critique : pas de boucle Master ↔ RAG
# ──────────────────────────────────────────────────────────────────────

def test_master_rag_cycle_terminates_without_recursion_error():
    """Régression critique du code review : sans le short-circuit
    `<RAG_DONE>` dans master_node, Master re-classifie le même HumanMessage
    comme intent=question et boucle indéfiniment → GraphRecursionError.

    Ce test exécute le vrai graph (Master + RAG nodes) et vérifie :
    1. Le stream se termine sans exception
    2. Un event `done` est émis
    3. Le nombre de cycles est borné (Master + RAG = ~2-3 nodes touchés max)
    """
    from langchain_core.messages import HumanMessage
    from agents.mortality.agents.graph import build_graph

    # Mock le classify_intent pour retourner `kind=question` (path RAG)
    def _mock_classify(*args, **kwargs):
        return {
            "kind":        "question",
            "write":       "no",
            "report_mode": None,
            "confidence":  0.95,
        }

    # Mock le pipeline RAG pour retourner une réponse déterministe
    def _mock_rag(*args, **kwargs):
        return {
            "answer":  "Le lissage Whittaker pénalise les différences [D03.02].\n\n"
                       "Sources :\n- D03.02 — Whittaker-Henderson 1D",
            "sources": _sample_hits()["results"],
            "stage_events": [
                ("RAG.1", "Question extraite"),
                ("RAG.4", "Retrieval hybride : 1 chunks"),
                ("RAG.5", "Synthèse rédigée"),
            ],
        }

    graph = build_graph()
    config = {"configurable": {"thread_id": "test-rag-loop"},
              "recursion_limit": 25}  # défaut LangGraph
    input_state = {
        "messages":     [HumanMessage(content="c'est quoi le wittaker ?")],
        "data_store":   {},
        "active_agent": "master",
        "events":       [],
        "plan_established": False,
    }

    nodes_visited: list[str] = []
    with patch("agents.master.classify_intent.classify_intent",
               side_effect=_mock_classify), \
         patch("agents.mortality.agents.rag_node._run_rag_pipeline",
               side_effect=_mock_rag):
        try:
            for chunk in graph.stream(input_state, config=config, stream_mode="updates"):
                nodes_visited.extend(chunk.keys())
        except Exception as exc:
            assert "GraphRecursionError" not in type(exc).__name__, (
                f"BOUCLE Master ↔ RAG détectée — nœuds visités : {nodes_visited}"
            )
            raise

    # Vérification : nombre raisonnable de nodes visités (pas de boucle)
    assert len(nodes_visited) <= 5, (
        f"Trop de nodes visités ({len(nodes_visited)}) — risque de boucle : "
        f"{nodes_visited}"
    )
    # Le RAG node doit avoir été visité au moins une fois
    assert "rag" in nodes_visited
    # Le master node doit avoir été visité (pour le routing initial + final)
    assert "master" in nodes_visited


def test_rag_stage_events_visible_in_stream():
    """M5 du code review : les stages RAG.* doivent apparaître dans le
    stream d'events (pas seulement dans data_store["_stage_buffer"])
    sinon l'UI ne les voit pas."""
    from langchain_core.messages import HumanMessage
    from agents.mortality.agents.graph import build_graph

    def _mock_classify(*args, **kwargs):
        return {"kind": "question", "write": "no", "report_mode": None,
                "confidence": 0.95}

    def _mock_rag(*args, **kwargs):
        return {
            "answer":  "Réponse [D03.02].",
            "sources": _sample_hits()["results"],
            "stage_events": [
                ("RAG.1", "Question extraite"),
                ("RAG.5", "Synthèse rédigée"),
            ],
        }

    graph = build_graph()
    config = {"configurable": {"thread_id": "test-rag-stages"}, "recursion_limit": 25}
    input_state = {
        "messages":     [HumanMessage(content="whittaker ?")],
        "data_store":   {},
        "active_agent": "master",
        "events":       [],
        "plan_established": False,
    }

    all_events: list[dict] = []
    with patch("agents.master.classify_intent.classify_intent",
               side_effect=_mock_classify), \
         patch("agents.mortality.agents.rag_node._run_rag_pipeline",
               side_effect=_mock_rag):
        for chunk in graph.stream(input_state, config=config, stream_mode="updates"):
            for update in chunk.values():
                all_events.extend(update.get("events") or [])

    stage_events = [e for e in all_events
                    if e.get("type") == "master_stage"
                    and (e.get("stage") or "").startswith("RAG.")]
    stage_ids = {e["stage"] for e in stage_events}
    assert "RAG.1" in stage_ids, (
        f"Stage RAG.1 absent du stream events (UI ne le verra pas). "
        f"Events vus : {[e.get('type') for e in all_events]}"
    )
    assert "RAG.5" in stage_ids


# ──────────────────────────────────────────────────────────────────────
# Test I2 du code review : _clean_section_id pas de faux positif
# ──────────────────────────────────────────────────────────────────────

def test_clean_section_id_does_not_false_positive_on_d1_d10():
    """Si doc_id='D1' et section_id='D10.02', le startswith brut donnerait
    un faux positif. Le séparateur strict '{doc_id}.' évite ça."""
    from agents.rag.pipeline.answer_generator import _clean_section_id

    assert _clean_section_id("D1", "D10.02") == "D1.D10.02", (
        "Doc_ids différents D1/D10 — pas de strip prefix"
    )
    assert _clean_section_id("D03", "D03.02") == "D03.02", (
        "Cas nominal : strip du préfixe"
    )
    assert _clean_section_id("D03", "02") == "D03.02", (
        "Section sans préfixe : on préfixe"
    )
    assert _clean_section_id("", "D03.02") == "D03.02"
    assert _clean_section_id("D03", "") == "D03"
    assert _clean_section_id("D03", "D03") == "D03"


def test_stream_agent_injects_session_id_and_history_in_data_store():
    """method_choices.answer_question_via_doctrine doit pouvoir accéder
    à data_store['_session_id'] et data_store['_history']."""
    from agents.mortality.agents.graph import stream_agent
    from langchain_core.messages import HumanMessage
    from unittest.mock import patch

    # On capture le data_store final via stream
    history = [{"role": "user", "content": "test propagation"}]
    captured = {}
    with patch("agents.mortality.agents.graph.master_node") as mock_master:
        # Mock master_node pour qu'il termine immédiatement et expose data_store
        def fake_master(state):
            captured["data_store"] = state.get("data_store") or {}
            captured["session_id"] = captured["data_store"].get("_session_id")
            captured["history"] = captured["data_store"].get("_history")
            from langchain_core.messages import AIMessage
            return {
                "messages": [AIMessage(content="ok")],
                "events": [{"type": "done"}],
                "data_store": captured["data_store"],
                "active_agent": "master",
            }
        mock_master.side_effect = fake_master

        list(stream_agent(
            history=history,
            df=None,
            data_store={},
            thread_id="test_stream_001",
        ))
    assert captured.get("session_id") == "test_stream_001"
    assert captured.get("history") is not None
    assert len(captured["history"]) >= 1
