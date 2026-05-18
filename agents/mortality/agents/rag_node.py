"""
agents/mortality/agents/rag_node.py
Nœud RAGAgent du graphe LangGraph (adapter mince).

Responsabilité unique : faire l'interface entre le state LangGraph et le
pipeline pur `agents.rag.pipeline.run_pipeline.run()`.

  - Pousse les stage_events dans `data_store["_stage_buffer"]` pour l'UI
    "internal agent"
  - Émet le signal `<RAG_DONE>` en fin d'AIMessage pour le routeur LangGraph
  - Force `active_agent="master"` dans le return (1 cycle = 1 réponse, retour
    immédiat au superviseur)
  - Émet les events `agent_switch` + `message` pour le canvas

Mirror exact du pattern utilisé par `report_node.py`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from agents.mortality.agents.state import AgentState

log = logging.getLogger(__name__)


def _run_rag_pipeline(state: dict, verify: bool = False) -> dict:
    """Indirection testable — patch ce nom dans les tests pour mocker
    le pipeline complet sans entrer dans les détails de chaque étape."""
    from agents.rag.pipeline.run_pipeline import run as run_rag
    return run_rag(state, verify=verify)


def rag_node(state: "AgentState") -> dict:
    """Adapter LangGraph : délègue à agents.rag.pipeline.run_pipeline."""
    data_store = dict(state.get("data_store") or {})

    try:
        result = _run_rag_pipeline(state, verify=False)
    except Exception as exc:
        log.exception("[rag_node] pipeline failure")
        return {
            "messages":     [AIMessage(content=f"Erreur agent RAG : {exc}\n\n<RAG_DONE>")],
            "events":       [{"type": "agent_switch", "agent": "RAGAgent"},
                             {"type": "error",        "message": str(exc)}],
            "active_agent": "master",
            "data_store":   data_store,
        }

    answer = result.get("answer", "")
    stage_events = result.get("stage_events", []) or []

    # ── Stage tracking : double cible ────────────────────────────────────
    # 1) data_store["_stage_buffer"] : conservé pour cohérence avec le
    #    pattern Master qui flush via _ret() en fin de tour.
    # 2) events directement : indispensable car au tour suivant le
    #    master_node initialise data_store["_stage_buffer"] = [] (ligne ~305),
    #    ce qui wiperait les stages RAG.* avant qu'ils ne soient émis vers
    #    le canvas. Pousser dans events garantit la visibilité immédiate UI.
    buf = list(data_store.get("_stage_buffer") or [])
    stage_event_dicts: list[dict] = []
    for stage_id, label in stage_events:
        ev = {"type": "master_stage", "stage": stage_id, "label": label}
        buf.append(ev)
        stage_event_dicts.append(ev)
    data_store["_stage_buffer"] = buf

    # ── Signal de routing + events canvas ────────────────────────────────
    content = f"{answer}\n\n<RAG_DONE>"
    events: list[dict] = [{"type": "agent_switch", "agent": "RAGAgent"}]
    events.extend(stage_event_dicts)
    events.append({"type": "message", "content": answer})

    return {
        "messages":     [AIMessage(content=content)],
        "events":       events,
        "active_agent": "master",
        "data_store":   data_store,
    }
