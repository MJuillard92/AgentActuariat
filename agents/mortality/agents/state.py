"""
agents/mortality/agents/state.py
État partagé entre les nœuds du graphe LangGraph.

Utilise LangGraph 1.x — Python 3.10+.
La mémoire de l'agent est gérée par MemorySaver (checkpointer) via thread_id.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────────────────
    # add_messages = reducer : accumule les messages sans écraser
    messages: Annotated[list[AnyMessage], add_messages]

    # ── Données portefeuille ──────────────────────────────────────────────────
    # df_json supprimé — le DataFrame est persisté une seule fois en Parquet
    # via session.DatasetStore ; les nodes le chargent via MemoryManager.load_dataframe()
    dataset_ref: Optional[str]   # session_id — clé de lookup vers l'artefact Parquet
    data_store: Dict[str, Any]   # résultats accumulés des tool calls
    context_docs: List[Any]      # documents uploadés par l'utilisateur

    # ── État de la session ────────────────────────────────────────────────────
    plan_established: bool
    active_agent: str            # "master" | "builder" | "writer"

    # ── Interface canvas (events streaming) ───────────────────────────────────
    events: List[Any]
    step_by_step: bool
    pending_tool_call: Optional[Dict[str, Any]]
