"""
report_agent/agents/state.py
État partagé entre les nœuds du graphe LangGraph.
"""
from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────────────────
    messages: Annotated[list[AnyMessage], add_messages]

    # ── Données portefeuille ──────────────────────────────────────────────────
    df_json: str | None          # DataFrame sérialisé (orient="split")
    data_store: dict             # résultats accumulés des tool calls
    context_docs: list[dict]     # documents uploadés par l'utilisateur

    # ── État de la session ────────────────────────────────────────────────────
    plan_established: bool       # True dès le premier tool call exécuté
    active_agent: str            # "calculation" | "writer"

    # ── Interface canvas (events streaming) ───────────────────────────────────
    events: list[dict]           # événements à consommer par le canvas
    step_by_step: bool
    pending_tool_call: dict | None
