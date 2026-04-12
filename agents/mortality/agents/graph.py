"""
agents/mortality/agents/graph.py
Assemblage du graphe LangGraph + adaptateur stream_agent().

Architecture — hub master :
  START → router
            ├── "master"      → master_node → _should_continue_master?
            │                       ├── "to_builder"  → builder_node
            │                       ├── "to_writer"   → writer_node
            │                       └── "end"         → END
            ├── "builder"     → builder_node → _should_continue_builder?
            │                       ├── "tools"       → tools_builder → builder_node
            │                       ├── "to_master"   → master_node
            │                       └── "end"         → END
            └── "writer"      → writer_node → _should_continue_writer?
                                    ├── "tools"       → tools_writer → writer_node
                                    ├── "to_master"   → master_node
                                    └── "end"         → END

Compat. ascendante :
  active_agent == "calculation" → router vers "builder" (alias)
  active_agent == "writer"      → router vers "writer"

stream_agent() expose la même interface que l'ancienne API.
"""
from __future__ import annotations

import threading
from typing import Generator

import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, START, StateGraph

from agents.mortality.agents.state import AgentState
from agents.mortality.agents.mortality_node import mortality_node   # legacy compat
from agents.mortality.agents.report_node import report_node         # legacy compat
from agents.mortality.agents.master_node import master_node
from agents.mortality.agents.builder_node import builder_node
from agents.mortality.agents.writer_node import writer_node
from agents.mortality.agents.tools_node import execute_tools


# ── Routing ───────────────────────────────────────────────────────────────────

def _router(state: AgentState) -> str:
    """Route vers le bon nœud selon active_agent."""
    agent = state.get("active_agent", "calculation")
    if agent == "master":
        return "master"
    if agent in ("builder", "calculation"):
        return "builder"
    if agent == "writer":
        return "writer"
    return "builder"  # default


def _should_continue_master(state: AgentState) -> str:
    """Routing post-master_node."""
    agent = state.get("active_agent", "master")
    if agent == "builder":
        return "to_builder"
    if agent == "writer":
        return "to_writer"
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return "end"
    return "end"


def _should_continue_builder(state: AgentState) -> str:
    """Routing post-builder_node."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return "end"
    if getattr(last, "tool_calls", None):
        return "tools"
    content = last.content or ""
    # Retour au master si BUILD_DONE ou HANDOFF_WRITER
    if "<BUILD_DONE>" in content or "<HANDOFF_WRITER>" in content:
        return "to_master"
    if "<MODEL_CHOICE_CHECKPOINT>" in content:
        return "end"  # Soft pause
    # Si active_agent a été basculé vers master par builder_node
    if state.get("active_agent") == "master":
        return "to_master"
    return "end"


def _should_continue_writer(state: AgentState) -> str:
    """Routing post-writer_node."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return "end"
    if getattr(last, "tool_calls", None):
        return "tools"
    content = last.content or ""
    # Retour au master si WRITE_DONE ou NEED_DATA
    if "<WRITE_DONE>" in content or "<NEED_DATA" in content:
        return "to_master"
    if state.get("active_agent") == "master":
        return "to_master"
    return "end"


# ── Nœuds avec closures pour step-by-step ────────────────────────────────────

def _make_tools_node(
    approval_event: threading.Event | None,
    cancel_flag: list[bool] | None,
    active_agent_ref: list[str],
):
    """Fabrique le nœud tools avec les handles step-by-step injectés."""
    def tools_node(state: AgentState) -> dict:
        result = execute_tools(state, approval_event=approval_event, cancel_flag=cancel_flag)
        result["active_agent"] = active_agent_ref[0]
        return result
    return tools_node


# ── Construction du graphe ────────────────────────────────────────────────────

def build_graph(
    approval_event: threading.Event | None = None,
    cancel_flag: list[bool] | None = None,
) -> StateGraph:
    """Construit et compile le graphe LangGraph."""
    active_agent_ref = ["builder"]

    def _master_node_w(state: AgentState) -> dict:
        active_agent_ref[0] = "master"
        result = master_node(state)
        evs = result.setdefault("events", [])
        evs.insert(0, {"type": "agent_switch", "agent": "MasterAgent"})
        return result

    def _builder_node_w(state: AgentState) -> dict:
        active_agent_ref[0] = state.get("active_agent", "builder")
        result = builder_node(state)
        evs = result.setdefault("events", [])
        evs.insert(0, {"type": "agent_switch", "agent": "BuilderAgent"})
        return result

    def _writer_node_w(state: AgentState) -> dict:
        active_agent_ref[0] = "writer"
        result = writer_node(state)
        evs = result.setdefault("events", [])
        evs.insert(0, {"type": "agent_switch", "agent": "WriterAgent"})
        return result

    tools_node = _make_tools_node(approval_event, cancel_flag, active_agent_ref)

    def _set_active(agent: str):
        def _node(state: AgentState) -> dict:
            return {"active_agent": agent}
        return _node

    graph = StateGraph(AgentState)

    # Nœuds
    graph.add_node("router",         lambda s: {"active_agent": s.get("active_agent", "builder")})
    graph.add_node("master",         _master_node_w)
    graph.add_node("builder",        _builder_node_w)
    graph.add_node("writer",         _writer_node_w)
    graph.add_node("tools_builder",  tools_node)
    graph.add_node("tools_writer",   tools_node)
    graph.add_node("set_builder",    _set_active("builder"))
    graph.add_node("set_writer",     _set_active("writer"))

    # Entrée
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", _router, {
        "master":  "master",
        "builder": "builder",
        "writer":  "writer",
    })

    # Master → builder / writer / END
    graph.add_conditional_edges("master", _should_continue_master, {
        "to_builder": "set_builder",
        "to_writer":  "set_writer",
        "end":        END,
    })
    graph.add_edge("set_builder", "builder")
    graph.add_edge("set_writer",  "writer")

    # Builder → tools / master / END
    graph.add_conditional_edges("builder", _should_continue_builder, {
        "tools":     "tools_builder",
        "to_master": "master",
        "end":       END,
    })
    graph.add_edge("tools_builder", "builder")

    # Writer → tools / master / END
    graph.add_conditional_edges("writer", _should_continue_writer, {
        "tools":     "tools_writer",
        "to_master": "master",
        "end":       END,
    })
    graph.add_edge("tools_writer", "writer")

    return graph.compile()


# ── Adaptateur canvas ─────────────────────────────────────────────────────────

def stream_agent(
    history: list[dict],
    df: "pd.DataFrame | None" = None,
    data_store: dict | None = None,
    context_docs: list[dict] | None = None,
    step_by_step: bool = False,
    approval_event: threading.Event | None = None,
    cancel_flag: list[bool] | None = None,
    catalogue_level: str | None = None,
) -> Generator[dict, None, None]:
    """
    Adaptateur : expose la même interface que l'ancien WriterAgent.run_agent_loop().
    Génère des events dict canvas identiques à l'ancienne API.

    active_agent par défaut = "builder" (compat. avec l'ancien "calculation").
    Passer active_agent="master" dans data_store pour activer le hub master.
    """
    if data_store is None:
        data_store = {}

    df_json: str | None = None
    if df is not None:
        try:
            df_json = df.to_json(orient="split")
        except Exception:
            pass

    lc_messages = []
    for h in history:
        role = h.get("role", "user")
        content = h.get("content", "")
        if not content:
            continue
        if role in ("assistant", "assistant_rag"):
            lc_messages.append(AIMessage(content=str(content)))
        else:
            lc_messages.append(HumanMessage(content=str(content)))

    if catalogue_level == "full":
        plan_established = True
    elif catalogue_level == "middle":
        plan_established = False
    else:
        plan_established = bool(data_store.get("_call_log"))

    # Déterminer l'agent actif initial
    # Si data_store indique "master" (sessions avec hub), utiliser master.
    # Sinon, utiliser "builder" (compat. avec l'ancien pipeline linéaire).
    initial_active = data_store.pop("_initial_active_agent", None) or "builder"

    initial_state: AgentState = {
        "messages":          lc_messages,
        "df_json":           df_json,
        "data_store":        data_store,
        "context_docs":      context_docs or [],
        "plan_established":  plan_established,
        "active_agent":      initial_active,
        "events":            [],
        "step_by_step":      step_by_step,
        "pending_tool_call": None,
    }

    graph = build_graph(
        approval_event=approval_event,
        cancel_flag=cancel_flag,
    )

    done_yielded = False

    try:
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            for node_name, update in chunk.items():
                for ev in update.get("events", []):
                    if ev.get("type") == "done":
                        done_yielded = True
                    yield ev

                if "data_store" in update:
                    data_store.update(update["data_store"])

    except Exception as exc:
        import traceback
        yield {"type": "error", "message": str(exc)}
        yield {"type": "error", "message": traceback.format_exc()}

    if not done_yielded:
        yield {"type": "done"}
