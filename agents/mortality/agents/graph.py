"""
agents/mortality/agents/graph.py
Graphe LangGraph multi-agents avec mémoire persistante (MemorySaver).

Architecture :
  router
    ├── "master"  → master_node  → _should_continue_master
    │                   ├── "to_builder" → builder_node
    │                   ├── "to_writer"  → writer_node
    │                   └── END
    ├── "builder" → builder_node → _should_continue_builder
    │                   ├── "tools"      → execute_tools → builder_node
    │                   ├── "to_master"  → master_node
    │                   └── END
    └── "writer"  → writer_node  → _should_continue_writer
                        ├── "tools"      → execute_tools → writer_node
                        ├── "to_master"  → master_node
                        └── END

Mémoire :
  MemorySaver checkpointer — l'état complet (messages + data_store) est
  persisté automatiquement par LangGraph entre les invocations.
  Le thread_id = session_id de la session canvas.

Audit :
  Chaque event émis est loggué dans sessions/{thread_id}_audit.json
  pour traçabilité humaine. Ce fichier n'est jamais lu par l'agent.
"""
from __future__ import annotations

import threading
from typing import Generator

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.mortality.agents.state import AgentState
from agents.mortality.agents.master_node import master_node
from agents.mortality.agents.builder_node import builder_node
from agents.mortality.agents.writer_node import writer_node
from agents.mortality.agents.tools_node import execute_tools


# ── Checkpointer global (partagé entre toutes les sessions) ──────────────────
_checkpointer = MemorySaver()


# ── Routing ───────────────────────────────────────────────────────────────────

def _router(state: AgentState) -> str:
    agent = state.get("active_agent", "master")
    if agent == "master":
        return "master"
    if agent in ("builder", "calculation"):
        return "builder"
    if agent == "writer":
        return "writer"
    return "master"


def _should_continue_master(state: AgentState) -> str:
    agent = state.get("active_agent", "master")
    if agent == "builder":
        return "to_builder"
    if agent == "writer":
        return "to_writer"
    msgs = state.get("messages") or []
    if not msgs:
        return END
    last = msgs[-1]
    if not isinstance(last, AIMessage):
        return END
    return END


def _should_continue_builder(state: AgentState) -> str:
    msgs = state.get("messages") or []
    if not msgs:
        return END
    last = msgs[-1]
    if not isinstance(last, AIMessage):
        return END

    # Priorité 1 : encore des tool_calls en attente
    if getattr(last, "tool_calls", None):
        return "tools"

    content = last.content or ""

    # Priorité 2 : signal explicite du Builder (legacy, conservé)
    if "<BUILD_DONE>" in content or "<HANDOFF_WRITER>" in content:
        return "to_master"
    if "<MODEL_CHOICE_CHECKPOINT>" in content:
        return END

    # Priorité 3 : vérification data_store — routing déterministe.
    # Si toutes les clés attendues pour le mode courant sont présentes, on rend
    # la main au Master sans attendre que le LLM Builder émette <BUILD_DONE>.
    data_store = state.get("data_store") or {}
    try:
        from agents.mortality.agents.master_node import _get_required_keys_for_current_mode
        expected = _get_required_keys_for_current_mode(data_store)
    except Exception:
        expected = []
    if expected and all(data_store.get(k) for k in expected):
        return "to_master"

    # Sécurité : si le Builder tourne trop longtemps sans produire les données,
    # revenir au Master pour éviter une boucle infinie.
    builder_turns = data_store.get("_builder_turns", 0)
    if builder_turns >= 5:
        return "to_master"

    if state.get("active_agent") == "master":
        return "to_master"
    return END


def _should_continue_writer(state: AgentState) -> str:
    msgs = state.get("messages") or []
    if not msgs:
        return END
    last = msgs[-1]
    if not isinstance(last, AIMessage):
        return END
    if getattr(last, "tool_calls", None):
        return "tools"
    content = last.content or ""
    if "<WRITE_DONE" in content or "<NEED_DATA" in content:
        return "to_master"
    if state.get("active_agent") == "master":
        return "to_master"
    return END


# ── Wrappers avec injection d'events agent_switch ────────────────────────────

def _master_node_w(state: AgentState) -> dict:
    result = master_node(state)
    result.setdefault("events", []).insert(0, {"type": "agent_switch", "agent": "MasterAgent"})
    return result


def _builder_node_w(state: AgentState) -> dict:
    result = builder_node(state)
    result.setdefault("events", []).insert(0, {"type": "agent_switch", "agent": "BuilderAgent"})
    return result


def _writer_node_w(state: AgentState) -> dict:
    result = writer_node(state)
    result.setdefault("events", []).insert(0, {"type": "agent_switch", "agent": "WriterAgent"})
    return result


def _tools_node_w(
    approval_event: threading.Event | None,
    cancel_flag: list | None,
):
    """Retourne un wrapper de execute_tools avec les flags step-by-step."""
    def _inner(state: AgentState) -> dict:
        return execute_tools(state, approval_event=approval_event, cancel_flag=cancel_flag)
    return _inner


# ── Construction du graphe ────────────────────────────────────────────────────

def build_graph(
    approval_event: threading.Event | None = None,
    cancel_flag: list | None = None,
):
    """
    Construit et compile le StateGraph LangGraph avec MemorySaver.
    Retourne un graphe compilé prêt à être invoqué avec un thread_id.
    """
    g = StateGraph(AgentState)

    g.add_node("master",  _master_node_w)
    g.add_node("builder", _builder_node_w)
    g.add_node("writer",  _writer_node_w)
    g.add_node("tools",   _tools_node_w(approval_event, cancel_flag))

    # Point d'entrée conditionnel
    g.set_conditional_entry_point(
        _router,
        {
            "master":  "master",
            "builder": "builder",
            "writer":  "writer",
        },
    )

    # Edges master
    g.add_conditional_edges(
        "master",
        _should_continue_master,
        {
            "to_builder": "builder",
            "to_writer":  "writer",
            END:          END,
        },
    )

    # Edges builder
    g.add_conditional_edges(
        "builder",
        _should_continue_builder,
        {
            "tools":      "tools",
            "to_master":  "master",
            END:          END,
        },
    )

    # Edges writer
    g.add_conditional_edges(
        "writer",
        _should_continue_writer,
        {
            "tools":     "tools",
            "to_master": "master",
            END:         END,
        },
    )

    # Tools retourne toujours vers le nœud qui l'a appelé
    # LangGraph ne supporte pas le routing dynamique depuis tools →
    # on utilise active_agent dans le state pour router
    g.add_conditional_edges(
        "tools",
        lambda s: s.get("active_agent", "builder"),
        {
            "builder": "builder",
            "writer":  "writer",
            "master":  "master",
        },
    )

    return g.compile(checkpointer=_checkpointer)


# ── Adaptateur canvas : stream_agent() ───────────────────────────────────────

def stream_agent(
    history: list,
    df: "pd.DataFrame | None" = None,
    data_store: dict | None = None,
    context_docs: list | None = None,
    step_by_step: bool = False,
    approval_event: threading.Event | None = None,
    cancel_flag: list | None = None,
    catalogue_level: str | None = None,
    thread_id: str | None = None,
) -> Generator[dict, None, None]:
    """
    Adaptateur canvas — source de vérité : MemoryManager (SessionState).

    Flux mémoire :
      1. MemoryManager.load()        → charge SessionState depuis disque
      2. mm.to_data_store()          → hydrate data_store initial
      3. graph.stream()              → LangGraph (MemorySaver RAM)
      4. mm.after_turn(data_store)   → persiste SessionState après le tour
    """
    from session.memory_manager import MemoryManager

    thread_id = thread_id or "default"

    # ── 1. Charger la business memory ────────────────────────────────────────
    mm = MemoryManager(thread_id)
    mm.load()

    # ── 2. Hydrater le data_store depuis le SessionState ─────────────────────
    # Priorité : data_store passé par canvas (contient les flags de la session
    # courante, ex. _initial_active_agent) > SessionState persisté
    persisted_ds = mm.to_data_store()
    if data_store:
        persisted_ds.update(data_store)   # les valeurs canvas écrasent si conflit
    data_store = persisted_ds

    # ── 3. Convertir l'historique Dash en messages LangChain ─────────────────
    lc_messages = []
    for h in history:
        role    = h.get("role", "user")
        content = h.get("content", "")
        if not content:
            continue
        if role in ("assistant", "assistant_rag"):
            lc_messages.append(AIMessage(content=str(content)))
        else:
            lc_messages.append(HumanMessage(content=str(content)))

    # ── 4. Compaction si historique trop long ────────────────────────────────
    lc_messages = mm.trim_messages(lc_messages)

    if catalogue_level == "full":
        plan_established = True
    elif catalogue_level == "middle":
        plan_established = False
    else:
        plan_established = bool(data_store.get("_call_log"))

    initial_active = data_store.pop("_initial_active_agent", None) or "master"

    # ── 5. Input LangGraph — dataset_ref remplace df_json ────────────────────
    # Le DataFrame brut n'entre plus dans l'état LangGraph.
    # Les nodes chargent via MemoryManager.load_dataframe(dataset_ref).
    # Si un df est passé (premier tour après upload), on l'enregistre ici.
    if df is not None and mm.state.dataset_meta is None:
        csv_filename = data_store.get("csv_filename")
        mm.register_dataset(df, csv_filename)

    input_state: dict = {
        "messages":          lc_messages,
        "dataset_ref":       thread_id if mm.state.dataset_meta else None,
        "data_store":        data_store,
        "context_docs":      context_docs or [],
        "plan_established":  plan_established,
        "active_agent":      initial_active,
        "events":            [],
        "step_by_step":      step_by_step,
        "pending_tool_call": None,
    }

    config = {"configurable": {"thread_id": thread_id}}
    graph  = build_graph(approval_event=approval_event, cancel_flag=cancel_flag)

    done_yielded = False
    final_data_store = data_store

    try:
        for chunk in graph.stream(input_state, config=config, stream_mode="updates"):
            for node_name, update in chunk.items():
                # Capturer le data_store mis à jour par les nodes
                if "data_store" in update:
                    final_data_store = update["data_store"]
                for ev in update.get("events") or []:
                    if ev.get("type") == "done":
                        done_yielded = True
                    yield ev

    except Exception as exc:
        import traceback
        yield {"type": "error", "message": str(exc)}
        yield {"type": "error", "message": traceback.format_exc()}

    # ── 6. Persister la business memory après le tour ─────────────────────────
    mm.after_turn(final_data_store, lc_messages)

    if not done_yielded:
        yield {"type": "done"}
