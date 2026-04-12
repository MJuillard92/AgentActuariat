"""
agents/mortality/agents/mortality_node.py
Nœud MortalityAgent du graphe LangGraph.

Responsabilités :
  - Orchestrer les tools actuariels (builder, statistical_analysis, graphs, reasoning)
  - Choisir le niveau de catalogue selon l'état (MIDDLE / FULL / LIGHT)
  - Signaler <HANDOFF_WRITER> quand les calculs sont terminés et un rapport est demandé

Niveau de catalogue :
  MIDDLE : 1er message user, plan non établi → qualification
  FULL   : message user avec plan existant   → replanification
  LIGHT  : dernier message = tool result     → exécution
"""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from langchain_core.messages import ToolMessage

if TYPE_CHECKING:
    from agents.mortality.agents.state import AgentState

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Tools accessibles au MortalityAgent
# build_pdf inclus : le handoff <HANDOFF_WRITER> est optionnel pour les sessions
# interactives complexes, mais le pipeline linéaire nécessite l'accès direct.
CALC_TOOLS = {"builder", "statistical_analysis", "graphs", "reasoning", "build_pdf"}


def _get_catalogue_level(state: "AgentState") -> str:
    """Détermine le niveau de catalogue selon l'état de la conversation."""
    messages = state.get("messages", [])
    if not messages:
        return "middle"
    last = messages[-1]
    if isinstance(last, ToolMessage):
        return "light"
    if not state.get("plan_established", False):
        return "middle"
    return "full"


def _build_system_prompt(state: "AgentState", level: str) -> str:
    """Charge le system prompt au niveau demandé via loader.py."""
    loader_path = _PROJECT_ROOT / "loader.py"
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("loader", loader_path)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        base = mod.get_system_prompt(level=level, agent_name="mortality")
    except Exception as exc:
        print(f"[MortalityAgent] loader error: {exc}", file=sys.stderr)
        fallback = _PROJECT_ROOT / "agents" / "mortality" / "agent_instructions" / "behavioral_contract.md"
        base = fallback.read_text(encoding="utf-8") if fallback.exists() else ""

    # Ajouter le mapping colonnes si df disponible
    df_json = state.get("df_json")
    if df_json:
        try:
            df = pd.read_json(StringIO(df_json), orient="split")
            from tools.tool_registry import get_capabilities
            from agents.mortality.dictionary.column_schema import build_mapping_report, COLUMN_SCHEMA
            caps = get_capabilities()
            report = build_mapping_report(df, caps)
            base += f"\n\n## Données : {len(df):,} lignes, {len(df.columns)} colonnes\n\n"
            base += "| Rôle | Colonne | Statut |\n|---|---|---|\n"
            for role, info in COLUMN_SCHEMA.items():
                if role in report["matched"]:
                    base += f"| {info['label']} | `{report['matched'][role]}` | ✓ |\n"
                else:
                    base += f"| {info['label']} | — | ❌ |\n"
        except Exception:
            pass

    # Documents de contexte
    context_docs = state.get("context_docs") or []
    if context_docs:
        base += "\n\n## Documents de contexte\n\n"
        for doc in context_docs:
            base += f"### {doc['name']}\n\n```\n{doc['content']}\n```\n\n"

    return base


def mortality_node(state: "AgentState") -> dict:
    """
    Nœud MortalityAgent : appelle le LLM avec le catalogue au bon niveau.
    Retourne la mise à jour de l'état LangGraph.
    """
    import openai

    level = _get_catalogue_level(state)
    system_prompt = _build_system_prompt(state, level)

    # Filtrer les tools du MortalityAgent
    from tools.tool_registry import get_openai_tools
    all_tools = get_openai_tools()
    tools = [t for t in all_tools if t["function"]["name"] in CALC_TOOLS]

    # Construire les messages
    MAX_HISTORY = 20
    raw_msgs = state["messages"]
    if len(raw_msgs) > MAX_HISTORY:
        raw_msgs = raw_msgs[-MAX_HISTORY:]

    messages = [{"role": "system", "content": system_prompt}]
    for msg in raw_msgs:
        messages.append(_to_openai_dict(msg))

    from agents.mortality.agents._utils import call_with_retry
    client = openai.OpenAI()
    new_events: list[dict] = []

    # ── Event : données envoyées à l'API ─────────────────────────────────────
    last_user = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    new_events.append({
        "type":        "llm_input",
        "agent":       "MortalityAgent",
        "model":       "gpt-4o",
        "n_messages":  len(messages),
        "max_tokens":  4000,
        "has_tools":   bool(tools),
        "last_user":   str(last_user)[:400],
        "system_head": system_prompt[:300],
    })

    try:
        response = call_with_retry(
            client,
            model="gpt-4o",
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            max_tokens=4000,
        )
    except Exception as exc:
        new_events.append({"type": "error", "message": f"Erreur API OpenAI : {exc}"})
        return {"messages": [], "events": new_events}

    choice = response.choices[0]
    msg = choice.message

    lc_msg = _from_openai_response(msg)

    # ── Event : réponse de l'API ──────────────────────────────────────────────
    usage = response.usage
    new_events.append({
        "type":               "llm_output",
        "agent":              "MortalityAgent",
        "finish_reason":      choice.finish_reason,
        "content_preview":    (msg.content or "")[:400],
        "prompt_tokens":      usage.prompt_tokens      if usage else None,
        "completion_tokens":  usage.completion_tokens  if usage else None,
        "total_tokens":       usage.total_tokens       if usage else None,
        "n_tool_calls":       len(msg.tool_calls or []),
    })

    if choice.finish_reason != "tool_calls":
        content = msg.content or ""
        if content:
            new_events.append({"type": "message", "content": content})
            state.get("data_store", {}).setdefault("_reasoning_log", []).append(content)
        new_events.append({"type": "done"})

    return {
        "messages": [lc_msg],
        "events": new_events,
    }


# ── Helpers format OpenAI ↔ LangChain ────────────────────────────────────────

def _to_openai_dict(msg) -> dict:
    """Convertit un message LangChain en dict OpenAI."""
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    if isinstance(msg, ToolMessage):
        return {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content}
    if isinstance(msg, AIMessage):
        d: dict = {"role": "assistant"}
        if msg.content:
            d["content"] = msg.content
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]) if isinstance(tc["args"], dict) else tc["args"],
                    },
                }
                for tc in msg.tool_calls
            ]
        return d
    return {"role": "user", "content": str(msg.content)}


def _from_openai_response(msg) -> object:
    """Convertit une réponse OpenAI en message LangChain."""
    from langchain_core.messages import AIMessage
    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "args": args,
                "type": "tool_call",
            })
    return AIMessage(
        content=msg.content or "",
        tool_calls=tool_calls,
    )
