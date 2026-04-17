"""
agents/mortality/agents/report_node.py
Nœud ReportAgent du graphe LangGraph.

Responsabilités :
  - Générer les rapports PDF (build_pdf.*)
  - Produire des graphiques (graphs.*)
  - Peut rappeler des tools de calcul (builder.*) si des données manquent
  - Utilise son propre system prompt dédié (agents/report/system_prompt.md)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import ToolMessage

if TYPE_CHECKING:
    from agents.mortality.agents.state import AgentState

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _build_system_prompt(state: "AgentState") -> str:
    """Charge le system prompt du ReportAgent via loader.py."""
    loader_path = _PROJECT_ROOT / "loader.py"
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("loader", loader_path)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.get_system_prompt(level="full", agent_name="report")
    except Exception as exc:
        print(f"[ReportAgent] loader error: {exc}", file=sys.stderr)
        fallback = _PROJECT_ROOT / "agents" / "report" / "agent_instructions" / "behavioral_contract.md"
        return fallback.read_text(encoding="utf-8") if fallback.exists() else ""


def report_node(state: "AgentState") -> dict:
    """
    Nœud ReportAgent : appelle le LLM pour la rédaction du rapport.
    Retourne la mise à jour de l'état LangGraph.
    """
    import openai
    from agents.mortality.agents.mortality_node import _to_openai_dict, _from_openai_response, sanitize_openai_messages

    system_prompt = _build_system_prompt(state)

    # Tous les tools disponibles pour la rédaction
    from tools.tool_registry import get_openai_tools
    tools = get_openai_tools()

    MAX_HISTORY = 20
    raw_msgs = state["messages"]
    if len(raw_msgs) > MAX_HISTORY:
        raw_msgs = raw_msgs[-MAX_HISTORY:]

    # Construire les messages
    messages = [{"role": "system", "content": system_prompt}]
    for msg in raw_msgs:
        messages.append(_to_openai_dict(msg))
    messages = sanitize_openai_messages(messages)

    from agents.mortality.agents._utils import call_with_retry
    client = openai.OpenAI()
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
        return {
            "messages": [],
            "events": [{"type": "error", "message": f"Erreur API OpenAI (ReportAgent) : {exc}"}],
        }

    choice = response.choices[0]
    msg = choice.message
    lc_msg = _from_openai_response(msg)
    new_events: list[dict] = []

    if choice.finish_reason != "tool_calls":
        content = msg.content or ""
        if content:
            new_events.append({"type": "message", "content": content})
        new_events.append({"type": "done"})

    return {
        "messages": [lc_msg],
        "events": new_events,
    }
