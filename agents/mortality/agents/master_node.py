"""
agents/mortality/agents/master_node.py
Nœud MasterAgent du graphe LangGraph — hub d'orchestration.

Responsabilités :
  - Qualifier l'intention de l'utilisateur
  - Router vers BuilderAgent (<GO_BUILD>) ou WriterAgent (<GO_WRITE>)
  - Recevoir les signaux BUILD_DONE / WRITE_DONE / NEED_DATA des sous-agents
  - Orchestrer itérativement si des données manquent (WriterAgent → BuilderAgent → WriterAgent)

Signaux reconnus :
  Sortants (depuis MasterAgent vers le graphe) :
    <GO_BUILD>  → activer BuilderAgent
    <GO_WRITE>  → activer WriterAgent
    <ROUTE:MORTALITY> → alias de <GO_BUILD>
    <ROUTE:REPORT>    → alias de <GO_WRITE>

  Entrants (depuis les sous-agents, lu dans l'historique) :
    <BUILD_DONE>     → les calculs sont terminés, master peut router vers writer
    <WRITE_DONE>     → le rapport est généré
    <NEED_DATA: ...> → le WriterAgent a besoin de données supplémentaires du Builder
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


def _build_system_prompt() -> str:
    """Charge le system prompt du MasterAgent via loader.py."""
    loader_path = _PROJECT_ROOT / "loader.py"
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("loader", loader_path)
        mod  = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.get_system_prompt(level="full", agent_name="master")
    except Exception as exc:
        print(f"[MasterAgent] loader error: {exc}", file=sys.stderr)
        fallback = _PROJECT_ROOT / "agents" / "master" / "agent_instructions" / "behavioral_contract.md"
        return fallback.read_text(encoding="utf-8") if fallback.exists() else ""


def _augment_with_data_store(prompt: str, data_store: dict) -> str:
    """Ajoute un résumé du data_store au system prompt si des calculs ont déjà tourné."""
    if not data_store:
        return prompt

    summary_lines = ["\n\n## État du data_store (résultats déjà calculés)"]
    key_labels = {
        "exposure_table":  "✓ Exposition calculée",
        "qx_table":        "✓ Taux bruts calculés",
        "smoothed_table":  "✓ Table lissée",
        "diagnostics":     "✓ Diagnostics crédibilité",
        "validation":      "✓ Validation statistique",
        "benchmarking":    "✓ Benchmarking référence",
        "cox_regression":  "✓ Régression Cox H/F",
        "logit_regression":"✓ Régression logit",
        "series":          "✓ Séries temporelles",
        "section_outputs": "✓ Sections rapport en cours",
    }
    for key, label in key_labels.items():
        if data_store.get(key):
            summary_lines.append(f"  {label}")

    # Summary stats if available
    smr = None
    if isinstance(data_store.get("benchmarking"), dict):
        smr = data_store["benchmarking"].get("smr_global")
    elif "smr_global" in data_store:
        smr = data_store["smr_global"]
    if smr is not None:
        summary_lines.append(f"  SMR global : {smr:.3f}")

    return prompt + "\n".join(summary_lines)


def master_node(state: "AgentState") -> dict:
    """
    Nœud MasterAgent : orchestre BuilderAgent et WriterAgent.
    Retourne la mise à jour de l'état LangGraph.
    """
    import openai
    from agents.mortality.agents.mortality_node import _to_openai_dict, _from_openai_response

    data_store = state.get("data_store") or {}
    system_prompt = _augment_with_data_store(_build_system_prompt(), data_store)

    # MasterAgent n'a pas de tools — il orchestre via signaux textuels
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
        "agent":       "MasterAgent",
        "model":       "gpt-4o",
        "n_messages":  len(messages),
        "max_tokens":  1500,
        "has_tools":   False,
        "last_user":   str(last_user)[:400],
        "system_head": system_prompt[:300],
    })

    try:
        response = call_with_retry(
            client,
            model="gpt-4o",
            messages=messages,
            tools=None,
            max_tokens=1500,
        )
    except Exception as exc:
        new_events.append({"type": "error", "message": f"Erreur API OpenAI (MasterAgent) : {exc}"})
        return {"messages": [], "events": new_events}

    choice = response.choices[0]
    msg_obj = choice.message
    lc_msg  = _from_openai_response(msg_obj)

    # ── Event : réponse de l'API ──────────────────────────────────────────────
    usage = response.usage
    new_events.append({
        "type":               "llm_output",
        "agent":              "MasterAgent",
        "finish_reason":      choice.finish_reason,
        "content_preview":    (msg_obj.content or "")[:400],
        "prompt_tokens":      usage.prompt_tokens      if usage else None,
        "completion_tokens":  usage.completion_tokens  if usage else None,
        "total_tokens":       usage.total_tokens       if usage else None,
        "n_tool_calls":       0,
    })

    content = msg_obj.content or ""

    # Routing signal detection
    goes_builder = "<GO_BUILD>" in content or "<ROUTE:MORTALITY>" in content
    goes_writer  = "<GO_WRITE>" in content  or "<ROUTE:REPORT>" in content

    if content and not goes_builder and not goes_writer:
        # Pure conversational response (qualification questions or status)
        new_events.append({"type": "message", "content": content})

    new_state: dict = {"messages": [lc_msg], "events": new_events}

    if goes_builder:
        new_state["active_agent"] = "builder"
    elif goes_writer:
        new_state["active_agent"] = "writer"
    # else: stay "master" — but graph will route to END for user response

    return new_state
