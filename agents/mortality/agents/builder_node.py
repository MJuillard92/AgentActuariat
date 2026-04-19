"""
agents/mortality/agents/builder_node.py
Nœud BuilderAgent du graphe LangGraph.

Evolucion depuis mortality_node.py — conserve toute la logique de calcul
actuariel et ajoute le signal <BUILD_DONE> pour retourner vers le MasterAgent
une fois les calculs terminés.

Signaux émis :
  <BUILD_DONE>  → calculs terminés, retour au MasterAgent
  <HANDOFF_WRITER> → (legacy) aussi reconnu pour compat. ascendante
  <MODEL_CHOICE_CHECKPOINT> → pause soft pour choix de modèle
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

# Tools accessibles au BuilderAgent (identique au MortalityAgent)
BUILDER_TOOLS = {"builder", "statistical_analysis", "graphs", "reasoning", "build_pdf"}


def _get_catalogue_level(state: "AgentState") -> str:
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
    loader_path = _PROJECT_ROOT / "loader.py"
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("loader", loader_path)
        mod  = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        base = mod.get_system_prompt(level=level, agent_name="mortality")
    except Exception as exc:
        print(f"[BuilderAgent] loader error: {exc}", file=sys.stderr)
        fallback = _PROJECT_ROOT / "agents" / "mortality" / "agent_instructions" / "behavioral_contract.md"
        base = fallback.read_text(encoding="utf-8") if fallback.exists() else ""

    # Ajouter le mapping colonnes — chargement depuis MemoryManager (Parquet)
    dataset_ref = state.get("dataset_ref")
    if dataset_ref:
        try:
            from session.memory_manager import MemoryManager
            mm = MemoryManager(dataset_ref)
            mm.load()
            df = mm.load_dataframe()
            if df is not None:
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
            # Injecter le contexte antérieur (résumé compacté)
            ctx_block = mm.get_context_block()
            if ctx_block:
                base += f"\n\n{ctx_block}"
        except Exception:
            pass

    # Paramètres d'étude confirmés (collectés via le formulaire de désambiguation)
    data_store = state.get("data_store") or {}
    study_plan = data_store.get("study_plan") or {}
    if study_plan:
        base += "\n\n## Paramètres d'étude confirmés par l'utilisateur\n\n"
        base += "Utilise ces paramètres directement dans les tool calls — ne les demande pas à l'utilisateur :\n\n"
        param_labels = {
            "observation_end_date":       "Date de fin d'observation",
            "observation_start_date":     "Date de début d'observation",
            "baseline_regulatory_table":  "Table de référence réglementaire",
            "sexe":                       "Sexe du portefeuille (pour benchmarking)",
            "age_min":                    "Âge minimum",
            "age_max":                    "Âge maximum",
            "smoothing_algorithm":        "Algorithme de lissage",
        }
        for k, v in study_plan.items():
            label = param_labels.get(k, k)
            base += f"- **{label}** : `{v}`\n"

    # Instructions sur <BUILD_DONE>
    base += (
        "\n\n## Signal de fin de calculs\n\n"
        "Quand les calculs actuariels sont terminés et qu'un rapport est demandé, "
        "émettre **exactement** le signal `<BUILD_DONE>` dans ta réponse. "
        "Le MasterAgent prendra le relais pour router vers le WriterAgent.\n"
        "Tu peux aussi utiliser `<HANDOFF_WRITER>` (équivalent legacy)."
    )

    # Documents de contexte
    context_docs = state.get("context_docs") or []
    if context_docs:
        base += "\n\n## Documents de contexte\n\n"
        for doc in context_docs:
            base += f"### {doc['name']}\n\n```\n{doc['content']}\n```\n\n"

    return base


def builder_node(state: "AgentState") -> dict:
    """
    Nœud BuilderAgent : orchestration des calculs actuariels.
    Retourne la mise à jour de l'état LangGraph.
    """
    import openai
    from agents.mortality.agents.mortality_node import _to_openai_dict, _from_openai_response, sanitize_openai_messages

    level = _get_catalogue_level(state)
    system_prompt = _build_system_prompt(state, level)

    from tools.tool_registry import get_openai_tools
    all_tools = get_openai_tools()
    tools = [t for t in all_tools if t["function"]["name"] in BUILDER_TOOLS]

    MAX_HISTORY = 20
    raw_msgs = state["messages"]
    if len(raw_msgs) > MAX_HISTORY:
        raw_msgs = raw_msgs[-MAX_HISTORY:]

    messages = [{"role": "system", "content": system_prompt}]
    for msg in raw_msgs:
        messages.append(_to_openai_dict(msg))
    messages = sanitize_openai_messages(messages)

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
        "agent":       "BuilderAgent",
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
        new_events.append({"type": "error", "message": f"Erreur API OpenAI (BuilderAgent) : {exc}"})
        return {"messages": [], "events": new_events}

    choice   = response.choices[0]
    msg_obj  = choice.message
    lc_msg   = _from_openai_response(msg_obj)

    # ── Event : réponse de l'API ──────────────────────────────────────────────
    usage = response.usage
    new_events.append({
        "type":               "llm_output",
        "agent":              "BuilderAgent",
        "finish_reason":      choice.finish_reason,
        "content_preview":    (msg_obj.content or "")[:400],
        "prompt_tokens":      usage.prompt_tokens      if usage else None,
        "completion_tokens":  usage.completion_tokens  if usage else None,
        "total_tokens":       usage.total_tokens       if usage else None,
        "n_tool_calls":       len(msg_obj.tool_calls or []),
    })

    data_store = state.get("data_store") or {}
    data_store["_builder_turns"] = data_store.get("_builder_turns", 0) + 1

    if choice.finish_reason != "tool_calls":
        content = msg_obj.content or ""
        if content:
            new_events.append({"type": "message", "content": content})
            data_store.setdefault("_reasoning_log", []).append(content)

    result: dict = {
        "messages":         [lc_msg],
        "events":           new_events,
        "plan_established": True,
        "data_store":       data_store,
    }
    # active_agent="master" si BUILD_DONE — _should_continue_builder le détecte aussi
    # via data_store, mais on le signale explicitement pour la lisibilité
    content = msg_obj.content or ""
    if "<BUILD_DONE>" in content or "<HANDOFF_WRITER>" in content:
        result["active_agent"] = "master"
    return result
