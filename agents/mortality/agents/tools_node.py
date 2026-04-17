"""
report_agent/agents/tools_node.py
Nœud d'exécution des tools actuariels dans le graphe LangGraph.

Reçoit l'état courant, exécute tous les tool_calls du dernier message
assistant, met à jour data_store, et retourne les nouveaux messages
(ToolMessage) + les events canvas.
"""
from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

import pandas as pd
from langchain_core.messages import ToolMessage

from tools.tool_registry import call_tool

if TYPE_CHECKING:
    from agents.mortality.agents.state import AgentState

# Mapping function_name → clé data_store
_RESULT_KEYS: dict[str, str] = {
    "portfolio_summary": "summary",
    "age_distribution":  "ages",
    "time_series":       "series",
    "segmentation":      "segmentation",
    "exposure":          "exposure_table",
    "crude_rates":       "qx_table",
    "smoothing":         "smoothed_table",
    "diagnostics":       "diagnostics",
    "validation":        "validation",
    "benchmarking":      "benchmarking",
}


def execute_tools(
    state: "AgentState",
    approval_event: threading.Event | None = None,
    cancel_flag: list[bool] | None = None,
) -> dict:
    """
    Exécute tous les tool_calls du dernier message assistant.

    Retourne un dict de mise à jour de l'état LangGraph :
      - messages : liste de ToolMessage (un par tool call)
      - data_store : dict mis à jour
      - events : nouveaux events canvas
      - plan_established : True après le premier tool call
      - pending_tool_call : tool en attente d'approbation (step_by_step)
    """
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", None) or []

    # Charger le DataFrame depuis MemoryManager (Parquet) — jamais depuis l'état LangGraph
    df: pd.DataFrame | None = None
    dataset_ref = state.get("dataset_ref")
    if dataset_ref:
        try:
            from session.memory_manager import MemoryManager
            df = MemoryManager(dataset_ref).load().load_dataframe()
        except Exception:
            pass

    data_store: dict = state.get("data_store") or {}
    step_by_step: bool = state.get("step_by_step", False)
    new_messages: list[ToolMessage] = []
    new_events: list[dict] = []
    pending: dict | None = None

    for tc in tool_calls:
        fn_name = tc["name"]
        try:
            fn_args = json.loads(tc["args"]) if isinstance(tc["args"], str) else tc["args"]
        except (json.JSONDecodeError, TypeError):
            fn_args = {}

        function_name = fn_args.get("function_name", "")
        params = fn_args.get("params", {})
        tc_id = tc["id"]

        # Event tool_call
        new_events.append({
            "type": "tool_call",
            "tool": fn_name,
            "function_name": function_name,
            "params": params,
            "tool_call_id": tc_id,
        })

        # Mode pas à pas — pause avant exécution
        if step_by_step and approval_event is not None:
            pending = {"tool": fn_name, "function_name": function_name, "params": params}
            new_events.append({
                "type": "awaiting_approval",
                "tool": fn_name,
                "function_name": function_name,
                "params": params,
                "tool_call_id": tc_id,
            })
            approval_event.clear()
            approval_event.wait(timeout=300)
            if cancel_flag and cancel_flag[0]:
                cancel_flag[0] = False
                rejection = {"erreur": "Étape annulée par l'utilisateur."}
                new_messages.append(ToolMessage(
                    content=json.dumps(rejection, ensure_ascii=False),
                    tool_call_id=tc_id,
                ))
                new_events.append({
                    "type": "tool_result",
                    "tool": fn_name,
                    "function_name": function_name,
                    "result": rejection,
                    "tool_call_id": tc_id,
                })
                continue

        # Exécution du tool
        context_for_tool = None
        if fn_name == "reasoning":
            history = [m for m in state["messages"] if hasattr(m, "type") and m.type == "human"]
            last_human = history[-1].content if history else ""
            context_for_tool = {
                "user_message": last_human,
                "history": [{"role": "user", "content": last_human}],
                "csv_columns": list(df.columns) if df is not None else [],
            }

        result = call_tool(
            tool_name=fn_name,
            function_name=function_name,
            params=params,
            df=df,
            data=data_store,
            context=context_for_tool,
        )

        # Stocker dans data_store
        if "erreur" not in result:
            if fn_name == "builder" and function_name == "exposure":
                # Stocker toutes les clés scalaires + exposure_table
                for k, v in result.items():
                    if k not in ("note", "lignes_exclues"):
                        data_store[k] = v
                # Alias pour load_yaml_template
                data_store.setdefault("cohort_min_age", result.get("age_min"))
                data_store.setdefault("cohort_max_age", result.get("age_max"))
                data_store.setdefault("total_exposure_years", result.get("total_exposure"))
            elif fn_name == "builder" and function_name == "crude_rates":
                data_store["qx_table"] = result.get("qx_table", [])
            elif fn_name == "builder" and function_name == "smoothing":
                data_store["smoothed_table"] = result.get("smoothed_table", [])
                data_store["smoothing_method"] = result.get("method", "whittaker")
                # Propager dans study_plan pour load_yaml_template
                sp = data_store.setdefault("study_plan", {})
                sp.setdefault("smoothing_algorithm", result.get("method", "whittaker_henderson"))
            elif fn_name == "builder" and function_name == "validation":
                # Merger dans data_store["validation"] au lieu d'écraser
                existing = data_store.get("validation") or {}
                if isinstance(existing, dict):
                    existing.update(result)
                    data_store["validation"] = existing
                else:
                    data_store["validation"] = result
            else:
                store_key = _RESULT_KEYS.get(function_name, function_name)
                data_store[store_key] = result

        # Log de session
        data_store.setdefault("_call_log", [])
        data_store["_call_log"].append({
            "step":          len(data_store["_call_log"]) + 1,
            "tool":          fn_name,
            "function_name": function_name,
            "params":        params,
            "result_summary": {
                k: (f"[{len(v)} lignes]" if isinstance(v, list) else str(v)[:300])
                for k, v in result.items()
                if k not in ("image_b64", "samples")
            },
            "has_error": "erreur" in result,
        })

        # Event tool_result
        new_events.append({
            "type": "tool_result",
            "tool": fn_name,
            "function_name": function_name,
            "result": result,
            "tool_call_id": tc_id,
        })

        # ToolMessage pour LangGraph (images tronquées)
        result_for_msg = {
            k: ("<image base64 tronquée>" if k == "image_b64" else v)
            for k, v in result.items()
        }
        new_messages.append(ToolMessage(
            content=json.dumps(result_for_msg, ensure_ascii=False, default=str)[:6000],
            tool_call_id=tc_id,
        ))

    return {
        "messages": new_messages,
        "data_store": data_store,
        "events": new_events,
        "plan_established": True,
        "pending_tool_call": pending,
    }
