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

def _msgpack_safe(obj):
    """Convertit récursivement un objet en types Python natifs pour
    sérialisation msgpack (LangGraph MemorySaver). Couvre les scalaires
    et tableaux numpy/pandas, qui sinon font crasher
    `ormsgpack.packb` avec 'Type is not msgpack serializable'.

    Ordre des checks important : numpy.bool_/integer/floating sont
    sous-classes de bool/int/float → vérifier numpy AVANT Python natif.
    """
    import numpy as _np
    if obj is None:
        return None
    if isinstance(obj, _np.bool_):
        return bool(obj)
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        v = float(obj)
        return v if v == v and v not in (float("inf"), float("-inf")) else None
    if isinstance(obj, _np.ndarray):
        return [_msgpack_safe(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [_msgpack_safe(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return [_msgpack_safe(x) for x in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        # NaN/Inf → None (msgpack-safe ET sémantiquement neutre)
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {str(k): _msgpack_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_msgpack_safe(x) for x in obj]
    return obj


def _inject_user_method(
    fn_name: str,
    function_name: str,
    params: dict,
    study_plan: dict | None,
) -> None:
    """Si l'utilisateur a explicitement choisi une méthode pour ce tool
    via le flux Master, écrit cette méthode dans `params` sous le bon
    nom de paramètre (méthod / function_name). Source de vérité :
    `study_plan["methods"][<tool>.<function>]`. Le nom du paramètre est
    redécouvert via le catalogue (cf. agents/master/method_choices.py).
    """
    if not study_plan or not isinstance(study_plan, dict):
        return
    methods = study_plan.get("methods") or {}
    if not methods:
        return
    tool_key = f"{fn_name}.{function_name}"
    chosen = methods.get(tool_key)
    if not chosen or chosen == "auto":
        return
    try:
        from agents.master.method_choices import _params_with_choices
        specs = _params_with_choices(tool_key)
    except Exception:
        return
    for pname, _values, _default in specs:
        params[pname] = chosen
        break


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
    data_store: dict = state.get("data_store") or {}

    # Priorité au Parquet normalisé écrit par maybe_normalize_records()
    # après validation UI du column_mapping + value_mapping. Ce fichier
    # contient déjà : colonnes renommées canoniques, valeurs enum mappées,
    # dates parsées en datetime64, sentinelles 2999 clippées.
    norm_path = data_store.get("dataset_ref_normalized")
    if norm_path:
        try:
            from pathlib import Path
            if Path(norm_path).exists():
                df = pd.read_parquet(norm_path)
        except Exception:
            df = None
    if df is None and dataset_ref:
        try:
            from session.memory_manager import MemoryManager
            df = MemoryManager(dataset_ref).load().load_dataframe()
        except Exception:
            pass

    # Appliquer column_mapping + value_mapping AVANT de passer df aux tools.
    # Sans ça les tools cherchent les colonnes canoniques (`date_naissance`,
    # `cause_sortie`, …) qui n'existent pas dans le CSV brut (CLINAISS,
    # STATUT, …) et plantent.
    # Format column_mapping stocké : {canonical: csv_col} ; df.rename attend
    # {csv_col: canonical} → on inverse.
    if df is not None:
        mapping = data_store.get("column_mapping") or {}
        if not mapping:
            # Fallback : auto-détection depuis COLUMN_SCHEMA. Le mapping
            # auto-détecté sort de candidates curées et est fiable même
            # sans confirmation UI explicite (column_mapping_confirmed).
            try:
                from agents.mortality.dictionary.column_schema import (
                    COLUMN_SCHEMA, find_col,
                )
                mapping = {
                    role: find_col(df, info["candidates"])
                    for role, info in COLUMN_SCHEMA.items()
                }
                mapping = {k: v for k, v in mapping.items() if v}
                if mapping:
                    data_store["column_mapping"] = mapping
            except Exception:
                mapping = {}
        rename_map = {v: k for k, v in mapping.items() if v and v in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)
        vmap = data_store.get("value_mapping") or {}
        if not vmap:
            # Auto-détection : si sexe et cause_sortie ont des valeurs non
            # canoniques mais reconnues par _SYNONYMS, on les mappe.
            # Heuristique sûre — toute valeur non reconnue reste intacte.
            try:
                from tools.master.suggest_value_mapping import run as _suggest
                enum_specs = {
                    "sexe":         ["H", "F"],
                    "cause_sortie": ["deces", "autre"],
                }
                cols_in_df = {k: v for k, v in enum_specs.items() if k in df.columns}
                if cols_in_df:
                    res = _suggest({"records": df, "enum_specs": cols_in_df}, {})
                    vmap = {
                        k: v for k, v in (res.get("value_mapping") or {}).items()
                        if v  # ignorer mappings vides
                    }
                    if vmap:
                        data_store["value_mapping"] = vmap
            except Exception:
                vmap = {}
        for col, m in vmap.items():
            if not m or col not in df.columns:
                continue
            df[col] = df[col].astype(str).map(lambda v, _m=m: _m.get(v, v))
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

        # Injection de la méthode choisie par l'utilisateur (via Master).
        # Quand plusieurs méthodes sont disponibles pour un tool et que
        # l'utilisateur a explicitement choisi via le flux _pending_need,
        # study_plan["methods"][<tool>.<function>] contient la valeur retenue.
        # On l'injecte ici, ce qui prime sur ce que le LLM aurait choisi.
        _inject_user_method(fn_name, function_name, params, data_store.get("study_plan"))

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

        # Sanitization msgpack systématique : convertit tout numpy/pandas
        # scalar ou DataFrame en types Python natifs. Sans ça, LangGraph
        # MemorySaver crashe sur ormsgpack.packb (numpy.float64 non
        # sérialisable). Doit s'appliquer AVANT tout stockage data_store.
        if isinstance(result, dict):
            result = _msgpack_safe(result)

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
                # Spread les sous-clés au top level pour que le chart
                # multi-séries puisse consommer `ci_table` comme source.
                if "ci_table" in result:
                    data_store["ci_table"] = result["ci_table"]

            # ── preprocessing.clean_records : spread cleaned_records + exclusion_report + total_records ──
            elif fn_name == "preprocessing" and function_name == "clean_records":
                if "cleaned_records" in result:
                    cr = result["cleaned_records"]
                    # LangGraph MemorySaver utilise msgpack qui ne sait pas
                    # sérialiser un DataFrame. On convertit en list[dict] —
                    # JSON-safe et exploitable par les tools en aval.
                    if isinstance(cr, pd.DataFrame):
                        cr = cr.to_dict(orient="records")
                    data_store["cleaned_records"] = cr
                if "exclusion_report" in result:
                    data_store["exclusion_report"] = result["exclusion_report"]
                    final = (result["exclusion_report"] or {}).get("final_count")
                    if final is not None:
                        data_store["total_records"] = final

            # ── statistical_analysis.segmentation ──
            # Deux consommateurs aval, deux noms :
            #   - YAML Writer attend `segmentations` (pluriel)
            #   - tools/graphs/analysis_plots.py lit `data["segmentation"]` (singulier)
            # On alimente les deux pour ne casser ni l'un ni l'autre.
            elif fn_name == "statistical_analysis" and function_name == "segmentation":
                data_store["segmentations"] = result.get("segmentations", {})
                data_store["segmentation"]  = result   # rétro-compat analysis_plots

            # ── statistical_analysis.time_series : spread serie + serie_h + serie_f ──
            elif fn_name == "statistical_analysis" and function_name == "time_series":
                if "serie" in result:
                    data_store["serie"] = result["serie"]
                if "serie_h" in result:
                    data_store["serie_h"] = result["serie_h"]
                if "serie_f" in result:
                    data_store["serie_f"] = result["serie_f"]
                # Aussi sauvegarder le résultat complet pour debug
                data_store["series"] = result

            # ── aggregation.exposure_deciles : spread qx_deciles_table ──
            elif fn_name == "aggregation" and function_name == "exposure_deciles":
                if "qx_deciles_table" in result:
                    data_store["qx_deciles_table"] = result["qx_deciles_table"]

            else:
                store_key = _RESULT_KEYS.get(function_name, function_name)
                # Sanitize : DataFrame → list[dict] pour msgpack-safety
                if isinstance(result, pd.DataFrame):
                    data_store[store_key] = result.to_dict(orient="records")
                elif isinstance(result, dict):
                    data_store[store_key] = {
                        k: (v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v)
                        for k, v in result.items()
                    }
                else:
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

        # Event tool_result — sanitize : on retire les DataFrames bruts
        # (msgpack ne sait pas les sérialiser et LangGraph checkpointe l'état
        # complet via msgpack). Les versions list[dict] sont déjà dans data_store.
        result_safe = {}
        for k, v in result.items():
            if isinstance(v, pd.DataFrame):
                result_safe[k] = v.to_dict(orient="records")
            elif k == "image_b64":
                result_safe[k] = "<image base64 tronquée>"
            else:
                result_safe[k] = v
        new_events.append({
            "type": "tool_result",
            "tool": fn_name,
            "function_name": function_name,
            "result": result_safe,
            "tool_call_id": tc_id,
        })

        # ToolMessage pour LangGraph (images tronquées + DataFrame → records)
        result_for_msg = result_safe
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
