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
BUILDER_TOOLS = {
    "builder",
    "preprocessing",        # tools/preprocessing/clean_records (R1-R6)
    "statistical_analysis",
    "aggregation",          # tools/aggregation/* — déciles d'exposition, etc.
    "graphs",
    "reasoning",
    "build_pdf",
}


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

    # Ajouter le mapping colonnes — préférer le Parquet normalisé (post-UI)
    # à l'original. Le mapping affiché doit refléter le df que les tools
    # vont effectivement consommer.
    dataset_ref = state.get("dataset_ref")
    if dataset_ref:
        try:
            from session.memory_manager import MemoryManager
            from session.dataset_store import DatasetStore
            mm = MemoryManager(dataset_ref)
            mm.load()
            df = DatasetStore.load_preferring_normalized(
                state.get("data_store") or {}, dataset_ref,
            )
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

    # Bloc exhaustif "capacités par section" dérivé du YAML (US report_mode).
    # Master indique dans son HumanMessage quelles sections sont actives pour
    # la session courante. Tu ne produis QUE les clés de ces sections.
    try:
        base += "\n\n" + _capabilities_block()
    except Exception as exc:
        print(f"[BuilderAgent] _capabilities_block error: {exc}", file=sys.stderr)

    # Règles de pilotage par sections actives + report_mode
    base += (
        "\n\n## Règle de session\n\n"
        "Le MasterAgent t'envoie un HumanMessage listant `Sections actives` et "
        "`Reste à produire` (les clés manquantes dans le data_store). Tu dois :\n"
        "  1. Produire UNIQUEMENT les clés listées dans `Reste à produire`.\n"
        "  2. NE PAS relancer les tools pour les clés listées dans `Déjà produit`.\n"
        "  3. Émettre **exactement** `<BUILD_DONE>` une fois toutes les clés produites.\n"
        "  4. Si `report_mode == 'raw_rates'`, NE PAS appeler `builder.smoothing` — "
        "la clé `smoothed_table` est produite par assimilation automatique des taux bruts "
        "par une branche déterministe du nœud Builder.\n"
        "  5. Si `report_mode == 'description'`, ne PAS appeler `builder.crude_rates`, "
        "`builder.smoothing`, `builder.validation`, `builder.benchmarking` — seules "
        "`builder.exposure` (optionnel) et les tools `preprocessing.clean_records` + "
        "`statistical_analysis.*` sont requis.\n"
    )

    # ── Override step 0 (data dictionary) si column_mapping confirmé ────────
    # Quand le mapping a déjà été validé via l'UI (column_mapping_confirmed),
    # on supprime explicitement la phase de confirmation du dictionnaire
    # de données — la step0 du contrat comportemental est court-circuitée.
    if data_store.get("column_mapping_confirmed"):
        base += (
            "\n\n## OVERRIDE — étape 0 (dictionnaire de données) DÉSACTIVÉE\n\n"
            "Le mapping des colonnes a déjà été confirmé par l'utilisateur via "
            "l'interface (`column_mapping_confirmed=True`). Tu DOIS donc :\n"
            "  - **NE PAS proposer** de tableau de validation des colonnes.\n"
            "  - **NE PAS attendre** de confirmation utilisateur sur les colonnes.\n"
            "  - Lancer directement les tools nécessaires pour produire les clés "
            "manquantes listées par le Master.\n"
            "Cette règle a priorité sur l'instruction de step0_data_dictionary.md.\n"
        )

    # Documents de contexte
    context_docs = state.get("context_docs") or []
    if context_docs:
        base += "\n\n## Documents de contexte\n\n"
        for doc in context_docs:
            base += f"### {doc['name']}\n\n```\n{doc['content']}\n```\n\n"

    return base


def _capabilities_block() -> str:
    """Génère la table 'section → clés → tools' exhaustive depuis le YAML.

    Le Builder reçoit ainsi, dans son system prompt, une carte complète de ce
    qu'il peut produire et pour quelle section. Master indique ensuite les
    sections actives via son HumanMessage d'invocation.
    """
    import re
    import yaml as _yaml
    from knowledge_base.report_template.template_loader import (
        build_manifest, DEFAULT_TEMPLATE,
    )

    placeholder_re = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

    manifest = build_manifest()
    # Index clé -> tool via produced_by (toutes catégories confondues)
    tool_for_key: dict[str, str] = {}
    for entry in (manifest.master_from_data + manifest.master_from_modeling
                  + manifest.builder_outputs):
        tool = (entry.produced_by or {}).get("tool")
        if tool:
            tool_for_key[entry.key] = tool

    with open(DEFAULT_TEMPLATE, encoding="utf-8") as f:
        tpl = _yaml.safe_load(f) or {}

    builder_outputs_keys = {e.key for e in manifest.builder_outputs}

    lines = ["## Capacités disponibles (par section du rapport)\n"]
    for section in (tpl.get("sections") or []):
        sid = section.get("id", "?")
        keys: set[str] = set()

        narrative = section.get("narrative") or {}
        for field in ("text", "text_default", "text_raw_rates"):
            text = narrative.get(field) or ""
            keys.update(placeholder_re.findall(text))

        directives = section.get("llm_directives") or {}
        pta = directives.get("post_table_analysis") or {}
        fse = pta.get("few_shot_example") or ""
        keys.update(placeholder_re.findall(fse))

        for v in (section.get("visual_specs") or []):
            src = v.get("source") or ""
            root = src.split(".")[0] if src else ""
            if root:
                keys.add(root)

        # Ne garder que les clés produites par le Builder (builder_outputs)
        builder_keys_for_section = sorted(keys & builder_outputs_keys)
        tools_for_section = sorted({
            tool_for_key[k] for k in builder_keys_for_section if k in tool_for_key
        })

        lines.append(f"### Section : {sid}")
        lines.append(f"  Clés à produire : {builder_keys_for_section}")
        lines.append(f"  Tools à appeler : {tools_for_section}")
        lines.append("")

    return "\n".join(lines)


def _has_pending_decision(messages: list) -> bool:
    """Retourne True si le dernier ToolMessage contient un marqueur
    `decision_required`. Sert de garde-fou : quand un tool a demandé une
    décision utilisateur, le LLM ne doit PAS enchaîner d'autres tool_calls.
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = str(getattr(msg, "content", "") or "")
            return "decision_required" in content
    return False


def builder_node(state: "AgentState") -> dict:
    """
    Nœud BuilderAgent : orchestration des calculs actuariels.
    Retourne la mise à jour de l'état LangGraph.
    """
    import openai
    from agents.mortality.agents.mortality_node import _to_openai_dict, _from_openai_response, sanitize_openai_messages

    data_store = state.get("data_store") or {}

    # ── Bloc C' : assimilation déterministe en mode raw_rates ───────────────
    # Si qx_table est présent et report_mode == "raw_rates", on copie les taux
    # bruts dans smoothed_table sans appeler builder.smoothing. Le LLM Builder
    # ne doit alors pas relancer le lissage (règle explicite dans son prompt).
    if (data_store.get("report_mode") == "raw_rates"
            and data_store.get("qx_table")
            and not data_store.get("smoothed_table")):
        data_store["smoothed_table"] = [
            {"age": r.get("age"), "q_x_brut": r.get("q_x_brut") or r.get("qx"),
             "q_x_lisse": r.get("q_x_brut") or r.get("qx")}
            for r in (data_store["qx_table"] or []) if r.get("age") is not None
        ]

    # ── Branches déterministes pour clés "dérivées" ─────────────────────────
    # Les tools `aggregation.exposure_deciles` et `builder.validation`
    # sont rarement choisis spontanément par le LLM ; on les exécute
    # déterministement dès que leurs prérequis sont satisfaits, pour
    # gagner des cycles Master↔Builder.
    if (data_store.get("smoothed_table")
            and data_store.get("qx_table")
            and not data_store.get("ci_table")):
        try:
            from tools.builder.validation import run as _validation_run
            # Respect du choix utilisateur (study_plan.methods.builder.validation)
            sp_user = data_store.get("study_plan") or {}
            user_methods = sp_user.get("methods") or {}
            chosen_validation = user_methods.get("builder.validation")
            validation_fn = (chosen_validation
                             if chosen_validation and chosen_validation != "auto"
                             else "confidence_intervals")
            res = _validation_run(
                data=data_store,
                params={"function_name": validation_fn,
                        "qx_col": "q_x_lisse", "alpha": 0.05},
            )
            if "ci_table" in res:
                data_store["ci_table"] = res["ci_table"]
                # Mirror dans data_store.validation pour compat
                v = data_store.get("validation") or {}
                if isinstance(v, dict):
                    v.update(res)
                    data_store["validation"] = v
        except Exception:
            pass

    if (data_store.get("qx_table")
            and not data_store.get("qx_deciles_table")):
        try:
            from tools.aggregation.exposure_deciles import run as _deciles_run
            res = _deciles_run(
                data={"qx_table":       data_store.get("qx_table"),
                      "smoothed_table": data_store.get("smoothed_table")},
                params={"n_buckets": 10},
            )
            if "qx_deciles_table" in res:
                data_store["qx_deciles_table"] = res["qx_deciles_table"]
        except Exception:
            pass

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
    from agents.mortality.agents.llm_config import get_llm_config
    _builder_cfg_for_event = get_llm_config("builder.llm")
    new_events.append({
        "type":        "llm_input",
        "agent":       "BuilderAgent",
        "model":       _builder_cfg_for_event.get("model", "?"),
        "n_messages":  len(messages),
        "max_tokens":  _builder_cfg_for_event.get("max_tokens", 4000),
        "has_tools":   bool(tools),
        "last_user":   str(last_user)[:400],
        "system_head": system_prompt[:300],
    })

    try:
        from agents.mortality.agents.llm_config import get_llm_config
        cfg = get_llm_config("builder.llm")
        response = call_with_retry(
            client,
            model=cfg["model"],
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            max_tokens=cfg.get("max_tokens", 4000),
            temperature=cfg.get("temperature", 0.0),
        )
    except Exception as exc:
        new_events.append({"type": "error", "message": f"Erreur API OpenAI (BuilderAgent) : {exc}"})
        return {"messages": [], "events": new_events}

    choice   = response.choices[0]
    msg_obj  = choice.message
    lc_msg   = _from_openai_response(msg_obj)

    # ── Garde-fou decision_required ──────────────────────────────────────────
    # Si un tool précédent a retourné un marqueur `decision_required`, on ne
    # laisse PAS le LLM enchaîner des tool_calls, qu'il ait émis du content ou
    # non. Il doit rendre la main à l'utilisateur pour qu'il choisisse parmi
    # les options proposées par le tool.
    if _has_pending_decision(raw_msgs):
        lc_tool_calls = getattr(lc_msg, "tool_calls", None)
        if lc_tool_calls:
            lc_msg.tool_calls = []
            if hasattr(msg_obj, "tool_calls"):
                msg_obj.tool_calls = []
            # Si le LLM n'a émis aucun texte, on force un message explicite pour
            # que l'UI n'affiche pas une réponse vide.
            content = getattr(lc_msg, "content", None) or ""
            if not content.strip():
                lc_msg.content = (
                    "[Décision utilisateur en attente — tool_calls supprimés par le garde-fou] "
                    "Merci de choisir parmi les options proposées avant toute nouvelle action."
                )
            new_events.append({
                "type":    "message",
                "content": "[garde-fou] Décision utilisateur en attente — tool_calls supprimés.",
            })

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
