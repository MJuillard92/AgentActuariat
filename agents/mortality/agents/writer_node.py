"""
agents/mortality/agents/writer_node.py
Nœud WriterAgent du graphe LangGraph.

Responsabilités :
  - Générer les rapports PDF à partir du data_store
  - Utiliser le template YAML (build_pdf.load_yaml_template)
  - Appeler les tools de rendu (table_renderer, graph_from_spec, assemble_sections)
  - Émettre <WRITE_DONE> quand le rapport est assemblé
  - Émettre <NEED_DATA: field1, field2> si des données builder sont manquantes

Signaux émis :
  <WRITE_DONE>         → rapport généré, retour au MasterAgent
  <NEED_DATA: f1, f2>  → le WriterAgent a besoin de données supplémentaires du BuilderAgent
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

# Tools accessibles au WriterAgent — PURE : aucun outil de calcul
# Le writer lit l'entrepôt, rédige et met en forme. Il ne calcule jamais.
WRITER_TOOLS = {
    "build_pdf",   # load_yaml_template, table_renderer, assemble_sections, certification_report
    "graphs",      # graph_from_spec, builder_plots
    "reasoning",   # think, plan
}


def _build_system_prompt(state: "AgentState") -> str:
    """Charge le system prompt du WriterAgent via loader.py."""
    loader_path = _PROJECT_ROOT / "loader.py"
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("loader", loader_path)
        mod  = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        base = mod.get_system_prompt(level="full", agent_name="report")
    except Exception as exc:
        print(f"[WriterAgent] loader error: {exc}", file=sys.stderr)
        fallback = _PROJECT_ROOT / "agents" / "report" / "agent_instructions" / "behavioral_contract.md"
        base = fallback.read_text(encoding="utf-8") if fallback.exists() else ""

    # Ajouter les instructions sur le workflow et les signaux
    base += (
        "\n\n## Workflow obligatoire\n\n"
        "**Étape 1 — Toujours commencer par** : appeler `build_pdf.load_yaml_template` "
        "pour charger la structure du rapport et identifier les sections prêtes vs manquantes. "
        "Passer `study_plan` dans les params si disponible dans le data_store.\n\n"
        "**Étape 2 — Si des champs requis manquent** : émettre IMMÉDIATEMENT "
        "`<NEED_DATA: field1, field2>` avec la liste exacte des champs manquants retournés "
        "par `load_yaml_template`. Ne jamais tenter de calculer ou dériver une valeur manquante. "
        "Le MasterAgent demandera au BuilderAgent de les produire.\n\n"
        "**Étape 3 — Rédaction section par section** : pour chaque section `ready: true`, "
        "dans cet ordre :\n"
        "  a. Rédiger le texte narratif (dans ton message).\n"
        "  b. Appeler `build_pdf.table_renderer` pour chaque tableau de la section (si applicable).\n"
        "  c. Appeler `graphs.graph_from_spec` pour chaque graphique (si applicable).\n"
        "  d. Appeler `build_pdf.write_section` avec `section_id`, `text` et les légendes — "
        "     cet appel consomme le dernier tableau et le dernier graphique générés "
        "     et les enregistre dans `section_outputs`.\n\n"
        "**Étape 4 — Assemblage** : appeler `build_pdf.assemble_sections` pour produire le PDF final.\n\n"
        "**Étape 5 — Signal de fin** : émettre exactement `<WRITE_DONE>` quand le PDF est produit.\n\n"
        "**Règles absolues** :\n"
        "- Tu ne calcules jamais une valeur manquante\n"
        "- Tu ne modifies jamais les données du data_store\n"
        "- Tu ne cites dans les textes narratifs QUE des chiffres présents dans le data_store\n"
        "- Si une section est marquée `ready: false` → elle est skippée ou `<NEED_DATA>` émis"
    )

    # Résumé du study_plan si disponible
    data_store = state.get("data_store") or {}
    study_plan = data_store.get("study_plan") or state.get("study_plan") or {}
    if study_plan:
        base += "\n\n## Paramètres d'étude (study_plan)\n"
        for k, v in list(study_plan.items())[:12]:
            base += f"  - `{k}` : {v}\n"

    # Résumé des données disponibles
    data_store = state.get("data_store") or {}
    if data_store:
        lines = ["\n\n## Données disponibles dans le data_store\n"]
        available_keys = [k for k in data_store.keys() if not k.startswith("_")]
        for key in available_keys[:20]:  # Limiter pour ne pas surcharger le prompt
            val = data_store[key]
            if isinstance(val, list):
                lines.append(f"  - `{key}` : {len(val)} enregistrements")
            elif isinstance(val, dict):
                lines.append(f"  - `{key}` : dict avec clés {list(val.keys())[:5]}")
            elif val is not None:
                lines.append(f"  - `{key}` : {val}")
        base += "\n".join(lines)

    return base


def writer_node(state: "AgentState") -> dict:
    """
    Nœud WriterAgent : rédaction et génération PDF.
    Retourne la mise à jour de l'état LangGraph.
    """
    import openai
    from agents.mortality.agents.mortality_node import _to_openai_dict, _from_openai_response

    system_prompt = _build_system_prompt(state)

    from tools.tool_registry import get_openai_tools
    all_tools = get_openai_tools()
    tools = [t for t in all_tools if t["function"]["name"] in WRITER_TOOLS]

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
        "agent":       "WriterAgent",
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
        new_events.append({"type": "error", "message": f"Erreur API OpenAI (WriterAgent) : {exc}"})
        return {"messages": [], "events": new_events}

    choice  = response.choices[0]
    msg_obj = choice.message
    lc_msg  = _from_openai_response(msg_obj)

    # ── Event : réponse de l'API ──────────────────────────────────────────────
    usage = response.usage
    new_events.append({
        "type":               "llm_output",
        "agent":              "WriterAgent",
        "finish_reason":      choice.finish_reason,
        "content_preview":    (msg_obj.content or "")[:400],
        "prompt_tokens":      usage.prompt_tokens      if usage else None,
        "completion_tokens":  usage.completion_tokens  if usage else None,
        "total_tokens":       usage.total_tokens       if usage else None,
        "n_tool_calls":       len(msg_obj.tool_calls or []),
    })

    if choice.finish_reason != "tool_calls":
        content = msg_obj.content or ""
        if content:
            new_events.append({"type": "message", "content": content})

        write_done = "<WRITE_DONE>" in content
        need_data  = "<NEED_DATA" in content

        if write_done:
            # Rapport généré — retour au MasterAgent
            return {
                "messages":     [lc_msg],
                "events":       new_events,
                "active_agent": "master",
            }

        if need_data:
            # Données manquantes — retour au MasterAgent pour re-router vers Builder
            return {
                "messages":     [lc_msg],
                "events":       new_events,
                "active_agent": "master",
            }

        new_events.append({"type": "done"})

    return {
        "messages": [lc_msg],
        "events":   new_events,
    }
