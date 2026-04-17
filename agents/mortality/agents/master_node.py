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


def _extract_study_plan_from_history(messages: list) -> dict:
    """
    Extrait les paramètres d'étude mentionnés dans la conversation et
    les retourne sous forme de study_plan dict.
    Appelle GPT-4o en JSON mode pour parser les intentions de l'utilisateur.
    """
    import openai
    import json
    from agents.mortality.agents._utils import call_with_retry

    # Reconstituer le texte de la conversation
    conv_lines = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            conv_lines.append(f"{role.upper()}: {str(content)[:300]}")
    if not conv_lines:
        return {}

    conversation_text = "\n".join(conv_lines[-20:])  # 20 derniers messages

    prompt = (
        "Extrait les paramètres d'étude actuarielle mentionnés dans cette conversation.\n"
        "Retourne UNIQUEMENT un JSON avec les clés présentes (ignore les absentes).\n\n"
        "Clés possibles :\n"
        "  observation_start_date (YYYY-MM-DD)\n"
        "  observation_end_date (YYYY-MM-DD)\n"
        "  observation_period_years (liste d'années ex: [2019,2020,2021])\n"
        "  study_objective (ex: 'prévoyance collective décès')\n"
        "  product_list (liste de codes produits)\n"
        "  smoothing_algorithm (ex: 'whittaker_henderson')\n"
        "  baseline_regulatory_table (ex: 'TH0002')\n"
        "  cohort_min_age (entier)\n"
        "  cohort_max_age (entier)\n"
        "  confidence_interval_level (ex: 0.95)\n"
        "  chi_squared_p_significance (ex: 0.05)\n\n"
        f"Conversation :\n{conversation_text}\n\n"
        "JSON uniquement, sans markdown :"
    )

    try:
        client = openai.OpenAI()
        response = call_with_retry(
            client,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=400,
        )
        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)
    except Exception:
        return {}


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


def _detect_need_data(messages: list) -> list[str]:
    """
    Parcourt les derniers messages pour trouver <NEED_DATA: field1, field2>.
    Retourne la liste des champs manquants, ou [] si pas de signal.
    """
    import re
    for msg in reversed(messages):
        content = getattr(msg, "content", "") or ""
        m = re.search(r"<NEED_DATA:\s*([^>]+)>", content)
        if m:
            fields = [f.strip() for f in m.group(1).split(",") if f.strip()]
            return fields
        # Chercher aussi dans les ToolMessages
        if hasattr(msg, "content") and isinstance(msg.content, str):
            m2 = re.search(r"<NEED_DATA:\s*([^>]+)>", msg.content)
            if m2:
                fields = [f.strip() for f in m2.group(1).split(",") if f.strip()]
                return fields
    return []


def master_node(state: "AgentState") -> dict:
    """
    Nœud MasterAgent : orchestre BuilderAgent et WriterAgent.
    Retourne la mise à jour de l'état LangGraph.
    """
    import openai
    from agents.mortality.agents.mortality_node import _to_openai_dict, _from_openai_response, sanitize_openai_messages

    data_store  = state.get("data_store") or {}
    dataset_ref = state.get("dataset_ref")
    # df_json supprimé — master n'a pas besoin du DataFrame brut
    # Il transmet dataset_ref aux sous-agents via l'état LangGraph

    # ── Court-circuit déterministe : <NEED_DATA> → re-router vers builder ────
    need_data_fields = _detect_need_data(state.get("messages") or [])
    if need_data_fields:
        attempts = data_store.get("_need_data_attempts", 0)
        if attempts < 2:
            data_store["_need_data_attempts"] = attempts + 1
            # Filtrer les champs que le builder peut effectivement produire
            # (les champs de study_plan ne peuvent pas venir du builder)
            _STUDY_PLAN_FIELDS = {
                "observation_period_years", "num_observation_years",
                "observation_start_date", "observation_end_date",
                "study_objective", "smoothing_algorithm", "baseline_regulatory_table",
            }
            builder_fields = [f for f in need_data_fields if f not in _STUDY_PLAN_FIELDS]

            if not builder_fields:
                # Uniquement des champs de study_plan manquants → le builder ne peut rien faire
                log.warning("[MasterAgent] NEED_DATA contient uniquement des champs study_plan "
                            "(%s) — passage en mode dégradé.", need_data_fields)
                # Laisser le pipeline continuer avec des données partielles
            else:
                from langchain_core.messages import HumanMessage
                instr = (
                    f"Le WriterAgent a besoin des données suivantes pour générer le rapport : "
                    f"{builder_fields}. "
                    f"Lance les outils nécessaires pour calculer ces données "
                    f"(builder.exposure, builder.crude_rates, builder.smoothing, "
                    f"builder.diagnostics, builder.validation, builder.benchmarking selon ce qui manque), "
                    f"puis émet <BUILD_DONE> quand c'est fait."
                )
                inject_msg = HumanMessage(content=instr)
                data_store["_need_data_attempts"] = attempts + 1
                new_events = [{
                    "type":  "agent_switch",
                    "agent": "MasterAgent",
                }, {
                    "type":    "message",
                    "content": f"[MasterAgent] Données manquantes : {builder_fields}. "
                               f"Relance BuilderAgent. (tentative {attempts + 1}/2)",
                }]
                return {
                    "messages":     [inject_msg],
                    "events":       new_events,
                    "active_agent": "builder",
                    "data_store":   data_store,
                }

    # ── Désambiguation : vérifier les prérequis si pas encore fait ───────────
    # _disambiguation_done persiste dans data_store toute la session.
    # Une fois True, master ne re-déclenche jamais le modal.
    if not data_store.get("_disambiguation_done"):
        messages_list = state.get("messages") or []
        last_human = next(
            (getattr(m, "content", "") for m in reversed(messages_list)
             if getattr(m, "type", "") == "human"),
            "",
        )
        if last_human:
            try:
                from agents.master.disambiguation import run_disambiguation
                disam = run_disambiguation(last_human, dataset_ref, data_store)
            except Exception:
                disam = {"status": "ready"}

            if disam["status"] == "needs_input":
                return {
                    "messages": [],
                    "events": [
                        {"type": "agent_switch", "agent": "MasterAgent"},
                        {
                            "type":                      "disambiguation_required",
                            "task_type":                 disam.get("task_type"),
                            "needs_column_mapping":      disam.get("needs_column_mapping", False),
                            "needs_form":                disam.get("needs_form", False),
                            "column_mapping_suggestion": disam.get("column_mapping_suggestion", {}),
                            "df_columns":                disam.get("df_columns", []),
                            "form_fields":               disam.get("form_fields", []),
                        },
                    ],
                    "data_store": data_store,
                    # Ne pas marquer running=False ici — canvas attend la confirmation
                }
            elif disam["status"] == "unclear":
                return {
                    "messages": [],
                    "events": [
                        {"type": "agent_switch", "agent": "MasterAgent"},
                        {"type": "message", "content": disam.get("message",
                         "Je n'ai pas bien compris votre demande. Pouvez-vous préciser ?")},
                    ],
                    "data_store": data_store,
                }
            # status == "ready" : tous les prérequis sont là, marquer comme fait
            data_store["_disambiguation_done"] = True

    system_prompt = _augment_with_data_store(_build_system_prompt(), data_store)

    # MasterAgent n'a pas de tools — il orchestre via signaux textuels
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
        new_events.append({"type": "message", "content": content})

    new_state: dict = {"messages": [lc_msg], "events": new_events}

    # ── Fix A : extraire et persister le study_plan avant tout routing ────────
    if goes_builder or goes_writer:
        existing_plan = data_store.get("study_plan") or {}
        if not existing_plan:
            extracted = _extract_study_plan_from_history(messages)
            if extracted:
                data_store["study_plan"] = extracted
                new_events.append({
                    "type":    "study_plan_extracted",
                    "content": f"study_plan extrait depuis la conversation : {list(extracted.keys())}",
                })
                new_state["data_store"] = data_store

    if goes_builder:
        new_state["active_agent"] = "builder"
    elif goes_writer:
        # ── Garde-fou : vérifier que le builder a bien produit les données de base ──
        # Si exposure_table est absent, le builder n'a pas fini — re-router vers lui
        # avec une instruction explicite plutôt que d'envoyer le writer dans le vide.
        _MINIMUM_BUILDER_KEYS = ["exposure_table", "smoothed_table"]
        missing_keys = [k for k in _MINIMUM_BUILDER_KEYS if not data_store.get(k)]
        if missing_keys:
            log.warning(
                "[MasterAgent] GO_WRITE demandé mais données manquantes : %s — "
                "re-route vers builder.", missing_keys
            )
            from langchain_core.messages import HumanMessage
            instr = (
                f"Le rapport ne peut pas encore être rédigé. "
                f"Il faut d'abord calculer les données de base ({missing_keys}). "
                f"Lance builder.exposure, builder.crude_rates, builder.smoothing "
                f"dans l'ordre, puis émet <BUILD_DONE>."
            )
            new_state["messages"] = [lc_msg, HumanMessage(content=instr)]
            new_state["active_agent"] = "builder"
        else:
            new_state["active_agent"] = "writer"

    return new_state
