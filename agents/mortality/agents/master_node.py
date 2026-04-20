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


# Données minimales que le Builder doit produire avant que le Writer puisse tourner
_ALL_BUILDER_KEYS: dict[str, str] = {
    "exposure_table": "exposition centrale",
    "qx_table":       "taux bruts q_x",
    "smoothed_table": "table lissée",
    "diagnostics":    "diagnostics de crédibilité",
    "validation":     "validation statistique",
    "benchmarking":   "benchmarking réglementaire",
}
_MINIMUM_BUILDER_KEYS = ["exposure_table", "smoothed_table"]


def _classify_intent(last_human: str, data_store: dict, dataset_ref: str | None) -> dict:
    """
    Classifie l'intention de l'utilisateur via un appel JSON structuré.
    Retourne {"intent": str, "reply": str}.
    Intents : build_only | write_only | build_and_write | question
    """
    import openai
    from agents.mortality.agents._utils import call_with_retry

    has_data  = bool(dataset_ref or data_store.get("_dataset_ref"))
    has_calcs = all(data_store.get(k) for k in _MINIMUM_BUILDER_KEYS)
    has_all   = all(data_store.get(k) for k in _ALL_BUILDER_KEYS)

    context = (
        f"Fichier CSV chargé : {'oui' if has_data else 'non'}. "
        f"Calculs de base effectués : {'oui' if has_calcs else 'non'}. "
        f"Calculs complets (prêt pour rapport) : {'oui' if has_all else 'non'}."
    )

    prompt = (
        "Tu es un routeur d'intention pour un système actuariel. "
        "Classifie la demande en une catégorie :\n"
        "- build_only : calculs uniquement (exposition, taux, lissage…)\n"
        "- write_only : rapport uniquement (les calculs sont supposés faits)\n"
        "- build_and_write : calculs ET rapport\n"
        "- question : question, explication, hors calculs/rapport\n\n"
        f"Contexte : {context}\n"
        f"Demande : {last_human[:500]}\n\n"
        "Réponds UNIQUEMENT en JSON :\n"
        '{"intent": "...", "reply": "confirmation courte en français (1-2 phrases max)"}'
    )

    try:
        client = openai.OpenAI()
        resp = call_with_retry(
            client,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as exc:
        print(f"[MasterAgent] _classify_intent error: {exc}", file=sys.stderr)
        return {"intent": "unclear", "reply": "Je n'ai pas compris votre demande. Pouvez-vous préciser ?"}


def _preflight_writer(data_store: dict) -> tuple[bool, list[str]]:
    """
    Vérifie que toutes les données nécessaires au WriterAgent sont présentes.
    Retourne (prêt, liste des labels manquants).
    """
    missing = [label for key, label in _ALL_BUILDER_KEYS.items() if not data_store.get(key)]
    return (len(missing) == 0, missing)


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


def _augment_with_data_store(prompt: str, data_store: dict, dataset_ref: str | None = None) -> str:
    """Ajoute un résumé du data_store au system prompt si des calculs ont déjà tourné."""
    if not data_store and not dataset_ref:
        return prompt

    summary_lines = ["\n\n## Contexte de session"]

    # Statut dataset — critique : le LLM doit savoir si un CSV est chargé
    if dataset_ref or data_store.get("_dataset_ref"):
        ref = dataset_ref or data_store.get("_dataset_ref")
        csv_name = data_store.get("csv_filename", "")
        summary_lines.append(
            f"✓ Fichier de données DÉJÀ CHARGÉ (session {ref}"
            + (f", fichier : {csv_name}" if csv_name else "")
            + "). NE PAS demander de fichier à l'utilisateur."
        )
        if data_store.get("column_mapping"):
            summary_lines.append(f"✓ Colonnes mappées : {list(data_store['column_mapping'].values())}")

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
    computed = [label for key, label in key_labels.items() if data_store.get(key)]
    if computed:
        summary_lines.append("\n## Résultats déjà calculés")
        summary_lines.extend(f"  {l}" for l in computed)

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
    Cherche <NEED_DATA: ...> UNIQUEMENT dans les messages postérieurs au dernier
    <BUILD_DONE>. Un BUILD_DONE invalide tous les NEED_DATA antérieurs.
    Retourne [] si aucun signal valide trouvé.
    """
    import re

    # Parcourir à rebours et s'arrêter dès qu'on voit BUILD_DONE
    for msg in reversed(messages):
        content = getattr(msg, "content", "") or ""
        if "<BUILD_DONE>" in content or "<HANDOFF_WRITER>" in content:
            return []  # BUILD_DONE invalide tout NEED_DATA antérieur
        m = re.search(r"<NEED_DATA:\s*([^>]+)>", content)
        if m:
            fields = [f.strip() for f in m.group(1).split(",") if f.strip()]
            # Ignorer les champs vagues ("inconnues", "unknown", etc.)
            known_prefixes = {
                "exposure", "qx", "smooth", "diag", "valid", "bench",
                "total_", "cohort_", "age_", "observation_", "num_", "smr",
            }
            real_fields = [
                f for f in fields
                if any(f.lower().startswith(p) for p in known_prefixes)
            ]
            return real_fields if real_fields else []
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

    messages_list = state.get("messages") or []

    # ── 1. WRITE_DONE : cycle complet — nettoyer et terminer ─────────────────
    last_write_done = next(
        (True for m in reversed(messages_list)
         if "<WRITE_DONE" in (getattr(m, "content", "") or "")),
        False,
    )
    if last_write_done:
        data_store.pop("_intent", None)
        data_store.pop("_need_data_attempts", None)
        return {
            "messages": [],
            "events":   [{"type": "agent_switch", "agent": "MasterAgent"},
                         {"type": "done"}],
            "data_store": data_store,
        }

    # ── 2. BUILD_DONE : routing déterministe vers Writer ─────────────────────
    last_build_done = next(
        (True for m in reversed(messages_list)
         if "<BUILD_DONE>" in (getattr(m, "content", "") or "")
         or "<HANDOFF_WRITER>" in (getattr(m, "content", "") or "")),
        False,
    )
    if last_build_done and all(data_store.get(k) for k in _MINIMUM_BUILDER_KEYS):
        data_store.pop("_need_data_attempts", None)
        intent = data_store.get("_intent", "build_and_write")
        if intent in ("build_and_write", "write_only"):
            return {
                "messages": [],
                "events":   [{"type": "agent_switch", "agent": "MasterAgent"},
                             {"type": "message",
                              "content": "Calculs terminés — lancement du WriterAgent."}],
                "active_agent": "writer",
                "data_store":   data_store,
            }
        # build_only : données prêtes, on termine
        return {
            "messages": [],
            "events":   [{"type": "agent_switch", "agent": "MasterAgent"},
                         {"type": "message", "content": "Calculs actuariels terminés."},
                         {"type": "done"}],
            "data_store": data_store,
        }

    # ── 3. NEED_DATA : re-router vers Builder avec liste précise ─────────────
    _STUDY_PLAN_FIELDS = {
        "observation_period_years", "num_observation_years",
        "observation_start_date", "observation_end_date",
        "study_objective", "smoothing_algorithm", "baseline_regulatory_table",
    }
    need_data_fields = _detect_need_data(messages_list)
    if need_data_fields:
        builder_fields = [f for f in need_data_fields if f not in _STUDY_PLAN_FIELDS]
        attempts = data_store.get("_need_data_attempts", 0)
        if builder_fields and attempts < 2:
            already_done = [k for k in _ALL_BUILDER_KEYS if data_store.get(k)]
            from langchain_core.messages import HumanMessage
            instr = (
                f"Le WriterAgent manque des données : {builder_fields}. "
                + (f"NE recalcule PAS : {already_done}. " if already_done else "")
                + "Lance uniquement les outils manquants puis émet <BUILD_DONE>."
            )
            data_store["_need_data_attempts"] = attempts + 1
            return {
                "messages":     [HumanMessage(content=instr)],
                "events":       [{"type": "agent_switch", "agent": "MasterAgent"},
                                 {"type": "message",
                                  "content": f"[MasterAgent] Données manquantes : {builder_fields} "
                                             f"— tentative {attempts + 1}/2."}],
                "active_agent": "builder",
                "data_store":   data_store,
            }
        elif not builder_fields:
            print(f"[MasterAgent] NEED_DATA study_plan uniquement ({need_data_fields}) "
                  "— mode dégradé.", file=sys.stderr)

    # ── 4. Désambiguation ────────────────────────────────────────────────────
    if not data_store.get("_disambiguation_done"):
        last_human = next(
            (getattr(m, "content", "") for m in reversed(messages_list)
             if getattr(m, "type", "") == "human"), "",
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
                    "events":   [{"type": "agent_switch", "agent": "MasterAgent"},
                                 {"type": "disambiguation_required",
                                  "task_type":                 disam.get("task_type"),
                                  "needs_column_mapping":      disam.get("needs_column_mapping", False),
                                  "needs_value_mapping":       disam.get("needs_value_mapping", False),
                                  "needs_form":                disam.get("needs_form", False),
                                  "column_mapping_suggestion": disam.get("column_mapping_suggestion", {}),
                                  "value_mapping_suggestion":  disam.get("value_mapping_suggestion", {}),
                                  "df_columns":                disam.get("df_columns", []),
                                  "form_fields":               disam.get("form_fields", [])}],
                    "data_store": data_store,
                }
            elif disam["status"] == "unclear":
                return {
                    "messages": [],
                    "events":   [{"type": "agent_switch", "agent": "MasterAgent"},
                                 {"type": "message",
                                  "content": disam.get("message",
                                             "Je n'ai pas bien compris. Pouvez-vous préciser ?")}],
                    "data_store": data_store,
                }
            # Normalisation automatique des records si les deux mappings
            # sont confirmés (US-14). No-op si l'un des drapeaux manque.
            try:
                from agents.master.disambiguation import maybe_normalize_records
                df_json_for_norm: str | None = None
                if dataset_ref:
                    try:
                        from session.dataset_store import DatasetStore
                        df_loaded = DatasetStore.load_by_session(dataset_ref)
                        if df_loaded is not None:
                            df_json_for_norm = df_loaded.to_json(orient="split")
                    except Exception:
                        pass
                norm_updates = maybe_normalize_records(data_store, df_json_for_norm)
                if norm_updates:
                    data_store.update(norm_updates)
            except Exception as exc:
                print(f"[MasterAgent] normalize error: {exc}", file=sys.stderr)

            data_store["_disambiguation_done"] = True

    # ── 5. Classification d'intention (remplace GO_BUILD / GO_WRITE LLM) ─────
    last_human = next(
        (getattr(m, "content", "") for m in reversed(messages_list)
         if getattr(m, "type", "") == "human"), "",
    )
    if not last_human:
        return {"messages": [], "events": [], "data_store": data_store}

    classification = _classify_intent(last_human, data_store, dataset_ref)
    intent = classification.get("intent", "unclear")
    reply  = classification.get("reply", "")
    data_store["_intent"] = intent

    # Extraire study_plan si pas encore fait
    if not data_store.get("study_plan"):
        from agents.mortality.agents.mortality_node import _to_openai_dict, sanitize_openai_messages
        raw_msgs = messages_list[-20:]
        msgs_dict = sanitize_openai_messages(
            [{"role": "system", "content": ""}] + [_to_openai_dict(m) for m in raw_msgs]
        )
        extracted = _extract_study_plan_from_history(msgs_dict)
        if extracted:
            data_store["study_plan"] = extracted

    new_events = [{"type": "agent_switch", "agent": "MasterAgent"}]
    if reply:
        new_events.append({"type": "message", "content": reply})

    # ── 6. Routing déterministe basé sur intent + data_store ─────────────────
    from langchain_core.messages import HumanMessage, AIMessage as LCAIMessage

    if intent in ("build_only", "build_and_write"):
        missing_min = [k for k in _MINIMUM_BUILDER_KEYS if not data_store.get(k)]
        if missing_min:
            data_store["_builder_turns"] = 0  # reset compteur safety
            already_done = [k for k in _ALL_BUILDER_KEYS if data_store.get(k)]
            instr = (
                "Lance l'ensemble des calculs actuariels : "
                "exposure, crude_rates, smoothing, diagnostics, validation, benchmarking. "
                + (f"Déjà calculés, NE PAS refaire : {already_done}. " if already_done else "")
                + "Émet <BUILD_DONE> quand tous les calculs sont terminés."
            )
            return {
                "messages":     [HumanMessage(content=instr)],
                "events":       new_events,
                "active_agent": "builder",
                "data_store":   data_store,
            }
        elif intent == "build_and_write":
            # Données déjà complètes → Writer directement
            return {
                "messages":     [],
                "events":       new_events,
                "active_agent": "writer",
                "data_store":   data_store,
            }
        else:
            # build_only + données déjà là
            new_events.append({"type": "done"})
            return {"messages": [], "events": new_events, "data_store": data_store}

    elif intent == "write_only":
        ready, missing_labels = _preflight_writer(data_store)
        if ready:
            return {
                "messages":     [],
                "events":       new_events,
                "active_agent": "writer",
                "data_store":   data_store,
            }
        # Données incomplètes → upgrader en build_and_write
        data_store["_intent"] = "build_and_write"
        already_done = [k for k in _ALL_BUILDER_KEYS if data_store.get(k)]
        instr = (
            f"Avant le rapport, il faut calculer : {missing_labels}. "
            + (f"Déjà calculés, NE PAS refaire : {already_done}. " if already_done else "")
            + "Lance les outils manquants puis émet <BUILD_DONE>."
        )
        return {
            "messages":     [HumanMessage(content=instr)],
            "events":       new_events,
            "active_agent": "builder",
            "data_store":   data_store,
        }

    elif intent == "question":
        # Appel LLM conversationnel — seul cas où le LLM rédige la réponse
        from agents.mortality.agents.mortality_node import (
            _to_openai_dict, _from_openai_response, sanitize_openai_messages
        )
        from agents.mortality.agents._utils import call_with_retry
        system_prompt = _augment_with_data_store(_build_system_prompt(), data_store, dataset_ref)
        raw_msgs = messages_list[-20:]
        messages = [{"role": "system", "content": system_prompt}]
        messages += [_to_openai_dict(m) for m in raw_msgs]
        messages = sanitize_openai_messages(messages)
        try:
            client = openai.OpenAI()
            response = call_with_retry(client, model="gpt-4o", messages=messages,
                                       tools=None, max_tokens=1500)
            lc_msg = _from_openai_response(response.choices[0].message)
            content = response.choices[0].message.content or ""
            if content:
                new_events.append({"type": "message", "content": content})
            new_events.append({"type": "done"})
            return {"messages": [lc_msg], "events": new_events, "data_store": data_store}
        except Exception as exc:
            new_events.append({"type": "error", "message": str(exc)})
            return {"messages": [], "events": new_events, "data_store": data_store}

    # unclear : demander à l'utilisateur de préciser
    new_events.append({"type": "done"})
    return {"messages": [], "events": new_events, "data_store": data_store}
