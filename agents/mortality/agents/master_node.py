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


def _get_builder_keys() -> list[str]:
    """Retourne la liste des clés `builder_outputs` du manifest YAML."""
    from knowledge_base.report_template.template_loader import build_manifest
    manifest = build_manifest()
    return [entry.key for entry in manifest.builder_outputs]


# ── Helpers report_mode → sections → clés requises ──────────────────────────

def _sections_for_mode(
    report_mode: str,
    gender_segmentation: str | None = None,
) -> list[str]:
    """Retourne les section ids actifs pour un `report_mode` donné.

    Lit le YAML via `build_manifest(context=...)`. Si `gender_segmentation` est
    fourni, on filtre aussi les sections sex-specific (unisex vs by_sex).
    """
    from knowledge_base.report_template.template_loader import build_manifest
    ctx: dict = {"report_mode": report_mode}
    if gender_segmentation:
        ctx["gender_segmentation"] = gender_segmentation
    manifest = build_manifest(context=ctx)
    return [s["id"] for s in manifest.sections if s.get("id")]


def _keys_for_sections(section_ids: list[str]) -> list[str]:
    """Collecte les clés data_store consommées par les sections données.

    Parcours : pour chaque section active, extraire
      - les placeholders `{{ key }}` de narrative.text / text_default /
        text_raw_rates + post_table_analysis.few_shot_example
      - les racines de `visual_specs[*].source` (sub-path `a.b.c` → on retient `a`)
    """
    import re
    import yaml as _yaml
    from knowledge_base.report_template.template_loader import DEFAULT_TEMPLATE

    placeholder_re = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

    with open(DEFAULT_TEMPLATE, encoding="utf-8") as f:
        tpl = _yaml.safe_load(f) or {}

    keys: set[str] = set()
    for section in (tpl.get("sections") or []):
        if section.get("id") not in section_ids:
            continue

        # narrative + variantes
        narrative = section.get("narrative") or {}
        for field in ("text", "text_default", "text_raw_rates"):
            text = narrative.get(field) or ""
            keys.update(placeholder_re.findall(text))

        # llm_directives.post_table_analysis.few_shot_example
        directives = section.get("llm_directives") or {}
        pta = directives.get("post_table_analysis") or {}
        fse = pta.get("few_shot_example") or ""
        keys.update(placeholder_re.findall(fse))

        # visual_specs.source (on prend la racine du sub-path).
        # Pour les charts multi-séries, on parcourt aussi `series[].source`.
        for v in (section.get("visual_specs") or []):
            src = v.get("source") or ""
            root = src.split(".")[0] if src else ""
            if root:
                keys.add(root)
            # Charts multi-séries : chaque série a sa propre source
            for s in (v.get("series") or []):
                ssrc = s.get("source") or ""
                sroot = ssrc.split(".")[0] if ssrc else ""
                if sroot:
                    keys.add(sroot)

    # On ne retient que les clés effectivement dans builder_outputs
    # (les placeholders peuvent référencer des master_from_data aussi, qu'on
    # laisse gérer au Master — mais pour le pilotage Builder on ne pousse que
    # les builder_outputs manquantes).
    builder_outputs_keys = set(_get_builder_keys())
    return sorted(keys & builder_outputs_keys)


def _get_required_keys_for_current_mode(data_store: dict) -> list[str]:
    """API utilisée par graph.py pour savoir si le Builder a terminé.

    Lit `data_store["report_mode"]` et `data_store["study_plan"]["gender_segmentation"]`,
    dérive les sections actives, retourne la liste des clés builder_outputs
    requises.
    """
    mode = data_store.get("report_mode", "full_report")
    sp = data_store.get("study_plan") or {}
    gender = sp.get("gender_segmentation") or data_store.get("gender_segmentation")
    sections = _sections_for_mode(mode, gender)
    return _keys_for_sections(sections)


# ── Wrappers locaux (rétro-compat tests + signature mortality-aware) ─────────
#
# Les implémentations sont désormais dans `agents.master.*` (domain-agnostic).
# On conserve ces wrappers minces pour ne pas casser :
#   - les tests qui patchent `mn._classify_intent` via patch.object
#   - le code existant qui appelle `_extract_gender_from_text`
#
# Toute évolution de la logique doit se faire dans les modules `agents.master.*`,
# pas ici.

def _extract_gender_from_text(text: str) -> str | None:
    """Wrapper rétro-compat. Voir `agents.master.extract_gender`."""
    from agents.master.extract_gender import extract_gender_from_text
    return extract_gender_from_text(text)


def _classify_intent(last_human: str, data_store: dict, dataset_ref: str | None) -> dict:
    """Wrapper rétro-compat. Voir `agents.master.classify_intent`.

    Calcule has_data / has_calcs depuis le contexte mortalité (builder_keys
    issus du YAML), puis délègue à la fonction domain-agnostic.
    """
    from agents.master.classify_intent import classify_intent as _generic_classify

    builder_keys = _get_builder_keys()
    has_data  = bool(dataset_ref or data_store.get("_dataset_ref"))
    # `is not None` plutôt que truthy : certaines builder_outputs peuvent être
    # des DataFrames (ex: cleaned_records) et `bool(df)` lève ValueError.
    has_calcs = bool(builder_keys) and all(
        data_store.get(k) is not None for k in builder_keys
    )
    # Contexte déjà tranché (transmis au LLM pour éviter qu'il dise
    # "je n'ai pas d'indication sur X" alors que X est connu).
    sp = data_store.get("study_plan") or {}
    known = {
        "gender_segmentation": sp.get("gender_segmentation")
                                or data_store.get("gender_segmentation"),
        "report_mode":         data_store.get("report_mode"),
        "write":               data_store.get("_write"),
    }
    known = {k: v for k, v in known.items() if v}
    return _generic_classify(
        last_human, has_data=has_data, has_calcs=has_calcs,
        known_context=known or None,
    )


def _preflight_writer(data_store: dict) -> tuple[bool, list[str]]:
    """
    Vérifie que toutes les données nécessaires au WriterAgent sont présentes.
    Retourne (prêt, liste des clés manquantes).
    """
    missing = [key for key in _get_builder_keys() if not data_store.get(key)]
    return (len(missing) == 0, missing)


def _extract_study_plan_from_history(messages: list) -> dict:
    """Wrapper rétro-compat. Voir `agents.master.extract_study_plan`."""
    from agents.master.extract_study_plan import extract_study_plan_from_history
    return extract_study_plan_from_history(messages)


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
        "total_exposure_years":         "✓ Exposition totale (années-personne)",
        "total_deaths":                 "✓ Décès observés",
        "portfolio_composition_by_sex": "✓ Composition par sexe",
        "deaths_by_year_series":        "✓ Décès par année",
        "cox_regression":               "✓ Régression Cox H/F",
        "logit_regression":             "✓ Régression logit",
        "series":                       "✓ Séries temporelles",
        "section_outputs":              "✓ Sections rapport en cours",
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
    # IMPORTANT (cf. Bug #7 BUILD_DONE) : un <WRITE_DONE> n'est valide que si
    # AUCUN nouveau HumanMessage n'a été reçu depuis. Sinon on est sur un
    # nouveau tour utilisateur qui doit passer par classification.
    last_write_done = False
    for m in reversed(messages_list):
        content = getattr(m, "content", "") or ""
        is_human = (
            getattr(m, "type", "") == "human"
            and (getattr(m, "additional_kwargs", None) or {}).get("source") != "master_synthetic"
        )
        if is_human:
            # Un HumanMessage récent invalide tout WRITE_DONE antérieur.
            break
        if "<WRITE_DONE" in content:
            last_write_done = True
            break
    if last_write_done:
        data_store.pop("_intent", None)
        data_store.pop("_need_data_attempts", None)
        # Nettoyage des compteurs de cycle pour ne pas polluer une demande
        # future (sinon la préservation _write se déclencherait à tort).
        data_store.pop("_master_builder_cycles", None)
        data_store.pop("_write_question_asked", None)
        data_store.pop("_questions_asked_this_cycle", None)
        return {
            "messages": [],
            "events":   [{"type": "agent_switch", "agent": "MasterAgent"},
                         {"type": "done"}],
            "data_store": data_store,
        }

    # ── 1b. Accumuler les messages user dans data_store["_user_messages"] ────
    # Sert de source de vérité pour le filtre question_filter (Niveau 2).
    # Ne stocke QUE les vrais HumanMessages user — pas les synthétiques émis
    # par le Master (marqués via additional_kwargs.source="master_synthetic").
    last_real_human = next(
        (m for m in reversed(messages_list)
         if getattr(m, "type", "") == "human"
         and (getattr(m, "additional_kwargs", None) or {}).get("source") != "master_synthetic"),
        None,
    )
    if last_real_human is not None:
        history = data_store.setdefault("_user_messages", [])
        content = getattr(last_real_human, "content", "") or ""
        if content and (not history or history[-1] != content):
            history.append(content)

    # ── 1c. Branche : need_user_input émis par le Builder ────────────────────
    # Le Builder peut émettre un AIMessage avec un marqueur additional_kwargs.
    # need_user_input. Master applique alors le filtre 3-niveaux : study_plan
    # → LLM mini → forward au user.
    from langchain_core.messages import AIMessage as _AIMsg, HumanMessage as _HMsg
    last_ai = next(
        (m for m in reversed(messages_list) if isinstance(m, _AIMsg)),
        None,
    )
    if last_ai is not None:
        from agents.master.question_filter import (
            detect_need_in_message, resolve_builder_question,
        )
        need = detect_need_in_message(last_ai)
        if need:
            # Garde-fou : limite de questions par cycle
            asked = data_store.get("_questions_asked_this_cycle", 0)
            MAX_QUESTIONS = 3
            if asked >= MAX_QUESTIONS:
                default_val = need.get("default")
                sp = data_store.setdefault("study_plan", {})
                sp[need["context_key"]] = default_val
                instr = _HMsg(
                    content=(
                        f"[Master] Limite de {MAX_QUESTIONS} questions atteinte dans ce cycle. "
                        f"Application du default pour '{need.get('context_key')}' : {default_val}."
                    ),
                    additional_kwargs={"source": "master_synthetic"},
                )
                return {
                    "messages":     [instr],
                    "events":       [{"type": "agent_switch", "agent": "MasterAgent"},
                                     {"type": "message",
                                      "content": f"Question '{need.get('context_key')}' "
                                                 f"forcée au default ({default_val}) — "
                                                 f"limite de {MAX_QUESTIONS} questions atteinte."}],
                    "active_agent": "builder",
                    "data_store":   data_store,
                }

            user_msgs = data_store.get("_user_messages") or []
            resolution = resolve_builder_question(need, data_store, user_msgs)
            data_store["_questions_asked_this_cycle"] = asked + 1

            if resolution.decision == "answered":
                # Cache + injection dans Builder
                sp = data_store.setdefault("study_plan", {})
                sp[need["context_key"]] = resolution.value
                instr = _HMsg(
                    content=(
                        f"[Master] Réponse à ta question '{need.get('context_key')}' : "
                        f"{resolution.value} (source: {resolution.source})."
                    ),
                    additional_kwargs={"source": "master_synthetic"},
                )
                return {
                    "messages":     [instr],
                    "events":       [{"type": "agent_switch", "agent": "MasterAgent"},
                                     {"type": "message",
                                      "content": f"Question '{need.get('context_key')}' "
                                                 f"résolue automatiquement (source: {resolution.source})."}],
                    "active_agent": "builder",
                    "data_store":   data_store,
                }
            else:  # forward
                data_store["_pending_need"] = need
                question_msg = _AIMsg(content=need.get("question", "Précision nécessaire."))
                return {
                    "messages":     [question_msg],
                    "events":       [{"type": "agent_switch", "agent": "MasterAgent"},
                                     {"type": "message", "content": need.get("question", "")}],
                    "data_store":   data_store,
                }

    # ── 1d. Si une question pendante existe et user vient de répondre ────────
    # IMPORTANT : tant que _pending_need est set, Master ne doit PAS classifier
    # le message user comme une nouvelle intention. Soit on extrait la réponse,
    # soit on re-pose la question avec un hint. Jamais de fallthrough.
    pending = data_store.get("_pending_need")
    if pending and last_real_human is not None:
        from agents.master.question_filter import extract_user_answer
        last_text = getattr(last_real_human, "content", "") or ""
        ctx_key = pending.get("context_key", "?")

        # ─── Désambiguation méthodes : déléguée à agents.master ──────────
        # Toute la logique (méta-question, branches auto/préciser,
        # inline-parse, fallback LLM, re-ask, enchaînement per-tool) vit
        # dans agents/master/method_choices.py. Ce nœud LangGraph ne
        # fait qu'invoquer le handler et retourner son update d'état.
        rep_mode = data_store.get("report_mode", "full_report")
        if ctx_key == "methods_choice_mode":
            from agents.master.method_choices import handle_methods_choice_response
            return handle_methods_choice_response(
                pending, last_text, data_store, report_mode=rep_mode,
            )
        if ctx_key.startswith("method_"):
            from agents.master.method_choices import handle_per_tool_method_response
            return handle_per_tool_method_response(
                pending, last_text, data_store, report_mode=rep_mode,
            )

        # ─── Cas général (gender, etc.) ──────────────────────────────────
        value = extract_user_answer(last_text, pending)
        if value is not None:
            sp = data_store.setdefault("study_plan", {})
            sp[pending["context_key"]] = value
            data_store.pop("_pending_need", None)
            instr = _HMsg(
                content=(
                    f"[Master] L'utilisateur a répondu '{pending.get('context_key')}' = {value}."
                ),
                additional_kwargs={"source": "master_synthetic"},
            )
            return {
                "messages":     [instr],
                "events":       [{"type": "agent_switch", "agent": "MasterAgent"},
                                 {"type": "message",
                                  "content": f"Réponse '{pending.get('context_key')}' enregistrée : {value}."}],
                "active_agent": "builder",
                "data_store":   data_store,
            }
        # Extract a échoué — re-poser la question avec un hint sans classify
        options = pending.get("options") or []
        options_str = " ou ".join(repr(o) for o in options) if options else "une réponse claire"
        question_msg = _AIMsg(content=(
            f"Je n'ai pas bien compris votre réponse '{last_text}'. "
            f"Pour la question sur '{ctx_key}', "
            f"merci de répondre par {options_str}."
        ))
        return {
            "messages":   [question_msg],
            "events":     [{"type": "agent_switch", "agent": "MasterAgent"},
                           {"type": "message", "content": question_msg.content}],
            "data_store": data_store,
        }

    # ── 2. BUILD_DONE : routing déterministe vers Writer ─────────────────────
    # IMPORTANT : un BUILD_DONE n'est valide que si AUCUN nouveau HumanMessage
    # n'a été reçu depuis. Sinon on est sur un nouveau tour de l'utilisateur,
    # qui doit passer par classification (cas typique : "fais l'analyse sans
    # rapport" puis "finalement, fais-moi le rapport PDF").
    last_build_done = False
    for m in reversed(messages_list):
        content = getattr(m, "content", "") or ""
        is_human = (getattr(m, "type", "") == "human"
                    and (getattr(m, "additional_kwargs", None) or {}).get("source") != "master_synthetic")
        if is_human:
            # Un HumanMessage récent invalide tout BUILD_DONE antérieur.
            break
        if "<BUILD_DONE>" in content or "<HANDOFF_WRITER>" in content:
            last_build_done = True
            break
    # On vérifie les clés attendues pour le mode courant (pas les 10 clés totales).
    _required = _get_required_keys_for_current_mode(data_store) or _get_builder_keys()
    if last_build_done and all(data_store.get(k) for k in _required):
        data_store.pop("_need_data_attempts", None)
        data_store.pop("_master_builder_cycles", None)
        write = data_store.get("_write", "yes")
        if write == "yes":
            return {
                "messages": [],
                "events":   [{"type": "agent_switch", "agent": "MasterAgent"},
                             {"type": "message",
                              "content": "Calculs terminés — lancement du WriterAgent."}],
                "active_agent": "writer",
                "data_store":   data_store,
            }
        # write=no (ou fallback d'un ask non résolu) : données prêtes, on termine.
        return {
            "messages": [],
            "events":   [{"type": "agent_switch", "agent": "MasterAgent"},
                         {"type": "message",
                          "content": "Calculs terminés. Résultats en mémoire — "
                                     "dis-moi si tu veux un rapport."},
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
            already_done = [k for k in _get_builder_keys() if data_store.get(k)]
            from langchain_core.messages import HumanMessage
            instr = (
                f"Le WriterAgent manque des données : {builder_fields}. "
                + (f"NE recalcule PAS : {already_done}. " if already_done else "")
                + "Lance uniquement les outils manquants puis émet <BUILD_DONE>."
            )
            data_store["_need_data_attempts"] = attempts + 1
            return {
                "messages":     [HumanMessage(content=instr, additional_kwargs={"source": "master_synthetic"})],
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
                norm_updates = maybe_normalize_records(
                    data_store, df_json_for_norm, dataset_ref=dataset_ref,
                )
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

    # ── 5a. Court-circuit : réponse à la question PDF en attente ─────────────
    # Si `_write_question_asked=True`, Master a posé la question "voulez-vous
    # un PDF ?" au tour précédent. La réponse de l'utilisateur ("oui"/"non"/
    # "yes"/"no"…) est interprétée DIRECTEMENT — ne pas la passer au classifier
    # qui, sans contexte, la classerait comme `kind=question`.
    if data_store.get("_write_question_asked"):
        ans = (last_human or "").strip().lower()
        YES_TOKENS = {"oui", "yes", "ok", "d'accord", "daccord", "oui svp",
                      "oui merci", "oui!", "carrément", "carrement",
                      "absolument", "bien sûr", "bien sur", "yep", "yeah"}
        NO_TOKENS  = {"non", "no", "nope", "pas maintenant", "non merci",
                      "pas de pdf", "pas de rapport", "sans rapport",
                      "pas tout de suite", "plus tard"}
        # On accepte aussi "oui …" / "non …" en début de phrase
        first_word = ans.split()[0] if ans else ""
        resolved_write: str | None = None
        if ans in YES_TOKENS or first_word in ("oui", "yes", "ok"):
            resolved_write = "yes"
        elif ans in NO_TOKENS or first_word in ("non", "no", "nope"):
            resolved_write = "no"
        # Détection rapport explicite ("oui, fais le rapport", "fais-moi le rapport")
        elif any(kw in ans for kw in ("rapport", "pdf", "document", "redige", "rédige")):
            resolved_write = "yes"
        elif any(kw in ans for kw in ("pas de rapport", "sans rapport", "pas de pdf")):
            resolved_write = "no"

        if resolved_write is not None:
            data_store["_write"] = resolved_write
            data_store.pop("_write_question_asked", None)
            # Extraction parallèle du gender depuis la même phrase : permet à
            # l'utilisateur de combiner la réponse PDF avec le choix de
            # segmentation (ex: "oui, fais le rapport unisex").
            gender_from_text = _extract_gender_from_text(last_human)
            classification = {
                "kind":                "task",
                "write":               resolved_write,
                "report_mode":         data_store.get("report_mode", "full_report"),
                "gender_segmentation": gender_from_text,   # None | "unisex" | "by_sex"
                "confidence":          1.0,                # décision déterministe
                "intent":              "build_and_write" if resolved_write == "yes" else "build_only",
                "reply":               ("D'accord, je lance les calculs avec rapport."
                                        if resolved_write == "yes"
                                        else "D'accord, je lance les calculs sans rapport."),
            }
        else:
            classification = _classify_intent(last_human, data_store, dataset_ref)
    else:
        classification = _classify_intent(last_human, data_store, dataset_ref)
    intent       = classification.get("intent", "unclear")
    reply        = classification.get("reply", "")
    kind         = classification.get("kind", "task")
    write        = classification.get("write", "ask")
    report_mode  = classification.get("report_mode", "full_report")
    # Défaut 1.0 : mocks de test n'incluent pas confidence, on les considère
    # comme fiables par défaut (compat ascendante).
    confidence   = classification.get("confidence", 1.0)
    reasoning    = classification.get("reasoning", "")
    # On stocke la confidence dans la classification pour `is_confident`
    classification["confidence"] = confidence

    # ── Branche reformulation : confiance LLM insuffisante ──────────────────
    # Si le LLM signale lui-même qu'il n'est pas sûr (confidence < seuil
    # configuré dans llm_models.yaml), on demande à l'utilisateur de
    # reformuler plutôt que de risquer une exécution erronée.
    #
    # Anti-boucle : on n'insiste pas plus de 2 fois. Au 3e tour ambigu, on
    # exécute avec les axes obtenus (fallback pessimiste).
    from agents.master.classify_intent import is_confident
    if not is_confident(classification):
        attempts = data_store.get("_reformulation_attempts", 0)
        MAX_REFORMULATIONS = 2
        if attempts < MAX_REFORMULATIONS:
            data_store["_reformulation_attempts"] = attempts + 1
            from langchain_core.messages import AIMessage as LCAIMessage
            hint = f" ({reasoning})" if reasoning else ""
            q_text = (
                f"Je ne suis pas sûr d'avoir compris votre demande{hint}. "
                "Pourriez-vous reformuler en précisant :\n"
                "• le type d'analyse souhaité (descriptive, taux bruts, taux lissés) ;\n"
                "• si vous voulez un rapport PDF ou juste les calculs ;\n"
                "• une table unisex ou des tables H/F séparées ?"
            )
            return {
                "messages":   [LCAIMessage(content=q_text)],
                "events":     [{"type": "agent_switch", "agent": "MasterAgent"},
                               {"type": "message", "content": q_text}],
                "data_store": data_store,
            }
        # Compteur épuisé : on exécute quand même en mode dégradé.
        data_store["_reformulation_attempts"] = 0
    else:
        # Classification fiable : reset le compteur (on est de nouveau au vert).
        data_store.pop("_reformulation_attempts", None)

    # Stocker les 3 axes + l'alias legacy
    # Préservation des axes _write et report_mode contre une rétrogradation
    # accidentelle en milieu de cycle (ex: user répond "ok", classify
    # retourne write=ask + report_mode=full_report par défaut, alors que
    # l'utilisateur avait précédemment précisé yes/raw_rates).
    # Règle : si un cycle est en cours et que la nouvelle classification est
    # AMBIGUË (write=ask), on conserve les axes antérieurs. Les changements
    # EXPLICITES (write=yes/no) restent toujours pris en compte.
    prev_write = data_store.get("_write")
    prev_report_mode = data_store.get("report_mode")
    cycle_in_progress = data_store.get("_master_builder_cycles", 0) >= 1
    classify_ambiguous = (write == "ask")
    if classify_ambiguous and cycle_in_progress:
        if prev_write in ("yes", "no"):
            write = prev_write
        if prev_report_mode in ("full_report", "raw_rates", "description"):
            report_mode = prev_report_mode

    data_store["_intent"]      = intent
    data_store["_kind"]        = kind
    data_store["_write"]       = write
    data_store["report_mode"]  = report_mode

    # ── Propagation du gender_segmentation détecté par le classifier ─────────
    # Le classifier LLM retourne désormais gender_segmentation comme 4e axe
    # (None / "unisex" / "by_sex"). Si la valeur n'est pas déjà fixée en
    # session, on l'adopte. Sinon Master posera la question via _pending_need.
    sp_now = data_store.get("study_plan") or {}
    if not (sp_now.get("gender_segmentation") or data_store.get("gender_segmentation")):
        gender_from_llm = classification.get("gender_segmentation")
        if gender_from_llm in ("unisex", "by_sex"):
            sp_now = data_store.setdefault("study_plan", {})
            sp_now["gender_segmentation"]   = gender_from_llm
            data_store["gender_segmentation"] = gender_from_llm

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

    # ── 6. Routing déterministe basé sur kind + write + report_mode ─────────
    from langchain_core.messages import HumanMessage, AIMessage as LCAIMessage

    # Branche "question" : conversation, aucun agent
    if kind == "question":
        # (suit le bloc historique `elif intent == "question"` ci-dessous)
        intent = "question"

    if intent in ("build_only", "build_and_write"):
        # ── Désambiguation write=ask AVANT de lancer le Builder ──────────────
        # Objectif : ne pas exécuter un pipeline coûteux si l'utilisateur n'est
        # pas sûr de vouloir un rapport. On pose la question UNE FOIS.
        if write == "ask" and not data_store.get("_write_question_asked"):
            data_store["_write_question_asked"] = True
            q = "Voulez-vous que je génère un rapport PDF à la fin des calculs ?"
            return {
                "messages":     [LCAIMessage(content=q)],
                "events":       new_events + [{"type": "message", "content": q}],
                "data_store":   data_store,
            }

        # ── Désambiguation gender_segmentation AVANT le Builder ─────────────
        # Master doit savoir si l'analyse est unisex (table agrégée) ou by_sex
        # (tables H/F séparées). Si la valeur n'est toujours pas connue
        # (l'extraction prématurée plus haut n'a rien trouvé), on demande à
        # l'user via le pattern need_user_input.
        sp = data_store.get("study_plan") or {}
        gender = sp.get("gender_segmentation") or data_store.get("gender_segmentation")
        if gender is None and not data_store.get("_pending_need"):
            data_store["_pending_need"] = {
                "context_key": "gender_segmentation",
                "question":    "Voulez-vous une table agrégée (unisex) ou des tables séparées par sexe (H/F) ?",
                "options":     ["unisex", "by_sex"],
                "default":     "unisex",
            }
            q_msg = LCAIMessage(content=data_store["_pending_need"]["question"])
            return {
                "messages":     [q_msg],
                "events":       new_events + [{"type": "message",
                                                "content": data_store["_pending_need"]["question"]}],
                "data_store":   data_store,
            }

        # ── Désambiguation choix de méthodes (délégué à agents.master) ──────
        from agents.master.method_choices import build_methods_meta_pending_need
        meta_pn = build_methods_meta_pending_need(
            report_mode, gender, data_store.get("study_plan"),
        )
        if (meta_pn
                and not data_store.get("_pending_need")
                and not data_store.get("_methods_question_done")):
            data_store["_pending_need"] = meta_pn
            q_msg = LCAIMessage(content=meta_pn["question"])
            return {
                "messages":   [q_msg],
                "events":     new_events + [{"type": "message",
                                              "content": meta_pn["question"]}],
                "data_store": data_store,
            }

        # ── Sections actives dérivées de report_mode + gender_segmentation ──
        active_sections = _sections_for_mode(report_mode, gender)
        required_keys = _keys_for_sections(active_sections)
        already_done = [k for k in required_keys if data_store.get(k)]
        missing_keys = [k for k in required_keys if not data_store.get(k)]

        # ── Compteur cumulatif Master ↔ Builder (filet anti-boucle) ─────────
        if missing_keys:
            cycles = data_store.get("_master_builder_cycles", 0) + 1
            data_store["_master_builder_cycles"] = cycles
            # Limite portée à 6 : le mode full_report nécessite ~5 batchs
            # de tools (descriptifs → crude_rates → smoothing → validation
            # → aggregation_deciles). 3 cycles bloquait avant convergence.
            if cycles > 6:
                new_events.append({
                    "type":    "message",
                    "content": (f"[MasterAgent] {cycles} cycles sans convergence — arrêt. "
                                f"Manquantes : {missing_keys}"),
                })
                new_events.append({"type": "done"})
                # IMPORTANT : reset active_agent à "master" pour que le routeur
                # graph.py:_should_continue_master retourne END (sinon il
                # voit active_agent="builder" de l'étape précédente et
                # reboucle indéfiniment).
                return {
                    "messages":     [],
                    "events":       new_events,
                    "active_agent": "master",
                    "data_store":   data_store,
                }

            # Détecter si l'intention de l'utilisateur est suffisamment
            # explicite pour skipper la phase de confirmation (cf. step3_client_
            # communication.md). Critère : write ∈ {yes, no} ET report_mode posé.
            # write="no" est aussi explicite : "calcule sans rapport" est aussi
            # clair que "fais-moi le rapport".
            intent_explicit = (
                write in ("yes", "no")
                and report_mode in ("full_report", "raw_rates", "description")
            )
            skip_confirm_line = (
                "L'utilisateur a déjà été explicite : NE demande PAS confirmation, "
                "ne ré-explique PAS le dictionnaire de données (déjà confirmé via "
                "le mapping UI), lance directement les tools nécessaires.\n"
                if intent_explicit else ""
            )
            # Hint pour by_sex : si serie_h/serie_f ou distribution_list_h/_f
            # sont attendus, appeler time_series + age_distribution avec by_sex=True
            # (un seul appel suffit, le tool retourne serie + serie_h + serie_f).
            hint_by_sex = ""
            if (gender == "by_sex"
                or any(k.endswith(("_h", "_f")) for k in missing_keys)):
                hint_by_sex = (
                    "Pour produire serie_h/serie_f : appelle "
                    "`statistical_analysis.time_series` UNE SEULE FOIS avec "
                    "params {by_sex: true}. Idem pour ages.distribution_list_h/_f : "
                    "`statistical_analysis.age_distribution` avec {by_sex: true}.\n"
                )
            # Hints explicites pour les clés réclamant un tool spécifique
            # rarement choisi spontanément par le LLM.
            extra_hints = []
            if "qx_deciles_table" in missing_keys:
                extra_hints.append(
                    "Pour `qx_deciles_table` : appelle `aggregation.exposure_deciles` "
                    "(prérequis : qx_table + smoothed_table déjà dans data_store).")
            if "ci_table" in missing_keys:
                extra_hints.append(
                    "Pour `ci_table` : appelle `builder.validation` avec params "
                    "{function_name: confidence_intervals, qx_col: q_x_lisse} "
                    "(prérequis : smoothed_table).")
            hint_extra = ("\n".join(extra_hints) + "\n") if extra_hints else ""

            instr = (
                f"Mode de rapport : {report_mode}\n"
                f"Sections actives : {active_sections}\n"
                f"Déjà produit (NE PAS relancer) : {already_done}\n"
                f"Reste à produire : {missing_keys}\n"
                + skip_confirm_line
                + hint_by_sex
                + hint_extra
                + "Émets <BUILD_DONE> quand toutes les clés ci-dessus sont dans le data_store."
            )
            return {
                "messages":     [HumanMessage(content=instr, additional_kwargs={"source": "master_synthetic"})],
                "events":       new_events,
                "active_agent": "builder",
                "data_store":   data_store,
            }

        # ── Toutes les clés sont présentes : route selon write ──────────────
        if write == "yes":
            return {
                "messages":     [],
                "events":       new_events,
                "active_agent": "writer",
                "data_store":   data_store,
            }
        # write == "no" → done. L'utilisateur peut demander un rapport plus tard,
        # le data_store est persisté et Master routera direct vers le Writer.
        new_events.append({
            "type":    "message",
            "content": "Calculs terminés. Résultats en mémoire — dis-moi si tu veux un rapport.",
        })
        new_events.append({"type": "done"})
        return {"messages": [], "events": new_events, "data_store": data_store}

    elif intent == "question":
        # Branche conversationnelle déléguée à agents.master.conversation.
        # Le LLM y a accès à un set restreint de tools (data_inspect,
        # plot_basic, eval_pandas, statistical_analysis.*) — pas aux tools
        # actuariels du Builder.
        from agents.master.conversation import respond_conversationally
        return respond_conversationally(messages_list, data_store, dataset_ref)

    # unclear : demander à l'utilisateur de préciser
    new_events.append({"type": "done"})
    return {"messages": [], "events": new_events, "data_store": data_store}
