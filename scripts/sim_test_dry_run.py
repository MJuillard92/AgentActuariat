"""Niveau 2 — Dry-run du Master sans appel LLM réel.

On mocke `_classify_intent` avec une fonction déterministe basée sur des
mots-clés. Le reste de la cinématique Master tourne pour de vrai :
désambiguation, désamb write=ask, instruction Builder dérivée, compteur
cumulatif, etc.

Aucun appel réseau, aucun coût OpenAI. Coût : 0 €.

Usage :
    python scripts/sim_test_dry_run.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import HumanMessage, AIMessage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────
# Mock déterministe de _classify_intent
# ──────────────────────────────────────────────────────────────────────────

def _fake_classify(last_human: str, data_store: dict, dataset_ref: str | None) -> dict:
    """Classification basique par mots-clés. Pas de LLM."""
    text = (last_human or "").lower()

    # kind
    kind = "question" if (text.startswith("c'est quoi") or text.startswith("comment ")
                          or text.startswith("explique")) else "task"

    # write : règle stricte. ORDRE IMPORTANT : tester "no" AVANT "yes",
    # car "sans rapport" contient le mot "rapport" en sous-chaîne.
    if any(w in text for w in ("sans rapport", "pas de rapport", "pas de pdf",
                               "juste les", "sans rédaction", "pas de document")):
        write = "no"
    elif text.strip().lower() in ("non", "no", "pas tout de suite"):
        write = "no"
    elif any(w in text for w in ("rapport", "pdf", "document", "rédige", "redige")):
        write = "yes"
    elif text.strip().lower() in ("oui", "yes", "ok", "d'accord"):
        write = "yes"
    else:
        write = "ask"

    # report_mode
    if "taux brut" in text or "sans lissage" in text:
        report_mode = "raw_rates"
    elif "descriptif" in text or "description" in text or "résumé" in text:
        report_mode = "description"
    else:
        report_mode = "full_report"

    # legacy intent (pour rétro-compat dans master_node)
    if kind == "question":
        legacy = "question"
    elif write == "yes":
        legacy = "build_and_write"
    elif write == "no":
        legacy = "build_only"
    else:
        legacy = "build_and_write"

    return {
        "kind":        kind,
        "write":       write,
        "report_mode": report_mode,
        "intent":      legacy,
        "reply":       f"[mock] kind={kind}, write={write}, mode={report_mode}",
    }


# ──────────────────────────────────────────────────────────────────────────
# Helpers d'affichage
# ──────────────────────────────────────────────────────────────────────────

def _line(c: str = "─", n: int = 78):
    print(c * n)


def _print_step(turn: int, user_msg: str | None, out: dict):
    """Affiche le résultat d'un appel master_node de façon compacte."""
    print()
    _line("═")
    print(f"  TOUR {turn}" + (f"  | message user : {user_msg!r}" if user_msg else ""))
    _line("═")

    new_msgs = out.get("messages") or []
    if new_msgs:
        for m in new_msgs:
            cls = type(m).__name__
            content = (getattr(m, "content", "") or "")[:200]
            print(f"  → {cls} : {content}")

    active = out.get("active_agent")
    if active:
        print(f"  → active_agent = {active}")

    events = out.get("events") or []
    for ev in events:
        if isinstance(ev, dict):
            etype = ev.get("type")
            if etype in ("done", "agent_switch", "message"):
                print(f"  → event: {etype} | {ev.get('content') or ev.get('agent') or ''}")

    ds = out.get("data_store") or {}
    keys_to_show = (
        "_kind", "_write", "report_mode", "_write_question_asked",
        "_master_builder_cycles", "_disambiguation_done",
    )
    interesting = {k: ds.get(k) for k in keys_to_show if k in ds}
    if interesting:
        print(f"  → data_store : {interesting}")


def _run_master_turn(state: dict, user_msg: str | None) -> dict:
    """Ajoute un message user et lance master_node avec le mock."""
    if user_msg is not None:
        state = dict(state)  # copie
        state["messages"] = list(state.get("messages") or [])
        state["messages"].append(HumanMessage(content=user_msg))
    from agents.mortality.agents import master_node as mn
    with patch.object(mn, "_classify_intent", _fake_classify):
        out = mn.master_node(state)
    # Apply updates to state for next turn
    state = dict(state)
    state["messages"] = list(state.get("messages") or [])
    state["messages"].extend(out.get("messages") or [])
    if "data_store" in out:
        state["data_store"] = out["data_store"]
    if out.get("active_agent"):
        state["active_agent"] = out["active_agent"]
    return state, out


# ──────────────────────────────────────────────────────────────────────────
# Scénarios
# ──────────────────────────────────────────────────────────────────────────

def scenario_1_explicit_yes_raw_rates():
    """User explicite : 'rapport' + 'taux bruts' → write=yes, mode=raw_rates."""
    print("\n\n" + "█" * 78)
    print("█  SCÉNARIO 1 : 'fais-moi le rapport avec les taux bruts'")
    print("█  Attendu : write=yes, mode=raw_rates → route DIRECTE vers Builder")
    print("█" * 78)

    state = {
        "messages": [],
        "data_store": {
            "_disambiguation_done": True,  # on saute la désambiguation
            "study_plan": {"gender_segmentation": "unisex"},
        },
        "dataset_ref": "fake_session",
    }
    state, out = _run_master_turn(state, "fais-moi le rapport avec les taux bruts")
    _print_step(1, "fais-moi le rapport avec les taux bruts", out)


def scenario_2_ask_then_yes():
    """User ambigu → Master pose la question → user dit oui."""
    print("\n\n" + "█" * 78)
    print("█  SCÉNARIO 2 : 'construis-moi une table de mortalité' (ambigu)")
    print("█  Attendu :")
    print("█    Tour 1 : Master demande 'voulez-vous un rapport ?'")
    print("█    Tour 2 : user répond 'oui' → route vers Builder")
    print("█" * 78)

    state = {
        "messages": [],
        "data_store": {
            "_disambiguation_done": True,
            "study_plan": {"gender_segmentation": "unisex"},
        },
        "dataset_ref": "fake_session",
    }
    state, out = _run_master_turn(state, "construis-moi une table de mortalité")
    _print_step(1, "construis-moi une table de mortalité", out)

    state, out = _run_master_turn(state, "oui")
    _print_step(2, "oui", out)


def scenario_3_explicit_no():
    """User explicite : pas de rapport."""
    print("\n\n" + "█" * 78)
    print("█  SCÉNARIO 3 : 'calcule sans rapport'")
    print("█  Attendu : write=no → Builder direct, pas de question")
    print("█" * 78)

    state = {
        "messages": [],
        "data_store": {
            "_disambiguation_done": True,
            "study_plan": {"gender_segmentation": "unisex"},
        },
        "dataset_ref": "fake_session",
    }
    state, out = _run_master_turn(state, "calcule les taux lissés sans rapport")
    _print_step(1, "calcule les taux lissés sans rapport", out)


def scenario_4_question():
    """Question conversationnelle : kind=question."""
    print("\n\n" + "█" * 78)
    print("█  SCÉNARIO 4 : 'c'est quoi le lissage Whittaker ?'")
    print("█  Attendu : kind=question → branche conversationnelle (LLM mock)")
    print("█" * 78)

    state = {
        "messages": [],
        "data_store": {
            "_disambiguation_done": True,
            "study_plan": {"gender_segmentation": "unisex"},
        },
        "dataset_ref": "fake_session",
    }
    state, out = _run_master_turn(state, "c'est quoi le lissage Whittaker ?")
    _print_step(1, "c'est quoi le lissage Whittaker ?", out)


def scenario_5_cycle_limit():
    """3 cycles Master ↔ Builder sans convergence → arrêt forcé."""
    print("\n\n" + "█" * 78)
    print("█  SCÉNARIO 5 : compteur cumulatif _master_builder_cycles > 3")
    print("█  Attendu : Master émet 'done' au lieu de relancer Builder")
    print("█" * 78)

    state = {
        "messages": [],
        "data_store": {
            "_disambiguation_done":         True,
            "_master_builder_cycles":       3,  # déjà 3 cycles
            "study_plan":                   {"gender_segmentation": "unisex"},
        },
        "dataset_ref": "fake_session",
    }
    state, out = _run_master_turn(state, "fais-moi le rapport")
    _print_step(1, "fais-moi le rapport (avec compteur déjà à 3)", out)


def scenario_6_later_report():
    """Calculs déjà faits + user demande rapport → Writer direct, zéro Builder."""
    print("\n\n" + "█" * 78)
    print("█  SCÉNARIO 6 : data_store complet + 'finalement le rapport'")
    print("█  Attendu : missing_keys=[] → Writer direct, pas de Builder")
    print("█" * 78)

    # Simule un data_store déjà complètement rempli (par un tour Builder antérieur)
    fake_full_data_store = {
        "_disambiguation_done":  True,
        "study_plan":            {"gender_segmentation": "unisex"},
        # Toutes les clés builder_outputs (Bloc A) :
        "cleaned_records":  [{"id": 1}, {"id": 2}],
        "exclusion_report": {"initial_count": 1000, "final_count": 950, "rules": []},
        "total_records":    950,
        "total_exposure":   1234.5,
        "total_deaths":     42,
        "segmentations":    {"sexe": [{"valeur": "H", "nb_contrats": 500, "nb_deces": 25,
                                       "pct_contrats": 50.0, "pct_deces": 59.5}]},
        "serie":            [{"annee": 2020, "nb_deces": 15}],
        "serie_h":          [{"annee": 2020, "nb_deces": 9}],
        "serie_f":          [{"annee": 2020, "nb_deces": 6}],
        "ages":             {"distribution_list": [{"tranche": "20-30", "nb_contrats": 100}]},
    }

    state = {
        "messages":    [],
        "data_store":  fake_full_data_store,
        "dataset_ref": "fake_session",
    }
    state, out = _run_master_turn(state, "finalement fais-moi le rapport")
    _print_step(1, "finalement fais-moi le rapport (data_store déjà rempli)", out)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("┌" + "─" * 76 + "┐")
    print("│  Dry-run cinématique Master                                                │")
    print("│  Aucun appel LLM réel — _classify_intent est mocké                         │")
    print("└" + "─" * 76 + "┘")

    scenario_1_explicit_yes_raw_rates()
    scenario_2_ask_then_yes()
    scenario_3_explicit_no()
    scenario_4_question()
    scenario_5_cycle_limit()
    scenario_6_later_report()

    print("\n\n" + "═" * 78)
    print("  Tous les scénarios passés. Lis attentivement les sorties pour valider :")
    print("    - active_agent (builder | writer | None)")
    print("    - data_store flags (_write, report_mode, _write_question_asked, ...)")
    print("    - messages générés par Master (HumanMessage instruction Builder, AIMessage question)")
    print("═" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
