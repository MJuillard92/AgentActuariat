"""Tests : Master ne doit pas rétrograder _write=yes vers ask en milieu de cycle.

Bug identifié au Niveau 3 du test manuel :
  Si l'utilisateur répond "ok" ou "continue" alors qu'un cycle de calculs
  est déjà en cours (write=yes posé), classify retourne write=ask (pas de
  mot-clé "rapport") et Master repose la question PDF alors qu'elle a déjà
  été tranchée.

Règle (scope #1) : si un cycle est en cours et que classify retourne ask,
on conserve le _write antérieur. Pas de rétrogradation accidentelle.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import HumanMessage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _fake_classify_returns(kind="task", write="ask", report_mode="full_report"):
    """Helper : produit un mock _classify_intent retournant des axes contrôlés."""
    def _f(last_human, data_store, dataset_ref):
        return {
            "kind":        kind,
            "write":       write,
            "report_mode": report_mode,
            "intent":      "build_and_write" if kind == "task" else "question",
            "reply":       "",
        }
    return _f


def test_master_instruction_says_skip_confirmation_when_intent_explicit():
    """Quand l'utilisateur a explicité write=yes et un report_mode non-défaut,
    l'instruction envoyée au Builder doit lui dire de NE PAS demander
    confirmation (sinon UX redondante)."""
    from agents.mortality.agents import master_node as mn

    state = {
        "messages":    [HumanMessage(content="construit les taux bruts et le rapport associé")],
        "data_store":  {
            "_disambiguation_done":     True,
            "_methods_question_done":   True,
            "study_plan":               {"gender_segmentation": "unisex",
                                         "methods_auto":        True},
        },
        "dataset_ref": None,
    }
    with patch.object(mn, "_classify_intent",
                      _fake_classify_returns(write="yes", report_mode="raw_rates")):
        out = mn.master_node(state)

    instr = (out.get("messages") or [None])[0]
    assert instr is not None, "Aucune instruction Builder émise"
    content = (instr.content or "").lower()
    # On veut une mention explicite qui désactive la confirmation préalable
    assert ("ne demande pas confirmation" in content
            or "skip confirmation" in content
            or "lance directement" in content), (
        f"Instruction Builder ne signale pas l'intent explicite (écrirait '"
        f"ne demande pas confirmation' / 'lance directement') : {instr.content[:300]}"
    )


def test_master_preserves_report_mode_when_classify_ambiguous():
    """En milieu de cycle (write=ask), report_mode ne doit pas non plus être
    rétrogradé vers le défaut full_report si le user a posé raw_rates avant."""
    from agents.mortality.agents import master_node as mn

    state = {
        "messages":    [HumanMessage(content="ok continue")],
        "data_store":  {
            "_disambiguation_done":   True,
            "_write":                 "yes",
            "_kind":                  "task",
            "report_mode":            "raw_rates",   # mode non-défaut posé avant
            "_master_builder_cycles": 1,
            "study_plan":             {"gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
    }
    # Le classify ambigu retourne write=ask + report_mode=full_report (défaut)
    with patch.object(mn, "_classify_intent",
                      _fake_classify_returns(write="ask", report_mode="full_report")):
        out = mn.master_node(state)

    assert out["data_store"]["report_mode"] == "raw_rates", (
        f"report_mode rétrogradé en plein cycle : "
        f"{out['data_store'].get('report_mode')!r}"
    )


def test_master_cleans_cycle_counter_on_write_done():
    """Sur réception de <WRITE_DONE>, Master doit nettoyer
    `_master_builder_cycles` pour que la session suivante reparte sur 0
    (sinon la préservation kicks in à tort)."""
    from agents.mortality.agents import master_node as mn
    from langchain_core.messages import AIMessage

    state = {
        "messages":    [AIMessage(content="<WRITE_DONE>")],
        "data_store":  {
            "_disambiguation_done":   True,
            "_master_builder_cycles": 2,
            "_intent":                "build_and_write",
            "_write":                 "yes",
        },
        "dataset_ref": None,
    }
    out = mn.master_node(state)
    assert "_master_builder_cycles" not in out["data_store"], (
        f"_master_builder_cycles non nettoyé après WRITE_DONE : "
        f"{out['data_store'].get('_master_builder_cycles')!r}"
    )


def test_master_allows_explicit_downgrade_yes_to_no_mid_cycle():
    """L'utilisateur doit pouvoir rétrograder explicitement de yes vers no
    en milieu de cycle. La préservation contre `ask` ne doit PAS bloquer
    une intention claire de stopper le rapport."""
    from agents.mortality.agents import master_node as mn

    state = {
        "messages":    [HumanMessage(content="finalement pas de rapport")],
        "data_store":  {
            "_disambiguation_done":   True,
            "_write":                 "yes",   # initialement explicite
            "_kind":                  "task",
            "report_mode":            "full_report",
            "_master_builder_cycles": 1,        # cycle en cours
            "study_plan":             {"gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
    }

    # classify détecte le refus explicite → write="no"
    with patch.object(mn, "_classify_intent", _fake_classify_returns(write="no")):
        out = mn.master_node(state)

    assert out["data_store"]["_write"] == "no", (
        f"Rétrogradation explicite bloquée : attendu 'no', obtenu "
        f"{out['data_store'].get('_write')!r}"
    )


def test_master_allows_explicit_no_to_yes_mid_cycle():
    """Inverse : l'utilisateur dit 'non' puis 'finalement oui' → doit pouvoir
    repasser à yes en plein cycle."""
    from agents.mortality.agents import master_node as mn

    state = {
        "messages":    [HumanMessage(content="finalement je veux un rapport")],
        "data_store":  {
            "_disambiguation_done":   True,
            "_write":                 "no",
            "_kind":                  "task",
            "report_mode":            "full_report",
            "_master_builder_cycles": 1,
            "study_plan":             {"gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
    }
    with patch.object(mn, "_classify_intent", _fake_classify_returns(write="yes")):
        out = mn.master_node(state)

    assert out["data_store"]["_write"] == "yes"


def test_master_preserves_write_yes_when_classify_returns_ask_mid_cycle():
    """En milieu de cycle (write=yes déjà posé), un classify=ask ne doit pas
    rétrograder write."""
    from agents.mortality.agents import master_node as mn

    state = {
        "messages":    [HumanMessage(content="continue stp")],
        "data_store":  {
            "_disambiguation_done":   True,
            "_write":                 "yes",   # déjà tranché au tour précédent
            "_kind":                  "task",
            "report_mode":            "full_report",
            "_master_builder_cycles": 1,        # cycle en cours
            "study_plan":             {"gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
    }

    with patch.object(mn, "_classify_intent", _fake_classify_returns(write="ask")):
        out = mn.master_node(state)

    # Le data_store final doit conserver write=yes
    assert out["data_store"]["_write"] == "yes", (
        f"_write rétrogradé : attendu 'yes' (préservé), obtenu "
        f"{out['data_store'].get('_write')!r}"
    )

    # Master ne doit PAS avoir reposé la question PDF
    assert not out["data_store"].get("_write_question_asked"), (
        "Master a posé _write_question_asked=True alors qu'on est en milieu de cycle"
    )
