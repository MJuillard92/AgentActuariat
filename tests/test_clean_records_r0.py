"""Tests régression : R0 (dates non parsables) dans clean_records,
+ détection erreur tool récurrente dans master_node.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────
# R0 — Dates non parsables
# ──────────────────────────────────────────────────────────────────────

def test_r0_excludes_invalid_birth_dates():
    """Régression bug terrain : 3 lignes avec date_naissance='0/0/0' qui
    plantaient builder.crude_rates_kaplan_meier. Doivent être exclues
    par R0 avant que les âges NaN ne polluent R2-R6."""
    from tools.preprocessing.clean_records import run
    df = pd.DataFrame({
        "date_naissance": ["01/01/1950", "0/0/0", "00/00/0000",
                           "qqchose",    "01/01/1980"],
        "date_entree":    ["01/01/2000"] * 5,
        "date_sortie":    ["01/01/2010", "01/01/2010", "01/01/2010",
                           "31/12/2999", "01/01/2010"],
        "cause_sortie":   ["deces", "deces", "vivant", "vivant", "deces"],
    })
    res = run(df)
    rules = {r["rule_id"]: r for r in res["exclusion_report"]["rules"]}
    assert "R0" in rules, "R0 absente du rapport d'exclusion"
    assert rules["R0"]["count"] == 3, (
        f"R0 devait capturer les 3 dates invalides, vu : {rules['R0']['count']}"
    )
    assert "non parsables" in rules["R0"]["rule_label"].lower() or \
           "invalides" in rules["R0"]["rule_label"].lower()
    # Vérif : les 3 lignes invalides sont bien retirées (final = 2)
    assert res["exclusion_report"]["final_count"] == 2


def test_r0_does_not_double_count_with_age_rules():
    """Une ligne avec date_naissance invalide doit être comptée UNE FOIS
    en R0, pas re-comptée en R2-R6 (car déjà retirée)."""
    from tools.preprocessing.clean_records import run
    df = pd.DataFrame({
        "date_naissance": ["0/0/0", "01/01/1950"],
        "date_entree":    ["01/01/2000", "01/01/2000"],
        "date_sortie":    ["01/01/2010", "01/01/2010"],
        "cause_sortie":   ["deces", "deces"],
    })
    res = run(df)
    rules = {r["rule_id"]: r["count"] for r in res["exclusion_report"]["rules"]}
    assert rules["R0"] == 1
    assert rules["R2"] == 0
    assert rules["R3"] == 0
    assert rules["R4"] == 0
    assert rules["R5"] == 0
    assert rules["R6"] == 0


def test_r0_keeps_valid_sentinels_2999():
    """La date sentinelle 31/12/2999 (contrat actif) ne doit PAS être
    considérée invalide. Le clipping à obs_end gère ce cas séparément."""
    from tools.preprocessing.clean_records import run
    df = pd.DataFrame({
        "date_naissance": ["01/01/1950", "01/01/1960", "01/01/1970"],
        "date_entree":    ["01/01/2000", "01/01/2000", "01/01/2000"],
        "date_sortie":    ["01/01/2010", "31/12/2999", "31/12/2999"],
        "cause_sortie":   ["deces", "vivant", "vivant"],
    })
    res = run(df)
    rules = {r["rule_id"]: r["count"] for r in res["exclusion_report"]["rules"]}
    # 31/12/2999 est parsable → pas exclu par R0
    assert rules["R0"] == 0
    # Les 3 lignes survivent à toutes les règles
    assert res["exclusion_report"]["final_count"] == 3


# ──────────────────────────────────────────────────────────────────────
# Détection erreur tool récurrente → stop early
# ──────────────────────────────────────────────────────────────────────

def test_master_stops_on_recurring_tool_error(monkeypatch):
    """Régression bug terrain : si le même tool plante 2+ fois,
    Master doit s'arrêter avec un message clair au lieu de continuer
    à boucler jusqu'à 6 cycles."""
    from langchain_core.messages import HumanMessage
    from agents.mortality.agents import master_node as mn

    def _fake_classify(*args, **kwargs):
        return {"kind": "task", "write": "yes", "report_mode": "full_report",
                "intent": "build_and_write", "reply": ""}
    monkeypatch.setattr(mn, "_classify_intent", _fake_classify)

    state = {
        "messages": [HumanMessage(content="lance les taux bruts")],
        "data_store": {
            "_disambiguation_done":     True,
            "_methods_question_done":   True,
            "study_plan":               {"gender_segmentation": "unisex",
                                         "methods_auto":        True},
            # Simuler : crude_rates a planté 2 fois sur la même erreur
            "_call_log": [
                {"tool": "builder", "function_name": "crude_rates",
                 "has_error": True, "result_summary": {"erreur": "KeyError: date_naissance"}},
                {"tool": "builder", "function_name": "exposure",
                 "has_error": False, "result_summary": {"exposure_table": "[100 lignes]"}},
                {"tool": "builder", "function_name": "crude_rates",
                 "has_error": True, "result_summary": {"erreur": "KeyError: date_naissance"}},
            ],
        },
        "dataset_ref": None,
    }
    out = mn.master_node(state)
    events = out.get("events") or []
    # Doit y avoir un event "done" et un message expliquant l'arrêt anticipé
    assert any(e.get("type") == "done" for e in events if isinstance(e, dict))
    stop_msgs = [e.get("content", "") for e in events
                 if isinstance(e, dict) and e.get("type") == "message"]
    joined = " ".join(stop_msgs)
    assert "récurrente" in joined or "boucle" in joined.lower(), (
        f"Pas de message d'arrêt anticipé : {stop_msgs}"
    )
    assert "builder.crude_rates" in joined, "Le tool fautif n'est pas nommé"
    # Pas de routing vers Builder (sinon boucle)
    assert out.get("active_agent") != "builder"
