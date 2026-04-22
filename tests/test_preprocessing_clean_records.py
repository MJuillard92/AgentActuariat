import pandas as pd
import pytest
from tools.preprocessing.clean_records import run


def _base_df():
    """DataFrame minimal valide (pas d'exclusion) pour usage dans les tests."""
    return pd.DataFrame({
        "date_naissance": ["1970-01-01", "1980-01-01"],
        "date_entree":    ["2010-01-01", "2011-01-01"],
        "date_sortie":    ["2015-01-01", "2016-01-01"],
        "cause_sortie":   ["deces", "autre"],
        "sexe":           ["H", "F"],
    })


def test_r1_removes_sans_objet_contracts():
    df = _base_df()
    df = pd.concat([df, pd.DataFrame({
        "date_naissance": ["1990-01-01"],
        "date_entree":    ["2010-01-01"],
        "date_sortie":    ["2015-01-01"],
        "cause_sortie":   ["sans_objet"],
        "sexe":           ["H"],
    })], ignore_index=True)

    result = run(df)

    assert len(result["cleaned_records"]) == 2
    report = result["exclusion_report"]
    assert report["initial_count"] == 3
    assert report["final_count"] == 2
    r1 = next(r for r in report["rules"] if r["rule_id"] == "R1")
    assert r1["count"] == 1
    assert r1["rule_label"] == "Contrats sans effet (cause de sortie \u00ab\u00a0sans objet\u00a0\u00bb)"


def _row(dn, de, ds, cs="autre", sx="H"):
    return {"date_naissance": dn, "date_entree": de, "date_sortie": ds,
            "cause_sortie": cs, "sexe": sx}


def test_r2_removes_negative_entry_age():
    df = pd.DataFrame([
        _row("2015-01-01", "2010-01-01", "2020-01-01"),  # âge entrée < 0
        _row("1970-01-01", "2010-01-01", "2020-01-01"),  # ok
    ])
    result = run(df)
    assert result["exclusion_report"]["final_count"] == 1
    r2 = next(r for r in result["exclusion_report"]["rules"] if r["rule_id"] == "R2")
    assert r2["count"] == 1


def test_r3_removes_negative_exit_age():
    df = pd.DataFrame([
        _row("1970-01-01", "2010-01-01", "1960-01-01"),  # âge sortie < 0
        _row("1970-01-01", "2010-01-01", "2020-01-01"),  # ok
    ])
    result = run(df)
    r3 = next(r for r in result["exclusion_report"]["rules"] if r["rule_id"] == "R3")
    assert r3["count"] == 1


def test_r4_removes_entry_age_over_100():
    df = pd.DataFrame([
        _row("1900-01-01", "2020-01-01", "2021-01-01"),  # 120 ans à l'entrée
        _row("1970-01-01", "2010-01-01", "2020-01-01"),  # ok
    ])
    result = run(df)
    r4 = next(r for r in result["exclusion_report"]["rules"] if r["rule_id"] == "R4")
    assert r4["count"] == 1


def test_r5_removes_exit_age_over_100():
    df = pd.DataFrame([
        _row("1900-01-01", "1950-01-01", "2005-01-01"),  # 105 à la sortie
        _row("1970-01-01", "2010-01-01", "2020-01-01"),  # ok
    ])
    result = run(df)
    r5 = next(r for r in result["exclusion_report"]["rules"] if r["rule_id"] == "R5")
    assert r5["count"] == 1


def test_r6_removes_exit_before_entry():
    df = pd.DataFrame([
        _row("1970-01-01", "2020-01-01", "2010-01-01"),  # sortie < entrée
        _row("1970-01-01", "2010-01-01", "2020-01-01"),  # ok
    ])
    result = run(df)
    r6 = next(r for r in result["exclusion_report"]["rules"] if r["rule_id"] == "R6")
    assert r6["count"] == 1


def test_rules_are_cumulative_no_double_counting():
    # Une ligne violait plusieurs règles : ne doit être comptée que dans la 1ère déclenchée.
    df = pd.DataFrame([
        # ligne aberrante : sans_objet ET âge entrée > 100
        _row("1900-01-01", "2020-01-01", "2021-01-01", cs="sans_objet"),
        _row("1970-01-01", "2010-01-01", "2020-01-01"),  # ok
    ])
    result = run(df)
    r1 = next(r for r in result["exclusion_report"]["rules"] if r["rule_id"] == "R1")
    r4 = next(r for r in result["exclusion_report"]["rules"] if r["rule_id"] == "R4")
    assert r1["count"] == 1
    assert r4["count"] == 0  # déjà retirée par R1
    assert result["exclusion_report"]["final_count"] == 1
