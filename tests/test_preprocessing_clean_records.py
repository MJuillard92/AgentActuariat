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
