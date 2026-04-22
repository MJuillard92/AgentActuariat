import pandas as pd
from tools.statistical_analysis.age_distribution import run


def _fixture_df():
    return pd.DataFrame({
        "date_naissance": [f"19{y}-01-01" for y in range(50, 90, 5)],
        "date_entree":    ["2010-01-01"] * 8,
        "sexe":           ["H", "F", "H", "F", "H", "F", "H", "F"],
    })


def test_distribution_list_is_list_of_dicts():
    result = run(_fixture_df())
    assert "distribution_list" in result
    assert isinstance(result["distribution_list"], list)
    for item in result["distribution_list"]:
        assert set(item.keys()) == {"tranche", "nb_contrats"}


def test_distribution_list_matches_distribution_dict():
    result = run(_fixture_df())
    dict_items = result["distribution"].items()
    list_items = [(r["tranche"], r["nb_contrats"]) for r in result["distribution_list"]]
    assert list(dict_items) == list_items


def test_by_sex_produces_distribution_list_h_and_f():
    result = run(_fixture_df(), params={"by_sex": True})
    assert "distribution_list_h" in result
    assert "distribution_list_f" in result
    total_h = sum(r["nb_contrats"] for r in result["distribution_list_h"])
    total_f = sum(r["nb_contrats"] for r in result["distribution_list_f"])
    assert total_h == 4
    assert total_f == 4
