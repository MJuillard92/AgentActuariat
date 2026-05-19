"""
HOTFIX-pre-refacto-2026-05 — Bug 9 : sanitize_node_result garantit que
le data_store retourné par un node ne contient pas de clés non-str ou
de valeurs numpy/pandas qui crashent ormsgpack (MemorySaver).

Symptôme prod : "Dict key must a type serializable with OPT_NON_STR_KEYS"
quand un tool écrit {2018: 12, 2019: 15} (int keys) dans data_store.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agents.mortality.agents._utils import msgpack_safe, sanitize_node_result


def test_msgpack_safe_str_keys_dict() -> None:
    """Un dict à clés int doit être converti en dict à clés str."""
    result = msgpack_safe({2018: 12, 2019: 15})
    assert result == {"2018": 12, "2019": 15}
    assert all(isinstance(k, str) for k in result.keys())


def test_msgpack_safe_nested_dict() -> None:
    """Les clés non-str sont converties récursivement."""
    inp = {"year_counts": {2018: 12, 2019: {"H": 5, 2: 7}}}
    out = msgpack_safe(inp)
    assert out == {"year_counts": {"2018": 12, "2019": {"H": 5, "2": 7}}}


def test_msgpack_safe_numpy_scalars() -> None:
    assert msgpack_safe(np.int64(42)) == 42
    assert isinstance(msgpack_safe(np.int64(42)), int)
    assert msgpack_safe(np.float64(3.14)) == 3.14
    assert msgpack_safe(np.bool_(True)) is True


def test_msgpack_safe_numpy_array() -> None:
    arr = np.array([1, 2, 3])
    assert msgpack_safe(arr) == [1, 2, 3]


def test_msgpack_safe_pandas_dataframe() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = msgpack_safe(df)
    assert out == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]


def test_msgpack_safe_nan_inf_to_none() -> None:
    assert msgpack_safe(float("nan")) is None
    assert msgpack_safe(float("inf")) is None
    assert msgpack_safe(float("-inf")) is None


def test_sanitize_node_result_passes_through_dict_keys() -> None:
    """sanitize_node_result doit nettoyer data_store en place."""
    node_result = {
        "messages":   [],
        "events":     [{"type": "x"}],
        "data_store": {"deaths_by_year": {2018: 12, 2019: 15}, "summary": "ok"},
    }
    out = sanitize_node_result(node_result)
    assert out["data_store"]["deaths_by_year"] == {"2018": 12, "2019": 15}
    assert out["data_store"]["summary"] == "ok"


def test_sanitize_node_result_idempotent() -> None:
    """Appliquer 2× ne change rien (les types déjà natifs passent)."""
    inp = {"data_store": {"already_str": {"a": 1, "b": 2}}}
    once = sanitize_node_result(inp)
    twice = sanitize_node_result(once)
    assert once == twice


def test_sanitize_node_result_handles_missing_keys() -> None:
    """data_store / events absents : retour sans crash."""
    assert sanitize_node_result({"messages": []}) == {"messages": []}
    assert sanitize_node_result({}) == {}


def test_sanitize_node_result_preserves_active_agent() -> None:
    """Les autres champs (active_agent, plan_established) sont préservés."""
    inp = {
        "data_store":       {"x": 1},
        "active_agent":     "builder",
        "plan_established": True,
    }
    out = sanitize_node_result(inp)
    assert out["active_agent"] == "builder"
    assert out["plan_established"] is True


def test_sanitize_node_result_non_dict_input() -> None:
    """Si l'input n'est pas un dict, retour tel quel (defensive)."""
    assert sanitize_node_result(None) is None
    assert sanitize_node_result("string") == "string"
    assert sanitize_node_result([1, 2, 3]) == [1, 2, 3]
