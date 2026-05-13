"""Tests du mode conversationnel enrichi du Master.

Couvre :
  - Scope : CONVERSATIONAL_TOOLS ne croise pas BUILDER_TOOLS sur les
    tools actuariels (builder, build_pdf, aggregation, preprocessing).
  - Filtrage : la liste tools passée à OpenAI est bien restreinte.
  - data_inspect : columns / shape / head / value_counts / date_range.
  - plot_basic : génère un PNG.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────
# Scope : Builder vs Conversation tools
# ──────────────────────────────────────────────────────────────────────

def test_builder_tools_do_not_include_conversation():
    """Le Builder n'a pas accès aux tools conversationnels — sinon il
    pourrait court-circuiter le pipeline normé."""
    from agents.mortality.agents.builder_node import BUILDER_TOOLS
    assert "conversation" not in BUILDER_TOOLS


def test_conversational_tools_exclude_actuarial_pipeline():
    """Réciproquement, le mode conversationnel n'a pas accès aux tools
    actuariels (le Builder seul orchestre le pipeline normé)."""
    from agents.master.conversation import CONVERSATIONAL_TOOLS
    forbidden = {"builder", "build_pdf", "aggregation", "preprocessing"}
    assert not (CONVERSATIONAL_TOOLS & forbidden), (
        f"Tools actuariels exposés au mode conversationnel : "
        f"{CONVERSATIONAL_TOOLS & forbidden}"
    )


def test_filtered_openai_tools_respects_whitelist():
    """get_openai_tools() filtré par CONVERSATIONAL_TOOLS ne retourne
    que les tools autorisés."""
    from agents.master.conversation import (
        _filtered_openai_tools, CONVERSATIONAL_TOOLS,
    )
    tools = _filtered_openai_tools()
    names = {t["function"]["name"] for t in tools}
    assert names <= CONVERSATIONAL_TOOLS, (
        f"Tools hors whitelist : {names - CONVERSATIONAL_TOOLS}"
    )
    # Sanity : au moins le namespace conversation est exposé
    assert "conversation" in names or "statistical_analysis" in names


# ──────────────────────────────────────────────────────────────────────
# Tool data_inspect
# ──────────────────────────────────────────────────────────────────────

def _sample_df():
    return pd.DataFrame({
        "age":          [20, 30, 40, 50, 60],
        "sexe":         ["H", "F", "H", "F", "H"],
        "date_sortie":  ["01/01/2010", "15/06/2015", "31/12/2020",
                         "10/03/2018", "01/01/2010"],
    })


def test_data_inspect_columns():
    from tools.conversation.data_inspect import run
    res = run(_sample_df(), {"function_name": "columns"})
    assert "result" in res
    names = [c["name"] for c in res["result"]]
    assert "age" in names and "sexe" in names


def test_data_inspect_shape():
    from tools.conversation.data_inspect import run
    res = run(_sample_df(), {"function_name": "shape"})
    assert res["result"] == {"rows": 5, "cols": 3}


def test_data_inspect_head():
    from tools.conversation.data_inspect import run
    res = run(_sample_df(), {"function_name": "head", "n": 3})
    assert len(res["result"]) == 3
    assert res["result"][0]["age"] == 20


def test_data_inspect_value_counts():
    from tools.conversation.data_inspect import run
    res = run(_sample_df(), {"function_name": "value_counts", "column": "sexe"})
    assert res["result"]["H"] == 3
    assert res["result"]["F"] == 2


def test_data_inspect_date_range():
    from tools.conversation.data_inspect import run
    res = run(_sample_df(), {"function_name": "date_range", "column": "date_sortie"})
    assert "min" in res["result"]
    assert "max" in res["result"]
    assert res["result"]["min"].startswith("2010")
    assert res["result"]["max"].startswith("2020")


# ──────────────────────────────────────────────────────────────────────
# Tool plot_basic
# ──────────────────────────────────────────────────────────────────────

def test_plot_basic_histogram_writes_png(tmp_path, monkeypatch):
    """plot_basic.histogram écrit un PNG accessible."""
    import os
    monkeypatch.chdir(tmp_path)  # redirige tmp/conversation_plots/
    from tools.conversation.plot_basic import run
    res = run(_sample_df(), {"function_name": "histogram", "column": "age", "bins": 5})
    assert "png_path" in res
    assert os.path.exists(res["png_path"]), f"PNG absent : {res['png_path']}"


def test_plot_basic_bar(tmp_path, monkeypatch):
    import os
    monkeypatch.chdir(tmp_path)
    from tools.conversation.plot_basic import run
    res = run(_sample_df(), {"function_name": "bar", "column": "sexe"})
    assert "png_path" in res
    assert os.path.exists(res["png_path"])


def test_plot_basic_unknown_column():
    from tools.conversation.plot_basic import run
    res = run(_sample_df(), {"function_name": "histogram", "column": "absente"})
    assert "erreur" in res
