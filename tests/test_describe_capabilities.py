"""Tests du tool conversation.describe_capabilities.

Vérifie que le système peut s'auto-décrire de manière structurée :
ce qu'il sait faire, ce qu'il attend de l'utilisateur, ce qu'il produit.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_returns_three_top_level_blocks():
    """Appel sans param → 3 blocs : capabilities, required_inputs, outputs_produced."""
    from tools.conversation.describe_capabilities import run
    out = run(None)
    assert "capabilities"     in out
    assert "required_inputs"  in out
    assert "outputs_produced" in out


def test_lists_exploration_tools():
    """capabilities.exploration contient les tools d'inspection conversationnels."""
    from tools.conversation.describe_capabilities import run
    out = run(None, {"function_name": "capabilities"})
    expl = out["capabilities"].get("exploration", [])
    tool_names = {item["tool"] for item in expl}
    # Au moins ces 3 tools doivent être listés
    assert "conversation.data_inspect" in tool_names
    assert "conversation.plot_basic"   in tool_names
    assert "conversation.eval_pandas"  in tool_names


def test_lists_actuarial_tools():
    """capabilities.calculs_actuariels contient le pipeline Builder."""
    from tools.conversation.describe_capabilities import run
    out = run(None, {"function_name": "capabilities"})
    calc = out["capabilities"].get("calculs_actuariels", [])
    tool_names = {item["tool"] for item in calc}
    # Au moins les piliers du pipeline
    expected = {"builder.crude_rates", "builder.smoothing"}
    assert expected & tool_names, (
        f"Tools actuariels attendus absents — vu : {tool_names}"
    )


def test_lists_report_modes():
    """capabilities.rapports_pdf liste les modes producibles."""
    from tools.conversation.describe_capabilities import run
    out = run(None, {"function_name": "capabilities"})
    modes = out["capabilities"].get("rapports_pdf", [])
    mode_ids = {m.get("mode") for m in modes if m.get("mode")}
    # Les 3 modes principaux sont déclarés dans le YAML
    assert mode_ids >= {"description", "raw_rates", "full_report"}
    # L'axe gender est mentionné
    assert any(m.get("axe") == "gender" for m in modes)


def test_required_inputs_from_user_contains_gender():
    """gender_segmentation a confirm_with_user=True dans le YAML → doit
    apparaître dans required_inputs.from_user."""
    from tools.conversation.describe_capabilities import run
    out = run(None, {"function_name": "required_inputs"})
    user_fields = {e["field"] for e in out["required_inputs"]["from_user"]}
    assert "gender_segmentation" in user_fields


def test_required_inputs_includes_file_description():
    """required_inputs.fichier décrit les colonnes attendues."""
    from tools.conversation.describe_capabilities import run
    out = run(None, {"function_name": "required_inputs"})
    fichier_desc = out["required_inputs"]["fichier"]
    assert "date_naissance" in fichier_desc
    assert "cause_sortie"   in fichier_desc


def test_outputs_includes_known_tables():
    """outputs_produced.tables liste les tableaux YAML connus."""
    from tools.conversation.describe_capabilities import run
    out = run(None, {"function_name": "outputs_produced"})
    table_ids = {t["id"] for t in out["outputs_produced"]["tables"]}
    # Tableaux référencés dans mortality_template.yaml
    assert "portfolio_composition" in table_ids
    assert "smoothing_table"       in table_ids


def test_function_name_filter_capabilities_only():
    """function_name='capabilities' ne retourne PAS required_inputs / outputs."""
    from tools.conversation.describe_capabilities import run
    out = run(None, {"function_name": "capabilities"})
    assert "capabilities"     in out
    assert "required_inputs"  not in out
    assert "outputs_produced" not in out


def test_client_visible_filter():
    """Aucun tool avec client_visible=false ne doit apparaître."""
    from tools.conversation.describe_capabilities import run
    from tools.catalogue import get_catalogue
    out = run(None, {"function_name": "capabilities"})

    cat = get_catalogue()
    hidden = {
        qn for qn, info in (cat.get("tools") or {}).items()
        if info.get("client_visible") is False
    }
    visible = set()
    for group_list in out["capabilities"].values():
        if isinstance(group_list, list):
            for item in group_list:
                t = item.get("tool")
                if t:
                    visible.add(t)
    assert not (visible & hidden), (
        f"Tools hidden exposés à tort : {visible & hidden}"
    )
