"""Tests pour tools/master/classify_request.py (US-9)."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.master.classify_request import run  # noqa: E402


def test_run_returns_objective_key():
    out = run({"request": "Construis-moi une table de mortalité"}, {})
    assert "objective" in out


def test_v1_always_returns_construction_table_mortalite():
    out = run({"request": "n'importe quoi"}, {})
    assert out["objective"] == "construction_table_mortalite"


def test_missing_request_accepts_empty():
    out = run({"request": ""}, {})
    assert out["objective"] == "construction_table_mortalite"


def test_contract_discoverable_by_registry():
    from knowledge_base.report_template.tool_registry import build_registry
    registry = build_registry(_PROJECT_ROOT / "tools")
    assert "master.classify_request" in registry
    spec = registry["master.classify_request"]
    assert "request" in spec["inputs"]
    assert "objective" in spec["outputs"]


def test_classify_detects_by_sex_mode():
    result = run({"request": "Construis-moi une table H/F"}, {})
    assert result["gender_mode"] == "by_sex"


def test_classify_detects_unisex_mode_by_default():
    result = run({"request": "Construis-moi une table de mortalité sur mon portefeuille"}, {})
    assert result["gender_mode"] == "unisex"


def test_classify_detects_unisex_explicit():
    result = run({"request": "Je veux une table unisex"}, {})
    assert result["gender_mode"] == "unisex"
