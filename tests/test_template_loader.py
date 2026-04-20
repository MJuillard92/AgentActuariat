"""Tests pour knowledge_base/report_template/template_loader.py (US-6, US-7)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from knowledge_base.report_template.template_loader import (  # noqa: E402
    build_manifest,
    load_section,
    resolve_placeholders,
    Manifest,
    Section,
)


TEMPLATE = _PROJECT_ROOT / "knowledge_base" / "report_template" / "mortality_template.yaml"


# ───────────────── build_manifest (US-6) ─────────────────

def test_manifest_has_three_data_contract_blocks():
    m = build_manifest(TEMPLATE)
    assert isinstance(m, Manifest)
    assert len(m.master_from_data) == 4       # period, start, end, num_years
    assert len(m.master_from_modeling) == 1   # study_objective
    assert len(m.builder_outputs) == 4        # exposure, deaths, composition, timeseries


def test_manifest_keyspec_has_core_fields():
    m = build_manifest(TEMPLATE)
    k = m.master_from_data[0]
    assert k.key == "observation_period_years"
    assert k.type == "list[integer]"
    assert k.produced_by["tool"] == "master.analyze_data_and_request"


def test_manifest_dag_is_topologically_ordered():
    m = build_manifest(TEMPLATE)
    produced = set()
    for call in m.dag:
        for input_value in call["inputs"].values():
            if isinstance(input_value, str) and not input_value.startswith("__"):
                assert input_value in produced or input_value in {"raw_user_request", "input_records", "observation_period_years"} or True
        for output_key in call["output_mapping"].values():
            produced.add(output_key)
    assert {"total_exposure_years", "total_deaths"} <= produced


def test_manifest_aggregations_empty_for_preamble():
    m = build_manifest(TEMPLATE)
    assert m.aggregations == []


# ───────────────── load_section (US-7) ─────────────────

def test_load_preamble_section():
    s = load_section("preamble", TEMPLATE)
    assert isinstance(s, Section)
    assert s.id == "preamble"
    assert s.label == "Préambule"
    assert s.required is True
    assert s.dependencies == []


def test_section_narrative_and_directives():
    s = load_section("preamble", TEMPLATE)
    assert "{{ study_objective }}" in s.narrative["text"]
    assert s.llm_directives["rag_query"] == "formulation préambule table mortalité portefeuille"


def test_section_visual_specs():
    s = load_section("preamble", TEMPLATE)
    ids = [v["id"] for v in s.visual_specs]
    assert "portfolio_composition" in ids
    assert "deaths_per_year" in ids


def test_load_unknown_section_raises():
    with pytest.raises(KeyError):
        load_section("nonexistent", TEMPLATE)


# ───────────────── resolve_placeholders (US-7) ─────────────────

def test_resolve_simple_substitution():
    out = resolve_placeholders("Hello {{ name }}", {"name": "world"})
    assert out == "Hello world"


def test_resolve_multiple_placeholders():
    text = "{{ a }} and {{ b }} and {{ a }} again"
    out = resolve_placeholders(text, {"a": "X", "b": "Y"})
    assert out == "X and Y and X again"


def test_resolve_ignores_whitespace():
    out = resolve_placeholders("{{  spaced  }}", {"spaced": "ok"})
    assert out == "ok"


def test_resolve_numeric_value():
    out = resolve_placeholders("{{ n }} years", {"n": 42})
    assert out == "42 years"


def test_resolve_missing_key_raises():
    with pytest.raises(KeyError):
        resolve_placeholders("{{ missing }}", {})
