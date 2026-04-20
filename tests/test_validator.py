"""
Tests pour knowledge_base/report_template/validator.py (US-2).

Couvre les checks bloquants et warnings listés dans l'ADR §Validation.
Chaque test construit une fixture YAML + un registry en mémoire.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from knowledge_base.report_template.validator import (  # noqa: E402
    validate_template,
    ValidationReport,
)


# ───────────────── Fixtures : registry minimal et YAML Design 3 ─────────────

@pytest.fixture
def registry() -> dict:
    """Registry en mémoire couvrant les tools référencés dans les YAML de test."""
    return {
        "master.analyze_data_and_request": {
            "inputs":  {"records": "table"},
            "outputs": {
                "period_years":     "list[int]",
                "first_death_year": "int",
                "last_death_year":  "int",
                "n_years":          "int",
            },
            "path": "/fake/master/analyze.py",
        },
        "master.classify_request": {
            "inputs":  {"request": "string"},
            "outputs": {"objective": "string"},
            "path": "/fake/master/classify.py",
        },
        "mortality.compute_exposure": {
            "inputs":  {"records": "table", "period": "list[int]"},
            "outputs": {"cumulative_exposure": "number"},
            "path": "/fake/mortality/exposure.py",
        },
    }


_VALID_YAML = textwrap.dedent("""
    session_inputs:
      - key: raw_user_request
        type: string
        required: true
      - key: input_records
        type: table
        required: true
        shape:
          - {key: date_naissance, type: date, format: "YYYY-MM-DD"}
          - {key: sexe, type: enum, allowed: [H, F]}

    data_contract:
      master_from_data:
        - key: observation_period_years
          type: list[integer]
          produced_by:
            tool: master.analyze_data_and_request
            inputs: {records: input_records}
            output_mapping: {period_years: observation_period_years}

      master_from_modeling:
        - key: study_objective
          type: string
          produced_by:
            tool: master.classify_request
            inputs: {request: raw_user_request}
            output_mapping: {objective: study_objective}

      builder_outputs:
        - key: total_exposure_years
          type: number
          produced_by:
            tool: mortality.compute_exposure
            inputs:
              records: input_records
              period: observation_period_years
            output_mapping: {cumulative_exposure: total_exposure_years}

    sections:
      - id: preamble
        label: "Préambule"
        required: true
        dependencies: []
        narrative:
          text: |
            Objectif : {{ study_objective }}.
            Exposition : {{ total_exposure_years }} années-personne.
        llm_directives:
          tone: "neutre"
        visual_specs: []
""").strip()


def _write(tmp_path: Path, content: str, name: str = "t.yaml") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ───────────────── Happy path ─────────────────

def test_valid_yaml_passes(tmp_path, registry):
    rep = validate_template(_write(tmp_path, _VALID_YAML), registry)
    assert rep.ok, f"Expected no errors, got: {rep.errors}"


def test_report_is_validation_report(tmp_path, registry):
    rep = validate_template(_write(tmp_path, _VALID_YAML), registry)
    assert isinstance(rep, ValidationReport)
    assert hasattr(rep, "errors")
    assert hasattr(rep, "warnings")


# ───────────────── Check 1 : YAML parse ─────────────────

def test_invalid_yaml_reported(tmp_path, registry):
    rep = validate_template(_write(tmp_path, "foo: [unclosed"), registry)
    assert not rep.ok
    assert any("parse" in e.message.lower() or "yaml" in e.message.lower() for e in rep.errors)


# ───────────────── Check 2 : produced_by.tool ∈ registry ─────────────────

def test_unknown_tool_detected(tmp_path, registry):
    bad = _VALID_YAML.replace("master.classify_request", "master.inexistant_tool")
    rep = validate_template(_write(tmp_path, bad), registry)
    assert not rep.ok
    assert any("inexistant_tool" in e.message for e in rep.errors)


# ───────────────── Check 3 : produced_by.inputs ⊆ signature ─────────────────

def test_unknown_input_param_detected(tmp_path, registry):
    bad = _VALID_YAML.replace(
        "inputs: {request: raw_user_request}",
        "inputs: {bogus_param: raw_user_request}",
    )
    rep = validate_template(_write(tmp_path, bad), registry)
    assert not rep.ok
    assert any("bogus_param" in e.message for e in rep.errors)


def test_input_value_references_existing_key(tmp_path, registry):
    bad = _VALID_YAML.replace(
        "inputs: {request: raw_user_request}",
        "inputs: {request: nonexistent_key}",
    )
    rep = validate_template(_write(tmp_path, bad), registry)
    assert not rep.ok
    assert any("nonexistent_key" in e.message for e in rep.errors)


# ───────────────── Check 4 : output_mapping / key ∈ outputs ─────────────────

def test_output_mapping_key_not_in_tool_outputs(tmp_path, registry):
    bad = _VALID_YAML.replace(
        "output_mapping: {objective: study_objective}",
        "output_mapping: {not_a_real_output: study_objective}",
    )
    rep = validate_template(_write(tmp_path, bad), registry)
    assert not rep.ok
    assert any("not_a_real_output" in e.message for e in rep.errors)


# ───────────────── Check 5 : placeholders résolvent ─────────────────

def test_unresolved_placeholder_detected(tmp_path, registry):
    bad = _VALID_YAML.replace(
        "{{ study_objective }}",
        "{{ missing_variable }}",
    )
    rep = validate_template(_write(tmp_path, bad), registry)
    assert not rep.ok
    assert any("missing_variable" in e.message for e in rep.errors)


# ───────────────── Check 6 : type: date a un format ─────────────────

def test_date_without_format_detected(tmp_path, registry):
    bad = _VALID_YAML.replace(
        '{key: date_naissance, type: date, format: "YYYY-MM-DD"}',
        "{key: date_naissance, type: date}",
    )
    rep = validate_template(_write(tmp_path, bad), registry)
    assert not rep.ok
    assert any("format" in e.message.lower() and "date_naissance" in e.message for e in rep.errors)


# ───────────────── Check 7 : type: enum a allowed ─────────────────

def test_enum_without_allowed_detected(tmp_path, registry):
    bad = _VALID_YAML.replace(
        "{key: sexe, type: enum, allowed: [H, F]}",
        "{key: sexe, type: enum}",
    )
    rep = validate_template(_write(tmp_path, bad), registry)
    assert not rep.ok
    assert any("allowed" in e.message.lower() and "sexe" in e.message for e in rep.errors)


# ───────────────── Check 8 : pas de cycle DAG ─────────────────

def test_cycle_in_dag_detected(tmp_path, registry):
    """total_exposure_years dépend de observation_period_years qui dépend... de total_exposure_years."""
    bad = _VALID_YAML.replace(
        "inputs: {records: input_records}\n        output_mapping: {period_years: observation_period_years}",
        "inputs: {records: total_exposure_years}\n        output_mapping: {period_years: observation_period_years}",
    )
    rep = validate_template(_write(tmp_path, bad), registry)
    assert not rep.ok
    assert any("cycle" in e.message.lower() for e in rep.errors)


# ───────────────── Check 9 : dependencies vers sections existantes ─────────

def test_invalid_dependency_detected(tmp_path, registry):
    bad = _VALID_YAML.replace(
        "dependencies: []",
        "dependencies: [nonexistent_section]",
    )
    rep = validate_template(_write(tmp_path, bad), registry)
    assert not rep.ok
    assert any("nonexistent_section" in e.message for e in rep.errors)


# ───────────────── Check 10 : unicité de production ─────────────────

def test_duplicate_key_detected(tmp_path, registry):
    """Une clé déclarée deux fois avec produced_by → erreur."""
    extra = (
        "    - key: study_objective\n"
        "      type: string\n"
        "      produced_by:\n"
        "        tool: master.classify_request\n"
        "        inputs: {request: raw_user_request}\n"
        "        output_mapping: {objective: study_objective}\n"
    )
    bad = _VALID_YAML.replace(
        "builder_outputs:\n    - key: total_exposure_years",
        "builder_outputs:\n" + extra + "    - key: total_exposure_years",
    )
    rep = validate_template(_write(tmp_path, bad), registry)
    assert not rep.ok
    assert any("study_objective" in e.message and "duplicat" in e.message.lower() for e in rep.errors)


# ───────────────── Warning : clé jamais consommée ─────────────────

def test_unused_key_produces_warning(tmp_path, registry):
    """Ajouter une clé builder_outputs qui n'est ni placeholder ni input d'un autre tool."""
    extra = (
        "    - key: orphan_key\n"
        "      type: number\n"
        "      produced_by:\n"
        "        tool: mortality.compute_exposure\n"
        "        inputs: {records: input_records, period: observation_period_years}\n"
        "        output_mapping: {cumulative_exposure: orphan_key}\n"
    )
    bad = _VALID_YAML.replace(
        "builder_outputs:\n    - key: total_exposure_years",
        "builder_outputs:\n" + extra + "    - key: total_exposure_years",
    )
    rep = validate_template(_write(tmp_path, bad), registry)
    assert rep.ok, f"Expected ok but got errors: {rep.errors}"
    assert any("orphan_key" in w.message for w in rep.warnings)
