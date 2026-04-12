"""
report_writer/tests/test_lacune1.py
Tests Lacune #1 — REQUIRED_FIELDS + _initialize_context()

Critères d'acceptation :
  1. Champ manquant dans study_plan → WriterError avec le champ listé
  2. Champ manquant dans data_store → WriterError avec le champ listé
  3. Valeur vide ([], None, {}, "") dans data_store → WriterError
  4. Champ optionnel absent → pas d'erreur, context[field] = None
  5. Tous les champs présents → pas d'erreur, context complet
"""
import pytest

from report_writer.errors import WriterError
from report_writer.report_builder import (
    ReportBuilder,
    REQUIRED_FIELDS_STUDY_PLAN,
    REQUIRED_FIELDS_DATA_STORE,
)

# ─── Fixtures minimales ───────────────────────────────────────────────────────

MOCK_STUDY_PLAN = {
    "study_objective":          "term life death contracts",
    "observation_start_date":   "2018-01-01",
    "observation_end_date":     "2022-12-31",
    "observation_period_years": [2018, 2019, 2020, 2021, 2022],
    "cohort_min_age":           20,
    "cohort_max_age":           80,
    "smoothing_algorithm":      "whittaker_henderson",
    "baseline_regulatory_table": "TH0002",
}

# Génère une table factice {age: valeur} pour les vecteurs par âge
def _age_vec(min_age: int = 20, max_age: int = 80, val: float = 1.0):
    return [{"age": a, "value": val} for a in range(min_age, max_age + 1)]

MOCK_DATA_STORE = {
    # section preamble
    "total_exposure_years": 500_000.0,
    "total_deaths":         3_500,
    # section data_submission
    "initial_record_count":   120_000,
    "final_record_count":     118_000,
    "exposure_by_age_male":   _age_vec(),
    "exposure_by_age_female": _age_vec(),
    "deaths_by_age_male":     _age_vec(val=5.0),
    "deaths_by_age_female":   _age_vec(val=3.0),
    "mean_age_cohort":        52.4,
    "gender_distribution":    {"male": 0.63, "female": 0.37},
    # section obs_vs_modeled
    "observed_deaths_by_age": _age_vec(val=8.0),
    "modeled_deaths_by_age":  _age_vec(val=8.5),
    "ci_lower_by_age":        _age_vec(val=7.0),
    "ci_upper_by_age":        _age_vec(val=10.0),
    "chi_squared_p":          0.042,
    # section regulatory
    "discount_by_age":        _age_vec(val=0.32),
    # section conclusion
    "avg_prudence_ratio":     1.72,
    # section annex
    "final_mortality_table_by_age": _age_vec(val=0.005),
}


def _builder(study_plan=None, data_store=None):
    return ReportBuilder(
        yaml_path="dummy.yaml",
        study_plan=study_plan if study_plan is not None else MOCK_STUDY_PLAN.copy(),
        calculation_agent_output=data_store if data_store is not None else MOCK_DATA_STORE.copy(),
    )


# ─── Test 1 : champ manquant dans study_plan ─────────────────────────────────

def test_missing_study_plan_field():
    """Un champ requis absent de study_plan → WriterError contenant ce champ."""
    sp = MOCK_STUDY_PLAN.copy()
    del sp["cohort_min_age"]

    b = _builder(study_plan=sp)
    with pytest.raises(WriterError) as exc_info:
        b._initialize_context()

    err = exc_info.value
    assert "cohort_min_age" in err.missing_inputs, (
        f"'cohort_min_age' absent mais non listé : {err.missing_inputs}"
    )
    assert err.source_should_be == "study_plan"


# ─── Test 2 : champ manquant dans data_store ─────────────────────────────────

def test_missing_data_store_field():
    """Un champ requis absent de calculation_agent_output → WriterError."""
    ds = MOCK_DATA_STORE.copy()
    del ds["modeled_deaths_by_age"]

    b = _builder(data_store=ds)
    with pytest.raises(WriterError) as exc_info:
        b._initialize_context()

    err = exc_info.value
    assert "modeled_deaths_by_age" in err.missing_inputs, (
        f"'modeled_deaths_by_age' absent mais non listé : {err.missing_inputs}"
    )
    assert err.source_should_be == "calculation_agent_output"


# ─── Test 3 : valeur vide dans data_store ────────────────────────────────────

@pytest.mark.parametrize("empty_val", [None, [], {}, ""])
def test_empty_value_raises_error(empty_val):
    """Une valeur vide (None, [], {}, '') est traitée comme absente → WriterError."""
    ds = MOCK_DATA_STORE.copy()
    ds["observed_deaths_by_age"] = empty_val

    b = _builder(data_store=ds)
    with pytest.raises(WriterError) as exc_info:
        b._initialize_context()

    assert "observed_deaths_by_age" in exc_info.value.missing_inputs


# ─── Test 4 : champ optionnel absent → pas d'erreur ──────────────────────────

def test_optional_field_no_error():
    """Un champ optionnel absent ne lève pas d'erreur ; context[field] = None."""
    ds = MOCK_DATA_STORE.copy()
    # Supprimer des champs optionnels
    ds.pop("cox_hazard_ratio", None)
    ds.pop("logit_slope", None)
    ds.pop("annual_prediction_ratio", None)

    b = _builder(data_store=ds)
    b._initialize_context()  # ne doit pas lever

    assert b.context.get("cox_hazard_ratio") is None
    assert b.context.get("logit_slope") is None
    assert b.context.get("annual_prediction_ratio") is None


# ─── Test 5 : tous les champs présents → pas d'erreur, context complet ───────

def test_complete_data_no_error():
    """Tous les champs requis présents → pas d'erreur, context entièrement rempli."""
    b = _builder()
    b._initialize_context()

    # Vérifier que tous les champs requis sont dans le context
    for field in REQUIRED_FIELDS_STUDY_PLAN:
        assert field in b.context, f"'{field}' absent du context"

    for fields in REQUIRED_FIELDS_DATA_STORE.values():
        for field in fields:
            assert field in b.context, f"'{field}' absent du context"
            assert b.context[field] is not None, f"'{field}' est None alors qu'il devrait être rempli"

    # num_observation_years doit être calculé
    assert b.context["num_observation_years"] == 5


# ─── Test 6 : plusieurs champs manquants → tous listés dans WriterError ───────

def test_multiple_missing_fields_all_reported():
    """Plusieurs champs manquants → WriterError liste TOUS les champs."""
    ds = MOCK_DATA_STORE.copy()
    del ds["modeled_deaths_by_age"]
    del ds["ci_lower_by_age"]
    del ds["discount_by_age"]

    b = _builder(data_store=ds)
    with pytest.raises(WriterError) as exc_info:
        b._initialize_context()

    missing = exc_info.value.missing_inputs
    assert "modeled_deaths_by_age" in missing
    assert "ci_lower_by_age" in missing
    assert "discount_by_age" in missing
    assert len(missing) == 3
