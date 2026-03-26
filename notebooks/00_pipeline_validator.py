"""
00_pipeline_validator.py
========================
Validates the coherence of the actuarial function library by running
the full pipeline chain with synthetic data and verifying every interface.

Usage (standalone):
    python notebooks/00_pipeline_validator.py

Usage (from kernel):
    from notebooks import 00_pipeline_validator as validator  # via importlib
    report = validator.validate_pipeline(verbose=True)

Each step verifies:
  - The function runs without exception
  - The output has the expected type and schema (columns, keys)
  - Value ranges are actuarially plausible
  - The output is compatible with every downstream function that consumes it
"""

import importlib.util
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------
_NOTEBOOKS_DIR = Path(__file__).parent


def _load(mod_file: str):
    """Load a module from notebooks/ by filename (without .py)."""
    path = _NOTEBOOKS_DIR / f"{mod_file}.py"
    spec = importlib.util.spec_from_file_location(mod_file, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Schema checkers
# ---------------------------------------------------------------------------

def _check_df(obj, required_cols, step_name=None, value_checks=None):
    """
    Verify obj is a DataFrame with required_cols.
    value_checks: list of (col, min_val, max_val, allow_nan)
    Returns (ok: bool, detail: str)
    """
    if not isinstance(obj, pd.DataFrame):
        return False, f"Expected DataFrame, got {type(obj).__name__}"
    missing = [c for c in required_cols if c not in obj.columns]
    if missing:
        return False, (
            f"Missing columns: {missing}. "
            f"Available: {list(obj.columns)}"
        )
    if len(obj) == 0:
        return False, "DataFrame is empty"

    warnings = []
    for col, lo, hi, allow_nan in (value_checks or []):
        if col not in obj.columns:
            continue
        series = obj[col].dropna()
        if not allow_nan and obj[col].isna().any():
            warnings.append(f"{col} has NaN values")
        if len(series) > 0:
            if lo is not None and series.min() < lo:
                warnings.append(f"{col} min={series.min():.4g} < {lo}")
            if hi is not None and series.max() > hi:
                warnings.append(f"{col} max={series.max():.4g} > {hi}")

    detail = f"{len(obj)} rows, cols={list(obj.columns)}"
    if warnings:
        detail += " | WARNINGS: " + "; ".join(warnings)
    return True, detail


def _check_dict(obj, required_keys, step_name=None):
    if not isinstance(obj, dict):
        return False, f"Expected dict, got {type(obj).__name__}"
    missing = [k for k in required_keys if k not in obj]
    if missing:
        return False, f"Missing keys: {missing}. Got: {list(obj.keys())}"
    return True, f"keys={list(obj.keys())}"


def _check_bytes(obj, step_name=None):
    if not isinstance(obj, bytes):
        return False, f"Expected bytes (PNG), got {type(obj).__name__}"
    if len(obj) < 100:
        return False, f"PNG too small ({len(obj)} bytes)"
    return True, f"PNG {len(obj):,} bytes"


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

class StepResult:
    def __init__(self, name, status, detail, output=None, warning=None):
        self.name = name          # str
        self.status = status      # "ok" | "fail" | "warning"
        self.detail = detail      # str — what was verified
        self.output = output      # the actual output (passed to next steps)
        self.warning = warning    # str | None

    def __repr__(self):
        icon = {"ok": "✓", "fail": "✗", "warning": "⚠"}.get(self.status, "?")
        w = f" | ⚠ {self.warning}" if self.warning else ""
        return f"  {icon} {self.name:<45} {self.detail}{w}"


def _run_step(name, fn, schema_check_fn, *args, **kwargs):
    """Run fn(*args, **kwargs), apply schema_check_fn to the result."""
    try:
        output = fn(*args, **kwargs)
    except Exception as exc:
        tb = traceback.format_exc().strip().split("\n")[-1]
        return StepResult(name, "fail", f"EXCEPTION: {tb}")

    ok, detail = schema_check_fn(output)
    warning = None
    if "WARNINGS:" in detail:
        parts = detail.split(" | WARNINGS: ", 1)
        detail = parts[0]
        warning = parts[1]
        status = "warning"
    else:
        status = "ok" if ok else "fail"

    return StepResult(name, status, detail, output=output if ok else None,
                      warning=warning)


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def validate_pipeline(n_contracts: int = 800, verbose: bool = True) -> list:
    """
    Run the full actuarial pipeline chain with synthetic data (n_contracts).
    Returns list[StepResult] — one per interface tested.

    Tests every producer→consumer pair to catch column naming mismatches,
    type errors, and range violations before they reach the agent.
    """
    import warnings as _warnings
    _warnings.filterwarnings("ignore")

    print(f"\n{'='*65}")
    print(f"  Pipeline Validator — {n_contracts} synthetic contracts")
    print(f"{'='*65}\n")

    # Load all modules
    try:
        dp  = _load("01_data_preparation")
        exp = _load("02_exposure")
        cr  = _load("03_crude_rates")
        sm  = _load("04_smoothing")
        dg  = _load("05_diagnostics")
        val = _load("06_validation")
        bm  = _load("07_benchmarking")
        viz = _load("08_visualization")
    except Exception as exc:
        print(f"✗ Module loading failed: {exc}")
        return []

    results = []

    def add(result):
        results.append(result)
        if verbose:
            print(repr(result))

    # ── PHASE 1 : Data preparation ───────────────────────────────────────────
    print("PHASE 1 — Data preparation")

    r = _run_step(
        "data_prep.generate_synthetic_data(n)",
        dp.generate_synthetic_data,
        lambda o: _check_df(o, ["id", "date_naissance", "date_entree",
                                 "date_sortie", "cause_sortie", "sexe"]),
        n=n_contracts, seed=42,
    )
    add(r)
    if r.status == "fail":
        print("\n✗ Cannot continue — data generation failed.\n")
        return results
    df_raw = r.output

    r = _run_step(
        "data_prep.clean_data(df_raw)",
        lambda df: dp.clean_data(df)[0],   # returns (df, report)
        lambda o: _check_df(o, ["date_naissance", "date_entree",
                                 "date_sortie", "cause_sortie"]),
        df_raw,
    )
    add(r)
    df_clean = r.output if r.output is not None else df_raw

    r = _run_step(
        "data_prep.compute_ages(df_clean)",
        dp.compute_ages,
        lambda o: _check_df(o, ["age_entree", "age_sortie", "duree_obs_ans"],
                             value_checks=[
                                 ("age_entree",    0, 120, False),
                                 ("age_sortie",    0, 120, False),
                                 ("duree_obs_ans", 0,  50, False),
                             ]),
        df_clean,
    )
    add(r)
    df = r.output if r.output is not None else df_clean

    r = _run_step(
        "data_prep.detect_anomalies(df)",
        dp.detect_anomalies,
        lambda o: _check_dict(o, ["duplicates", "missing_values",
                                   "severity", "recommendations"]),
        df,
    )
    add(r)

    # ── PHASE 2 : Exposure ───────────────────────────────────────────────────
    print("\nPHASE 2 — Exposure")

    r = _run_step(
        "exposure.compute_exposure_by_age(df)",
        lambda d: exp.compute_exposure_by_age(d, age_min=35, age_max=85),
        lambda o: _check_df(o, ["age", "E_x", "D_x", "mu_x", "q_x_brut"],
                             value_checks=[
                                 ("E_x",      0,   None, False),
                                 ("D_x",      0,   None, False),
                                 ("q_x_brut", 0,   1.0,  True),
                             ]),
        df,
    )
    add(r)
    exposure_table = r.output

    r = _run_step(
        "exposure.exposure_summary(exposure_table)",
        exp.exposure_summary,
        lambda o: _check_dict(o, ["total_exposure", "total_deaths",
                                   "pct_low_credibility"]),
        exposure_table,
    )
    add(r)

    # ── PHASE 3 : Crude rates (3 méthodes × 2 entrées) ──────────────────────
    print("\nPHASE 3 — Crude rates")

    CRUDE_COLS = ["age", "E_x", "D_x", "qx"]
    CRUDE_CHECKS = [("qx", 0, 1.0, True)]

    # central depuis exposure_table (q_x_brut)
    r = _run_step(
        "crude_rates.crude_rates_central(exposure_table)",
        cr.crude_rates_central,
        lambda o: _check_df(o, CRUDE_COLS, value_checks=CRUDE_CHECKS),
        exposure_table,
    )
    add(r)
    cr_central = r.output

    # binomial depuis exposure_table
    r = _run_step(
        "crude_rates.crude_rates_binomial(exposure_table)",
        cr.crude_rates_binomial,
        lambda o: _check_df(o, CRUDE_COLS, value_checks=CRUDE_CHECKS),
        exposure_table,
    )
    add(r)

    # KM depuis df individuel
    r = _run_step(
        "crude_rates.crude_rates_kaplan_meier(df, 35, 85)",
        lambda d: cr.crude_rates_kaplan_meier(d, age_min=35, age_max=85),
        lambda o: _check_df(o, ["age", "qx"], value_checks=CRUDE_CHECKS),
        df,
    )
    add(r)

    # ── PHASE 4 : Smoothing (chaque méthode × 2 sources) ────────────────────
    print("\nPHASE 4 — Smoothing (× 2 input types)")

    SMOOTH_KEYS = ["ages", "qx_smoothed", "method", "n_non_monotone_after_40"]
    SMOOTH_CHECK = lambda o: _check_dict(o, SMOOTH_KEYS)

    for source_name, source in [
        ("exposure_table (q_x_brut)", exposure_table),
        ("crude_rates_central (qx)",  cr_central),
    ]:
        for fn_name, fn in [
            ("smooth_whittaker",       sm.smooth_whittaker),
            ("smooth_gompertz",        sm.smooth_gompertz),
            ("smooth_makeham",         sm.smooth_makeham),
            ("smooth_spline",          sm.smooth_spline),
            ("smooth_local_polynomial",sm.smooth_local_polynomial),
        ]:
            r = _run_step(
                f"smoothing.{fn_name}({source_name})",
                fn,
                SMOOTH_CHECK,
                source,
            )
            add(r)

    # Keep Whittaker result from exposure_table for downstream tests
    try:
        wh_result = sm.smooth_whittaker(exposure_table)
        gom_result = sm.smooth_gompertz(exposure_table)
    except Exception:
        wh_result = gom_result = None

    # ── PHASE 5 : Diagnostics ────────────────────────────────────────────────
    print("\nPHASE 5 — Diagnostics")

    r = _run_step(
        "diagnostics.diagnose_credibility(exposure_table)",
        dg.diagnose_credibility,
        lambda o: _check_dict(o, ["low_credibility_ages", "pct_low", "recommendation"]),
        exposure_table,
    )
    add(r)

    r = _run_step(
        "diagnostics.diagnose_monotonicity(qx_array, age_array)",
        lambda: dg.diagnose_monotonicity(
            exposure_table["q_x_brut"].values,
            exposure_table["age"].values,
        ),
        lambda o: _check_dict(o, ["n_violations", "violation_ages", "is_monotone"]),
    )
    add(r)

    if wh_result and gom_result:
        r = _run_step(
            "diagnostics.compare_smoothers({'WH': ..., 'Gom': ...})",
            lambda: dg.compare_smoothers(
                {"Whittaker": wh_result, "Gompertz": gom_result},
                exposure_table,
            ),
            lambda o: (
                isinstance(o, tuple) and len(o) == 2
                and isinstance(o[0], pd.DataFrame)
                and isinstance(o[1], dict),
                f"(DataFrame, dict) — keys={list(o[1].keys()) if isinstance(o, tuple) else '?'}",
            )[0] and (True, f"(DataFrame, dict)") or (False, "Wrong output type"),
        )
        add(r)

    r = _run_step(
        "diagnostics.compute_smr(exposure_table)",
        dg.compute_smr,
        lambda o: _check_dict(o, ["smr_global", "ci_lower", "ci_upper",
                                   "d_observed", "interpretation"]),
        exposure_table,
    )
    add(r)
    smr_result = r.output

    # ── PHASE 6 : Validation ─────────────────────────────────────────────────
    print("\nPHASE 6 — Validation")

    r = _run_step(
        "validation.confidence_intervals(exposure_table)",
        val.confidence_intervals,
        lambda o: _check_df(o, ["age", "qx", "ci_lower", "ci_upper"],
                             value_checks=[
                                 ("ci_lower", 0, 1.0, True),
                                 ("ci_upper", 0, 1.0, True),
                             ]),
        exposure_table,
    )
    add(r)
    ci_result = r.output

    r = _run_step(
        "validation.chi_square_test(exposure_table)",
        val.chi_square_test,
        lambda o: _check_dict(o, ["statistic", "p_value", "conclusion"]),
        exposure_table,
    )
    add(r)

    r = _run_step(
        "validation.prudence_margin(exposure_table)",
        val.prudence_margin,
        lambda o: _check_dict(o, ["prudence_level", "is_prudent"]),
        exposure_table,
    )
    add(r)

    r = _run_step(
        "validation.cox_model(df, covariates=['sexe'])",
        lambda d: val.cox_model(d, covariates=["sexe"]),
        lambda o: _check_dict(o, ["hazard_ratios", "interpretation"]),
        df,
    )
    add(r)

    # ── PHASE 7 : Benchmarking ───────────────────────────────────────────────
    print("\nPHASE 7 — Benchmarking")

    for table_name in ["TH0002", "TF0002", "TD8890", "TPRV93"]:
        r = _run_step(
            f"benchmarking.load_reference_table('{table_name}')",
            lambda n=table_name: bm.load_reference_table(n, "H"),
            lambda o: _check_df(o, ["age", "qx_ref"],
                                 value_checks=[("qx_ref", 0, 1.0, False)]),
        )
        add(r)

    r = _run_step(
        "benchmarking.abatement_factors(exposure_table)",
        bm.abatement_factors,
        lambda o: (
            isinstance(o, tuple) and len(o) == 2
            and isinstance(o[0], pd.DataFrame),
            f"(DataFrame[{len(o[0])} rows], summary_dict)" if isinstance(o, tuple) else "Wrong type",
        ),
        exposure_table,
    )
    add(r)

    r = _run_step(
        "benchmarking.logit_regression(exposure_table)",
        bm.logit_regression,
        lambda o: _check_dict(o, ["a", "b", "r_squared", "interpretation"]),
        exposure_table,
    )
    add(r)

    r = _run_step(
        "benchmarking.export_table(exposure_table, '/tmp/test_export.csv')",
        lambda e: bm.export_table(e, file_path="/tmp/test_validator_export.csv"),
        lambda o: (isinstance(o, str), f"path={o}"),
        exposure_table,
    )
    add(r)

    # ── PHASE 8 : Visualization ──────────────────────────────────────────────
    print("\nPHASE 8 — Visualization")

    smoothed_dict = {}
    if wh_result:
        smoothed_dict["Whittaker"] = wh_result["qx_smoothed"]
    if gom_result:
        smoothed_dict["Gompertz"] = gom_result["qx_smoothed"]

    for fn_name, fn_call in [
        ("plot_exposure_by_age",
         lambda: viz.plot_exposure_by_age(exposure_table)),
        ("plot_deaths_by_age",
         lambda: viz.plot_deaths_by_age(exposure_table)),
        ("plot_crude_vs_smoothed",
         lambda: viz.plot_crude_vs_smoothed(exposure_table, smoothed_dict)),
        ("plot_smr_by_age",
         lambda: viz.plot_smr_by_age(smr_result) if smr_result else b"skip"),
        ("plot_confidence_bands",
         lambda: viz.plot_confidence_bands(exposure_table, ci_result=ci_result)),
        ("plot_observed_vs_expected",
         lambda: viz.plot_observed_vs_expected(exposure_table)),
        ("plot_survival_curve",
         lambda: viz.plot_survival_curve(exposure_table)),
    ]:
        r = _run_step(
            f"visualization.{fn_name}(...)",
            fn_call,
            lambda o: _check_bytes(o, fn_name) if o != b"skip" else (True, "skipped"),
        )
        add(r)

    # ── Summary ──────────────────────────────────────────────────────────────
    n_ok      = sum(1 for r in results if r.status == "ok")
    n_warn    = sum(1 for r in results if r.status == "warning")
    n_fail    = sum(1 for r in results if r.status == "fail")
    n_total   = len(results)

    print(f"\n{'='*65}")
    print(f"  RESULT: {n_ok}/{n_total} OK  |  {n_warn} warnings  |  {n_fail} failures")
    if n_fail == 0:
        print("  ✓ All interfaces are coherent — pipeline is safe to use.")
    else:
        print("  ✗ Failures detected — fix before running the agent.")
        for r in results:
            if r.status == "fail":
                print(f"    → {r.name}: {r.detail}")
    print(f"{'='*65}\n")

    return results


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = validate_pipeline(n_contracts=800, verbose=True)
