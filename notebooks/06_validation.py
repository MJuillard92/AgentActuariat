"""
06_validation.py
================
Actuarial validation library for experience mortality tables.
Functions: confidence_intervals, chi_square_test, prudence_margin, cox_model
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2, norm as _norm

warnings.filterwarnings('ignore')

try:
    from actuary_logger import LOGGER as _LOGGER
except ImportError:
    class _NoLogger:
        def log(self, *a, **k): pass
    _LOGGER = _NoLogger()

# ---------------------------------------------------------------------------
# Reference table: TH/TF 00-02
# ---------------------------------------------------------------------------
_AGES_REF = np.array([20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100], dtype=float)
_QX_H = np.array([
    0.000830, 0.000860, 0.001100, 0.001450, 0.001840, 0.002650,
    0.003960, 0.006180, 0.009480, 0.014870, 0.024010, 0.039840,
    0.070050, 0.120300, 0.200490, 0.310000, 0.420000
])
_QX_F = np.array([
    0.000340, 0.000320, 0.000350, 0.000420, 0.000560, 0.000800,
    0.001220, 0.001940, 0.003040, 0.004860, 0.007900, 0.013650,
    0.026100, 0.050780, 0.098400, 0.170000, 0.280000
])


def _qx_ref_default(age: float, sexe: str = 'H') -> float:
    tbl = _QX_H if sexe == 'H' else _QX_F
    return float(np.exp(np.interp(float(np.clip(age, 20, 100)), _AGES_REF, np.log(tbl))))


# ---------------------------------------------------------------------------
# confidence_intervals
# ---------------------------------------------------------------------------

def confidence_intervals(exposure_table: pd.DataFrame,
                         qx_col: str = None,
                         alpha: float = 0.05) -> pd.DataFrame:
    """
    WHEN TO USE:
        After smoothing, to quantify uncertainty around the estimated mortality
        rates. Poisson CI on D_x provides the standard actuarial confidence band.

    INPUTS:
        exposure_table : DataFrame — Must have columns: age, E_x, D_x, qx_col.
        qx_col         : str       — Column name for smoothed qx (default 'q_x_lisse').
        alpha          : float     — Significance level (default 0.05 -> 95% CI).

    OUTPUTS:
        DataFrame with columns:
            age      : int   — Integer age.
            qx       : float — Smoothed qx.
            ci_lower : float — Lower bound of Poisson-based CI.
            ci_upper : float — Upper bound of Poisson-based CI.
            width    : float — CI half-width (ci_upper - ci_lower).
    """
    t = exposure_table.copy()
    if qx_col is None:
        for candidate in ('q_x_lisse', 'q_x_brut', 'qx'):
            if candidate in t.columns:
                qx_col = candidate
                break
        else:
            raise ValueError("[confidence_intervals] No qx column found. Pass qx_col explicitly.")
    if qx_col not in t.columns:
        raise ValueError(f"[confidence_intervals] Column '{qx_col}' not found.")

    rows = []
    for _, row in t.iterrows():
        age = int(row['age'])
        E = float(row['E_x'])
        D = int(row['D_x'])
        qx = row.get(qx_col, np.nan)

        if E <= 0 or np.isnan(qx):
            rows.append({'age': age, 'qx': qx, 'ci_lower': np.nan,
                         'ci_upper': np.nan, 'width': np.nan})
            continue

        # Exact Poisson CI for mu_x = D/E, then convert to qx
        mu_lo = _chi2.ppf(alpha / 2, 2 * D) / (2 * E) if D > 0 else 0.0
        mu_hi = _chi2.ppf(1 - alpha / 2, 2 * (D + 1)) / (2 * E)

        qx_lo = 1.0 - np.exp(-mu_lo)
        qx_hi = 1.0 - np.exp(-mu_hi)

        rows.append({
            'age': age,
            'qx': float(qx),
            'ci_lower': float(qx_lo),
            'ci_upper': float(qx_hi),
            'width': float(qx_hi - qx_lo),
        })

    result = pd.DataFrame(rows)
    mean_width = result['width'].dropna().mean()
    n_valid = result['width'].notna().sum()
    ci_pct = int(100 * (1 - alpha))
    print(f"[confidence_intervals] {ci_pct}% Poisson CI computed for {n_valid} ages. "
          f"Mean CI width: {mean_width:.6f}")
    _LOGGER.log("confidence_intervals",
                f"Intervalles de confiance {ci_pct}% (Poisson) : {n_valid} âges, "
                f"largeur moyenne {mean_width:.4%}",
                {"niveau_confiance_pct": ci_pct, "n_ages_valides": int(n_valid),
                 "largeur_moyenne_IC": round(float(mean_width), 6)})
    return result


# ---------------------------------------------------------------------------
# chi_square_test
# ---------------------------------------------------------------------------

def chi_square_test(exposure_table: pd.DataFrame,
                    qx_col: str = None,
                    reference_fn=None,
                    sexe: str = 'H') -> dict:
    """
    WHEN TO USE:
        Formal goodness-of-fit test comparing observed deaths to those expected
        under the smoothed (or reference) table. Complements the SMR.

    INPUTS:
        exposure_table : DataFrame      — Must have columns: age, E_x, D_x.
        qx_col         : str            — Smoothed qx column (default 'q_x_lisse').
        reference_fn   : callable|None  — callable(age, sexe) -> qx. If None,
                                          uses the qx_col column from exposure_table
                                          as the expected model.
        sexe           : str            — 'H' or 'F' (used only if reference_fn is set).

    OUTPUTS:
        dict with keys:
            statistic        : float — Chi-square test statistic.
            p_value          : float — P-value (low = poor fit).
            df               : int   — Degrees of freedom (number of age cells tested).
            conclusion       : str   — 'Good fit' or 'Significant deviation'.
            significant_ages : list  — Ages where local deviation is significant (p < 0.05).
    """
    t = exposure_table.copy()
    if qx_col is None:
        for candidate in ('q_x_lisse', 'q_x_brut', 'qx'):
            if candidate in t.columns:
                qx_col = candidate
                break
    if reference_fn is not None:
        t['qx_model'] = t['age'].apply(lambda a: reference_fn(a, sexe))
    elif qx_col and qx_col in t.columns:
        t['qx_model'] = t[qx_col]
    else:
        raise ValueError(f"[chi_square_test] Column '{qx_col}' not found and no reference_fn.")

    t = t.dropna(subset=['qx_model'])
    t = t[t['E_x'] > 0].copy()

    E_expected = t['E_x'] * t['qx_model']
    D_obs = t['D_x'].values.astype(float)
    D_exp = E_expected.values

    # Aggregate cells with very low expected (E < 5 rule for chi-square validity)
    mask_valid = D_exp >= 1.0
    if mask_valid.sum() < 2:
        print("[chi_square_test] WARNING: Too few cells with E >= 1. Test may be unreliable.")

    chi2_local = np.where(
        D_exp > 0,
        (D_obs - D_exp) ** 2 / D_exp,
        0.0
    )
    stat = float(chi2_local[mask_valid].sum())
    df_test = int(mask_valid.sum())

    p_value = float(1.0 - _chi2.cdf(stat, df=df_test)) if df_test > 0 else np.nan
    conclusion = 'Good fit (p >= 0.05)' if p_value >= 0.05 else 'Significant deviation (p < 0.05)'

    # Individual age significance
    p_local = np.where(D_exp > 0, 1.0 - _chi2.cdf(chi2_local, df=1), np.nan)
    sig_ages = t['age'].values[p_local < 0.05].tolist() if not np.all(np.isnan(p_local)) else []

    print(f"[chi_square_test] Chi2={stat:.2f}, df={df_test}, p={p_value:.4f} "
          f"-> {conclusion}")
    print(f"  Significant ages (local): {sig_ages[:20]}")
    _LOGGER.log("chi_square_test",
                f"Test chi² : statistique={stat:.2f}, ddl={df_test}, p={p_value:.4f} → {conclusion}",
                {"chi2_stat": round(stat, 2), "df": df_test,
                 "p_value": round(float(p_value), 4), "conclusion": conclusion,
                 "n_ages_significatifs": len(sig_ages),
                 "ages_significatifs": sig_ages[:10]})
    return {
        'statistic': stat,
        'p_value': p_value,
        'df': df_test,
        'conclusion': conclusion,
        'significant_ages': sig_ages,
    }


# ---------------------------------------------------------------------------
# prudence_margin
# ---------------------------------------------------------------------------

def prudence_margin(exposure_table: pd.DataFrame,
                    qx_col: str = None,
                    ci_upper_col: str = None,
                    ci_result: pd.DataFrame = None,
                    reference_fn=None,
                    sexe: str = 'H',
                    alpha: float = 0.05) -> dict:
    """
    WHEN TO USE:
        For annuity/life insurance reserve purposes, to verify that the experience
        table used for pricing is prudent (conservative). A prudent table has qx
        above the observed upper CI at most ages.

    INPUTS:
        exposure_table : DataFrame      — Must have columns: age, E_x, D_x, qx_col.
        qx_col         : str            — Smoothed qx column (default 'q_x_lisse').
        ci_upper_col   : str|None       — Column name of precomputed CI upper bound.
                                          If None, Poisson CI is computed internally.
        ci_result      : DataFrame|None — Output of confidence_intervals(). If provided,
                                          used directly instead of recomputing.
        reference_fn   : callable|None  — callable(age, sexe) -> qx to use as
                                          the reference (e.g. a regulatory table).
                                          If None, compares qx to the CI upper bound.
        sexe           : str            — 'H' or 'F'.
        alpha          : float          — CI level (default 0.05 -> 95% CI).

    OUTPUTS:
        dict with keys:
            is_prudent            : bool  — True if global_prudence_margin > 0.
            pct_ages_above_ci     : float — % of ages where qx > CI upper bound.
            global_prudence_margin : float — (mean qx_ref - mean qx_exp) / mean qx_ref.
            prudence_level        : str  — 'insufficient', 'adequate', 'conservative'.
            interpretation        : str  — Human-readable assessment.
    """
    t = exposure_table.copy()
    if qx_col is None:
        for candidate in ('q_x_lisse', 'q_x_brut', 'qx'):
            if candidate in t.columns:
                qx_col = candidate
                break
    if qx_col not in t.columns:
        raise ValueError(f"[prudence_margin] Column '{qx_col}' not found.")

    # Compute or use provided CI
    if ci_result is not None and 'ci_upper' in ci_result.columns:
        t = t.merge(ci_result[['age', 'ci_upper']], on='age', how='left')
        ci_upper_col = 'ci_upper'
    elif ci_upper_col is None or ci_upper_col not in t.columns:
        ci_df = confidence_intervals(t, qx_col=qx_col, alpha=alpha)
        t = t.merge(ci_df[['age', 'ci_upper']], on='age', how='left')
        ci_upper_col = 'ci_upper'

    t = t.dropna(subset=[qx_col, ci_upper_col])

    if reference_fn is not None:
        t['qx_ref_pm'] = t['age'].apply(lambda a: reference_fn(a, sexe))
        qx_ref_vals = t['qx_ref_pm'].values
        qx_exp_vals = t[qx_col].values
        above_ci = (qx_exp_vals > t[ci_upper_col].values).mean()
        global_margin = float(
            (qx_ref_vals - qx_exp_vals).mean() / np.mean(qx_ref_vals)
            if np.mean(qx_ref_vals) > 0 else 0.0
        )
    else:
        # Compare experience qx to its own CI upper bound (less useful, but valid)
        qx_exp_vals = t[qx_col].values
        above_ci = (qx_exp_vals > t[ci_upper_col].values).mean()
        global_margin = 0.0

    is_prudent = global_margin > 0

    if global_margin >= 0.10:
        prudence_level = 'conservative'
        interpretation = (
            f"The reference table has a {global_margin*100:.1f}% global margin above "
            f"experience. The table is conservative / prudent."
        )
    elif global_margin >= 0:
        prudence_level = 'adequate'
        interpretation = (
            f"The reference table has a {global_margin*100:.1f}% global margin above "
            f"experience. Marginally prudent."
        )
    else:
        prudence_level = 'insufficient'
        interpretation = (
            f"The reference table is {abs(global_margin)*100:.1f}% BELOW "
            f"experience mortality on average. The table is NOT prudent."
        )

    print(f"[prudence_margin] {prudence_level.upper()} | "
          f"Global margin: {global_margin*100:.2f}% | "
          f"Ages above CI upper: {above_ci*100:.1f}%")
    print(f"  {interpretation}")
    _LOGGER.log("prudence_margin",
                f"Marge de prudence : {prudence_level.upper()} — "
                f"marge globale {global_margin*100:.2f}% — {interpretation}",
                {"prudence_level": prudence_level, "is_prudent": is_prudent,
                 "global_prudence_margin_pct": round(global_margin * 100, 2),
                 "pct_ages_above_ci": round(float(above_ci * 100), 1)})
    return {
        'is_prudent': is_prudent,
        'pct_ages_above_ci': float(above_ci * 100),
        'global_prudence_margin': float(global_margin),
        'prudence_level': prudence_level,
        'interpretation': interpretation,
    }


# ---------------------------------------------------------------------------
# cox_model
# ---------------------------------------------------------------------------

def cox_model(df: pd.DataFrame,
              duration_col: str = None,
              event_col: str = 'cause_sortie',
              covariates: list = None,
              dob_col: str = 'date_naissance',
              entry_col: str = 'date_entree',
              exit_col: str = 'date_sortie') -> dict:
    """
    WHEN TO USE:
        When you have covariate information (gender, product type, etc.) and want
        to measure their effect on mortality differentials. Uses lifelines if
        available, otherwise falls back to a manual log-rank test.

    INPUTS:
        df           : DataFrame  — Individual-level cleaned portfolio data.
        duration_col : str|None   — Column for observation duration in years.
                                    If None, computed from dob_col and entry/exit dates
                                    as age-at-exit minus age-at-entry.
        event_col    : str        — Column indicating death ('deces' = event = 1).
        covariates   : list       — List of covariate column names (default ['sexe']).
        dob_col      : str        — Date-of-birth column name.
        entry_col    : str        — Entry-date column name.
        exit_col     : str        — Exit-date column name.

    OUTPUTS:
        dict with keys:
            hazard_ratios    : dict  — {covariate: HR}.
            p_values         : dict  — {covariate: p_value}.
            concordance_index : float — C-index (0.5 = random, 1.0 = perfect).
            interpretation   : str  — Human-readable summary.
            warning          : str  — Set if lifelines is not available.
    """
    if covariates is None:
        covariates = ['sexe']

    df = df.copy()

    # Compute duration if not provided
    if duration_col is None or duration_col not in df.columns:
        for col in [dob_col, entry_col, exit_col]:
            df[col] = pd.to_datetime(df[col])
        df['_duration'] = (df[exit_col] - df[entry_col]).dt.days / 365.25
        duration_col = '_duration'

    df['_event'] = (df[event_col] == 'deces').astype(int)

    # Encode string covariates
    df_model = df[[duration_col, '_event'] + covariates].copy()
    for cov in covariates:
        if df_model[cov].dtype == object:
            df_model[cov] = pd.Categorical(df_model[cov]).codes

    df_model = df_model.dropna()

    warning_msg = ''
    try:
        from lifelines import CoxPHFitter
        cph = CoxPHFitter()
        cph.fit(df_model, duration_col=duration_col, event_col='_event',
                show_progress=False)
        summary = cph.summary
        hazard_ratios = dict(np.exp(summary['coef']))
        p_values = dict(summary['p'])
        concordance = float(cph.concordance_index_)

        interp_parts = []
        for cov in covariates:
            if cov in hazard_ratios:
                hr = hazard_ratios[cov]
                pv = p_values.get(cov, np.nan)
                sig = "significant" if pv < 0.05 else "not significant"
                direction = "higher" if hr > 1 else "lower"
                interp_parts.append(
                    f"{cov}: HR={hr:.3f} ({direction} mortality, p={pv:.4f}, {sig})"
                )
        interpretation = "; ".join(interp_parts) if interp_parts else "No significant covariates."

    except ImportError:
        warning_msg = (
            "lifelines not installed. Falling back to log-rank test for binary covariates."
        )
        print(f"[cox_model] WARNING: {warning_msg}")

        hazard_ratios = {}
        p_values = {}
        concordance = np.nan
        interp_parts = []

        for cov in covariates:
            if cov not in df_model.columns:
                continue
            groups = df_model[cov].unique()
            if len(groups) != 2:
                p_values[cov] = np.nan
                hazard_ratios[cov] = np.nan
                continue

            # Log-rank test (Mantel-Cox)
            g0 = df_model[df_model[cov] == groups[0]]
            g1 = df_model[df_model[cov] == groups[1]]

            # Event counts and total times per group
            O0, O1 = int(g0['_event'].sum()), int(g1['_event'].sum())
            E0 = len(g0) * (O0 + O1) / max(len(df_model), 1)
            E1 = len(g1) * (O0 + O1) / max(len(df_model), 1)

            chi2_lr = ((O0 - E0) ** 2 / max(E0, 1) + (O1 - E1) ** 2 / max(E1, 1))
            from scipy.stats import chi2 as _chi2_lr
            pv = float(1.0 - _chi2_lr.cdf(chi2_lr, df=1))

            # Rough HR estimate
            hr = (O1 / max(E1, 1e-9)) / (O0 / max(E0, 1e-9))
            hazard_ratios[cov] = hr
            p_values[cov] = pv

            sig = "significant" if pv < 0.05 else "not significant"
            direction = "higher" if hr > 1 else "lower"
            interp_parts.append(
                f"{cov}: approx HR={hr:.3f} ({direction} mortality for group {groups[1]}, "
                f"log-rank p={pv:.4f}, {sig})"
            )

        interpretation = "; ".join(interp_parts) if interp_parts else "No covariates tested."

    print(f"[cox_model] Covariates: {covariates} | C-index: {concordance:.3f}")
    print(f"  {interpretation}")
    _LOGGER.log("cox_model",
                f"Modèle de Cox : C-index={concordance:.3f} — {interpretation}",
                {"covariates": covariates,
                 "concordance_index": round(float(concordance), 3) if not np.isnan(concordance) else None,
                 "hazard_ratios": {k: round(float(v), 3) for k, v in hazard_ratios.items()
                                   if v is not None and not np.isnan(float(v))},
                 "p_values": {k: round(float(v), 4) for k, v in p_values.items()
                              if v is not None and not np.isnan(float(v))}})
    return {
        'hazard_ratios': hazard_ratios,
        'p_values': p_values,
        'concordance_index': concordance,
        'interpretation': interpretation,
        'warning': warning_msg,
    }
