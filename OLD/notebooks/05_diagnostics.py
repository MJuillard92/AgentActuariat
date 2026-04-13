"""
05_diagnostics.py
=================
Actuarial diagnostics library for experience mortality tables.
Functions: diagnose_credibility, diagnose_monotonicity, compare_smoothers, compute_smr
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    from actuary_logger import LOGGER as _LOGGER
except ImportError:
    class _NoLogger:
        def log(self, *a, **k): pass
    _LOGGER = _NoLogger()

# ---------------------------------------------------------------------------
# Reference table: TH/TF 00-02 (embedded, log-interpolated)
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


def _qx_ref_array_default(ages, sexe: str = 'H') -> np.ndarray:
    tbl = _QX_H if sexe == 'H' else _QX_F
    return np.exp(np.interp(
        np.clip(np.asarray(ages, dtype=float), 20, 100), _AGES_REF, np.log(tbl)))


# ---------------------------------------------------------------------------
# diagnose_credibility
# ---------------------------------------------------------------------------

def diagnose_credibility(exposure_table: pd.DataFrame,
                         threshold: int = 10) -> dict:
    """
    WHEN TO USE:
        ALWAYS before choosing a smoothing method. This function tells you which
        ages have insufficient data and recommends the most appropriate smoother.

    INPUTS:
        exposure_table : DataFrame — Output of compute_exposure_by_age().
                                     Must have columns: age, E_x, D_x.
        threshold      : int       — E_x below this value is considered low-credibility
                                     (default 10 person-years).

    OUTPUTS:
        dict with keys:
            low_credibility_ages  : list  — Ages where 0 < E_x < threshold.
            n_low                 : int   — Count of low-credibility ages.
            pct_low               : float — Percentage of all ages that are low-credibility.
            zero_exposure_ages    : list  — Ages where E_x = 0.
            recommendation        : str  — 'parametric', 'non-parametric', or 'mixed'.
            recommendation_reason : str  — Human-readable explanation.
    """
    t = exposure_table.copy()
    n_total = len(t)

    zero_exp = t[t['E_x'] == 0]['age'].tolist()
    low_cred = t[(t['E_x'] > 0) & (t['E_x'] < threshold)]['age'].tolist()

    n_zero = len(zero_exp)
    n_low = len(low_cred)
    pct_low = 100.0 * (n_low + n_zero) / max(n_total, 1)

    if pct_low >= 30:
        recommendation = 'parametric'
        reason = (
            f"{pct_low:.0f}% of ages have E_x < {threshold} or zero exposure. "
            f"Use smooth_gompertz() or smooth_makeham() which can extrapolate "
            f"beyond sparse data."
        )
    elif pct_low >= 10:
        recommendation = 'mixed'
        reason = (
            f"{pct_low:.0f}% of ages have low credibility (E_x < {threshold}). "
            f"Consider smooth_whittaker() with high lambda, or smooth_gompertz() "
            f"for the sparse age range and smooth_whittaker() elsewhere."
        )
    else:
        recommendation = 'non-parametric'
        reason = (
            f"Data is dense ({pct_low:.0f}% low-credibility ages). "
            f"smooth_whittaker() is the default and appropriate choice. "
            f"smooth_spline() or smooth_local_polynomial() are also suitable."
        )

    print(f"[diagnose_credibility] Zero exposure: {n_zero} ages | "
          f"Low credibility (E_x<{threshold}): {n_low} ages ({pct_low:.1f}%)")
    print(f"  Recommendation: {recommendation.upper()} — {reason}")
    _LOGGER.log("diagnose_credibility",
                f"Crédibilité : {n_low} âges faible crédibilité ({pct_low:.1f}%) — "
                f"recommandation : {recommendation.upper()}",
                {"n_low_credibility": n_low, "pct_low_credibility": round(pct_low, 1),
                 "n_zero_exposure": n_zero, "recommendation": recommendation,
                 "recommendation_reason": reason})
    return {
        'low_credibility_ages': low_cred,
        'n_low': n_low,
        'pct_low': pct_low,
        'zero_exposure_ages': zero_exp,
        'recommendation': recommendation,
        'recommendation_reason': reason,
    }


# ---------------------------------------------------------------------------
# diagnose_monotonicity
# ---------------------------------------------------------------------------

def diagnose_monotonicity(qx_series, age_series,
                          age_start_check: int = 40) -> dict:
    """
    WHEN TO USE:
        After smoothing, before finalising the table. Monotone increasing qx
        for ages 40+ is a regulatory requirement in most jurisdictions.

    INPUTS:
        qx_series       : array-like — Smoothed qx values.
        age_series      : array-like — Corresponding integer ages.
        age_start_check : int        — Start checking monotonicity from this age
                                       (default 40; below 40 non-monotonicity
                                        is acceptable due to accident hump).

    OUTPUTS:
        dict with keys:
            n_violations  : int   — Number of age steps where qx decreases.
            violation_ages : list  — Ages x where qx[x] > qx[x+1].
            max_reversal  : float — Largest decrease magnitude.
            is_monotone   : bool  — True if n_violations == 0.
    """
    qx = np.asarray(qx_series, dtype=float)
    ages = np.asarray(age_series, dtype=int)

    mask = ages >= age_start_check
    qx_sub = qx[mask]
    ages_sub = ages[mask]

    if len(qx_sub) < 2:
        return {'n_violations': 0, 'violation_ages': [],
                'max_reversal': 0.0, 'is_monotone': True}

    diffs = np.diff(qx_sub)
    bad_idx = np.where(diffs < 0)[0]
    violation_ages = ages_sub[bad_idx].tolist()
    max_reversal = float(abs(diffs[bad_idx]).max()) if len(bad_idx) > 0 else 0.0
    n_violations = len(bad_idx)
    is_monotone = n_violations == 0

    if is_monotone:
        print(f"[diagnose_monotonicity] OK — qx is monotone increasing for ages >= {age_start_check}.")
    else:
        print(f"[diagnose_monotonicity] WARNING: {n_violations} monotonicity violation(s) "
              f"for ages >= {age_start_check}. Max reversal: {max_reversal:.6f} "
              f"at ages: {violation_ages[:10]}")
    _LOGGER.log("diagnose_monotonicity",
                f"Monotonicité : {'OK' if is_monotone else f'{n_violations} violation(s)'} "
                f"pour âges ≥ {age_start_check}",
                {"is_monotone": is_monotone, "n_violations": n_violations,
                 "max_reversal": round(max_reversal, 6),
                 "violation_ages": violation_ages[:10]})
    return {
        'n_violations': n_violations,
        'violation_ages': violation_ages,
        'max_reversal': max_reversal,
        'is_monotone': is_monotone,
    }


# ---------------------------------------------------------------------------
# compare_smoothers
# ---------------------------------------------------------------------------

def compare_smoothers(smoothers_dict: dict,
                      exposure_table: pd.DataFrame) -> tuple:
    """
    WHEN TO USE:
        After fitting two or more smoothers. Use this to select the best method
        before finalising the experience table.

    INPUTS:
        smoothers_dict : dict      — {'method_name': smooth_*() result dict, ...}
                                     Each value must have keys: ages, qx_smoothed.
        exposure_table : DataFrame — Output of compute_exposure_by_age().
                                     Must have columns: age, E_x, D_x, q_x_brut.

    OUTPUTS:
        (DataFrame, dict) where:
          - DataFrame has columns:
              method, AIC_poisson, BIC_poisson, MSE_vs_crude, n_non_monotone,
              max_reversal
            sorted ascending by AIC_poisson.
          - dict has keys:
              recommended : str — Method name with lowest AIC.
              reason      : str — Short explanation.
    """
    t = exposure_table.copy()
    rows = []

    for method_name, result in smoothers_dict.items():
        ages_fit = np.asarray(result['ages'], dtype=int)
        qx_fit = np.asarray(result['qx_smoothed'], dtype=float)

        # Map fitted qx back to exposure table
        fit_map = dict(zip(ages_fit, qx_fit))
        t_sub = t[t['age'].isin(ages_fit)].copy()
        t_sub['qx_fit'] = t_sub['age'].map(fit_map)
        t_sub = t_sub.dropna(subset=['qx_fit', 'q_x_brut'])

        # Poisson AIC: -2*logL + 2*k (k = number of unique ages used)
        # logL = sum_x [ D_x * log(E_x * mu_x) - E_x * mu_x ] for Poisson
        mu_fit = np.where(
            t_sub['qx_fit'] < 1.0,
            -np.log(1.0 - t_sub['qx_fit'].clip(1e-9, 1 - 1e-9)),
            1.0,
        )
        D = t_sub['D_x'].values
        E = t_sub['E_x'].values
        mu_fit = np.clip(mu_fit, 1e-12, None)
        E_mu = E * mu_fit
        log_lik = np.sum(
            np.where(E_mu > 0,
                     D * np.log(E_mu + 1e-300) - E_mu - D * np.log(D + 1e-300) + D,
                     -E_mu)
        )
        n_params = len(ages_fit)
        n_obs = len(t_sub)
        aic = -2.0 * log_lik + 2.0 * n_params
        bic = -2.0 * log_lik + np.log(max(n_obs, 1)) * n_params

        # MSE on log scale vs crude rates
        valid_crude = t_sub['q_x_brut'].notna() & (t_sub['q_x_brut'] > 0)
        if valid_crude.sum() > 0:
            log_crude = np.log(t_sub.loc[valid_crude, 'q_x_brut'].values.clip(1e-9))
            log_fit = np.log(t_sub.loc[valid_crude, 'qx_fit'].values.clip(1e-9))
            mse = float(np.mean((log_crude - log_fit) ** 2))
        else:
            mse = np.nan

        # Monotonicity
        mono = diagnose_monotonicity(qx_fit, ages_fit, age_start_check=40)
        n_nm = mono['n_violations']
        max_rev = mono['max_reversal']

        rows.append({
            'method': method_name,
            'AIC_poisson': round(aic, 2),
            'BIC_poisson': round(bic, 2),
            'MSE_vs_crude': round(mse, 6) if not np.isnan(mse) else np.nan,
            'n_non_monotone': n_nm,
            'max_reversal': round(max_rev, 6),
        })

    df_result = pd.DataFrame(rows).sort_values('AIC_poisson').reset_index(drop=True)

    best = df_result.iloc[0]['method']
    best_aic = df_result.iloc[0]['AIC_poisson']
    reason = f"Lowest Poisson AIC = {best_aic:.1f}"
    if df_result.iloc[0]['n_non_monotone'] > 0:
        reason += f" (WARNING: {df_result.iloc[0]['n_non_monotone']} non-monotone steps — "
        reason += "consider a smoother with higher lambda or parametric constraint)"

    meta = {'recommended': best, 'reason': reason}

    print(f"[compare_smoothers] Best method: {best} (AIC={best_aic:.1f})")
    print(df_result.to_string(index=False))
    _LOGGER.log("compare_smoothers",
                f"Comparaison lisseurs : meilleure méthode = {best} (AIC={best_aic:.1f})",
                {"best_method": best, "AIC_best": round(best_aic, 1),
                 "methods_compared": df_result['method'].tolist(),
                 "AIC_all": {r['method']: r['AIC_poisson']
                             for r in df_result.to_dict('records')}})
    return df_result, meta


# ---------------------------------------------------------------------------
# compute_smr
# ---------------------------------------------------------------------------

def compute_smr(exposure_table: pd.DataFrame,
                qx_col: str = 'q_x_lisse',
                reference_fn=None,
                sexe: str = 'H',
                alpha: float = 0.05) -> dict:
    """
    WHEN TO USE:
        After smoothing. Compares your portfolio mortality to a reference table.
        SMR < 1 = portfolio less mortal than reference (favourable selection).
        SMR = 1 = portfolio in line with reference.
        SMR > 1 = excess mortality.

    INPUTS:
        exposure_table : DataFrame       — Must have columns: age, E_x, D_x,
                                           and qx_col.
        qx_col         : str             — Column name for smoothed qx (default
                                           'q_x_lisse').
        reference_fn   : callable|None   — callable(age, sexe) -> float.
                                           If None, uses built-in TH/TF 00-02.
        sexe           : str             — 'H' or 'F'. Used if reference_fn is None.
        alpha          : float           — Significance level for CI (default 0.05 -> 95%).

    OUTPUTS:
        dict with keys:
            smr_global     : float     — D_observed / D_expected.
            ci_lower       : float     — Lower bound of (1-alpha) CI.
            ci_upper       : float     — Upper bound of (1-alpha) CI.
            d_observed     : int       — Total observed deaths.
            d_expected     : float     — Total expected deaths under reference.
            smr_by_decade  : DataFrame — [decade, D_obs, D_exp, SMR] per decade.
            interpretation : str       — Human-readable conclusion.
    """
    if reference_fn is None:
        reference_fn = lambda age, s: _qx_ref_default(age, s)

    t = exposure_table.copy()
    t['qx_ref'] = t['age'].apply(lambda a: reference_fn(a, sexe))
    t['D_expected'] = t['E_x'] * t['qx_ref']

    D_obs = int(t['D_x'].sum())
    D_exp = float(t['D_expected'].sum())

    if D_exp <= 0:
        raise ValueError("[compute_smr] Total expected deaths = 0. Check exposure and reference.")

    smr = D_obs / D_exp
    z = float(np.abs(np.percentile(
        np.random.standard_normal(100_000), 100 * (1 - alpha / 2))))
    # Exact Poisson CI for the count D_obs divided by D_exp
    from scipy.stats import chi2 as _chi2
    ci_lower = _chi2.ppf(alpha / 2, 2 * D_obs) / (2 * D_exp) if D_obs > 0 else 0.0
    ci_upper = _chi2.ppf(1 - alpha / 2, 2 * (D_obs + 1)) / (2 * D_exp)

    # SMR by decade
    decade_rows = []
    for ds in range(20, 100, 10):
        m = (t['age'] >= ds) & (t['age'] < ds + 10)
        d_o = int(t.loc[m, 'D_x'].sum())
        d_a = float(t.loc[m, 'D_expected'].sum())
        smr_d = d_o / d_a if d_a > 0 else np.nan
        decade_rows.append({
            'decade': f'{ds}-{ds+9}',
            'D_obs': d_o,
            'D_exp': round(d_a, 1),
            'SMR': round(smr_d, 4) if not np.isnan(smr_d) else np.nan,
        })
    smr_by_decade = pd.DataFrame(decade_rows)

    # Interpretation
    if smr < 0.90:
        interpretation = (
            f"SMR = {smr:.4f}: Portfolio is significantly less mortal than the "
            f"reference (favourable selection effect)."
        )
    elif smr > 1.10:
        interpretation = (
            f"SMR = {smr:.4f}: Portfolio shows excess mortality relative to "
            f"the reference table."
        )
    else:
        interpretation = (
            f"SMR = {smr:.4f}: Portfolio mortality is in line with the "
            f"reference table (within ±10%)."
        )

    ci_pct = int(100 * (1 - alpha))
    print(f"[compute_smr] D_obs={D_obs:,} | D_exp={D_exp:.1f} | "
          f"SMR={smr:.4f} | {ci_pct}% CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  {interpretation}")
    print(smr_by_decade.to_string(index=False))
    _LOGGER.log("compute_smr",
                f"SMR = {smr:.4f} [{ci_lower:.4f} – {ci_upper:.4f}] — "
                f"{D_obs:,} décès observés / {D_exp:.1f} attendus — {interpretation}",
                {"SMR_global": round(smr, 4), "CI_lower": round(ci_lower, 4),
                 "CI_upper": round(ci_upper, 4), "D_observed": D_obs,
                 "D_expected": round(D_exp, 1), "sexe": sexe,
                 "interpretation": interpretation,
                 "SMR_par_decennie": smr_by_decade[['decade', 'SMR']].to_dict('records')})
    return {
        'smr_global': smr,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'd_observed': D_obs,
        'd_expected': D_exp,
        'smr_by_decade': smr_by_decade,
        'interpretation': interpretation,
    }
