"""
03_crude_rates.py
=================
Actuarial crude mortality rate estimation library.
Three methods: central exposure, binomial (Balducci), Kaplan-Meier.
Each returns a DataFrame with columns [age, E_x, D_x, qx, method_name].
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
# crude_rates_central
# ---------------------------------------------------------------------------

def crude_rates_central(exposure_table: pd.DataFrame) -> pd.DataFrame:
    """
    WHEN TO USE:
        When exposure is expressed in person-years (central exposure), which is
        the standard output of compute_exposure_by_age. This is the most common
        method for portfolio experience studies.

    INPUTS:
        exposure_table : DataFrame — Must have columns: age, E_x, D_x.
                                     Typically the output of compute_exposure_by_age.

    OUTPUTS:
        DataFrame with columns:
            age         : int   — Integer age x.
            E_x         : float — Central exposure (person-years).
            D_x         : int   — Observed deaths.
            qx          : float — Annual death probability: 1 - exp(-D_x/E_x).
                                   NaN where E_x = 0.
            method_name : str   — 'central'.
    """
    t = exposure_table.copy()
    t = t[['age', 'E_x', 'D_x']].copy()

    mu_x = np.where(t['E_x'] > 0, t['D_x'] / t['E_x'], np.nan)
    qx = np.where(~np.isnan(mu_x), 1.0 - np.exp(-mu_x), np.nan)

    t['qx'] = qx
    t['method_name'] = 'central'

    valid = t['qx'].notna().sum()
    print(f"[crude_rates_central] {valid} ages with valid qx "
          f"(out of {len(t)} total)")
    qx_valid = t.loc[t['qx'].notna(), 'qx']
    _LOGGER.log("crude_rates_central",
                f"Taux bruts (méthode centrale) : {valid} âges valides, "
                f"qx moyen {qx_valid.mean():.4%}, "
                f"qx à 65 ans ≈ {t.loc[t['age']==65, 'qx'].values[0]:.4%}" if 65 in t['age'].values else
                f"Taux bruts (méthode centrale) : {valid} âges valides, qx moyen {qx_valid.mean():.4%}",
                {"methode": "central", "n_ages_valides": int(valid),
                 "qx_moyen": round(float(qx_valid.mean()), 6),
                 "qx_min": round(float(qx_valid.min()), 6),
                 "qx_max": round(float(qx_valid.max()), 6)})
    return t[['age', 'E_x', 'D_x', 'qx', 'method_name']].reset_index(drop=True)


# ---------------------------------------------------------------------------
# crude_rates_binomial
# ---------------------------------------------------------------------------

def crude_rates_binomial(exposure_table: pd.DataFrame) -> pd.DataFrame:
    """
    WHEN TO USE:
        When exposure is expressed as the initial number of lives at the start of
        each year of age (initial exposure). Uses the Balducci (uniform distribution
        of deaths) assumption to convert central to initial exposure, then applies
        the binomial estimator.

        Note: if your exposure table already contains central exposure (person-years),
        this function applies the Balducci approximation E_x_initial ≈ E_x + D_x/2.

    INPUTS:
        exposure_table : DataFrame — Must have columns: age, E_x, D_x.

    OUTPUTS:
        DataFrame with columns:
            age         : int   — Integer age x.
            E_x         : float — Central exposure (as input).
            D_x         : int   — Observed deaths.
            qx          : float — Annual death probability under Balducci:
                                   D_x / (E_x + D_x/2). NaN where E_x + D_x = 0.
            method_name : str   — 'binomial_balducci'.
    """
    t = exposure_table.copy()
    t = t[['age', 'E_x', 'D_x']].copy()

    # Balducci: initial exposure = central + half the deaths
    E_initial = t['E_x'] + t['D_x'] / 2.0
    qx = np.where(
        E_initial > 0,
        t['D_x'] / E_initial,
        np.nan
    )
    # Clip to valid probability range
    qx = np.clip(qx, 0.0, 1.0)

    t['qx'] = qx
    t['method_name'] = 'binomial_balducci'

    valid = (~np.isnan(qx)).sum()
    print(f"[crude_rates_binomial] {valid} ages with valid qx "
          f"(Balducci assumption, out of {len(t)} total)")
    qx_valid = t.loc[t['qx'].notna(), 'qx']
    _LOGGER.log("crude_rates_binomial",
                f"Taux bruts (méthode binomiale Balducci) : {valid} âges valides, "
                f"qx moyen {qx_valid.mean():.4%}",
                {"methode": "binomial_balducci", "n_ages_valides": int(valid),
                 "qx_moyen": round(float(qx_valid.mean()), 6),
                 "qx_min": round(float(qx_valid.min()), 6),
                 "qx_max": round(float(qx_valid.max()), 6)})
    return t[['age', 'E_x', 'D_x', 'qx', 'method_name']].reset_index(drop=True)


# ---------------------------------------------------------------------------
# crude_rates_kaplan_meier
# ---------------------------------------------------------------------------

def crude_rates_kaplan_meier(df: pd.DataFrame,
                             age_min: int = 20,
                             age_max: int = 90,
                             dob_col: str = 'date_naissance',
                             entry_col: str = 'date_entree',
                             exit_col: str = 'date_sortie',
                             death_col: str = 'cause_sortie') -> pd.DataFrame:
    """
    WHEN TO USE:
        Small portfolios (fewer than ~5,000 contracts), when individual-level
        survival data is available and a non-parametric estimate is preferred.
        Does not require the lifelines package — implemented entirely in NumPy.

    INPUTS:
        df        : DataFrame — Individual-level cleaned portfolio data.
        age_min   : int       — Start of age range (default 20).
        age_max   : int       — End of age range (default 90).
        dob_col   : str       — Date-of-birth column name.
        entry_col : str       — Entry-date column name.
        exit_col  : str       — Exit-date column name.
        death_col : str       — Cause-of-exit column name.

    OUTPUTS:
        DataFrame with columns:
            age         : int   — Integer age x.
            E_x         : float — Effective risk set size (number at risk) at age x.
            D_x         : int   — Deaths at age x.
            survival    : float — Kaplan-Meier S(x) = product of (1 - d_i/n_i).
            qx          : float — 1 - S(x+1)/S(x), estimated from KM curve.
            method_name : str   — 'kaplan_meier'.
    """
    df = df.copy()
    for col in [dob_col, entry_col, exit_col]:
        df[col] = pd.to_datetime(df[col])

    # Compute exact ages at entry, exit, death
    df['_age_entry'] = (df[entry_col] - df[dob_col]).dt.days / 365.25
    df['_age_exit'] = (df[exit_col] - df[dob_col]).dt.days / 365.25
    df['_is_death'] = (df[death_col] == 'deces').astype(int)

    # Build KM estimate at integer age boundaries
    # Use the Greenwood approach on (t, event) pairs where t = exact age at exit
    ages = np.arange(age_min, age_max + 2, dtype=float)  # boundaries
    km_values = {}  # age_boundary -> S(age)

    S = 1.0
    km_values[float(age_min)] = S

    for age in range(age_min, age_max + 1):
        # Risk set: those who entered at or before this age and have not yet exited
        at_risk_mask = (df['_age_entry'] <= age) & (df['_age_exit'] > age)
        n_risk = int(at_risk_mask.sum())

        # Deaths at this integer age year
        death_mask = at_risk_mask & (df['_age_exit'].values.astype(int) == age) & \
                     (df['_is_death'] == 1)
        n_deaths = int(death_mask.sum())

        if n_risk > 0:
            S = S * (1.0 - n_deaths / n_risk)
        km_values[float(age + 1)] = S

    # Derive qx from S(x) and S(x+1)
    results = []
    for age in range(age_min, age_max + 1):
        S_x = km_values.get(float(age), np.nan)
        S_x1 = km_values.get(float(age + 1), np.nan)

        at_risk_mask = (df['_age_entry'] <= age) & (df['_age_exit'] > age)
        E_x = float(at_risk_mask.sum())

        death_mask = at_risk_mask & (df['_age_exit'].values.astype(int) == age) & \
                     (df['_is_death'] == 1)
        D_x = int(death_mask.sum())

        if not np.isnan(S_x) and not np.isnan(S_x1) and S_x > 0:
            qx = 1.0 - S_x1 / S_x
        else:
            qx = np.nan

        results.append({
            'age': age,
            'E_x': E_x,
            'D_x': D_x,
            'survival': S_x,
            'qx': qx,
            'method_name': 'kaplan_meier',
        })

    table = pd.DataFrame(results)
    valid = table['qx'].notna().sum()
    print(f"[crude_rates_kaplan_meier] KM estimator on {len(df):,} individuals, "
          f"{valid} ages with valid qx")
    qx_valid = table.loc[table['qx'].notna(), 'qx']
    _LOGGER.log("crude_rates_kaplan_meier",
                f"Taux bruts Kaplan-Meier sur {len(df):,} individus : "
                f"{valid} âges valides, qx moyen {qx_valid.mean():.4%}",
                {"methode": "kaplan_meier", "n_individus": len(df),
                 "n_ages_valides": int(valid),
                 "qx_moyen": round(float(qx_valid.mean()), 6),
                 "qx_min": round(float(qx_valid.min()), 6),
                 "qx_max": round(float(qx_valid.max()), 6)})
    return table[['age', 'E_x', 'D_x', 'survival', 'qx', 'method_name']].reset_index(drop=True)
