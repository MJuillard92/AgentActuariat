"""
02_exposure.py
==============
Actuarial exposure computation library.
Functions: compute_exposure_by_age, compute_exposure_by_year, exposure_summary
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
# compute_exposure_by_age
# ---------------------------------------------------------------------------

def compute_exposure_by_age(df: pd.DataFrame,
                            age_min: int = 20,
                            age_max: int = 90,
                            dob_col: str = 'date_naissance',
                            entry_col: str = 'date_entree',
                            exit_col: str = 'date_sortie',
                            death_col: str = 'cause_sortie') -> pd.DataFrame:
    """
    WHEN TO USE:
        Always, immediately after clean_data. This is the standard central
        exposure method used for experience mortality table construction. Run
        once per study population (a single gender at a time is recommended).

    INPUTS:
        df        : DataFrame — Cleaned portfolio with date columns.
        age_min   : int       — Lowest integer age to compute (default 20).
        age_max   : int       — Highest integer age to compute (default 90).
        dob_col   : str       — Date-of-birth column name.
        entry_col : str       — Entry-date column name.
        exit_col  : str       — Exit-date column name.
        death_col : str       — Cause-of-exit column name ('deces' = death).

    OUTPUTS:
        DataFrame with columns:
            age      : int   — Integer age x.
            E_x      : float — Central exposure (person-years at age x).
            D_x      : int   — Observed deaths at age x.
            mu_x     : float — Central death rate (D_x / E_x); NaN if E_x = 0.
            q_x_brut : float — Crude annual probability of death
                               (1 - exp(-mu_x)); NaN if E_x = 0.
    """
    df = df.copy()
    for col in [dob_col, entry_col, exit_col]:
        df[col] = pd.to_datetime(df[col])

    results = []
    for age in range(age_min, age_max + 1):
        # Individuals who were observed at integer age `age`
        mask = (
            ((df[entry_col] - df[dob_col]).dt.days / 365.25 <= age) &
            ((df[exit_col] - df[dob_col]).dt.days / 365.25 > age)
        )
        subset = df[mask].copy()

        if len(subset) == 0:
            results.append({'age': age, 'E_x': 0.0, 'D_x': 0,
                            'mu_x': np.nan, 'q_x_brut': np.nan})
            continue

        # Birthday at exact integer age
        age_start = subset[dob_col] + pd.to_timedelta(age * 365, unit='D')
        age_end = subset[dob_col] + pd.to_timedelta((age + 1) * 365, unit='D')

        # Intersection of [entry, exit] with [age_start, age_end]
        obs_start = pd.concat(
            [subset[entry_col].reset_index(drop=True),
             age_start.reset_index(drop=True)], axis=1
        ).max(axis=1)
        obs_end = pd.concat(
            [subset[exit_col].reset_index(drop=True),
             age_end.reset_index(drop=True)], axis=1
        ).min(axis=1)

        exposure = ((obs_end - obs_start).dt.days / 365.25).clip(lower=0)
        E_x = float(exposure.sum())

        # Deaths: cause == 'deces' AND death occurred within this age year
        age_sortie = (subset[exit_col] - subset[dob_col]).dt.days / 365.25
        D_x = int(
            ((subset[death_col] == 'deces') &
             (age_sortie.values.astype(int) == age)).sum()
        )

        mu_x = D_x / E_x if E_x > 0 else np.nan
        q_x = 1.0 - np.exp(-mu_x) if not np.isnan(mu_x) else np.nan

        results.append({
            'age': age,
            'E_x': round(E_x, 4),
            'D_x': D_x,
            'mu_x': mu_x,
            'q_x_brut': q_x,
        })

    table = pd.DataFrame(results)

    # Summary printout
    total_E = table['E_x'].sum()
    total_D = table['D_x'].sum()
    n_zero = (table['E_x'] == 0).sum()
    n_low = ((table['E_x'] > 0) & (table['E_x'] < 10)).sum()

    print(f"[compute_exposure_by_age] Age range: {age_min}-{age_max}")
    print(f"  Total exposure : {total_E:,.1f} person-years")
    print(f"  Total deaths   : {int(total_D):,}")
    print(f"  Empty ages     : {n_zero}")
    print(f"  Ages with E_x < 10 (low credibility): {n_low}")
    q_global = round(total_D / total_E, 6) if total_E > 0 else None
    _LOGGER.log("compute_exposure_by_age",
                f"Exposition calculée ({age_min}–{age_max} ans) : "
                f"{total_E:,.1f} années-personnes, {int(total_D):,} décès, "
                f"taux brut global {q_global:.4%}" if q_global else
                f"Exposition calculée ({age_min}–{age_max} ans) : {total_E:,.1f} py, {int(total_D):,} décès",
                {"age_min": age_min, "age_max": age_max,
                 "total_exposure_py": round(total_E, 1),
                 "total_deaths": int(total_D),
                 "q_x_brut_global": q_global,
                 "n_ages_zero_exposure": int(n_zero),
                 "n_ages_low_credibility": int(n_low)})
    return table


# ---------------------------------------------------------------------------
# compute_exposure_by_year
# ---------------------------------------------------------------------------

def compute_exposure_by_year(df: pd.DataFrame,
                             age_min: int = 20,
                             age_max: int = 90,
                             dob_col: str = 'date_naissance',
                             entry_col: str = 'date_entree',
                             exit_col: str = 'date_sortie',
                             death_col: str = 'cause_sortie') -> pd.DataFrame:
    """
    WHEN TO USE:
        Temporal stability analysis — use after compute_exposure_by_age to check
        whether mortality rates are stable across calendar years. Useful for
        detecting trend, pandemic effects, or data collection changes.

    INPUTS:
        df        : DataFrame — Cleaned portfolio with date columns.
        age_min   : int       — Lowest integer age to compute (default 20).
        age_max   : int       — Highest integer age to compute (default 90).
        dob_col   : str       — Date-of-birth column name.
        entry_col : str       — Entry-date column name.
        exit_col  : str       — Exit-date column name.
        death_col : str       — Cause-of-exit column name.

    OUTPUTS:
        DataFrame with columns:
            year : int   — Calendar year.
            age  : int   — Integer age x.
            E_x  : float — Central exposure in that year for that age.
            D_x  : int   — Observed deaths in that year for that age.
    """
    df = df.copy()
    for col in [dob_col, entry_col, exit_col]:
        df[col] = pd.to_datetime(df[col])

    year_min = int(df[entry_col].dt.year.min())
    year_max = int(df[exit_col].dt.year.max())

    results = []
    for year in range(year_min, year_max + 1):
        year_start = pd.Timestamp(f'{year}-01-01')
        year_end = pd.Timestamp(f'{year}-12-31')

        # Contracts active during this year
        active = df[
            (df[entry_col] <= year_end) & (df[exit_col] >= year_start)
        ].copy()

        if len(active) == 0:
            continue

        # Clip entry/exit to this calendar year
        active_entry = active[entry_col].clip(lower=year_start)
        active_exit = active[exit_col].clip(upper=year_end + pd.Timedelta(days=1))

        age_at_entry = ((active_entry - active[dob_col]).dt.days / 365.25).values
        age_at_exit = ((active_exit - active[dob_col]).dt.days / 365.25).values
        is_death = (active[death_col] == 'deces').values
        death_age = ((active[exit_col] - active[dob_col]).dt.days / 365.25).values

        for age in range(age_min, age_max + 1):
            # Individuals who crossed age `age` during this year
            mask = (age_at_entry <= age) & (age_at_exit > age)
            if not mask.any():
                continue

            age_start_exact = active.loc[mask, dob_col] + pd.to_timedelta(age * 365, unit='D')
            age_end_exact = active.loc[mask, dob_col] + pd.to_timedelta((age + 1) * 365, unit='D')
            obs_start = pd.concat(
                [active.loc[mask, entry_col].clip(lower=year_start).reset_index(drop=True),
                 age_start_exact.reset_index(drop=True)], axis=1
            ).max(axis=1)
            obs_end = pd.concat(
                [active.loc[mask, exit_col].clip(upper=year_end + pd.Timedelta(days=1)
                                                 ).reset_index(drop=True),
                 age_end_exact.reset_index(drop=True)], axis=1
            ).min(axis=1)

            E_x = float(((obs_end - obs_start).dt.days / 365.25).clip(lower=0).sum())

            mask_np = mask.values if hasattr(mask, 'values') else mask
            D_x = int(
                (is_death[mask_np] &
                 (death_age[mask_np].astype(int) == age) &
                 (active.loc[mask, exit_col].dt.year.values == year)).sum()
            )

            results.append({'year': year, 'age': age,
                            'E_x': round(E_x, 4), 'D_x': D_x})

    table = pd.DataFrame(results) if results else pd.DataFrame(
        columns=['year', 'age', 'E_x', 'D_x'])

    print(f"[compute_exposure_by_year] {len(table):,} (year, age) cells computed "
          f"across years {year_min}-{year_max}")
    return table


# ---------------------------------------------------------------------------
# exposure_summary
# ---------------------------------------------------------------------------

def exposure_summary(exposure_table: pd.DataFrame) -> dict:
    """
    WHEN TO USE:
        After compute_exposure_by_age. Quick overview of exposure quality before
        choosing a smoothing method.

    INPUTS:
        exposure_table : DataFrame — Output of compute_exposure_by_age, must have
                                     columns age, E_x, D_x.

    OUTPUTS:
        dict with keys:
            total_exposure         : float — Sum of E_x across all ages.
            total_deaths           : int   — Sum of D_x.
            age_range              : tuple — (min_age_with_data, max_age_with_data).
            n_ages_empty           : int   — Ages with E_x = 0.
            n_ages_low_credibility : int   — Ages with 0 < E_x < 10.
            pct_low_credibility    : float — Percentage of ages with E_x < 10.
            crude_rate_overall     : float — Total D_x / Total E_x.
    """
    t = exposure_table.copy()
    total_E = float(t['E_x'].sum())
    total_D = int(t['D_x'].sum())

    has_data = t[t['E_x'] > 0]
    age_range = (
        int(has_data['age'].min()) if len(has_data) > 0 else None,
        int(has_data['age'].max()) if len(has_data) > 0 else None,
    )

    n_empty = int((t['E_x'] == 0).sum())
    n_low = int(((t['E_x'] > 0) & (t['E_x'] < 10)).sum())
    total_ages = len(t)
    pct_low = 100.0 * n_low / max(total_ages, 1)

    crude_rate = total_D / total_E if total_E > 0 else np.nan

    summary = {
        'total_exposure': total_E,
        'total_deaths': total_D,
        'age_range': age_range,
        'n_ages_empty': n_empty,
        'n_ages_low_credibility': n_low,
        'pct_low_credibility': pct_low,
        'crude_rate_overall': crude_rate,
    }

    print(f"[exposure_summary] Exposure: {total_E:,.1f} py | Deaths: {total_D:,} | "
          f"Ages: {age_range[0]}-{age_range[1]} | "
          f"Low credibility ages: {n_low} ({pct_low:.1f}%)")
    _LOGGER.log("exposure_summary",
                f"Synthèse exposition : {total_E:,.1f} py | {total_D:,} décès | "
                f"âges {age_range[0]}–{age_range[1]} | {n_low} âges faible crédibilité ({pct_low:.1f}%)",
                {**summary})
    return summary
