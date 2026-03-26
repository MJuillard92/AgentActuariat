"""
07_benchmarking.py
==================
Actuarial benchmarking library for experience mortality tables.
Functions: load_reference_table, abatement_factors, logit_regression, export_table

Reference tables embedded as hardcoded arrays (no external file dependency).
"""

import os
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
# Embedded reference table data
# Sparse knots — interpolated logarithmically to all integer ages 20-100.
# ---------------------------------------------------------------------------

_SPARSE_AGES = np.array(
    [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    dtype=float
)

_TABLES = {
    # French 2000-2002 experience tables
    'TH0002': np.array([
        0.000830, 0.000860, 0.001100, 0.001450, 0.001840, 0.002650,
        0.003960, 0.006180, 0.009480, 0.014870, 0.024010, 0.039840,
        0.070050, 0.120300, 0.200490, 0.310000, 0.420000
    ]),
    'TF0002': np.array([
        0.000340, 0.000320, 0.000350, 0.000420, 0.000560, 0.000800,
        0.001220, 0.001940, 0.003040, 0.004860, 0.007900, 0.013650,
        0.026100, 0.050780, 0.098400, 0.170000, 0.280000
    ]),
    # French 1988-1990 tables
    'TD8890_H': np.array([
        0.001200, 0.001300, 0.001600, 0.002100, 0.002800, 0.004100,
        0.006200, 0.009800, 0.015200, 0.024000, 0.039000, 0.063000,
        0.105000, 0.170000, 0.265000, 0.380000, 0.490000
    ]),
    'TD8890_F': np.array([
        0.000520, 0.000490, 0.000540, 0.000660, 0.000900, 0.001300,
        0.002000, 0.003200, 0.005100, 0.008200, 0.013500, 0.023000,
        0.042000, 0.079000, 0.147000, 0.240000, 0.360000
    ]),
    # French annuity table 1993 (males only — use same ages)
    'TPRV93': np.array([
        0.000500, 0.000520, 0.000660, 0.000870, 0.001100, 0.001590,
        0.002380, 0.003710, 0.005690, 0.008920, 0.014410, 0.023900,
        0.042030, 0.072180, 0.120290, 0.186000, 0.252000
    ]),
}

# Canonical name -> internal key mapping
_TABLE_ALIASES = {
    'TH0002': 'TH0002', 'TH00-02': 'TH0002', 'TH': 'TH0002',
    'TF0002': 'TF0002', 'TF00-02': 'TF0002', 'TF': 'TF0002',
    'TD8890': None,      # needs sexe
    'TD88-90': None,
    'TPRV93': 'TPRV93', 'TPRV': 'TPRV93',
}

_INTEGER_AGES = np.arange(20, 101, dtype=int)


def _interpolate_table(sparse_qx: np.ndarray) -> np.ndarray:
    """Log-interpolate sparse qx to all integer ages 20-100."""
    return np.exp(np.interp(
        _INTEGER_AGES.astype(float), _SPARSE_AGES, np.log(sparse_qx)
    ))


# Pre-compute integer-age versions of all tables
_TABLES_FULL = {k: _interpolate_table(v) for k, v in _TABLES.items()}


# ---------------------------------------------------------------------------
# load_reference_table
# ---------------------------------------------------------------------------

def load_reference_table(name: str = 'TH0002', sexe: str = 'H') -> pd.DataFrame:
    """
    WHEN TO USE:
        To obtain a standard French reference mortality table for comparison.
        Embedded in the library — no external file needed.

    INPUTS:
        name : str — Table identifier. Supported values:
                     'TH0002' / 'TF0002' : French experience 2000-2002
                     'TD8890'            : French experience 1988-1990
                     'TPRV93'            : French annuity table 1993
        sexe : str — 'H' (male) or 'F' (female). Used to select TH/TF or TD/H/F
                     variants. Ignored for TPRV93 (male-only).

    OUTPUTS:
        DataFrame with columns:
            age        : int   — Integer ages 20-100.
            qx_ref     : float — Mortality probability qx (log-interpolated).
            table_name : str   — Name of the table.
    """
    name_upper = name.upper().strip()

    # Resolve alias
    if name_upper in ('TD8890', 'TD88-90', 'TD88_90'):
        key = 'TD8890_H' if sexe == 'H' else 'TD8890_F'
        display_name = f'TD8890_{sexe}'
    elif name_upper in ('TH0002', 'TH00-02', 'TH00_02'):
        key = 'TH0002'
        display_name = 'TH0002'
    elif name_upper in ('TF0002', 'TF00-02', 'TF00_02'):
        key = 'TF0002'
        display_name = 'TF0002'
    elif name_upper in ('TPRV93', 'TPRV'):
        key = 'TPRV93'
        display_name = 'TPRV93'
    else:
        # Auto-select by sexe from common names
        if sexe == 'F' and name_upper.startswith('T'):
            key = 'TF0002'
            display_name = 'TF0002'
        else:
            key = 'TH0002'
            display_name = 'TH0002'
        print(f"[load_reference_table] Unknown table '{name}', "
              f"falling back to {display_name}.")

    if key not in _TABLES_FULL:
        raise ValueError(f"[load_reference_table] Internal key '{key}' not found.")

    qx_arr = _TABLES_FULL[key]
    df = pd.DataFrame({
        'age': _INTEGER_AGES,
        'qx_ref': qx_arr,
        'table_name': display_name,
    })

    print(f"[load_reference_table] Table '{display_name}' loaded: "
          f"ages 20-100, qx range [{qx_arr.min():.6f}, {qx_arr.max():.4f}]")
    return df


# ---------------------------------------------------------------------------
# abatement_factors
# ---------------------------------------------------------------------------

def abatement_factors(exposure_table: pd.DataFrame,
                      qx_exp_col: str = None,
                      reference_table: pd.DataFrame = None,
                      reference_name: str = 'TH0002',
                      sexe: str = 'H') -> tuple:
    """
    WHEN TO USE:
        To quantify by how much the portfolio mortality differs from the
        reference table at each age. A factor < 1 means the portfolio is
        less mortal than the reference.

    INPUTS:
        exposure_table  : DataFrame       — Must have columns: age, qx_exp_col.
        qx_exp_col      : str             — Smoothed experience qx column.
        reference_table : DataFrame|None  — Output of load_reference_table().
                                            If None, loaded automatically.
        reference_name  : str             — Table name (passed to load_reference_table).
        sexe            : str             — 'H' or 'F'.

    OUTPUTS:
        (DataFrame, dict) where:
          - DataFrame columns: age, qx_exp, qx_ref, abatement_factor
          - dict keys: global_factor, min_factor, max_factor,
                       age_min_factor, age_max_factor
    """
    if reference_table is None:
        reference_table = load_reference_table(name=reference_name, sexe=sexe)

    t = exposure_table.copy()
    if qx_exp_col is None:
        for candidate in ('q_x_lisse', 'q_x_brut', 'qx'):
            if candidate in t.columns:
                qx_exp_col = candidate
                break
    if qx_exp_col not in t.columns:
        raise ValueError(f"[abatement_factors] Column '{qx_exp_col}' not found.")

    ref = reference_table[['age', 'qx_ref']].copy()
    merged = t[['age', qx_exp_col]].rename(columns={qx_exp_col: 'qx_exp'}).merge(
        ref, on='age', how='inner')
    merged = merged.dropna(subset=['qx_exp', 'qx_ref'])
    merged = merged[merged['qx_ref'] > 0].copy()

    merged['abatement_factor'] = merged['qx_exp'] / merged['qx_ref']

    alpha = merged['abatement_factor']
    global_f = float(merged['qx_exp'].sum() / merged['qx_ref'].sum())
    min_f = float(alpha.min())
    max_f = float(alpha.max())
    age_min_f = int(merged.loc[alpha.idxmin(), 'age'])
    age_max_f = int(merged.loc[alpha.idxmax(), 'age'])

    summary = {
        'global_factor': global_f,
        'min_factor': min_f,
        'max_factor': max_f,
        'age_min_factor': age_min_f,
        'age_max_factor': age_max_f,
    }

    _LOGGER.log("abatement_factors",
                f"Facteurs d'abattement vs référence : global={global_f:.4f}, "
                f"min={min_f:.4f} (âge {age_min_f}), max={max_f:.4f} (âge {age_max_f})",
                {"global_factor": round(global_f, 4), "min_factor": round(min_f, 4),
                 "max_factor": round(max_f, 4), "age_min_factor": age_min_f,
                 "age_max_factor": age_max_f})
    print(f"[abatement_factors] Global factor: {global_f:.4f} | "
          f"Min: {min_f:.4f} (age {age_min_f}) | "
          f"Max: {max_f:.4f} (age {age_max_f})")
    if global_f < 0.90:
        print("  Portfolio is significantly less mortal than the reference (alpha < 90%).")
    elif global_f > 1.10:
        print("  Portfolio shows excess mortality vs the reference (alpha > 110%).")
    else:
        print("  Portfolio mortality is in line with the reference (alpha within 90-110%).")

    return merged[['age', 'qx_exp', 'qx_ref', 'abatement_factor']].reset_index(drop=True), summary


# ---------------------------------------------------------------------------
# logit_regression
# ---------------------------------------------------------------------------

def logit_regression(exposure_table: pd.DataFrame,
                     qx_exp_col: str = None,
                     reference_table: pd.DataFrame = None,
                     reference_name: str = 'TH0002',
                     sexe: str = 'H') -> dict:
    """
    WHEN TO USE:
        When the abatement factor (qx_exp / qx_ref) is not constant across ages
        — for instance, when the portfolio is younger or older than the reference.
        logit regression allows fitting a linear relationship on the logit scale.

    INPUTS:
        exposure_table  : DataFrame       — Must have columns: age, qx_exp_col.
        qx_exp_col      : str             — Smoothed experience qx column.
        reference_table : DataFrame|None  — Output of load_reference_table().
        reference_name  : str             — Used if reference_table is None.
        sexe            : str             — 'H' or 'F'.

    OUTPUTS:
        dict with keys:
            a             : float      — Intercept of logit(qx_exp) = a + b * logit(qx_ref).
            b             : float      — Slope.
            r_squared     : float      — OLS R² on logit scale.
            fitted_qx     : np.ndarray — Fitted qx values at each age.
            ages          : np.ndarray — Corresponding ages.
            interpretation : str       — Human-readable explanation of a and b.
    """
    if reference_table is None:
        reference_table = load_reference_table(name=reference_name, sexe=sexe)

    t = exposure_table.copy()
    if qx_exp_col is None:
        for candidate in ('q_x_lisse', 'q_x_brut', 'qx'):
            if candidate in t.columns:
                qx_exp_col = candidate
                break
    if qx_exp_col not in t.columns:
        raise ValueError(f"[logit_regression] Column '{qx_exp_col}' not found.")

    ref = reference_table[['age', 'qx_ref']].copy()
    merged = t[['age', qx_exp_col]].rename(columns={qx_exp_col: 'qx_exp'}).merge(
        ref, on='age', how='inner')
    merged = merged.dropna(subset=['qx_exp', 'qx_ref'])
    valid = (merged['qx_exp'] > 0) & (merged['qx_exp'] < 1) & \
            (merged['qx_ref'] > 0) & (merged['qx_ref'] < 1)
    merged = merged[valid].copy()

    if len(merged) < 3:
        raise ValueError("[logit_regression] Not enough valid age points for regression.")

    def _logit(p):
        return np.log(p / (1.0 - p))

    logit_exp = _logit(merged['qx_exp'].values)
    logit_ref = _logit(merged['qx_ref'].values)

    X = np.column_stack([np.ones(len(logit_ref)), logit_ref])
    coeffs, _, _, _ = np.linalg.lstsq(X, logit_exp, rcond=None)
    a, b = coeffs

    fitted_logit = a + b * logit_ref
    ss_res = np.sum((logit_exp - fitted_logit) ** 2)
    ss_tot = np.sum((logit_exp - logit_exp.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    fitted_qx = 1.0 / (1.0 + np.exp(-fitted_logit))

    # Interpretation
    if abs(b - 1.0) < 0.05 and abs(a) < 0.1:
        interpretation = (
            f"a={a:.4f}, b={b:.4f}: nearly identical shape to reference (b≈1, a≈0). "
            f"A constant abatement factor is appropriate."
        )
    elif b > 1.05:
        interpretation = (
            f"a={a:.4f}, b={b:.4f}: portfolio diverges from reference at high ages "
            f"(b={b:.3f} > 1). Abatement factor increases with age."
        )
    elif b < 0.95:
        interpretation = (
            f"a={a:.4f}, b={b:.4f}: portfolio converges toward reference at high ages "
            f"(b={b:.3f} < 1). Abatement factor decreases with age."
        )
    else:
        interpretation = (
            f"a={a:.4f}, b={b:.4f}: moderate shape difference vs reference. "
            f"R²={r2:.4f}."
        )

    print(f"[logit_regression] a={a:.4f}, b={b:.4f}, R²={r2:.4f}")
    print(f"  {interpretation}")
    _LOGGER.log("logit_regression",
                f"Régression logit : a={a:.4f}, b={b:.4f}, R²={r2:.4f} — {interpretation}",
                {"a_intercept": round(float(a), 4), "b_slope": round(float(b), 4),
                 "r_squared": round(float(r2), 4), "interpretation": interpretation})
    return {
        'a': float(a),
        'b': float(b),
        'r_squared': float(r2),
        'fitted_qx': fitted_qx,
        'ages': merged['age'].values.astype(int),
        'interpretation': interpretation,
    }


# ---------------------------------------------------------------------------
# export_table
# ---------------------------------------------------------------------------

def export_table(exposure_table: pd.DataFrame,
                 qx_col: str = None,
                 file_path: str = None,
                 sexe: str = 'H',
                 smr: float = None) -> str:
    """
    WHEN TO USE:
        Final step. Exports the complete experience mortality table to CSV.
        Computes reference qx and local SMR if not already present.

    INPUTS:
        exposure_table : DataFrame  — Must have columns: age, E_x, D_x, q_x_brut,
                                      and qx_col.
        qx_col         : str        — Smoothed qx column to export (default 'q_x_lisse').
        file_path      : str|None   — Destination file path. If None, uses
                                      'table_mortalite_{sexe}.csv' in current directory.
        sexe           : str        — 'H' or 'F' (used for file name and reference lookup).
        smr            : float|None — Global SMR to print in output; computed if None.

    OUTPUTS:
        str — Absolute path of the saved CSV file.
    """
    ref_df = load_reference_table(name='TH0002' if sexe == 'H' else 'TF0002', sexe=sexe)
    ref_map = dict(zip(ref_df['age'], ref_df['qx_ref']))

    t = exposure_table.copy()

    # Auto-detect qx column
    if qx_col is None:
        for candidate in ('q_x_lisse', 'q_x_brut', 'qx'):
            if candidate in t.columns:
                qx_col = candidate
                break
        else:
            raise ValueError(f"[export_table] No qx column found. Available: {list(t.columns)}")

    # Ensure required columns
    missing_cols = [c for c in ['age', 'E_x', 'D_x', qx_col]
                    if c not in t.columns]
    if missing_cols:
        raise ValueError(f"[export_table] Missing columns: {missing_cols}")

    t = t.rename(columns={qx_col: 'q_x_lisse'}) if qx_col != 'q_x_lisse' else t

    t['q_x_ref'] = t['age'].map(ref_map)
    t['SMR_local'] = np.where(
        t['q_x_ref'] > 0,
        t['D_x'] / (t['E_x'] * t['q_x_ref']),
        np.nan
    )

    output_cols = ['age', 'E_x', 'D_x', 'q_x_brut', 'q_x_lisse', 'q_x_ref', 'SMR_local']
    existing = [c for c in output_cols if c in t.columns]
    table_out = t[existing].copy()

    for col in ['q_x_brut', 'q_x_lisse', 'q_x_ref', 'SMR_local']:
        if col in table_out.columns:
            table_out[col] = table_out[col].round(6)

    if file_path is None:
        file_path = os.path.join(os.getcwd(),
                                 f'table_mortalite_{sexe}.csv')

    file_path = os.path.abspath(file_path)
    table_out.to_csv(file_path, index=False)

    # Summary
    n_covered = (table_out['E_x'] > 0).sum()
    lisse_valid = table_out['q_x_lisse'].dropna()
    qx_min = lisse_valid.min() if len(lisse_valid) > 0 else np.nan
    qx_max = lisse_valid.max() if len(lisse_valid) > 0 else np.nan

    if smr is None and 'SMR_local' in table_out.columns:
        d_obs = table_out['D_x'].sum()
        d_exp = (table_out['E_x'] * table_out['q_x_ref']).sum()
        smr = d_obs / d_exp if d_exp > 0 else np.nan

    print(f"[export_table] Saved: {file_path}")
    print(f"  Ages covered : {int(table_out['age'].min())}-{int(table_out['age'].max())} "
          f"({n_covered} ages with E_x > 0)")
    if not np.isnan(qx_min):
        print(f"  qx lisse     : [{qx_min:.6f}, {qx_max:.6f}]")
    if smr is not None and not np.isnan(smr):
        print(f"  SMR global   : {smr:.4f}")
    _LOGGER.log("export_table",
                f"Table exportée vers {file_path} : "
                f"{int(table_out['age'].min())}–{int(table_out['age'].max())} ans, "
                f"{n_covered} âges avec exposition, SMR global={smr:.4f}" if smr else
                f"Table exportée vers {file_path}",
                {"file_path": file_path, "n_ages_couverts": int(n_covered),
                 "age_min": int(table_out['age'].min()), "age_max": int(table_out['age'].max()),
                 "qx_lisse_min": round(float(qx_min), 6) if not np.isnan(qx_min) else None,
                 "qx_lisse_max": round(float(qx_max), 6) if not np.isnan(qx_max) else None,
                 "SMR_global": round(float(smr), 4) if smr and not np.isnan(smr) else None})
    return file_path
