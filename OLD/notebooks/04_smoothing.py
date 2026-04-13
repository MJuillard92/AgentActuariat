"""
04_smoothing.py
===============
Actuarial mortality rate smoothing library.
Methods: Whittaker-Henderson, Gompertz, Makeham, Spline, Local Polynomial.
All functions accept a qx_table DataFrame and return a standardised result dict.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

warnings.filterwarnings('ignore')

try:
    from actuary_logger import LOGGER as _LOGGER
except ImportError:
    class _NoLogger:
        def log(self, *a, **k): pass
    _LOGGER = _NoLogger()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _count_non_monotone(qx_array: np.ndarray, ages: np.ndarray,
                        age_start: int = 40) -> int:
    """Count non-monotone steps in qx for ages >= age_start."""
    mask = ages >= age_start
    q = qx_array[mask]
    if len(q) < 2:
        return 0
    return int((np.diff(q) < 0).sum())


def _prepare_log_qx(qx_table: pd.DataFrame,
                    age_min_fit=None,
                    age_max_fit=None,
                    qx_col: str = None) -> tuple:
    """Extract valid (age, log_qx, weight) arrays from a qx_table.

    Auto-detects the qx column if qx_col is None:
    tries 'q_x_brut', 'qx', 'q_x_lisse' in that order.
    """
    t = qx_table.copy()
    if qx_col is None:
        for candidate in ('q_x_brut', 'qx', 'q_x_lisse'):
            if candidate in t.columns:
                qx_col = candidate
                break
        else:
            raise ValueError(
                f"[_prepare_log_qx] No qx column found. "
                f"Available columns: {list(t.columns)}")
    if qx_col not in t.columns:
        raise ValueError(
            f"[_prepare_log_qx] Column '{qx_col}' not found. "
            f"Available: {list(t.columns)}")
    if age_min_fit is not None:
        t = t[t['age'] >= age_min_fit]
    if age_max_fit is not None:
        t = t[t['age'] <= age_max_fit]
    valid = t[qx_col].notna() & (t['E_x'] > 0) & (t[qx_col] > 0)
    ages = t.loc[valid, 'age'].values.astype(float)
    log_qx = np.log(t.loc[valid, qx_col].values.clip(1e-9))
    weights = t.loc[valid, 'E_x'].values
    return ages, log_qx, weights


# ---------------------------------------------------------------------------
# smooth_whittaker
# ---------------------------------------------------------------------------

def smooth_whittaker(qx_table: pd.DataFrame,
                     lambda_wh: float = 100.0,
                     d: int = 2,
                     age_min_fit: int = None,
                     age_max_fit: int = None) -> dict:
    """
    WHEN TO USE:
        Dense data (E_x > 10 for most ages), no parametric assumption needed,
        and a good general-purpose smoother is required. This is the default
        choice for most experience mortality studies.

    INPUTS:
        qx_table    : DataFrame — Must have columns: age, E_x, D_x, q_x_brut.
                                  Typically output of crude_rates_central().
        lambda_wh   : float     — Smoothing penalty (default 100). Higher = smoother.
        d           : int       — Order of difference penalty (default 2 = curvature).
        age_min_fit : int|None  — Restrict fitting to ages >= this value.
        age_max_fit : int|None  — Restrict fitting to ages <= this value.

    OUTPUTS:
        dict with keys:
            ages                  : np.ndarray — Integer ages.
            qx_smoothed           : np.ndarray — Smoothed qx values.
            method                : str        — 'whittaker_henderson'.
            params                : dict       — {'lambda': lambda_wh, 'd': d}.
            n_non_monotone_after_40 : int      — Count of inversions for age >= 40.
    """
    ages, log_qx, weights = _prepare_log_qx(
        qx_table, age_min_fit, age_max_fit)

    if len(ages) < d + 2:
        raise ValueError(
            f"[smooth_whittaker] Not enough valid data points ({len(ages)}) "
            f"to fit (need at least {d+2}).")

    n = len(log_qx)
    W = np.diag(weights)
    D = np.diff(np.eye(n), n=d, axis=0)
    A = W + lambda_wh * D.T @ D
    log_smooth = np.linalg.solve(A, W @ log_qx)
    qx_smooth = np.exp(log_smooth)

    n_nm = _count_non_monotone(qx_smooth, ages.astype(int), age_start=40)

    print(f"[smooth_whittaker] lambda={lambda_wh}, d={d}, "
          f"n_ages_fitted={n}, "
          f"non_monotone_after_40={n_nm}")
    if n_nm > 0:
        print(f"  WARNING: {n_nm} non-monotone steps after age 40 — "
              f"consider increasing lambda_wh.")
    _LOGGER.log("smooth_whittaker",
                f"Lissage Whittaker-Henderson : λ={lambda_wh}, {n} âges lissés, "
                f"{n_nm} inversions après 40 ans",
                {"lambda_wh": lambda_wh, "d": d, "n_ages": n,
                 "n_non_monotone_after_40": n_nm,
                 "qx_age_50": round(float(qx_smooth[ages.astype(int) == 50][0]), 6) if 50 in ages.astype(int) else None,
                 "qx_age_65": round(float(qx_smooth[ages.astype(int) == 65][0]), 6) if 65 in ages.astype(int) else None})
    return {
        'ages': ages.astype(int),
        'qx_smoothed': qx_smooth,
        'method': 'whittaker_henderson',
        'params': {'lambda': lambda_wh, 'd': d},
        'n_non_monotone_after_40': n_nm,
    }


# ---------------------------------------------------------------------------
# smooth_gompertz
# ---------------------------------------------------------------------------

def smooth_gompertz(qx_table: pd.DataFrame,
                    age_min_fit: int = 40,
                    age_max_fit: int = 90) -> dict:
    """
    WHEN TO USE:
        Sparse data at high ages, or when extrapolation beyond the observed range
        is required. Gompertz assumes log-linear increase of mortality with age,
        which is a good approximation for ages 40+.

    INPUTS:
        qx_table    : DataFrame — Must have columns: age, E_x, D_x, q_x_brut.
        age_min_fit : int       — Lower fitting boundary (default 40).
        age_max_fit : int       — Upper fitting boundary / extrapolation target (default 90).

    OUTPUTS:
        dict with keys:
            ages                  : np.ndarray — Integer ages (age_min_fit to age_max_fit).
            qx_smoothed           : np.ndarray — Fitted qx values.
            method                : str        — 'gompertz'.
            params                : dict       — {'a': float, 'b': float}
                                                 where log(mu_x) = a + b*x.
            r_squared             : float      — Coefficient of determination on log scale.
            n_non_monotone_after_40 : int.
    """
    ages, log_qx, weights = _prepare_log_qx(
        qx_table, age_min_fit, age_max_fit)

    if len(ages) < 3:
        raise ValueError(
            f"[smooth_gompertz] Not enough valid data points ({len(ages)}) "
            f"in age range {age_min_fit}-{age_max_fit}.")

    # Weighted OLS: log(qx) = a + b * age
    X = np.column_stack([np.ones_like(ages), ages])
    W_diag = np.sqrt(weights)
    Xw = X * W_diag[:, None]
    yw = log_qx * W_diag
    coeffs, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    a, b = coeffs

    # R-squared
    log_fitted = a + b * ages
    ss_res = np.sum(weights * (log_qx - log_fitted) ** 2)
    ss_tot = np.sum(weights * (log_qx - np.average(log_qx, weights=weights)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Extrapolate to full range
    out_ages = np.arange(age_min_fit, age_max_fit + 1, dtype=float)
    mu_fitted = np.exp(a + b * out_ages)
    qx_smooth = 1.0 - np.exp(-mu_fitted)

    n_nm = _count_non_monotone(qx_smooth, out_ages.astype(int), age_start=40)

    print(f"[smooth_gompertz] a={a:.4f}, b={b:.4f}, R²={r2:.4f}, "
          f"ages {age_min_fit}-{age_max_fit}, non_monotone_after_40={n_nm}")

    return {
        'ages': out_ages.astype(int),
        'qx_smoothed': qx_smooth,
        'method': 'gompertz',
        'params': {'a': a, 'b': b},
        'r_squared': r2,
        'n_non_monotone_after_40': n_nm,
    }


# ---------------------------------------------------------------------------
# smooth_makeham
# ---------------------------------------------------------------------------

def smooth_makeham(qx_table: pd.DataFrame,
                   age_min_fit: int = 30,
                   age_max_fit: int = 90) -> dict:
    """
    WHEN TO USE:
        When accidental/background mortality is non-negligible at younger ages.
        The Makeham term A adds a constant hazard on top of the Gompertz component,
        which better captures the 'accident hump' in younger populations.

    INPUTS:
        qx_table    : DataFrame — Must have columns: age, E_x, D_x, q_x_brut.
        age_min_fit : int       — Lower fitting boundary (default 30).
        age_max_fit : int       — Upper fitting boundary (default 90).

    OUTPUTS:
        dict with keys:
            ages                  : np.ndarray — Integer ages.
            qx_smoothed           : np.ndarray — Fitted qx values.
            method                : str        — 'makeham'.
            params                : dict       — {'A': float, 'B': float, 'c': float}
                                                 where mu_x = A + B * exp(c * x).
            n_non_monotone_after_40 : int.
    """
    ages, log_qx, weights = _prepare_log_qx(
        qx_table, age_min_fit, age_max_fit)

    if len(ages) < 4:
        raise ValueError(
            f"[smooth_makeham] Not enough valid data points ({len(ages)}) "
            f"in age range {age_min_fit}-{age_max_fit}.")

    qx_vals = np.exp(log_qx)

    # Gompertz initial guess for warm-start
    X = np.column_stack([np.ones_like(ages), ages])
    W_diag = np.sqrt(weights)
    Xw = X * W_diag[:, None]
    yw = log_qx * W_diag
    g_coeffs, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    a0, b0 = g_coeffs
    A0 = 1e-4
    B0 = max(np.exp(a0), 1e-6)
    c0 = max(b0, 0.05)

    def makeham_mu(x, A, B, c):
        return np.clip(A + B * np.exp(c * x), 1e-9, None)

    def makeham_qx(x, A, B, c):
        mu = makeham_mu(x, A, B, c)
        return 1.0 - np.exp(-mu)

    try:
        popt, _ = curve_fit(
            makeham_qx,
            ages, qx_vals,
            p0=[A0, B0, c0],
            sigma=1.0 / np.sqrt(np.maximum(weights, 1e-6)),
            bounds=([0, 1e-9, 0.01], [0.1, 10.0, 0.5]),
            maxfev=10_000,
        )
        A_fit, B_fit, c_fit = popt
    except RuntimeError as e:
        print(f"[smooth_makeham] curve_fit did not converge: {e}. "
              f"Falling back to Gompertz (A=0).")
        A_fit, B_fit, c_fit = 0.0, B0, c0

    out_ages = np.arange(age_min_fit, age_max_fit + 1, dtype=float)
    qx_smooth = makeham_qx(out_ages, A_fit, B_fit, c_fit)

    n_nm = _count_non_monotone(qx_smooth, out_ages.astype(int), age_start=40)

    print(f"[smooth_makeham] A={A_fit:.6f}, B={B_fit:.6f}, c={c_fit:.4f}, "
          f"ages {age_min_fit}-{age_max_fit}, non_monotone_after_40={n_nm}")

    return {
        'ages': out_ages.astype(int),
        'qx_smoothed': qx_smooth,
        'method': 'makeham',
        'params': {'A': A_fit, 'B': B_fit, 'c': c_fit},
        'n_non_monotone_after_40': n_nm,
    }


# ---------------------------------------------------------------------------
# smooth_spline
# ---------------------------------------------------------------------------

def smooth_spline(qx_table: pd.DataFrame,
                  smoothing_factor: float = None,
                  age_min_fit: int = None,
                  age_max_fit: int = None) -> dict:
    """
    WHEN TO USE:
        Dense data where local flexibility is more important than parsimony.
        More flexible than Whittaker-Henderson, less constrained than Gompertz.
        Uses scipy's UnivariateSpline on log(qx), weighted by exposure.

    INPUTS:
        qx_table        : DataFrame  — Must have columns: age, E_x, D_x, q_x_brut.
        smoothing_factor : float|None — Smoothing factor s passed to UnivariateSpline.
                                        None = cross-validated (scipy default).
        age_min_fit     : int|None   — Lower fitting boundary.
        age_max_fit     : int|None   — Upper fitting boundary.

    OUTPUTS:
        dict with keys:
            ages                  : np.ndarray — Integer ages.
            qx_smoothed           : np.ndarray — Smoothed qx values.
            method                : str        — 'spline'.
            params                : dict       — {'smoothing_factor': s, 'knots': int}.
            n_non_monotone_after_40 : int.
    """
    ages, log_qx, weights = _prepare_log_qx(
        qx_table, age_min_fit, age_max_fit)

    if len(ages) < 5:
        raise ValueError(
            f"[smooth_spline] Not enough valid data points ({len(ages)}) for spline fitting.")

    # Normalise weights for scipy (they act as 1/sigma)
    w_norm = weights / weights.max()

    spl = UnivariateSpline(
        ages, log_qx,
        w=w_norm,
        s=smoothing_factor,
        k=3,
        ext=3,          # extrapolate with boundary value
    )

    out_ages = np.arange(int(ages.min()), int(ages.max()) + 1, dtype=float)
    log_smooth = spl(out_ages)
    qx_smooth = np.exp(log_smooth)

    n_knots = len(spl.get_knots())
    n_nm = _count_non_monotone(qx_smooth, out_ages.astype(int), age_start=40)

    print(f"[smooth_spline] smoothing_factor={smoothing_factor}, "
          f"knots={n_knots}, n_ages_fitted={len(ages)}, "
          f"non_monotone_after_40={n_nm}")

    return {
        'ages': out_ages.astype(int),
        'qx_smoothed': qx_smooth,
        'method': 'spline',
        'params': {'smoothing_factor': smoothing_factor, 'knots': n_knots},
        'n_non_monotone_after_40': n_nm,
    }


# ---------------------------------------------------------------------------
# smooth_local_polynomial
# ---------------------------------------------------------------------------

def smooth_local_polynomial(qx_table: pd.DataFrame,
                            bandwidth: int = 5,
                            degree: int = 2) -> dict:
    """
    WHEN TO USE:
        When local structure matters more than global fit — for example, when
        mortality exhibits heterogeneous behaviour across age ranges. LOESS-style
        smoother implemented manually using numpy (no external LOESS dependency).

    INPUTS:
        qx_table  : DataFrame — Must have columns: age, E_x, D_x, q_x_brut.
        bandwidth : int       — Half-window size in ages (default 5).
                                Actual window = 2*bandwidth+1 ages.
        degree    : int       — Polynomial degree (default 2 = quadratic).

    OUTPUTS:
        dict with keys:
            ages                  : np.ndarray — Integer ages.
            qx_smoothed           : np.ndarray — Smoothed qx values.
            method                : str        — 'local_polynomial'.
            params                : dict       — {'bandwidth': bandwidth, 'degree': degree}.
            n_non_monotone_after_40 : int.
    """
    ages, log_qx, weights = _prepare_log_qx(
        qx_table, None, None)

    if len(ages) < degree + 2:
        raise ValueError(
            f"[smooth_local_polynomial] Not enough data points ({len(ages)}).")

    n = len(ages)
    log_smooth = np.empty(n)

    for i in range(n):
        # Local window indices
        lo = max(0, i - bandwidth)
        hi = min(n, i + bandwidth + 1)
        idx = np.arange(lo, hi)

        x_loc = ages[idx] - ages[i]           # centre on current age
        y_loc = log_qx[idx]
        w_loc = weights[idx]

        # Tricubic distance weights (LOESS kernel)
        max_dist = max(abs(x_loc).max(), 1e-9)
        u = np.abs(x_loc) / max_dist
        kernel = np.where(u < 1, (1.0 - u ** 3) ** 3, 0.0)
        combined_w = w_loc * kernel + 1e-9

        # Weighted polynomial fit
        X = np.column_stack([x_loc ** k for k in range(degree + 1)])
        W_sqrt = np.sqrt(combined_w)
        Xw = X * W_sqrt[:, None]
        yw = y_loc * W_sqrt

        try:
            coeffs, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
            log_smooth[i] = coeffs[0]   # value at x_loc = 0
        except np.linalg.LinAlgError:
            log_smooth[i] = log_qx[i]  # fallback

    qx_smooth = np.exp(log_smooth)
    n_nm = _count_non_monotone(qx_smooth, ages.astype(int), age_start=40)

    print(f"[smooth_local_polynomial] bandwidth={bandwidth}, degree={degree}, "
          f"n_ages={n}, non_monotone_after_40={n_nm}")

    return {
        'ages': ages.astype(int),
        'qx_smoothed': qx_smooth,
        'method': 'local_polynomial',
        'params': {'bandwidth': bandwidth, 'degree': degree},
        'n_non_monotone_after_40': n_nm,
    }
