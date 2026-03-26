"""
08_visualization.py
===================
Actuarial visualization library for experience mortality tables.
All functions return bytes (PNG). All use a cream/light theme.
"""

import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

try:
    from actuary_logger import LOGGER as _LOGGER
except ImportError:
    class _NoLogger:
        def log(self, *a, **k): pass
    _LOGGER = _NoLogger()

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
_BG_COLOR = '#FBF8F1'
_GRID_COLOR = '#E8E3D8'
_ACCENT_BLUE = '#2C5F8A'
_ACCENT_RED = '#C0392B'
_ACCENT_GREEN = '#27AE60'
_ACCENT_ORANGE = '#E67E22'

# ---------------------------------------------------------------------------
# Reference table (embedded)
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


def _qx_ref_array(ages, sexe='H'):
    tbl = _QX_H if sexe == 'H' else _QX_F
    return np.exp(np.interp(
        np.clip(np.asarray(ages, dtype=float), 20, 100), _AGES_REF, np.log(tbl)))


def _apply_theme(fig, axes_list):
    """Apply cream background theme to figure and all axes."""
    fig.patch.set_facecolor(_BG_COLOR)
    for ax in axes_list:
        ax.set_facecolor(_BG_COLOR)
        ax.grid(True, color=_GRID_COLOR, linewidth=0.8, alpha=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def _fig_to_bytes(fig) -> bytes:
    """Render figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=_BG_COLOR)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# plot_exposure_by_age
# ---------------------------------------------------------------------------

def plot_exposure_by_age(exposure_table: pd.DataFrame,
                         title_suffix: str = '') -> bytes:
    """
    WHEN TO USE:
        Immediately after compute_exposure_by_age to visualise data density.
        Red bars highlight ages with insufficient data for credible estimation.

    INPUTS:
        exposure_table : DataFrame — Must have columns: age, E_x.
        title_suffix   : str       — Optional text appended to the chart title.

    OUTPUTS:
        bytes — PNG image (render with IPython.display.Image or save to file).
    """
    t = exposure_table.copy()
    ages = t['age'].values
    ex = t['E_x'].values

    colors = np.where(ex < 10, _ACCENT_RED,
             np.where(ex < 50, _ACCENT_ORANGE, _ACCENT_GREEN))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(ages, ex, color=colors, alpha=0.85, width=0.8, edgecolor='none')

    # Legend proxies
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=_ACCENT_GREEN,  label='High credibility (E_x ≥ 50)'),
        Patch(color=_ACCENT_ORANGE, label='Medium credibility (10 ≤ E_x < 50)'),
        Patch(color=_ACCENT_RED,    label='Low credibility (E_x < 10)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9,
              facecolor=_BG_COLOR, edgecolor=_GRID_COLOR)

    title = f'Central Exposure by Age{" — " + title_suffix if title_suffix else ""}'
    subtitle = (f'Total: {ex.sum():,.0f} person-years | '
                f'Low-credibility ages: {(ex < 10).sum()}')
    ax.set_title(f'{title}\n{subtitle}', fontsize=11, loc='left', pad=8)
    ax.set_xlabel('Age', fontsize=10)
    ax.set_ylabel('Person-years (E_x)', fontsize=10)

    _apply_theme(fig, [ax])
    print(f"[plot_exposure_by_age] Chart rendered.")
    _LOGGER.log("plot_exposure_by_age",
                f"Graphique exposition par âge généré ({ex.sum():,.0f} py total, "
                f"{(ex < 10).sum()} âges faible crédibilité)",
                {"total_exposure_py": round(float(ex.sum()), 1),
                 "n_ages_low_credibility": int((ex < 10).sum())})
    return _fig_to_bytes(fig)


# ---------------------------------------------------------------------------
# plot_deaths_by_age
# ---------------------------------------------------------------------------

def plot_deaths_by_age(exposure_table: pd.DataFrame,
                       title_suffix: str = '') -> bytes:
    """
    WHEN TO USE:
        To inspect the distribution of observed deaths across ages, and spot
        ages with zero or very few deaths that may affect smoothing.

    INPUTS:
        exposure_table : DataFrame — Must have columns: age, D_x.
        title_suffix   : str       — Optional text appended to the chart title.

    OUTPUTS:
        bytes — PNG image.
    """
    t = exposure_table.copy()
    ages = t['age'].values
    dx = t['D_x'].values

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(ages, dx, color=_ACCENT_BLUE, alpha=0.8, width=0.8, edgecolor='none')

    total_d = int(dx.sum())
    zero_d = int((dx == 0).sum())
    title = f'Observed Deaths by Age{" — " + title_suffix if title_suffix else ""}'
    subtitle = f'Total deaths: {total_d:,} | Ages with 0 deaths: {zero_d}'
    ax.set_title(f'{title}\n{subtitle}', fontsize=11, loc='left', pad=8)
    ax.set_xlabel('Age', fontsize=10)
    ax.set_ylabel('Deaths (D_x)', fontsize=10)

    _apply_theme(fig, [ax])
    print(f"[plot_deaths_by_age] Chart rendered.")
    _LOGGER.log("plot_deaths_by_age", "Graphique décès par âge généré",
                {"total_deaths": int(t['D_x'].sum()), "n_ages_zero_deaths": int((t['D_x'] == 0).sum())})
    return _fig_to_bytes(fig)


# ---------------------------------------------------------------------------
# plot_crude_vs_smoothed
# ---------------------------------------------------------------------------

def plot_crude_vs_smoothed(exposure_table: pd.DataFrame,
                           smoothed_dict: dict,
                           sexe: str = 'H',
                           reference_fn=None,
                           title_suffix: str = '') -> bytes:
    """
    WHEN TO USE:
        After smoothing. Visual sanity check to confirm the smoother follows
        the crude rates without over- or under-fitting.

    INPUTS:
        exposure_table : DataFrame — Must have columns: age, E_x, q_x_brut.
        smoothed_dict  : dict      — {'Method Name': qx_array_aligned_to_ages, ...}
                                     or {'Method Name': smooth_*() result dict, ...}
                                     where each value has keys 'ages', 'qx_smoothed'.
        sexe           : str       — 'H' or 'F' for reference table.
        reference_fn   : callable|None — callable(age, sexe) -> float. If None,
                                         uses built-in TH/TF 00-02.
        title_suffix   : str       — Optional text appended to the chart title.

    OUTPUTS:
        bytes — PNG image.
    """
    t = exposure_table.copy()
    valid = t['q_x_brut'].notna() & (t['E_x'] > 0)
    crude_ages = t.loc[valid, 'age'].values
    crude_qx = t.loc[valid, 'q_x_brut'].values

    # Reference
    all_ages = t['age'].values
    if reference_fn is not None:
        qx_ref_vals = np.array([reference_fn(a, sexe) for a in all_ages])
    else:
        qx_ref_vals = _qx_ref_array(all_ages, sexe)

    fig, ax = plt.subplots(figsize=(13, 6))

    # Crude rates scatter
    ax.scatter(crude_ages, crude_qx, s=20, alpha=0.55, color=_ACCENT_BLUE,
               label='Crude rates', zorder=2)

    # Smoothed lines
    palette = [_ACCENT_RED, '#8E44AD', '#16A085', '#D35400', '#2980B9']
    for idx, (method_name, val) in enumerate(smoothed_dict.items()):
        color = palette[idx % len(palette)]
        if isinstance(val, dict):
            sm_ages = np.asarray(val['ages'], dtype=int)
            sm_qx = np.asarray(val['qx_smoothed'], dtype=float)
        else:
            sm_qx = np.asarray(val, dtype=float)
            # Align with valid crude ages if sizes match, else all ages
            if len(sm_qx) == len(crude_ages):
                sm_ages = crude_ages
            else:
                sm_ages = t['age'].values[:len(sm_qx)]
        ax.plot(sm_ages, sm_qx, color=color, linewidth=2,
                label=method_name, zorder=3)

    # Reference line
    ax.plot(all_ages, qx_ref_vals, color='#2C3E50', linewidth=1.5,
            linestyle='--', alpha=0.7,
            label=f'Reference TH/TF 00-02 ({sexe})', zorder=4)

    ax.set_yscale('log')
    ax.set_xlabel('Age', fontsize=10)
    ax.set_ylabel('qx (log scale)', fontsize=10)
    n_methods = len(smoothed_dict)
    title = (f'Crude Rates vs Smoothed Mortality'
             f'{" — " + title_suffix if title_suffix else ""}')
    subtitle = f'{n_methods} smoother(s) shown | Log scale'
    ax.set_title(f'{title}\n{subtitle}', fontsize=11, loc='left', pad=8)
    ax.legend(fontsize=9, facecolor=_BG_COLOR, edgecolor=_GRID_COLOR, loc='upper left')

    _apply_theme(fig, [ax])
    print(f"[plot_crude_vs_smoothed] Chart rendered ({n_methods} smoother(s)).")
    _LOGGER.log("plot_crude_vs_smoothed", f"Graphique taux bruts vs lissés ({n_methods} lisseur(s))",
                {"n_methods": n_methods})
    return _fig_to_bytes(fig)


# ---------------------------------------------------------------------------
# plot_smr_by_age
# ---------------------------------------------------------------------------

def plot_smr_by_age(smr_result: dict,
                    title_suffix: str = '') -> bytes:
    """
    WHEN TO USE:
        After compute_smr(). Visualises where the portfolio deviates from the
        reference — green bars indicate sub-mortality, red bars excess mortality.

    INPUTS:
        smr_result  : dict — Output of compute_smr(). Must have keys:
                             smr_by_decade (DataFrame with columns decade, SMR),
                             smr_global, d_observed, d_expected.
        title_suffix : str — Optional text appended to the chart title.

    OUTPUTS:
        bytes — PNG image.
    """
    smr_decade = smr_result.get('smr_by_decade', pd.DataFrame())
    smr_global = smr_result.get('smr_global', np.nan)
    d_obs = smr_result.get('d_observed', '?')
    d_exp = smr_result.get('d_expected', np.nan)

    if smr_decade.empty or 'SMR' not in smr_decade.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No SMR by decade data available.',
                ha='center', va='center', transform=ax.transAxes)
        _apply_theme(fig, [ax])
        return _fig_to_bytes(fig)

    decades = smr_decade['decade'].values
    smr_vals = smr_decade['SMR'].values.astype(float)
    x = np.arange(len(decades))

    bar_colors = np.where(smr_vals < 0.90, _ACCENT_GREEN,
                 np.where(smr_vals <= 1.10, _ACCENT_BLUE, _ACCENT_RED))

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(x, smr_vals, color=bar_colors, alpha=0.85, width=0.6, edgecolor='none')

    # Reference lines
    ax.axhline(1.0,  color='#2C3E50', linewidth=1.5, linestyle='-',  alpha=0.8, label='SMR = 1.0')
    ax.axhline(0.90, color=_ACCENT_GREEN,  linewidth=1.2, linestyle='--', alpha=0.7, label='0.90')
    ax.axhline(1.10, color=_ACCENT_RED,    linewidth=1.2, linestyle='--', alpha=0.7, label='1.10')

    ax.set_xticks(x)
    ax.set_xticklabels(decades, fontsize=9)
    ax.set_xlabel('Age decade', fontsize=10)
    ax.set_ylabel('SMR', fontsize=10)

    d_exp_str = f'{d_exp:.1f}' if not np.isnan(float(d_exp)) else '?'
    title = f'SMR by Age Decade{" — " + title_suffix if title_suffix else ""}'
    subtitle = (f'Global SMR = {smr_global:.4f} | '
                f'Observed = {d_obs:,} | Expected = {d_exp_str}')
    ax.set_title(f'{title}\n{subtitle}', fontsize=11, loc='left', pad=8)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=_ACCENT_GREEN, label='Sub-mortality (SMR < 0.90)'),
        Patch(color=_ACCENT_BLUE,  label='Normal (0.90 ≤ SMR ≤ 1.10)'),
        Patch(color=_ACCENT_RED,   label='Excess mortality (SMR > 1.10)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9,
              facecolor=_BG_COLOR, edgecolor=_GRID_COLOR)

    _apply_theme(fig, [ax])
    print(f"[plot_smr_by_age] Chart rendered.")
    _LOGGER.log("plot_smr_by_age", "Graphique SMR par âge généré",
                {"smr_global": round(float(smr_result.get("smr_global", 0)), 4)})
    return _fig_to_bytes(fig)


# ---------------------------------------------------------------------------
# plot_confidence_bands
# ---------------------------------------------------------------------------

def plot_confidence_bands(exposure_table: pd.DataFrame,
                          qx_col: str = None,
                          ci_result: pd.DataFrame = None,
                          reference_fn=None,
                          sexe: str = 'H',
                          title_suffix: str = '') -> bytes:
    """
    WHEN TO USE:
        After confidence_intervals(). Shows where uncertainty is high (wide
        bands) and whether the reference falls within the confidence band.

    INPUTS:
        exposure_table : DataFrame       — Must have columns: age, qx_col.
        qx_col         : str             — Smoothed qx column.
        ci_result      : DataFrame|None  — Output of confidence_intervals().
                                           If None, CI is not shown.
        reference_fn   : callable|None   — callable(age, sexe) -> float.
        sexe           : str             — 'H' or 'F'.
        title_suffix   : str             — Optional text appended to the chart title.

    OUTPUTS:
        bytes — PNG image.
    """
    t = exposure_table.copy()
    if qx_col is None:
        for candidate in ('q_x_lisse', 'q_x_brut', 'qx'):
            if candidate in t.columns:
                qx_col = candidate
                break
    if qx_col not in t.columns:
        raise ValueError(f"[plot_confidence_bands] Column '{qx_col}' not found.")

    valid = t[qx_col].notna()
    ages = t.loc[valid, 'age'].values
    qx_vals = t.loc[valid, qx_col].values

    if reference_fn is not None:
        qx_ref_vals = np.array([reference_fn(a, sexe) for a in ages])
    else:
        qx_ref_vals = _qx_ref_array(ages, sexe)

    fig, ax = plt.subplots(figsize=(13, 6))

    # Confidence band
    if ci_result is not None and 'ci_lower' in ci_result.columns:
        ci = ci_result.dropna(subset=['ci_lower', 'ci_upper'])
        ax.fill_between(ci['age'], ci['ci_lower'], ci['ci_upper'],
                        alpha=0.25, color=_ACCENT_BLUE, label='95% CI (Poisson)')

    # Experience line
    ax.plot(ages, qx_vals, color=_ACCENT_RED, linewidth=2,
            label=f'Experience ({qx_col})', zorder=3)

    # Reference line
    ax.plot(ages, qx_ref_vals, color='#2C3E50', linewidth=1.5,
            linestyle='--', alpha=0.75,
            label=f'Reference TH/TF 00-02 ({sexe})', zorder=4)

    ax.set_yscale('log')
    ax.set_xlabel('Age', fontsize=10)
    ax.set_ylabel('qx (log scale)', fontsize=10)
    title = f'Smoothed qx with 95% Confidence Bands{" — " + title_suffix if title_suffix else ""}'
    ax.set_title(title, fontsize=11, loc='left', pad=8)
    ax.legend(fontsize=9, facecolor=_BG_COLOR, edgecolor=_GRID_COLOR, loc='upper left')

    _apply_theme(fig, [ax])
    print(f"[plot_confidence_bands] Chart rendered.")
    _LOGGER.log("plot_confidence_bands", "Graphique intervalles de confiance généré", {})
    return _fig_to_bytes(fig)


# ---------------------------------------------------------------------------
# plot_observed_vs_expected
# ---------------------------------------------------------------------------

def plot_observed_vs_expected(exposure_table: pd.DataFrame,
                              qx_col: str = 'q_x_lisse',
                              reference_fn=None,
                              sexe: str = 'H',
                              title_suffix: str = '') -> bytes:
    """
    WHEN TO USE:
        After smoothing and SMR computation. Scatter of observed vs expected
        deaths highlights systematic over- or under-estimation by age.

    INPUTS:
        exposure_table : DataFrame      — Must have columns: age, E_x, D_x, qx_col.
        qx_col         : str            — Smoothed qx column.
        reference_fn   : callable|None  — callable(age, sexe) -> float.
                                          If None, uses built-in TH/TF 00-02.
        sexe           : str            — 'H' or 'F'.
        title_suffix   : str            — Optional text appended to the chart title.

    OUTPUTS:
        bytes — PNG image.
    """
    t = exposure_table.copy()
    if reference_fn is not None:
        t['qx_ref'] = t['age'].apply(lambda a: reference_fn(a, sexe))
    else:
        t['qx_ref'] = _qx_ref_array(t['age'].values, sexe)

    t['D_exp_ref'] = t['E_x'] * t['qx_ref']
    if qx_col in t.columns:
        t['D_exp_model'] = t['E_x'] * t[qx_col]

    t = t[(t['E_x'] > 0) & (t['D_x'] >= 0)].dropna(subset=['D_exp_ref'])

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter: observed vs reference-expected
    sc = ax.scatter(t['D_exp_ref'], t['D_x'],
                    c=t['age'], cmap='plasma', s=45, alpha=0.8,
                    zorder=3, label='Obs vs Ref. Expected')
    plt.colorbar(sc, ax=ax, label='Age')

    # Model-expected vs reference-expected
    if 'D_exp_model' in t.columns:
        ax.scatter(t['D_exp_ref'], t['D_exp_model'],
                   color=_ACCENT_BLUE, s=25, alpha=0.5, marker='+',
                   label='Model Expected vs Ref. Expected', zorder=4)

    # Diagonal
    max_val = max(t['D_exp_ref'].max(), t['D_x'].max())
    line_range = np.linspace(0, max_val * 1.05, 100)
    ax.plot(line_range, line_range, color='#2C3E50', linewidth=1.5,
            linestyle='--', alpha=0.7, label='Perfect fit')

    ax.set_xlabel(f'Expected deaths (Ref: TH/TF 00-02 {sexe})', fontsize=10)
    ax.set_ylabel('Observed deaths', fontsize=10)
    title = f'Observed vs Expected Deaths{" — " + title_suffix if title_suffix else ""}'
    subtitle = f'Each point = one age | Color = age value'
    ax.set_title(f'{title}\n{subtitle}', fontsize=11, loc='left', pad=8)
    ax.legend(fontsize=9, facecolor=_BG_COLOR, edgecolor=_GRID_COLOR)

    _apply_theme(fig, [ax])
    print(f"[plot_observed_vs_expected] Chart rendered.")
    _LOGGER.log("plot_observed_vs_expected", "Graphique observés vs attendus généré", {})
    return _fig_to_bytes(fig)


# ---------------------------------------------------------------------------
# plot_smr_heatmap
# ---------------------------------------------------------------------------

def plot_smr_heatmap(smr_by_year_age: pd.DataFrame,
                     title_suffix: str = '') -> bytes:
    """
    WHEN TO USE:
        After compute_exposure_by_year followed by an SMR calculation per
        (year, age) cell. Reveals temporal trends in excess/sub-mortality.

    INPUTS:
        smr_by_year_age : DataFrame — index=age, columns=year, values=SMR.
        title_suffix    : str       — Optional text appended to the chart title.

    OUTPUTS:
        bytes — PNG image.
    """
    data = smr_by_year_age.copy()

    if data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No SMR by (year, age) data available.',
                ha='center', va='center', transform=ax.transAxes)
        _apply_theme(fig, [ax])
        return _fig_to_bytes(fig)

    # Diverging colormap centred at 1.0
    max_dev = max(abs(data.values[~np.isnan(data.values)] - 1.0).max(), 0.3)
    vmin, vcenter, vmax = 1.0 - max_dev, 1.0, 1.0 + max_dev
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(10, len(data.columns) * 0.8), 8))
    im = ax.imshow(
        data.values,
        aspect='auto',
        cmap='RdYlGn_r',
        norm=norm,
        interpolation='nearest',
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('SMR', fontsize=10)
    cbar.ax.axhline(y=1.0, color='black', linewidth=1.5)

    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=8)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Age', fontsize=10)

    title = f'SMR Heatmap by Year and Age{" — " + title_suffix if title_suffix else ""}'
    subtitle = 'Green = sub-mortality | Red = excess mortality | White = in line'
    ax.set_title(f'{title}\n{subtitle}', fontsize=11, loc='left', pad=8)

    _apply_theme(fig, [ax])
    print(f"[plot_smr_heatmap] Chart rendered "
          f"({len(data.index)} ages x {len(data.columns)} years).")
    return _fig_to_bytes(fig)


# ---------------------------------------------------------------------------
# plot_survival_curve
# ---------------------------------------------------------------------------

def plot_survival_curve(exposure_table: pd.DataFrame,
                        qx_col: str = None,
                        reference_fn=None,
                        sexe: str = 'H',
                        title_suffix: str = '') -> bytes:
    """
    WHEN TO USE:
        After smoothing. Shows the complete survival function implied by the
        experience table compared to the reference — intuitive for insurance
        practitioners and actuarial presentations.

    INPUTS:
        exposure_table : DataFrame      — Must have columns: age, qx_col.
        qx_col         : str            — Smoothed qx column.
        reference_fn   : callable|None  — callable(age, sexe) -> float.
        sexe           : str            — 'H' or 'F'.
        title_suffix   : str            — Optional text appended to the chart title.

    OUTPUTS:
        bytes — PNG image.
    """
    t = exposure_table.copy().sort_values('age')
    if qx_col is None:
        for candidate in ('q_x_lisse', 'q_x_brut', 'qx'):
            if candidate in t.columns:
                qx_col = candidate
                break
    if qx_col not in t.columns:
        raise ValueError(f"[plot_survival_curve] Column '{qx_col}' not found.")

    valid = t[qx_col].notna()
    ages = t.loc[valid, 'age'].values
    qx_vals = t.loc[valid, qx_col].values.clip(0, 1)

    if reference_fn is not None:
        qx_ref_vals = np.array([reference_fn(a, sexe) for a in ages])
    else:
        qx_ref_vals = _qx_ref_array(ages, sexe)

    # Survival: S(x) = prod_{t=age_min}^{x-1} (1 - qx_t)
    px_exp = 1.0 - qx_vals
    px_ref = 1.0 - np.clip(qx_ref_vals, 0, 1)
    S_exp = np.cumprod(px_exp)
    S_ref = np.cumprod(px_ref)

    # e0 (life expectancy at entry age) as area under survival curve
    e0_exp = np.trapz(S_exp, ages) if len(ages) > 1 else np.nan
    e0_ref = np.trapz(S_ref, ages) if len(ages) > 1 else np.nan

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(ages, S_exp, color=_ACCENT_RED, linewidth=2.5,
            label=f'Experience (from {qx_col})', zorder=3)
    ax.plot(ages, S_ref, color='#2C3E50', linewidth=1.8,
            linestyle='--', alpha=0.8,
            label=f'Reference TH/TF 00-02 ({sexe})', zorder=4)
    ax.fill_between(ages, S_exp, S_ref,
                    where=(S_exp < S_ref), alpha=0.15,
                    color=_ACCENT_RED, label='Higher mortality zone')
    ax.fill_between(ages, S_exp, S_ref,
                    where=(S_exp >= S_ref), alpha=0.15,
                    color=_ACCENT_GREEN, label='Lower mortality zone')

    ax.set_xlabel('Age', fontsize=10)
    ax.set_ylabel('Survival S(x)', fontsize=10)
    ax.set_ylim(0, 1.02)

    e0_str = f'e0_exp={e0_exp:.1f} | e0_ref={e0_ref:.1f}' \
             if not np.isnan(e0_exp) else ''
    title = f'Survival Curve S(x){" — " + title_suffix if title_suffix else ""}'
    subtitle = f'Experience vs Reference TH/TF 00-02 ({sexe}) | {e0_str}'
    ax.set_title(f'{title}\n{subtitle}', fontsize=11, loc='left', pad=8)
    ax.legend(fontsize=9, facecolor=_BG_COLOR, edgecolor=_GRID_COLOR, loc='upper right')

    _apply_theme(fig, [ax])
    print(f"[plot_survival_curve] Chart rendered. "
          f"e0 (experience): {e0_exp:.1f} | e0 (reference): {e0_ref:.1f}")
    _LOGGER.log("plot_survival_curve",
                f"Courbe de survie générée : espérance de vie e0={e0_exp:.1f} ans (expérience) "
                f"vs {e0_ref:.1f} ans (référence)",
                {"e0_experience": round(float(e0_exp), 1), "e0_reference": round(float(e0_ref), 1)})
    return _fig_to_bytes(fig)
