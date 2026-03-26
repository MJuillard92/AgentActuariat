"""
01_data_preparation.py
======================
Actuarial data preparation library for experience mortality table construction.
Functions: load_data, generate_synthetic_data, clean_data, compute_ages, detect_anomalies
"""

import io
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

# Lignes supprimées lors du dernier appel à clean_data (DataFrame avec colonne removal_reason)
df_removed: pd.DataFrame = pd.DataFrame()

# ---------------------------------------------------------------------------
# Reference table: TH/TF 00-02 (French 2000-2002 experience table)
# Sparse knots — interpolated logarithmically in helpers below.
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


def qx_ref(age: float, sexe: str = 'H') -> float:
    """Return TH/TF 00-02 reference qx for a single age (log-interpolated)."""
    tbl = _QX_H if sexe == 'H' else _QX_F
    return float(np.exp(np.interp(float(np.clip(age, 20, 100)), _AGES_REF, np.log(tbl))))


def qx_ref_array(ages, sexe: str = 'H') -> np.ndarray:
    """Return TH/TF 00-02 reference qx for an array of ages (log-interpolated)."""
    tbl = _QX_H if sexe == 'H' else _QX_F
    return np.exp(np.interp(np.clip(np.asarray(ages, dtype=float), 20, 100),
                            _AGES_REF, np.log(tbl)))


# ---------------------------------------------------------------------------
# normalize_column_names — appelée automatiquement par load_data
# ---------------------------------------------------------------------------

# Synonymes connus → nom canonique attendu par le pipeline
_COLUMN_SYNONYMS = {
    # Genre
    "sexe":            "sexe",
    "sex":             "sexe",
    "gender":          "sexe",
    "genre":           "sexe",
    # Cause de sortie
    "cause_sortie":    "cause_sortie",
    "cause":           "cause_sortie",
    "exit_reason":     "cause_sortie",
    "reason":          "cause_sortie",
    # Dates
    "date_naissance":  "date_naissance",
    "birth_date":      "date_naissance",
    "dob":             "date_naissance",
    "date_de_naissance": "date_naissance",
    "date_entree":     "date_entree",
    "entry_date":      "date_entree",
    "date_d_entree":   "date_entree",
    "date_sortie":     "date_sortie",
    "exit_date":       "date_sortie",
    "date_de_sortie":  "date_sortie",
    # Identifiant
    "id":              "id",
    "identifiant":     "id",
    "contract_id":     "id",
    "num_contrat":     "id",
}


def normalize_column_names(df: pd.DataFrame) -> tuple:
    """Normalise les noms de colonnes d'un DataFrame vers les noms attendus par le pipeline.

    Appliqué automatiquement dans load_data() pour qu'un CSV avec des colonnes
    en majuscules (SEXE, DATE_NAISSANCE…) ou des synonymes anglais (gender, dob…)
    fonctionne sans modification.

    Étapes :
      1. Minuscules sur tous les noms de colonnes (SEXE → sexe).
      2. Suppression des espaces en début/fin (\" sexe \" → \"sexe\").
      3. Remplacement des espaces internes par \"_\" (\"date naissance\" → \"date_naissance\").
      4. Correspondance aux synonymes connus (_COLUMN_SYNONYMS).

    Returns:
        (df_normalisé, mapping_dict) où mapping_dict liste les renommages effectués.
    """
    rename_map: dict[str, str] = {}
    for col in df.columns:
        normalised = col.strip().lower().replace(" ", "_").replace("-", "_")
        canonical = _COLUMN_SYNONYMS.get(normalised, normalised)
        if canonical != col:
            rename_map[col] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"[load_data] Colonnes renommées : {rename_map}")
    return df, rename_map


# ---------------------------------------------------------------------------
# _load_flat_file — détection automatique du séparateur CSV/TXT
# ---------------------------------------------------------------------------

def _load_flat_file(path: str, encoding: str = 'utf-8'):
    """Charge un fichier plat (CSV ou TXT) en détectant automatiquement le séparateur.

    Stratégie :
      1. csv.Sniffer sur les 4 premiers Ko pour détecter le séparateur.
      2. Essai forcé avec ';', ',', tabulation, '|'.
      3. Réessaie avec latin-1 / cp1252 si utf-8 échoue.
      4. Fallback ultime : pandas par défaut (latin-1).

    Retourne (DataFrame, fmt_str).
    """
    import csv as _csv

    SEP_CANDIDATES = [';', ',', '\t', '|']
    ENCODINGS = [encoding] if encoding not in ('utf-8', '') else ['utf-8', 'latin-1', 'cp1252']

    def _try(enc, sep):
        return pd.read_csv(path, sep=sep, encoding=enc, engine='python')

    def _sniff(enc):
        try:
            with open(path, encoding=enc, errors='replace') as fh:
                sample = fh.read(4096)
            return _csv.Sniffer().sniff(sample, delimiters=';,\t|').delimiter
        except Exception:
            return None

    for enc in ENCODINGS:
        sniffed = _sniff(enc)
        if sniffed:
            try:
                df = _try(enc, sniffed)
                if len(df.columns) > 1:
                    print(f"[load_data] sep={sniffed!r} enc={enc} détectés automatiquement")
                    return df, f"CSV(sep={sniffed!r})"
            except Exception:
                pass
        for sep in SEP_CANDIDATES:
            try:
                df = _try(enc, sep)
                if len(df.columns) > 1:
                    print(f"[load_data] sep={sep!r} enc={enc}")
                    return df, f"CSV(sep={sep!r})"
            except Exception:
                pass

    # Fallback ultime
    try:
        df = pd.read_csv(path, encoding='latin-1')
        print("[load_data] Chargement fallback (latin-1, sep=',')")
        return df, "CSV(fallback)"
    except Exception as exc:
        raise ValueError(
            f"Impossible de lire '{path}'. "
            f"Formats supportés : CSV/TXT (sep ',' ';' ou tabulation), Excel. "
            f"Erreur : {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

def load_data(path: str,
              date_cols: list = None,
              encoding: str = 'utf-8',
              column_mapping: dict = None,
              value_mapping: dict = None) -> tuple:
    """
    WHEN TO USE:
        First step of any pipeline. Loads a portfolio file (CSV or Excel) into a
        standardised DataFrame and returns a quick quality summary.

    INPUTS:
        path           : str  — Absolute or relative path to a CSV or Excel file.
        date_cols      : list — Column names to parse as dates. Default:
                                ['date_naissance', 'date_entree', 'date_sortie']
        encoding       : str  — File encoding for CSV (default 'utf-8').
        column_mapping : dict — {raw_col_name: canonical_name} for non-standard
                                column names not covered by the auto-normalisation.
                                Example: {"Gender": "sexe", "DateNaiss": "date_naissance"}
        value_mapping  : dict — {canonical_col: {raw_value: canonical_value}} to
                                remap categorical values after column renaming.
                                Example: {"sexe": {"M": "H", "F": "F"},
                                          "cause_sortie": {"death": "deces", "alive": "autre"}}

    OUTPUTS:
        (DataFrame, dict) where dict keys are:
            n_rows          : int   — Number of rows loaded.
            n_cols          : int   — Number of columns.
            missing_by_col  : dict  — {col: n_missing} for every column.
            dtypes_detected : dict  — {col: dtype_str} after parsing.
    """
    if date_cols is None:
        date_cols = ['date_naissance', 'date_entree', 'date_sortie']

    ext = os.path.splitext(path)[-1].lower()
    if ext in ('.xlsx', '.xls', '.ods'):
        df = pd.read_excel(path)
        fmt = 'Excel'
    else:
        # Détection automatique du séparateur et de l'encodage pour CSV/TXT
        df, fmt = _load_flat_file(path, encoding)

    # Apply user-provided column mapping FIRST (raw names still intact before normalization)
    if column_mapping:
        extra_renames = {k: v for k, v in column_mapping.items() if k in df.columns}
        if extra_renames:
            df = df.rename(columns=extra_renames)
            print(f"[load_data] Mapping colonnes utilisateur : {extra_renames}")

    # Then normalise column names (uppercase → lowercase, synonyms → canonical names)
    df, _col_renames = normalize_column_names(df)

    # Apply user-provided value mapping (e.g. M→H, death→deces)
    if value_mapping:
        for col, val_map in value_mapping.items():
            if col in df.columns:
                df[col] = df[col].map(lambda x, vm=val_map: vm.get(str(x), x))
                print(f"[load_data] Remappage valeurs '{col}' : {val_map}")

    # Parse date columns that exist in the file (after normalisation)
    existing_date_cols = [c for c in date_cols if c in df.columns]
    for col in existing_date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    missing_by_col = {col: int(df[col].isna().sum()) for col in df.columns}
    dtypes_detected = {col: str(df[col].dtype) for col in df.columns}

    summary = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'missing_by_col': missing_by_col,
        'dtypes_detected': dtypes_detected,
    }

    print(f"[load_data] Loaded {fmt}: {len(df):,} rows x {len(df.columns)} cols from '{path}'")
    total_missing = sum(missing_by_col.values())
    if total_missing > 0:
        print(f"[load_data] Total missing values: {total_missing:,}")
    _LOGGER.log("load_data",
                f"Fichier chargé ({fmt}) : {len(df):,} lignes × {len(df.columns)} colonnes",
                {"n_rows": len(df), "n_cols": len(df.columns),
                 "total_missing": total_missing, "format": fmt, "path": path})
    return df, summary


# ---------------------------------------------------------------------------
# generate_synthetic_data
# ---------------------------------------------------------------------------

def generate_synthetic_data(n: int = 50_000,
                            sexe: str = 'H',
                            date_fin: str = '2023-12-31',
                            date_debut: str = '2010-01-01',
                            lambda_lapse: float = 0.08,
                            seed: int = 42) -> pd.DataFrame:
    """
    WHEN TO USE:
        When no real portfolio is available. Generates a realistic synthetic
        portfolio calibrated on TH/TF 00-02 mortality — useful for testing and
        demonstration.

    INPUTS:
        n             : int   — Number of contracts to generate (default 50,000).
        sexe          : str   — 'H' (male) or 'F' (female). Controls reference table.
        date_fin      : str   — End-of-observation date (ISO 'YYYY-MM-DD').
        date_debut    : str   — Earliest possible entry date (ISO 'YYYY-MM-DD').
        lambda_lapse  : float — Annual lapse rate (default 0.08 = 8 %).
        seed          : int   — Random seed for reproducibility.

    OUTPUTS:
        DataFrame with columns:
            id               : str  — Unique contract identifier 'C000001' …
            date_naissance   : date — Date of birth.
            date_entree      : date — Date of entry into observation.
            date_sortie      : date — Date of exit (death, lapse, or end of obs.).
            cause_sortie     : str  — 'deces' or 'autre'.
            sexe             : str  — Same as parameter sexe.
    """
    np.random.seed(seed)
    DATE_FIN = pd.Timestamp(date_fin)
    DATE_DEB = pd.Timestamp(date_debut)
    n_days_range = (DATE_FIN - DATE_DEB).days

    # Entry ages: bell-shaped around 52 years, range 35-75
    age_range = np.arange(35, 76)
    weights = np.exp(-0.5 * ((age_range - 52) / 12) ** 2)
    weights /= weights.sum()
    ages_entree = np.random.choice(age_range, n, p=weights)

    # Entry dates spread over observation window
    entry_offsets = np.random.randint(0, max(1, int(n_days_range * 0.8)), n)
    dates_entree = DATE_DEB + pd.to_timedelta(entry_offsets, unit='D')

    # Birth dates derived from entry age
    birth_offsets = ages_entree * 365 + np.random.randint(0, 365, n)
    dates_naissance = dates_entree - pd.to_timedelta(birth_offsets, unit='D')

    # Remaining observation time
    remaining = (DATE_FIN - dates_entree).days.astype(float)

    # Mortality intensity calibrated on TH/TF 00-02
    mu = -np.log(1.0 - qx_ref_array(ages_entree, sexe))
    mu = np.clip(mu, 1e-9, None)

    # Simulate time to death (exponential) and time to lapse
    t_death = np.random.exponential(365.25 / mu)
    t_lapse = np.random.exponential(365.25 / lambda_lapse, n)

    is_death = t_death < t_lapse
    t_exit = np.where(is_death, t_death, t_lapse)

    censored = t_exit > remaining
    t_exit_final = np.clip(np.minimum(t_exit, remaining), 0, None).astype(int)
    cause_sortie = np.where(censored, 'autre', np.where(is_death, 'deces', 'autre'))

    dates_sortie_raw = dates_entree + pd.to_timedelta(t_exit_final, unit='D')
    dates_sortie = pd.DatetimeIndex([min(d, DATE_FIN) for d in dates_sortie_raw])

    df = pd.DataFrame({
        'id': [f'C{i:06d}' for i in range(n)],
        'date_naissance': dates_naissance.normalize(),
        'date_entree': dates_entree.normalize(),
        'date_sortie': dates_sortie.normalize(),
        'cause_sortie': cause_sortie,
        'sexe': np.full(n, sexe),
    })

    n_deces = (df['cause_sortie'] == 'deces').sum()
    print(f"[generate_synthetic_data] {n:,} contracts generated (sexe={sexe}, "
          f"calibrated on TH/TF 00-02)")
    print(f"[generate_synthetic_data] Deaths: {n_deces:,} | "
          f"Other exits: {n - n_deces:,}")
    return df


# ---------------------------------------------------------------------------
# clean_data
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame,
               dob_col: str = 'date_naissance',
               entry_col: str = 'date_entree',
               exit_col: str = 'date_sortie',
               death_col: str = 'cause_sortie',
               sexe_col: str = 'sexe',
               sexe_filter: str = None,
               age_min: int = 20,
               age_max: int = 100,
               date_fin_observation: str = '2023-12-31') -> tuple:
    """
    WHEN TO USE:
        Always run after load_data or generate_synthetic_data, before any
        exposure computation. Removes invalid rows and reports reasons.

    INPUTS:
        df           : DataFrame — Raw portfolio data.
        dob_col      : str       — Name of date-of-birth column.
        entry_col    : str       — Name of entry-date column.
        exit_col     : str       — Name of exit-date column.
        death_col    : str       — Name of cause-of-exit column.
        sexe_col     : str       — Name of gender column.
        sexe_filter  : str|None  — If 'H' or 'F', keep only that gender.
        age_min      : int       — Minimum entry age to keep (default 20).
        age_max      : int       — Maximum entry age to keep (default 100).

    OUTPUTS:
        (cleaned_df, dict) where dict keys are:
            n_initial        : int  — Rows before cleaning.
            n_removed        : int  — Total rows removed.
            n_final          : int  — Rows in output.
            removal_reasons  : dict — {reason_str: count} for each filter applied.

    Side-effect:
        data_prep.df_removed — DataFrame of all removed rows with a
        'removal_reason' column indicating why each row was excluded.
    """
    global df_removed
    df_input = df                  # référence à l'original (non modifié)
    df = df.copy()
    n_initial = len(df)
    removal_reasons = {}
    _reason = pd.Series([''] * n_initial, index=df.index, dtype=str)

    # 1. Required columns
    required = [dob_col, entry_col, exit_col, death_col, sexe_col]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[clean_data] Missing required columns: {missing_cols}")

    # 2. Parse dates
    for col in [dob_col, entry_col, exit_col]:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # 2b. Fill NaT exit dates : la valeur sentinelle 31/12/2999 ne peut pas être
    #     parsée par pandas (max ~2262) → NaT signifie "encore actif" → on remplace
    #     par date_fin_observation pour conserver ces individus dans l'analyse.
    t_fin = pd.to_datetime(date_fin_observation)
    mask_no_exit = df[exit_col].isna()
    if mask_no_exit.any():
        df.loc[mask_no_exit, exit_col] = t_fin
        print(f"[clean_data] {mask_no_exit.sum():,} date_sortie NaT (sentinelle) → {date_fin_observation}")

    # 3. Drop rows with null values in required columns
    mask_null = df[required].isna().any(axis=1)
    n_null = int(mask_null.sum())
    if n_null > 0:
        removal_reasons['null_required_fields'] = n_null
        _reason[mask_null & (_reason == '')] = 'null_required_fields'
        df = df[~mask_null].copy()

    # 4. Date consistency: entry < exit and dob < entry
    mask_dates = (
        (df[entry_col] >= df[exit_col]) |
        (df[dob_col] >= df[entry_col]) |
        (df[entry_col] < pd.Timestamp('1900-01-01'))
    )
    n_dates = int(mask_dates.sum())
    if n_dates > 0:
        removal_reasons['date_inconsistency'] = n_dates
        _reason[df.index[mask_dates]] = 'date_inconsistency'
        df = df[~mask_dates].copy()

    # 5. Valid cause_sortie values
    mask_cause = ~df[death_col].isin(['deces', 'autre'])
    n_cause = int(mask_cause.sum())
    if n_cause > 0:
        removal_reasons['invalid_cause_sortie'] = n_cause
        _reason[df.index[mask_cause]] = 'invalid_cause_sortie'
        df = df[~mask_cause].copy()

    # 6. Valid sexe values
    mask_sexe_invalid = ~df[sexe_col].isin(['H', 'F'])
    n_sexe_invalid = int(mask_sexe_invalid.sum())
    if n_sexe_invalid > 0:
        removal_reasons['invalid_sexe_value'] = n_sexe_invalid
        _reason[df.index[mask_sexe_invalid]] = 'invalid_sexe_value'
        df = df[~mask_sexe_invalid].copy()

    # 7. Gender filter
    if sexe_filter in ('H', 'F'):
        mask_sexe_filter = df[sexe_col] != sexe_filter
        n_sexe_filter = int(mask_sexe_filter.sum())
        if n_sexe_filter > 0:
            removal_reasons[f'sexe_not_{sexe_filter}'] = n_sexe_filter
            _reason[df.index[mask_sexe_filter]] = f'sexe_not_{sexe_filter}'
        df = df[df[sexe_col] == sexe_filter].copy()

    # 8. Age range filter
    age_entree_tmp = (df[entry_col] - df[dob_col]).dt.days / 365.25
    mask_age = (age_entree_tmp < age_min) | (age_entree_tmp > age_max)
    n_age = int(mask_age.sum())
    if n_age > 0:
        removal_reasons[f'age_outside_{age_min}_{age_max}'] = n_age
        _reason[df.index[mask_age]] = f'age_outside_{age_min}_{age_max}'
        df = df[~mask_age].copy()

    n_final = len(df)
    n_removed = n_initial - n_final

    # Stocker les lignes supprimées avec leur raison (accessible via data_prep.df_removed)
    removed_idx = _reason[_reason != ''].index
    df_removed = df_input.loc[removed_idx].copy()
    df_removed['removal_reason'] = _reason[removed_idx]

    summary = {
        'n_initial': n_initial,
        'n_removed': n_removed,
        'n_final': n_final,
        'removal_reasons': removal_reasons,
    }

    print(f"[clean_data] Initial: {n_initial:,} | Removed: {n_removed:,} "
          f"| Final: {n_final:,}")
    if removal_reasons:
        for reason, count in removal_reasons.items():
            print(f"  - {reason}: {count:,}")
    _LOGGER.log("clean_data",
                f"Nettoyage : {n_initial:,} → {n_final:,} contrats "
                f"({n_removed:,} supprimés, {100*n_removed/max(n_initial,1):.1f}%)",
                {"n_initial": n_initial, "n_removed": n_removed, "n_final": n_final,
                 "removal_reasons": removal_reasons})
    return df.reset_index(drop=True), summary


# ---------------------------------------------------------------------------
# compute_ages
# ---------------------------------------------------------------------------

def compute_ages(df: pd.DataFrame,
                 dob_col: str = 'date_naissance',
                 entry_col: str = 'date_entree',
                 exit_col: str = 'date_sortie') -> pd.DataFrame:
    """
    WHEN TO USE:
        After clean_data and before compute_exposure_by_age. Adds age columns
        needed by downstream functions.

    INPUTS:
        df        : DataFrame — Cleaned portfolio data.
        dob_col   : str       — Name of date-of-birth column.
        entry_col : str       — Name of entry-date column.
        exit_col  : str       — Name of exit-date column.

    OUTPUTS:
        DataFrame copy with three added columns:
            age_entree    : float — Age at entry (fractional years).
            age_sortie    : float — Age at exit (fractional years).
            duree_obs_ans : float — Observation duration in years.
    """
    df = df.copy()
    df[dob_col] = pd.to_datetime(df[dob_col])
    df[entry_col] = pd.to_datetime(df[entry_col])
    df[exit_col] = pd.to_datetime(df[exit_col])

    df['age_entree'] = (df[entry_col] - df[dob_col]).dt.days / 365.25
    df['age_sortie'] = (df[exit_col] - df[dob_col]).dt.days / 365.25
    df['duree_obs_ans'] = (df[exit_col] - df[entry_col]).dt.days / 365.25

    print(f"[compute_ages] Age at entry: "
          f"min={df['age_entree'].min():.1f}, "
          f"mean={df['age_entree'].mean():.1f}, "
          f"max={df['age_entree'].max():.1f}")
    print(f"[compute_ages] Observation duration: "
          f"mean={df['duree_obs_ans'].mean():.2f} years")
    _LOGGER.log("compute_ages",
                f"Âges calculés : entrée {df['age_entree'].min():.1f}–{df['age_entree'].max():.1f} ans "
                f"(moy {df['age_entree'].mean():.1f}), durée moy {df['duree_obs_ans'].mean():.2f} ans",
                {"age_entree_min": round(float(df['age_entree'].min()), 1),
                 "age_entree_mean": round(float(df['age_entree'].mean()), 1),
                 "age_entree_max": round(float(df['age_entree'].max()), 1),
                 "duree_obs_mean_ans": round(float(df['duree_obs_ans'].mean()), 2)})
    return df


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------

def detect_anomalies(df: pd.DataFrame,
                     dob_col: str = 'date_naissance',
                     entry_col: str = 'date_entree',
                     exit_col: str = 'date_sortie',
                     death_col: str = 'cause_sortie') -> dict:
    """
    WHEN TO USE:
        Exploratory step, run before or alongside clean_data to understand
        data quality issues. Returns a structured report with recommendations.

    INPUTS:
        df        : DataFrame — Raw or cleaned portfolio data.
        dob_col   : str       — Name of date-of-birth column.
        entry_col : str       — Name of entry-date column.
        exit_col  : str       — Name of exit-date column.
        death_col : str       — Name of cause-of-exit column.

    OUTPUTS:
        dict with keys:
            duplicates             : int  — Number of duplicate rows.
            missing_values         : dict — {col: n_missing} for all columns.
            date_inconsistencies   : int  — Rows with date ordering problems.
            extreme_ages           : int  — Rows with entry age <18 or >110.
            severity               : str  — 'low', 'medium', or 'high'.
            recommendations        : list[str] — Actionable messages.
    """
    result = {}
    recommendations = []

    # Duplicates
    n_dup = int(df.duplicated().sum())
    result['duplicates'] = n_dup
    if n_dup > 0:
        recommendations.append(f"Drop {n_dup} duplicate rows with df.drop_duplicates().")

    # Missing values
    missing = {col: int(df[col].isna().sum()) for col in df.columns}
    result['missing_values'] = missing
    for col, count in missing.items():
        if count > 0:
            pct = 100 * count / max(len(df), 1)
            recommendations.append(f"Column '{col}' has {count:,} missing values ({pct:.1f}%).")

    # Date inconsistencies
    date_issues = 0
    for col in [dob_col, entry_col, exit_col]:
        if col in df.columns:
            df_tmp = df.copy()
            df_tmp[col] = pd.to_datetime(df_tmp[col], errors='coerce')
    present = [c for c in [dob_col, entry_col, exit_col] if c in df.columns]
    if len(present) == 3:
        dob = pd.to_datetime(df[dob_col], errors='coerce')
        entry = pd.to_datetime(df[entry_col], errors='coerce')
        exit_ = pd.to_datetime(df[exit_col], errors='coerce')
        date_issues = int(((entry >= exit_) | (dob >= entry)).sum())
    result['date_inconsistencies'] = date_issues
    if date_issues > 0:
        recommendations.append(
            f"{date_issues} rows have date ordering problems (dob>=entry or entry>=exit).")

    # Extreme ages
    extreme_ages = 0
    if dob_col in df.columns and entry_col in df.columns:
        dob = pd.to_datetime(df[dob_col], errors='coerce')
        entry = pd.to_datetime(df[entry_col], errors='coerce')
        age_e = (entry - dob).dt.days / 365.25
        extreme_ages = int(((age_e < 18) | (age_e > 110)).sum())
    result['extreme_ages'] = extreme_ages
    if extreme_ages > 0:
        recommendations.append(
            f"{extreme_ages} rows have extreme entry ages (<18 or >110 years).")

    # Severity scoring
    n = max(len(df), 1)
    issues_pct = 100 * (n_dup + date_issues + extreme_ages) / n
    if issues_pct < 1 and sum(missing.values()) / (n * max(len(df.columns), 1)) < 0.01:
        severity = 'low'
    elif issues_pct < 5:
        severity = 'medium'
    else:
        severity = 'high'
    result['severity'] = severity
    result['recommendations'] = recommendations

    if not recommendations:
        recommendations.append("No major anomalies detected. Data looks clean.")

    print(f"[detect_anomalies] Severity: {severity.upper()}")
    for r in recommendations:
        print(f"  -> {r}")
    _LOGGER.log("detect_anomalies",
                f"Qualité des données : sévérité {severity.upper()} — "
                f"{n_dup} doublons, {date_issues} incohérences de dates, "
                f"{extreme_ages} âges extrêmes",
                {"severity": severity, "duplicates": n_dup,
                 "date_inconsistencies": date_issues, "extreme_ages": extreme_ages,
                 "recommendations": recommendations})
    return result
