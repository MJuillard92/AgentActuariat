"""
TOOL CONTRACT — preprocessing.clean_records
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : preprocessing.clean_records
domain        : preprocessing
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-21

DESCRIPTION
-----------
Premier nœud du DAG Builder. Applique 6 règles figées de retraitement
(R1 contrats sans effet, R2–R5 âges aberrants, R6 sortie < entrée),
produit la base assainie et un rapport d'exclusions détaillé.

WHEN TO USE
-----------
Systématiquement, avant tout tool statistical_analysis.* ou builder.*
consommant des records. Les tools en aval reçoivent cleaned_records,
jamais input_records brut.

WHEN NOT TO USE
---------------
N/A — toujours appelé.

PREREQUISITES
-------------
required_tools: [master.normalize_records]
required_data_store_keys: []
Note: reçoit df (DataFrame) déjà normalisé par Master (column_mapping,
value_mapping appliqués).

INPUTS
------
params: {}

OUTPUTS
-------
data_store_keys_written:
  - cleaned_records : DataFrame — records après exclusions
  - exclusion_report : dict — {initial_count, final_count, rules}
  - rules : list[dict] — records {rule_id, rule_label, count, detail} (sous-champ de exclusion_report.rules)
return_payload:
  cleaned_records  : table — records après retraitement (R1–R6)
  exclusion_report : dict — initial_count, final_count, rules
  total_records    : integer — nombre de lignes post-retraitement (exclusion_report.final_count)

QUALITY GATES
-------------
BLOCKING:
  - final_count == 0 → retourne erreur.
NON-BLOCKING:
  - final_count < 0.5 × initial_count → warning.

CATALOGUE METADATA
------------------
display_name      : Retraitement des données aberrantes
short_description : Applique 6 règles de retraitement et produit un rapport d'exclusions.
domain            : preprocessing
capability_group  : preprocessing
depends_on        : [master.normalize_records]
required_by       : [builder.exposure, statistical_analysis.time_series, statistical_analysis.age_distribution, statistical_analysis.segmentation]
client_visible    : true
"""
from __future__ import annotations
import pandas as pd


def _ages(df: pd.DataFrame, obs_end: "pd.Timestamp | None" = None) -> tuple[pd.Series, pd.Series]:
    # `format="mixed"` + `dayfirst=True` : sans ça les dates au format
    # DD/MM/YYYY (standard français) sont mal-parsées en NaT, et toutes
    # les règles R2-R6 ratent les lignes concernées (vu en prod : R6
    # devrait exclure 52k lignes mais n'en voyait que 3k).
    dn = pd.to_datetime(df["date_naissance"], format="mixed", dayfirst=True, errors="coerce")
    de = pd.to_datetime(df["date_entree"],    format="mixed", dayfirst=True, errors="coerce")
    ds = pd.to_datetime(df["date_sortie"],    format="mixed", dayfirst=True, errors="coerce")
    # Clipping des sortie sentinelles (31/12/2999 = contrats actifs) à
    # l'observation_end. Sans ça, R5 (age sortie > 100) exclut tous les
    # contrats actifs alors qu'ils sont valides (~228k lignes en pratique).
    if obs_end is not None:
        ds = ds.where(ds <= obs_end, obs_end)
    age_entree = (de - dn).dt.days / 365.25
    age_sortie = (ds - dn).dt.days / 365.25
    return age_entree, age_sortie


def _resolve_obs_end(df: pd.DataFrame, params: dict | None) -> "pd.Timestamp | None":
    """Détermine la date de fin d'observation pour clipper les sentinelles.
    Priorité : params.observation_end > dernier décès observé > None."""
    params = params or {}
    obs_end_str = params.get("observation_end")
    if obs_end_str:
        try:
            return pd.to_datetime(str(obs_end_str), dayfirst=True)
        except Exception:
            pass
    # Auto-détection : max(date_sortie) parmi les décès
    if "cause_sortie" in df.columns and "date_sortie" in df.columns:
        is_dead = df["cause_sortie"].astype(str).str.lower().isin(
            {"deces", "décès", "decede", "décédé", "d", "1", "true"}
        )
        ds = pd.to_datetime(df.loc[is_dead, "date_sortie"], errors="coerce")
        ds = ds[ds.notna() & (ds.dt.year < 2100)]
        if len(ds) > 0:
            return ds.max()
    return None


def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    initial_count = len(df)
    rules_report: list[dict] = []
    current = df.copy()
    obs_end = _resolve_obs_end(df, params)

    def _apply(mask: pd.Series, rule_id: str, rule_label: str, detail: dict | None = None) -> None:
        nonlocal current
        m = mask.reindex(current.index, fill_value=False)
        count = int(m.sum())
        current = current[~m].copy()
        rules_report.append({
            "rule_id":    rule_id,
            "rule_label": rule_label,
            "count":      count,
            "detail":     detail or {},
        })

    # R1 — Contrats sans effet
    mask = current["cause_sortie"].astype(str).str.lower() == "sans_objet"
    _apply(mask, "R1", "Contrats sans effet (cause de sortie \u00ab\u00a0sans objet\u00a0\u00bb)")

    # R2–R5 — âges aberrants
    ae, as_ = _ages(current, obs_end)
    _apply(ae < 0,    "R2", "Âge à l'entrée négatif")
    ae, as_ = _ages(current, obs_end)  # recalcul après R2
    _apply(as_ < 0,   "R3", "Âge à la sortie négatif")
    ae, as_ = _ages(current, obs_end)
    _apply(ae > 100,  "R4", "Âge à l'entrée supérieur à 100 ans")
    ae, as_ = _ages(current, obs_end)
    _apply(as_ > 100, "R5", "Âge à la sortie supérieur à 100 ans")

    ae, as_ = _ages(current, obs_end)
    _apply(as_ < ae, "R6", "Âge à la sortie inférieur à l'âge à l'entrée")

    return {
        "cleaned_records": current.reset_index(drop=True),
        "exclusion_report": {
            "initial_count": initial_count,
            "final_count":   len(current),
            "rules":         rules_report,
        },
    }
