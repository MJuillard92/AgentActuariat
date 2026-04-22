"""
TOOL CONTRACT — preprocessing.clean_records
═══════════════════════════════════════════

Premier nœud du DAG Builder. Reçoit les records normalisés par Master,
applique les règles de retraitement figées, produit la base assainie
et le rapport d'exclusions consommé par la section data_preprocessing.
"""
from __future__ import annotations
import pandas as pd


def _ages(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    dn = pd.to_datetime(df["date_naissance"], errors="coerce")
    de = pd.to_datetime(df["date_entree"],    errors="coerce")
    ds = pd.to_datetime(df["date_sortie"],    errors="coerce")
    age_entree = (de - dn).dt.days / 365.25
    age_sortie = (ds - dn).dt.days / 365.25
    return age_entree, age_sortie


def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    initial_count = len(df)
    rules_report: list[dict] = []
    current = df.copy()

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
    ae, as_ = _ages(current)
    _apply(ae < 0,    "R2", "Âge à l'entrée négatif")
    ae, as_ = _ages(current)  # recalcul après R2
    _apply(as_ < 0,   "R3", "Âge à la sortie négatif")
    ae, as_ = _ages(current)
    _apply(ae > 100,  "R4", "Âge à l'entrée supérieur à 100 ans")
    ae, as_ = _ages(current)
    _apply(as_ > 100, "R5", "Âge à la sortie supérieur à 100 ans")

    ae, as_ = _ages(current)
    _apply(as_ < ae, "R6", "Âge à la sortie inférieur à l'âge à l'entrée")

    return {
        "cleaned_records": current.reset_index(drop=True),
        "exclusion_report": {
            "initial_count": initial_count,
            "final_count":   len(current),
            "rules":         rules_report,
        },
    }
