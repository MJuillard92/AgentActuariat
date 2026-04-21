"""
TOOL CONTRACT — preprocessing.clean_records
═══════════════════════════════════════════

Premier nœud du DAG Builder. Reçoit les records normalisés par Master,
applique les règles de retraitement figées, produit la base assainie
et le rapport d'exclusions consommé par la section data_preprocessing.
"""
from __future__ import annotations
import pandas as pd


def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    initial_count = len(df)
    rules_report: list[dict] = []
    current = df.copy()

    # R1 — Contrats sans effet
    mask_r1 = current["cause_sortie"].astype(str).str.lower() == "sans_objet"
    count_r1 = int(mask_r1.sum())
    current = current[~mask_r1].copy()
    rules_report.append({
        "rule_id":    "R1",
        "rule_label": "Contrats sans effet (cause de sortie \u00ab\u00a0sans objet\u00a0\u00bb)",
        "count":      count_r1,
        "detail":     {},
    })

    return {
        "cleaned_records": current.reset_index(drop=True),
        "exclusion_report": {
            "initial_count": initial_count,
            "final_count":   len(current),
            "rules":         rules_report,
        },
    }
