"""
certification_report.py
Rapport de certification réglementaire.

Nécessite le tool build_table (Phase 2). En Phase 1, retourne une erreur explicite.

Interface : run(data, params) -> dict
"""
from __future__ import annotations


def run(data: dict, params: dict | None = None) -> dict:
    """
    Génère un rapport de certification PDF.

    NOTE Phase 1 : nécessite les calculs actuariels du tool build_table
    (lissage, SMR, chi2). Ces calculs ne sont pas encore disponibles.
    Utilisez build_pdf.descriptive_report pour un rapport descriptif.
    """
    return {
        "erreur": (
            "Le rapport de certification nécessite le tool build_table (Phase 2 — "
            "non encore disponible). "
            "Pour l'instant, utilisez build_pdf.descriptive_report pour un rapport descriptif "
            "du portefeuille."
        ),
        "disponible": False,
    }
