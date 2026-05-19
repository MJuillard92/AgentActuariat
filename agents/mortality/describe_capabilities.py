"""
agents.mortality.describe_capabilities

Self-describe du MortalityAgent — appelé au démarrage par capability_registry.

Le MortalityAgent construit une table de mortalité d'expérience à partir d'un
portefeuille. Il enchaîne preprocessing → exposition → taux bruts → lissage
→ validation → benchmarking → fermeture grands âges.

Les tools sont introspectés dynamiquement depuis tools/catalogue.yaml pour
rester en phase avec l'état réel du code (toute nouvelle docstring de tool
préfixée builder./statistical_analysis./preprocessing./aggregation. est
automatiquement remontée ici).
"""
from __future__ import annotations


# Namespaces des tools owned par cet agent
_OWNED_NAMESPACES = ("builder", "statistical_analysis", "preprocessing", "aggregation")


def _list_owned_tools() -> list[dict]:
    """Filtre le catalogue pour ne garder que les tools de cet agent."""
    try:
        from tools.catalogue import get_catalogue
    except Exception:
        return []
    cat = get_catalogue() or {}
    tools_flat = cat.get("tools") or {}
    owned: list[dict] = []
    for qualified_name, info in tools_flat.items():
        if info.get("client_visible") is False:
            continue
        if not qualified_name.startswith(tuple(f"{ns}." for ns in _OWNED_NAMESPACES)):
            continue
        owned.append({
            "name":        qualified_name,
            "display":     info.get("display_name") or qualified_name,
            "description": info.get("short_description") or "",
            "methods":     info.get("methods") or [],
        })
    return sorted(owned, key=lambda x: x["name"])


def describe() -> dict:
    """Retourne le descriptif des capacités de calcul du MortalityAgent."""
    return {
        "agent":   "mortality",
        "display": "MortalityAgent — Calculs actuariels",
        "purpose": (
            "Construit une table de mortalité d'expérience à partir d'un "
            "portefeuille (assurances vie, prévoyance, retraite). Enchaîne "
            "preprocessing, exposition, taux bruts, lissage, validation et "
            "benchmarking contre les tables réglementaires françaises."
        ),
        "tools":           _list_owned_tools(),
        "methods_summary": {
            "preprocessing":         ["clean_records (sentinelles, dates, doublons)"],
            "exposition":            ["compute_exposure (E_x, D_x centraux)"],
            "taux_bruts":            ["Kaplan-Meier", "Nelson-Aalen"],
            "lissage_non_paramétrique": ["Whittaker-Henderson 1D", "LOESS"],
            "lissage_paramétrique":  ["Gompertz", "Makeham", "Heligman-Pollard", "Beard"],
            "fermeture_grands_ages": ["Coale-Kisker", "Denuit-Goderniaux"],
            "validation":            ["chi-2 d'ajustement", "intervalles confiance binomiaux", "SMR"],
            "benchmarking":          ["facteurs d'abattement vs TH/TF, TGH/TGF"],
            "régression":            ["logit", "Cox proportional hazards"],
            "analyses_descriptives": ["distribution par âge", "qualité données", "segmentation portefeuille"],
        },
        "inputs":  [
            "fichier CSV de portefeuille",
            "study_plan validé (sexe, période d'observation, table de référence)",
            "choix des méthodes ou mode auto",
        ],
        "outputs": [
            "exposure_table",
            "qx_table (taux bruts)",
            "smoothed_table (taux lissés)",
            "diagnostics (crédibilité par âge)",
            "validation (chi-2, IC)",
            "benchmarking (SMR, facteurs d'abattement)",
        ],
    }
