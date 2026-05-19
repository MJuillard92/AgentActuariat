"""
agents.master.describe_capabilities

Self-describe du MasterAgent — appelé au démarrage par capability_registry.

Le Master n'exécute aucun calcul actuariel. Son rôle :
- Classifier les demandes utilisateur (intent : calcul / question / rapport)
- Router vers le sous-agent compétent (MortalityAgent / ReportAgent / RAGAgent)
- Qualifier les prérequis (sexe, période, méthodes)
- Orchestrer les enchaînements (calculs → rapport)
"""
from __future__ import annotations


def describe() -> dict:
    """Retourne le descriptif des capacités orchestrationnelles du Master."""
    return {
        "agent":   "master",
        "display": "MasterAgent — Orchestrateur",
        "purpose": (
            "Comprend la demande utilisateur, qualifie les prérequis, "
            "et route vers le sous-agent compétent. N'exécute aucun calcul."
        ),
        "tools": [
            {
                "name":        "master.classify_request",
                "display":     "Classification d'intention",
                "description": "Détermine kind={calcul, question, rapport, unclear} + paramètres write/report_mode.",
            },
            {
                "name":        "master.analyze_data_and_request",
                "display":     "Analyse fichier + demande",
                "description": "Inspecte le CSV et la formulation user pour suggérer un study_plan initial.",
            },
            {
                "name":        "master.suggest_value_mapping",
                "display":     "Suggestion de mapping valeurs",
                "description": "Propose des correspondances pour les valeurs catégorielles non standard (sexe, statut, etc.).",
            },
            {
                "name":        "master.normalize_records",
                "display":     "Normalisation déterministe",
                "description": "Applique le value_mapping + nettoyage léger (dates, sentinelles).",
            },
        ],
        "routes": [
            "MortalityAgent : intent=calcul (table de mortalité)",
            "ReportAgent   : intent=rapport (PDF certification ou descriptif)",
            "RAGAgent      : intent=question (doctrine actuarielle)",
        ],
        "inputs":  ["fichier CSV de portefeuille", "demande utilisateur en langage naturel"],
        "outputs": ["routing vers sous-agent", "study_plan validé", "data_store hydraté"],
    }
