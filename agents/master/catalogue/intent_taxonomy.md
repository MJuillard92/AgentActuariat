# Taxonomie des intentions — MasterAgent

## Types de demandes reconnues

| Intention | Indices linguistiques | Sous-agent | Prérequis |
|---|---|---|---|
| Construction de table | "table de mortalité", "taux de décès", "lissage", "Whittaker", "Gompertz", "exposition" | MortalityAgent | données + sexe |
| Benchmarking | "comparer à TH0002", "facteurs d'abattement", "SMR", "table de référence" | MortalityAgent | données + sexe + référence |
| Rapport PDF | "rapport", "certification", "PDF", "document", "notebook", "log de session" | ReportAgent | calculs préalables |
| Pipeline complet | "tout faire", "pipeline complet", "de A à Z", "analyse complète" | Mortality → Report | données + sexe |
| Exploration descriptive | "décris le portefeuille", "distribution des âges", "résumé des données" | MortalityAgent | données |

## Questions de qualification (poser UNE SEULE FOIS si non disponibles)

1. Avez-vous un fichier de données ? (colonnes attendues : date_naissance, date_entrée, date_sortie, cause_sortie)
2. Sexe à analyser : H, F, ou les deux ?
3. Table de référence pour le benchmarking ? (TH0002 par défaut si non précisé)

## Règles de routing

- Si **données manquantes** → demander l'upload avant tout routing
- Si **sexe non précisé et analyse demandée** → demander H/F/tous
- Si **rapport demandé sans calculs préalables** → informer que les calculs sont nécessaires d'abord
- Si **pipeline complet** → router vers MortalityAgent en premier, puis ReportAgent automatiquement

## Ce que le MasterAgent NE fait PAS

- Il ne propose pas de plan de travail détaillé (→ MortalityAgent)
- Il ne choisit pas les méthodes de lissage (→ MortalityAgent)
- Il ne commente pas les résultats actuariels (→ MortalityAgent)
- Il ne génère pas de rapports (→ ReportAgent)
