# Carte d'identité — MortalityAgent

## Domaine
Construction de tables de mortalité d'expérience à partir de données de portefeuille.

## Requêtes types qu'il traite
- "Construis une table de mortalité"
- "Calcule les taux de décès sur mon portefeuille"
- "Lisse la table avec Whittaker-Henderson"
- "Benchmark vs TH0002 / TD 88-90"
- "Donne-moi les diagnostics de crédibilité"
- "Calcule l'exposition et les taux bruts"

## Prérequis à confirmer AVANT de router

1. **Fichier de données** uploadé avec les colonnes : date_naissance, date_entrée, date_sortie, cause_sortie
2. **Sexe** à analyser : H, F, ou les deux séparément
3. **Période d'observation** (optionnel — peut être déduite des données)
4. **Table de référence** pour le benchmarking (optionnel — TH0002 par défaut)

## Ce qu'il produit
- `exposure_table` : exposition centrale par âge
- `qx_table` : taux bruts de mortalité
- `smoothed_table` : table lissée
- `benchmarking` : facteurs d'abattement vs table de référence
- Présente son plan de travail au client avant d'exécuter

## Ce qu'il ne fait PAS
- Rédaction de rapport PDF (→ ReportAgent)
- Tarification dommages, IBNR, chain-ladder
- Accès à des données externes
