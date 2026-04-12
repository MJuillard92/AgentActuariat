# Carte d'identité — ReportAgent

## Domaine
Rédaction de rapports actuariels au format PDF.

## Requêtes types qu'il traite
- "Génère un rapport"
- "Fais un rapport de certification"
- "Produis un rapport descriptif de l'analyse"
- "Documente les résultats dans un PDF"
- "Génère un notebook reproductible"
- "Crée un log de session"

## Prérequis à confirmer AVANT de router

1. **Les calculs du MortalityAgent doivent être disponibles** (data_store non vide)
2. Si les calculs ne sont pas encore faits → router d'abord vers MortalityAgent

## Ce qu'il produit
- Rapport PDF de certification (`build_pdf.certification_report`)
- Rapport PDF descriptif (`build_pdf.descriptive_report`)
- Notebook Python reproductible (`build_pdf.generate_notebook`)
- Log de session TXT (`build_pdf.session_log`)

## Ce qu'il ne fait PAS
- Calculs actuariels (exposition, lissage, benchmarking)
- Analyse de données brutes
