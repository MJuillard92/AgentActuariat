# Spécifications fonctionnelles — Agent Actuariat

## Architecture générale

```
canvas_app.py
├── Onglet "Rapport guidé"
│     ├── Upload CSV → dcc.Store (DataFrame sérialisé)
│     └── Chat → WriterAgent → tool calls → affichage résultats
└── Onglet "DEV"
      ├── Capacités : cartes par tool (depuis builder_capabilities.json)
      └── Code : arborescence + éditeur + sauvegarde

report_agent/
├── writer_agent.py          — boucle OpenAI tool-calling
├── writer_dialog_prompt.md  — instructions de l'agent
├── builder_capabilities.json — registre des tools (source de vérité)
├── dictionary/
│     └── column_schema.py   — mapping colonnes CSV → rôles actuariels
└── tools/
      ├── tool_registry.py   — dispatch tool_name.function_name → .py
      ├── statistical_analysis/  — analyses descriptives (4 fonctions)
      ├── build_pdf/             — génération PDF (2 fonctions)
      ├── builder/               — construction table mortalité (6 fonctions)
      ├── graphs/                — graphiques (2 fonctions)
      └── reasoning/             — compréhension métier (1 fonction)

notebooks/                   — fonctions actuarielles source (wrappées par builder/)
├── 01_data_preparation.py   — chargement, nettoyage, données synthétiques
├── 02_exposure.py           — exposition centrale par âge
├── 03_crude_rates.py        — taux bruts (centrale, binomiale, Kaplan-Meier)
├── 04_smoothing.py          — lissage (Whittaker, Gompertz, Makeham, spline)
├── 05_diagnostics.py        — crédibilité, comparaison lisseurs, SMR
├── 06_validation.py         — IC Poisson, test chi2
├── 07_benchmarking.py       — abattements vs référence (TH/TF 00-02, TD 88-90)
├── 08_visualization.py      — graphiques PNG (exposition, taux, SMR)
└── actuarial_params.py      — paramètres de la pipeline (seuils, méthodes)
```

## Flux de données

### Rapport guidé — analyse descriptive

```
CSV uploadé
    → parse_csv() → DataFrame → dcc.Store("store-df-json")
    → WriterAgent._build_system_prompt(df)
         → build_mapping_report(df, caps) : mapping automatique colonnes
         → injecté dans le system prompt
    → Dialogue utilisateur
    → tool call : statistical_analysis.portfolio_summary(df, params)
    → tool call : statistical_analysis.age_distribution(df, params)
    → tool call : graphs.analysis_plots(data, {chart: "age_pyramid"})
         → retourne image_b64 (PNG)
    → tool call : build_pdf.descriptive_report(data, {output_path: ...})
```

### Rapport guidé — construction table

```
CSV uploadé
    → WriterAgent
    → tool call : builder.exposure(df, {age_min, age_max})
         → data_store["exposure_table"] = [{age, E_x, D_x, mu_x, q_x_brut}]
    → tool call : builder.crude_rates(data, {method: "central"})
         → data_store["qx_table"] = [{age, E_x, D_x, qx}]
    → tool call : builder.diagnostics(data, {function_name: "credibility"})
    → tool call : builder.smoothing(data, {method: "whittaker"})
         → data_store["smoothed_table"] = [{age, q_x_lisse}]
    → tool call : graphs.builder_plots(data, {chart: "crude_smoothed"})
    → tool call : builder.validation(data, {function_name: "chi_square"})
    → tool call : builder.benchmarking(data, {function_name: "abatement_factors"})
```

## Colonnes CSV reconnues (COLUMN_SCHEMA)

| Rôle | Label | Candidats reconnus |
|---|---|---|
| date_entree | Date d'entrée | date_entree, ctreffet, entry_date |
| date_sortie | Date de sortie | date_sortie, exit_date |
| date_naissance | Date de naissance | date_naissance, clinaiss, dob |
| cause_sortie | Cause de sortie | cause_sortie, statut, status, cause |
| sexe | Sexe | sexe, sexeref, gender, sex |
| produit | Produit | cdprod, produit, product |
| duree_obs_ans | Durée observation | duree_obs_ans, exposition, exposure |

## Format des tools

Toute fonction dans `report_agent/tools/` suit l'interface :
- `run(df: pd.DataFrame, params: dict) -> dict` pour statistical_analysis et builder.exposure
- `run(data: dict, params: dict) -> dict` pour builder (autres), graphs, build_pdf
- `run(context: dict, params: dict) -> dict` pour reasoning

Chaque résultat contient soit les données calculées, soit `{"erreur": "message"}`.
