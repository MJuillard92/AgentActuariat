# Design — Section `data_analysis` (migration V1 → Design 3)

**Date** : 2026-04-21
**Scope** : réintroduction de la section `data_submission` du V1 archive, renommée `data_analysis`, en conformité avec la structure Design 3 (session_inputs / data_contract / sections). Cette section se décline en deux variantes mutuellement exclusives : `data_analysis_unisex` et `data_analysis_by_sex`.
**Contexte** : suite à la livraison de `preamble` (Design 3, 189 lignes) et à l'objectif E2E US-26, on réintroduit les sections V1 une par une. `data_analysis` est la première après le préambule.

---

## 1. Objectif fonctionnel

La section `data_analysis` présente une analyse descriptive et interprétative de la base de données ayant servi à la construction de la table de mortalité :
- ouverture narrative factuelle (volumétrie, période),
- tableau annuel des grandeurs clés (effectif, âge moyen, décès, taux, âge au décès),
- analyse interprétative LLM post-tableau (tendances + causes probables),
- graphique de distribution des effectifs par âge.

Le traitement se dédouble selon le choix méthodologique `gender_segmentation` :
- `unisex` → analyse agrégée (1 table + 1 chart)
- `by_sex` → analyse ventilée H/F (2 tables + 2 charts, analyse LLM comparative)

## 2. Nouveautés `data_contract`

### 2.1 `master_from_data`

| Clé | Type | Tool | Description |
|---|---|---|---|
| `total_records` | integer | `master.analyze_data_and_request` | Nombre de lignes post-normalisation. `confirm_with_user: true` |

### 2.2 `master_from_modeling`

| Clé | Type | Tool | Description |
|---|---|---|---|
| `gender_segmentation` | enum `[unisex, by_sex]` | `master.classify_request` | Choix méthodologique : table unisex agrégée vs tables séparées H/F. `confirm_with_user: true` |

### 2.3 `builder_outputs`

| Clé | Type | Tool | group_by |
|---|---|---|---|
| `annual_statistics_table` | list[dict] | `mortality.compute_annual_statistics` | `null` |
| `annual_statistics_table_male` | list[dict] | `mortality.compute_annual_statistics` | `{sexe: H}` |
| `annual_statistics_table_female` | list[dict] | `mortality.compute_annual_statistics` | `{sexe: F}` |
| `exposure_by_age` | list[dict] | `mortality.compute_exposure_by_age` | `null` |
| `exposure_by_age_male` | list[dict] | `mortality.compute_exposure_by_age` | `{sexe: H}` |
| `exposure_by_age_female` | list[dict] | `mortality.compute_exposure_by_age` | `{sexe: F}` |

Chaque ligne d'`annual_statistics_table*` : `{year, n_lives, mean_age, deaths, death_rate, mean_age_at_death}`.
Chaque ligne d'`exposure_by_age*` : `{age, exposure}`.

### 2.4 Conservation délibérée

`deaths_by_year_series` (preamble) est **conservé** malgré la redondance fonctionnelle avec `annual_statistics_table` (qui contient déjà la colonne `deaths` par année). Raison : ne pas toucher le préambule avant E2E US-26 vert. Dette tracée dans `memory/project_refactor_mortality_describe.md` — la refacto `mortality.describe()` générique collapsera `deaths_by_year_series` et les 6 tools `compute_*`.

## 3. Nouveaux tools

### 3.1 `mortality.compute_annual_statistics`

**Signature** : `(records, period, group_by: dict | null) -> {stats: list[dict]}`
**Sortie** : par année de la période, `{year, n_lives, mean_age, deaths, death_rate, mean_age_at_death}`.
**Filtre** : si `group_by={sexe: H}`, calcule sur le sous-ensemble `records[records.sexe == 'H']`.

### 3.2 `mortality.compute_exposure_by_age`

**Signature** : `(records, group_by: dict | null) -> {distribution: list[dict]}`
**Sortie** : par âge entier, `{age, exposure}`, snapshot agrégé sur toute la période.
**Filtre** : idem.

## 4. Sections YAML

### 4.1 `data_analysis_unisex`

```yaml
- id: data_analysis_unisex
  label: "Analyse des données — base agrégée"
  required: true
  dependencies: [preamble]
  activation:
    key: gender_segmentation
    equals: unisex

  narrative:
    text: |
      La base de données utilisée regroupe {{ total_records }} lignes,
      réparties sur {{ num_observation_years }} années d'observation,
      conduisant à {{ total_exposure_years }} années-personne. Une analyse
      statistique des effectifs sous risque par année est présentée
      ci-après.

  llm_directives:
    tone: "professionnel, actuariel, descriptif"
    length_words: [120, 200]
    rag_query: "présentation base données mortalité effectifs observation"

    post_table_analysis:
      instruction: >
        À partir du tableau annuel, produire 3 à 5 bullets analysant
        l'évolution. Pour chaque bullet : (1) constater la tendance
        avec valeurs saillantes ; (2) proposer une explication probable,
        métier (montée en puissance, changement de cible) ou statistique
        (faiblesse des effectifs, variance). Style actuariel, factuel,
        pas de spéculation non étayée.
      length_words: [80, 150]
      few_shot_example: |
        - forte croissance de l'effectif sous risque, traduisant une
          montée en puissance de la commercialisation ;
        - taux de décès annuel très volatil, s'expliquant par la faiblesse
          de l'effectif sur les premières années ;
        - âge moyen relativement stable ;
        - âge moyen au décès volatil.

  visual_specs:
    - id: annual_statistics
      type: table
      purpose: "Statistiques annuelles agrégées du portefeuille."
      source: annual_statistics_table
      columns:
        - {key: year,              label: "Année",                format: int}
        - {key: n_lives,           label: "Effectif sous risque", format: int}
        - {key: mean_age,          label: "Âge moyen",            format: float2}
        - {key: deaths,            label: "Nombre de décès",      format: int}
        - {key: death_rate,        label: "Taux de décès",        format: percent2}
        - {key: mean_age_at_death, label: "Âge moyen au décès",   format: float2}
      highlight_rule: totals_row

    - id: exposure_distribution_by_age
      type: chart
      chart_type: line
      purpose: "Distribution des effectifs sous risque par âge."
      source: exposure_by_age
      x_axis: {key: age,      label: "Âge"}
      y_axis: {key: exposure, label: "Effectif"}
```

### 4.2 `data_analysis_by_sex`

```yaml
- id: data_analysis_by_sex
  label: "Analyse des données — ventilation par sexe"
  required: true
  dependencies: [preamble]
  activation:
    key: gender_segmentation
    equals: by_sex

  narrative:
    text: |
      La base de données utilisée regroupe {{ total_records }} lignes,
      réparties sur {{ num_observation_years }} années d'observation,
      conduisant à {{ total_exposure_years }} années-personne. L'étude
      étant conduite par sexe, les statistiques annuelles et la
      distribution des expositions par âge sont présentées séparément
      pour les hommes et les femmes.

  llm_directives:
    tone: "professionnel, actuariel, descriptif"
    length_words: [150, 250]
    rag_query: "analyse par sexe mortalité effectifs portefeuille"

    post_table_analysis:
      instruction: >
        À partir des deux tableaux annuels (H et F), produire 4 à 6 bullets
        analysant l'évolution pour chaque sexe ET comparant les tendances
        entre sexes. Pour chaque constat : (1) valeur saillante ;
        (2) explication probable (métier ou statistique). Mettre en
        évidence les écarts H/F significatifs.
      length_words: [120, 200]
      few_shot_example: |
        - effectif masculin en forte croissance, effectif féminin stable,
          traduisant une cible commerciale majoritairement masculine ;
        - taux de décès systématiquement supérieur chez les hommes
          (ratio ~2:1), cohérent avec la littérature actuarielle ;
        - âge moyen au décès plus élevé chez les femmes d'environ X ans ;
        - volatilité des taux comparable entre sexes sur la période.

  visual_specs:
    - id: annual_statistics_male
      type: table
      purpose: "Statistiques annuelles — hommes."
      source: annual_statistics_table_male
      columns:
        - {key: year,              label: "Année",                format: int}
        - {key: n_lives,           label: "Effectif sous risque", format: int}
        - {key: mean_age,          label: "Âge moyen",            format: float2}
        - {key: deaths,            label: "Nombre de décès",      format: int}
        - {key: death_rate,        label: "Taux de décès",        format: percent2}
        - {key: mean_age_at_death, label: "Âge moyen au décès",   format: float2}
      highlight_rule: totals_row

    - id: annual_statistics_female
      type: table
      purpose: "Statistiques annuelles — femmes."
      source: annual_statistics_table_female
      columns:
        - {key: year,              label: "Année",                format: int}
        - {key: n_lives,           label: "Effectif sous risque", format: int}
        - {key: mean_age,          label: "Âge moyen",            format: float2}
        - {key: deaths,            label: "Nombre de décès",      format: int}
        - {key: death_rate,        label: "Taux de décès",        format: percent2}
        - {key: mean_age_at_death, label: "Âge moyen au décès",   format: float2}
      highlight_rule: totals_row

    - id: exposure_distribution_male
      type: chart
      chart_type: line
      purpose: "Distribution des effectifs par âge — hommes."
      source: exposure_by_age_male
      x_axis: {key: age,      label: "Âge"}
      y_axis: {key: exposure, label: "Effectif"}

    - id: exposure_distribution_female
      type: chart
      chart_type: line
      purpose: "Distribution des effectifs par âge — femmes."
      source: exposure_by_age_female
      x_axis: {key: age,      label: "Âge"}
      y_axis: {key: exposure, label: "Effectif"}
```

## 5. Mécanisme d'activation conditionnelle

### 5.1 Syntaxe

Chaque section peut déclarer un bloc `activation` :

```yaml
activation:
  key: <nom_clé_data_contract>
  equals: <valeur_attendue>
```

Si le bloc est absent, la section est toujours active (comportement actuel de `preamble`).

### 5.2 Sémantique

- Évaluation **après** que le Master a résolu la clé référencée (ici `gender_segmentation` via `master.classify_request` + dialogue 3-modes).
- Une section dont la condition est fausse est **exclue** du manifest produit par `build_manifest()` : ni ses `produced_by` ne sont exécutés (pas de gaspillage compute), ni son narrative/visual_specs ne sont rendus.

### 5.3 Impact implémentation

- **Validator `scripts/check_template.py`** :
  - reconnaître le champ `activation` sur une section ;
  - vérifier que `key` référence une clé déclarée dans `master_from_modeling` (ou `master_from_data`) et de type `enum` ;
  - vérifier que **l'union des `equals` sur toutes les sections qui partagent le même `key`** couvre **toutes les valeurs de l'enum** (pas d'impasse : pour chaque valeur possible, au moins une section s'active).
- **Template_loader `build_manifest`** :
  - accepter un contexte de résolution (dict clé → valeur) ;
  - filtrer les sections dont `activation` est non satisfaite avant calcul du DAG builder.
- **Master orchestration** :
  - résoudre `gender_segmentation` **avant** le GO_BUILD (donc en phase preflight, cohérent avec US-17/18/19).

### 5.4 Justification du choix (vs alternatives)

Syntaxe retenue : déclarative `{key, equals}`. Écartées :
- **Expression `{{ ... }}`** : nécessite un évaluateur d'expressions, surface d'attaque plus large, validation statique plus faible.
- **`variant_group`** : introduit un concept de groupe pour juste 2 sections mutex, disproportionné.

La syntaxe `{key, equals}` est extensible sans breaking change (ajout futur de `in`, `not_equals`, `and`) si un besoin réel apparaît.

## 6. Ordre de rendu final

**Section `data_analysis_unisex`** :
1. Paragraphe narratif (seed reformulé par LLM)
2. Table `annual_statistics`
3. Bullets d'analyse post-table (LLM)
4. Chart `exposure_distribution_by_age`

**Section `data_analysis_by_sex`** :
1. Paragraphe narratif
2. Table `annual_statistics_male`
3. Table `annual_statistics_female`
4. Bullets d'analyse comparative H/F (LLM)
5. Chart `exposure_distribution_male`
6. Chart `exposure_distribution_female`

## 6bis. Prérequis style

Les formats `int`, `float2`, `percent2` utilisés dans `columns.format` ne sont pas encore définis dans `knowledge_base/report_template/style.yaml` (qui ne couvre aujourd'hui que couleurs + typographie). Deux options :

- étendre `style.yaml` avec une section `formats: {int: "{:d}", float2: "{:.2f}", percent2: "{:.2%}"}` dans le cadre de l'implémentation de cette spec ;
- déléguer à `table_renderer.py` une logique de formats par défaut, avec `style.yaml` n'ayant que les surcharges.

À arbitrer au moment de la planification (writing-plans). Non bloquant pour le présent design.

## 7. Hors scope

- Refactorisation des 6 tools `mortality.compute_*` en un tool générique `mortality.describe(records, group_by, metrics)` — cf. `memory/project_refactor_mortality_describe.md`. À traiter après E2E `data_analysis` vert.
- Suppression de `deaths_by_year_series` — idem, collapse prévu dans la refacto `describe()`.
- Enrichissement du LLM avec contexte métier supplémentaire (`product_list`, `underwriting_rules` du V1) — à évaluer lorsqu'on constatera la qualité réelle des bullets interprétatifs ; sinon, reste générique.

## 8. Critères de done

- `knowledge_base/report_template/mortality_template.yaml` contient les deux sections + nouveautés data_contract.
- Tools `mortality.compute_annual_statistics` et `mortality.compute_exposure_by_age` implémentés et testés (TDD).
- `scripts/check_template.py` reconnaît `activation` et vérifie la couverture d'enum.
- `template_loader.build_manifest()` accepte un contexte et filtre les sections inactives.
- Tests : `pytest tests/` vert ; `python scripts/check_template.py` vert.
- E2E (après US-26 preamble vert) : génération d'un rapport sur un dataset fictif dans les deux modes (`unisex` et `by_sex`).
