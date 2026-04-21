# Design — Section `data_analysis` (migration V1 → Design 3)

**Date** : 2026-04-21 (v2 — rebasé sur tools génériques existants)
**Scope** : réintroduction de la section `data_submission` du V1 archive, renommée `data_analysis`, sous deux variantes mutuellement exclusives : `data_analysis_unisex` et `data_analysis_by_sex`. Activation pilotée par une clé méthodologique `gender_segmentation`.
**Contexte** : après le revert US-20 qui a retiré les tools `mortality.compute_*`, le préambule est rewiré sur les tools génériques (`builder.exposure`, `statistical_analysis.segmentation`, `statistical_analysis.time_series`). Cette spec réutilise ces mêmes tools, avec deux extensions ciblées.

**Versions antérieures** : une v1 (commit `ebc143b`) a été rédigée sur la base des tools `mortality.compute_*` — désormais obsolète, reposant sur des tools supprimés.

---

## 1. Objectif fonctionnel

La section `data_analysis` présente une analyse descriptive et interprétative de la base de données :
- ouverture narrative factuelle (volumétrie, période),
- tableau annuel des grandeurs clés (exposition, âge moyen, décès, taux, âge moyen au décès),
- analyse interprétative LLM post-tableau (tendances + causes probables),
- graphique de distribution des âges par tranches.

Selon `gender_segmentation` :
- `unisex` → analyse agrégée (1 table + 1 chart)
- `by_sex` → analyse ventilée H/F (2 tables + 2 charts, analyse LLM comparative)

## 2. Réutilisation des tools existants

| Tool | Rôle dans data_analysis | Statut |
|---|---|---|
| `statistical_analysis.time_series` | Table annuelle (serie + serie_h/serie_f) | **À étendre** (§3.1) |
| `statistical_analysis.age_distribution` | Chart distribution par tranches d'âge | **À étendre** (§3.2) |
| `statistical_analysis.segmentation` | Déjà consommé par preamble pour `segmentations.sexe` | Inchangé |
| `master.analyze_data_and_request` | Exposer `total_records` | **À étendre** (§3.3) |
| `master.classify_request` | Résoudre `gender_segmentation` | **À étendre** (§3.4) |

## 3. Extensions de tools (minimales)

### 3.1 `statistical_analysis.time_series` — nouveaux champs + mode by_sex

Ajouter aux outputs actuels (`annee, nb_entres, nb_deces, exposition_pa`) :
- `age_moyen_entres` (float) : âge moyen des contrats entrants cette année
- `age_moyen_deces` (float) : âge moyen au décès (null si aucun décès dans l'année)
- `taux_deces` (float) : `nb_deces / exposition_pa × 1000`, exprimé pour 1000 PA

Ajouter un paramètre :
- `by_sex: bool` (default `false`) — si `true`, le tool produit en plus `serie_h` et `serie_f` (même schéma que `serie`), en filtrant sur la colonne sexe.

Sortie enrichie :
```
serie     : list[dict]  # columns: annee, nb_entres, nb_deces, exposition_pa,
                        #          age_moyen_entres, age_moyen_deces, taux_deces
serie_h   : list[dict]  # (si by_sex=true)
serie_f   : list[dict]  # (si by_sex=true)
annee_min : int
annee_max : int
nb_annees : int
anomalies : list[str]   # inchangé
```

### 3.2 `statistical_analysis.age_distribution` — format list pour consommation chart

Ajouter aux outputs :
- `distribution_list: list[{tranche, nb_contrats}]` — équivalent liste de `distribution` (dict), pour consommation directe par `visual_spec.chart` (qui référence x_axis.key / y_axis.key).
- Si `by_sex=True` : `distribution_list_h` et `distribution_list_f`.

Pas de changement fonctionnel — simple reformat de l'output existant.

### 3.3 `master.analyze_data_and_request` — exposer `n_records`

Le tool doit retourner (en plus de ses outputs actuels utilisés pour `observation_period_years`, `start_year`, `end_year`, `num_observation_years`) un champ `n_records: int` (nombre de lignes post-normalisation). Trivial : `len(records)` sur le DataFrame normalisé.

### 3.4 `master.classify_request` — exposer `gender_mode`

Ajouter un output `gender_mode: "unisex" | "by_sex"` pour alimenter `gender_segmentation`. Inféré depuis la requête naturelle (ex : "construis-moi une table H/F" → `by_sex` ; "table unisex" ou défaut → `unisex`). `confirm_with_user: true` côté YAML.

## 4. Nouveautés `data_contract`

### 4.1 `master_from_data`

| Clé | Type | `produced_by.tool` | `output_mapping` |
|---|---|---|---|
| `total_records` | integer | `master.analyze_data_and_request` | `{n_records: total_records}` |

### 4.2 `master_from_modeling`

| Clé | Type | `produced_by.tool` | `output_mapping` |
|---|---|---|---|
| `gender_segmentation` | enum `[unisex, by_sex]` | `master.classify_request` | `{gender_mode: gender_segmentation}` |

`confirm_with_user: true`.

### 4.3 `builder_outputs` (nouveautés)

| Clé | Type | `produced_by.tool` | `produced_by.inputs` | `output_mapping` |
|---|---|---|---|---|
| `serie_h` | list[dict] | `statistical_analysis.time_series` | `{by_sex: true}` | `{serie_h: serie_h}` |
| `serie_f` | list[dict] | `statistical_analysis.time_series` | `{by_sex: true}` | `{serie_f: serie_f}` |
| `ages` | dict | `statistical_analysis.age_distribution` | `{by_sex: true}` | `{*: ages}` (racine) |

**Note sur `ages`** : dict complet (avec `distribution_list`, `distribution_list_h`, `distribution_list_f`, `age_min`, `age_moyen`, etc.) stocké en data_contract ; les visual_specs accèdent aux sous-chemins (`ages.distribution_list`, `ages.distribution_list_h`, …) selon la section active.

**Clés existantes** (déjà dans le YAML, préservées) : `total_exposure`, `total_deaths`, `segmentations`, `serie`. La clé `serie` existante est également enrichie par l'extension §3.1 (colonnes `age_moyen_entres`, `age_moyen_deces`, `taux_deces`).

### 4.4 Invocations uniques côté DAG

Les tools `time_series` et `age_distribution` sont déclarés **une seule fois** dans `data_contract`, avec `by_sex: true`. L'appel unique produit **à la fois** les agrégats globaux (`serie`, `distribution_list`) et les ventilations H/F (`serie_h`, `serie_f`, `distribution_list_h/_f`).

Les sections `data_analysis_unisex` et `data_analysis_by_sex` pointent ensuite vers le sous-chemin approprié via `visual_specs.source`. La section inactive (filtrée par le mécanisme d'activation §6) n'est pas rendue, mais les tools sous-jacents ont été exécutés une fois pour toutes — coût négligeable vs la complexité d'un DAG conditionnel.

## 5. Sections YAML

### 5.1 `data_analysis_unisex`

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
      conduisant à {{ total_exposure }} années-personne. Une analyse
      statistique annuelle est présentée ci-après.

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
        (faiblesse des effectifs, variance). Style actuariel, factuel.
      length_words: [80, 150]
      few_shot_example: |
        - forte croissance de l'exposition, traduisant une montée en
          puissance de la commercialisation ;
        - taux de décès annuel très volatil, s'expliquant par la faiblesse
          de l'effectif sur les premières années ;
        - âge moyen à l'entrée relativement stable ;
        - âge moyen au décès volatil.

  visual_specs:
    - id: annual_statistics
      type: table
      purpose: "Statistiques annuelles agrégées."
      source: serie
      columns:
        - {key: annee,            label: "Année",                      format: int}
        - {key: exposition_pa,    label: "Exposition (années-personne)", format: float2}
        - {key: age_moyen_entres, label: "Âge moyen",                  format: float2}
        - {key: nb_deces,         label: "Nombre de décès",            format: int}
        - {key: taux_deces,       label: "Taux de décès (‰ PA)",       format: float2}
        - {key: age_moyen_deces,  label: "Âge moyen au décès",         format: float2}
      highlight_rule: totals_row

    - id: exposure_distribution_by_age
      type: chart
      chart_type: bar
      purpose: "Distribution des âges à l'entrée par tranches."
      source: ages.distribution_list
      x_axis: {key: tranche,     label: "Âge (tranches)"}
      y_axis: {key: nb_contrats, label: "Effectif"}
```

### 5.2 `data_analysis_by_sex`

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
      conduisant à {{ total_exposure }} années-personne. L'étude étant
      conduite par sexe, les statistiques annuelles et la distribution
      des âges sont présentées séparément pour les hommes et les femmes.

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
        - exposition masculine en forte croissance, exposition féminine
          stable, traduisant une cible commerciale majoritairement masculine ;
        - taux de décès systématiquement supérieur chez les hommes
          (ratio ~2:1), cohérent avec la littérature actuarielle ;
        - âge moyen au décès plus élevé chez les femmes ;
        - volatilité des taux comparable entre sexes sur la période.

  visual_specs:
    - id: annual_statistics_male
      type: table
      purpose: "Statistiques annuelles — hommes."
      source: serie_h
      columns:
        - {key: annee,            label: "Année",                      format: int}
        - {key: exposition_pa,    label: "Exposition (années-personne)", format: float2}
        - {key: age_moyen_entres, label: "Âge moyen",                  format: float2}
        - {key: nb_deces,         label: "Nombre de décès",            format: int}
        - {key: taux_deces,       label: "Taux de décès (‰ PA)",       format: float2}
        - {key: age_moyen_deces,  label: "Âge moyen au décès",         format: float2}
      highlight_rule: totals_row

    - id: annual_statistics_female
      type: table
      purpose: "Statistiques annuelles — femmes."
      source: serie_f
      columns:
        - {key: annee,            label: "Année",                      format: int}
        - {key: exposition_pa,    label: "Exposition (années-personne)", format: float2}
        - {key: age_moyen_entres, label: "Âge moyen",                  format: float2}
        - {key: nb_deces,         label: "Nombre de décès",            format: int}
        - {key: taux_deces,       label: "Taux de décès (‰ PA)",       format: float2}
        - {key: age_moyen_deces,  label: "Âge moyen au décès",         format: float2}
      highlight_rule: totals_row

    - id: exposure_distribution_male
      type: chart
      chart_type: bar
      purpose: "Distribution des âges — hommes."
      source: ages.distribution_list_h
      x_axis: {key: tranche,     label: "Âge (tranches)"}
      y_axis: {key: nb_contrats, label: "Effectif"}

    - id: exposure_distribution_female
      type: chart
      chart_type: bar
      purpose: "Distribution des âges — femmes."
      source: ages.distribution_list_f
      x_axis: {key: tranche,     label: "Âge (tranches)"}
      y_axis: {key: nb_contrats, label: "Effectif"}
```

## 6. Mécanisme d'activation conditionnelle

### 6.1 Syntaxe

Chaque section peut déclarer un bloc `activation` :

```yaml
activation:
  key: <nom_clé_data_contract>
  equals: <valeur_attendue>
```

Absence de bloc = section toujours active (comportement `preamble` actuel).

### 6.2 Sémantique

- Évaluation après résolution de la clé référencée par le Master.
- Section inactive → exclue du manifest (`build_manifest()`) ; ses `produced_by` ne sont pas exécutés (pas de gaspillage compute) ; narrative et visual_specs non rendus.

### 6.3 Impact implémentation

- **Validator `scripts/check_template.py`** :
  - reconnaître le champ `activation` ;
  - vérifier que `key` pointe vers une clé enum déclarée dans `master_from_modeling` (ou `master_from_data`) ;
  - vérifier que l'union des `equals` sur les sections partageant la même `key` **couvre toutes les valeurs de l'enum** (aucune valeur sans section).
- **Template_loader `build_manifest`** :
  - accepter un contexte (dict clé → valeur) ;
  - filtrer les sections inactives avant calcul du DAG.
- **Master** :
  - résoudre `gender_segmentation` avant le GO_BUILD (phase preflight, cohérent avec US-17/18/19).

### 6.4 Justification

Syntaxe déclarative `{key, equals}` préférée à :
- **Expression `{{ ... }}`** — nécessite un évaluateur, validation statique plus faible.
- **`variant_group`** — sur-design pour 2 sections mutex.

Extensible plus tard (ajout futur de `in:`, `not_equals`, …) sans breaking change.

## 7. Ordre de rendu

**`data_analysis_unisex`** : narrative → table `annual_statistics` → bullets LLM post-table → chart `exposure_distribution_by_age`.

**`data_analysis_by_sex`** : narrative → table H → table F → bullets LLM comparative → chart H → chart F.

## 8. Hors scope

- Refactorisation future `statistical_analysis.describe(records, group_by, metrics)` pour collapse de `time_series`, `age_distribution`, `segmentation`, `portfolio_summary` — cf. `memory/project_refactor_mortality_describe.md`. Hors scope de cette section.
- Enrichissement LLM avec contexte métier supplémentaire (`product_list`, `underwriting_rules` V1) — à évaluer après observation de la qualité réelle des bullets interprétatifs.
- Formats `int`, `float2` dans `columns.format` : déjà utilisés dans le préambule (`segmentations.sexe`), supposés supportés par `table_renderer`. Si un format manque (`percent2` non utilisé dans cette spec), arbitrage lors de l'implémentation.

## 9. Critères de done

- `knowledge_base/report_template/mortality_template.yaml` : deux sections + 3 clés nouvelles data_contract.
- `statistical_analysis.time_series` étendu (3 nouveaux champs + param `by_sex`).
- `statistical_analysis.age_distribution` étendu (outputs `distribution_list[_h|_f]`).
- `master.analyze_data_and_request` expose `n_records`.
- `master.classify_request` expose `gender_mode`.
- `scripts/check_template.py` reconnaît `activation` et vérifie la couverture d'enum.
- `template_loader.build_manifest()` accepte un contexte et filtre les sections inactives.
- Tests : `pytest tests/` vert ; `python scripts/check_template.py` vert.
- E2E (après US-26 preamble vert) : génération d'un rapport dans les deux modes `unisex` et `by_sex`.
