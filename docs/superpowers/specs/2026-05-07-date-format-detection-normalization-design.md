# Détection automatique du format de dates + normalisation centralisée

> **Spec issue de la session de brainstorming du 2026-05-07.**

## Contexte

Les CSV chargés par les actuaires arrivent avec des formats de dates hétérogènes : ISO `2018-03-15`, européen `15/03/2018`, parfois mélangés au sein d'un même portefeuille. Les tools actuariels (`builder.exposure`, `statistical_analysis.data_quality`, `time_series`, etc.) parsent les dates avec leur propre logique, ce qui produit deux symptômes observés en production :

- `pd.to_datetime(s, dayfirst=True)` rejette silencieusement les dates ISO → ~50% des lignes du CSV de test exclues sans alerte (`total_exposure=567` au lieu de `1546`).
- Incohérence potentielle entre tools : un tool peut accepter un format que l'autre rejette → résultats divergents.

L'absence d'une étape de normalisation centralisée est la cause racine. La spec décrit comment intégrer la détection de format au moment du mapping des colonnes, normaliser une fois pour toutes en `datetime64[ns]`, et exposer aux tools une seule version "propre" du DataFrame.

## Objectif

1. Détecter automatiquement le format dominant pour chaque colonne date d'un CSV chargé.
2. Alerter l'utilisateur si plus de 5% des lignes d'une colonne ne matchent pas le format dominant.
3. Convertir les colonnes date en `datetime64[ns]` au moment de la normalisation des records.
4. Persister le DataFrame normalisé dans un Parquet dédié, lu par tous les tools de façon transparente.
5. Éliminer la dépendance des tools au format brut (plus de `pd.to_datetime(..., dayfirst=True)` dispersé).

## Architecture

### Pipeline de bout en bout

```
CSV brut (strings, formats potentiellement mixtes)
    │
    ▼
DatasetStore.store(session_id, df_raw)
    └─► sessions/{id}_dataset.parquet            ← brut, archivage
                                                   (jamais touché ensuite)
    │
    ▼ (canvas_app : confirmation column_mapping)
    │
detect_date_formats(df_raw, date_columns)
    └─► {col: {format, n_outliers, pct_outliers, examples}}
    │
    ▼ (master/disambiguation.py)
    │
si pct_outliers > 5% pour au moins 1 colonne :
    │
    ├─ Master émet need_user_input avec
    │  context_key="date_outliers_decision",
    │  options=["continue", "cancel"]
    │
    └─ pause utilisateur (pattern existant _pending_need)
    │
    ▼ (réponse user)
    │
maybe_normalize_records :
    1. Lit df_raw depuis Parquet
    2. Renomme les colonnes (csv → canoniques)
    3. Normalise les valeurs (cause_sortie, sexe)
    4. Pour chaque colonne date : pd.to_datetime(s, format=fmt_detected, errors="coerce")
    5. Drop les outliers (rows avec NaT après parsing)
    │
    ▼
DatasetStore.store_normalized(session_id, df_normalized)
    └─► sessions/{id}_dataset_normalized.parquet  ← source unique pour les tools
    │
    ▼
data_store["input_records"] = df_normalized
data_store["study_plan"]["date_formats"] = {col: fmt}  ← audit / debug
    │
    ▼
Tools (via MemoryManager.load_dataframe) :
    - récupèrent automatiquement le Parquet normalisé si présent
    - sinon fallback sur le brut (pour les sessions où la normalisation n'a pas
      encore eu lieu — rétro-compatibilité)
    - reçoivent des Series datetime64[ns], plus aucune logique de format à gérer
```

### Single source of truth

Après normalisation, **les tools ne voient qu'une seule version du DataFrame** : la version normalisée (Parquet `_normalized`). Le brut est conservé pour audit mais jamais lu par les tools. La méthode `MemoryManager.load_dataframe` priorise le Parquet normalisé.

Conséquence : un tool ne peut pas "oublier" de gérer le format. Il reçoit `df["date_entree"]` comme `Series[datetime64[ns]]` directement utilisable :
```python
ages = (df["date_sortie"] - df["date_naissance"]).dt.days / 365.25  # marche tel quel
```

### Format standard : `datetime64[ns]` natif pandas

Choix architectural : on convertit les dates au type **natif pandas** (Timestamp), pas en string ISO. Avantages :
- Opérations datetime natives (soustraction, extraction d'année, comparaisons).
- Type-safe : un tool qui attend une date ne peut pas recevoir une string par erreur.
- Sérialisation JSON propre (`'2018-03-15T00:00:00'`) si jamais besoin.

Le format string détecté (`"%Y-%m-%d"`, etc.) est conservé dans `study_plan["date_formats"]` **uniquement pour audit / debug** ; aucun tool ne le consomme.

## Composants à créer / modifier

### Création

| Fichier | Rôle |
|---|---|
| `tools/utils/date_parsing.py` | Helper partagé : `detect_dominant_format(series)` + `parse_with_dominant_format(series)`. Liste de candidats : `[%Y-%m-%d, %d/%m/%Y, %Y/%m/%d, %d-%m-%Y, %m/%d/%Y, %d.%m.%Y]`. |
| `tools/statistical_analysis/detect_date_formats.py` | Tool exposé au catalogue : profile chaque colonne date d'un df, retourne `{col: {format, n_outliers, pct_outliers, examples}}`. Utilise le helper. |

### Modification

| Fichier | Modification |
|---|---|
| `agents/master/disambiguation.py` | Après confirmation du `column_mapping`, appelle `detect_date_formats`. Si `max(pct_outliers) > 5%`, pose un `_pending_need` (UI Option B retenu lors du brainstorm) avec `context_key="date_outliers_decision"` et options `["continue", "cancel"]`. Sinon stocke directement les formats dans `study_plan["date_formats"]`. |
| `tools/master/normalize_records.py` | Étape supplémentaire : après le rename + value_mapping, parse les colonnes date au format choisi (`pd.to_datetime(s, format=fmt, errors="coerce")`) et drop les rows avec NaT (= outliers). |
| `session/dataset_store.py` | Ajouter `store_normalized(session_id, df)` qui écrit dans `{id}_dataset_normalized.parquet` et `load_normalized_by_session(session_id)`. |
| `session/memory_manager.py` | `load_dataframe()` priorise le `_normalized` si présent, fallback sur le brut. |
| `tools/builder/exposure.py` | Suppression du parsing `pd.to_datetime(..., dayfirst=True)` — le df reçu a déjà des `datetime64`. |
| `tools/statistical_analysis/data_quality.py` | Idem — `_try_parse_date` n'a plus besoin de tester plusieurs formats (les colonnes datetime sont déjà parsées). Il vérifie juste `dtype == 'datetime64[ns]'`. |
| `tools/statistical_analysis/time_series.py` | Idem — utiliser `df["date_xxx"].dt.year` direct. |
| `canvas_app.py` | Affichage de l'alerte UI lorsque le `need_user_input` `date_outliers_decision` est posé : *"⚠ Dans `date_entree`, X lignes (Y%) ne matchent pas le format dominant. Exemples : ... Continuer en excluant ces lignes ou annuler ?"*. |

### Tests à créer

| Fichier | Couverture |
|---|---|
| `tests/test_date_parsing_helper.py` | Détection format dominant sur série uniforme, mixte, vide. Helper `parse_with_dominant_format` retourne datetime64 + count d'outliers. |
| `tests/test_detect_date_formats_tool.py` | Tool sur df réaliste : 3 colonnes en formats différents, certaines avec outliers. |
| `tests/test_disambiguation_date_outliers.py` | Disambiguation : ≤ 5% → silencieux, formats stockés ; > 5% → `_pending_need` posé. |
| `tests/test_normalize_records_datetime.py` | Normalize_records produit un df avec `dtype==datetime64[ns]` sur les colonnes date. |
| `tests/test_dataset_store_normalized.py` | Round-trip Parquet normalisé : store + load. |
| `tests/test_tools_consume_normalized_df.py` | Intégration : appel à `builder.exposure` reçoit un df déjà parsé, calcul correct sur le CSV de test. |

## UI : alerte outliers > 5%

Quand `max(pct_outliers) > 5%`, Master pose la question via le pattern `_pending_need`. Format :

```yaml
context_key: "date_outliers_decision"
question: |
  ⚠ La colonne `date_entree` contient 87 lignes (8.7%) qui ne matchent
  pas le format dominant détecté (%Y-%m-%d).
  Exemples : "20/13/2020", "00/00/0000", "invalid".
  Voulez-vous continuer en excluant ces lignes, ou annuler pour
  corriger le CSV ?
options: ["continue", "cancel"]
default: "continue"
```

User répond `continue` → normalisation procède, outliers exclus.  
User répond `cancel` → Master réinitialise la session, retour à l'étape de chargement CSV.

## Garanties

1. **Source unique** : un seul Parquet de référence pour les tools, plus jamais d'incohérence brut/normalisé.
2. **Idempotence** : si le mapping change, on régénère le `_normalized.parquet`. Tous les tools reprennent la version à jour automatiquement.
3. **Audit** : le brut original est conservé. Un dev qui veut comprendre la transformation peut comparer les deux Parquets.
4. **Performance** : parsing fait une fois au moment du mapping. Tools n'ont rien à parser.
5. **Pas d'oubli possible** : un tool reçoit `Series[datetime64[ns]]`. Une opération datetime fonctionne nativement. Une erreur de type produit une exception immédiate (pas un faux positif silencieux).

## Hors scope

- Édition manuelle ligne par ligne des outliers (option C écartée lors du brainstorm — over-engineering).
- Choix d'un format alternatif via dropdown UI (jugé inutile — la détection auto est fiable à >99%).
- Re-détection automatique si l'utilisateur ré-importe un CSV différent dans la même session (cas rare, à traiter dans une US dédiée).
- Migration des sessions existantes (rétro-compatibilité gérée par fallback sur le brut).

## Critères d'acceptation

- [ ] Sur le CSV de test `Portefeuille/portefeuille_test_1000.csv` (ISO), `total_exposure` après pipeline = ~1546 années-personne (vs 567 actuellement).
- [ ] Aucun tool aval ne contient plus de `pd.to_datetime(..., dayfirst=True)`.
- [ ] Une session avec ≤ 5% outliers passe silencieusement, formats stockés dans `study_plan["date_formats"]`.
- [ ] Une session avec > 5% outliers déclenche un `_pending_need` UI affiché dans canvas_app.
- [ ] Tous les tests existants restent verts (209 baseline) + ~20 nouveaux tests.
