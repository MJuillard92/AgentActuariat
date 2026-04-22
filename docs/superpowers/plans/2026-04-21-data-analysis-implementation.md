# Data Analysis Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduire la section `data_analysis` (avec préambule `data_preprocessing`) dans `mortality_template.yaml`, selon le design [2026-04-21-data-analysis-design.md](../specs/2026-04-21-data-analysis-design.md) (v3).

**Architecture:** Un nouveau tool `preprocessing.clean_records` (1er nœud du DAG Builder), trois tools existants étendus (`statistical_analysis.time_series`, `.age_distribution`, `master.classify_request`), un mécanisme d'activation conditionnelle dans `template_loader` + `check_template`, puis trois nouvelles sections YAML (`data_preprocessing`, `data_analysis_unisex`, `data_analysis_by_sex`) avec rebranchement de tous les builder tools existants sur `cleaned_records`.

**Tech Stack:** Python 3.11 + pandas, pytest, PyYAML.

**Prérequis :** US-26 (E2E preamble vert) — à valider avant de démarrer ce plan.

**Convention :** chaque US = une tranche TDD (test rouge → impl minimale → test vert → commit). Les US sont numérotées US-27+ en continuité du [plan refactor antérieur](2026-04-20-refactor-yaml-master-builder-writer.md).

---

## Structure fichiers

**Créés :**
- `tools/preprocessing/__init__.py`
- `tools/preprocessing/clean_records.py`
- `tests/test_preprocessing_clean_records.py`
- `tests/test_time_series_extensions.py`
- `tests/test_age_distribution_extensions.py`
- `tests/test_activation_mechanism.py`
- `tests/test_data_analysis_e2e.py`

**Modifiés :**
- `tools/statistical_analysis/time_series.py` (colonnes + param `by_sex`)
- `tools/statistical_analysis/age_distribution.py` (outputs `distribution_list[_h|_f]`)
- `tools/master/classify_request.py` (output `gender_mode`)
- `tools/catalogue.yaml` (enregistrement `preprocessing.clean_records`)
- `tools/tool_registry.py` (si enregistrement manuel requis)
- `scripts/check_template.py` (reconnaissance `activation`)
- `knowledge_base/report_template/template_loader.py` (`build_manifest` context-aware)
- `knowledge_base/report_template/mortality_template.yaml` (nouveaux keys + sections + rebranch)
- `tests/test_master_classify_request.py` (tests `gender_mode`)
- `tests/test_template_loader.py` (tests activation filter)
- `tests/test_validator.py` (tests `activation` field)

---

### US-27 — Tool `preprocessing.clean_records` : squelette + règle R1 (contrats sans effet)

**Files:**
- Create: `tools/preprocessing/__init__.py` (vide)
- Create: `tools/preprocessing/clean_records.py`
- Create: `tests/test_preprocessing_clean_records.py`

- [ ] **Étape 1 — Écrire le test rouge pour R1**

```python
# tests/test_preprocessing_clean_records.py
import pandas as pd
import pytest
from tools.preprocessing.clean_records import run


def _base_df():
    """DataFrame minimal valide (pas d'exclusion) pour usage dans les tests."""
    return pd.DataFrame({
        "date_naissance": ["1970-01-01", "1980-01-01"],
        "date_entree":    ["2010-01-01", "2011-01-01"],
        "date_sortie":    ["2015-01-01", "2016-01-01"],
        "cause_sortie":   ["deces", "autre"],
        "sexe":           ["H", "F"],
    })


def test_r1_removes_sans_objet_contracts():
    df = _base_df()
    df = pd.concat([df, pd.DataFrame({
        "date_naissance": ["1990-01-01"],
        "date_entree":    ["2010-01-01"],
        "date_sortie":    ["2015-01-01"],
        "cause_sortie":   ["sans_objet"],
        "sexe":           ["H"],
    })], ignore_index=True)

    result = run(df)

    assert len(result["cleaned_records"]) == 2
    report = result["exclusion_report"]
    assert report["initial_count"] == 3
    assert report["final_count"] == 2
    r1 = next(r for r in report["rules"] if r["rule_id"] == "R1")
    assert r1["count"] == 1
    assert r1["rule_label"] == "Contrats sans effet (cause de sortie « sans objet »)"
```

- [ ] **Étape 2 — Exécuter pour vérifier l'échec**

Run: `pytest tests/test_preprocessing_clean_records.py::test_r1_removes_sans_objet_contracts -v`
Expected: `ModuleNotFoundError: tools.preprocessing.clean_records`.

- [ ] **Étape 3 — Implémentation minimale**

```python
# tools/preprocessing/__init__.py
```
(fichier vide)

```python
# tools/preprocessing/clean_records.py
"""
TOOL CONTRACT — preprocessing.clean_records
═══════════════════════════════════════════

Premier nœud du DAG Builder. Reçoit les records normalisés par Master,
applique les règles de retraitement figées, produit la base assainie
et le rapport d'exclusions consommé par la section data_preprocessing.
"""
from __future__ import annotations
import pandas as pd


def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    initial_count = len(df)
    rules_report: list[dict] = []
    current = df.copy()

    # R1 — Contrats sans effet
    mask_r1 = current["cause_sortie"].astype(str).str.lower() == "sans_objet"
    count_r1 = int(mask_r1.sum())
    current = current[~mask_r1].copy()
    rules_report.append({
        "rule_id":    "R1",
        "rule_label": "Contrats sans effet (cause de sortie « sans objet »)",
        "count":      count_r1,
        "detail":     {},
    })

    return {
        "cleaned_records": current.reset_index(drop=True),
        "exclusion_report": {
            "initial_count": initial_count,
            "final_count":   len(current),
            "rules":         rules_report,
        },
    }
```

- [ ] **Étape 4 — Exécuter pour vérifier le succès**

Run: `pytest tests/test_preprocessing_clean_records.py::test_r1_removes_sans_objet_contracts -v`
Expected: PASS.

- [ ] **Étape 5 — Commit**

```bash
git add tools/preprocessing/__init__.py tools/preprocessing/clean_records.py tests/test_preprocessing_clean_records.py
git commit -m "feat(US-27): preprocessing.clean_records — règle R1 (contrats sans effet)"
```

---

### US-28 — `clean_records` : règles R2–R5 (âges aberrants)

**Files:**
- Modify: `tools/preprocessing/clean_records.py`
- Modify: `tests/test_preprocessing_clean_records.py`

- [ ] **Étape 1 — Ajouter les tests rouges**

```python
# Ajouter dans tests/test_preprocessing_clean_records.py

def _row(dn, de, ds, cs="autre", sx="H"):
    return {"date_naissance": dn, "date_entree": de, "date_sortie": ds,
            "cause_sortie": cs, "sexe": sx}


def test_r2_removes_negative_entry_age():
    df = pd.DataFrame([
        _row("2015-01-01", "2010-01-01", "2020-01-01"),  # âge entrée < 0
        _row("1970-01-01", "2010-01-01", "2020-01-01"),  # ok
    ])
    result = run(df)
    assert result["exclusion_report"]["final_count"] == 1
    r2 = next(r for r in result["exclusion_report"]["rules"] if r["rule_id"] == "R2")
    assert r2["count"] == 1


def test_r3_removes_negative_exit_age():
    df = pd.DataFrame([
        _row("1970-01-01", "2010-01-01", "1960-01-01"),  # âge sortie < 0
        _row("1970-01-01", "2010-01-01", "2020-01-01"),  # ok
    ])
    result = run(df)
    r3 = next(r for r in result["exclusion_report"]["rules"] if r["rule_id"] == "R3")
    assert r3["count"] == 1


def test_r4_removes_entry_age_over_100():
    df = pd.DataFrame([
        _row("1900-01-01", "2020-01-01", "2021-01-01"),  # 120 ans à l'entrée
        _row("1970-01-01", "2010-01-01", "2020-01-01"),  # ok
    ])
    result = run(df)
    r4 = next(r for r in result["exclusion_report"]["rules"] if r["rule_id"] == "R4")
    assert r4["count"] == 1


def test_r5_removes_exit_age_over_100():
    df = pd.DataFrame([
        _row("1900-01-01", "1950-01-01", "2005-01-01"),  # 105 à la sortie
        _row("1970-01-01", "2010-01-01", "2020-01-01"),  # ok
    ])
    result = run(df)
    r5 = next(r for r in result["exclusion_report"]["rules"] if r["rule_id"] == "R5")
    assert r5["count"] == 1
```

- [ ] **Étape 2 — Vérifier l'échec**

Run: `pytest tests/test_preprocessing_clean_records.py -v -k "r2 or r3 or r4 or r5"`
Expected: 4 FAIL (règles inexistantes ou count=0).

- [ ] **Étape 3 — Implémentation**

Remplacer le corps de `run` dans `tools/preprocessing/clean_records.py` :

```python
def _ages(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    dn = pd.to_datetime(df["date_naissance"], errors="coerce")
    de = pd.to_datetime(df["date_entree"],    errors="coerce")
    ds = pd.to_datetime(df["date_sortie"],    errors="coerce")
    age_entree = (de - dn).dt.days / 365.25
    age_sortie = (ds - dn).dt.days / 365.25
    return age_entree, age_sortie


def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    initial_count = len(df)
    rules_report: list[dict] = []
    current = df.copy()

    def _apply(mask: pd.Series, rule_id: str, rule_label: str, detail: dict | None = None) -> None:
        nonlocal current
        m = mask.reindex(current.index, fill_value=False)
        count = int(m.sum())
        current = current[~m].copy()
        rules_report.append({
            "rule_id":    rule_id,
            "rule_label": rule_label,
            "count":      count,
            "detail":     detail or {},
        })

    # R1 — Contrats sans effet
    mask = current["cause_sortie"].astype(str).str.lower() == "sans_objet"
    _apply(mask, "R1", "Contrats sans effet (cause de sortie « sans objet »)")

    # R2–R5 — âges aberrants
    ae, as_ = _ages(current)
    _apply(ae < 0,    "R2", "Âge à l'entrée négatif")
    ae, as_ = _ages(current)  # recalcul après R2
    _apply(as_ < 0,   "R3", "Âge à la sortie négatif")
    ae, as_ = _ages(current)
    _apply(ae > 100,  "R4", "Âge à l'entrée supérieur à 100 ans")
    ae, as_ = _ages(current)
    _apply(as_ > 100, "R5", "Âge à la sortie supérieur à 100 ans")

    return {
        "cleaned_records": current.reset_index(drop=True),
        "exclusion_report": {
            "initial_count": initial_count,
            "final_count":   len(current),
            "rules":         rules_report,
        },
    }
```

- [ ] **Étape 4 — Vérifier**

Run: `pytest tests/test_preprocessing_clean_records.py -v`
Expected: 5 PASS.

- [ ] **Étape 5 — Commit**

```bash
git add tools/preprocessing/clean_records.py tests/test_preprocessing_clean_records.py
git commit -m "feat(US-28): clean_records — règles R2–R5 (âges aberrants)"
```

---

### US-29 — `clean_records` : règle R6 (sortie < entrée) + cumulativité

**Files:**
- Modify: `tools/preprocessing/clean_records.py`
- Modify: `tests/test_preprocessing_clean_records.py`

- [ ] **Étape 1 — Tests rouges**

```python
def test_r6_removes_exit_before_entry():
    df = pd.DataFrame([
        _row("1970-01-01", "2020-01-01", "2010-01-01"),  # sortie < entrée
        _row("1970-01-01", "2010-01-01", "2020-01-01"),  # ok
    ])
    result = run(df)
    r6 = next(r for r in result["exclusion_report"]["rules"] if r["rule_id"] == "R6")
    assert r6["count"] == 1


def test_rules_are_cumulative_no_double_counting():
    # Une ligne violait plusieurs règles : ne doit être comptée que dans la 1ère déclenchée.
    df = pd.DataFrame([
        # ligne aberrante : sans_objet ET âge entrée > 100
        _row("1900-01-01", "2020-01-01", "2021-01-01", cs="sans_objet"),
        _row("1970-01-01", "2010-01-01", "2020-01-01"),  # ok
    ])
    result = run(df)
    r1 = next(r for r in result["exclusion_report"]["rules"] if r["rule_id"] == "R1")
    r4 = next(r for r in result["exclusion_report"]["rules"] if r["rule_id"] == "R4")
    assert r1["count"] == 1
    assert r4["count"] == 0  # déjà retirée par R1
    assert result["exclusion_report"]["final_count"] == 1
```

- [ ] **Étape 2 — Vérifier l'échec**

Run: `pytest tests/test_preprocessing_clean_records.py::test_r6_removes_exit_before_entry tests/test_preprocessing_clean_records.py::test_rules_are_cumulative_no_double_counting -v`
Expected: 2 FAIL (R6 inexistante).

- [ ] **Étape 3 — Impl : ajouter R6 en fin de chaîne**

Dans `tools/preprocessing/clean_records.py`, après R5, ajouter :

```python
    ae, as_ = _ages(current)
    _apply(as_ < ae, "R6", "Âge à la sortie inférieur à l'âge à l'entrée")
```

- [ ] **Étape 4 — Vérifier**

Run: `pytest tests/test_preprocessing_clean_records.py -v`
Expected: 7 PASS.

- [ ] **Étape 5 — Commit**

```bash
git add tools/preprocessing/clean_records.py tests/test_preprocessing_clean_records.py
git commit -m "feat(US-29): clean_records — règle R6 + test cumulativité"
```

---

### US-30 — Enregistrement `preprocessing.clean_records` dans le catalogue + registry

**Files:**
- Modify: `tools/preprocessing/clean_records.py` (ajouter header CATALOGUE METADATA)
- Regenerate: `tools/catalogue.yaml` (via `python tools/catalogue.py --force`)

- [ ] **Étape 1 — Test rouge : vérifier la présence du tool dans le catalogue**

```python
# tests/test_preprocessing_clean_records.py (append)
import yaml
from pathlib import Path

def test_tool_registered_in_catalogue():
    catalogue = yaml.safe_load(Path("tools/catalogue.yaml").read_text())
    assert "preprocessing.clean_records" in catalogue["tools"]
    entry = catalogue["tools"]["preprocessing.clean_records"]
    assert entry["domain"] == "preprocessing"
    assert entry["client_visible"] is True
```

Run: `pytest tests/test_preprocessing_clean_records.py::test_tool_registered_in_catalogue -v`
Expected: FAIL (tool absent du catalogue).

- [ ] **Étape 2 — Compléter le docstring avec le format CATALOGUE METADATA**

En tête de `tools/preprocessing/clean_records.py`, ajouter le docstring complet (calqué sur `tools/statistical_analysis/time_series.py` §IDENTITY → §CATALOGUE METADATA) :

```python
"""
TOOL CONTRACT — preprocessing.clean_records
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : preprocessing.clean_records
domain        : preprocessing
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-21

DESCRIPTION
-----------
Premier nœud du DAG Builder. Applique 6 règles figées de retraitement
(R1 contrats sans effet, R2–R5 âges aberrants, R6 sortie < entrée),
produit la base assainie et un rapport d'exclusions détaillé.

WHEN TO USE
-----------
Systématiquement, avant tout tool statistical_analysis.* ou builder.*
consommant des records. Les tools en aval reçoivent cleaned_records,
jamais input_records brut.

WHEN NOT TO USE
---------------
N/A — toujours appelé.

PREREQUISITES
-------------
required_tools: [master.normalize_records]
required_data_store_keys: []
Note: reçoit df (DataFrame) déjà normalisé par Master (column_mapping,
value_mapping appliqués).

INPUTS
------
params: {}

OUTPUTS
-------
data_store_keys_written:
  - cleaned_records : DataFrame — records après exclusions
  - exclusion_report : dict — initial_count, final_count, rules

QUALITY GATES
-------------
BLOCKING:
  - final_count == 0 → retourne erreur.
NON-BLOCKING:
  - final_count < 0.5 × initial_count → warning.

CATALOGUE METADATA
------------------
display_name      : Retraitement des données aberrantes
short_description : Applique 6 règles de retraitement et produit un rapport d'exclusions.
domain            : preprocessing
capability_group  : preprocessing
depends_on        : [master.normalize_records]
required_by       : [builder.exposure, statistical_analysis.time_series, statistical_analysis.age_distribution, statistical_analysis.segmentation]
client_visible    : true
"""
```

- [ ] **Étape 3 — Régénérer le catalogue**

Run: `python tools/catalogue.py --force`
Expected: `catalogue.yaml` régénéré, mentionne `preprocessing.clean_records`.

- [ ] **Étape 4 — Vérifier**

Run: `pytest tests/test_preprocessing_clean_records.py -v`
Expected: 8 PASS.

- [ ] **Étape 5 — Commit**

```bash
git add tools/preprocessing/clean_records.py tools/catalogue.yaml tests/test_preprocessing_clean_records.py
git commit -m "feat(US-30): enregistre preprocessing.clean_records dans catalogue"
```

---

### US-31 — Extension `statistical_analysis.time_series` : colonnes `age_moyen_entres`, `age_moyen_deces`, `taux_deces`

**Files:**
- Modify: `tools/statistical_analysis/time_series.py`
- Create: `tests/test_time_series_extensions.py`

- [ ] **Étape 1 — Test rouge**

```python
# tests/test_time_series_extensions.py
import pandas as pd
from tools.statistical_analysis.time_series import run


def _fixture_df():
    return pd.DataFrame({
        "date_naissance": ["1970-01-01", "1980-01-01", "1975-01-01"],
        "date_entree":    ["2010-06-01", "2010-03-01", "2011-01-01"],
        "date_sortie":    ["2015-04-01", "2013-09-01", "2014-12-01"],
        "cause_sortie":   ["deces",      "deces",      "autre"],
        "sexe":           ["H",          "F",          "H"],
    })


def test_series_includes_age_moyen_entres():
    result = run(_fixture_df())
    row_2010 = next(r for r in result["serie"] if r["annee"] == 2010)
    assert "age_moyen_entres" in row_2010
    assert 30 <= row_2010["age_moyen_entres"] <= 50


def test_series_includes_age_moyen_deces():
    result = run(_fixture_df())
    row_2013 = next(r for r in result["serie"] if r["annee"] == 2013)
    # un décès en 2013 (date_sortie=2013-09-01, dn=1980) ~ 33 ans
    assert 30 <= row_2013["age_moyen_deces"] <= 40


def test_series_includes_taux_deces():
    result = run(_fixture_df())
    for row in result["serie"]:
        if row["exposition_pa"] > 0:
            expected = row["nb_deces"] / row["exposition_pa"] * 1000
            assert abs(row["taux_deces"] - expected) < 1e-6
```

- [ ] **Étape 2 — Vérifier l'échec**

Run: `pytest tests/test_time_series_extensions.py -v`
Expected: 3 FAIL.

- [ ] **Étape 3 — Impl**

Dans `tools/statistical_analysis/time_series.py`, à l'intérieur de la boucle `for year in range(...)`, après le calcul de `expo`, ajouter le calcul des trois colonnes :

```python
        # Âge moyen à l'entrée pour les contrats entrés dans l'année
        if len(entres) > 0:
            ent_dn = pd.to_datetime(df.loc[entres.index, _find_col(df, _CS["date_naissance"]["candidates"])],
                                    format="mixed", dayfirst=True, errors="coerce")
            ages_entres = (entres["_entree"] - ent_dn).dt.days / 365.25
            age_moyen_entres = round(float(ages_entres.mean()), 2) if ages_entres.notna().any() else None
        else:
            age_moyen_entres = None

        # Âge moyen au décès pour les décès survenus dans l'année
        if nb_deces > 0 and exit_col:
            deces_mask = valid["_is_dead"] & (valid["_sortie"].dt.year == year)
            dec_dn = pd.to_datetime(df.loc[valid[deces_mask].index, _find_col(df, _CS["date_naissance"]["candidates"])],
                                    format="mixed", dayfirst=True, errors="coerce")
            ages_deces = (valid.loc[deces_mask, "_sortie"] - dec_dn).dt.days / 365.25
            age_moyen_deces = round(float(ages_deces.mean()), 2) if ages_deces.notna().any() else None
        else:
            age_moyen_deces = None

        # Taux de décès (‰ PA)
        taux_deces = round(nb_deces / expo * 1000, 2) if expo > 0 else 0.0

        rows.append({
            "annee":             year,
            "nb_entres":         nb_entres,
            "nb_deces":          nb_deces,
            "exposition_pa":     round(float(expo), 1),
            "age_moyen_entres":  age_moyen_entres,
            "age_moyen_deces":   age_moyen_deces,
            "taux_deces":        taux_deces,
        })
```

Remplacer l'ancien `rows.append({...})` par le nouveau bloc.

- [ ] **Étape 4 — Vérifier**

Run: `pytest tests/test_time_series_extensions.py tests/test_pipeline_preamble_e2e.py -v`
Expected: PASS (incluant non-régression préambule).

- [ ] **Étape 5 — Commit**

```bash
git add tools/statistical_analysis/time_series.py tests/test_time_series_extensions.py
git commit -m "feat(US-31): time_series — colonnes age_moyen_entres/deces + taux_deces"
```

---

### US-32 — Extension `statistical_analysis.time_series` : param `by_sex` → `serie_h`, `serie_f`

**Files:**
- Modify: `tools/statistical_analysis/time_series.py`
- Modify: `tests/test_time_series_extensions.py`

- [ ] **Étape 1 — Test rouge**

```python
def test_by_sex_produces_serie_h_and_serie_f():
    result = run(_fixture_df(), params={"by_sex": True})
    assert "serie_h" in result
    assert "serie_f" in result
    # H : 2 contrats, F : 1 contrat
    total_entres_h = sum(r["nb_entres"] for r in result["serie_h"])
    total_entres_f = sum(r["nb_entres"] for r in result["serie_f"])
    assert total_entres_h == 2
    assert total_entres_f == 1


def test_by_sex_false_omits_sex_keys():
    result = run(_fixture_df(), params={"by_sex": False})
    assert "serie_h" not in result
    assert "serie_f" not in result
```

- [ ] **Étape 2 — Vérifier l'échec**

Run: `pytest tests/test_time_series_extensions.py -k "by_sex" -v`
Expected: 2 FAIL.

- [ ] **Étape 3 — Impl**

Refactorer `run` pour extraire la logique annuelle en helper, et l'appeler 1× (global) ou 3× (global + H + F) selon `by_sex`.

```python
def _compute_annual(valid: pd.DataFrame, df: pd.DataFrame, exit_col: str | None,
                    year_min: int, year_max: int) -> list[dict]:
    # (déplacer ici toute la boucle for year ... rows.append({...}))
    ...


def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    p = params or {}
    by_sex = bool(p.get("by_sex", False))

    # (garder le parsing initial jusqu'à valid, year_min, year_max)
    ...

    result: dict = {
        "serie":     _compute_annual(valid, df, exit_col, year_min, year_max),
        "annee_min": year_min,
        "annee_max": year_max,
        "nb_annees": year_max - year_min + 1,
    }

    if by_sex:
        from agents.mortality.dictionary.column_schema import find_col as _find_col, COLUMN_SCHEMA as _CS
        sexe_col = _find_col(df, _CS["sexe"]["candidates"])
        if sexe_col:
            sexe = df[sexe_col].astype(str).str.upper().str.strip()
            mask_h = sexe.isin(["H", "M", "HOMME", "MALE", "1"])
            mask_f = sexe.isin(["F", "FEMME", "FEMALE", "2"])
            result["serie_h"] = _compute_annual(valid[mask_h.reindex(valid.index, fill_value=False)], df, exit_col, year_min, year_max)
            result["serie_f"] = _compute_annual(valid[mask_f.reindex(valid.index, fill_value=False)], df, exit_col, year_min, year_max)

    # (garder anomalies et retour)
    ...
    return result
```

- [ ] **Étape 4 — Vérifier**

Run: `pytest tests/test_time_series_extensions.py tests/test_pipeline_preamble_e2e.py -v`
Expected: PASS.

- [ ] **Étape 5 — Commit**

```bash
git add tools/statistical_analysis/time_series.py tests/test_time_series_extensions.py
git commit -m "feat(US-32): time_series — param by_sex (serie_h, serie_f)"
```

---

### US-33 — Extension `statistical_analysis.age_distribution` : outputs `distribution_list[_h|_f]`

**Files:**
- Modify: `tools/statistical_analysis/age_distribution.py`
- Create: `tests/test_age_distribution_extensions.py`

- [ ] **Étape 1 — Test rouge**

```python
# tests/test_age_distribution_extensions.py
import pandas as pd
from tools.statistical_analysis.age_distribution import run


def _fixture_df():
    return pd.DataFrame({
        "date_naissance": [f"19{y}-01-01" for y in range(50, 90, 5)],
        "date_entree":    ["2010-01-01"] * 8,
        "sexe":           ["H", "F", "H", "F", "H", "F", "H", "F"],
    })


def test_distribution_list_is_list_of_dicts():
    result = run(_fixture_df())
    assert "distribution_list" in result
    assert isinstance(result["distribution_list"], list)
    for item in result["distribution_list"]:
        assert set(item.keys()) == {"tranche", "nb_contrats"}


def test_distribution_list_matches_distribution_dict():
    result = run(_fixture_df())
    dict_items = result["distribution"].items()
    list_items = [(r["tranche"], r["nb_contrats"]) for r in result["distribution_list"]]
    assert list(dict_items) == list_items


def test_by_sex_produces_distribution_list_h_and_f():
    result = run(_fixture_df(), params={"by_sex": True})
    assert "distribution_list_h" in result
    assert "distribution_list_f" in result
    total_h = sum(r["nb_contrats"] for r in result["distribution_list_h"])
    total_f = sum(r["nb_contrats"] for r in result["distribution_list_f"])
    assert total_h == 4
    assert total_f == 4
```

- [ ] **Étape 2 — Vérifier l'échec**

Run: `pytest tests/test_age_distribution_extensions.py -v`
Expected: 3 FAIL.

- [ ] **Étape 3 — Impl**

Dans `tools/statistical_analysis/age_distribution.py`, après la construction de `result["distribution"]`, ajouter :

```python
    result["distribution_list"] = [
        {"tranche": k, "nb_contrats": v} for k, v in result["distribution"].items()
    ]

    if by_sex:
        if "distribution_h" in result:
            result["distribution_list_h"] = [
                {"tranche": k, "nb_contrats": v} for k, v in result["distribution_h"].items()
            ]
        if "distribution_f" in result:
            result["distribution_list_f"] = [
                {"tranche": k, "nb_contrats": v} for k, v in result["distribution_f"].items()
            ]
```

- [ ] **Étape 4 — Vérifier**

Run: `pytest tests/test_age_distribution_extensions.py -v`
Expected: 3 PASS.

- [ ] **Étape 5 — Commit**

```bash
git add tools/statistical_analysis/age_distribution.py tests/test_age_distribution_extensions.py
git commit -m "feat(US-33): age_distribution — outputs distribution_list[_h|_f]"
```

---

### US-34 — Extension `master.classify_request` : output `gender_mode`

**Files:**
- Modify: `tools/master/classify_request.py`
- Modify: `tests/test_master_classify_request.py`

- [ ] **Étape 1 — Test rouge**

```python
# tests/test_master_classify_request.py (append)

def test_classify_detects_by_sex_mode():
    result = run({"request": "Construis-moi une table H/F"}, {})
    assert result["gender_mode"] == "by_sex"


def test_classify_detects_unisex_mode_by_default():
    result = run({"request": "Construis-moi une table de mortalité sur mon portefeuille"}, {})
    assert result["gender_mode"] == "unisex"


def test_classify_detects_unisex_explicit():
    result = run({"request": "Je veux une table unisex"}, {})
    assert result["gender_mode"] == "unisex"
```

- [ ] **Étape 2 — Vérifier l'échec**

Run: `pytest tests/test_master_classify_request.py -k "gender" -v`
Expected: 3 FAIL.

- [ ] **Étape 3 — Impl**

Modifier `tools/master/classify_request.py` :

```python
_BY_SEX_PATTERNS = ("h/f", "h / f", "par sexe", "par genre", "masculin et féminin")


def run(data: dict, params: dict) -> dict:
    request = str(data.get("request", "")).lower()
    gender_mode = "by_sex" if any(p in request for p in _BY_SEX_PATTERNS) else "unisex"
    return {
        "objective":   _ALLOWED[0],
        "gender_mode": gender_mode,
    }
```

- [ ] **Étape 4 — Vérifier**

Run: `pytest tests/test_master_classify_request.py -v`
Expected: tous PASS.

- [ ] **Étape 5 — Commit**

```bash
git add tools/master/classify_request.py tests/test_master_classify_request.py
git commit -m "feat(US-34): classify_request — output gender_mode (heuristique mots-clés)"
```

---

### US-35 — Validator `check_template` : reconnaissance du champ `activation`

**Files:**
- Modify: `scripts/check_template.py`
- Modify: `tests/test_validator.py`

- [ ] **Étape 1 — Tests rouges**

```python
# tests/test_validator.py (append)

def test_activation_field_is_recognized(tmp_path, minimal_template):
    minimal_template["sections"][0]["activation"] = {"key": "study_objective", "equals": "construction_table_mortalite"}
    tpl = tmp_path / "t.yaml"
    tpl.write_text(yaml.safe_dump(minimal_template))
    errors = validate(tpl)
    assert errors == []


def test_activation_key_must_reference_enum_in_data_contract(tmp_path, minimal_template):
    minimal_template["sections"][0]["activation"] = {"key": "nonexistent_key", "equals": "foo"}
    tpl = tmp_path / "t.yaml"
    tpl.write_text(yaml.safe_dump(minimal_template))
    errors = validate(tpl)
    assert any("nonexistent_key" in e for e in errors)


def test_activation_coverage_must_be_exhaustive(tmp_path, minimal_template):
    # Deux sections avec activation sur gender_segmentation (enum [unisex, by_sex])
    # mais seule unisex est couverte.
    minimal_template["data_contract"]["master_from_modeling"].append({
        "key": "gender_segmentation",
        "type": "enum",
        "allowed": ["unisex", "by_sex"],
        "description": "...",
        "produced_by": {"tool": "master.classify_request", "inputs": {}, "output_mapping": {"gender_mode": "gender_segmentation"}},
        "confirm_with_user": True,
    })
    minimal_template["sections"].append({
        "id": "variant_unisex",
        "label": "U",
        "required": True,
        "dependencies": [],
        "activation": {"key": "gender_segmentation", "equals": "unisex"},
        "narrative": {"text": ""},
        "llm_directives": {"tone": "", "length_words": [1, 2], "rag_query": ""},
        "visual_specs": [],
    })
    tpl = tmp_path / "t.yaml"
    tpl.write_text(yaml.safe_dump(minimal_template))
    errors = validate(tpl)
    assert any("by_sex" in e for e in errors)
```

Ajouter la fixture `minimal_template` si elle n'existe pas déjà (s'inspirer de `tests/test_template_contract.py`).

- [ ] **Étape 2 — Vérifier l'échec**

Run: `pytest tests/test_validator.py -k activation -v`
Expected: 3 FAIL.

- [ ] **Étape 3 — Impl**

Dans `scripts/check_template.py`, ajouter une fonction `_validate_activation(template, errors)` appelée depuis la validation principale :

```python
def _validate_activation(template: dict, errors: list[str]) -> None:
    """Vérifie la syntaxe et la couverture d'enum des champs `activation`."""
    # Index des enums dans master_from_data + master_from_modeling
    enums: dict[str, list[str]] = {}
    for group in ("master_from_data", "master_from_modeling"):
        for entry in template.get("data_contract", {}).get(group, []):
            if entry.get("type") == "enum":
                enums[entry["key"]] = entry.get("allowed", [])

    # Collecter les activations par clé référencée
    covered: dict[str, set[str]] = {}
    for section in template.get("sections", []):
        act = section.get("activation")
        if act is None:
            continue
        if not isinstance(act, dict) or "key" not in act or "equals" not in act:
            errors.append(f"Section {section['id']} : activation doit être un dict {{key, equals}}")
            continue
        key = act["key"]
        if key not in enums:
            errors.append(f"Section {section['id']} : activation.key '{key}' absent des enums master_*")
            continue
        if act["equals"] not in enums[key]:
            errors.append(
                f"Section {section['id']} : activation.equals '{act['equals']}' absent de allowed={enums[key]}"
            )
            continue
        covered.setdefault(key, set()).add(act["equals"])

    for key, seen in covered.items():
        missing = set(enums[key]) - seen
        if missing:
            errors.append(
                f"Enum '{key}' : valeurs sans section activable {sorted(missing)}"
            )
```

- [ ] **Étape 4 — Vérifier**

Run: `pytest tests/test_validator.py -v && python scripts/check_template.py knowledge_base/report_template/mortality_template.yaml`
Expected: tests verts, check_template toujours vert (aucune `activation` dans le YAML actuel).

- [ ] **Étape 5 — Commit**

```bash
git add scripts/check_template.py tests/test_validator.py
git commit -m "feat(US-35): validator — champ activation (syntaxe + couverture enum)"
```

---

### US-36 — `template_loader.build_manifest` : filtre de sections inactives

**Files:**
- Modify: `knowledge_base/report_template/template_loader.py`
- Modify: `tests/test_template_loader.py`

- [ ] **Étape 1 — Tests rouges**

```python
# tests/test_template_loader.py (append)

def test_build_manifest_filters_inactive_sections(tmp_path):
    tpl = {
        "session_inputs": [],
        "data_contract": {
            "master_from_data": [],
            "master_from_modeling": [
                {"key": "gender_segmentation", "type": "enum", "allowed": ["unisex", "by_sex"],
                 "description": "", "produced_by": {"tool": "master.classify_request", "inputs": {}, "output_mapping": {}},
                 "confirm_with_user": True}
            ],
            "builder_outputs": [],
        },
        "sections": [
            {"id": "a", "label": "A", "required": True, "dependencies": [],
             "activation": {"key": "gender_segmentation", "equals": "unisex"},
             "narrative": {"text": ""},
             "llm_directives": {"tone": "", "length_words": [1, 2], "rag_query": ""},
             "visual_specs": []},
            {"id": "b", "label": "B", "required": True, "dependencies": [],
             "activation": {"key": "gender_segmentation", "equals": "by_sex"},
             "narrative": {"text": ""},
             "llm_directives": {"tone": "", "length_words": [1, 2], "rag_query": ""},
             "visual_specs": []},
        ],
    }
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(tpl))
    manifest = build_manifest(path, context={"gender_segmentation": "unisex"})
    ids = [s["id"] for s in manifest["sections"]]
    assert ids == ["a"]


def test_build_manifest_without_context_keeps_all_sections(tmp_path):
    # Backward compat : pas de contexte → toutes sections retenues.
    tpl = {...}  # identique à ci-dessus
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(tpl))
    manifest = build_manifest(path)
    ids = [s["id"] for s in manifest["sections"]]
    assert set(ids) == {"a", "b"}
```

- [ ] **Étape 2 — Vérifier l'échec**

Run: `pytest tests/test_template_loader.py -k activation -v`
Expected: 2 FAIL.

- [ ] **Étape 3 — Impl**

Modifier la signature de `build_manifest` dans `knowledge_base/report_template/template_loader.py` :

```python
def build_manifest(template_path: Path, context: dict | None = None) -> dict:
    """
    Construit le manifest ordonné des sections à rendre.
    Si `context` est fourni, filtre les sections dont l'activation n'est pas satisfaite.
    """
    template = yaml.safe_load(Path(template_path).read_text())
    sections = template.get("sections", [])

    if context is not None:
        sections = [s for s in sections if _is_active(s, context)]

    # (garder le tri par dependencies existant)
    ...
    return {"sections": sections, ...}


def _is_active(section: dict, context: dict) -> bool:
    act = section.get("activation")
    if act is None:
        return True
    return context.get(act["key"]) == act["equals"]
```

- [ ] **Étape 4 — Vérifier**

Run: `pytest tests/test_template_loader.py tests/test_pipeline_preamble_e2e.py -v`
Expected: tous PASS (régression préambule vert).

- [ ] **Étape 5 — Commit**

```bash
git add knowledge_base/report_template/template_loader.py tests/test_template_loader.py
git commit -m "feat(US-36): build_manifest — filtre sections inactives via context"
```

---

### US-37 — YAML : ajouter `gender_segmentation` + rebranchement sur `cleaned_records`

**Files:**
- Modify: `knowledge_base/report_template/mortality_template.yaml`

- [ ] **Étape 1 — Inspecter l'état actuel**

Run: `python scripts/check_template.py knowledge_base/report_template/mortality_template.yaml`
Expected: OK.

- [ ] **Étape 2 — Ajouter `gender_segmentation` dans `master_from_modeling` + `sans_objet` dans l'enum cause_sortie**

Dans `knowledge_base/report_template/mortality_template.yaml` :

1. Dans `session_inputs.input_records.shape`, étendre l'enum :
```yaml
  - {key: cause_sortie, type: enum, allowed: [deces, autre, sans_objet]}
```

2. Dans `data_contract.master_from_modeling`, **après** `study_objective`, ajouter :
```yaml
    - key: gender_segmentation
      type: enum
      allowed: [unisex, by_sex]
      description: "Table unisex agrégée vs tables séparées H/F."
      produced_by:
        tool: master.classify_request
        inputs: {request: raw_user_request}
        output_mapping: {gender_mode: gender_segmentation}
      confirm_with_user: true
```

3. Dans `data_contract.builder_outputs`, **tout en tête**, ajouter les 3 clés du preprocessing :
```yaml
    - key: cleaned_records
      type: table
      description: "Records après retraitement (R1–R6)."
      produced_by:
        tool: preprocessing.clean_records
        inputs: {}
        output_mapping: {cleaned_records: cleaned_records}

    - key: exclusion_report
      type: dict
      description: "Rapport des exclusions : initial_count, final_count, rules."
      produced_by:
        tool: preprocessing.clean_records
        inputs: {}
        output_mapping: {exclusion_report: exclusion_report}

    - key: total_records
      type: integer
      description: "Nombre de lignes post-retraitement (= exclusion_report.final_count)."
      produced_by:
        tool: preprocessing.clean_records
        inputs: {}
        output_mapping: {exclusion_report.final_count: total_records}
```

4. Modifier les `inputs` des builder tools existants qui sont actuellement `{}` pour pointer sur `cleaned_records` :
```yaml
    - key: total_exposure
      ...
      produced_by:
        tool: builder.exposure
        inputs: {records: cleaned_records}   # <- modifié
        output_mapping: {total_exposure: total_exposure}

    - key: total_deaths
      ...
      produced_by:
        tool: builder.exposure
        inputs: {records: cleaned_records}   # <- modifié
        output_mapping: {total_deaths: total_deaths}

    - key: segmentations
      ...
      produced_by:
        tool: statistical_analysis.segmentation
        inputs: {records: cleaned_records}   # <- modifié
        output_mapping: {segmentations: segmentations}

    - key: serie
      ...
      produced_by:
        tool: statistical_analysis.time_series
        inputs: {records: cleaned_records, by_sex: true}   # <- modifié
        output_mapping: {serie: serie}
```

5. Ajouter les clés `serie_h`, `serie_f`, `ages` :
```yaml
    - key: serie_h
      type: list[dict]
      description: "Série temporelle ventilée — hommes."
      produced_by:
        tool: statistical_analysis.time_series
        inputs: {records: cleaned_records, by_sex: true}
        output_mapping: {serie_h: serie_h}

    - key: serie_f
      type: list[dict]
      description: "Série temporelle ventilée — femmes."
      produced_by:
        tool: statistical_analysis.time_series
        inputs: {records: cleaned_records, by_sex: true}
        output_mapping: {serie_f: serie_f}

    - key: ages
      type: dict
      description: >
        Résultat complet age_distribution (age_min, age_max, distribution_list,
        distribution_list_h, distribution_list_f, …).
      produced_by:
        tool: statistical_analysis.age_distribution
        inputs: {records: cleaned_records, by_sex: true}
        output_mapping: {*: ages}
```

- [ ] **Étape 3 — Exécuter `check_template`**

Run: `python scripts/check_template.py knowledge_base/report_template/mortality_template.yaml`
Expected: OK (aucune couverture activation requise pour l'instant puisqu'aucune section n'utilise encore `activation`).

- [ ] **Étape 4 — Non-régression préambule**

Run: `pytest tests/test_pipeline_preamble_e2e.py -v`
Expected: PASS (les clés `total_exposure`, `total_deaths`, `segmentations`, `serie` sont toujours produites, juste avec un input explicite).

- [ ] **Étape 5 — Commit**

```bash
git add knowledge_base/report_template/mortality_template.yaml
git commit -m "feat(US-37): YAML — gender_segmentation, preprocessing outputs, rebranch cleaned_records"
```

---

### US-38 — YAML : section `data_preprocessing`

**Files:**
- Modify: `knowledge_base/report_template/mortality_template.yaml`

- [ ] **Étape 1 — Test rouge (contractuel)**

```python
# tests/test_template_contract.py (append)

def test_data_preprocessing_section_exists():
    import yaml
    tpl = yaml.safe_load(open("knowledge_base/report_template/mortality_template.yaml"))
    ids = [s["id"] for s in tpl["sections"]]
    assert "data_preprocessing" in ids


def test_data_preprocessing_has_exclusion_table():
    import yaml
    tpl = yaml.safe_load(open("knowledge_base/report_template/mortality_template.yaml"))
    section = next(s for s in tpl["sections"] if s["id"] == "data_preprocessing")
    vs_ids = [v["id"] for v in section["visual_specs"]]
    assert "exclusion_table" in vs_ids
```

Run: `pytest tests/test_template_contract.py -k "data_preprocessing" -v`
Expected: 2 FAIL.

- [ ] **Étape 2 — Ajouter la section dans le YAML**

Dans `knowledge_base/report_template/mortality_template.yaml`, dans la liste `sections`, après `preamble`, ajouter le YAML copié verbatim de [la spec §5.0](../specs/2026-04-21-data-analysis-design.md#50-data_preprocessing-nouvelle-section-toujours-active).

- [ ] **Étape 3 — Vérifier**

Run: `python scripts/check_template.py knowledge_base/report_template/mortality_template.yaml && pytest tests/test_template_contract.py -v`
Expected: OK + tous PASS.

- [ ] **Étape 4 — Commit**

```bash
git add knowledge_base/report_template/mortality_template.yaml tests/test_template_contract.py
git commit -m "feat(US-38): YAML — section data_preprocessing"
```

---

### US-39 — YAML : sections `data_analysis_unisex` + `data_analysis_by_sex`

**Files:**
- Modify: `knowledge_base/report_template/mortality_template.yaml`

- [ ] **Étape 1 — Tests rouges**

```python
# tests/test_template_contract.py (append)

def test_data_analysis_unisex_and_by_sex_exist():
    import yaml
    tpl = yaml.safe_load(open("knowledge_base/report_template/mortality_template.yaml"))
    ids = [s["id"] for s in tpl["sections"]]
    assert "data_analysis_unisex" in ids
    assert "data_analysis_by_sex" in ids


def test_data_analysis_sections_have_activation():
    import yaml
    tpl = yaml.safe_load(open("knowledge_base/report_template/mortality_template.yaml"))
    for sid in ("data_analysis_unisex", "data_analysis_by_sex"):
        section = next(s for s in tpl["sections"] if s["id"] == sid)
        assert section["activation"]["key"] == "gender_segmentation"
```

Run: `pytest tests/test_template_contract.py -k "data_analysis" -v`
Expected: 2 FAIL.

- [ ] **Étape 2 — Ajouter les deux sections dans le YAML**

Dans `sections`, après `data_preprocessing`, ajouter les deux sections verbatim depuis [la spec §5.1 et §5.2](../specs/2026-04-21-data-analysis-design.md#51-data_analysis_unisex).

- [ ] **Étape 3 — Vérifier**

Run: `python scripts/check_template.py knowledge_base/report_template/mortality_template.yaml && pytest tests/test_template_contract.py -v`
Expected: check_template OK (couverture enum `gender_segmentation` complète : unisex + by_sex) + tests PASS.

- [ ] **Étape 4 — Commit**

```bash
git add knowledge_base/report_template/mortality_template.yaml tests/test_template_contract.py
git commit -m "feat(US-39): YAML — sections data_analysis_{unisex,by_sex}"
```

---

### US-40 — E2E rendering : mode `unisex` (data_store fictif + build_manifest filtré)

**Files:**
- Create: `tests/test_data_analysis_e2e.py`

Suit le pattern de `tests/test_pipeline_preamble_e2e.py` : on construit un `data_store` inline avec toutes les clés que les nouvelles sections consomment, on appelle `build_manifest(..., context=...)` puis on vérifie que les sections actives sont bien rendues.

- [ ] **Étape 1 — Test rouge**

```python
# tests/test_data_analysis_e2e.py
from pathlib import Path
from knowledge_base.report_template.template_loader import build_manifest

TEMPLATE = Path("knowledge_base/report_template/mortality_template.yaml")


def _data_store_unisex() -> dict:
    """data_store minimal couvrant preamble + data_preprocessing + data_analysis_unisex."""
    return {
        "study_objective":        "construction_table_mortalite",
        "gender_segmentation":    "unisex",
        "start_year":             2019,
        "end_year":               2021,
        "num_observation_years":  3,
        "total_records":          900,
        "total_exposure":         2700.0,
        "total_deaths":           42,
        "exclusion_report": {
            "initial_count": 1000,
            "final_count":   900,
            "rules": [
                {"rule_id": f"R{i}", "rule_label": f"Règle {i}", "count": 0, "detail": {}}
                for i in range(1, 7)
            ],
        },
        "segmentations": {"sexe": [
            {"valeur": "H", "nb_contrats": 500, "nb_deces": 25, "pct_contrats": 55.6, "pct_deces": 59.5},
            {"valeur": "F", "nb_contrats": 400, "nb_deces": 17, "pct_contrats": 44.4, "pct_deces": 40.5},
        ]},
        "serie": [
            {"annee": 2019, "nb_entres": 300, "nb_deces": 10, "exposition_pa": 900.0,
             "age_moyen_entres": 45.1, "age_moyen_deces": 62.3, "taux_deces": 11.11},
        ],
        "ages": {
            "age_min": 30, "age_max": 85, "age_moyen": 47.5,
            "distribution_list": [{"tranche": "30-34", "nb_contrats": 50},
                                  {"tranche": "35-39", "nb_contrats": 120}],
        },
    }


def test_unisex_manifest_activates_correct_sections():
    context = {"gender_segmentation": "unisex"}
    manifest = build_manifest(TEMPLATE, context=context)
    ids = [s["id"] for s in manifest["sections"]]
    assert "preamble" in ids
    assert "data_preprocessing" in ids
    assert "data_analysis_unisex" in ids
    assert "data_analysis_by_sex" not in ids


def test_unisex_preprocessing_renders_exclusion_table():
    from agents.report.pipeline._01_load_plan import load_plan
    from agents.report.pipeline._04_redaction import _run_tables
    context = {"gender_segmentation": "unisex"}
    plan = load_plan(_data_store_unisex(), context=context)
    preprocessing = next(s for s in plan.sections if s.section_id == "data_preprocessing")
    tables = _run_tables(preprocessing, _data_store_unisex())
    assert len(tables) == 1
    assert len(tables[0]["rows"]) == 7  # 1 header + 6 rules
```

Run: `pytest tests/test_data_analysis_e2e.py -v`
Expected: FAIL (pas encore implémenté ; selon si `load_plan` supporte déjà `context=`, FAIL différents).

- [ ] **Étape 2 — Adapter `load_plan` pour accepter un context**

Si `load_plan(data_store)` n'accepte pas encore `context`, l'ajouter pour propager à `build_manifest`. Modifier la signature en gardant la rétro-compatibilité :

```python
# agents/report/pipeline/_01_load_plan.py
def load_plan(data_store: dict, context: dict | None = None) -> Plan:
    # extraire context depuis data_store si non fourni explicitement :
    if context is None:
        context = {k: data_store[k] for k in ("gender_segmentation", "study_objective") if k in data_store}
    manifest = build_manifest(TEMPLATE_PATH, context=context)
    ...
```

- [ ] **Étape 3 — Vérifier**

Run: `pytest tests/test_data_analysis_e2e.py tests/test_pipeline_preamble_e2e.py -v`
Expected: tous PASS (non-régression préambule + nouveaux tests verts).

- [ ] **Étape 4 — Commit**

```bash
git add tests/test_data_analysis_e2e.py agents/report/pipeline/_01_load_plan.py
git commit -m "test(US-40): E2E data_analysis mode unisex (manifest + exclusion table)"
```

---

### US-41 — E2E rendering : mode `by_sex`

**Files:**
- Modify: `tests/test_data_analysis_e2e.py`

- [ ] **Étape 1 — Test rouge**

```python
def _data_store_by_sex() -> dict:
    base = _data_store_unisex()
    base["gender_segmentation"] = "by_sex"
    base["serie_h"] = [{"annee": 2019, "nb_entres": 180, "nb_deces": 6, "exposition_pa": 540.0,
                        "age_moyen_entres": 45.1, "age_moyen_deces": 62.3, "taux_deces": 11.11}]
    base["serie_f"] = [{"annee": 2019, "nb_entres": 120, "nb_deces": 4, "exposition_pa": 360.0,
                        "age_moyen_entres": 43.2, "age_moyen_deces": 64.1, "taux_deces": 11.11}]
    base["ages"]["distribution_list_h"] = [{"tranche": "30-34", "nb_contrats": 30}]
    base["ages"]["distribution_list_f"] = [{"tranche": "30-34", "nb_contrats": 20}]
    return base


def test_by_sex_manifest_activates_correct_sections():
    context = {"gender_segmentation": "by_sex"}
    manifest = build_manifest(TEMPLATE, context=context)
    ids = [s["id"] for s in manifest["sections"]]
    assert "data_analysis_by_sex" in ids
    assert "data_analysis_unisex" not in ids


def test_by_sex_section_has_four_visuals():
    from agents.report.pipeline._01_load_plan import load_plan
    context = {"gender_segmentation": "by_sex"}
    plan = load_plan(_data_store_by_sex(), context=context)
    by_sex_section = next(s for s in plan.sections if s.section_id == "data_analysis_by_sex")
    vs_ids = [v.spec_id for v in by_sex_section.visual_specs]
    assert vs_ids == ["annual_statistics_male", "annual_statistics_female",
                      "exposure_distribution_male", "exposure_distribution_female"]
```

Run: `pytest tests/test_data_analysis_e2e.py -k by_sex -v`
Expected: 2 PASS.

- [ ] **Étape 2 — Full pytest de non-régression**

Run: `pytest tests/ -v`
Expected: tous verts.

- [ ] **Étape 3 — Commit**

```bash
git add tests/test_data_analysis_e2e.py
git commit -m "test(US-41): E2E data_analysis mode by_sex (4 visuals H+F)"
```

---

## Done global

- [ ] `pytest tests/` : tous verts (y compris non-régression US-26 preamble).
- [ ] `python scripts/check_template.py` : vert.
- [ ] Les 3 sections (`data_preprocessing`, `data_analysis_unisex`, `data_analysis_by_sex`) sont rendues correctement selon `gender_segmentation`.
- [ ] Mettre à jour `memory/project_refactor_report_agent.md` pour marquer US-27..41 terminées.
- [ ] Ouvrir le ticket refacto `mortality.describe()` (cf. `memory/project_refactor_mortality_describe.md`) une fois ce plan vert.
