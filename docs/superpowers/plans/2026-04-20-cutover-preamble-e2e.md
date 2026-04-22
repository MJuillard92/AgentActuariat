# Cutover Preamble E2E — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rendre le pipeline `Master → Builder → Writer → PDF` fonctionnel de bout en bout sur la seule section `preamble` du YAML Design 3. À la fin, un utilisateur peut charger un CSV, lancer le pipeline, et obtenir un PDF contenant uniquement le préambule avec placeholders résolus, 1 tableau et 1 graphe.

**Architecture:** Le YAML actuel est déjà en Design 3 (`data_contract` + section preamble isolée). Le Master et le Builder continuent d'utiliser l'ancien vocabulaire (`_ALL_BUILDER_KEYS` hardcodé = `exposure_table, qx_table, ...`). Le Writer lit le YAML via `tools/build_pdf/load_yaml_template.py` qui parse la structure v1 (`processing_sequence, subsections, content: [{purpose, word_count}]`), incompatible avec Design 3. Ce plan réaligne les trois couches sur Design 3 via `template_loader` (déjà livré en US-6/7) et les nouveaux tools `mortality/compute_*` (livrés hors-US en préparation de ce cutover).

**Tech Stack:** Python 3.11, pytest, pandas, LangGraph, `template_loader` (US-6/7), tools/mortality/compute_* (préamble).

**Bundle = 6 User Stories** du plan principal `docs/superpowers/plans/2026-04-20-refactor-yaml-master-builder-writer.md` :
US-15, US-20, US-22, US-23, US-24, US-25. Elles doivent impérativement être commitées ensemble pour que le pipeline reste fonctionnel en bout de chaîne.

**Hors scope de ce plan** : US-17/18/19 (cinématique Master étendue), US-26 (test E2E formel), UI frontend (événements déjà émis, front non modifié).

---

## File Structure

**Modifications** :
- [agents/mortality/agents/master_node.py](agents/mortality/agents/master_node.py) — supprimer `_ALL_BUILDER_KEYS` et `_MINIMUM_BUILDER_KEYS`, consommer `build_manifest()`, dériver les clés dynamiquement.
- [agents/mortality/agents/builder_node.py](agents/mortality/agents/builder_node.py) — ajouter une branche déterministe "exécute le DAG du manifest" qui remplace l'appel LLM quand `active_agent=builder` avec `study_plan` complet.
- [agents/report/pipeline/_01_load_plan.py](agents/report/pipeline/_01_load_plan.py) — réécrire `load_plan()` sur `template_loader.load_section()` + `resolve_placeholders()` au lieu de `tools/build_pdf/load_yaml_template`.
- [agents/report/pipeline/_03_completion_plan.py](agents/report/pipeline/_03_completion_plan.py) — supprimer `_SECTION_QUERIES`, lire `llm_directives.rag_query` via loader.
- [agents/report/pipeline/_04_redaction.py](agents/report/pipeline/_04_redaction.py) — simplifier `_hydrate_table_spec` : mapping direct `column.key → data_store[source][column.key]`. Adapter `_run_tables`/`_run_graphs` aux visual_specs Design 3.
- [agents/report/pipeline/run_pipeline.py](agents/report/pipeline/run_pipeline.py) — supprimer étape `_02_validation_plan` (3 étapes au lieu de 4).
- [tools/build_pdf/load_yaml_template.py](tools/build_pdf/load_yaml_template.py) — retirer `_SECTION_REQUIRED` (laisser le reste du fichier en place pour l'instant, il n'est plus appelé).

**Suppressions** :
- [agents/report/pipeline/_02_validation_plan.py](agents/report/pipeline/_02_validation_plan.py) — fichier entier.

**Créations (tests)** :
- `tests/test_master_node_manifest.py` — US-15 unitaire.
- `tests/test_builder_node_dag.py` — US-20 unitaire (exécution déterministe du DAG preamble).
- `tests/test_load_plan_design3.py` — US-22 unitaire sur le nouveau `load_plan`.
- `tests/test_redaction_preamble.py` — US-24 intégration `_04_redaction` sur preamble.
- `tests/test_pipeline_preamble_e2e.py` — orchestration des 3 étapes (load → completion → redaction) sur fixture data_store preamble.

---

## Prerequisites Check

**À vérifier avant Task 1** :

- [ ] **Step 0.1: Confirmer que les tools preamble existent**

Run: `ls tools/mortality/compute_exposure.py tools/mortality/compute_deaths.py tools/mortality/compute_composition.py tools/mortality/compute_deaths_timeseries.py`
Expected: 4 fichiers listés, aucune erreur.

- [ ] **Step 0.2: Confirmer que la suite de tests est verte au départ**

Run: `python -m pytest tests/ && python scripts/check_template.py`
Expected: 98 passed, `✓ template valide`.

- [ ] **Step 0.3: Créer une branche dédiée**

```bash
git checkout -b feat/cutover-preamble-e2e
```

---

## Task 1 (US-15): Master consomme `build_manifest()`

**Contexte** : Le Master utilise aujourd'hui `_ALL_BUILDER_KEYS` (6 clés legacy : `exposure_table, qx_table, smoothed_table, diagnostics, validation, benchmarking`) et `_MINIMUM_BUILDER_KEYS = ["exposure_table", "smoothed_table"]`. Ces deux constantes doivent disparaître au profit d'une dérivation dynamique depuis `build_manifest()`. Les nouvelles clés sont `total_exposure_years, total_deaths, portfolio_composition_by_sex, deaths_by_year_series`.

**Files:**
- Modify: `agents/mortality/agents/master_node.py:53-62` (constantes) puis toutes leurs utilisations.
- Create: `tests/test_master_node_manifest.py`

- [ ] **Step 1.1: Écrire le test `test_master_builder_keys_from_manifest`**

```python
# tests/test_master_node_manifest.py
"""Tests US-15 : master lit les clés Builder depuis build_manifest()."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_get_builder_keys_from_manifest_returns_preamble_keys():
    """La fonction helper doit renvoyer les 4 clés builder_outputs du preamble."""
    from agents.mortality.agents.master_node import _get_builder_keys

    keys = _get_builder_keys()

    assert set(keys) == {
        "total_exposure_years",
        "total_deaths",
        "portfolio_composition_by_sex",
        "deaths_by_year_series",
    }


def test_preflight_writer_ready_when_all_manifest_keys_present():
    from agents.mortality.agents.master_node import _preflight_writer

    data_store = {
        "total_exposure_years":          1234.5,
        "total_deaths":                  42,
        "portfolio_composition_by_sex":  [{"sexe": "H"}],
        "deaths_by_year_series":         [{"year": 2020, "deaths": 10}],
    }

    ready, missing = _preflight_writer(data_store)

    assert ready is True
    assert missing == []


def test_preflight_writer_missing_keys():
    from agents.mortality.agents.master_node import _preflight_writer

    data_store = {"total_exposure_years": 100}
    ready, missing = _preflight_writer(data_store)

    assert ready is False
    assert len(missing) == 3
```

- [ ] **Step 1.2: Lancer le test (doit échouer sur ImportError `_get_builder_keys`)**

Run: `python -m pytest tests/test_master_node_manifest.py -x`
Expected: ImportError ou AttributeError sur `_get_builder_keys`.

- [ ] **Step 1.3: Implémenter `_get_builder_keys` + refactoriser**

Remplacer [agents/mortality/agents/master_node.py:53-62](agents/mortality/agents/master_node.py#L53-L62) :

```python
# ── Clés Builder attendues (dérivées du YAML Design 3) ──────────────────────

def _get_builder_keys() -> list[str]:
    """Retourne la liste des clés `builder_outputs` du manifest YAML."""
    from knowledge_base.report_template.template_loader import build_manifest
    manifest = build_manifest()
    return [k.key for k in manifest.builder_outputs]
```

Puis remplacer toutes les occurrences :
- `_ALL_BUILDER_KEYS` (dict clé→label) → `_get_builder_keys()` (list)
- `_MINIMUM_BUILDER_KEYS` → `_get_builder_keys()` (les 4 clés deviennent toutes "minimales" en mode preamble)
- `for key, label in _ALL_BUILDER_KEYS.items()` → dérivation sans label (ou label = key)

Points précis à éditer (chercher chaque occurrence dans le fichier) :
- ligne ~76 `has_all = all(data_store.get(k) for k in _ALL_BUILDER_KEYS)` → `has_all = all(data_store.get(k) for k in _get_builder_keys())`
- ligne ~117 `_preflight_writer` : boucler sur `_get_builder_keys()`, retourner la key comme label
- ligne ~292 `all(data_store.get(k) for k in _MINIMUM_BUILDER_KEYS)` → `_get_builder_keys()`
- ligne ~324 `already_done = [k for k in _ALL_BUILDER_KEYS if ...]` → `[k for k in _get_builder_keys() if ...]`
- ligne ~414 `missing_min = [k for k in _MINIMUM_BUILDER_KEYS ...]` → `_get_builder_keys()`
- ligne ~417 `already_done = [k for k in _ALL_BUILDER_KEYS ...]` → `_get_builder_keys()`
- ligne ~454 idem
- lignes `key_labels = {...}` dans `_augment_with_data_store` (ligne ~196) : elles sont cosmétiques, laisser les anciennes clés supprimées de la liste mais ne pas casser. Supprimer les clés `exposure_table, qx_table, smoothed_table, diagnostics, validation, benchmarking` du dict `key_labels` et ajouter les 4 nouvelles avec labels FR.

- [ ] **Step 1.4: Lancer les tests US-15 (doivent passer)**

Run: `python -m pytest tests/test_master_node_manifest.py -x -v`
Expected: 3 passed.

- [ ] **Step 1.5: Lancer la suite complète (garde-fou régression)**

Run: `python -m pytest tests/ && python scripts/check_template.py`
Expected: 101 passed, template valide.

- [ ] **Step 1.6: Commit US-15**

```bash
git add agents/mortality/agents/master_node.py tests/test_master_node_manifest.py
git commit -m "feat(US-15): master_node dérive les clés builder depuis build_manifest()

Supprime _ALL_BUILDER_KEYS et _MINIMUM_BUILDER_KEYS hardcodés. Les
4 clés du preamble (total_exposure_years, total_deaths,
portfolio_composition_by_sex, deaths_by_year_series) sont lues
dynamiquement via template_loader.build_manifest()."
```

---

## Task 2 (US-20): Builder exécute le DAG du manifest

**Contexte** : Aujourd'hui `builder_node` appelle uniquement un LLM avec tools (`BUILDER_TOOLS`). En mode preamble, on veut une **branche déterministe** : quand Master envoie `GO_BUILD`, le Builder exécute le DAG du manifest (4 tools mortality) sans LLM, stocke les sorties dans `data_store`, émet `<BUILD_DONE>`. Le mode LLM reste pour les sections qui arriveront plus tard (hors scope preamble).

**Décision** : on ajoute une nouvelle fonction `_execute_manifest_dag(data_store, dataset_ref) -> dict` qui tourne systématiquement au début du `builder_node`, en amont de l'appel LLM. Si toutes les clés `builder_outputs` sont remplies après exécution, on émet `<BUILD_DONE>` et on retourne sans appeler le LLM. Sinon on laisse tomber en mode LLM (legacy, permet de rester fonctionnel pour les sections futures).

**Files:**
- Modify: `agents/mortality/agents/builder_node.py` (ajouter helper + branche déterministe).
- Create: `tests/test_builder_node_dag.py`

- [ ] **Step 2.1: Écrire le test du DAG deterministic**

```python
# tests/test_builder_node_dag.py
"""Tests US-20 : builder exécute le DAG du manifest preamble."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _make_records():
    return pd.DataFrame({
        "date_naissance": ["1960-01-01", "1965-06-15", "1970-03-20"],
        "date_entree":    ["2019-01-01", "2019-01-01", "2019-01-01"],
        "date_sortie":    ["2020-06-01", "2021-12-31", "2021-07-15"],
        "cause_sortie":   ["deces",      "autre",      "deces"],
        "sexe":           ["H",          "F",          "H"],
    })


def test_execute_manifest_dag_fills_all_builder_outputs():
    from agents.mortality.agents.builder_node import _execute_manifest_dag

    data_store = {
        "input_records":       _make_records(),
        "raw_user_request":    "construis-moi une table de mortalité",
    }

    updates = _execute_manifest_dag(data_store)

    assert updates is not None
    for key in (
        "total_exposure_years",
        "total_deaths",
        "portfolio_composition_by_sex",
        "deaths_by_year_series",
        "observation_period_years",
        "start_year",
        "end_year",
        "num_observation_years",
        "study_objective",
    ):
        assert key in updates, f"clé manquante: {key}"


def test_execute_manifest_dag_noop_without_records():
    from agents.mortality.agents.builder_node import _execute_manifest_dag

    updates = _execute_manifest_dag({"input_records": None})

    assert updates is None
```

- [ ] **Step 2.2: Lancer le test (ImportError attendue)**

Run: `python -m pytest tests/test_builder_node_dag.py -x`
Expected: ImportError sur `_execute_manifest_dag`.

- [ ] **Step 2.3: Implémenter `_execute_manifest_dag`**

Ajouter en haut de [agents/mortality/agents/builder_node.py](agents/mortality/agents/builder_node.py), juste avant `def builder_node(...)` :

```python
def _execute_manifest_dag(data_store: dict) -> dict | None:
    """Exécute en une passe le DAG `builder_outputs` du manifest YAML.

    Retourne les updates à appliquer au data_store, ou None si pas de
    records exploitables.

    Chaque entrée du DAG est un appel direct au tool nommé dans `produced_by.tool`
    avec les inputs résolus depuis le data_store. L'output_mapping renomme les
    sorties vers les clés canoniques du data_contract.
    """
    if not isinstance(data_store.get("input_records"), (pd.DataFrame, list, dict)):
        return None

    from knowledge_base.report_template.template_loader import build_manifest
    import importlib

    manifest = build_manifest()
    produced: dict = dict(data_store)  # copie pour résolution d'inputs au fil de l'eau
    updates: dict = {}

    for call in manifest.dag:
        tool_name = call["tool"]          # ex: "master.analyze_data_and_request"
        inputs_spec = call["inputs"]      # {local_name: data_store_key_or_literal}
        output_mapping = call["output_mapping"]  # {tool_output: canonical_key}

        # Résolution inputs
        tool_inputs: dict = {}
        for local_name, ref in inputs_spec.items():
            if isinstance(ref, str) and ref in produced:
                tool_inputs[local_name] = produced[ref]
            else:
                tool_inputs[local_name] = ref  # littéral (list, dict, scalar)

        # Import dynamique du module tool
        module_path = "tools." + tool_name.replace(".", ".")
        try:
            mod = importlib.import_module(module_path)
            result = mod.run(tool_inputs, {})
        except Exception as exc:
            print(f"[BuilderAgent] tool {tool_name} a échoué: {exc}", file=sys.stderr)
            continue

        # Application de l'output_mapping
        for tool_out, canonical in output_mapping.items():
            if tool_out in result:
                produced[canonical] = result[tool_out]
                updates[canonical] = result[tool_out]

    return updates or None
```

- [ ] **Step 2.4: Lancer les tests DAG**

Run: `python -m pytest tests/test_builder_node_dag.py -x -v`
Expected: 2 passed.

- [ ] **Step 2.5: Brancher la branche déterministe dans `builder_node`**

Modifier [agents/mortality/agents/builder_node.py:124-148](agents/mortality/agents/builder_node.py#L124-L148), au tout début de `builder_node(state)` (juste après `import openai`) :

```python
def builder_node(state: "AgentState") -> dict:
    """..."""
    data_store = state.get("data_store") or {}

    # ── Branche déterministe : exécute le DAG du manifest (US-20) ────────────
    # Si les 4 clés builder_outputs sont remplies après exécution,
    # on saute l'appel LLM et on émet <BUILD_DONE>.
    dag_updates = _execute_manifest_dag(data_store)
    if dag_updates:
        data_store.update(dag_updates)
        from knowledge_base.report_template.template_loader import build_manifest
        needed = [k.key for k in build_manifest().builder_outputs]
        if all(data_store.get(k) is not None for k in needed):
            from langchain_core.messages import AIMessage
            return {
                "messages":     [AIMessage(content="Calculs preamble terminés. <BUILD_DONE>")],
                "events":       [{"type": "agent_switch", "agent": "BuilderAgent"},
                                 {"type": "message", "content": "Preamble calculé (4 clés). <BUILD_DONE>"}],
                "active_agent": "master",
                "data_store":   data_store,
                "plan_established": True,
            }
    # ── Fallback LLM (legacy — sections futures hors preamble) ────────────────
    import openai
    # ... (reste du corps actuel inchangé)
```

**Important** : ne pas dupliquer la ligne `data_store = state.get("data_store") or {}` — la déplacer en tête de la fonction si nécessaire.

- [ ] **Step 2.6: Lancer la suite complète**

Run: `python -m pytest tests/ && python scripts/check_template.py`
Expected: 103 passed, template valide.

- [ ] **Step 2.7: Commit US-20**

```bash
git add agents/mortality/agents/builder_node.py tests/test_builder_node_dag.py
git commit -m "feat(US-20): builder_node exécute le DAG preamble en une passe

Branche déterministe en tête de builder_node : _execute_manifest_dag
appelle les 4 tools mortality.compute_* (+ master tools pour
observation_*, study_objective) via le manifest build_manifest().
Si toutes les builder_outputs sont remplies, <BUILD_DONE> est
émis directement sans appel LLM."
```

---

## Task 3 (US-25 partiel) : Supprimer `_SECTION_REQUIRED` dans load_yaml_template

**Contexte** : Le dict `_SECTION_REQUIRED` dans [tools/build_pdf/load_yaml_template.py:144-154](tools/build_pdf/load_yaml_template.py#L144) référence les clés legacy (`exposure_table`, `qx_table`, etc.). Il est utilisé ligne ~493 pour marquer les sections comme "not ready" si ces clés manquent. En mode preamble Design 3, les clés produites sont différentes ; le dict bloque alors la section preamble comme "non prête". On le supprime avant de toucher au reste pour éviter un effet de bord.

**Files:**
- Modify: `tools/build_pdf/load_yaml_template.py:144-154` et `tools/build_pdf/load_yaml_template.py:493`.

- [ ] **Step 3.1: Inspecter le fichier**

Run: `sed -n '140,160p' tools/build_pdf/load_yaml_template.py && echo '---' && sed -n '488,500p' tools/build_pdf/load_yaml_template.py`
(ou équivalent Read). Repérer précisément la définition et l'utilisation.

- [ ] **Step 3.2: Supprimer le dict et son utilisation**

Remplacer le dict complet par un commentaire :

```python
# _SECTION_REQUIRED supprimé (US-25) : le statut "ready" d'une section
# est désormais dérivé des placeholders effectivement résolus via le
# template_loader (Design 3).
```

À l'endroit d'utilisation (ligne ~493), remplacer `required_keys = _SECTION_REQUIRED.get(str(sec_id), [])` par `required_keys = []` (les sections sont considérées prêtes si les placeholders sont résolus, logique déléguée à `_01_load_plan` v2).

- [ ] **Step 3.3: Lancer la suite**

Run: `python -m pytest tests/ && python scripts/check_template.py`
Expected: 103 passed (aucun test ne couvre `_SECTION_REQUIRED` directement).

- [ ] **Step 3.4: Commit US-25 partiel**

```bash
git add tools/build_pdf/load_yaml_template.py
git commit -m "refactor(US-25): supprime _SECTION_REQUIRED hardcodé

Préparation du cutover preamble Design 3 : les clés legacy
(exposure_table, qx_table, ...) ne sont plus produites par le
Builder, le dict bloquait artificiellement les sections.
Le ready sera dérivé des placeholders par template_loader."
```

---

## Task 4 (US-22): Réécrire `_01_load_plan.py` sur Design 3

**Contexte** : C'est le morceau le plus gros. L'actuel `load_plan()` utilise `tools/build_pdf/load_yaml_template.run()` (parser v1) et construit des `SectionPlan` avec `table_specs`, `graph_specs`, `stat_specs` dérivés de la structure v1. On réécrit sur `template_loader.load_section()` + `resolve_placeholders()` (Design 3).

**Nouveau contrat** :
- `SectionPlan.visual_specs: list[dict]` remplace `table_specs`/`graph_specs`/`stat_specs` (conforme Design 3).
- `SectionPlan.prompt` : assemble via narrative + llm_directives de la section.
- `SectionPlan.context_snapshot: dict` : placeholders résolus (via `resolve_placeholders`).
- Plus de notion de "subsections" (abandonné en Design 3).

**Files:**
- Modify: `agents/report/pipeline/_01_load_plan.py` (réécriture quasi-complète — 446 lignes → ~130).
- Create: `tests/test_load_plan_design3.py`.

- [ ] **Step 4.1: Écrire les tests du nouveau load_plan**

```python
# tests/test_load_plan_design3.py
"""Tests US-22 : load_plan v2 lit Design 3 via template_loader."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.report.pipeline._01_load_plan import load_plan, SectionPlan, ReportPlan  # noqa: E402


def _preamble_data_store():
    return {
        "study_objective":                "construction_table_mortalite",
        "start_year":                     2019,
        "end_year":                       2021,
        "num_observation_years":          3,
        "total_exposure_years":           1234.5,
        "total_deaths":                   42,
        "portfolio_composition_by_sex":   [
            {"sexe": "H", "n_lives": 500, "exposure": 700.0, "deaths": 25},
            {"sexe": "F", "n_lives": 500, "exposure": 534.5, "deaths": 17},
        ],
        "deaths_by_year_series":          [
            {"year": 2019, "deaths": 10},
            {"year": 2020, "deaths": 15},
            {"year": 2021, "deaths": 17},
        ],
    }


def test_load_plan_returns_one_section_for_preamble_yaml():
    plan = load_plan(_preamble_data_store())
    assert isinstance(plan, ReportPlan)
    assert len(plan.sections) == 1
    assert plan.sections[0].section_id == "preamble"


def test_section_plan_has_resolved_narrative():
    plan = load_plan(_preamble_data_store())
    preamble = plan.sections[0]

    # Les placeholders doivent être résolus dans le prompt
    assert "{{ study_objective }}" not in preamble.prompt
    assert "construction_table_mortalite" in preamble.prompt
    assert "2019" in preamble.prompt
    assert "2021" in preamble.prompt


def test_section_plan_visual_specs_pass_through():
    plan = load_plan(_preamble_data_store())
    preamble = plan.sections[0]
    ids = [v["id"] for v in preamble.visual_specs]
    assert "portfolio_composition" in ids
    assert "deaths_per_year" in ids


def test_section_plan_ready_when_all_placeholders_resolvable():
    plan = load_plan(_preamble_data_store())
    assert plan.sections[0].ready is True
    assert plan.missing_fields == []


def test_section_plan_not_ready_on_missing_placeholder():
    ds = _preamble_data_store()
    del ds["total_deaths"]
    plan = load_plan(ds)
    assert plan.sections[0].ready is False
    assert "total_deaths" in plan.missing_fields
```

- [ ] **Step 4.2: Vérifier les tests échouent**

Run: `python -m pytest tests/test_load_plan_design3.py -x`
Expected: au moins un test échoue (import ou assertion).

- [ ] **Step 4.3: Réécrire `_01_load_plan.py`**

Remplacer intégralement le fichier par :

```python
"""
agents/report/pipeline/_01_load_plan.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPE 1 — Déterministe, zéro LLM (Design 3)

Lit le YAML via template_loader, résout les placeholders depuis le
data_store, assemble un prompt de rédaction par section.

Interface publique :
    load_plan(data_store, study_plan=None, yaml_path=None) -> ReportPlan
    ReportPlan, SectionPlan (dataclasses)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SectionPlan:
    section_id:       str
    label:            str
    ready:            bool
    missing_inputs:   list[str]
    prompt:           str
    visual_specs:     list[dict]       # passé tel quel aux étapes aval
    context_snapshot: dict


@dataclass
class ReportPlan:
    sections:       list[SectionPlan]
    context:        dict
    missing_fields: list[str]
    n_ready:        int
    n_total:        int
    yaml_path:      str


_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def _extract_placeholder_keys(text: str) -> list[str]:
    return list(dict.fromkeys(_PLACEHOLDER_RE.findall(text or "")))


def _resolve_or_placeholder(text: str, context: dict) -> tuple[str, list[str]]:
    """Substitue les placeholders ; renvoie le texte résolu + la liste des
    clés manquantes. Les clés manquantes sont remplacées par '—' pour garder
    le prompt lisible mais sont remontées dans missing_inputs."""
    missing: list[str] = []

    def _sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in context or context[key] in (None, "", []):
            missing.append(key)
            return "—"
        val = context[key]
        if isinstance(val, float):
            return f"{round(val, 4):g}"
        if isinstance(val, (list, dict)):
            return str(val)[:80]
        return str(val)

    return _PLACEHOLDER_RE.sub(_sub, text or ""), missing


def _build_prompt(section, context: dict, missing_in_narrative: list[str]) -> str:
    """Assemble le prompt de rédaction d'une section Design 3."""
    narrative_text, _ = _resolve_or_placeholder(
        (section.narrative.get("text") or ""), context
    )
    directives = section.llm_directives or {}
    tone = directives.get("tone", "neutre, descriptif")
    length = directives.get("length_words", [])
    length_str = f"{length[0]}-{length[1]} mots" if isinstance(length, list) and len(length) == 2 else ""

    lines = [
        f"# Section {section.label} — Instructions de rédaction",
        "",
        "## Rôle",
        f"Tu rédiges la section '{section.label}' d'un rapport actuariel.",
        "Tu cites UNIQUEMENT des valeurs présentes dans les données ci-dessous.",
        "",
        f"## Ton attendu : {tone}",
    ]
    if length_str:
        lines.append(f"## Longueur cible : {length_str}")
    lines += [
        "",
        "## Narrative de référence (placeholders résolus)",
        narrative_text,
        "",
    ]

    visuals = section.visual_specs or []
    if visuals:
        lines += ["## Visuels à produire (détails dans SectionPlan.visual_specs)"]
        for v in visuals:
            lines.append(f"- `{v.get('id', '?')}` ({v.get('type', '?')}) — {v.get('purpose', '')}")
        lines.append("")

    lines += [
        "## Règles",
        "- Ne cite QUE des chiffres présents dans la narrative ou les visual_specs",
        "- Ne dépasse pas 10% au-delà de la longueur cible",
        "- Français, style professionnel actuariel",
    ]
    return "\n".join(lines)


def load_plan(
    data_store: dict,
    study_plan: dict | None = None,
    yaml_path:  str | Path | None = None,
) -> ReportPlan:
    """Charge le YAML Design 3 et assemble un ReportPlan."""
    from knowledge_base.report_template.template_loader import (
        DEFAULT_TEMPLATE, build_manifest, load_section,
    )
    import yaml as _yaml

    yaml_path = Path(yaml_path) if yaml_path else DEFAULT_TEMPLATE

    # Contexte = data_store + study_plan (fusion)
    context: dict = dict(data_store or {})
    if study_plan:
        context.update(study_plan)

    # Énumérer les sections actives dans le YAML
    with open(yaml_path, encoding="utf-8") as f:
        tpl = _yaml.safe_load(f) or {}
    active_section_ids = [s["id"] for s in (tpl.get("sections") or []) if "id" in s]

    sections: list[SectionPlan] = []
    missing_fields_global: set[str] = set()

    for sid in active_section_ids:
        sec = load_section(sid, yaml_path)
        narrative_text = sec.narrative.get("text") or ""
        _, missing_narrative = _resolve_or_placeholder(narrative_text, context)

        # Collecter les clés des visual_specs (source, columns[].key)
        for v in sec.visual_specs or []:
            source = v.get("source")
            if source and source not in context:
                missing_narrative.append(source)

        missing_fields_global.update(missing_narrative)
        prompt = _build_prompt(sec, context, missing_narrative)

        sections.append(SectionPlan(
            section_id       = sec.id,
            label            = sec.label,
            ready            = len(missing_narrative) == 0,
            missing_inputs   = list(dict.fromkeys(missing_narrative)),
            prompt           = prompt,
            visual_specs     = list(sec.visual_specs or []),
            context_snapshot = {k: context[k] for k in _extract_placeholder_keys(narrative_text) if k in context},
        ))

    n_ready = sum(1 for s in sections if s.ready)

    return ReportPlan(
        sections       = sections,
        context        = context,
        missing_fields = sorted(missing_fields_global),
        n_ready        = n_ready,
        n_total        = len(sections),
        yaml_path      = str(yaml_path),
    )
```

- [ ] **Step 4.4: Lancer les tests US-22**

Run: `python -m pytest tests/test_load_plan_design3.py -x -v`
Expected: 5 passed.

- [ ] **Step 4.5: Lancer la suite complète — attention aux régressions**

Run: `python -m pytest tests/ -x`
Expected: tous verts. Des tests anciens pouvaient tester `load_plan` legacy → s'ils échouent, les marquer `@pytest.mark.skip(reason="Design 1 obsolete, remplacé par US-22")` avec commentaire. Justifier dans le commit.

- [ ] **Step 4.6: Commit US-22**

```bash
git add agents/report/pipeline/_01_load_plan.py tests/test_load_plan_design3.py
git commit -m "feat(US-22): _01_load_plan réécrit sur template_loader (Design 3)

446 lignes → ~140. Supprime la dépendance à tools.build_pdf.load_yaml_template
(parser v1). Utilise template_loader.load_section() +
resolve_placeholders(). SectionPlan.visual_specs remplace les dicts
table/graph/stat_specs hardcodés."
```

---

## Task 5 (US-23): `_03_completion_plan` lit `rag_query` via loader

**Contexte** : `_SECTION_QUERIES` dict (lignes 44-76) hardcode les queries RAG pour 8 sections legacy. Design 3 stocke `llm_directives.rag_query` dans chaque section.

**Files:**
- Modify: `agents/report/pipeline/_03_completion_plan.py`.

- [ ] **Step 5.1: Test rapide qu'une query est bien lue depuis YAML**

Ajouter à `tests/test_load_plan_design3.py` :

```python
def test_completion_plan_reads_rag_query_from_yaml():
    from agents.report.pipeline._03_completion_plan import _query_for_section
    q = _query_for_section("preamble", "Préambule")
    assert q == "formulation préambule table mortalité portefeuille"
```

- [ ] **Step 5.2: Lancer → doit échouer (query vient encore du dict legacy)**

Run: `python -m pytest tests/test_load_plan_design3.py::test_completion_plan_reads_rag_query_from_yaml -x`
Expected: assertion échoue (query legacy différente).

- [ ] **Step 5.3: Modifier `_query_for_section` et supprimer `_SECTION_QUERIES`**

Remplacer dans [agents/report/pipeline/_03_completion_plan.py:44-85](agents/report/pipeline/_03_completion_plan.py#L44) :

```python
def _query_for_section(section_id: str, label: str) -> str:
    """Retourne la query RAG depuis llm_directives.rag_query du YAML."""
    try:
        from knowledge_base.report_template.template_loader import load_section
        sec = load_section(section_id)
        q = (sec.llm_directives or {}).get("rag_query")
        if q:
            return q
    except Exception:
        pass
    return f"rédaction professionnelle de la section '{label}' d'un rapport actuariel"
```

Supprimer entièrement le dict `_SECTION_QUERIES`.

- [ ] **Step 5.4: Tests**

Run: `python -m pytest tests/ -x`
Expected: tous verts.

- [ ] **Step 5.5: Commit US-23**

```bash
git add agents/report/pipeline/_03_completion_plan.py tests/test_load_plan_design3.py
git commit -m "feat(US-23): _03_completion_plan lit rag_query via template_loader

Supprime le dict _SECTION_QUERIES hardcodé (8 entrées legacy).
La query RAG est désormais lue dans llm_directives.rag_query
du YAML Design 3 via load_section()."
```

---

## Task 6 (US-24): `_04_redaction` — hydrate_table_spec simplifié

**Contexte** : `_hydrate_table_spec` (134 lignes, [lignes 36-169](agents/report/pipeline/_04_redaction.py#L36-L169)) transforme les specs de visuels v1 en données hydratées. En Design 3, `visual_specs[i].source` pointe directement vers une clé du data_store (ex: `portfolio_composition_by_sex`, `deaths_by_year_series`) qui contient déjà la donnée finale (list[dict]). Le "hydrate" devient un simple lookup.

**Files:**
- Modify: `agents/report/pipeline/_04_redaction.py` (simplification massive).
- Create: `tests/test_redaction_preamble.py`.

- [ ] **Step 6.1: Écrire le test d'hydratation Design 3**

```python
# tests/test_redaction_preamble.py
"""Tests US-24 : _04_redaction sur visual_specs Design 3."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.report.pipeline._04_redaction import _hydrate_visual_spec  # noqa: E402


def test_hydrate_table_reads_direct_from_data_store():
    spec = {
        "id": "portfolio_composition",
        "type": "table",
        "source": "portfolio_composition_by_sex",
        "columns": [
            {"key": "sexe",     "label": "Sexe"},
            {"key": "n_lives",  "label": "Vies"},
            {"key": "exposure", "label": "Exposition"},
            {"key": "deaths",   "label": "Décès"},
        ],
    }
    data_store = {
        "portfolio_composition_by_sex": [
            {"sexe": "H", "n_lives": 500, "exposure": 700.0, "deaths": 25},
            {"sexe": "F", "n_lives": 500, "exposure": 534.5, "deaths": 17},
        ],
    }

    out = _hydrate_visual_spec(spec, data_store)

    assert out["type"] == "table"
    assert out["headers"] == ["Sexe", "Vies", "Exposition", "Décès"]
    assert out["rows"] == [
        ["H", 500, 700.0, 25],
        ["F", 500, 534.5, 17],
    ]


def test_hydrate_chart_reads_direct_from_data_store():
    spec = {
        "id": "deaths_per_year",
        "type": "chart",
        "chart_type": "bar",
        "source": "deaths_by_year_series",
        "x_axis": {"key": "year",   "label": "Année"},
        "y_axis": {"key": "deaths", "label": "Décès"},
    }
    data_store = {
        "deaths_by_year_series": [
            {"year": 2019, "deaths": 10},
            {"year": 2020, "deaths": 15},
        ],
    }

    out = _hydrate_visual_spec(spec, data_store)

    assert out["type"] == "chart"
    assert out["chart_type"] == "bar"
    assert out["x_values"] == [2019, 2020]
    assert out["y_values"] == [10, 15]
    assert out["x_label"] == "Année"
    assert out["y_label"] == "Décès"


def test_hydrate_missing_source_returns_error_marker():
    spec = {"id": "foo", "type": "table", "source": "absent", "columns": []}
    out = _hydrate_visual_spec(spec, {})
    assert out["error"] is not None
```

- [ ] **Step 6.2: Lancer (doit échouer — `_hydrate_visual_spec` absente)**

Run: `python -m pytest tests/test_redaction_preamble.py -x`
Expected: ImportError.

- [ ] **Step 6.3: Ajouter `_hydrate_visual_spec` + simplifier**

Au début de [agents/report/pipeline/_04_redaction.py](agents/report/pipeline/_04_redaction.py), remplacer `_hydrate_table_spec` (lignes 36-169) par :

```python
def _hydrate_visual_spec(spec: dict, data_store: dict) -> dict:
    """Design 3 : `source` pointe vers une clé data_store contenant la donnée.
    Pour un tableau, on extrait (headers, rows). Pour un chart, (x_values, y_values)."""
    source = spec.get("source")
    data = data_store.get(source) if source else None
    stype = spec.get("type")

    if data is None:
        return {**spec, "error": f"source '{source}' absente du data_store"}

    if stype == "table":
        columns = spec.get("columns", [])
        headers = [c.get("label", c.get("key", "")) for c in columns]
        rows    = [[row.get(c["key"]) for c in columns] for row in (data or [])]
        return {**spec, "headers": headers, "rows": rows, "error": None}

    if stype == "chart":
        x_key = (spec.get("x_axis") or {}).get("key")
        y_key = (spec.get("y_axis") or {}).get("key")
        return {
            **spec,
            "x_values": [row.get(x_key) for row in (data or [])],
            "y_values": [row.get(y_key) for row in (data or [])],
            "x_label":  (spec.get("x_axis") or {}).get("label", x_key),
            "y_label":  (spec.get("y_axis") or {}).get("label", y_key),
            "error":    None,
        }

    return {**spec, "error": f"type non supporté: {stype}"}
```

- [ ] **Step 6.4: Adapter `_run_tables` et `_run_graphs`**

Dans `_04_redaction.py`, ces fonctions itéraient sur `section.table_specs` / `section.graph_specs`. Passer à `section.visual_specs` et filtrer par type :

```python
def _run_tables(section, data_store: dict) -> list[dict]:
    results = []
    for spec in section.visual_specs:
        if spec.get("type") == "table":
            results.append(_hydrate_visual_spec(spec, data_store))
    return results


def _run_graphs(section, data_store: dict) -> list[str]:
    # Pour l'instant : on rend un graphe minimal via graph_from_spec existant,
    # ou on retourne une URI placeholder si le renderer n'est pas adapté
    # Design 3. Détail du renderer laissé à une US suivante.
    paths = []
    for spec in section.visual_specs:
        if spec.get("type") == "chart":
            hydrated = _hydrate_visual_spec(spec, data_store)
            # Appel au renderer existant (adapter la signature au besoin)
            try:
                from tools.build_pdf.graph_from_spec import run as _graph
                result = _graph(
                    data={"x_values": hydrated["x_values"], "y_values": hydrated["y_values"]},
                    params={"chart_type": hydrated.get("chart_type", "bar"),
                            "x_label":    hydrated.get("x_label", ""),
                            "y_label":    hydrated.get("y_label", "")},
                )
                if result.get("path"):
                    paths.append(result["path"])
            except Exception as exc:
                log.warning("[04_redaction] graph render failed for %s: %s", spec.get("id"), exc)
    return paths
```

**NB** : la signature effective de `graph_from_spec.run` doit être vérifiée au moment de l'édition. Si elle diffère, adapter ; au pire, laisser les graphes désactivés pour le preamble (retourner `[]`) et noter l'amélioration future.

Supprimer `_run_stats` (Design 3 preamble n'a pas de stat_specs) ou conserver en no-op.

- [ ] **Step 6.5: Lancer les tests US-24**

Run: `python -m pytest tests/test_redaction_preamble.py -x -v`
Expected: 3 passed.

- [ ] **Step 6.6: Lancer la suite**

Run: `python -m pytest tests/ -x`
Expected: tous verts.

- [ ] **Step 6.7: Commit US-24**

```bash
git add agents/report/pipeline/_04_redaction.py tests/test_redaction_preamble.py
git commit -m "feat(US-24): _04_redaction hydrate visual_specs Design 3

Supprime _hydrate_table_spec (134 lignes d'agrégation) et le remplace
par _hydrate_visual_spec : source → data_store lookup direct, colonnes
→ headers/rows pour tables, x_axis/y_axis → x_values/y_values pour
charts. _run_tables et _run_graphs itèrent sur section.visual_specs."
```

---

## Task 7 (US-25): Supprimer `_02_validation_plan.py` + pipeline 3 étapes

**Contexte** : `_02_validation_plan` validait le plan legacy. En Design 3, la validation est faite en amont par `template_loader` + `scripts/check_template.py`. On supprime le fichier et on retire l'étape 02 de `run_pipeline.py`.

**Files:**
- Delete: `agents/report/pipeline/_02_validation_plan.py`.
- Modify: `agents/report/pipeline/run_pipeline.py` (supprimer l'étape 02).

- [ ] **Step 7.1: Identifier les appels à validate_plan dans run_pipeline**

Run: `grep -n "validate_plan\|_02_validation" agents/report/pipeline/run_pipeline.py`

- [ ] **Step 7.2: Supprimer l'étape 02 dans run_pipeline.py**

Retirer l'import `from agents.report.pipeline._02_validation_plan import validate_plan` et l'appel correspondant. Le pipeline passe directement de `load_plan` (01) à `complete_plan` (03).

- [ ] **Step 7.3: Supprimer le fichier**

```bash
git rm agents/report/pipeline/_02_validation_plan.py
```

- [ ] **Step 7.4: Vérifier qu'aucun autre import ne pointe vers ce module**

Run: `grep -rn "_02_validation_plan\|validate_plan" agents/ tests/ --include="*.py"`
Expected: zéro résultat (ou uniquement des imports dans le fichier supprimé).

- [ ] **Step 7.5: Lancer la suite**

Run: `python -m pytest tests/ -x`
Expected: tous verts.

- [ ] **Step 7.6: Commit US-25 (fin)**

```bash
git add agents/report/pipeline/run_pipeline.py
git commit -m "feat(US-25): pipeline 3 étapes, _02_validation_plan supprimé

La validation est faite en amont par check_template +
template_loader. Plus besoin d'une étape dédiée qui
dupliquait le check contractuel."
```

---

## Task 8: Test d'intégration pipeline preamble

- [ ] **Step 8.1: Écrire le test d'intégration**

```python
# tests/test_pipeline_preamble_e2e.py
"""Intégration : load_plan → complete_plan → _04_redaction sur preamble."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _data_store():
    return {
        "study_objective":               "construction_table_mortalite",
        "start_year":                    2019,
        "end_year":                      2021,
        "num_observation_years":         3,
        "total_exposure_years":          1234.5,
        "total_deaths":                  42,
        "portfolio_composition_by_sex":  [
            {"sexe": "H", "n_lives": 500, "exposure": 700.0, "deaths": 25},
            {"sexe": "F", "n_lives": 500, "exposure": 534.5, "deaths": 17},
        ],
        "deaths_by_year_series":         [
            {"year": 2019, "deaths": 10},
            {"year": 2020, "deaths": 15},
            {"year": 2021, "deaths": 17},
        ],
    }


def test_load_plan_produces_ready_preamble():
    from agents.report.pipeline._01_load_plan import load_plan
    plan = load_plan(_data_store())
    assert plan.n_ready == 1
    assert plan.sections[0].ready


def test_redaction_hydrates_both_visuals():
    from agents.report.pipeline._01_load_plan import load_plan
    from agents.report.pipeline._04_redaction import _run_tables

    plan = load_plan(_data_store())
    tables = _run_tables(plan.sections[0], _data_store())
    assert len(tables) == 1
    assert tables[0]["headers"][0] == "Sexe"
    assert len(tables[0]["rows"]) == 2
```

- [ ] **Step 8.2: Vérifier et commit**

Run: `python -m pytest tests/test_pipeline_preamble_e2e.py -x -v`
Expected: 2 passed.

```bash
git add tests/test_pipeline_preamble_e2e.py
git commit -m "test: intégration pipeline preamble (load → redaction)"
```

---

## Task 9: Vérification finale

- [ ] **Step 9.1: Suite complète + template check**

Run: `python -m pytest tests/ && python scripts/check_template.py`
Expected: ≥110 passed, template valide.

- [ ] **Step 9.2: Test manuel par l'utilisateur**

L'utilisateur lance l'app canvas, charge `Portefeuille/portefeuille_test_1000.csv`, demande "construis une table de mortalité", traverse la désambiguation (colonnes + valeurs) et lance la génération du rapport. Le PDF doit s'ouvrir avec :
- Une page preamble contenant le texte narratif avec placeholders résolus
- Un tableau "Composition du portefeuille par sexe"
- Un graphe bar "Décès par année"

Tout échec visuel → nouvelle US de correction, ne pas patcher dans ce bundle.

- [ ] **Step 9.3: Merger la branche**

```bash
# Une fois le test manuel validé
git checkout main
git merge --no-ff feat/cutover-preamble-e2e -m "feat: cutover preamble E2E (US-15,20,22,23,24,25)"
```

---

## Dépendances et risques

**Dépendances** : chaque task dépend de la précédente (ordre imposé). Master (US-15) avant Builder (US-20) parce que le Builder émet vers Master ; Builder avant Writer parce que Writer lit les clés produites par Builder ; `_SECTION_REQUIRED` (US-25 partiel) avant load_plan (US-22) parce que son existence peut masquer des erreurs de résolution ; validation_plan supprimé (US-25 fin) en dernier pour éviter un pipeline cassé au milieu.

**Risques** :
1. **Incompatibilités cachées dans `_04_redaction.py`** : le fichier (852 lignes) contient bien d'autres fonctions qui référencent les anciens specs. Garder le reste intact au maximum ; adapter uniquement les fonctions appelées par `redact_plan` dans le flux preamble. Tester après chaque modification.
2. **Tools LLM du Writer (`_call_llm_redaction`)** : non modifié par ce plan. Le prompt produit par US-22 doit rester compatible avec l'API OpenAI utilisée.
3. **Canvas/UI** : les vues canvas_app qui affichaient les clés legacy (`exposure_table`, etc.) deviendront vides. Cosmétique. Une US cosmétique suivra si besoin.
4. **Graph renderer** : la signature actuelle de `tools/build_pdf/graph_from_spec.run` peut ne pas correspondre à ce que le Design 3 produit. Si besoin, dans Task 6, on laisse `_run_graphs` retourner `[]` et on ajoute une US dédiée renderer.
