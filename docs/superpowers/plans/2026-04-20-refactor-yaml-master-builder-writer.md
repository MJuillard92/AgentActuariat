# Refactor YAML + redistribution Master / Builder / Writer — User Stories

> Document d'exécution dérivé de l'ADR `/Users/macbook14/.claude/plans/squishy-jingling-koala.md` (v3, 2026-04-20). Fil rouge : **template réduit au preamble** jusqu'à E2E vert, puis extension section par section.

---

## Contexte

Le sous-agent Writer s'est vu confier des responsabilités qui ne lui reviennent pas (agrégation métier, validation LLM des calculs). L'analyse conjointe a révélé que la cause racine est le YAML lui-même : duplications (`processing_sequence` ↔ `sections`), dict Python qui shadow les exigences YAML (`_SECTION_REQUIRED`), 6 blocs dormants. L'ADR acte une réécriture complète du YAML (**Design 3** : `data_contract` à la racine + sections purement rédactionnelles) puis une redistribution des rôles : Master garant qualité + orchestrateur, Builder single-pass, Writer rédacteur pur.

Ce document décompose l'ADR en ~25 User Stories indépendamment testables, ordonnées par dépendances, chacune livrable en 0.5-2 jours.

---

## Principes de décomposition

- **Testable isolément** : chaque US livre une fonction/un fichier avec tests unitaires qui passent seuls.
- **Dépendances explicites** : `Dépend de: US-X` en tête. Pas de US bloquante cachée.
- **Fil rouge preamble** : jusqu'à US-26, le template actif ne contient que le preamble. Les autres sections sont commentées en bloc dans le YAML.
- **Clean break** : pas de couche de compat. Chaque PR laisse le système fonctionnel.
- **Gate CI** : à partir de US-3, `check_template` vert est bloquant.

---

## Format d'une US

```
### US-N — <titre court>

**Dépend de** : US-X, US-Y (ou: aucune)
**Bloque**    : US-Z (listes inverses utiles pour planification)
**Estimation**: 0.5d / 1d / 2d

**Objectif** (1 phrase)

**Fichiers**
- Créer : `path/to/new.py`
- Modifier : `path/to/existing.py` (zone ciblée)
- Tests : `tests/path/test_xxx.py`

**Contrat** (si tool) : contrat catalogue minimal (inputs/outputs).

**Critères d'acceptation**
- [ ] critère 1 (mesurable)
- [ ] critère 2
- [ ] tests passent
- [ ] `check_template` vert (à partir de US-3)
```

---

## Vue d'ensemble — 9 phases

| Phase | Objet                             | US      | Livrable                              |
|-------|-----------------------------------|---------|---------------------------------------|
| 1     | Fondations (registry + validator) | US-1..3 | `check_template` vert sur YAML actuel |
| 2     | YAML v2 minimal                   | US-4..5 | Template v2 avec preamble seul actif  |
| 3     | Loader + style                    | US-6..8 | `template_loader.build_manifest()`    |
| 4     | Tools Master                      | US-9..12| 4 nouveaux tools sous `tools/master/` |
| 5     | Disambiguation étendue            | US-13..14 | `value_mapping` + `normalize_records` |
| 6     | Master refactor                   | US-15..19 | Cinématique 10 étapes complète      |
| 7     | Builder single-pass               | US-20..21 | DAG exécuté en une passe             |
| 8     | Writer amincissement              | US-22..25 | Pipeline 3 étapes, no hydrate        |
| 9     | E2E preamble + extension          | US-26..+  | PDF preamble vert, puis sections     |

**Chemin critique** : US-1 → US-2 → US-3 → US-4 → US-6 → US-10/11 → US-15 → US-17 → US-20 → US-26.

**Parallélisation possible** après US-5 :
- Branche A (infra) : US-6, US-7, US-8
- Branche B (tools Master) : US-9, US-10, US-11, US-12
- Branche C (aggregation) : tool `tools/aggregation/rules.py` si besoin preamble

---

## Phase 1 — Fondations

### US-1 — Tool registry

**Dépend de** : aucune
**Bloque**    : US-2
**Estimation**: 1d

**Objectif** : scanner `tools/` et exposer un registry `{tool_name: {inputs_schema, outputs_schema}}` construit à partir des en-têtes catalogue (pattern existant dans `tools/build_pdf/search_exemplars.py:1-113`).

**Fichiers**
- Créer : `knowledge_base/report_template/tool_registry.py`
- Tests : `tests/test_tool_registry.py`

**API**
```python
def build_registry(tools_root: Path = Path("tools")) -> dict[str, ToolSpec]
# ToolSpec = {inputs: dict[name, type], outputs: dict[name, type], path: str}
```

**Critères d'acceptation**
- [ ] Parse le header `"""...TOOL CONTRACT..."""` des fichiers `tools/**/*.py`
- [ ] Extrait nom (`build_pdf.search_exemplars`), inputs, outputs depuis les sections `INPUTS` / `OUTPUTS`
- [ ] Retourne les tools existants sans erreur (≥ 5 tools découverts dans le repo actuel)
- [ ] Détecte les collisions de noms → lève `ValueError`
- [ ] Tests unitaires : fixture minimale avec 2 faux tools, registry correctement construit

### US-2 — Validator contractuel YAML ↔ registry

**Dépend de** : US-1
**Bloque**    : US-3
**Estimation**: 1.5d

**Objectif** : logique réutilisable qui valide un YAML (Design 3) contre le registry. Consommée par le script CLI (US-3) et par le preflight Master (US-19).

**Fichiers**
- Créer : `knowledge_base/report_template/validator.py`
- Tests : `tests/test_validator.py`

**API**
```python
def validate_template(yaml_path: Path, registry: Registry) -> ValidationReport
# ValidationReport = {errors: list[Issue], warnings: list[Issue]}
# Issue = {severity, location, message}
```

**Checks bloquants implémentés** (cf. ADR §Validation) :
- YAML parse
- Chaque `produced_by.tool` ∈ registry
- `produced_by.inputs` ⊆ signature du tool
- `output_mapping` ou clé directe ∈ outputs du tool
- Chaque `{{ placeholder }}` dans `narrative.text` et `visual_specs.columns.label` résout contre `session_inputs` ∪ `data_contract`
- `type: date` dans un `shape` a un `format`
- `type: enum` a `allowed`
- Pas de cycle dans le DAG (détection topologique)
- `dependencies: [...]` pointe vers des sections existantes
- Unicité de production (une clé = un seul `produced_by`)

**Check warning** :
- Clé déclarée jamais consommée (ni placeholder, ni input d'un autre `produced_by`, ni source de visual_spec)

**Critères d'acceptation**
- [ ] Tous les checks ci-dessus couverts par un test unitaire dédié
- [ ] Rapport lisible avec pointeur `section/key` pour chaque erreur
- [ ] Passe sur un YAML Design 3 minimal valide (fixture)
- [ ] Échoue avec message actionable sur chaque cas cassé (fixtures)

### US-3 — Script `check_template` + test pytest

**Dépend de** : US-2
**Bloque**    : US-5 (gate CI bloquant à partir d'ici)
**Estimation**: 0.5d

**Objectif** : entry point CLI + wrapper pytest. Exit 0/1 avec rapport lisible. Bloquant en pré-commit et CI.

**Fichiers**
- Créer : `scripts/check_template.py`
- Créer : `tests/test_template_contract.py`

**Critères d'acceptation**
- [ ] `python scripts/check_template.py` → exit 0 si YAML valide, 1 sinon
- [ ] Sortie colorée lisible (optionnel via `--no-color`)
- [ ] `pytest tests/test_template_contract.py` équivalent
- [ ] Documentation d'usage dans docstring du script

---

## Phase 2 — YAML v2 minimal (preamble-only)

### US-4 — Réécriture `mortality_template.yaml` Design 3 preamble-only

**Dépend de** : US-2 (pour valider pendant l'écriture)
**Bloque**    : US-5, US-6
**Estimation**: 1d

**Objectif** : remplacer le YAML actuel par la version Design 3 avec **seul le preamble actif** ; les autres sections (construction, analysis, results, conclusion, annex) sont présentes mais commentées en bloc YAML (`# ...`).

**Fichiers**
- Modifier : `knowledge_base/report_template/mortality_template.yaml` (réécriture complète)

**Contenu cible** : copie exacte de l'annexe `Préambule v7` de l'ADR (`session_inputs` + `data_contract` avec 4 clés `master_from_data` + 1 clé `master_from_modeling` + 4 clés `builder_outputs` + 1 section `preamble`).

**Critères d'acceptation**
- [ ] YAML parse sans erreur
- [ ] Tous les 6 blocs dormants supprimés (`processing_sequence`, `inputs:` racine, `conditional_sections`, `agent_writer_instructions`, `generation_rules`, `quality_gates`)
- [ ] `check_template` vert (une fois les tools Master stubs créés — voir US-9..11 ; temporairement ignorer via flag `--skip-registry` si besoin, à retirer en US-12)
- [ ] Sections non-preamble présentes en commentaire pour référence

### US-5 — `check_template` vert en CI sur la nouvelle version

**Dépend de** : US-3, US-4
**Bloque**    : toutes les US suivantes qui touchent YAML ou tools
**Estimation**: 0.5d

**Objectif** : activer le gate CI bloquant. Rollback du flag `--skip-registry` si utilisé en US-4.

**Critères d'acceptation**
- [ ] Job CI ajouté (ou pre-commit hook) qui exécute `scripts/check_template.py`
- [ ] Build rouge si YAML ou tools désalignés

---

## Phase 3 — Loader + style

### US-6 — `template_loader.build_manifest()`

**Dépend de** : US-4
**Bloque**    : US-15
**Estimation**: 1d

**Objectif** : API Master. Projette `data_contract` en manifest consommable + DAG d'exécution.

**Fichiers**
- Créer : `knowledge_base/report_template/template_loader.py`
- Tests : `tests/test_template_loader.py`

**API**
```python
def build_manifest(yaml_path: Path = DEFAULT) -> Manifest
# Manifest = {
#   master_from_data:     list[KeySpec],
#   master_from_modeling: list[KeySpec],
#   builder_outputs:      list[KeySpec],
#   aggregations:         list[Aggregation],  # depuis visual_specs.aggregation
#   dag:                  list[ToolCall],     # ordonné topologiquement
# }
```

**Critères d'acceptation**
- [ ] Projection directe des 3 listes (pas de consolidation par section)
- [ ] DAG ordonné topologiquement, cycle → exception (redondant avec validator, mais défense en profondeur)
- [ ] Tests sur fixture preamble v7

### US-7 — `template_loader.load_section()` + `resolve_placeholders()`

**Dépend de** : US-6
**Bloque**    : US-22
**Estimation**: 0.5d

**Objectif** : API Writer. Livre narrative + directives LLM + visual_specs d'une section. Déménage `_resolve_placeholders` depuis `_01_load_plan.py:203`.

**API**
```python
def load_section(sid: str, yaml_path: Path = DEFAULT) -> Section
def resolve_placeholders(text: str, data_store: dict) -> str
```

**Critères d'acceptation**
- [ ] `load_section("preamble")` retourne narrative + llm_directives + visual_specs conformes au schéma
- [ ] `resolve_placeholders` substitue `{{ key }}` par `str(data_store[key])` par regex simple, lève `KeyError` si clé manquante
- [ ] Tests sur texte preamble avec 5 placeholders

### US-8 — `style.yaml` + lecture depuis `table_renderer`

**Dépend de** : aucune (parallélisable)
**Bloque**    : US-22
**Estimation**: 0.5d

**Objectif** : extraire la charte graphique hardcodée de `tools/build_pdf/table_renderer.py:76-79` dans un YAML dédié.

**Fichiers**
- Créer : `knowledge_base/report_template/style.yaml`
- Modifier : `tools/build_pdf/table_renderer.py` (lit `style.yaml` au lieu des constantes)

**Critères d'acceptation**
- [ ] Couleurs, polices, paddings déplacés
- [ ] Tests existants de `table_renderer` toujours verts
- [ ] Fallback sur defaults si `style.yaml` absent (warning, pas crash)

---

## Phase 4 — Tools Master

Chaque tool suit le pattern catalogue en tête (`"""TOOL CONTRACT..."""`), entry point `run(data, params)`.

### US-9 — `tools/master/classify_request.py`

**Dépend de** : US-1 (registry doit le voir)
**Bloque**    : US-4 (gate green), US-17
**Estimation**: 0.5d

**Objectif** : V1 trivial. Classifie la demande utilisateur en `study_objective`. Allowed pour V1 : `[construction_table_mortalite]`.

**Contrat**
- Inputs : `request: string`
- Outputs : `objective: enum[construction_table_mortalite]`

**Critères d'acceptation**
- [ ] Retourne toujours `construction_table_mortalite` en V1 (commentaire TODO pour extension multi-classes)
- [ ] Contrat catalogue parsable par le registry
- [ ] Test : un exemple de `raw_user_request` → retourne bonne valeur

### US-10 — `tools/master/analyze_data_and_request.py`

**Dépend de** : US-1
**Bloque**    : US-4 (gate green), US-17
**Estimation**: 1.5d

**Objectif** : inférences factuelles depuis les records. Retourne `observation_period_years`, `start_year`, `end_year`, `num_observation_years` (tous via `output_mapping` en un appel).

**Contrat**
- Inputs : `records: table` (records normalisés)
- Outputs : `period_years: list[int]`, `first_death_year: int`, `last_death_year: int`, `n_years: int`

**Logique**
- Filtrer records avec `cause_sortie == "deces"`
- Extraire l'année depuis `date_sortie`
- `first_death_year = min`, `last_death_year = max`, `n_years = last - first + 1`, `period_years = list(range(first, last+1))`

**Critères d'acceptation**
- [ ] Test avec CSV synthétique : années 2010-2015 → `first=2010, last=2015, n=6, period=[2010..2015]`
- [ ] Cas dégénéré (0 décès) → retourne valeurs `None` + warning, ne crash pas
- [ ] Contrat catalogue cohérent avec les 4 `output_mapping` du YAML preamble

### US-11 — `tools/master/suggest_value_mapping.py`

**Dépend de** : US-1
**Bloque**    : US-13
**Estimation**: 1d

**Objectif** : détecte les valeurs enum non conformes dans un DataFrame et propose un mapping.

**Contrat**
- Inputs : `records: table`, `enum_specs: dict[column, list[allowed]]`
- Outputs : `value_mapping: dict[column, dict[observed, canonical]]`, `unmapped: dict[column, list[value]]`

**Logique**
- Pour chaque colonne enum : compare valeurs observées aux `allowed`
- Heuristique simple : lower + strip accents + normalisation (`décédé` → `deces`, `vivant` → `autre`, `M` → `H` ?). Une table de synonymes basique suffit pour V1.
- Valeurs sans mapping évident → `unmapped` (user devra trancher)

**Critères d'acceptation**
- [ ] Fixture : `cause_sortie` avec `[decede, vivant]` → mappe vers `[deces, autre]`
- [ ] Fixture : valeur inconnue → apparaît dans `unmapped`
- [ ] Pas de modification du DataFrame (pure fonction)

### US-12 — `tools/master/normalize_records.py`

**Dépend de** : US-11
**Bloque**    : US-17
**Estimation**: 0.5d

**Objectif** : applique `column_mapping` et `value_mapping` sur un DataFrame. Produit la copie qui remplace `input_records` pour la suite.

**Contrat**
- Inputs : `records: table`, `column_mapping: dict`, `value_mapping: dict`
- Outputs : `normalized_records: table`

**Critères d'acceptation**
- [ ] Colonnes renommées selon `column_mapping`
- [ ] Valeurs enum substituées selon `value_mapping`
- [ ] Colonnes non mappées conservées telles quelles
- [ ] Original non muté (copie)

---

## Phase 5 — Disambiguation étendue

### US-13 — Confirmation value_mapping (UI dialogue Master)

**Dépend de** : US-11
**Bloque**    : US-17
**Estimation**: 1d

**Objectif** : intégrer `suggest_value_mapping` dans la chaîne runtime du Master (parallèle à la confirmation `column_mapping` existante, voir `agents/master/disambiguation.py`).

**Fichiers**
- Modifier : `agents/master/disambiguation.py` (ajouter `confirm_value_mapping`)

**Critères d'acceptation**
- [ ] Fonction qui présente les mappings suggérés au user, récupère les overrides
- [ ] Cas `unmapped` non vide → bloque avec message clair
- [ ] Tests : mock user input, vérifie que le mapping validé est correctement appliqué

### US-14 — Chaîne disambiguation complète

**Dépend de** : US-12, US-13
**Bloque**    : US-17
**Estimation**: 0.5d

**Objectif** : câbler la séquence `suggest_column_mapping → confirm_column_mapping → suggest_value_mapping → confirm_value_mapping → normalize_records`. Résultat : `input_records` du data_store est la version normalisée.

**Critères d'acceptation**
- [ ] Test d'intégration avec CSV non conforme → pipeline produit records normalisés conformes au shape YAML
- [ ] Log d'audit des mappings appliqués dans `data_store["_audit"]`

---

## Phase 6 — Master refactor

### US-15 — Master consomme `build_manifest()` (suppression `_ALL_BUILDER_KEYS`)

**Dépend de** : US-6
**Bloque**    : US-17, US-20
**Estimation**: 0.5d

**Objectif** : remplacer la liste hardcodée `_ALL_BUILDER_KEYS` dans `agents/mortality/agents/master_node.py:54-61` par un appel `template_loader.build_manifest()`.

**Critères d'acceptation**
- [ ] Constante supprimée, test existant toujours vert
- [ ] Manifest tiré du YAML au démarrage du Master

### US-16 — Dialogue 3-modes (choix utilisateur)

**Dépend de** : aucune (UX pur)
**Bloque**    : US-17, US-18
**Estimation**: 1d

**Objectif** : fonction utilitaire qui propose (a) autonome / (b) user-first / (c) proposition+validation au début de chaque phase (data, modeling). Mode stocké dans `data_store["_session"]["mode_<phase>"]`.

**Fichiers**
- Créer : `agents/master/dialogue_modes.py`
- Tests : `tests/test_dialogue_modes.py`

**Critères d'acceptation**
- [ ] 3 modes supportés, par défaut (c)
- [ ] Trace dans audit
- [ ] UX minimum : fonctionne en CLI ; canvas à itérer ensuite (hors scope)

### US-17 — Exécution `master_from_data` + dialogue

**Dépend de** : US-14, US-15, US-16
**Bloque**    : US-18, US-19
**Estimation**: 1.5d

**Objectif** : après disambiguation, le Master exécute le sous-DAG `master_from_data` du manifest (ici : `master.analyze_data_and_request` une fois, qui remplit les 4 clés). Puis lance le dialogue 3-modes du bloc data.

**Critères d'acceptation**
- [ ] Les 4 clés du preamble `master_from_data` présentes et typées dans `data_store`
- [ ] User peut override via mode (c), les overrides sont tracés
- [ ] Test E2E sur CSV synthétique

### US-18 — Exécution `master_from_modeling` + dialogue

**Dépend de** : US-17
**Bloque**    : US-19
**Estimation**: 0.5d

**Objectif** : sous-DAG `master_from_modeling` (preamble : `study_objective` seul). Dialogue du bloc modeling.

**Critères d'acceptation**
- [ ] `study_objective = construction_table_mortalite` dans data_store
- [ ] Test E2E

### US-19 — Preflight Master (Phase 0 + 1 + 2)

**Dépend de** : US-2, US-18
**Bloque**    : US-20
**Estimation**: 1d

**Objectif** : avant `GO_BUILD` : Phase 0 (validator au démarrage) ; après `BUILD_DONE` : Phase 1 (présence de toutes les clés manifest) + Phase 2 (structure : colonnes/types/sous-champs).

**Fichiers**
- Modifier : `agents/mortality/agents/master_node.py` (`_preflight_writer` étendu)

**Critères d'acceptation**
- [ ] Phase 0 exécutée au boot, crash early avec message actionable si YAML/registry désalignés
- [ ] Phase 1 : manque une clé → `NEED_DATA: <clés>` renvoyé au Builder (au lieu de continuer)
- [ ] Phase 2 : test avec structure cassée (colonne manquante dans table) → bloque
- [ ] Phase 3 (LLM optionnel) : stub avec flag `enable_llm_check=False` par défaut

---

## Phase 7 — Builder single-pass

### US-20 — Builder consomme manifest + exécute DAG

**Dépend de** : US-15, US-19
**Bloque**    : US-26
**Estimation**: 1.5d

**Objectif** : le Builder reçoit manifest via `GO_BUILD`. Exécute le sous-DAG `builder_outputs` en une passe, sans ré-ouvrir le YAML.

**Fichiers**
- Modifier : `agents/mortality/agents/builder_node.py`

**Critères d'acceptation**
- [ ] Ne lit plus `mortality_template.yaml`
- [ ] Exécute les 4 tools `mortality.compute_*` du preamble
- [ ] Les 4 clés `builder_outputs` présentes après `BUILD_DONE`
- [ ] Self-check avant `BUILD_DONE` (présence minimale)

### US-21 — Agrégation générique `tools/aggregation/rules.py`

**Dépend de** : US-1
**Bloque**    : US-26 (sections avec aggregation)
**Estimation**: 1d

**Objectif** : tool générique exposant les règles (`exposure_share_min`, `fixed_width`, `equal_count`, `none`). Appelable par Builder pour pré-agréger avant `BUILD_DONE`.

**Contrat**
- Inputs : `source: table`, `rule: enum`, `params: dict`, `weight: table?`
- Outputs : `aggregated: table`

**Critères d'acceptation**
- [ ] 4 règles implémentées et testées sur fixtures
- [ ] Tool registry le voit
- [ ] Preamble actuel n'en a pas besoin (tables sources déjà agrégées par `compute_composition`) — US peut être parallélisée

---

## Phase 8 — Writer amincissement

### US-22 — `_01_load_plan.py` délègue au loader

**Dépend de** : US-7, US-8
**Bloque**    : US-26
**Estimation**: 0.5d

**Objectif** : `_01_load_plan.py` utilise `template_loader.load_section` + `resolve_placeholders`. Suppression des locaux.

**Critères d'acceptation**
- [ ] Fichier réduit à ~50 lignes
- [ ] Tests pipeline existants verts

### US-23 — `_03_completion_plan.py` lit RAG query depuis YAML

**Dépend de** : US-7
**Bloque**    : US-26
**Estimation**: 0.5d

**Objectif** : supprimer `_SECTION_QUERIES` (lignes 44-76). Lit `llm_directives.rag_query` via le loader.

**Critères d'acceptation**
- [ ] Dict supprimé
- [ ] Test sur preamble : query lue = `"formulation préambule table mortalité portefeuille"`

### US-24 — `_04_redaction.py` — suppression `_hydrate_table_spec`

**Dépend de** : US-20 (Builder pré-agrège)
**Bloque**    : US-26
**Estimation**: 1d

**Objectif** : le hydrate devient trivial : `column.key → data_store[source][column.key]`. Supprimer les 134 lignes d'agrégation (lignes 36-169).

**Critères d'acceptation**
- [ ] Fonction `_hydrate_table_spec` supprimée
- [ ] Remplacée par un mapping direct de ~10 lignes
- [ ] Prompts déménagés vers `agents/report/prompts/`
- [ ] Test PDF preamble vert

### US-25 — Pipeline 3 étapes + suppression `_02_validation_plan.py`

**Dépend de** : US-19, US-22, US-23, US-24
**Bloque**    : US-26
**Estimation**: 0.5d

**Objectif** : `run_pipeline.py` : `01 load → 03 RAG → 04 write`. Suppression de `_02_validation_plan.py` (absorbé par preflight Master US-19).

**Critères d'acceptation**
- [ ] Fichier `_02_validation_plan.py` supprimé
- [ ] `run_pipeline.py` à 3 étapes
- [ ] Dict `_SECTION_REQUIRED` dans `load_yaml_template.py:144-154` supprimé

---

## Phase 9 — E2E preamble + extension

### US-26 — E2E preamble vert

**Dépend de** : US-20, US-24, US-25
**Bloque**    : US-27+
**Estimation**: 1d

**Objectif** : scénario complet sur un CSV réel (`Portefeuille/portefeuille_test_1000.csv`) → PDF preamble généré, placeholders résolus, tables/graph présents.

**Critères d'acceptation**
- [ ] PDF généré sans warning
- [ ] Les 5 placeholders du preamble résolus
- [ ] Table `portfolio_composition` et chart `deaths_per_year` présents
- [ ] `check_template` vert
- [ ] Run < 60s

### US-27+ — Extension section par section

**Dépend de** : US-26
**Estimation**: 0.5-1d par section × 5 sections = 2.5-5d

**Objectif** : décommenter et finaliser `construction`, `analysis`, `results`, `conclusion`, `annex` dans le YAML, une section à la fois. Pour chaque section :

1. Rédiger `data_contract` (nouvelles clés uniquement)
2. Rédiger `narrative` + `llm_directives` + `visual_specs`
3. Ajouter les tools Builder manquants si besoin
4. `check_template` vert
5. E2E section vert

Chaque section = 1 US (US-27 construction, US-28 analysis, etc.).

---

## Graphe de dépendances (chemin critique)

```
US-1 ─┬─ US-2 ─ US-3 ─┬─ US-4 ─ US-5 ─┬─ US-6 ─┬─ US-15 ─┬─ US-17 ─ US-18 ─ US-19 ─ US-20 ─┐
      │                │              │        │         │                                 │
      │                │              └─ US-7 ─┤         │                                 │
      │                │                       │         │                                 │
      │                │              US-8 ────┘         │                                 │
      │                │                                 │                                 │
      └─ US-9,10,11 ───┤                                 │                                 │
                      US-12 ─ US-13 ─ US-14 ─────────────┘                                 │
                                                                                           │
                                                US-16 ─────────────────────────────────────┤
                                                                                           │
                                                US-21 (parallèle) ─────────────────────────┤
                                                                                           │
                                                US-22,23,24,25 ────────────────────────────┴─ US-26 ─ US-27+
```

**Durée estimée critique** : ~20 jours-homme si séquentiel ; ~12-14 jours avec parallélisation Phase 4 + Phase 8.

---

## Règles d'exécution

1. **Branche par US** : `feat/us-N-<slug>`. PR dédiée. Review obligatoire.
2. **Tests d'abord** : TDD sur toutes les US avec composant isolé (tools, validator, loader).
3. **`check_template` bloquant** dès US-5.
4. **Pas de "while I'm here"** : l'US ne touche que ce que son périmètre exige. Les nettoyages opportunistes partent en US séparée.
5. **Commit frequent** : un commit par step significatif (test écrit, implémentation, refactor).
6. **Invoquer systématiquement** `superpowers:writing-plans` puis `superpowers:executing-plans` / `subagent-driven-development` pour chaque US.

---

## Points ouverts (tranchés au fil de l'eau)

- UX dialogue 3-modes (canvas vs CLI) — décidé en US-16 sur version minimale CLI
- Liste définitive des règles d'agrégation — dépend des sections US-27+
- Few-shot examples : YAML vs RAG — tranché à l'inventaire analysis/results
