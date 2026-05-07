# Plan de test rigoureux — refactor `report_mode` + anti-boucle

> Objectif : valider que le refactor sur la boucle Master ↔ Builder (plan `abundant-chasing-kazoo`) tient la route sur l'ensemble des situations utilisateur, sans consommer plus de 50 000 tokens par session. À lancer après chaque modification structurelle.

---

## 1. État courant à l'issue du refactor

- **161 tests unitaires verts** (`pytest tests/ -q`).
- **Template YAML valide** (`scripts/check_template.py` → ✓ valide, 2 warnings non bloquants).
- **15 nouveaux tests** dans `tests/test_report_mode_three_axes.py` couvrant les 3 axes + les garde-fous.

---

## 2. Couverture automatique actuelle

### 2.1. Loader / YAML (6 tests)

| Test | Ce qu'il vérifie |
|---|---|
| `test_is_active_old_format_scalar` | Format ancien `{key, equals}` toujours supporté (rétro-compat) |
| `test_is_active_new_format_list_match` | Format `{field: [values]}` valide la présence dans la liste |
| `test_is_active_new_format_multi_key_and` | AND implicite multi-clés (report_mode ET gender_segmentation) |
| `test_is_active_missing_context_key_is_tolerant` | Clé d'activation absente du contexte → contrainte non évaluée |
| `test_load_section_default_variant` | `load_section(ctx={report_mode: raw_rates})` choisit `text_raw_rates` |
| `test_load_section_single_text_backward_compat` | Section avec `narrative.text` seulement : inchangée |

### 2.2. Helpers Master (3 tests)

| Test | Ce qu'il vérifie |
|---|---|
| `test_sections_for_mode_full_report` | Sections actives en `full_report + unisex` (au moins preamble, data_preprocessing, data_analysis_unisex) |
| `test_sections_for_mode_by_sex` | Sections actives en `by_sex` (data_analysis_by_sex, pas unisex) |
| `test_keys_for_sections_subset_of_builder_outputs` | Les clés retournées sont un sous-ensemble strict de `_get_builder_keys()` |

### 2.3. Cinématique Master (3 tests)

| Test | Ce qu'il vérifie |
|---|---|
| `test_master_asks_write_question_before_builder` | `write=ask` → Master émet la question, pas de route Builder, `_write_question_asked=True` |
| `test_master_routes_to_builder_when_write_yes` | `write=yes` → Master émet instruction Builder contenant "Sections actives" et "Reste à produire" |
| `test_master_cumulative_cycle_limit` | 3 cycles atteints → Master émet `done` au lieu de relancer Builder |

### 2.4. Builder (3 tests)

| Test | Ce qu'il vérifie |
|---|---|
| `test_capabilities_block_non_empty_and_references_sections` | Le bloc généré du YAML mentionne au moins preamble + data_preprocessing + formatage "Clés/Tools" |
| `test_builder_raw_rates_assimilation_is_deterministic` | `report_mode=raw_rates` + `qx_table` → `smoothed_table` auto-produit, pas de LLM sur smoothing |
| `test_decision_gate_erases_tool_calls_even_without_content` | `decision_required` pending + LLM émet tool_calls sans content → tool_calls écrasés + content forcé |

### 2.5. Validator (tous les tests existants toujours verts)

- `test_activation_field_is_recognized` — format ancien OK.
- `test_activation_key_must_reference_enum_in_data_contract` — clé inconnue en format ancien rejetée.
- `test_activation_coverage_must_be_exhaustive` — couverture d'enum OK.
- Nouveaux cas couverts implicitement par le template `mortality_template.yaml` qui passe en format multi-clés.

---

## 3. Tests à ajouter (non bloquants, à enrichir en cours de route)

Ces tests ne sont pas encore écrits faute de mocking OpenAI suffisant ou de scénarios d'intégration trop complexes à simuler, mais ils valent la peine d'être ajoutés au fur et à mesure :

| Test à ajouter | Description | Priorité |
|---|---|---|
| `test_classify_intent_three_axes_keywords.py` | Mocker `call_with_retry` pour forcer 10 classifications différentes et valider que keys retournées sont bien `{kind, write, report_mode}` | Haute |
| `test_master_later_report_request.py` | data_store déjà complet + user tape "fais-moi le rapport" → Master voit missing_keys=[] et route direct Writer (zéro Builder) | Haute |
| `test_builder_full_report_pipeline_order.py` | Mocker les tools pour vérifier que le Builder appelle exposure → crude → diagnostics → smoothing → validation → benchmarking dans cet ordre | Moyenne |
| `test_builder_description_mode_skips_build_tools.py` | `report_mode=description` → Builder n'appelle jamais `builder.crude_rates`, `builder.smoothing`, etc. | Moyenne |
| `test_pipeline_e2e_report_modes.py` | 3 scénarios synthétiques bout en bout (Writer rend un PDF structurellement différent par mode) | Basse (E2E hors scope immédiat) |

---

## 4. Tests manuels avec l'application

**Prérequis** : `AGENT_SMOOTHING_STUB=1 streamlit run canvas_app.py` — le stub permet au mode `full_report` de converger sans violations de monotonie, pour tester la cinématique. Une fois la cinématique validée, relancer sans `AGENT_SMOOTHING_STUB` pour valider le cas réel.

### 4.1. Matrice des 8 scénarios principaux

Chaque ligne = une session fraîche (Ctrl+C Streamlit puis relance).

| # | Phrase utilisateur initiale | Attente `classify_intent` | Comportement Master attendu | Résultat final attendu |
|---|---|---|---|---|
| 1 | "construis-moi une table de mortalité" | kind=task, write=ask, mode=full_report | Émet question "Voulez-vous un rapport PDF ?" AVANT de router Builder | User répond "oui" → Builder complet + PDF complet |
| 2 | "fais-moi le rapport avec les taux bruts" | kind=task, write=yes, mode=raw_rates | Route direct Builder + Writer (pas de question) | PDF avec encart "taux bruts assimilés" ; aucun appel `builder.smoothing` |
| 3 | "fais une analyse descriptive" | kind=task, write=ask, mode=description | Émet question | User "non" → Builder stats only + done (pas de PDF) |
| 4 | "calcule les taux lissés sans rapport" | kind=task, write=no, mode=full_report | Route direct Builder (pas de question) | Builder complet, pas de Writer, message final "Résultats en mémoire..." |
| 5 | "rédige-moi un rapport descriptif" | kind=task, write=yes, mode=description | Route direct Builder + Writer | Builder stats + PDF allégé (sans section table_construction/smoothing/validation/benchmarking) |
| 6 | "pas de PDF, juste les taux bruts" | kind=task, write=no, mode=raw_rates | Route direct Builder | Builder (assimilation), pas de PDF |
| 7 | "c'est quoi le lissage Whittaker ?" | kind=question | Réponse LLM conversationnelle | Pas de Builder, pas de Writer, done |
| 8 | Scénario 4 suivi de "finalement fais-moi le rapport" | 2e classify → write=yes | Voit missing_keys=[] (data_store déjà rempli) | Writer direct, aucun nouveau tool Builder |

### 4.2. Métriques à monitorer par scénario

Pour chaque scénario, relever dans les logs Streamlit :

| Métrique | Où la trouver | Valeur cible |
|---|---|---|
| Valeur de `_master_builder_cycles` à la fin | data_store dans les events streamés | ≤ 2 |
| Nombre d'appels `gpt-4o` total | Events `llm_input` + `llm_output` | ≤ 10 pour scénarios 3/6, ≤ 25 pour 1/2/4/5 |
| Total tokens consommés (prompt + completion) | Events `llm_output.total_tokens` | ≤ 50 000 par session |
| Passage par la branche déterministe `raw_rates` | Absence de trace d'appel `builder.smoothing` dans les events | Vrai uniquement pour scénarios 2 et 6 |

### 4.3. Cas de régression à vérifier explicitement

- **Scénario boucle historique** (bug à l'origine de ce refactor) :
  - Session fraîche, sans `AGENT_SMOOTHING_STUB`.
  - "fais-moi le rapport" → le Builder tourne, le smoothing émet `decision_required` sur monotonie.
  - **Attendu** : Builder s'arrête avec la question affichée. Pas de tool_calls supplémentaires. `_master_builder_cycles ≤ 2`.
  - **Rejet** : si le nombre d'appels gpt-4o dépasse 30 ou si le smoothing est lancé plusieurs fois de suite.

- **Scénario persistance inter-messages** :
  - Session fraîche : "calcule sans rapport" → Builder tourne.
  - Puis dans la même session, user tape "finalement fais le rapport".
  - **Attendu** : Master reclassify → write=yes → missing_keys=[] → Writer direct. Aucun nouveau tool Builder.
  - **Rejet** : si le Builder retourne ET exécute des tools.

---

## 5. Plan de validation finale avant merge

À exécuter **dans cet ordre** :

### Étape A — Non-régression

```bash
cd /Users/macbook14/Python_projects/AgentActuariat
python -m pytest tests/ -q
python scripts/check_template.py
```

Cible : **161 passed** + **✓ template valide**.

### Étape B — Tests manuels prioritaires (durée ~30 min)

Lancer l'app avec le stub :
```bash
AGENT_SMOOTHING_STUB=1 streamlit run canvas_app.py
```

Passer **au minimum** les scénarios 1, 2, 3, 7, 8 de la matrice 4.1. Chaque scénario doit respecter ses métriques 4.2.

### Étape C — Test boucle réelle (durée ~10 min)

Relancer sans stub :
```bash
streamlit run canvas_app.py
```

Passer uniquement le scénario "Régression boucle historique" de 4.3. Objectif : valider que la cause racine (désalignement Master/Builder) est corrigée et que le gate `decision_required` tient.

### Étape D — Tests d'intégration à ajouter

Écrire progressivement les tests de la section 3 (à raison d'un par jour idéalement), pour étendre la couverture.

---

## 6. Signaux d'alerte (à monitorer en production)

Ces signaux indiquent une régression subtile :

| Signal | Cause probable | Action |
|---|---|---|
| `_master_builder_cycles` atteint 3 dans une session | Master ↔ Builder boucle malgré tout | Vérifier que `missing_keys` diminue à chaque cycle ; sinon le Builder produit des clés qui ne matchent pas celles attendues par Master |
| Le message "Voulez-vous un rapport PDF ?" s'affiche alors que l'user a déjà dit rapport | Classification LLM hésitante ou flag `_write_question_asked` mal réinitialisé | Vérifier `_classify_intent` ; tester avec la phrase exacte |
| `builder.smoothing` tourne en mode `raw_rates` | Le Builder LLM ignore la règle du system prompt | Vérifier que la règle `raw_rates → pas de smoothing` est dans le prompt ; durcir avec un garde-fou au tool_registry (rejeter l'appel) |
| Le PDF a une section smoothing vide en mode `description` | Writer pas aligné sur activation | Vérifier `build_manifest(context={"report_mode": "description"})` retourne bien les bonnes sections |

---

## 7. Annexe — Liste des fichiers modifiés

| Fichier | Changement |
|---|---|
| `knowledge_base/report_template/mortality_template.yaml` | Ajout `activation.report_mode` sur les 4 sections existantes |
| `knowledge_base/report_template/template_loader.py` | `_is_active` multi-format + `_select_narrative_variant` + `load_section(context=)` |
| `knowledge_base/report_template/validator.py` | `_validate_activation` accepte les 2 formats |
| `agents/mortality/agents/master_node.py` | `_classify_intent` 3 axes + `_sections_for_mode` / `_keys_for_sections` / `_get_required_keys_for_current_mode` + désambiguation `ask` + compteur cumulatif + instruction Builder dérivée |
| `agents/mortality/agents/builder_node.py` | `_capabilities_block` injecté dans system prompt + branche raw_rates + garde-fou `decision_required` durci |
| `agents/mortality/agents/graph.py` | `_MINIMUM_BUILDER_KEYS` → lecture dynamique via master helper |
| `tests/test_template_contract.py` | Test activation adapté au nouveau format |
| `tests/test_report_mode_three_axes.py` | **Nouveau** — 15 tests sur les 3 axes + garde-fous |

---

## 8. Prochaines US recommandées (non-urgentes)

- **US-XX** : ajouter `table_construction`, `smoothing`, `validation`, `benchmarking`, `conclusion` sections dans le YAML avec leurs activations `report_mode` appropriées. Aujourd'hui ces sections n'existent pas encore — le YAML ne contient que les 4 sections Bloc A.
- **US-XX** : déclarer `report_mode` comme enum dans `data_contract.master_from_modeling` avec `allowed: [full_report, raw_rates, description]` pour que le validator checke la couverture d'enum.
- **US-XX** : garde-fou tool_registry pour rejeter `builder.smoothing` en mode `raw_rates` (deuxième ceinture, protège contre une désobéissance LLM au prompt).
- **US-XX** : ajouter test E2E pipeline complet pour les 3 modes (simulation sans vraie génération PDF).
