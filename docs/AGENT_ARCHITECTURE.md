# Document factuel — Architecture et logique de l'agent AgentActuariat

> **But du document** : décrire *précisément* ce que fait l'agent (pas ce qu'il devrait faire) pour permettre une critique externe de la logique.
> Toutes les affirmations sont sourcées par chemin de fichier + numéro de ligne.
> Quand une information n'a pas été trouvée dans le code, c'est indiqué `NON TROUVÉ` — jamais inventé.
>
> Périmètre couvert : MasterAgent + BuilderAgent + WriterAgent + Pipeline de rapport + SessionState.

---

## 0. Vue d'ensemble

`AgentActuariat` est un agent multi-nœud LangGraph orienté construction et certification de tables de mortalité d'expérience. L'UI est une application Dash ([canvas_app.py](../canvas_app.py)), l'orchestration un `StateGraph` LangGraph, le calcul actuariel une suite d'outils Python (pandas/numpy/scipy), le rapport un pipeline PDF déterministe + appels LLM ciblés.

**Modèle LLM utilisé partout** : `gpt-4o` via OpenAI SDK. Aucun autre modèle détecté.

**Responsabilités par composant** (à haut niveau) :

| Composant | Rôle | LLM | Fichier principal |
|---|---|---|---|
| **Canvas UI** | Dialogue utilisateur, upload CSV, rendu des events | non | [canvas_app.py](../canvas_app.py) |
| **MasterAgent** (node) | Classification d'intention, routing, désambiguïsation, réponses directes aux questions | oui (3-4 prompts ciblés) | [agents/mortality/agents/master_node.py](../agents/mortality/agents/master_node.py) |
| **BuilderAgent** (node) | Calculs actuariels — choisit et enchaîne les tools par function-calling | oui (ReAct, `tool_choice=auto`) | [agents/mortality/agents/builder_node.py](../agents/mortality/agents/builder_node.py) |
| **WriterAgent** (node LangGraph) | Wrapper qui appelle directement `run_pipeline()` — aucun LLM au niveau du nœud | non | [agents/mortality/agents/writer_node.py](../agents/mortality/agents/writer_node.py) |
| **ReportNode** (node LangGraph) | Variante LLM tool-calling (non utilisée par le WriterNode actuel) | oui | [agents/mortality/agents/report_node.py](../agents/mortality/agents/report_node.py) |
| **Tools node** | Exécute séquentiellement les tool-calls produits par Builder/Writer | non | [agents/mortality/agents/tools_node.py](../agents/mortality/agents/tools_node.py) |
| **Pipeline de rapport** | 6 étapes déterministes + 3 appels LLM (02, 04, 06) | partiel | [agents/report/pipeline/](../agents/report/pipeline/) |
| **SessionState + MemoryManager** | Source de vérité métier persistée sur disque ; LangGraph MemorySaver = cache RAM | non | [session/](../session/) |

**Deux mémoires distinctes** :
- `MemorySaver` de LangGraph = **working memory** par session (RAM, `thread_id = session_id`). Perdue si process redémarre.
- `SessionState` (Pydantic) = **business memory** persistée en JSON sur disque (`session/data/{session_id}_state.json`) + DataFrame du portefeuille en Parquet (écrit une seule fois).

---

## 1. Graphe LangGraph

Fichier : [agents/mortality/agents/graph.py](../agents/mortality/agents/graph.py)

### 1.1 Topologie

4 nodes + edges conditionnelles :

```
START
  │
  └─ _router (conditional) ─── selon state["active_agent"] ───┐
                                                              │
                                ┌─────────────┬───────────────┤
                                ▼             ▼               ▼
                            "master"    "builder"         "writer"
                                │             │               │
        ┌───────────────────────┤             │               │
        │                       │             │               │
        │   _should_continue_master           │               │
        │   ├── "to_builder" ─┐               │               │
        │   ├── "to_writer"  ─┼──────────────▶│               │
        │   └── "done"       ─┼──────────────────────────────▶│
        │                     │               │               │
        ▼                     ▼               ▼               ▼
      END ◀─────┐        ┌─ tools ─┐    _should_continue_builder
                │        │          │    ├── "tools" ──────▶ tools
                │        │          │    ├── "to_master" ──▶ master
                │        │          │    └── "done" ──────▶ END
                │        └──────────┘
                │                          (idem pour writer)
                │
                └── _should_continue_writer / tools → lambda → {master, builder, writer}
```

**Nodes** (`graph.py:176-179`) : `master`, `builder`, `writer`, `tools`.
**Checkpointer** : `MemorySaver()` unique partagé entre sessions (`graph.py:47`). `thread_id = session_id` format `yymmddhhmm` ([canvas_app.py:54]).
**Routeurs conditionnels** : `_should_continue_master/builder/writer` (`graph.py:63-131`).
**Borne de sécurité** : le builder est limité à **5 itérations** pour éviter une boucle infinie LLM→tools→LLM (`graph.py:108-110`).

### 1.2 State partagé

TypedDict `AgentState` ([agents/mortality/agents/state.py:17-37](../agents/mortality/agents/state.py)) :

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # reducer standard
    dataset_ref: Optional[str]                           # session_id → Parquet
    data_store: Dict[str, Any]                           # résultats accumulés
    context_docs: List[Any]                              # docs uploadés
    plan_established: bool                               # plan validé ?
    active_agent: str                                    # "master" | "builder" | "writer"
    events: List[Any]                                    # events pour canvas
    step_by_step: bool                                   # mode pas à pas ?
    pending_tool_call: Optional[Dict[str, Any]]          # tool en attente (approval)
```

**Mutations observées** :
- `messages` : tous les nodes (reducer concat) ;
- `data_store` : `tools_node.py:206`, `master_node.py:282/302/310/339/369/378/428/436/464`, `graph.py:331-333` ;
- `active_agent` : `master_node.py:301/338/427/435/449/463` (seul le master change le routing) ;
- `plan_established` : `tools_node.py:209` ;
- `pending_tool_call` : `tools_node.py:210`.

**Le DataFrame brut n'entre PAS dans l'état** — il est lu à la demande via `MemoryManager.load_dataframe()` depuis Parquet (`graph.py:303-308`).

### 1.3 Entrée — un tour utilisateur

1. Click "Envoyer" dans le canvas → callback Dash → `_run_writer_in_thread()` ([canvas_app.py:133-224]).
2. `stream_agent(history, df, data_store, context_docs, step_by_step, thread_id)` ([graph.py:195-345]) :
   - Charge `SessionState` via `MemoryManager` (`graph.py:267-268`).
   - Hydrate `data_store` depuis `SessionState.to_data_store()` (`graph.py:273-276`).
   - Convertit l'historique Dash (list[dict]) en messages LangChain (`graph.py:279-288`).
   - Compacte si `len(messages) > COMPACT_THRESHOLD` (`graph.py:291`).
   - Invoque le graphe compilé avec `thread_id` (`graph.py:329`).
3. À la fin du stream : `mm.after_turn(final_data_store, lc_messages)` persiste (`graph.py:345`).

**Nombre d'appels LLM par tour** (typique) :
- Master : 1 appel (classification d'intention) + 0-2 optionnels (study plan extraction, réponse directe à une question) = **1-3**
- Builder (si invoqué) : 1 appel LLM qui déclenche N tool calls, éventuellement suivi d'un nouvel appel LLM. Borné à 5 tours.
- Writer (si invoqué) : 0 appel LLM au niveau nœud, mais **le pipeline de rapport** déclenche 3 appels LLM internes (étapes 02, 04 par section, 06).
- **Total par tour** : 2-4 appels LLM hors pipeline de rapport ; +10-15 dans le pipeline de rapport si déclenché.

---

## 2. MasterAgent

Fichier : [agents/mortality/agents/master_node.py](../agents/mortality/agents/master_node.py)

### 2.1 Responsabilités

Le MasterAgent **ne tient aucun tool OpenAI**. Il fait uniquement :
1. Classification d'intention (LLM).
2. Désambiguïsation au premier tour si CSV chargé.
3. Extraction du `study_plan` depuis l'historique (LLM).
4. Routing vers `builder` / `writer` / réponse directe.
5. Réponse conversationnelle aux questions (LLM).

### 2.2 Appels LLM

Tous `gpt-4o`. `temperature` non spécifiée (défaut OpenAI = 1.0).

| Fonction | Ligne | max_tokens |
|---|---|---|
| `_classify_intent()` | 99-105 | 200 |
| `_extract_study_plan_from_history()` | 164-170 | 400 |
| `_classify_with_llm()` (désambiguïsation) | 196-202 | 80 |
| Réponse directe à un intent `"question"` | 480-481 | 1500 |

**System prompt** : chargé dynamiquement via `loader.py` ([master_node.py:46]) en appelant `mod.get_system_prompt(level="full", agent_name="master")`. Le fallback pointe vers `agents/master/agent_instructions/behavioral_contract.md` (chargement dynamique non vérifié dans cette enquête — le contenu exact n'a pas été lu).

**Augmentation du prompt** (`_augment_with_data_store()`, lignes 177-222) — injecte dans le system prompt :
- Statut dataset chargé (`l.189-192`)
- Colonnes mappées (`l.194`)
- Résultats déjà calculés (`l.208-211`)
- SMR global si dispo (`l.219-220`)

### 2.3 Logique de routing

Les intents classifiés sont : `build_only`, `build_and_write`, `write_only`, `question`, `unclear`. Ordre de la logique (`master_node.py:269-494`) :

1. **WRITE_DONE** (`l.269-283`) : flag posé par le writer → cycle fini → END.
2. **BUILD_DONE** (`l.285-311`) : si intent demandait un write → router vers `writer`.
3. **NEED_DATA** (`l.319-343`) : si le writer manque des données → re-router vers `builder`.
4. **Désambiguïsation** (`l.345-380`) : au premier tour, appelle `run_disambiguation()`.
5. **Classification** (`l.390-393`) : `_classify_intent()`.
6. **Extraction study_plan** (`l.395-404`) si absent.
7. **Routing final** (`l.413-493`) :
   - `build_only` / `build_and_write` → `builder`
   - `write_only` → `writer` ou upgrade en `build_and_write` si inputs manquants
   - `question` → LLM conversationnel direct
   - `unclear` → `done`

**Pas de boucle ReAct au niveau master** : single-shot par tour, chaque appel produit un message et un routing, point.

### 2.4 Désambiguïsation

Fichier : [agents/master/disambiguation.py](../agents/master/disambiguation.py)

- **Trigger** : `not data_store.get("_disambiguation_done")` (`master_node.py:346`).
- **`classify_intent()`** (`l.389`) : règles mots-clés d'abord (`_INTENT_RULES`, `l.30-55`), fallback LLM JSON-mode si confiance < 0.5 (`l.176-209`). Task_type ∈ `{mortality_table, report, descriptive, replay, unknown}`.
- **`check_prerequisites()`** (`l.418`) : charge `catalogue.yaml`, lit `tools_section[task_type].user_inputs_required`, vérifie présence dans `study_plan` ou `data_store` (`l.214-280`).
- **`suggest_column_mapping()`** (`l.285-320`) : exact match d'abord, puis alias via `EXPECTED_COLUMNS` (`l.58-78`), fallback LLM si ambigu (`l.315, 323-361`).

Le dict retourné est stocké dans `data_store["_disambiguation_done"] = True` (`master_node.py:380`), le reste utilisé pour émettre un event `disambiguation_required` vers le canvas (`l.362-370`).

---

## 3. BuilderAgent

Fichier : [agents/mortality/agents/builder_node.py](../agents/mortality/agents/builder_node.py)

### 3.1 Appel LLM

```python
# l.171-177
response = call_with_retry(
    client,
    model="gpt-4o",
    messages=messages,
    tools=tools if tools else None,
    tool_choice="auto" if tools else None,
    max_tokens=4000,
)
```

- `temperature` : **non spécifiée** dans le code → défaut OpenAI 1.0.
- Retry : backoff exponentiel 15s/30s/60s, max 4 tentatives, uniquement sur erreurs rate-limit/transitoires ([`_utils.py:39-83`](../agents/mortality/agents/_utils.py)).

### 3.2 Tools exposés

```python
# builder_node.py:31, 104-106
BUILDER_TOOLS = {"builder", "statistical_analysis", "graphs", "reasoning", "build_pdf"}
all_tools = get_openai_tools()
tools = [t for t in all_tools if t["function"]["name"] in BUILDER_TOOLS]
```

**Mécanisme** : OpenAI function-calling standard. Le LLM choisit seul les tools et leur ordre. C'est **un vrai agent ReAct** (au sens "Reason + Act"), borné par la limite de 5 itérations dans `graph.py`.

### 3.3 System prompt — 3 niveaux

La fonction `_build_system_prompt(state, level)` (`l.46-121`) choisit un niveau :
- `"light"` (`l.39`) : si le dernier message est `ToolMessage` (le LLM vient d'exécuter un tool, prompt court pour économiser tokens).
- `"middle"` (`l.37`) : 1er message user, plan pas encore établi.
- `"full"` (`l.42`) : plan établi, messages utilisateur récents.

Le contenu vient de `loader.py:get_system_prompt(level, agent_name="mortality")`. Augmentations :
- Mapping de colonnes depuis `MemoryManager` (`l.60-84`)
- `study_plan` confirmé (`l.88-103`)
- Instruction `BUILD_DONE` à émettre pour signaler la fin des calculs (`l.106-112`)
- Documents de contexte uploadés par l'utilisateur (`l.115-119`)

### 3.4 Historique conversationnel tronqué

L'historique passé au LLM est limité à **20 messages** (`l.139`). Au-delà, le `Summarizer` produit un `ContextSummary` structuré (voir §7).

### 3.5 Absence de script d'orchestration

Aucun dispatcher Python n'impose l'ordre `exposure → smoothing → validation`. Le LLM voit les outils et les dépendances déclarées dans `catalogue.yaml`, et il choisit. Les garde-fous sont **passifs** : un tool retourne `{"erreur": "exposure_table absent"}` si le prérequis n'est pas là.

---

## 4. Tools node

Fichier : [agents/mortality/agents/tools_node.py](../agents/mortality/agents/tools_node.py)

### 4.1 Exécution

```python
# l.72-87
for tool_call in last_msg.tool_calls:
    args = json.loads(tool_call["args"])
    result = call_tool(
        tool_name   = tool_call["name"],
        function_name = args.get("function_name"),
        params      = args.get("params", {}),
        data        = local_data_store,
        df          = df,
    )
    ...
```

- Dispatch via `tools/tool_registry.call_tool()` ([`tool_registry.py:131`](../tools/tool_registry.py)).
- DataFrame chargé **ici** depuis `MemoryManager` (Parquet), jamais depuis le state (`l.58-64`).
- Pour chaque tool call : stockage dans `data_store` (`l.141-169`), log dans `_call_log` (`l.172-184`), création d'un `ToolMessage` (`l.200-203`).

### 4.2 Mode step-by-step

Si `step_by_step=True` (`l.93-118`) : avant chaque tool, émet `awaiting_approval`, attend sur `approval_event` ; le canvas peut approuver (`approval_event.set()`) ou annuler (`cancel_flag[0] = True`).

---

## 5. WriterAgent / WriterNode / ReportNode

Il existe **trois constructions distinctes** qui portent la notion d'agent "writer". Seule WriterNode est effectivement câblée dans le graphe.

### 5.1 WriterNode (actif)

Fichier : [agents/mortality/agents/writer_node.py](../agents/mortality/agents/writer_node.py)

- **Aucun appel LLM au niveau du nœud**.
- Appelle directement `run_pipeline.run(data_store, initial_request, output_path)` (`l.61`).
- Retourne un `AIMessage` avec le chemin du PDF et un flag `WRITE_DONE` pour que le master l'intercepte au tour suivant.

### 5.2 ReportNode (inactif dans le flow standard)

Fichier : [agents/mortality/agents/report_node.py](../agents/mortality/agents/report_node.py)

- Appelle `gpt-4o` avec `get_openai_tools()` et `tool_choice="auto"` (`l.71-76`).
- `max_tokens=4000`, pas de temperature spécifiée.
- **Pas branché par défaut** dans le graphe actuel (le nom du nœud registré est `writer` et il pointe sur `writer_node`).

### 5.3 WriterAgent legacy

Fichier : [agents/mortality/writer_agent.py](../agents/mortality/writer_agent.py)

Classe autonome avec une boucle `run_agent_loop()` generator (`l.47-256`). Appelle `gpt-4o` avec tool-calling (`l.103-107`). **Pas utilisée par LangGraph** — vestige de l'architecture précédente.

---

## 6. Pipeline de rapport (6 étapes)

Interface publique : `run_pipeline.run(data_store, initial_request, output_path, yaml_path) -> PipelineResult` ([agents/report/pipeline/run_pipeline.py:32-141](../agents/report/pipeline/run_pipeline.py)).

**Étapes séquentielles** :

### 6.1 Étape 01 — `load_plan` (déterministe, zéro LLM)

Fichier : [agents/report/pipeline/_01_load_plan.py](../agents/report/pipeline/_01_load_plan.py).

- Lit le YAML (`mortality_template.yaml` par défaut).
- Appelle `load_yaml_template.run()` qui résout **{{ placeholders }}** depuis `_PLACEHOLDER_MAP` (scalaires + dérivations + fallbacks).
- Post-enrichit le `context` avec des clés supplémentaires de `data_store` listées dans `_EXTRA_CONTEXT_KEYS` (scalaires + dicts actuariels comme `cox_regression`, `logit_regression`, `validation`, `benchmarking`, `diagnostics`).
- Construit un `ReportPlan` avec une `SectionPlan` par section du `processing_sequence`. Chaque `SectionPlan` a un `prompt` **déjà assemblé et autonome** contenant :
  - Rôle + contenu narratif YAML (purpose/word_count/tone)
  - Templates narratifs avec placeholders résolus
  - Liste des tableaux/graphiques/stats attendus
  - **Bloc « Données disponibles pour la rédaction »** injectant en JSON les scalaires et objets métier (Cox, logit, etc.), avec `_round_floats(4)` pour éviter les valeurs à 15 décimales.
  - Règles absolues ("ne cite que des chiffres présents", "omets les phrases si donnée manque").

### 6.2 Étape 02 — `validation_plan` (LLM gpt-4o, JSON mode)

Fichier : [agents/report/pipeline/_02_validation_plan.py](../agents/report/pipeline/_02_validation_plan.py).

- Prompt : envoie `plan.context` + statut de chaque section au LLM.
- Demande un JSON `{sections: [{section_id, valid, reason, missing_or_insufficient}]}`.
- Si `all_valid=False` : le pipeline retourne immédiatement `PipelineResult(status="need_data", need_data=...)` → le master renvoie au Builder au tour suivant.
- `max_tokens` et `temperature` : `NON TROUVÉ` dans le périmètre lu.

### 6.3 Étape 03 — `completion_plan` (RAG parallèle, pas de LLM de rédaction)

Fichier : [agents/report/pipeline/_03_completion_plan.py](../agents/report/pipeline/_03_completion_plan.py).

- Requête ChromaDB par section via `tools.build_pdf.search_exemplars.run()` (`l.89-111`).
- Queries prédéfinies dans `_SECTION_QUERIES` (`l.44-76`).
- Résultats filtrés par distance ≤ 1.2 (`l.32, 105`), tronqués à 600 chars, ajoutés au `sec.prompt` comme bloc `## Exemples de rédaction`.
- ThreadPoolExecutor max 4, timeout 20s par section. Non-bloquant.
- Sections éligibles au RAG : `{preamble, data_submission, construction, obs_vs_modeled, regulatory_positioning, conclusion}` (`l.36-39`).
- Modèle d'embeddings : `NON TROUVÉ` (délégué à ChromaDB).

### 6.4 Étape 04 — `redaction` (parallèle LLM + tools déterministes)

Fichier : [agents/report/pipeline/_04_redaction.py](../agents/report/pipeline/_04_redaction.py).

**Architecture par section** :
1. **Tools déterministes d'abord** (`_run_tables` + `_run_stats` + `_run_graphs`) :
   - Tables : `_hydrate_table_spec()` (`l.36-169`) injecte les `rows` depuis le contexte pour les IDs YAML connus (`table_construction`, `exposure_stats`, `death_stats`, `table_comparison` agrégé par classes de 5 ans, `mortality_table`).
   - Stats : mapping `_STAT_TYPE_BY_ID` (`cox_model → cox_proportional_hazards`, `logit_fit → logit_regression`, etc.) injecté avant `render_statistical_output()`.
   - Graphes : `_enrich_graph_context()` (`l.172-241`) dérive les dicts `{age: valeur}` attendus par `builder_plots` depuis `validation.ci_table`, `exposure_table`, `benchmarking.abatement_table`, `precedent_comparison.comparison_table`.

2. **LLM rédige ensuite** (`_call_llm_redaction`, `l.649-660`) :
   - `gpt-4o`, `temperature=0.4`, `max_tokens=1200`.
   - **System prompt `_SYSTEM_PROMPT_REDACTION`** (`l.465-549`, ~85 lignes) : charte de style, markup markdown autorisé, formules LaTeX obligatoires (`$…$` / `$$…$$`), virgule décimale FR, espace fine milliers, structure type section.
   - **User prompt** = prompt assemblé par étape 01 (enrichi par 03) + tableaux rendus en intégralité (pas de troncature) + liste des PNG générés + consignes finales ("cite les chiffres clés", "omets si manque", "n'invente pas").

3. **Validator traçabilité** (`_enforce_traceability`, `l.576-634`) : voir §7.

4. **Bypass LLM pour `annex`** (`_ANNEX_SECTION_IDS`, `l.379`) : une intro figée (`_ANNEX_INTRO`) est posée à la place du LLM, la table q_x complète est insérée par le renderer déterministe.

5. **Parallélisme** : ThreadPoolExecutor `max_workers=5` (protection 429 TPM). Snapshot read-only du `data_store` par worker. Écriture séquentielle à la fin pour préserver l'ordre.

6. **Cas multi-tables/multi-graphs par section** : la boucle d'écriture appelle `write_section` **N fois** pour qu'aucun élément ne soit écrasé (write_section consomme `_last_table_rows` / `_last_graph_path` singuliers).

### 6.5 Étape 05 — `assemble` (déterministe, ReportLab)

Fichier : [agents/report/pipeline/_05_assemble.py](../agents/report/pipeline/_05_assemble.py).

- Wrapper sur `tools.build_pdf.assemble_sections.run()`.
- Ordre fixe des sections : `preamble → data_submission → construction → analysis → conclusion → annex` (`assemble_sections.py:73-75`).
- Titre du rapport + bandeau portefeuille (période, années-personnes, décès) déduits de `template_context`.

### 6.6 Étape 06 — `validate_report` (LLM gpt-4o, JSON mode)

Fichier : [agents/report/pipeline/_06_validation.py](../agents/report/pipeline/_06_validation.py).

- **System prompt** (`l.141-144`) : « Tu es un réviseur qualité actuariel senior. Tu réponds UNIQUEMENT en JSON valide. »
- **User prompt** (`l.52-123`) : demande initiale + paramètres d'étude + **résumé ≤300 mots** de chaque section (pas le texte intégral). Liste critères anomalies mineures vs majeures.
- **Sortie JSON** : `{verdict: "ok"|"minor"|"major", summary, anomalies: [{severity, section_id, description, suggestion}]}`.
- **Retry ciblé** (`run_pipeline.py:114-127`) : si `verdict="minor"`, re-exécute étape 04 uniquement pour les sections KO (1 fois max).
- **Fallback** (`l.242-251`) : si LLM indisponible, `verdict="ok"` d'office, rapport livré.
- `max_tokens=1500`. `temperature` : `NON TROUVÉ`.

---

## 7. Validator de traçabilité

Fichier : [agents/report/pipeline/traceability.py](../agents/report/pipeline/traceability.py).

**Algorithme** :

1. **`extract_numbers(text)`** (`l.79-86`) : regex `_NUMBER_RE` (supporte milliers espacés `2 041 523`, virgule ou point décimal, `%` optionnel) → `list[float]`.
2. **`collect_numbers(data)`** (`l.91-122`) : parcours récursif dict/list/valeur, extrait tous les nombres présents comme référentiel.
3. **`_is_traceable(value, refs, rel_tol, abs_tol)`** (`l.131-147`) : une valeur est traçable si une référence correspond à :
   - ±tol exact
   - `value × 100` (fraction → pourcent)
   - `value / 100` (pourcent → fraction)
   - `value / 1000` (‰ → fraction).
   Pas de `value × 1000` (évite qu'un HR=2.14 matche 2140).
4. **Tolérances** : `rel_tol=0.02` (±2 %), `abs_tol=1e-4` (plancher pour arrondis d'affichage).
5. **`_BAD_TOKEN_RE`** (`l.151-156`) : détecte `[donnée non disponible]`, `[key]` non substitué, `{{ key }}` non substitué.
6. **`_WHITELIST_EXACT`** (`l.161-166`) : `{95, 99, 100, 0.01, 0.05, 0.1, 0.95, 0.99, 1.96, 2.576, 1000}`. Match exact (< 1e-9) — pas de tolérance — pour éviter qu'un hallucination `3,99` ne matche `4.0`.
7. **`validate_section()`** retourne `TraceabilityResult(ok, numbers_cited, untraceable, bad_tokens)` + `feedback_for_retry()` qui construit un message chirurgical pour le LLM.

**Branchement dans l'étape 04** (`_enforce_traceability`, `l.576-634`) :
- Construit les références à partir de `all_tables + data_store["summary"|"cox_regression"|"logit_regression"|"validation"|"benchmarking"|"diagnostics"|"precedent_comparison"|"exposure_table"|"smoothed_table"|"qx_table"|...]`.
- Si `result.ok` : on garde le texte.
- Sinon : retry 1× avec le feedback. Si le retry dégrade (plus d'`untraceable`) → on garde l'original.

---

## 8. Template YAML et rendering

### 8.1 Structure du template

Fichier : [knowledge_base/report_template/mortality_template.yaml](../knowledge_base/report_template/mortality_template.yaml).

- `report_template.id`, `version`, etc.
- **`processing_sequence`** : liste ordonnée des sections (preamble, data_submission, construction, analysis, conclusion, annex) avec `inputs_required`, `tool_calls`, `validation_checkpoints`.
- **`sections`** : liste parallèle avec le contenu narratif (subsections, `narrative_elements` avec `{{ placeholders }}`, specs de tables/graphiques/stats). Indexée par `section_id` / `subsection_id`.
- **`inputs`** : dictionnaire de placeholders attendus (valeurs `null` dans le template).

Les `tool_call` sont des dicts `{tool_call: render_table_from_spec | generate_graph_from_spec | render_statistical_output, inputs: {spec: {...}}}`. Extraits récursivement par `_extract_section_specs()` ([load_yaml_template.py:157-217](../tools/build_pdf/load_yaml_template.py)) avec déduplication par `id` (évite qu'un spec partagé entre subsections soit rendu plusieurs fois).

### 8.2 `_PLACEHOLDER_MAP`

[tools/build_pdf/load_yaml_template.py:83-141](../tools/build_pdf/load_yaml_template.py) — ~41 entrées mappant placeholder YAML → `(source_type, source_path)` :
- `"study_plan"` : lecture directe dans le study_plan
- `"data_store_multi"` : plusieurs chemins séparés par `|`, premier trouvé gagne
- `"derived"` : calculé par `_resolve_derived()` (ex. `deaths_by_age()`, `mean_age_cohort()`, `modeled_deaths_by_age()`)

### 8.3 Rendering déterministe

- **`table_renderer.render_table_from_spec(spec, data)`** ([tools/build_pdf/table_renderer.py](../tools/build_pdf/table_renderer.py)) :
  - `_normalize_columns()` accepte colonnes string ou dict.
  - `_resolve_inline()` substitue les `{{ }}` dans les labels de colonnes (sinon le placeholder brut apparaît dans le PDF).
  - Trois modes de construction de lignes : `dynamic`/`age_indexed` (construit via `_build_age_rows` depuis les data series du contexte), liste statique (`_build_static_rows`), ou vide.
  - Rend un objet ReportLab `Table` + HTML de fallback.

- **`generate_graph_from_spec(spec, data)`** ([tools/graphs/graph_from_spec.py](../tools/graphs/graph_from_spec.py)) :
  - Dispatcher `_DISPATCH` : mappe `spec.id` (ex. `graph_exposure_by_age`) → `chart_name` (ex. `exposure`) implémenté dans `builder_plots.py`.
  - `_flatten_for_builder_plots()` aplatit les structures imbriquées (`benchmarking.abatement_table` → `abatement_table` au root).
  - Si dispatch échoue et si le spec a `series: [...]`, fallback matplotlib générique.
  - Retourne un chemin PNG ou `""` en cas d'échec.

- **`assemble_sections.py`** : itère `_SECTION_ORDER`, insère pour chaque section le titre + texte narratif (markdown → ReportLab) + tableaux + PNG.

---

## 9. Outils actuariels (data_store schema)

Fichiers : [tools/builder/*.py](../tools/builder/).

Tous les tools suivent `run(data, params)` ou `run(df, params)` et retournent un dict JSON-safe. Dépendances déclarées dans [catalogue.yaml](../catalogue.yaml) mais **non imposées au runtime** (garde-fou passif : le tool retourne `{erreur: ...}` si prérequis manque).

| Tool | Input requis | Clé(s) écrite(s) dans data_store | Dépend de |
|---|---|---|---|
| `builder.exposure` | DataFrame (colonnes `date_naissance`, `date_entree`, `date_sortie`, `cause_sortie`) | `exposure_table`, `age_min`, `age_max`, `total_exposure`, `total_deaths`, `lignes_exclues` | — |
| `builder.crude_rates` | `exposure_table` | `qx_table`, `method` | exposure |
| `builder.smoothing` | `qx_table` | `smoothed_table`, `method`, `n_non_monotone`, `aic_poisson`, `bic_poisson` | crude_rates |
| `builder.diagnostics` | `exposure_table` | `diagnostics.{regime, pct_low_credibility, n_low, recommendation}` ou comparateur smoothers | exposure |
| `builder.validation` | `exposure_table` (+ `smoothed_table`) | `validation.{ci_table, alpha}` ou `validation.{chi2_stat, p_value, df}` | exposure (+smoothing) |
| `builder.benchmarking` | `exposure_table` (+ `smoothed_table`) | `benchmarking.{abatement_table, smr_global, reference_name}` | exposure (+smoothing) |
| `builder.cox_regression` | `exposure_table` + DataFrame | `cox_regression.{hazard_ratio, ci_lower_95, ci_upper_95, cox_pvalue, deaths_male, deaths_female, …}` | exposure |
| `builder.logit_regression` | `smoothed_table` + table réglementaire | `logit_regression.{slope_alpha, intercept_beta, r_squared, scatter_data, …}` | smoothing + benchmarking |
| `builder.precedent_comparison` | `smoothed_table` + table antérieure | `precedent_comparison.{comparison_table, drift_global, ages_derive_forte}` | smoothing |
| `statistical_analysis.portfolio_summary` | DataFrame | `summary.{nb_contrats, nb_deces, exposition_totale_pa, age_moyen, ratio_h_f, qualite_donnees, warnings}` | — |

**Qualité gate bloquante** (une seule observée) : `smoothing` retourne erreur si `n_non_monotone > 0` après âge 40 ([smoothing.py:82-87](../tools/builder/smoothing.py)). Le LLM doit relancer avec un `lambda` plus élevé.

**Garde-fou bloquant absent** : il n'y a **pas** de refus proactif (ex : refuser `benchmarking` si `smoothing` n'a pas été fait). Le LLM peut théoriquement appeler les tools dans un ordre incohérent.

---

## 10. Persistance et mémoire

Fichiers : [session/session_state.py](../session/session_state.py), [session/memory_manager.py](../session/memory_manager.py), [session/dataset_store.py](../session/dataset_store.py), [session/summarizer.py](../session/summarizer.py).

### 10.1 Quatre couches de mémoire

| Couche | Contenu | Persistance | Cycle de vie |
|---|---|---|---|
| **Working memory** | AgentState LangGraph complet | RAM (MemorySaver) | Jusqu'au redémarrage du process |
| **Business memory** | `SessionState` Pydantic (study_plan, column_mapping, tool_results, context_summary) | `session/data/{session_id}_state.json` | Permanente |
| **Dataset memory** | DataFrame du portefeuille | `session/data/artifacts/{session_id}_dataset.parquet` (écrit une seule fois) | Permanente |
| **Conversation memory** | Messages LangGraph (reducer `add_messages`) | État LangGraph, compactée à 15+ messages | Jusqu'au redémarrage |

### 10.2 Cycle de vie d'un tour

Extrait de `stream_agent()` ([graph.py:267-345](../agents/mortality/agents/graph.py)) :

1. **Début de tour** : `mm = get_memory_manager(session_id)` charge `SessionState` (ou crée un neuf).
2. `data_store` initialisé depuis `SessionState.to_data_store()` (hydrate study_plan + tool_results whitelistés).
3. Historique Dash → messages LangChain.
4. Si `len(messages) > COMPACT_THRESHOLD (15)` : `Summarizer.summarize(old_messages)` produit un `ContextSummary` qui remplace les vieux messages. Le `ContextSummary` contient : `decisions_prises`, `ambiguites_levees`, `hypotheses_actives`, `objets_construits` (clés data_store), `donnees_manquantes`, `prochaine_etape`.
5. LangGraph invoqué avec ce state.
6. **Fin de tour** : `mm.after_turn(final_data_store, lc_messages)` → `SessionState.update_from_data_store()` (whitelist `_TOOL_RESULT_KEYS`) + `SessionState.save()`.

### 10.3 `_TOOL_RESULT_KEYS` (whitelist)

[session_state.py:197-204](../session/session_state.py) — clés du `data_store` effectivement persistées sur disque entre tours :

```python
{
    "exposure_table", "qx_table", "smoothed_table", "diagnostics",
    "validation", "benchmarking", "certification_report", "summary",
    "cox_regression", "logit_regression", "series",
    "total_deaths", "total_exposure_years", "cohort_min_age", "cohort_max_age",
    "age_min", "age_max", "total_exposure", "n_insured",
}
```

Tout ce qui est dans `data_store` mais **hors de cette liste** est perdu au prochain tour.

---

## 11. Points de décision LLM vs déterministe

Récapitulatif par composant :

| Décision | Par qui | Commentaire |
|---|---|---|
| Classifier l'intent de l'utilisateur | LLM (master) | `_classify_intent`, max_tokens=200 |
| Désambiguïser les colonnes CSV | règles + LLM si ambigu | `suggest_column_mapping` |
| Choisir quels tools actuariels appeler et dans quel ordre | **LLM (builder) via function-calling** | Aucun script |
| Enchaîner les étapes 01→06 du pipeline | déterministe | `run_pipeline.run` |
| Résoudre les placeholders du template YAML | déterministe | `_PLACEHOLDER_MAP` + `_resolve_derived` |
| Hydrater les spec tables avec des lignes de données | déterministe | `_hydrate_table_spec` (mortalité-spécifique) |
| Rendu des tables / graphes | déterministe | ReportLab / matplotlib |
| Juger si les données sont "suffisantes" (étape 02) | LLM gpt-4o JSON mode | Peut renvoyer `need_data` → retour au builder |
| Rédiger chaque section du rapport | LLM gpt-4o, `temperature=0.4`, `max_tokens=1200` | 1 appel par section, parallèles (max 5) |
| Vérifier la traçabilité des chiffres cités | déterministe (regex + set matching) | `traceability.validate_section` |
| Réparer les chiffres non traçables | LLM (retry ciblé, 1×) | Avec feedback chirurgical |
| Juger si le rapport final est acceptable | LLM gpt-4o JSON mode | Verdict ok/minor/major, peut déclencher retry 04 |

---

## 12. Choses volontairement absentes / zones d'incertitude

Ce que l'enquête n'a **pas** tranché — angles d'attaque potentiels pour une critique :

1. **Contenu exact des system prompts `loader.py`** pour master et builder (chargés dynamiquement depuis des `.md` dans `agents/*/agent_instructions/` qui n'ont pas été lus dans le périmètre de cette enquête).
2. **`temperature`** des appels LLM étape 02, 06, et ReportNode : non spécifiée dans le code → défaut OpenAI 1.0 (potentiellement trop élevé pour un livrable de certification).
3. **Modèle d'embeddings** utilisé par ChromaDB pour le RAG (étape 03) : délégué, pas dans le code du pipeline.
4. **Logique d'auto-régénération de `catalogue.yaml`** depuis les docstrings `TOOL CONTRACT` : existe mais pas analysée en profondeur.
5. **Conditions exactes déclenchant WRITE_DONE / BUILD_DONE / NEED_DATA** : flags posés par le builder/writer, interceptés par le master ; le contrat exact des messages AIMessage n'a pas été documenté verbatim.
6. **Comportement sur CSV mal formé** : la désambiguïsation peut-elle boucler ou l'agent abandonne-t-il proprement ? Non testé.
7. **Gestion des tours LLM en erreur** : si `_call_llm_redaction` retourne `""` pour une section, la section reste vide dans le PDF et l'étape 06 doit le détecter — mais elle n'a accès qu'à un résumé 300 mots, pas aux flags internes.

---

## 13. Index rapide des fichiers

Pour aller vite :

| Domaine | Fichier |
|---|---|
| UI | [canvas_app.py](../canvas_app.py) |
| Graphe LangGraph | [agents/mortality/agents/graph.py](../agents/mortality/agents/graph.py) |
| État | [agents/mortality/agents/state.py](../agents/mortality/agents/state.py) |
| MasterAgent | [agents/mortality/agents/master_node.py](../agents/mortality/agents/master_node.py) |
| BuilderAgent | [agents/mortality/agents/builder_node.py](../agents/mortality/agents/builder_node.py) |
| Tools node | [agents/mortality/agents/tools_node.py](../agents/mortality/agents/tools_node.py) |
| Désambiguïsation | [agents/master/disambiguation.py](../agents/master/disambiguation.py) |
| WriterNode | [agents/mortality/agents/writer_node.py](../agents/mortality/agents/writer_node.py) |
| Pipeline report | [agents/report/pipeline/](../agents/report/pipeline/) |
| Traceability validator | [agents/report/pipeline/traceability.py](../agents/report/pipeline/traceability.py) |
| Template YAML | [knowledge_base/report_template/mortality_template.yaml](../knowledge_base/report_template/mortality_template.yaml) |
| Resolver YAML | [tools/build_pdf/load_yaml_template.py](../tools/build_pdf/load_yaml_template.py) |
| Renderer tables | [tools/build_pdf/table_renderer.py](../tools/build_pdf/table_renderer.py) |
| Renderer graphs | [tools/graphs/graph_from_spec.py](../tools/graphs/graph_from_spec.py) |
| Builder plots | [tools/graphs/builder_plots.py](../tools/graphs/builder_plots.py) |
| Assemble PDF | [tools/build_pdf/assemble_sections.py](../tools/build_pdf/assemble_sections.py) |
| Registry tools | [tools/tool_registry.py](../tools/tool_registry.py) |
| Catalogue | [catalogue.yaml](../catalogue.yaml) |
| SessionState | [session/session_state.py](../session/session_state.py) |
| MemoryManager | [session/memory_manager.py](../session/memory_manager.py) |
| Summarizer | [session/summarizer.py](../session/summarizer.py) |
| DatasetStore | [session/dataset_store.py](../session/dataset_store.py) |
| Outils actuariels | [tools/builder/](../tools/builder/) |
