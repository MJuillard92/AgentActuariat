# session/ — Couche mémoire de l'agent actuariel

Ce module implémente les **4 couches de mémoire** de la plateforme.
Il est indépendant de LangGraph et peut être utilisé seul (CLI, tests).

---

## Architecture des 4 couches

```
┌─────────────────────────────────────────────────────────────────┐
│  BusinessMemory  →  SessionState  (session/data/{id}_state.json)│
│  WorkingMemory   →  AgentState LangGraph (MemorySaver — RAM)    │
│  ConversationMem →  messages tronqués + ContextSummary (system) │
│  AuditLog        →  session/data/{id}_audit.json (append-only)  │
└─────────────────────────────────────────────────────────────────┘
```

| Couche | Support | Durée de vie | Responsable |
|---|---|---|---|
| BusinessMemory | JSON sur disque | Permanente | `MemoryManager` |
| WorkingMemory | RAM (MemorySaver LangGraph) | Session process | `graph.py` |
| ConversationMemory | Messages tronqués + résumé injecté | Tour courant | `Summarizer` |
| AuditLog | JSON append-only sur disque | Permanente | `canvas_app.py` |

---

## Fichiers du module

### `session_state.py` — Modèles Pydantic

Schémas canoniques pour la persistance.

```
SessionState
├── session_id: str
├── study_plan: StudyPlan | None
├── dataset_meta: DatasetMeta | None
├── context_summary: ContextSummary | None
├── column_mapping: dict[str, str]
├── csv_filename: str | None
├── tool_results: dict[str, Any]
└── created_at / updated_at: str
```

**`StudyPlan`** — paramètres de l'étude actuarielle :
```
observation_start_date, observation_end_date, cohort_min_age,
cohort_max_age, smoothing_algorithm, baseline_regulatory_table,
study_objective, observation_period_years
+ is_complete() → bool
```

**`DatasetMeta`** — référence stable vers l'artefact Parquet :
```
path: str           # chemin absolu vers session/data/artifacts/{id}_dataset.parquet
sha256: str         # 12 premiers chars du hash SHA-256 du CSV
n_rows, n_cols: int
columns: list[str]
created_at: str
```

**`ContextSummary`** — résumé structuré de la conversation (produit par GPT-4o) :
```
decisions_prises: list[str]
ambiguites_levees: list[str]
hypotheses_actives: list[str]
objets_construits: list[str]
donnees_manquantes: list[str]
prochaine_etape: str
+ to_system_block() → str   # injecté dans le system prompt
```

---

### `dataset_store.py` — Artefact Parquet (écriture unique)

Le DataFrame initial est sérialisé **une seule fois** en Parquet.
Toutes les opérations suivantes lisent depuis ce fichier — le DataFrame
n'est jamais re-sérialisé dans l'AgentState LangGraph.

```
DatasetStore
├── store(session_id, df) → DatasetMeta   # idempotent : ne réécrit pas si déjà présent
├── load(meta)            → pd.DataFrame  # charge depuis meta.path
├── load_by_session(id)   → pd.DataFrame | None
└── exists(session_id)    → bool
```

Chemins :
```
session/data/artifacts/{session_id}_dataset.parquet
```

**Règle d'idempotence** : si le fichier Parquet existe déjà pour cette session,
`store()` reconstruit le `DatasetMeta` depuis le fichier existant sans réécrire.

---

### `summarizer.py` — Compaction de la conversation

Déclenché automatiquement quand la conversation dépasse 15 messages.

```
Summarizer
├── should_compact(messages) → bool   # True si len > COMPACT_THRESHOLD (15)
├── compact(messages, data_store) → ContextSummary
│     └── appel GPT-4o (JSON mode) → ContextSummary
└── trim_messages(messages) → list    # garde les 5 derniers messages verbatim
```

En cas d'échec GPT-4o, le résumé de fallback liste les clés calculées du data_store.

---

### `memory_manager.py` — Orchestrateur principal

Interface publique utilisée par `graph.py` et `canvas_app.py`.

```
MemoryManager(session_id)
├── load()                              → self      # charge depuis disque
├── save()                              → None      # persiste sur disque
├── to_data_store()                     → dict      # hydrate l'AgentState LangGraph
├── after_turn(data_store, messages)    → None      # persist + compaction si nécessaire
├── get_context_block()                 → str       # bloc system prompt (résumé + dataset meta)
├── load_dataframe()                    → df | None # charge le DataFrame depuis Parquet
├── register_dataset(df, csv_filename)  → DatasetMeta  # écriture unique idempotente
└── trim_messages(messages)             → list      # tronque si résumé disponible
```

**Cycle de vie typique dans `graph.py`** :
```python
mm = MemoryManager(session_id)
mm.load()
data_store = mm.to_data_store()     # 1. Hydrater depuis état persisté

# ... exécution LangGraph ...

mm.after_turn(data_store, messages)  # 2. Persister + compacter si besoin
```

---

## Structure des fichiers runtime

```
session/
├── data/                          # gitignored — données runtime
│   ├── {session_id}_state.json    # SessionState sérialisé (BusinessMemory)
│   ├── {session_id}_audit.json    # AuditLog append-only (canvas_app.py)
│   └── artifacts/
│       └── {session_id}_dataset.parquet   # DataFrame initial (écriture unique)
├── __init__.py
├── session_state.py
├── dataset_store.py
├── summarizer.py
├── memory_manager.py
└── README.md
```

Le dossier `session/data/` est dans `.gitignore` — il contient uniquement
des données runtime locales (données utilisateur sensibles, artefacts Parquet).

---

## Flux de données

```
Upload CSV
    │
    ▼
canvas_app.py : mm.register_dataset(df, filename)
    │   └── DatasetStore.store() → session/data/artifacts/{id}.parquet
    │   └── SessionState.dataset_meta = DatasetMeta(...)
    │   └── mm.save() → session/data/{id}_state.json
    │
    ▼
stream_agent() (graph.py)
    │   mm.load() → hydrate data_store avec dataset_meta, study_plan, tool_results
    │   df = mm.load_dataframe()  ← lu depuis Parquet (jamais re-sérialisé)
    │
    ▼
builder_node / tools_node
    │   df rechargé via MemoryManager(dataset_ref).load_dataframe()
    │   résultats ajoutés à data_store
    │
    ▼
mm.after_turn(data_store, messages)
    │   SessionState.update_from_data_store(data_store)
    │   si len(messages) > 15 → Summarizer.compact() → ContextSummary
    │   mm.save() → session/data/{id}_state.json
```
