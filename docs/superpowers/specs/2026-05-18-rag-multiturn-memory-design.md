# RAG multi-turn memory — Design

> Brainstorming: 2026-05-18 — Marc Juillard + Claude
> Scope: RAG only (autres agents inchangés)
> Pattern: LangChain-style buffer + summary + vectorstore (option (b))

---

## Context

Le RAG actuel (agent top-level commité dans `4c377e1`) est **stateless** — chaque appel à `run_pipeline.run()` ne voit que la dernière question utilisateur, sans contexte conversationnel.

**Limites observées** :
- `"compare-les"` après deux tours sur Whittaker et Kaplan-Meier → retriever paumé, réponse hors-sujet.
- `"et pour les femmes ?"` après discussion sur taux bruts → manque le contexte.
- Toute référence anaphorique (`les`, `ça`, `cette méthode`, `son`, `et pour`) est cassée.

**Décision business** : on s'apprête à brancher un agent **provisionnement non-vie** (peer de mortality/report/rag). C'est le bon moment pour solidifier le RAG conversationnel — il sera l'unique agent purement conversationnel du projet (les autres sont state-driven).

**Périmètre confirmé** : seul le RAG évolue. Master/Builder/Writer/Mortality/Report restent sur leurs patterns actuels (state machines + 20 derniers messages pour le tool-calling Builder).

---

## Architecture overview

Le RAG reçoit 3 niveaux de mémoire conversationnelle, **tous consommés par le `query_rewriter`** (single source of truth pour la gestion du contexte multi-turn). Le `answer_generator` reste stateless.

```
Conversation user
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ RAGMemoryStore (per session, RAM-only, lazy rebuild)            │
│   ┌──────────────────┬───────────────────┬───────────────────┐  │
│   │ Niveau 1 BUFFER  │ Niveau 2 SUMMARY  │ Niveau 3 VECTOR   │  │
│   │ 4 derniers tours │ Pydantic compact  │ FAISS embeddings  │  │
│   │ verbatim         │ (topics, focus,   │ de tous les Q/A   │  │
│   │                  │  facts, citations)│ MiniLM-384        │  │
│   └──────────────────┴───────────────────┴───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│query_rewriter│ ← buffer + summary + vectorstore_top_k
│   (nano)     │ → produit une requête self-contained
└──────┬───────┘
       │
       ▼
┌──────────────┐
│search_doctrine│ ← requête self-contained (stateless)
│  (FAISS+BM25)│ → chunks doctrine
└──────┬───────┘
       │
       ▼
┌──────────────┐
│answer_generator│ ← original_query + chunks (INCHANGÉ)
│    (mini)    │ → réponse rédigée avec citations [Dxx.yy]
└──────────────┘
```

**Principe directeur** : `query_rewriter` est la seule porte d'entrée pour la conversation history. Si quelque chose lui échappe, ajouter du contexte au générateur ne le rattrapera pas (les chunks sont déjà fixés par la requête reformulée).

---

## Les 3 niveaux de mémoire

### Niveau 1 — Buffer (verbatim)

- **Contenu** : N derniers tours `(user_question, rag_answer)` mot pour mot
- **Taille** : `BUFFER_SIZE = 4` (soit 8 messages user+ai)
- **Usage** : résoudre les anaphores immédiates ("compare-les" sur N-1, N-2)
- **Coût prompt** : ~1500 tokens max au rewriter

### Niveau 2 — Summary (résumé structuré)

- **Contenu** : `RAGSummary` Pydantic avec 4 champs :
  ```python
  topics_covered:    list[str]   # ["Whittaker-Henderson", "TH 00-02", "chi-2"]
  user_focus:        str         # "comprendre méthodes lissage et validation"
  key_facts_stated:  list[str]   # ["paramètre h optimisé par CV", ...]
  citations_used:    list[str]   # ["D03.02", "D03.04", ...]
  ```
- **Déclencheur** : update synchrone à la fin de `run_pipeline.run()` si `len(complete_history) > SUMMARY_TRIGGER` (= 10 tours)
- **Mode** : incrémental — on fournit le summary existant + nouveaux tours au LLM nano, qui produit le summary mis à jour. Évite la dérive.
- **Usage** : injecté dans le prompt du rewriter pour donner le thème de fond
- **Coût prompt** : ~200 tokens max

### Niveau 3 — Vectorstore (FAISS embeddings)

- **Contenu** : chaque tour `(user_q, rag_answer)` embedded avec MiniLM-384 (modèle déjà utilisé par `search_doctrine`)
- **Index** : FAISS local par session, en RAM
- **Retrieval** : `VECTORSTORE_TOP_K = 3` Q/A passés sémantiquement similaires à la nouvelle question, filtrés par `VECTORSTORE_MIN_SCORE = 0.7`
- **Usage** : ramener au rewriter les Q/A pertinents éloignés du buffer (conversations longues > 20 tours)
- **Cas typique** : T1 sur TH 00-02 → 30 tours sur autres sujets → T31 "et pour la version 2005 ?" → vectorstore remonte T1, rewriter produit "TGH 05 TGF 05"

---

## Storage architecture

**Décision : RAM-only avec lazy rebuild depuis `history`**.

### Justification

- L'historique des messages est déjà la source de vérité côté **canvas (browser)** — envoyé au serveur à chaque tour.
- `SessionState` (disque) ne stocke pas les messages, seulement le métier (study_plan, tool_results, etc.).
- LangGraph `MemorySaver` est déjà RAM-only pour l'`AgentState`.
- Cold start après restart Flask : ~1s pour re-embedder 20 Q/A (négligeable).
- Coût de la persistance disque : ~300 lignes (sérialisation FAISS, fichiers, sync, cleanup) vs ~100 RAM-only.

### Pattern d'accès

```python
class RAGMemoryStore:
    _cache: dict[str, "RAGMemoryStore"] = {}  # module-level, par session_id

    @classmethod
    def for_session(cls, session_id: str, history: list) -> "RAGMemoryStore":
        store = cls._cache.get(session_id)
        if store is None:
            store = cls(session_id)
            store._rebuild_from_history(history)  # cold start
            cls._cache[session_id] = store
        return store
```

### Cycle de vie

| Événement | Action |
|---|---|
| Premier message d'une session | Cache miss → store vide créé → `_rebuild_from_history([])` no-op |
| Tour N suivant | Cache hit → buffer/summary/vectorstore prêts |
| Restart Flask | Cache vidé. Au tour suivant, `_rebuild_from_history(history)` reconstruit buffer + vectorstore depuis le full history envoyé par canvas |
| Session abandonnée 1h | Cache reste en RAM jusqu'à expiration (TTL optionnel, pas v1) |

---

## query_rewriter — Modifications

### Signature

```python
def rewrite(
    query: str,
    buffer: list[RAGTurn] | None = None,
    summary: RAGSummary | None = None,
    vectorstore_hits: list[RAGTurn] | None = None,
) -> str:
    ...
```

### Construction du prompt user

```
[Conversation récente]
T-2 user: C'est quoi Whittaker-Henderson ?
T-2 assistant: ...méthode de lissage [D03.02]...
T-1 user: Et Kaplan-Meier ?
T-1 assistant: ...estimateur non paramétrique [D02.01]...

[Résumé contexte antérieur]
Topics couverts : TH 00-02 régl., Whittaker, Kaplan-Meier
User focus : méthodes lissage et estimation

[Échanges passés pertinents]
(T-7) user: "Différence taux brut vs lissé ?"
(T-7) assistant: "...le taux brut estime q_x par l'observation..."

[Nouvelle question]
compare-les

Reformule la nouvelle question en requête de recherche self-contained.
```

### `should_rewrite()` étendu

```python
_ANAPHORA_PATTERNS = (
    " les ", " ça", " ca ", "cette ", "celle", "celui",
    "et pour", "et avec", "et sur", "et ", "compare",
    "leur ", "leurs ", " son ", " sa ", " ses ",
)
_INTERROGATIVE_MARKERS = ("?", "qu'est", "comment", "pourquoi", "explique",
                          "donne-moi", "c'est quoi", "différence")

def should_rewrite(query: str, buffer_size: int = 0) -> bool:
    if not query:
        return False
    lower = f" {query.lower()} "

    # PRIORITÉ ABSOLUE : anaphore + contexte dispo → FORCER
    if buffer_size > 0 and any(p in lower for p in _ANAPHORA_PATTERNS):
        return True

    # Query déjà déclarative (pas de marqueur interrogatif) → skip
    if not any(m in lower for m in _INTERROGATIVE_MARKERS):
        return False

    # Courte + terme du corpus → skip (heuristique single-turn)
    if len(query) <= _SHORT_QUERY_THRESHOLD:
        lexicon = get_lexicon()  # auto-derived from meta.json
        return not any(term in lower for term in lexicon)

    return True
```

### `_TECHNICAL_TERMS` remplacé par lexique auto-derivé

```python
# agents/rag/pipeline/_corpus_lexicon.py
def build_lexicon_from_meta() -> set[str]:
    """Construit le lexique depuis le meta.json de l'index FAISS doctrine."""
    with _META_PATH.open() as f:
        meta = json.load(f)
    terms: set[str] = set()
    for chunk in meta.get("chunks", []):
        # section_titles, doc_ids, section_ids, tags
        ...
    return terms

_LEXICON_CACHE: set[str] | None = None
_LEXICON_MTIME: float = 0.0

def get_lexicon() -> set[str]:
    """Cache module-level avec check mtime — recharge si meta.json modifié."""
    ...
```

**Bénéfice** : tout enrichissement du corpus (re-ingest) met automatiquement à jour le lexique. Plus de liste hardcodée à maintenir.

---

## answer_generator — INCHANGÉ

Pas de modification de code, de prompt, ni de tests. Toute la mécanique multi-turn est encapsulée dans `rewriter + memory_store`.

Bénéfice secondaire : si on veut désactiver le multi-turn (debug, A/B test), il suffit de ne pas appeler le memory_store dans le rewriter. Pipeline marche identiquement.

---

## Sizing parameters

| Paramètre | Valeur | Justification |
|---|:-:|---|
| `BUFFER_SIZE` | **4** tours (8 messages) | Couvre anaphores immédiates sans saturer prompt nano |
| `SUMMARY_TRIGGER` | **10** tours (20 messages) | En dessous, buffer suffit |
| `VECTORSTORE_TOP_K` | **3** | Plus = bruit |
| `VECTORSTORE_MIN_SCORE` | **0.7** | Évite remontée de Q/A faiblement liées |

### Coût LLM par message utilisateur

| Conversation | Rewriter (nano) | Generator (mini) | Summary (nano, occasionnel) | Total |
|---|---|---|---|---|
| 1-3 tours | $0.00005 | $0.0008 | — | **~$0.001** |
| 4-10 tours | $0.00007 | $0.0008 | — | **~$0.001** |
| 11-30 tours | $0.00008 | $0.0008 | $0.00002 / ~5 tours | **~$0.001** |

Surcoût multi-turn vs single-turn : **< 5 %**.

---

## Summary — Stratégie de génération

**Mode synchrone** dans `append_turn()` : si le seuil est atteint, on appelle le LLM nano avant de retourner la réponse à l'user. Pénalité ~1s sur ~1 tour sur 5. Pas de thread async pour la v1 (simplicité).

### Prompt du summarizer (LLM nano `rag.summarizer`)

```
Tu es archiviste conversationnel. Maintiens un résumé STRUCTURÉ JSON
de la conversation RAG actuarielle.

Résumé existant (à enrichir, pas à écraser) :
{existing_summary_json or "vide"}

Nouveaux tours à intégrer :
{old_turns formatés}

Produis le résumé mis à jour au format JSON strict :
{
  "topics_covered":     ["..."],
  "user_focus":         "...",
  "key_facts_stated":   ["..."],
  "citations_used":     ["..."],
  "n_turns_summarized": <int>
}

Règles :
- max 8 topics_covered (dédupliquer)
- max 10 key_facts_stated
- user_focus en 1 phrase, max 15 mots
- Conserver l'existant SAUF contradiction
```

Config : `rag.summarizer = gpt-5.4-nano`, temp 0.0, max 500 tok, JSON mode.

---

## run_pipeline.run() — Orchestration finale

7 étapes (au lieu de 5/6) :

```python
def run(state, verify=False):
    # RAG.0 — NOUVEAU : Hydrater la mémoire conversationnelle
    session_id = state.get("session_id")
    history    = state.get("messages") or []
    memory     = RAGMemoryStore.for_session(session_id, history)

    # RAG.1 — Extract user query (inchangé)
    user_query = _extract_user_query(history)

    # RAG.2 — Normalize (inchangé)
    normalized = query_normalizer.normalize(user_query)

    # RAG.3 — Rewrite (ÉTENDU)
    buffer = memory.get_buffer(n=BUFFER_SIZE)
    if query_rewriter.should_rewrite(normalized, buffer_size=len(buffer)):
        rewritten = query_rewriter.rewrite(
            normalized,
            buffer=buffer,
            summary=memory.get_summary(),
            vectorstore_hits=memory.retrieve_similar(
                normalized, k=VECTORSTORE_TOP_K, min_score=VECTORSTORE_MIN_SCORE,
            ),
        )
    else:
        rewritten = normalized

    # RAG.4 — Retrieve (inchangé)
    hits = search_doctrine.run(None, {"query": rewritten, "k": 5})

    # RAG.5 — Generate (inchangé)
    answer = answer_generator.generate(user_query, hits.get("results") or [])

    # RAG.6 — Verify (optionnel, inchangé)
    if verify and chunks:
        ok, reason = grounding_check.verify(answer, chunks)

    # RAG.7 — NOUVEAU : Persister le tour en mémoire (synchrone)
    memory.append_turn(
        user_q=user_query,
        rag_answer=answer,
        sources=hits.get("results") or [],
    )
    # append_turn() : buffer fifo + embedding vectorstore + trigger summary si seuil

    return {"answer": answer, "sources": ..., "stage_events": [...]}
```

### Stage events émis (UI internal agent)

```
RAG.0 — Mémoire conversationnelle hydratée (buffer=N, summary=oui/non)
RAG.1 — Question extraite
RAG.2 — Normalisation typos
RAG.3 — Reformulation avec contexte (sources utilisées : buffer/summary/vectorstore)
RAG.4 — Retrieval hybride (n chunks)
RAG.5 — Synthèse rédigée
RAG.6 — Self-check (si verify=True)
RAG.7 — Mémoire mise à jour (summary regénéré ? oui/non)
```

---

## Path pending (`method_choices.answer_question_via_doctrine`)

**Compatibilité automatique** : `answer_question_via_doctrine` appelle déjà `run_pipeline.run()`. Le multi-turn est inherit sans modification.

**Subtilité** : il faut propager `session_id` + `history` complet dans le `fake_state` :

```python
def answer_question_via_doctrine(last_text, data_store, pending=None):
    # ...
    fake_state = {
        "messages":   data_store.get("_history") or [HumanMessage(content=last_text)],
        "session_id": data_store.get("_session_id"),
    }
    result = _run_rag(fake_state, verify=False)
```

→ nécessite que `stream_agent()` stocke `_history` + `_session_id` dans le `data_store`.

**Bénéfice** : mémoire partagée entre les deux paths (0.c pending et 0.e normal). Même `session_id` ⇒ même `RAGMemoryStore`.

---

## Tests strategy

### Unit (mocks LLM, ~20 tests)

- `test_rag_memory_store.py`
  - buffer ring-fifo respecte `BUFFER_SIZE`
  - lazy rebuild from history reconstruit buffer + vectorstore correctement
  - vectorstore add/retrieve avec filtre min_score
  - summary trigger respecte `SUMMARY_TRIGGER`
  - cache module-level partage le store entre appels même session_id

- `test_rag_query_rewriter_multiturn.py`
  - anaphore "compare-les" résolue avec buffer non vide
  - vectorstore consulté si query mentionne sujet absent du buffer
  - skip si query déjà self-contained

- `test_rag_should_rewrite_anaphora.py`
  - table de décision complète (anaphore + buffer / anaphore + no buffer / declarative / etc.)

- `test_rag_corpus_lexicon.py`
  - extraction depuis meta.json mockée
  - cache mtime invalidé si meta.json modifié

### Pipeline E2E (mocks LLM, retriever réel ou mocké, ~6 tests)

- `test_rag_pipeline_multiturn_e2e.py`
  - Scénario "compare-les" : T1 Whittaker → T2 Kaplan-Meier → T3 compare → rewriter inclut les deux noms
  - Scénario "et pour 2005 ?" : T1 TH 00-02 → T2-T30 autres sujets → T31 référence éloignée → vectorstore remonte T1
  - Scénario short conv : T1 single question → pas de buffer/summary inutile, marche comme single-turn

### Graph E2E (graph réel + mocks LLM, ~3 tests)

- `test_rag_e2e_graph_multiturn.py`
  - Tour Master → RAG avec history multi-tours, termine sans recursion error
  - Stage events RAG.0 + RAG.7 émis dans le stream
  - 2 questions consécutives partagent la mémoire (cache module-level fonctionne)

**Couverture cible** : ~30 nouveaux tests, total ~390 tests verts.

---

## File impact

| Fichier | Action | Lignes (estim.) |
|---|---|---|
| `agents/rag/memory/__init__.py` | Créer | 5 |
| `agents/rag/memory/schemas.py` | Créer (RAGTurn, RAGSummary) | 40 |
| `agents/rag/memory/rag_memory_store.py` | Créer | 200 |
| `agents/rag/memory/summarizer.py` | Créer | 80 |
| `agents/rag/pipeline/_corpus_lexicon.py` | Créer | 50 |
| `agents/rag/pipeline/query_rewriter.py` | Modifier | +80 / -20 |
| `agents/rag/pipeline/run_pipeline.py` | Modifier (RAG.0 + RAG.7) | +40 |
| `agents/rag/agent_instructions/query_rewriter_prompt.md` | Modifier | +40 |
| `agents/master/method_choices.py` | Modifier (history + session_id) | +10 |
| `agents/mortality/agents/graph.py` (stream_agent) | Modifier (propager `_history` + `_session_id` dans data_store) | +5 |
| `config/llm_models.yaml` | Ajouter `rag.summarizer` | +6 |
| Tests | Créer ~7 fichiers | ~600 |

**Total estimatif** : ~1100 lignes (dont 600 de tests).
**Effort** : 4-6h en mode TDD subagent-driven.

---

## Risques identifiés

| Risque | Mitigation |
|---|---|
| `session_id` non accessible dans le state LangGraph 1.x | Plan B : injecter via `data_store["_session_id"]` au démarrage de `stream_agent()` |
| Cold start (re-embedding 20 Q/A) > 1s perçu par user | Acceptable pour v1. Si problème en prod : pré-warming + cache disque (refacto futur) |
| Cache module-level grossit indéfiniment (sessions abandonnées) | TTL via OrderedDict + LRU (à ajouter si memory leak observé). Pas v1. |
| Summarizer hallucine ou produit JSON malformé | JSON mode OpenAI + Pydantic validation + fallback (garder summary existant si parse fail) |
| Vectorstore retourne Q/A obsolètes (l'user a changé d'avis) | Filtrer par `min_score=0.7` réduit le risque. Pas de purge automatique pour v1. |

---

## Verification end-to-end

```bash
# 1. Tests unit + régression
python -m pytest tests/ -q
# attendu : ~390 verts

# 2. Test scope graph (rag node toujours là, pas de régression)
python -c "
from agents.mortality.agents.graph import build_graph
g = build_graph()
print('rag' in g.nodes)
"

# 3. Test multi-turn isolé
python -m pytest tests/test_rag_pipeline_multiturn_e2e.py -v

# 4. Test manuel session UI
python canvas_app.py
# T1 : 'qu'est-ce que la méthode de wittaker ?'
# T2 : 'et kaplan-meier ?'
# T3 : 'compare-les'
# Vérifier : T3 produit une comparaison des deux (pas une réponse générique)
```

---

## Ordre d'implémentation suggéré

1. `agents/rag/memory/schemas.py` (Pydantic, isolé)
2. `agents/rag/pipeline/_corpus_lexicon.py` + tests (Python pur, isolé)
3. `agents/rag/memory/rag_memory_store.py` (buffer + vectorstore + rebuild) + tests
4. `agents/rag/memory/summarizer.py` (LLM nano, mocké) + tests
5. `agents/rag/memory/rag_memory_store.py` : intégrer summarizer (trigger + append)
6. `agents/rag/pipeline/query_rewriter.py` : signature + prompt + should_rewrite étendu + tests
7. `config/llm_models.yaml` : rôle `rag.summarizer`
8. `agents/rag/pipeline/run_pipeline.py` : RAG.0 + RAG.7 + tests E2E pipeline
9. `agents/mortality/agents/graph.py` (stream_agent) : propager `_history` + `_session_id`
10. `agents/master/method_choices.py` : propager history dans fake_state
11. Tests E2E graph LangGraph
12. Test manuel + commit + push

---

## Critères d'acceptation

1. `RAGMemoryStore` instanciable, buffer fifo respecté, vectorstore add/retrieve fonctionne.
2. `query_rewriter` produit une requête self-contained à partir de buffer + summary + vectorstore.
3. `should_rewrite()` détecte anaphores et force réécriture si buffer non vide.
4. `answer_generator` non modifié — passes les tests existants.
5. Summary update synchrone déclenché au-delà de 10 tours.
6. Lexique auto-derivé depuis `meta.json` (régénéré au prochain ingest).
7. Multi-turn fonctionne dans les DEUX paths (0.e normal RAG + 0.c pending).
8. Régression : 360 tests existants verts + ~30 nouveaux RAG multi-turn.
9. Test manuel : T1 Whittaker → T2 KM → T3 "compare-les" → réponse pertinente.
