# Pipeline end-to-end — détaillé

Document de travail. On part du message utilisateur et on déroule.

---

## Étape 0 : User pose une question → arrivée dans Master

### Que cherche à faire Master ?

Master est un **routeur + interlocuteur enrichi**. Il décide de la suite en suivant une **séquence de checks ordonnés** (court-circuit dès qu'un check matche).

### Séquence de décision (ordre RÉEL)

À chaque message, Master déroule cette checklist dans l'ordre. Il **court-circuite** dès qu'un check matche.

```
Message utilisateur arrive
    │
    ├─ [0.a] RAPPELER LA MÉMOIRE DE LA SESSION
    │        Quels paramètres l'utilisateur a-t-il déjà confirmés ?
    │        (mode de rapport, segmentation H/F, méthodes choisies,
    │         calculs déjà tournés, fichier déjà chargé…)
    │
    ├─ [0.b] PRÉPARER LE FICHIER DE DONNÉES (si pas encore fait)
    │        L'utilisateur a-t-il déjà validé le mapping des colonnes
    │        et des valeurs via l'interface ?
    │        Si OUI et pas encore fait : on génère un fichier "propre"
    │        à utiliser pour la suite (colonnes renommées, dates
    │        parsées, sentinelles 31/12/2999 ramenées à la fin
    │        d'observation réelle).
    │
    ├─ [0.c] AI-JE UNE QUESTION EN ATTENTE ?
    │        Au tour précédent, Master a peut-être posé une question
    │        et attend la réponse (ex: "voulez-vous une table H/F ou
    │        unisex ?", "kaplan_meier ou central ?").
    │       ├── OUI → ACTION 1 : enregistrer la réponse et passer à
    │       │         l'étape suivante (souvent Builder).
    │       └── NON → continuer.
    │
    ├─ [0.d] COMPRENDRE CE QUE VEUT L'UTILISATEUR
    │        Un mini-LLM (gpt-5.4) classe le message sur 4 axes :
    │          • C'est une question ou une demande de calcul ?
    │          • Veut-il un PDF à la fin ?
    │          • Quel niveau de rapport (description seule, taux
    │            bruts, rapport complet) ?
    │          • Table unisex ou H/F séparée ?
    │
    └─ [0.e] DÉCIDER QUOI FAIRE
            │
            ├── C'est une QUESTION (exploration, vérification du
            │     fichier, demande de stat ponctuelle…) :
            │     → ACTION 2 : Master répond directement,
            │       en s'appuyant sur des outils d'inspection
            │       (lire des colonnes, tracer un graphique,
            │       lancer une stat ad hoc).
            │
            └── C'est une DEMANDE DE CALCUL :
                  │
                  ├── Il manque une info essentielle ?
                  │     → ACTION 3 : Master pose une question
                  │       (PDF ?, sexe ?, méthode de calcul ?).
                  │
                  ├── Tout est précisé, les calculs ne sont
                  │     pas encore tous faits :
                  │     → ACTION 4 : Master passe la main au
                  │       Builder (qui exécute le pipeline
                  │       actuariel normé).
                  │
                  └── Les calculs sont tous prêts et un PDF
                        est demandé :
                        → ACTION 5 : Master passe la main au
                          Writer (rédaction du rapport).
```

### Les 5 résultats possibles d'un passage dans Master

| # | Résultat | Déclencheur | Effet |
|---|---|---|---|
| 1 | **Enregistrer la réponse à une question précédente** | Une question était en attente | Suivant la réponse : reprend le cycle, route vers Builder, ou re-pose si réponse ambiguë |
| 2 | **Répondre directement (exploration)** | Message classé "question" | Master répond avec texte + éventuels tableaux/graphiques d'inspection. Pas de routing. |
| 3 | **Poser une question de clarification** | Message classé "demande de calcul" mais info manquante | Master attend la réponse au tour suivant |
| 4 | **Lancer / continuer les calculs (Builder)** | Demande de calcul + toutes les infos OK + calculs non finis | Builder prend la main |
| 5 | **Lancer la rédaction du PDF (Writer)** | Demande de calcul + calculs finis + PDF demandé | Writer prend la main |

**À noter** : "passer la main au Builder" (action 4) n'arrive qu'**après** que toutes les clarifications ont été tranchées. Ce n'est pas un check prioritaire, c'est le résultat final naturel d'un cycle où il n'y a plus rien à clarifier.

### Programmes principaux impliqués

| Fichier | Rôle |
|---|---|
| [agents/mortality/agents/master_node.py](agents/mortality/agents/master_node.py) | Entry point LangGraph — orchestre 0.a → 0.e |
| [agents/master/conversation.py](agents/master/conversation.py) | **Mode conversationnel enrichi** (0.e branche question) — boucle tool-calling LLM avec whitelist `CONVERSATIONAL_TOOLS` |
| [agents/master/classify_intent.py](agents/master/classify_intent.py) | 0.d — classification LLM 4-axes (gpt-5.4 JSON mode) |
| [agents/master/disambiguation.py](agents/master/disambiguation.py) | 0.b — mapping colonnes/valeurs + écriture Parquet normalisé |
| [agents/master/method_choices.py](agents/master/method_choices.py) | 0.e (sous-cas task) — désambiguation méthodes de calcul |
| [agents/master/question_filter.py](agents/master/question_filter.py) | 0.c — extraction de réponse user à `_pending_need` |
| [agents/master/extract_study_plan.py](agents/master/extract_study_plan.py) | 0.a — extraction des paramètres d'étude depuis l'historique (LLM mini) |
| [agents/master/extract_gender.py](agents/master/extract_gender.py) | 0.d/extras — détection déterministe "unisex"/"by_sex" |
| [session/memory_manager.py](session/memory_manager.py) | 0.a — hydratation SessionState depuis disque |
| [session/dataset_store.py](session/dataset_store.py) | 0.a — chargement Parquet (préfère le normalisé) |
| [session/session_state.py](session/session_state.py) | Schéma Pydantic SessionState |

### Mode conversationnel — détail (0.e branche kind="question")

Quand Master entre en mode conversationnel, il **n'est plus muet** : il peut interroger le DataFrame via une boucle tool-calling LLM. Le LLM (gpt-5.4-nano) choisit entre :

| Tool | Capacité |
|---|---|
| `conversation.data_inspect` | columns, shape, head, describe, value_counts, date_range |
| `conversation.plot_basic` | histogram, bar, scatter, time_series → PNG |
| `conversation.eval_pandas` | expression Python sandboxée (`df`, `pd`, `np`, `plt`, `sns`, `stats`, `ll`/lifelines, `datetime`) — AST whitelist refuse `import`, dunders, `open`/`exec`/`eval`, méthodes I/O |
| `statistical_analysis.*` | data_quality, age_distribution, time_series, segmentation, portfolio_summary |

**Scope strict** : ces tools sont définis dans `CONVERSATIONAL_TOOLS` ([agents/master/conversation.py:30](agents/master/conversation.py#L30)). Le Builder (`BUILDER_TOOLS` dans [agents/mortality/agents/builder_node.py:34](agents/mortality/agents/builder_node.py#L34)) ne les voit jamais — il reste sur son pipeline normé.

**L'utilisateur peut sortir du mode actuariel à tout moment** : s'il pose une question d'exploration en milieu de pipeline, `classify_intent` détecte `kind=question` → respond_conversationally() → réponse + plots inline → le Builder reprend si l'utilisateur le redemande après.

### Sortie de Master (vers le graphe LangGraph)

```python
return {
    "messages":     [AIMessage(content="…") | HumanMessage(content="[Master] …") | ToolMessage(...)],
    "events":       [{"type": "agent_switch"|"message"|"tool_call"|"tool_result"|"image"|"done", ...}],
    "active_agent": "builder" | "writer" | "master",
    "data_store":   {...},
}
```

En mode conversationnel, `events` peut contenir plusieurs `tool_call`/`tool_result` + des `image` (paths PNG) que le canvas affiche.

### Sécurité du mode conversationnel

- **L'utilisateur ne tape jamais de code** — c'est le LLM qui génère expressions ou tool calls.
- `eval_pandas` parse l'expression en AST et refuse 26+ patterns dangereux AVANT exécution (cf. [tests/test_eval_pandas_safety.py](tests/test_eval_pandas_safety.py)).
- Namespace d'exécution explicit : `{"__builtins__": <subset>, "df": ..., "pd": pd, "np": np, "stats": stats, "plt": plt, "sns": sns, "ll": lifelines, "datetime": ...}` — pas d'`os`, `sys`, env vars, filesystem, network.
- Boucle tool-calling bornée à 5 itérations.
- Plots écrits exclusivement dans `tmp/conversation_plots/` (gitignored).

### Modèles LLM impliqués dans l'étape 0

| Sous-étape | Rôle YAML | Modèle | Raison |
|---|---|---|---|
| 0.d classify_intent | `master.classify_intent` | gpt-5.4 (full) | Routage critique — pas droit à l'erreur |
| 0.a extract_study_plan | `master.extract_study_plan` | gpt-5.4-mini | JSON extraction, mini suffit |
| 0.e method resolution fallback | `master.method_resolution` | gpt-5.4-mini | JSON contraint |
| 0.e conversation tool-calling | `master.conversation` | **gpt-5.4-nano** | Tâche peu critique, coût ÷10-15× |

---

(étapes 1 à 9 — à détailler après que l'étape 0 soit validée)
