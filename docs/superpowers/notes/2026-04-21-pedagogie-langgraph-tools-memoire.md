# Pédagogie — Comment fonctionne notre système LLM agent (LangGraph + tools + mémoire)

> Support de référence pédagogique — session du 2026-04-21.
> À utiliser pour construire un support de présentation dans une autre session.

---

## 1. Comment le LLM choisit un tool (tool_call selection)

### 1.1 À chaque tour, on passe 3 choses au LLM

Dans [builder_node.py](../../../agents/mortality/agents/builder_node.py) :

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,       # ← historique de conversation
    tools=tools,             # ← CATALOGUE des tools disponibles
    tool_choice="auto",      # ← mode de sélection
    ...
)
```

- **`messages`** : tout l'historique (system prompt + tours user/assistant/tool).
- **`tools`** : la liste des outils avec leur description et leur schéma de paramètres (JSON Schema).
- **`tool_choice="auto"`** : on laisse le LLM décider s'il appelle un tool ou pas.

### 1.2 À quoi ressemble le catalogue fourni

Pour chaque tool :

```json
{
  "type": "function",
  "function": {
    "name": "builder.smoothing",
    "description": "Lisse la table de taux bruts par Whittaker, Gompertz, Makeham ou spline…",
    "parameters": {
      "type": "object",
      "properties": {
        "method":    {"type": "string", "enum": ["whittaker","gompertz","makeham","spline"]},
        "lambda_wh": {"type": "number", "description": "Paramètre de lissage Whittaker"}
      },
      "required": ["method"]
    }
  }
}
```

Ce catalogue est construit par `get_openai_tools()` depuis `tools/catalogue.yaml`.

### 1.3 Le LLM fait son choix

Le modèle lit :
- **La demande de l'utilisateur** ("construis-moi une table de mortalité").
- **Son prompt système** ("tu es un actuaire, suis cette séquence : exposure → crude_rates → smoothing → …").
- **Les descriptions des tools** (chaque tool dit quand l'utiliser dans `description` et `when_to_use`).

Il compare sémantiquement la situation aux descriptions et choisit le tool qui "match le mieux". Il remplit les paramètres en respectant le JSON Schema.

### 1.4 Trois sources d'influence, par ordre d'importance

1. **Les descriptions des tools** (`description`, `when_to_use`, `when_not_to_use` dans le catalogue).
2. **Les instructions système** (`step1_planning.md`, `behavioral_contract.md`) qui imposent un ordre canonique.
3. **Le contexte de la conversation** (ce qui a déjà été fait, retours précédents).

### 1.5 Format de la réponse LLM

| Réponse | Signification |
|---|---|
| `content: "..."`, `tool_calls: null` | pas de tool, juste une réponse à l'utilisateur |
| `content: null`, `tool_calls: [...]` | un/plusieurs tool sans message |
| `content: "..."`, `tool_calls: [...]` | texte **ET** tool en parallèle (source de bugs !) |

---

## 2. Qui orchestre le passage d'un tool à l'autre : LangGraph

### 2.1 Structure du graphe

Dans [graph.py](../../../agents/mortality/agents/graph.py), deux nœuds pour le Builder :
- `builder_node` → appelle l'LLM.
- `tools` (un `ToolNode` LangGraph) → exécute les fonctions Python des tools.

Et **une fonction d'aiguillage** entre les deux : `_should_continue_builder`.

### 2.2 Le cycle de bouclage

```
[user message]
     │
     ▼
builder_node ───► appelle LLM ───► ajoute AIMessage dans state
     │
     ▼
_should_continue_builder inspecte le dernier message
     │
     ├─ si AIMessage.tool_calls non vide  ──► "tools"
     │                                          │
     │                                          ▼
     │                                    ToolNode exécute la/les fonctions
     │                                          │
     │                                          ▼
     │                                    ToolMessage ajouté dans state
     │                                          │
     │                                          ▼
     │                                    edge inconditionnelle revient vers builder_node
     │                                          │
     │                                          └──► ( on boucle : LLM relit + décide encore )
     │
     ├─ si "<BUILD_DONE>" dans content    ──► "to_master"
     │
     └─ sinon                               ──► END (pause)
```

### 2.3 Qui décide quoi

| Décision | Qui ? |
|---|---|
| Quel tool appeler (parmi le catalogue) | **LLM** |
| Avec quels paramètres | **LLM** (en respectant le JSON Schema) |
| Rappeler ou pas le LLM après un tool | **LangGraph** (via `_should_continue_builder`) |
| Router vers un autre nœud (Master, Writer) | **LangGraph** (sur `<BUILD_DONE>`, `active_agent`, etc.) |
| Exécuter effectivement la fonction Python | **ToolNode** (Python) |

**LangGraph n'est pas "intelligent"** au sens où il ne choisit pas les tools — il est la plomberie déterministe qui fait circuler les messages entre les deux acteurs (LLM et tools) tant qu'il y a du travail.

---

## 3. Qu'est-ce que ToolNode ?

### 3.1 Définition

Classe fournie par LangGraph qui sert de "porte d'entrée" automatique pour exécuter les fonctions Python des tools.

### 3.2 Ce qu'elle fait concrètement

Quand LangGraph route un message vers un nœud de type `ToolNode` :

1. Elle **lit le dernier `AIMessage`** de l'historique.
2. Elle **récupère la liste `tool_calls`** dans ce message.
3. Pour chaque tool_call :
   - **Retrouve la fonction Python** correspondante.
   - **Extrait les arguments** JSON du tool_call.
   - **Appelle la fonction** avec ces arguments.
   - **Enveloppe le résultat** dans un `ToolMessage`.
4. Elle **ajoute ces `ToolMessage`** dans l'historique du state.
5. LangGraph rend la main au prochain nœud (généralement le nœud LLM).

### 3.3 Code équivalent sans ToolNode

```python
def execute_tools(state):
    last = state["messages"][-1]
    tool_messages = []
    for tc in last.tool_calls:
        fn = REGISTRY[tc["name"]]
        args = json.loads(tc["arguments"])
        result = fn(**args)
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )
    return {"messages": state["messages"] + tool_messages}
```

### 3.4 Synthèse

> **ToolNode = le bout de plomberie qui, entre chaque tour du LLM, regarde ce que l'LLM a demandé, exécute les fonctions Python correspondantes, et remet les résultats dans la conversation.**

---

## 4. Qu'est-ce que le state LangGraph

### 4.1 Définition

Un **dict typé partagé entre tous les nœuds du graphe**. Chaque nœud le reçoit en entrée, peut le lire, et retourne des mises à jour que LangGraph merge dedans.

Dans [state.py](../../../agents/mortality/agents/state.py) :

```python
class AgentState(TypedDict):
    messages:         list[AnyMessage]   # ← l'historique de conversation
    dataset_ref:      str | None          # ← référence du CSV uploadé
    data_store:       dict                # ← résultats accumulés des tools
    context_docs:     list                # ← docs uploadés par l'user
    plan_established: bool
    active_agent:     str                 # ← "master" | "builder" | "writer"
    events:           list                # ← événements streamés vers le front
    step_by_step:     bool
    pending_tool_call: dict | None
```

### 4.2 Le champ `messages` en particulier

Liste chronologique de messages. 4 types :

| Type de message | Qui l'émet | Exemple de contenu |
|---|---|---|
| `HumanMessage` | l'utilisateur | "construis-moi une table de mortalité" |
| `AIMessage` | le LLM | "Je lance les calculs" + éventuellement `tool_calls` |
| `ToolMessage` | le framework après exécution d'un tool | le dict JSON retourné par `builder.exposure` |
| `SystemMessage` | instructions cadres | rarement — on passe plutôt le system prompt à chaque tour |

### 4.3 Exemple d'évolution

**T0** : `state["messages"] = []`

**T1** — utilisateur charge le CSV et demande :
```python
state["messages"] = [
    HumanMessage("construis-moi une table de mortalité"),
]
```

**T2** — Master classify_intent puis route vers Builder :
```python
state["messages"] = [
    HumanMessage("construis-moi une table de mortalité"),
    AIMessage("Je route vers le BuilderAgent."),
    HumanMessage("Lance l'ensemble des calculs actuariels : exposure, crude_rates, …"),
]
state["active_agent"] = "builder"
```

**T3** — Builder LLM émet un tool_call :
```python
state["messages"] += [
    AIMessage(content=None, tool_calls=[{"name": "builder.exposure", "arguments": {...}}]),
]
```

**T4** — ToolNode exécute et ajoute le résultat :
```python
state["messages"] += [
    ToolMessage(content='{"total_exposure": 1234.5, "total_deaths": 42, ...}'),
]
state["data_store"]["total_exposure"] = 1234.5
state["data_store"]["total_deaths"]   = 42
```

**T5** — Builder LLM relit (avec ToolMessage) et émet son prochain tool_call. Et ainsi de suite.

### 4.4 Accumulation via reducer

```python
messages: Annotated[list[AnyMessage], add_messages]
```

`add_messages` est un **reducer** LangGraph : au lieu d'**écraser** la liste quand un nœud retourne `{"messages": [...]}`, il **ajoute** ces messages. C'est ce qui permet l'accumulation continue de l'historique.

---

## 5. Gestion de la mémoire — les trois couches

### 5.1 Couche 1 — mémoire courte : fenêtre glissante de messages

Dans [builder_node.py](../../../agents/mortality/agents/builder_node.py) :
```python
MAX_HISTORY = 20
raw_msgs = state["messages"]
if len(raw_msgs) > MAX_HISTORY:
    raw_msgs = raw_msgs[-MAX_HISTORY:]
```

- **Avantage** : simple, naturel pour le LLM.
- **Limite** : si un info clé est plus vieille que 20 tours, elle est perdue.

### 5.2 Couche 2 — mémoire "système" : infos critiques dans le system prompt

À chaque tour, on reconstruit le system prompt avec les infos qu'on veut que le LLM n'oublie **JAMAIS** :
- Contrat comportemental + instructions step-by-step (fichiers markdown fixes).
- Catalogue des tools disponibles à ce niveau (light/middle/full).
- Mapping des colonnes du CSV.
- `study_plan` confirmé par l'utilisateur.
- Résumé compacté de la conversation précédente (`MemoryManager.get_context_block()`).

- **Avantage** : règle dure qui survit à toute la conversation.
- **Limite** : le system prompt grossit → tokens consommés.

### 5.3 Couche 3 — mémoire longue : `data_store` + Parquet

Quand un tool retourne son résultat :
1. Le dict est **enveloppé dans un `ToolMessage`** (couche 1).
2. Certaines clés sont **extraites dans `state["data_store"]`** (couche longue).

Exemple après `builder.exposure` :
```python
state["data_store"]["total_exposure"]   = 1234.5
state["data_store"]["total_deaths"]     = 42
state["data_store"]["exposure_table"]   = [...]
```

Et en parallèle :
- Le DataFrame CSV est stocké en **Parquet sur disque** via `session/dataset_store.py`.
- Un `MemoryManager` génère un **résumé compact** réinjecté dans le system prompt.

- **Avantage** : des tonnes de données structurées sans jamais les montrer au LLM directement.
- **Limite** : le LLM n'y accède que via des tools ou via le résumé.

### 5.4 Schéma des trois couches ensemble

```
┌────────────────────────────── À CHAQUE TOUR ──────────────────────────────┐
│                                                                           │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │ SYSTEM PROMPT (reconstruit)                                      │    │
│   │  • instructions fixes (behavioral contract, step1_planning…)     │    │
│   │  • catalogue tools (schémas JSON)                                │    │
│   │  • mapping colonnes                                              │    │
│   │  • study_plan confirmé                                           │    │
│   │  • résumé compact des tours passés                               │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │ 20 DERNIERS MESSAGES (HumanMessage / AIMessage / ToolMessage)    │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│                             ⬇   ENVOI LLM                                 │
└───────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────── PERSISTÉ ───────────────────────────────────┐
│                                                                           │
│   state["messages"]              ← historique complet (toutes les T turns)│
│   state["data_store"]            ← dict {clé: valeur} cumulatif           │
│   session/{id}/records.parquet   ← DataFrame du CSV                       │
│   session/{id}/memory.json       ← résumé progressif + contexte agent     │
│                                                                           │
│   ⬆ alimente le system prompt du prochain tour                            │
└───────────────────────────────────────────────────────────────────────────┘
```

### 5.5 Règle de conception

> **"Ce qui est stable et crucial → system prompt.**
> **Ce qui est récent → messages window.**
> **Ce qui est volumineux ou structuré → data_store / fichiers.**
> **Ce qui est ancien mais pertinent → résumé dans le system prompt."**

### 5.6 En pratique — les 3 questions à te poser

Quand tu conçois un nouveau tool ou une nouvelle phase :

1. **Un tour futur aura-t-il besoin de cette info ?** → écris dans `data_store`.
2. **Le LLM doit-il la voir à chaque tour ?** → injecte dans le system prompt.
3. **Est-ce juste un échange transitoire ?** → laisse ça dans `messages` (sera fenêtré).

---

## 6. Pourquoi limiter à 20 messages ?

Trois raisons :

1. **Coût** — gpt-4o facture par token, un tour peut atteindre 5k-10k tokens. Sans fenêtrage, chaque tour coûterait de plus en plus cher.
2. **Latence** — plus le contexte est long, plus la génération est lente.
3. **Qualité** — paradoxalement, trop de contexte "noie" le LLM. Les 20 derniers suffisent : la dernière demande + tool_calls récents + résultats.

## 7. Risque et contournement

Si un élément crucial est plus vieux que 20 messages, le LLM le perd. Exemple :
- Message 1 : "l'observation s'arrête au 31/12/2023"
- Messages 2-25 : calculs, allers-retours
- Message 26 : Le LLM ne voit plus le 31/12/2023 → redemande ou utilise un défaut.

Contournement dans notre code : on injecte `study_plan` dans le **system prompt** à chaque tour (couche 2) → toujours visible même hors fenêtre.

---

## 8. Ce qui est envoyé au LLM à chaque tour — résumé

| Contenu | Provenance |
|---|---|
| system prompt (instructions + catalogue tools + study_plan + mapping colonnes) | **reconstruit** via `_build_system_prompt()` |
| 20 derniers messages de `state["messages"]` | **fenêtrage** sur l'historique |
| Liste des tools disponibles (schéma JSON) | injectée via `tools=[...]` de l'API |

---

## Fichiers clés référencés

- [agents/mortality/agents/graph.py](../../../agents/mortality/agents/graph.py) — construction du graphe, `_should_continue_*`
- [agents/mortality/agents/builder_node.py](../../../agents/mortality/agents/builder_node.py) — appel LLM + fenêtrage
- [agents/mortality/agents/master_node.py](../../../agents/mortality/agents/master_node.py) — orchestration inter-agents
- [agents/mortality/agents/state.py](../../../agents/mortality/agents/state.py) — définition du state typé
- [agents/mortality/agent_instructions/](../../../agents/mortality/agent_instructions/) — instructions markdown du LLM
- [tools/catalogue.yaml](../../../tools/catalogue.yaml) — catalogue de tools
- [session/dataset_store.py](../../../session/dataset_store.py) — persistance Parquet
- [session/memory_manager.py](../../../session/memory_manager.py) — résumé compact
