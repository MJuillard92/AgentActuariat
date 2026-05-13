# Pipeline end-to-end — détaillé

Document de travail. On part du message utilisateur et on déroule.

---

## Étape 0 : User pose une question → arrivée dans Master

### Que cherche à faire Master ?

Master est un **routeur** + **interlocuteur**. Quand un message user arrive, son job en une phrase : **décider de l'action suivante** parmi 5 possibilités :

1. **Répondre conversationnellement** (l'utilisateur pose une question hors calculs)
2. **Lancer / continuer un calcul** (l'utilisateur demande un rapport, des taux, etc.) → route vers Builder
3. **Poser une question de clarification** (informations indispensables manquantes : column mapping, mode du rapport, méthode de calcul, sexe…)
4. **Résoudre une question pendante** (Master attendait une réponse précise du précédent tour)
5. **Router vers Writer** (les calculs sont finis, il faut rédiger le PDF)

### Sous-étapes internes (ordre d'exécution)

```
Message user arrive
    ↓
[0.a] Hydratation : récupérer l'historique et le state
    ↓
[0.b] Normalisation déterministe (si mappings UI confirmés)
    ↓
[0.c] Y a-t-il un `_pending_need` à résoudre ?
    ├── oui → extraire la réponse → mise à jour state → router
    └── non → continuer
    ↓
[0.d] Classifier l'intention de l'utilisateur (4 axes)
    ↓
[0.e] Décider de l'action en fonction de la classification
    ├── kind=question → répondre conversationnellement
    ├── kind=task + besoin de précisions → poser une question
    └── kind=task + tout OK → router vers Builder
```

### Programmes principaux impliqués

| Fichier | Rôle |
|---|---|
| [agents/mortality/agents/master_node.py](agents/mortality/agents/master_node.py) | Entry point LangGraph — orchestration des sous-étapes |
| [agents/master/classify_intent.py](agents/master/classify_intent.py) | Classification LLM 4-axes (kind / write / report_mode / gender) |
| [agents/master/disambiguation.py](agents/master/disambiguation.py) | Mapping colonnes/valeurs + normalisation Parquet |
| [agents/master/method_choices.py](agents/master/method_choices.py) | Désambiguation méthodes de calcul (méta-question + per-tool) |
| [agents/master/question_filter.py](agents/master/question_filter.py) | Extraction de réponse user à une question pendante |
| [agents/master/extract_study_plan.py](agents/master/extract_study_plan.py) | Extraction structurée des paramètres d'étude depuis l'historique |
| [agents/master/extract_gender.py](agents/master/extract_gender.py) | Détection déterministe "unisex" / "by_sex" dans le texte user |
| [session/memory_manager.py](session/memory_manager.py) | Hydratation : recharge SessionState (study_plan, flags) depuis disque |
| [session/dataset_store.py](session/dataset_store.py) | Chargement Parquet (original ou normalisé) |
| [session/session_state.py](session/session_state.py) | Schéma Pydantic SessionState (study_plan, column_mapping, methods, …) |

### Sortie de Master (vers le graphe LangGraph)

```python
return {
    "messages":     [AIMessage(content="…") | HumanMessage(content="[Master] …")],
    "events":       [{"type": "agent_switch", "agent": "MasterAgent"}, ...],
    "active_agent": "builder" | "writer" | "master",   # routing
    "data_store":   {...},                              # state enrichi
}
```

### Ce que tu peux préciser pour cette étape

- Comment l'historique est compacté (`MemoryManager.trim_messages`) ?
- Quand `_disambiguation_done` passe à True ?
- Comment Master différencie "réponse à une question pendante" vs "nouvelle intention" ?
- Comportement quand confidence < 0.80 (reformulation) ?

---

(étapes 1 à 9 — à détailler après que l'étape 0 soit validée)
