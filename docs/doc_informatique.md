# Documentation informatique — Agent Actuariat

Pour les développeurs qui veulent comprendre ou étendre le projet.

## Comment ajouter une nouvelle fonction à un tool existant

### Via l'interface DEV (recommandé)

1. Lancer `python canvas_app.py`
2. Aller dans l'onglet **DEV → Capacités**
3. Cliquer **Ajouter** sur la carte du tool cible (ex : `statistical_analysis`)
4. Remplir : nom, description, colonnes requises/optionnelles, paramètres
5. Le code est généré automatiquement et modifiable directement
6. Cliquer **Créer la fonction**

→ Le fichier `.py` est créé dans `report_agent/tools/{tool}/` et `builder_capabilities.json` est mis à jour automatiquement.

### Manuellement

1. Créer `report_agent/tools/{tool}/{fonction}.py` avec la structure :

```python
"""
INPUTS / OUTPUT docstring obligatoire
"""
from __future__ import annotations
import pandas as pd
from report_agent.dictionary.column_schema import find_col_by_role

def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    params = params or {}
    # Détecter les colonnes
    date_col = find_col_by_role(df, "date_entree")
    # Calculer...
    return {"resultat": ...}
```

2. Ajouter l'entrée dans `report_agent/builder_capabilities.json` :

```json
"ma_fonction": {
  "description": "Ce que fait la fonction",
  "required_columns": ["date_entree"],
  "optional_columns": ["sexe"],
  "column_notes": "Explication sur les colonnes",
  "params": {"mon_param": "int — explication"}
}
```

## Structure de COLUMN_SCHEMA

`report_agent/dictionary/column_schema.py` est la **source unique** pour les noms de colonnes. Chaque rôle définit :
- `label` : nom lisible en français
- `question` : question à poser à l'utilisateur si la colonne est absente
- `candidates` : liste de noms acceptés (insensible à la casse)

Pour ajouter un rôle :
```python
"mon_role": {
    "label": "Ma colonne",
    "question": "Quelle colonne correspond à... ?",
    "candidates": ["mon_col", "my_col", "col_alias"],
},
```

## Comment fonctionne tool_registry.py

```
call_tool("builder", "smoothing", params, df=df, data=data_store)
    ↓
1. Vérifie que "builder" existe dans builder_capabilities.json
2. Vérifie que "smoothing" existe et est disponible
3. Importe dynamiquement report_agent.tools.builder.smoothing
4. Appelle smoothing.run(data_store, params)
5. Retourne le résultat (dict) ou {"erreur": "..."}
```

Routing par type de tool :
- `statistical_analysis` → `run(df, params)`
- `builder.exposure` → `run(df, params)` (seule fonction builder avec df)
- `builder.*` (autres) → `run(data_store, params)`
- `graphs` → `run(data_store, params)`
- `reasoning` → `run(context, params)`

## Comment fonctionne le WriterAgent

```python
writer = WriterAgent()
for event in writer.run_agent_loop(history, df=df):
    # events : tool_call | tool_result | message | done | error
    print(event)
```

La boucle :
1. Construit le system prompt depuis `writer_dialog_prompt.md` + mapping CSV + capabilities JSON
2. Appelle `openai.chat.completions.create(..., tools=get_openai_tools())`
3. Si `finish_reason == "tool_calls"` : exécute chaque tool via `call_tool()`, stocke dans `data_store`
4. Si `finish_reason == "stop"` : yield `message` puis `done`

## Comment fonctionne le builder/ par rapport aux notebooks/

Les fonctions dans `notebooks/` sont les **sources actuarielles** développées par les actuaires. Elles sont autonomes et peuvent être utilisées directement en Python.

Les fonctions dans `tools/builder/` sont des **wrappers minces** qui :
- Détectent les colonnes via `column_schema` (au lieu de noms hardcodés)
- Convertissent les paramètres du format JSON vers les paramètres Python
- Uniformisent la sortie au format dict sérialisable

Le chargement des notebooks se fait via `tools/builder/_nb_loader.py` (importlib.util), ce qui évite de rendre les notebooks dépendants de la structure de packages.

## Gestion des images (graphs/)

Les fonctions graphs retournent `{"image_b64": "<PNG encodé en base64>"}`. Dans `canvas_app.py`, ces images sont affichées directement avec `html.Img(src="data:image/png;base64,...")`.

Dans `writer_agent.py`, l'image est **tronquée** dans les messages OpenAI pour ne pas saturer le contexte (`<image base64 tronquée>`). L'image reste dans `_writer_state["events"]` pour l'affichage UI.

## Lancer l'application

```bash
cd "Agent actuariat"
python canvas_app.py
# → http://localhost:8050
```
