# Instruction à coller dans une nouvelle conversation Claude

---

Tu vas construire une application Python complète qui reproduit le mécanisme de la plateforme URLab pour les actuaires. L'application comporte 3 fichiers : `app.py`, `notebook_runner.py`, et `agent.py`. Tu les génères tous les trois en une seule réponse, complets et fonctionnels.

---

## Contexte

URLab est une plateforme actuarielle où :
1. Un actuaire senior a rédigé un notebook `.ipynb` qui décrit une méthodologie validée (ici : construction d'une table de mortalité en Python)
2. Un agent IA lit ce notebook comme guide et l'exécute étape par étape en s'adaptant aux données fournies par l'utilisateur
3. L'interface est un split-view : panneau gauche = chat avec l'agent, panneau droit = notebook avec les cellules et leurs outputs

L'objectif est de reproduire ce mécanisme avec Streamlit + l'API Anthropic Claude.

---

## Ce que tu dois construire

### Fichier 1 : `notebook_runner.py`

Ce module lit un fichier `.ipynb` et expose ses cellules. Il doit :
- Charger le notebook avec `nbformat`
- Retourner une liste de cellules sous forme de dicts : `{"id", "type" (code/markdown), "source", "output"}`
- Exposer une fonction `execute_cell(cell_source: str, kernel_state: dict) -> str` qui exécute du code Python via `exec()` dans un namespace partagé et capture stdout + la valeur de retour
- Gérer les erreurs d'exécution proprement (retourner le traceback comme output)
- Exposer une fonction `get_notebook_as_context(path: str) -> str` qui retourne le contenu du notebook formaté en texte brut (pour injection dans le contexte Claude) — format : pour chaque cellule, `[MARKDOWN]\n{source}` ou `[CODE]\n{source}`

### Fichier 2 : `agent.py`

Ce module gère l'agent Claude. Il doit :
- Utiliser `openAI` Python SDK pour la partie LLM (on externalise dans un fichier séparé les clés et code secrets) 
- Exposer une fonction `stream_agent_response(user_message: str, notebook_context: str, conversation_history: list) -> generator`
- Le system prompt de l'agent est le suivant (à inclure tel quel dans le code) :

```
Tu es un agent actuariel spécialisé dans la construction de tables de mortalité d'expérience.

Tu disposes d'un notebook blueprint qui décrit la méthodologie validée. Ce notebook est ta référence absolue.

NOTEBOOK BLUEPRINT :
{notebook_context}

Ton rôle :
- Suivre le notebook étape par étape
- Pour chaque étape, expliquer ce que tu fais en une phrase, puis donner le code Python à exécuter entre balises <code> et </code>
- Adapter le code aux données de l'utilisateur (chemin fichier, noms de colonnes)
- Signaler toute anomalie (données manquantes, SMR anormal, qx non monotones)
- Ne jamais changer la méthode de lissage (Whittaker-Henderson) sans demande explicite

Format de réponse pour chaque étape :
**Étape X — [nom]**
[explication en une phrase]
<code>
[code Python]
</code>
[commentaire sur le résultat attendu]

Quand l'utilisateur envoie ses données pour la première fois, commence par poser ces 3 questions :
1. Sexe dominant du portefeuille (H / F / mixte) ?
2. Date de fin d'observation ?
3. Paramètre de lissage λ souhaité (défaut : 100) ?
```

- La fonction `stream_agent_response` doit utiliser `client.messages.stream()` et yielder les chunks de texte au fur et à mesure
- `conversation_history` est une liste de `{"role": "user"/"assistant", "content": "..."}` — l'inclure dans l'appel API pour maintenir le contexte multi-tour
- Modèle : `GPT5 nano`

### Fichier 3 : `app.py`

Interface Streamlit qui reproduit le split-view URLab. Elle doit :

**Layout** :
- `st.set_page_config(layout="wide")`
- Deux colonnes : gauche 40% (chat), droite 60% (notebook viewer)
- Titre en haut : "Actuarial Notebook Agent"

**Colonne gauche — Chat** :
- Upload CSV en haut (`st.file_uploader`) — quand un fichier est uploadé, sauvegarder dans `./uploads/` et injecter le chemin dans le premier message utilisateur automatiquement
- Affichage de l'historique de conversation avec `st.chat_message`
- Input utilisateur avec `st.chat_input`
- Quand l'utilisateur envoie un message :
  1. Appeler `stream_agent_response` et afficher la réponse en streaming avec `st.write_stream`
  2. Parser la réponse pour extraire les blocs `<code>...</code>`
  3. Pour chaque bloc de code extrait, appeler `execute_cell` et stocker l'output dans `st.session_state.cell_outputs`
  4. Ajouter la réponse à `conversation_history` dans session_state

**Colonne droite — Notebook viewer** :
- Charger le notebook `mortality_table_blueprint.ipynb` au démarrage
- Afficher chaque cellule dans une carte :
  - Cellule markdown : rendu avec `st.markdown`, badge gris "Markdown"
  - Cellule code : `st.code(source, language='python')`, badge vert "Code"
  - Si un output existe dans `st.session_state.cell_outputs` pour cette cellule : l'afficher dans un `st.container` avec fond gris clair en dessous du code
- Un bouton "Run all" en haut de la colonne qui exécute toutes les cellules de code dans l'ordre
- Indicateur de statut kernel en haut à droite : "Kernel actif" (vert) ou "Kernel inactif" (gris)

**Session state à initialiser** :
```python
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'cell_outputs' not in st.session_state:
    st.session_state.cell_outputs = {}
if 'kernel_namespace' not in st.session_state:
    st.session_state.kernel_namespace = {}
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
```

**Gestion de la clé API** :
- Lire depuis `st.secrets["OPENAI_API_KEY"]` ou variable d'environnement `OPENAI_API_KEY`
- Si absente, afficher `st.error` et `st.stop()`
le pgm lit par lui même ses éléments (je n'ai pas à les données quand j'appelle le pgm)
---

## Dépendances

Génère aussi un fichier `requirements.txt` :
```
streamlit>=1.32.0
nbformat>=5.9.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
```
je te laisse y ajouter les autres librairies (dont open AI)
---

## Contraintes importantes

- Tout le code doit être en français pour les commentaires et messages UI
- Pas de base de données, pas de Docker — tout tourne en local
- Le notebook `mortality_table_blueprint.ipynb` doit être dans le même dossier que `app.py`
- Le `kernel_namespace` dans session_state sert de mémoire partagée entre les cellules — c'est lui qui maintient l'état (comme `data`, `table`, `SMR`) entre les exécutions
- Les imports du notebook (`pandas`, `numpy`, etc.) doivent être pré-chargés dans le namespace au démarrage

---

## Comment lancer

À la fin de ta réponse, donne les commandes de lancement :
```bash
pip install -r requirements.txt
streamlit run app.py
```
