# Documentation informatique — Agent Actuariat

**Pour les développeurs qui maintiennent et étendent le projet.**
Version du document : mars 2026 — correspond à la v2.0 de l'application.

---

## Table des matières

1. [Architecture globale](#1-architecture-globale)
2. [Lancer l'application](#2-lancer-lapplication)
3. [Les tools actuariels — liste complète et interface](#3-les-tools-actuariels--liste-complète-et-interface)
4. [data_store — structure et accumulation](#4-data_store--structure-et-accumulation)
5. [WriterAgent — cycle de vie](#5-writeragent--cycle-de-vie)
6. [tool_registry.py — routage](#6-tool_registrypy--routage)
7. [column_schema.py — mapping automatique](#7-column_schemapy--mapping-automatique)
8. [builder_capabilities.json — format](#8-builder_capabilitiesjson--format)
9. [Ajouter un nouveau tool — guide pas-à-pas](#9-ajouter-un-nouveau-tool--guide-pas-à-pas)
10. [Gestion des images et PDF](#10-gestion-des-images-et-pdf)
11. [Documents de contexte (uploads additionnels)](#11-documents-de-contexte-uploads-additionnels)

---

## 1. Architecture globale

### Arborescence des fichiers importants

```
Agent actuariat/
│
├── canvas_app.py                        # Point d'entrée — interface Dash + callbacks
├── config.py                            # Configuration globale (clés API, chemins)
│
├── report_agent/
│   ├── writer_agent.py                  # Orchestrateur maître (boucle tool-calling OpenAI)
│   ├── writer_dialog_prompt.md          # Prompt système de base injecté dans chaque session
│   ├── builder_capabilities.json        # Catalogue des tools (source de vérité)
│   │
│   ├── dictionary/
│   │   └── column_schema.py             # Mapping rôles → noms de colonnes CSV
│   │
│   └── tools/
│       ├── tool_registry.py             # Routage central : call_tool(), get_openai_tools()
│       │
│       ├── statistical_analysis/        # Analyse descriptive du portefeuille
│       │   ├── portfolio_summary.py
│       │   ├── age_distribution.py
│       │   ├── time_series.py
│       │   ├── segmentation.py
│       │   └── data_quality.py
│       │
│       ├── builder/                     # Construction de table de mortalité
│       │   ├── _nb_loader.py            # Chargeur de notebooks Python (importlib)
│       │   ├── exposure.py
│       │   ├── crude_rates.py
│       │   ├── smoothing.py
│       │   ├── diagnostics.py
│       │   ├── validation.py
│       │   └── benchmarking.py
│       │
│       ├── graphs/                      # Génération de graphiques (PNG base64)
│       │   ├── analysis_plots.py
│       │   ├── builder_plots.py
│       │   └── sample_gallery.py
│       │
│       ├── build_pdf/                   # Génération de rapports et livrables
│       │   ├── descriptive_report.py
│       │   ├── certification_report.py
│       │   ├── session_log.py
│       │   └── generate_notebook.py
│       │
│       └── reasoning/
│           └── understand_request.py    # Classifie l'intent avant tout calcul
│
├── notebooks/                           # Modules actuariels (sources de calcul)
│   ├── 01_data_preparation.py
│   ├── 02_exposure.py
│   ├── 03_crude_rates.py
│   ├── 04_smoothing.py
│   ├── 05_diagnostics.py
│   ├── 06_validation.py
│   ├── 07_benchmarking.py
│   ├── 08_visualization.py
│   ├── actuarial_params.py
│   └── smoothing_selector.py
│
├── docs/
│   ├── doc_informatique.md              # Ce fichier
│   ├── doc_utilisateur.md
│   └── spec_fonctionnels.md
│
└── outputs/                             # Traces de sessions (logs Markdown)
```

### Flux de données

```
Utilisateur
    │
    │  upload CSV + message texte
    ▼
canvas_app.py  (Dash — thread principal)
    │
    │  parse CSV → DataFrame pandas
    │  sérialise df en JSON (dcc.Store)
    │  lance un thread background
    ▼
_run_writer_in_thread()
    │
    │  rehydrate df depuis JSON
    │  récupère data_store persisté (_writer_state["data_store"])
    ▼
WriterAgent.run_agent_loop(history, df, data_store)
    │
    │  construit system_prompt
    │    ← writer_dialog_prompt.md
    │    ← builder_capabilities.json (catalogue)
    │    ← column_schema.py (mapping colonnes CSV)
    │
    │  appelle openai.chat.completions.create(tools=get_openai_tools())
    │
    │  BOUCLE :
    │    si finish_reason == "tool_calls"
    │      → call_tool(tool_name, function_name, params, df, data_store)
    │           │
    │           ├── statistical_analysis/*.py  → run(df, params)
    │           ├── builder/exposure.py        → run(df, params)
    │           ├── builder/*.py               → run(data_store, params)
    │           │     └── charge notebooks/*.py via _nb_loader
    │           ├── graphs/*.py                → run(data_store, params)
    │           ├── build_pdf/*.py             → run(data_store, params)
    │           └── reasoning/*.py             → run(context, params)
    │      → stocke résultat dans data_store
    │      → yield event "tool_result"
    │    si finish_reason == "stop"
    │      → yield event "message" puis "done"
    │
    ▼
canvas_app.py  (callback poll_agent — toutes les 400 ms)
    │
    │  lit _writer_state["events"]
    │  affiche bulles de chat (texte, tableaux, images)
    │  déclenche téléchargement si output_path dans résultat
    ▼
Utilisateur  (PDF, TXT, .ipynb téléchargés automatiquement)
```

### Qui appelle quoi

| Appelant | Appelé | Méthode |
|---|---|---|
| `canvas_app.py` | `WriterAgent` | Thread background |
| `canvas_app.py` | `tool_registry.get_capabilities()` | Import direct |
| `canvas_app.py` | `column_schema.build_mapping_report()` | Import direct |
| `WriterAgent` | `tool_registry.call_tool()` | Import direct |
| `WriterAgent` | `tool_registry.get_openai_tools()` | Import direct |
| `tool_registry` | `builder/exposure.py`, etc. | `importlib.import_module()` dynamique |
| `builder/*.py` | `notebooks/*.py` | `_nb_loader.load_nb()` via `importlib.util` |
| `builder/*.py` | `column_schema.find_col_by_role()` | Import direct |

---

## 2. Lancer l'application

### Prérequis

```bash
pip install dash dash-bootstrap-components pandas openai reportlab matplotlib
```

La variable d'environnement `OPENAI_API_KEY` doit être définie (ou configurée dans `config.py`).

### Démarrage

```bash
cd "Agent actuariat"
python canvas_app.py
# → http://localhost:8050
```

L'application s'ouvre sur deux onglets :

- **Rapport guidé** : upload CSV à gauche, chat avec l'agent à droite.
- **DEV** : vue des capacités enregistrées (cards), éditeur de code en ligne avec arborescence des fichiers.

### Mode DEV — fonctionnalités utiles

L'onglet DEV permet sans redémarrer l'application :

- Voir le code source de chaque fonction (bouton **Code** sur chaque carte)
- Modifier et sauvegarder un fichier `.py` directement dans l'éditeur
- Ajouter une nouvelle fonction via le formulaire modal (bouton **Ajouter** sur chaque carte) — génère le squelette de code et met à jour `builder_capabilities.json` automatiquement

---

## 3. Les tools actuariels — liste complète et interface

Chaque tool est un répertoire dans `report_agent/tools/`. Chaque fonction est un fichier `.py` exposant une fonction `run()`.

### Convention d'interface

Toutes les fonctions respectent l'une des deux signatures suivantes :

```python
def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    """Pour les tools qui lisent directement le CSV."""

def run(data: dict | None, params: dict | None = None) -> dict:
    """Pour les tools qui consomment des résultats déjà calculés (data_store)."""
```

En cas d'erreur, le résultat contient toujours la clé `"erreur"` (avec un message descriptif). Les tools à succès ne contiennent jamais cette clé.

---

### 3.1 Tool `reasoning`

**Fichier :** `report_agent/tools/reasoning/understand_request.py`

**Interface :** `run(context: dict, params: dict) -> dict`

Le `context` est un dict construit par `WriterAgent` :

```python
context = {
    "user_message": str,   # dernier message de l'utilisateur
    "history":      list,  # historique complet du dialogue
    "csv_columns":  list,  # liste des colonnes du CSV chargé
}
```

**Params :** aucun paramètre requis.

**Output :**
```python
{
    "intent":    str,   # "analyse_descriptive" | "table_mortalite" | "rapport_pdf" | "graphique"
    "entities":  dict,  # entités métier détectées (sexe, période, produit...)
    "questions": list,  # questions à poser si la demande est ambiguë
}
```

**Usage :** appelé en premier, avant tout tool de calcul, pour classifier l'intent.

---

### 3.2 Tool `statistical_analysis`

**Répertoire :** `report_agent/tools/statistical_analysis/`

Toutes les fonctions reçoivent `run(df: pd.DataFrame, params: dict) -> dict`.

#### `data_quality`

**Fichier :** `statistical_analysis/data_quality.py`

**Colonnes requises :** aucune (utilise les colonnes de dates reconnues si présentes)
**Colonnes optionnelles :** `date_entree`, `date_sortie`, `date_naissance`, `cause_sortie`, `sexe`

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `focus` | `str` | `"dates"` | `"dates"` inspecte les colonnes de dates uniquement ; `"all"` ajoute les colonnes catégorielles |
| `max_rows` | `int` | `8` | Nombre max de lignes d'exemple retournées |
| `column` | `str` | — | Nom d'une colonne spécifique à inspecter |

**Output :** `{ "nb_erreurs": int, "table": list[dict], "colonnes_inspectees": list }`

---

#### `portfolio_summary`

**Fichier :** `statistical_analysis/portfolio_summary.py`

**Colonnes requises :** `date_entree`, `cause_sortie`
**Colonnes optionnelles :** `date_sortie`, `date_naissance`, `duree_obs_ans`, `sexe`

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `observation_end` | `str` | — | Date de fin d'observation pour tronquer les dates futures (ex : `"2024-12-31"`) |

**Output :**
```python
{
    "nb_contrats":                  int,
    "nb_deces":                     int,
    "exposition_totale_pa":         float,
    "taux_brut_deces_pour_1000_pa": float,
    "age_min":                      int,
    "age_max":                      int,
    "age_moyen":                    float,
    "date_entree_min":              str,
    "date_entree_max":              str,
    "date_sortie_max":              str,
}
```

**Stocké dans data_store sous la clé :** `"summary"`

---

#### `age_distribution`

**Fichier :** `statistical_analysis/age_distribution.py`

**Colonnes requises :** `date_entree`, `date_naissance`
**Colonnes optionnelles :** `sexe`

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `by_sex` | `bool` | `false` | Si `true`, produit une distribution séparée H/F |
| `band_width` | `int` | `5` | Largeur des tranches d'âge en années |

**Output :**
```python
{
    "distribution":   dict,  # {"20-24": 120, "25-29": 340, ...}
    # si by_sex=true :
    "distribution_H": dict,
    "distribution_F": dict,
}
```

**Stocké dans data_store sous la clé :** `"ages"`

---

#### `time_series`

**Fichier :** `statistical_analysis/time_series.py`

**Colonnes requises :** `date_entree`
**Colonnes optionnelles :** `date_sortie`, `cause_sortie`

**Params :** aucun

**Output :**
```python
{
    "serie": [
        {"annee": int, "nb_entres": int, "nb_deces": int, "exposition_pa": float},
        ...
    ],
    "anomalies": list[str],   # liste de messages d'avertissement si des années sont anormales
}
```

**Stocké dans data_store sous la clé :** `"series"`

---

#### `segmentation`

**Fichier :** `statistical_analysis/segmentation.py`

**Colonnes requises :** aucune
**Colonnes optionnelles :** `sexe`, `produit`, `cause_sortie`

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `columns` | `list[str]` | `[]` | Colonnes à analyser. Si vide, utilise sexe, produit, statut par défaut. |

**Output :**
```python
{
    "segmentations": {
        "sexe": [
            {"valeur": "H", "nb_contrats": int, "nb_deces": int, "pct_contrats": float},
            ...
        ],
        "produit": [...],
    }
}
```

**Stocké dans data_store sous la clé :** `"segmentation"`

---

### 3.3 Tool `builder`

**Répertoire :** `report_agent/tools/builder/`

**Important :** `builder.exposure` reçoit `run(df, params)` (accède au CSV brut). Toutes les autres fonctions `builder` reçoivent `run(data, params)` (consomment le data_store).

Les calculs actuariels réels sont délégués aux modules `notebooks/*.py`, chargés via `_nb_loader.load_nb("02_exposure")` etc.

#### `exposure`

**Fichier :** `builder/exposure.py`
**Interface :** `run(df: pd.DataFrame, params: dict) -> dict`

**Colonnes requises :** `date_naissance`, `date_entree`, `date_sortie`, `cause_sortie`

**Valeurs reconnues comme décès :** `deces`, `décès`, `dcd`, `d`, `dead`, `mort`, `1`, `true`, `oui`, `yes`, `decede`, `deceased`, `death`

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `age_min` | `int` | `20` | Âge minimum du tableau |
| `age_max` | `int` | `90` | Âge maximum du tableau |
| `observation_end` | `str` | `"31/12/2023"` | Date de fin d'observation (les dates sentinelles type 2999 y sont remplacées) |

**Output :**
```python
{
    "exposure_table": [
        {"age": int, "E_x": float, "D_x": int, "mu_x": float, "q_x_brut": float},
        ...
    ],
    "age_min":        int,
    "age_max":        int,
    "total_exposure": float,
    "total_deaths":   int,
    "lignes_exclues": int,   # présent seulement si des lignes ont été exclues
    "note":           str,
}
```

**Stocké dans data_store sous la clé :** `"exposure_table"` (liste directe, pas le dict entier)

---

#### `crude_rates`

**Fichier :** `builder/crude_rates.py`
**Interface :** `run(data: dict, params: dict) -> dict`

**Requiert dans data_store :** `data["exposure_table"]`

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `method` | `str` | `"central"` | `"central"` (méthode du taux central) ou `"binomial"` |

**Output :**
```python
{
    "qx_table": [
        {"age": int, "q_x": float, "var_q_x": float},
        ...
    ]
}
```

**Stocké dans data_store sous la clé :** `"qx_table"` (liste directe)

---

#### `smoothing`

**Fichier :** `builder/smoothing.py`
**Interface :** `run(data: dict, params: dict) -> dict`

**Requiert dans data_store :** `data["qx_table"]`

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `method` | `str` | `"whittaker"` | `"whittaker"` \| `"gompertz"` \| `"makeham"` \| `"spline"` |
| `lambda_wh` | `float` | `100` | Pénalité de lissage Whittaker-Henderson |
| `age_min_fit` | `int` | `40` | Âge de début d'ajustement pour Gompertz/Makeham |

**Output :**
```python
{
    "smoothed_table": [
        {"age": int, "q_x_lisse": float},
        ...
    ],
    "method": str,
}
```

**Stocké dans data_store sous la clé :** `"smoothed_table"` (liste directe)

---

#### `diagnostics`

**Fichier :** `builder/diagnostics.py`
**Interface :** `run(data: dict, params: dict) -> dict`

**Requiert dans data_store :** `data["exposure_table"]` (et `data["smoothers_dict"]` pour `compare_smoothers`)

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `function_name` | `str` | `"credibility"` | `"credibility"` \| `"compare_smoothers"` \| `"smr"` |
| `threshold` | `int` | `10` | Seuil E_x pour la crédibilité |
| `sexe` | `str` | `"H"` | `"H"` ou `"F"` pour la table de référence SMR |

**Output (credibility) :**
```python
{
    "credibility": [
        {"age": int, "E_x": float, "credible": bool},
        ...
    ],
    "nb_credible": int,
    "pct_credible": float,
}
```

**Stocké dans data_store sous la clé :** `"diagnostics"`

---

#### `validation`

**Fichier :** `builder/validation.py`
**Interface :** `run(data: dict, params: dict) -> dict`

**Requiert dans data_store :** `data["exposure_table"]` (et optionnellement `data["smoothed_table"]`)

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `function_name` | `str` | `"confidence_intervals"` | `"confidence_intervals"` \| `"chi_square"` |
| `alpha` | `float` | `0.05` | Niveau de risque |
| `sexe` | `str` | `"H"` | Table de référence pour le test chi2 |

**Output (confidence_intervals) :**
```python
{
    "ci_table": [
        {"age": int, "q_x": float, "lower": float, "upper": float},
        ...
    ]
}
```

**Stocké dans data_store sous la clé :** `"validation"`

---

#### `benchmarking`

**Fichier :** `builder/benchmarking.py`
**Interface :** `run(data: dict, params: dict) -> dict`

**Requiert dans data_store :** `data["exposure_table"]` (pour `abatement_factors`)

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `function_name` | `str` | `"abatement_factors"` | `"abatement_factors"` \| `"load_reference_table"` |
| `reference_name` | `str` | `"TH0002"` | `"TH0002"` \| `"TF0002"` \| `"TD8890"` \| `"TPRV93"` |
| `sexe` | `str` | `"H"` | `"H"` ou `"F"` |
| `qx_exp_col` | `str` | `"q_x_lisse"` | Colonne des taux d'expérience à comparer |

**Output (abatement_factors) :**
```python
{
    "abatement_table": [
        {"age": int, "q_x_exp": float, "q_x_ref": float, "abatement_factor": float},
        ...
    ],
    "smr_global":     float,
    "reference_name": str,
    "summary":        dict,
}
```

**Stocké dans data_store sous la clé :** `"benchmarking"`

---

### 3.4 Tool `graphs`

**Répertoire :** `report_agent/tools/graphs/`

Toutes les fonctions reçoivent `run(data: dict, params: dict) -> dict` et retournent un PNG encodé en base64.

#### `sample_gallery`

**Fichier :** `graphs/sample_gallery.py`
**Ne nécessite pas de données.** Génère des mini-graphiques avec données synthétiques.

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `filter` | `str` | `"all"` | `"descriptive"` \| `"builder"` \| `"all"` |

**Output :**
```python
{
    "samples": [
        {"title": str, "description": str, "image_b64": str},
        ...
    ],
    "n_samples": int,
}
```

---

#### `analysis_plots`

**Fichier :** `graphs/analysis_plots.py`

**Requiert dans data_store :** selon le graphique demandé (`ages`, `series`, `segmentation`)

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `chart` | `str` | `"age_pyramid"` | `"age_pyramid"` \| `"time_series"` \| `"segmentation"` |
| `by_sex` | `bool` | `false` | Pyramide H/F séparée si `true` |
| `title_suffix` | `str` | `""` | Texte ajouté au titre du graphique |

**Output :** `{"chart": str, "image_b64": str}`

---

#### `builder_plots`

**Fichier :** `graphs/builder_plots.py`

**Requiert dans data_store :** selon le graphique (`exposure_table`, `smoothed_table`, `smr`)

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `chart` | `str` | `"exposure"` | `"exposure"` \| `"crude_smoothed"` \| `"smr"` |
| `sexe` | `str` | `"H"` | `"H"` ou `"F"` pour la courbe de référence TH/TF |
| `title_suffix` | `str` | `""` | Texte ajouté au titre |

**Output :** `{"chart": str, "image_b64": str}`

---

### 3.5 Tool `build_pdf`

**Répertoire :** `report_agent/tools/build_pdf/`

Toutes les fonctions reçoivent `run(data: dict, params: dict) -> dict`.

#### `descriptive_report`

**Fichier :** `build_pdf/descriptive_report.py`

**Requiert dans data_store :** `"summary"` (obligatoire), `"ages"`, `"series"`, `"segmentation"`, `"narrative"` (tous optionnels)

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `output_path` | `str` | `"/tmp/rapport_descriptif.pdf"` | Chemin de sortie du PDF |
| `title` | `str` | `"Analyse descriptive du portefeuille"` | Titre du rapport |

**Output :** `{"succes": true, "output_path": str, "nb_pages_estimees": int}`

---

#### `certification_report`

**Fichier :** `build_pdf/certification_report.py`

Rapport PDF complet de la table de mortalité d'expérience. Appeler après avoir exécuté toute la pipeline builder.

**Requiert dans data_store :** résultats de `exposure`, `crude_rates`, `smoothing`, `diagnostics`, `validation`, `benchmarking`

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `output_path` | `str` | `"/tmp/rapport_certification.pdf"` | Chemin de sortie |
| `title` | `str` | — | Titre du rapport |
| `portfolio_info` | `str` | — | Description du portefeuille (ex : `"45 000 lignes, 2000-2023"`) |
| `sexe` | `str` | `"H"` | Table de référence à utiliser |

**Output :** `{"succes": true, "output_path": str}`

---

#### `session_log`

**Fichier :** `build_pdf/session_log.py`

Génère un fichier TXT contenant le raisonnement de l'agent, la séquence complète d'appels et un bloc JSON REPLAY pour rejouer l'analyse.

**Requiert dans data_store :** `_call_log` et `_reasoning_log` (accumulés automatiquement)

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `output_path` | `str` | `"/tmp/session_actuarielle.txt"` | Chemin de sortie |
| `portfolio_info` | `str` | — | Description courte du portefeuille |

**Output :** `{"succes": true, "output_path": str}`

---

#### `generate_notebook`

**Fichier :** `build_pdf/generate_notebook.py`

Génère un Jupyter notebook `.ipynb` reproduisant toute la session d'analyse. Chaque appel de fonction devient une cellule Python exécutable.

**Requiert dans data_store :** `_call_log` et `_reasoning_log`

**Params :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `output_path` | `str` | `"/tmp/analyse_actuarielle.ipynb"` | Chemin de sortie |
| `portfolio_info` | `str` | — | Description du portefeuille |
| `csv_filename` | `str` | — | Nom du fichier CSV à utiliser dans le notebook |

**Output :** `{"succes": true, "output_path": str}`

---

## 4. data_store — structure et accumulation

### Principe

Le `data_store` est un dictionnaire Python ordinaire qui persiste pendant toute la durée d'une session (entre les messages de l'utilisateur). Il est stocké dans `_writer_state["data_store"]` (variable globale de `canvas_app.py`, protégée par un `threading.Lock`).

Lorsqu'un nouvel upload CSV est effectué, le `data_store` est réinitialisé à `{}`.

### Clés accumulées pendant une session

| Clé | Type | Produit par | Description |
|---|---|---|---|
| `"summary"` | `dict` | `statistical_analysis.portfolio_summary` | Résumé global du portefeuille |
| `"ages"` | `dict` | `statistical_analysis.age_distribution` | Distribution des âges |
| `"series"` | `dict` | `statistical_analysis.time_series` | Série temporelle annuelle |
| `"segmentation"` | `dict` | `statistical_analysis.segmentation` | Répartitions catégorielles |
| `"exposure_table"` | `list[dict]` | `builder.exposure` | Table E_x / D_x par âge |
| `"qx_table"` | `list[dict]` | `builder.crude_rates` | Taux bruts q_x par âge |
| `"smoothed_table"` | `list[dict]` | `builder.smoothing` | Taux lissés q_x par âge |
| `"diagnostics"` | `dict` | `builder.diagnostics` | Résultats diagnostics (crédibilité, SMR...) |
| `"validation"` | `dict` | `builder.validation` | Intervalles de confiance, test chi2 |
| `"benchmarking"` | `dict` | `builder.benchmarking` | Facteurs d'abattement, SMR global |
| `"_call_log"` | `list[dict]` | `WriterAgent` (automatique) | Log de chaque appel tool de la session |
| `"_reasoning_log"` | `list[str]` | `WriterAgent` (automatique) | Textes de raisonnement produits par l'agent |

### Structure de `_call_log`

Chaque entrée est ajoutée automatiquement par `WriterAgent` après chaque appel tool :

```python
{
    "step":          int,    # numéro de l'appel (commence à 1)
    "tool":          str,    # ex : "builder"
    "function_name": str,    # ex : "exposure"
    "params":        dict,   # paramètres passés
    "result_summary": dict,  # résumé tronqué (listes → "[N lignes]", str tronquées à 300 chars)
    "has_error":     bool,   # True si la clé "erreur" est dans le résultat
}
```

Les clés `image_b64` et `samples` sont exclues du `result_summary` pour éviter de saturer la mémoire.

### Structure de `_reasoning_log`

Liste de chaînes de caractères. Chaque message texte de l'agent (hors tool calls) y est ajouté dans l'ordre.

### Persistance entre appels

Lors de chaque message de l'utilisateur, le thread background récupère le `data_store` existant :

```python
with _writer_lock:
    data_store = _writer_state["data_store"]
```

L'objet est passé par référence à `WriterAgent.run_agent_loop()`, qui le mute directement. Ainsi, une session peut enchaîner plusieurs messages et chaque nouveau message bénéficie des résultats calculés dans les messages précédents.

---

## 5. WriterAgent — cycle de vie

**Fichier :** `report_agent/writer_agent.py`

### Instanciation

```python
from report_agent.writer_agent import WriterAgent
writer = WriterAgent(model="gpt-4o")
```

Le client OpenAI est créé en lazy (premier appel à `_llm`). Il utilise `openai.OpenAI()` (lit la clé depuis la variable d'environnement `OPENAI_API_KEY`).

### Boucle principale

```python
for event in writer.run_agent_loop(history, df=df, data_store=data_store):
    print(event)
```

**Signature complète :**
```python
def run_agent_loop(
    self,
    history:    list[dict],         # historique [{role, content}]
    df:         pd.DataFrame | None = None,
    data_store: dict | None         = None,
    csv_path:   str | None          = None,  # charge le CSV si df=None
) -> Generator[dict, None, None]:
```

### Types d'événements yielded

| Type | Clés | Signification |
|---|---|---|
| `"tool_call"` | `tool`, `function_name`, `params`, `tool_call_id` | L'agent va appeler ce tool |
| `"tool_result"` | `tool`, `function_name`, `result`, `tool_call_id` | Résultat du tool (dict complet) |
| `"message"` | `content` | Message texte de l'agent |
| `"done"` | — | Fin normale de la session |
| `"error"` | `message` | Erreur fatale (API, limite d'étapes, etc.) |

### Construction du system prompt

La méthode `_build_system_prompt(df)` :

1. Charge `report_agent/writer_dialog_prompt.md` (instructions comportementales de l'agent)
2. Charge `builder_capabilities.json` via `get_capabilities()` et l'injecte en JSON
3. Si un DataFrame est fourni :
   - Appelle `build_mapping_report(df, caps)` depuis `column_schema.py`
   - Injecte un tableau Markdown des colonnes détectées (rôle → colonne → statut)
   - Injecte un tableau de disponibilité par fonction
   - Ajoute des questions à poser si des colonnes requises sont manquantes

### Mécanisme de boucle tool-calling OpenAI

```
messages = [system_prompt] + history

BOUCLE (max 20 étapes) :
  1. POST openai.chat.completions.create(messages, tools=get_openai_tools())
  2. Si finish_reason == "tool_calls" :
       Pour chaque tool_call :
         - Extraire fn_name, fn_args (JSON)
         - Appeler call_tool(fn_name, function_name, params, df, data_store)
         - Stocker résultat dans data_store (clé dépend de function_name)
         - Ajouter dans _call_log
         - Yield "tool_result"
         - Ajouter message tool dans messages (images tronquées : "<image base64 tronquée>")
  3. Si finish_reason == "stop" :
       - Yield "message" (contenu textuel)
       - Si "stop" ou "<FIN>" dans contenu → yield "done", return
```

La limite `MAX_STEPS = 20` protège contre les boucles infinies.

### Troncature des images dans les messages

Lors de la construction du message `role: "tool"` injecté dans l'historique OpenAI, la clé `image_b64` est remplacée par la chaîne `"<image base64 tronquée>"`. Cela empêche de saturer la fenêtre de contexte (un PNG en base64 peut représenter plusieurs dizaines de Ko). L'image complète reste dans `_writer_state["events"]` pour l'affichage UI.

---

## 6. tool_registry.py — routage

**Fichier :** `report_agent/tools/tool_registry.py`

### `get_capabilities() -> dict`

Charge et retourne le contenu de `builder_capabilities.json`. Utilisé par `WriterAgent` (system prompt) et `canvas_app.py` (affichage DEV).

### `get_openai_tools() -> list[dict]`

Construit la liste des tools au format OpenAI function-calling. Pour chaque entrée `tools.*` dans les capabilities, crée un tool OpenAI avec :

- Un paramètre `function_name` (enum des fonctions disponibles)
- Un paramètre `params` (objet libre)

Les fonctions avec `"disponible": false` sont exclues de l'enum.

### `call_tool(tool_name, function_name, params, df, data, context) -> dict`

Logique de routage :

```
call_tool("builder", "exposure", params, df=df, data=data_store)
    │
    ├─ 1. Vérifie que "builder" existe dans builder_capabilities.json["tools"]
    │      → sinon vérifie dans ["hors_perimetre"] → retourne erreur métier
    │
    ├─ 2. Vérifie que "exposure" existe dans les fonctions du tool
    │      → sinon retourne {"erreur": "Fonction inconnue"}
    │
    ├─ 3. Vérifie que disponible != false
    │
    ├─ 4. Importe dynamiquement report_agent.tools.builder.exposure
    │      → importlib.import_module("report_agent.tools.builder.exposure")
    │
    └─ 5. Dispatche selon le type de tool :
          ┌─ statistical_analysis   → mod.run(df, params)
          ├─ builder.exposure       → mod.run(df, params)
          ├─ builder.*              → mod.run(data_store, params)
          ├─ reasoning              → mod.run(context, params)
          └─ graphs, build_pdf      → mod.run(data_store, params)
```

En cas d'exception Python dans `run()`, retourne :
```python
{"erreur": "Erreur lors de l'exécution de ...", "traceback": str}
```

### Constantes de routage

```python
_DF_TOOLS = {"statistical_analysis"}          # reçoivent toujours df
_BUILDER_DF_FUNCTIONS = {"exposure"}          # builder avec df
```

---

## 7. column_schema.py — mapping automatique

**Fichier :** `report_agent/dictionary/column_schema.py`

### Contenu de `COLUMN_SCHEMA`

Dictionnaire `{role: {label, question, candidates}}`. C'est la **source unique** pour tous les noms de colonnes. Rôles disponibles :

| Rôle | Label | Candidates (insensible à la casse) |
|---|---|---|
| `date_entree` | Date d'entrée en observation | `date_entree`, `ctreffet`, `entry_date`, `date_d_entree` |
| `date_sortie` | Date de sortie | `date_sortie`, `exit_date`, `date_de_sortie` |
| `date_naissance` | Date de naissance | `date_naissance`, `clinaiss`, `dob`, `birth_date` |
| `cause_sortie` | Cause de sortie | `cause_sortie`, `statut`, `status`, `cause` |
| `sexe` | Sexe de l'assuré | `sexe`, `sexeref`, `gender`, `sex` |
| `produit` | Produit / type de contrat | `cdprod`, `produit`, `product`, `type_contrat` |
| `duree_obs_ans` | Durée d'observation (P-A) | `duree_obs_ans`, `duree_obs`, `exposition`, `exposure` |

### `find_col_by_role(df, role) -> str | None`

Fonction principale utilisée par tous les tools pour détecter les colonnes :

```python
from report_agent.dictionary.column_schema import find_col_by_role

date_col = find_col_by_role(df, "date_entree")
# Retourne le nom de colonne réel dans df, ou None si absent
# Exemple : si le CSV a "ctreffet", retourne "ctreffet"
```

Mécanisme : compare les candidats du rôle (en minuscules) aux colonnes du DataFrame (en minuscules). Retourne le nom de colonne tel qu'il apparaît dans le DataFrame (casse originale préservée).

### `build_mapping_report(df, capabilities) -> dict`

Utilisée par `canvas_app.py` (badge visuel) et `WriterAgent` (system prompt) :

```python
report = build_mapping_report(df, caps)
# {
#   "matched":      {"date_entree": "ctreffet", "sexe": "sexeref", ...},
#   "unmatched":    {"produit": {"label": "...", "question": "..."}},
#   "unknown_cols": ["col_inconnue_1"],
#   "fn_readiness": {
#       "portfolio_summary": {"ready": True, "missing_required": [], "missing_optional": ["duree_obs_ans"]},
#       "exposure":          {"ready": False, "missing_required": ["date_sortie"], ...},
#       ...
#   }
# }
```

### Ajouter un nouveau rôle

Éditer `column_schema.py` et ajouter une entrée dans `COLUMN_SCHEMA` :

```python
"mon_role": {
    "label": "Ma colonne métier",
    "question": "Quelle colonne correspond à ... ?",
    "candidates": ["mon_col", "my_col", "alias_possible"],
},
```

Le rôle sera alors automatiquement :
- détecté dans les CSV uploadés
- affiché dans le badge de mapping de l'interface
- injecté dans le system prompt de l'agent
- disponible dans les tools via `find_col_by_role(df, "mon_role")`

---

## 8. builder_capabilities.json — format

**Fichier :** `report_agent/builder_capabilities.json`

C'est la **source de vérité** de toutes les capacités de l'agent. Il est lu dynamiquement à chaque session (pas de cache au démarrage).

### Structure complète

```json
{
  "version": "2.0",

  "tools": {
    "<tool_name>": {
      "description": "Description du tool pour l'agent et l'UI DEV.",
      "functions": {
        "<function_name>": {
          "description":       "Description de la fonction (injectée dans le tool OpenAI).",
          "required_columns":  ["role1", "role2"],
          "optional_columns":  ["role3"],
          "column_notes":      "Explications sur les colonnes pour l'agent.",
          "params": {
            "param1": "type — description (défaut : valeur)",
            "param2": "type — description"
          },
          "disponible": true   // optionnel, défaut true. Si false, la fonction est masquée.
        }
      }
    }
  },

  "hors_perimetre": {
    "<tool_name>": {
      "disponible": false,
      "raison": "Explication pourquoi ce tool n'est pas disponible."
    }
  }
}
```

### Champs importants

- **`required_columns`** et **`optional_columns`** : liste de **rôles** (clés de `COLUMN_SCHEMA`), pas de noms de colonnes bruts. Utilisés par `build_mapping_report()` pour calculer `fn_readiness`.
- **`params`** : dictionnaire informatif (valeurs = description en texte libre). Ces descriptions sont injectées dans le system prompt pour guider l'agent. Elles ne sont pas validées par le code.
- **`disponible: false`** : masque la fonction dans `get_openai_tools()` (elle n'est pas proposée à l'agent). Un appel à `call_tool()` sur une fonction masquée retourne `{"erreur": "... non disponible"}`.
- **`hors_perimetre`** : tools qui ne seront jamais implémentés (ex : modules non-vie). Permet de retourner un message d'erreur métier explicite si l'agent tente de les appeler.

### Ajouter une nouvelle fonction à un tool existant

```json
// Dans tools.statistical_analysis.functions :
"lapse_analysis": {
  "description": "Analyse des rachats par ancienneté de contrat.",
  "required_columns": ["date_entree", "cause_sortie"],
  "optional_columns": ["produit"],
  "column_notes": "cause_sortie doit contenir les valeurs rachat reconnues.",
  "params": {
    "bin_size": "int — taille des tranches en années (défaut : 1)"
  }
}
```

La fonction sera immédiatement visible dans l'onglet DEV et proposée à l'agent dès la prochaine session.

---

## 9. Ajouter un nouveau tool — guide pas-à-pas

### Étape 1 : créer le fichier `.py`

Créer `report_agent/tools/<tool>/<fonction>.py`. Le nom du fichier = le nom de la fonction.

**Template pour une fonction qui lit le DataFrame :**

```python
"""
report_agent/tools/<tool>/<fonction>.py
<description courte>

INPUTS
  Colonnes requises : date_entree, cause_sortie
  Params :
    mon_param : int — description (défaut : 10)

OUTPUT (dict)
  resultat : list[dict]
  total    : float

Interface : run(df, params) -> dict
"""
from __future__ import annotations
import pandas as pd
from report_agent.dictionary.column_schema import find_col_by_role


def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    params = params or {}

    # Détecter les colonnes via le schema
    date_col  = find_col_by_role(df, "date_entree")
    death_col = find_col_by_role(df, "cause_sortie")

    if date_col is None:
        return {"erreur": "Colonne date_entree non trouvée dans le CSV."}

    mon_param = int(params.get("mon_param", 10))

    # ... calcul ...
    resultat = []
    total = 0.0

    return {"resultat": resultat, "total": total}
```

**Template pour une fonction qui consomme le data_store :**

```python
from __future__ import annotations


def run(data: dict | None, params: dict | None = None) -> dict:
    data = data or {}
    params = params or {}

    # Vérifier que les données requises sont présentes
    exposure = data.get("exposure_table")
    if not exposure:
        return {"erreur": "exposure_table manquant. Appeler builder.exposure d'abord."}

    # ... calcul ...
    return {"mon_resultat": ...}
```

### Étape 2 : enregistrer dans `builder_capabilities.json`

Ajouter la fonction dans la section `tools.<tool_name>.functions` :

```json
"ma_fonction": {
  "description": "Ce que fait la fonction (vu par l'agent et l'UI DEV).",
  "required_columns": ["date_entree"],
  "optional_columns": ["sexe"],
  "column_notes": "Explication pour l'agent sur l'usage des colonnes.",
  "params": {
    "mon_param": "int — description (défaut : 10)"
  }
}
```

Si c'est un nouveau tool (nouveau répertoire), ajouter aussi l'entrée `tools.<nouveau_tool>` avec son `description`.

### Étape 3 : vérifier le routage dans `tool_registry.py`

Le routage est automatique pour les cas standards. Vérifier uniquement si le nouveau tool ne rentre pas dans les catégories existantes :

- Nouveau tool avec `run(df, params)` → ajouter son nom dans `_DF_TOOLS`
- Nouveau tool avec `run(data, params)` → aucun changement nécessaire
- Cas particulier (contexte custom) → ajouter un `elif tool_name == "mon_tool"` dans `call_tool()`

### Étape 4 : tester manuellement

```python
import pandas as pd
from report_agent.tools.tool_registry import call_tool

df = pd.read_csv("mon_fichier.csv", sep=";")

# Test fonction avec df
result = call_tool(
    tool_name="statistical_analysis",
    function_name="ma_fonction",
    params={"mon_param": 5},
    df=df,
)
print(result)

# Test fonction avec data_store
data_store = {"exposure_table": [...]}
result = call_tool(
    tool_name="builder",
    function_name="ma_fonction",
    params={},
    data=data_store,
)
print(result)
```

### Étape 5 : vérifier dans l'interface DEV

Lancer `python canvas_app.py`, aller dans l'onglet DEV → Capacités → cliquer "Rafraîchir". La nouvelle fonction doit apparaître sur la carte du tool correspondant.

---

## 10. Gestion des images et PDF

### Graphiques matplotlib — encodage base64

Toutes les fonctions `graphs/*.py` suivent le même pattern :

```python
import base64, io
import matplotlib
matplotlib.use("Agg")  # backend non-interactif (pas besoin d'écran)
import matplotlib.pyplot as plt

def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#FBF8F1")
    plt.close(fig)  # libère la mémoire
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def run(data, params=None) -> dict:
    fig, ax = plt.subplots(...)
    # ... construction du graphique ...
    return {"chart": "nom_du_chart", "image_b64": _to_b64(fig)}
```

### Affichage dans le chat (canvas_app.py)

Lorsque le résultat d'un tool contient `image_b64`, `canvas_app.py` l'affiche via :

```python
html.Img(
    src=f"data:image/png;base64,{result['image_b64']}",
    style={"maxWidth": "100%", "borderRadius": "6px"},
)
```

Le cas `samples` (galerie multi-images) affiche plusieurs images en grille 2 colonnes.

### Troncature dans le contexte OpenAI

Pour éviter de saturer la fenêtre de contexte, `WriterAgent` remplace `image_b64` par `"<image base64 tronquée>"` dans le message `role: "tool"` envoyé à l'API :

```python
result_for_msg = {
    k: ("<image base64 tronquée>" if k == "image_b64" else v)
    for k, v in result.items()
}
```

L'image complète reste dans `_writer_state["events"]` pour l'affichage UI.

### Génération PDF avec ReportLab

Les fonctions `build_pdf/*.py` utilisent ReportLab (bibliothèque pure Python, pas de dépendance système).

```python
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate(output_path, pagesize=A4, ...)
story = [Paragraph("Titre", style), Table(data, ...), ...]
doc.build(story)
```

### Patch md5 Python 3.9+ / OpenSSL

Sur certaines distributions macOS/Linux avec OpenSSL 3, `hashlib.md5()` lève une exception si appelé sans `usedforsecurity=False`. ReportLab utilise `md5()` en interne. Le patch est appliqué juste avant l'import de ReportLab et restauré après :

```python
import hashlib as _hashlib
_orig_md5 = _hashlib.md5

def _md5_compat(*a, **kw):
    kw.pop("usedforsecurity", None)  # supprime le paramètre si présent
    return _orig_md5(*a, **kw)

_hashlib.md5 = _md5_compat

try:
    from reportlab.platypus import ...
    # ... génération du PDF ...
    doc.build(story)
finally:
    _hashlib.md5 = _orig_md5  # toujours restaurer
```

Ce patch est dupliqué dans `descriptive_report.py` et `certification_report.py`.

### Téléchargement automatique dans l'interface

Lorsque `build_pdf` retourne `{"succes": true, "output_path": "/tmp/fichier.pdf"}`, le callback `poll_agent` de `canvas_app.py` détecte l'extension et met à jour le store correspondant (`store-pdf-path`, `store-txt-path`, `store-notebook-path`). Les callbacks de téléchargement déclenchent alors `dcc.send_file(path)` automatiquement.

---

## 11. Documents de contexte (uploads additionnels)

### Situation actuelle

Dans la version actuelle (v2.0), **l'interface n'expose pas de zone d'upload pour des documents de contexte** (PDFs, notes, tables de référence externes). L'upload de `canvas_app.py` est dédié uniquement au fichier CSV de portefeuille.

### Fonctionnement des données du portefeuille

Le CSV uploadé est :
1. Parsé par `_parse_csv()` avec détection automatique du séparateur (`;`, `,`, `\t`, `|`) et de l'encodage (`utf-8`, `latin-1`)
2. Sérialisé en JSON via `df.to_json(orient="split")` et stocké dans `dcc.Store` (côté client Dash, dans le navigateur)
3. Rehydraté à chaque appel agent via `pd.read_json(StringIO(df_json), orient="split")`

### Injection dans le system prompt

La connaissance du CSV est transmise à l'agent via le system prompt (pas directement dans les messages) :

```
## Données du portefeuille chargées — 45 231 lignes, 8 colonnes

### Mapping automatique des colonnes

| Rôle | Colonne détectée | Statut |
|---|---|---|
| Date d'entrée en observation | `ctreffet` | ✓ auto |
| Date de sortie | `date_sortie` | ✓ auto |
| Sexe de l'assuré | — | ❌ absent |
...

### Disponibilité des fonctions

| Fonction | Prêt | Colonnes requises manquantes | Notes |
|---|---|---|---|
| `portfolio_summary` | ✓ | — | Optionnel absent : duree_obs_ans |
| `exposure` | ✓ | — | — |
...
```

### Ajouter le support de documents de contexte (évolution future)

Pour permettre l'upload de PDFs ou notes en contexte :

1. Ajouter un second composant `dcc.Upload` dans `_writer_tab()` (acceptant `.pdf`, `.txt`, `.md`)
2. Décoder le contenu base64, extraire le texte (pdfplumber ou PyPDF2 pour les PDFs)
3. Stocker le texte extrait dans un `dcc.Store` dédié
4. Dans `WriterAgent._build_system_prompt()`, injecter le texte en fin de prompt :

```python
if context_docs:
    prompt += "\n\n## Documents de contexte uploadés par le client\n\n"
    for doc in context_docs:
        prompt += f"### {doc['filename']}\n\n{doc['text'][:3000]}\n\n"
```

---

## Annexe — Séquences d'appels types

### Analyse descriptive complète

```
reasoning.understand_request
statistical_analysis.portfolio_summary       → data_store["summary"]
statistical_analysis.age_distribution        → data_store["ages"]
statistical_analysis.time_series             → data_store["series"]
statistical_analysis.segmentation            → data_store["segmentation"]
graphs.analysis_plots (chart=age_pyramid)    → image_b64
graphs.analysis_plots (chart=time_series)    → image_b64
build_pdf.descriptive_report                 → /tmp/rapport_descriptif.pdf
```

### Pipeline de table de mortalité

```
reasoning.understand_request
builder.exposure           → data_store["exposure_table"]
builder.crude_rates        → data_store["qx_table"]
builder.diagnostics        → data_store["diagnostics"]       (crédibilité)
builder.smoothing          → data_store["smoothed_table"]
graphs.builder_plots       → image_b64                       (crude_smoothed)
builder.validation         → data_store["validation"]        (IC Poisson)
builder.benchmarking       → data_store["benchmarking"]      (abattements TH0002)
graphs.builder_plots       → image_b64                       (smr)
build_pdf.certification_report → /tmp/rapport_certification.pdf
```

### Génération de livrables après analyse

```
build_pdf.session_log           → /tmp/session_actuarielle.txt
build_pdf.generate_notebook     → /tmp/analyse_actuarielle.ipynb
```
