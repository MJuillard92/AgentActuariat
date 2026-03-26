# Agent Actuariel — Architecture

## Point d'entrée

| Fichier | Rôle |
|---|---|
| `canvas_app.py` | Application Dash (UI principale) — lancer avec `python canvas_app.py` |
| `config.py` | Paramètres globaux (chemins, modèle LLM, répertoires) |
| `.env` | Clé API OpenAI (ne pas versionner) |

## Moteur agent

| Fichier | Rôle |
|---|---|
| `agent.py` | Boucle ReAct — raisonnement + appels d'outils via l'API OpenAI |
| `workflow.py` | Modèle de données du workflow (nœuds = notebooks, arêtes = dépendances) |
| `workflow_executor.py` | Exécution du workflow + création du kernel Python partagé |
| `notebook_runner.py` | Chargement et exécution des cellules de notebooks `.ipynb` |
| `word_generator.py` | Génération du rapport Word final |

## Bibliothèque actuarielle (`notebooks/`)

Deux types de fichiers coexistent :

### Modules `.py` — fonctions appelables par l'agent

| Fichier | Contenu |
|---|---|
| `01_data_preparation.py` | Chargement, nettoyage, calcul des âges, détection d'anomalies |
| `02_exposure.py` | Calcul des expositions (méthode centrale, par année) |
| `03_crude_rates.py` | Taux bruts : méthode centrale, binomiale, Kaplan-Meier |
| `04_smoothing.py` | Lissage : Whittaker-Henderson, Gompertz, Makeham, spline, polynomial local |
| `05_diagnostics.py` | Diagnostic : crédibilité, monotonie, comparaison de modèles, SMR |
| `06_validation.py` | Validation : IC Poisson, chi-deux, marge de prudence, modèle de Cox |
| `07_benchmarking.py` | Tables de référence (TH/TF 00-02, TD 88-90, TPRV93), abattements, régression logit |
| `08_visualization.py` | Graphiques (retournent des bytes PNG) |

### Notebooks `.ipynb` — documentation narrative de l'ancienne approche

Scripts séquentiels d'origine, conservés comme référence méthodologique.

## Données

| Dossier | Contenu |
|---|---|
| `Portefeuille/` | Données d'entrée (portefeuilles assurés) |
| `uploads/` | Fichiers déposés via l'interface |
| `outputs/` | Tables de mortalité générées (CSV) |
| `offline/` | Module autonome — pipeline offline (ne pas modifier) |

## Documentation

| Dossier/Fichier | Contenu |
|---|---|
| `docs/` | Documents techniques (LaTeX, PDF) |
| `instruction_pour_claude.md` | Instructions de contexte pour l'IA |
