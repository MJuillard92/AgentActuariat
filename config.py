# ─────────────────────────────────────────────────────────────────────────────
# Configuration de l'agent actuariel
# Modifier ce fichier pour changer les modèles, les limites ou les chemins.
#
# Architecture 4-LLM :
#   PLANNING_MODEL   — phase de planification (raisonnement structuré, JSON)
#   REASONING_MODEL  — boucle ReAct de l'agent (code, décisions métier)
#   FORMATTER_MODEL  — réponses RAG et mise en forme (léger, bon marché)
#   ANALYSIS_MODEL   — analyse structurée des PDFs (extraction JSON)
#
# Préconisations OpenAI (performance / coût) :
#   Planning      → o4-mini         : raisonnement structuré, JSON fiable (~$1.1/1M)
#   ReAct/code    → gpt-4o-mini     : rapide, excellent code Python (~$0.15/1M)
#   Formatage     → gpt-4o-mini     : suffisant pour synthèse et RAG
#   Analyse PDF   → gpt-4o          : nécessaire pour extraction JSON complexe
# ─────────────────────────────────────────────────────────────────────────────

# Phase de planification — génère le plan structuré, formules, méthodes
PLANNING_MODEL = "o4-mini"

# Agent ReAct principal — exécution code Python, décisions métier step par step
REASONING_MODEL = "gpt-4o-mini"

# Réponses RAG et mise en forme des rapports — synthèse, questions/réponses
FORMATTER_MODEL = "gpt-4o-mini"

# Analyse structurée des PDFs de référence — extraction JSON du template
ANALYSIS_MODEL = "gpt-4o"

# Alias de compatibilité (ancienne variable utilisée dans certains modules)
MODEL = REASONING_MODEL

# Paramètres de génération de l'agent
MAX_TOKENS = 16000          # max_tokens pour modèles non-o (gpt-4o, gpt-4o-mini : limite 16384)
MAX_COMPLETION_TOKENS = 40000  # max_completion_tokens pour modèles o-series (o4-mini, o3…)
TEMPERATURE = 0.2

# Chemins
NOTEBOOKS_DIR = "./notebooks"          # Répertoire contenant les notebooks individuels
UPLOADS_DIR = "./uploads"              # Répertoire pour les fichiers CSV uploadés
REPORT_BLUEPRINTS_DIR = "./offline/outputs"  # Répertoire des blueprints JSON validés offline

# Paramètres métier actuariels — re-export depuis actuarial_params pour accès unifié
# Modifier les seuils dans actuarial_params.py (pas ici)
from actuarial_params import PARAMS as ACTUARIAL_PARAMS  # noqa: E402
