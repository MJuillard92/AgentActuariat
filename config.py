# ─────────────────────────────────────────────────────────────────────────────
# Configuration de l'agent actuariel
# Modifier ce fichier pour changer les modèles, les limites ou les chemins.
#
# Architecture 3-LLM :
#   REASONING_MODEL  — boucle ReAct de l'agent (raisonnement complexe, multi-étapes)
#   FORMATTER_MODEL  — réponses RAG et mise en forme des rapports (léger, bon marché)
#   ANALYSIS_MODEL   — analyse structurée des PDFs de référence (extraction JSON)
#
# Préconisations OpenAI (performance / coût) :
#   Raisonnement  → gpt-4o          : meilleur rapport qualité/coût pour ReAct
#   Formatage     → gpt-4o-mini     : suffisant pour synthèse et RAG
#   Analyse PDF   → gpt-4o          : nécessaire pour l'extraction JSON structurée
# ─────────────────────────────────────────────────────────────────────────────

# Agent ReAct principal — raisonnement, planification, décisions métier
REASONING_MODEL = "o4-mini"

# Réponses RAG et mise en forme des rapports — synthèse, questions/réponses
FORMATTER_MODEL = "gpt-4o-mini"

# Analyse structurée des PDFs de référence — extraction JSON du template
ANALYSIS_MODEL = "gpt-4o"

# Alias de compatibilité (ancienne variable utilisée dans certains modules)
MODEL = REASONING_MODEL

# Paramètres de génération de l'agent
MAX_TOKENS = 40000   # o3-mini consomme des tokens de raisonnement internes en plus du code
TEMPERATURE = 0.2

# Chemins
NOTEBOOKS_DIR = "./notebooks"          # Répertoire contenant les notebooks individuels
UPLOADS_DIR = "./uploads"              # Répertoire pour les fichiers CSV uploadés
REPORT_BLUEPRINTS_DIR = "./offline/outputs"  # Répertoire des blueprints JSON validés offline

# Paramètres métier actuariels — re-export depuis actuarial_params pour accès unifié
# Modifier les seuils dans actuarial_params.py (pas ici)
from actuarial_params import PARAMS as ACTUARIAL_PARAMS  # noqa: E402
