"""
domain_config.py
Registre des domaines disponibles pour l'agent générique.

Chaque domaine définit :
  - label       : nom affiché dans le sélecteur UI
  - kb_dir      : chemin relatif au projet vers la base de connaissances JSON
  - prompt_file : chemin relatif vers un fichier .txt contenant le system prompt
                  (None → utilise SYSTEM_PROMPT_TEMPLATE dans agent.py)
  - default_message : message pré-rempli suggéré à l'utilisateur

Pour ajouter un domaine : ajouter une entrée dans DOMAINS et créer
le fichier prompt et le répertoire KB correspondants.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent

DOMAINS: dict[str, dict] = {
    "mortality": {
        "label": "Tables de mortalité (TD/TF)",
        "kb_dir": "Knowledge Base",
        "prompt_file": None,   # utilise SYSTEM_PROMPT_TEMPLATE dans agent.py
        "default_message": (
            "Construis la table de mortalité d'expérience à partir du fichier de données. "
            "Suis la procédure standard : chargement, nettoyage, calcul des expositions, "
            "taux bruts, lissage, validation et export."
        ),
    },
    "nonlife_reserving": {
        "label": "Provisionnement non-vie (IBNR / Chain-Ladder)",
        "kb_dir": "Knowledge Base/nonlife",
        "prompt_file": "prompts/nonlife_reserving.txt",
        "default_message": (
            "Calcule les provisions IBNR à partir du triangle de développement. "
            "Utilise la méthode Chain-Ladder et compare avec Bornhuetter-Ferguson."
        ),
    },
    "generic": {
        "label": "Agent générique (Python libre)",
        "kb_dir": "Knowledge Base",
        "prompt_file": "prompts/generic_agent.txt",
        "default_message": "Décris ce que tu veux analyser.",
    },
}

_GENERIC_PROMPT = """\
Tu es un agent d'analyse de données expert. Tu exécutes du code Python pour répondre \
à la demande de l'utilisateur. Tu disposes de l'outil execute_python pour exécuter du code \
dans un kernel partagé, et de search_documentation pour consulter la base de connaissances.

RÈGLES :
- Chaque appel execute_python fait UNE seule chose.
- Utilise display(df) pour afficher les DataFrames, jamais print().
- Bibliothèques autorisées : pandas, numpy, scipy, matplotlib, seaborn, pathlib, json.

{notebook_context}
"""


def list_domains() -> list[dict]:
    """Retourne la liste des domaines pour un dcc.Dropdown."""
    return [{"label": v["label"], "value": k} for k, v in DOMAINS.items()]


def get_domain(domain_id: str) -> dict:
    """Retourne la config d'un domaine (fallback : mortality)."""
    return DOMAINS.get(domain_id, DOMAINS["mortality"])


def load_system_prompt(domain_id: str) -> str | None:
    """Charge le system prompt pour un domaine.

    Returns:
        str  : texte du prompt si prompt_file existe et est lisible
        None : si prompt_file est None ou fichier absent (→ agent.py utilisera son SYSTEM_PROMPT_TEMPLATE)
    """
    domain = get_domain(domain_id)
    prompt_file = domain.get("prompt_file")
    if prompt_file is None:
        return None

    path = _PROJECT_ROOT / prompt_file
    if not path.exists():
        if domain_id == "generic":
            return _GENERIC_PROMPT
        return None

    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def load_kb_context(domain_id: str, modules: list[str] | None = None) -> str:
    """Charge la base de connaissances pour un domaine.

    Délègue à agent.load_knowledge_base_context() avec le kb_dir du domaine.
    """
    from agent import load_knowledge_base_context  # import local pour éviter circulaire

    domain = get_domain(domain_id)
    kb_dir = _PROJECT_ROOT / domain["kb_dir"]
    return load_knowledge_base_context(modules=modules, kb_dir=kb_dir)


def get_default_message(domain_id: str) -> str:
    """Retourne le message par défaut suggéré pour un domaine."""
    return get_domain(domain_id).get("default_message", "")
