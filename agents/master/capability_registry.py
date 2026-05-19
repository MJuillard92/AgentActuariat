"""
agents.master.capability_registry

Agrège les self-describes des sous-agents en un registry mémoire,
construit une fois au startup. Permet au Master de répondre aux questions
méta-capacité ("que sais-tu faire ?") sans appel LLM, en <10ms.

Pattern :
    - Chaque sous-agent expose `describe()` à `agents/<name>/describe_capabilities.py`
    - Master appelle tous les describes au startup → CAPABILITY_REGISTRY (module-level)
    - `is_capability_question(text)` détecte les marqueurs méta
    - `format_capabilities_answer(registry)` produit un message Markdown prêt à servir
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)


# ── Construction du registry au startup (lazy + cache) ──────────────────────

_REGISTRY_CACHE: dict | None = None


def build_registry() -> dict:
    """Agrège les capacities de tous les sous-agents.

    Import paresseux pour éviter le cycle si un sous-agent importe ce module.
    Graceful : si un describe échoue, on log et on continue avec les autres.
    """
    registry: dict = {"agents": {}}
    for agent_name in ("master", "mortality", "report", "rag"):
        try:
            mod = __import__(
                f"agents.{agent_name}.describe_capabilities",
                fromlist=["describe"],
            )
            registry["agents"][agent_name] = mod.describe()
        except Exception as exc:
            log.warning("[capability_registry] %s.describe() failed: %s", agent_name, exc)
            registry["agents"][agent_name] = {
                "agent":   agent_name,
                "display": f"{agent_name.title()}Agent",
                "purpose": f"(introspection failed : {exc})",
                "tools":   [],
            }
    return registry


def get_registry() -> dict:
    """Cache module-level : build_registry() appelé 1x par process."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = build_registry()
    return _REGISTRY_CACHE


def reset_registry_cache() -> None:
    """Pour les tests : force un rebuild au prochain get_registry()."""
    global _REGISTRY_CACHE
    _REGISTRY_CACHE = None


# ── Détection des questions méta-capacité ───────────────────────────────────

_CAPABILITY_PATTERNS = (
    # FR — questions générales
    r"\bque\s+(sais|peux|fais|peut|fait)[- ]?(tu|vous|on)\s+faire",
    r"\b(quelles?\s+sont\s+)?tes?\s+(capacit[ée]s?|fonctionnalit[ée]s?|comp[ée]tences?)",
    r"\bce\s+que\s+tu\s+(sais|peux)\s+faire",
    # FR — listes
    r"\b(liste|donne[- ]?moi\s+la\s+liste)\s+(de\s+)?(tes|vos)\s+(outils?|tools?|fonctions?|m[ée]thodes?)",
    r"\b(quels?|quelles?)\s+(outils?|tools?|m[ée]thodes?|agents?|rapports?|calculs?)",
    # FR — capacité spécifique
    r"\b(peux|sais)[- ]?tu\s+(faire|calculer|g[ée]n[ée]rer|produire|construire)\b",
    # EN
    r"\bwhat\s+(can\s+you|do\s+you)\s+(do|know)\b",
    r"\b(list|show)\s+(your\s+)?(tools?|capabilities?|methods?)",
)
_CAPABILITY_RE = re.compile("|".join(_CAPABILITY_PATTERNS), re.IGNORECASE)


def is_capability_question(text: str) -> bool:
    """True si la question concerne les capacités du système."""
    if not text:
        return False
    return bool(_CAPABILITY_RE.search(text))


# ── Formatage de la réponse Markdown ────────────────────────────────────────

def format_capabilities_answer(registry: dict | None = None) -> str:
    """Produit une réponse Markdown lisible décrivant les capacités du système.

    Déterministe, instantanée, aucune dépendance LLM. Toujours à jour
    (régénérée au startup depuis le code).
    """
    if registry is None:
        registry = get_registry()
    agents = registry.get("agents") or {}

    lines: list[str] = [
        "Je suis un système actuariel multi-agents pour construire des tables "
        "de mortalité d'expérience et générer des rapports certifiés.",
        "",
        "Voici ce que je peux faire :",
        "",
    ]

    # Master en premier (orchestrateur)
    order = ("master", "mortality", "report", "rag")
    for name in order:
        info = agents.get(name)
        if not info:
            continue
        lines.append(f"### {info.get('display', name)}")
        purpose = info.get("purpose") or ""
        if purpose:
            lines.append(purpose)
            lines.append("")

        # Méthodes (MortalityAgent a un methods_summary structuré)
        ms = info.get("methods_summary") or {}
        if ms:
            for category, items in ms.items():
                pretty = category.replace("_", " ").capitalize()
                lines.append(f"- **{pretty}** : {', '.join(items)}")
            lines.append("")

        # Modes (ReportAgent)
        modes = info.get("report_modes") or {}
        if modes:
            for mode, desc in modes.items():
                lines.append(f"- **{mode}** : {desc}")
            lines.append("")

        # Topics (RAG)
        topics = info.get("topics_covered") or []
        if topics:
            lines.append("Sujets couverts :")
            for t in topics[:6]:
                lines.append(f"- {t}")
            if len(topics) > 6:
                lines.append(f"- *...et {len(topics) - 6} autres*")
            lines.append("")

        # Routes (Master)
        routes = info.get("routes") or []
        if routes:
            for r in routes:
                lines.append(f"- {r}")
            lines.append("")

    lines.append("Pour démarrer : uploadez un fichier CSV de portefeuille "
                 "et décrivez votre objectif (calcul, rapport, ou question méthodologique).")

    return "\n".join(lines)
