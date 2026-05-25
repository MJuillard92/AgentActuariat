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
#
# DEUX NIVEAUX (plan disambiguation 2026-05-25) :
#
#   1. _CAPABILITY_CANONICAL_RE — ultra-strict, utilisé par
#      is_capability_question() pour le COURT-CIRCUIT pre-LLM dans
#      master_node. Match seulement les phrases qui ne peuvent rien dire
#      d'autre qu'une question méta (aucun objet métier possible).
#
#   2. _CAPABILITY_HINT_RE et _COMMAND_HINT_RE — utilisés par
#      regex_kind_hint() pour le CROSS-CHECK post-LLM. Plus permissifs :
#      ils peuvent désaccorder avec le LLM → l'utilisateur tranche via
#      désambiguïsation (pas d'override silencieux).

_CAPABILITY_CANONICAL_PATTERNS = (
    # FR — questions générales canoniques (ancrage début/fin obligatoire)
    r"^\s*qu['e]\s+(sais|peux|fais)[- ]?tu\s+faire\s*\??\s*$",
    r"^\s*ce\s+que\s+tu\s+(sais|peux)\s+faire\s*\??\s*$",
    r"^\s*(quelles?\s+sont\s+)?tes\s+(capacit[ée]s?|fonctionnalit[ée]s?|comp[ée]tences?)\s*\??\s*$",
    r"^\s*[àa]\s+quoi\s+sers[- ]?tu\s*\??\s*$",
    r"^\s*tu\s+fais\s+quoi\s*\??\s*$",
    # FR — listes canoniques
    r"^\s*liste\s+(tes|vos)\s+(outils?|m[ée]thodes?|fonctions?|capacit[ée]s?)\s*\??\s*$",
    r"^\s*donne[- ]?moi\s+la\s+liste\s+de\s+tes\s+(outils?|m[ée]thodes?|fonctions?)\s*\??\s*$",
    # EN canoniques
    r"^\s*what\s+can\s+you\s+do\s*\??\s*$",
    r"^\s*list\s+(your\s+)?(tools?|capabilities?|methods?)\s*\??\s*$",
)
_CAPABILITY_CANONICAL_RE = re.compile(
    "|".join(_CAPABILITY_CANONICAL_PATTERNS), re.IGNORECASE,
)


def is_capability_question(text: str) -> bool:
    """True UNIQUEMENT si la question est manifestement une demande
    méta-capacité (ultra-strict, canonique).

    Utilisé en court-circuit PRE-LLM dans master_node. La phrase doit être
    canonique (aucun objet métier possible). Pour le check post-LLM, voir
    `regex_kind_hint()`.
    """
    if not text:
        return False
    return bool(_CAPABILITY_CANONICAL_RE.search(text))


# ── Hints pour cross-check post-LLM (plus permissifs) ───────────────────────

# Char-class large pour les accents FR (é, è, ê, e).
_E = r"[eéèê]"

_CAPABILITY_HINT_PATTERNS = (
    *_CAPABILITY_CANONICAL_PATTERNS,
    # Méta avec verbe abstrait + objet ABSTRAIT (des/de la/du suivi de
    # n'importe quoi non concret). « peux-tu faire des calculs » =
    # méta — l'objet « des calculs » est générique, pas un job précis.
    rf"\b(peux|sais)[- ]?tu\s+(faire|calculer|g{_E}n{_E}rer|produire|construire)"
    rf"\s+(des|de\s+la|du)\s+\w+",
    # Méta avec verbe abstrait SANS objet (fin de phrase ou pronom flou).
    rf"\b(peux|sais)[- ]?tu\s+(faire|calculer|g{_E}n{_E}rer|produire|construire)\s*"
    r"(n'importe\s+quoi|ce\s+genre|ça|cela|tout)?\s*\??\s*$",
    # « est-ce que tu sais faire » sans objet
    rf"\b(est[- ]?ce\s+que\s+)?tu\s+(sais|peux)\s+(faire|calculer|g{_E}n{_E}rer)\s*\??\s*$",
)
_CAPABILITY_HINT_RE = re.compile("|".join(_CAPABILITY_HINT_PATTERNS), re.IGNORECASE)

_COMMAND_HINT_PATTERNS = (
    # 1. Impératif direct en début de phrase + (déterminant | -moi)
    # Note : `[eéèê]` couvre é/è/ê pour « génère », « rédige », « écris ».
    rf"^\s*(calcule|g{_E}n{_E}re|construis|produis|lance|ex{_E}cute|fais|"
    rf"r{_E}dige|{_E}cris|build|run|generate|create|compute)"
    r"(?:\s+(?:le|la|les|un|une|des|mon|ma|mes|ce|cette|ces|moi)\b|[- ]?moi\b)",
    # 2. « peux-tu/sais-tu VERBE DÉTERMINANT NOM » — le bug d'origine
    rf"\b(peux|sais)[- ]?tu\s+(calculer|g{_E}n{_E}rer|construire|produire|"
    rf"lancer|faire|r{_E}diger|{_E}crire)\s+"
    r"(le|la|les|un|une|mon|ma|mes|ce|cette|ces)\s+\w+",
    # 3. Politesse + verbe d'action (infinitif inclus : calculer, générer…)
    rf"\b(j['e]\s*(aimerais|voudrais|veux)|on\s+veut)\s+"
    rf"(que\s+tu\s+)?(calcule(?:r|s)?|g{_E}n{_E}re(?:r|s)?|"
    r"construis|construire|produise(?:s|r)?|lance(?:r|s)?)\b",
    # 4. « fais-moi / donne-moi le rapport/calcul/etc. »
    r"\b(fais|donne|sors|montre)[- ]?moi\s+(le|la|un|une)\s+"
    r"(rapport|tableau|table|calcul|pdf|graphique|analyse)\b",
)
_COMMAND_HINT_RE = re.compile("|".join(_COMMAND_HINT_PATTERNS), re.IGNORECASE)


def regex_kind_hint(text: str) -> str | None:
    """Hint regex pour cross-checker la classification LLM.

    Retourne :
      - "question" si le texte ressemble à une demande méta-capacité
      - "task"     si le texte ressemble à une commande concrète
      - None       si la regex ne sait pas trancher (cas par défaut)

    Plus permissif que `is_capability_question` (qui court-circuite avant
    le LLM). Sert à détecter les désaccords avec le LLM ; en cas de
    désaccord c'est l'utilisateur qui tranche (pas d'override silencieux).

    Priorité : COMMAND > CAPABILITY (en cas de double match, le verbe
    d'action + objet concret l'emporte sur le pattern méta).
    """
    if not text:
        return None
    if _COMMAND_HINT_RE.search(text):
        return "task"
    if _CAPABILITY_HINT_RE.search(text):
        return "question"
    return None


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
