"""
agents/master/classify_intent.py
Classification d'intention 3-axes (kind / write / report_mode) du Master,
avec score de confiance auto-déclaré par le LLM.

Domain-agnostic : appelle le LLM en JSON mode et lui demande de joindre
un `confidence` ∈ [0, 1]. Le caller décide ensuite (via `is_confident`)
si la classification est suffisamment fiable pour être exécutée ou si
l'utilisateur doit reformuler.

Interface publique :
    classify_intent(last_human, *, has_data, has_calcs) -> dict
        retourne {"kind", "write", "report_mode", "confidence",
                  "reasoning", "intent", "reply"}

    is_confident(classification, threshold=...) -> bool
    confidence_threshold() -> float                # lit la config YAML
"""
from __future__ import annotations

import json
import sys


# ── Configuration : seuil de confiance ─────────────────────────────────────

_DEFAULT_THRESHOLD = 0.80


def confidence_threshold() -> float:
    """Lit le seuil de confiance depuis `config/llm_models.yaml`.

    Permet d'ajuster sans toucher au code. Fallback sur 0.80 si la clé
    n'est pas définie.
    """
    try:
        from agents.mortality.agents.llm_config import get_llm_config
        cfg = get_llm_config("master.classify_intent")
        return float(cfg.get("confidence_threshold", _DEFAULT_THRESHOLD))
    except Exception:
        return _DEFAULT_THRESHOLD


def is_confident(classification: dict, threshold: float | None = None) -> bool:
    """True si la classification atteint le seuil de confiance.

    Args:
        classification : dict retourné par `classify_intent`.
        threshold      : seuil custom (sinon lit la config).
    """
    if threshold is None:
        threshold = confidence_threshold()
    try:
        return float(classification.get("confidence", 0.0)) >= threshold
    except (ValueError, TypeError):
        return False


# ── Dérivation legacy intent ───────────────────────────────────────────────

def _derive_legacy_intent(kind: str, write: str) -> str:
    """Dérive l'`intent` historique (pre-3-axes) pour rétro-compat."""
    if kind == "question":
        return "question"
    if write == "yes":
        return "build_and_write"
    if write == "no":
        return "build_only"
    return "build_and_write"   # ask : pessimiste, on prépare un rapport


# ── Appel LLM ──────────────────────────────────────────────────────────────

def _llm_classify(
    last_human: str,
    has_data: bool,
    has_calcs: bool,
    known_context: dict | None = None,
) -> dict:
    """Appel OpenAI — JSON mode, modèle configuré, temp=0."""
    import openai
    from agents.mortality.agents._utils import call_with_retry
    from agents.mortality.agents.llm_config import get_llm_config

    ctx = (
        f"Fichier CSV chargé : {'oui' if has_data else 'non'}. "
        f"Calculs complets (prêt pour rapport) : {'oui' if has_calcs else 'non'}."
    )
    # Si certains axes ont déjà été tranchés lors de tours précédents, on
    # les indique au LLM. Sans ça le modèle peut faire reply="Je n'ai pas
    # d'indication explicite sur la segmentation par sexe" alors que la
    # valeur est connue depuis le 1er tour.
    if known_context:
        bits = []
        if known_context.get("gender_segmentation"):
            bits.append(f"gender_segmentation déjà connu : {known_context['gender_segmentation']}")
        if known_context.get("report_mode"):
            bits.append(f"report_mode déjà connu : {known_context['report_mode']}")
        if known_context.get("write"):
            bits.append(f"write déjà connu : {known_context['write']}")
        if bits:
            ctx += (
                "\nContexte déjà acquis (NE PAS re-poser ces axes, NE PAS "
                "mentionner qu'ils sont 'inconnus' dans reply) : "
                + " ; ".join(bits) + "."
            )

    prompt = (
        "Tu es un routeur pour un système actuariel. Classifie la demande "
        "en 4 axes orthogonaux ET fournis un score de confiance global.\n\n"
        "=== AXES ===\n\n"
        "Axe 1 — kind :\n"
        "  - task      : calculs / rapport / action concrète\n"
        "  - question  : explication, conversation hors calculs\n\n"
        "Axe 2 — write (uniquement si kind=task) :\n"
        "  - yes : l'utilisateur veut un rapport PDF (mots-clés clairs : "
        "          'rapport', 'PDF', 'document', 'rédige', ou réponse 'oui' "
        "          à une question PDF précédente)\n"
        "  - no  : refus explicite ('sans rapport', 'pas de PDF', 'juste les calculs')\n"
        "  - ask : intention de calcul claire mais aucun signal sur le rapport. "
        "          Master posera la question PDF séparément.\n\n"
        "Axe 3 — compute_scope (uniquement si kind=task) — pilote AUSSI bien les\n"
        "calculs que le contenu du rapport :\n"
        "  - full_report : pipeline complet avec lissage (défaut). Inclut\n"
        "                  exposure, taux bruts, lissage, validation, benchmarking.\n"
        "  - raw_rates   : 'taux bruts', 'sans lissage', 'brut', 'non lissé'.\n"
        "                  Inclut exposure + taux bruts SANS lissage.\n"
        "  - description : 'description', 'analyse descriptive', 'résumé du\n"
        "                  portefeuille', 'analyse statistique'. Pas de calcul\n"
        "                  de taux de mortalité, uniquement stats descriptives.\n\n"
        "Axe 4 — gender_segmentation (uniquement si kind=task) :\n"
        "  - unisex  : 'unisex', 'agrégé', 'table agrégée', 'sans distinction\n"
        "              de sexe', 'tous sexes confondus'\n"
        "  - by_sex  : 'H/F', 'par sexe', 'tables séparées', 'hommes et femmes',\n"
        "              'différencié par sexe', 'homme/femme'\n"
        "  - unknown : aucun signal explicite (Master posera la question\n"
        "              séparément si nécessaire)\n\n"
        "=== SCORE DE CONFIANCE (confidence) ===\n\n"
        "Note GLOBALE sur la fiabilité de ta classification des 4 axes.\n"
        "  0.95-1.00 : intention parfaitement claire, sans ambiguïté\n"
        "  0.80-0.95 : intention claire, une légère ambiguïté mineure\n"
        "  0.60-0.80 : plusieurs interprétations plausibles\n"
        "  0.00-0.60 : très ambigu, l'utilisateur devrait reformuler\n\n"
        "IMPORTANT : un axe en `ask`/`unknown` ne fait PAS baisser le score —\n"
        "c'est une absence d'information, pas une ambiguïté. Master sait gérer\n"
        "ces valeurs (il posera la question manquante). Ne baisse le score que\n"
        "quand la demande GLOBALE est floue.\n\n"
        "Sois HONNÊTE : préfère un score bas si tu hésites — l'agent\n"
        "demandera une reformulation, ce qui est mieux qu'agir mal.\n\n"
        f"=== CONTEXTE ===\n{ctx}\n\n"
        f"=== DEMANDE UTILISATEUR ===\n{last_human[:500]}\n\n"
        "=== RÉPONSE ===\n"
        "Réponds UNIQUEMENT en JSON :\n"
        "{\n"
        '  "kind":                "task" | "question",\n'
        '  "write":               "yes" | "no" | "ask",\n'
        '  "report_mode":         "full_report" | "raw_rates" | "description",\n'
        '  "gender_segmentation": "unisex" | "by_sex" | "unknown",\n'
        '  "confidence":          0.0-1.0,\n'
        '  "reasoning":           "1 phrase courte : pourquoi cette classification",\n'
        '  "reply":               "phrase de confirmation à afficher (1-2 phrases max)"\n'
        "}\n"
    )

    cfg = get_llm_config("master.classify_intent")
    client = openai.OpenAI()
    resp = call_with_retry(
        client,
        model=cfg["model"],
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=cfg.get("max_tokens", 400),
        temperature=cfg.get("temperature", 0.0),
    )
    return json.loads(resp.choices[0].message.content or "{}")


# ── Point d'entrée public ──────────────────────────────────────────────────

def classify_intent(
    last_human: str,
    *,
    has_data: bool = False,
    has_calcs: bool = False,
    known_context: dict | None = None,
) -> dict:
    """Classifie une demande utilisateur en 3 axes avec score de confiance.

    Args:
        last_human: dernier message utilisateur.
        has_data:   True si un CSV est chargé pour la session.
        has_calcs:  True si toutes les clés builder requises sont déjà
                    présentes dans le data_store.

    Returns:
        dict avec :
          - kind        : "task" | "question"
          - write       : "yes" | "no" | "ask"
          - report_mode : "full_report" | "raw_rates" | "description"
          - confidence  : float [0.0, 1.0] — score auto-déclaré du LLM
          - reasoning   : str — raison de la classification (surtout si confidence basse)
          - intent      : alias legacy
          - reply       : phrase de confirmation
    """
    try:
        parsed = _llm_classify(last_human, has_data, has_calcs,
                               known_context=known_context)
    except Exception as exc:
        print(f"[classify_intent] LLM error: {exc}", file=sys.stderr)
        return {
            "kind":                "task",
            "write":               "ask",
            "report_mode":         "full_report",
            "gender_segmentation": None,
            "confidence":          0.0,
            "reasoning":           f"Erreur LLM : {exc}",
            "intent":              "unclear",
            "reply":               "Je n'ai pas compris votre demande. Pouvez-vous préciser ?",
        }

    kind        = parsed.get("kind",        "task")
    write       = parsed.get("write",       "ask")
    report_mode = parsed.get("report_mode", "full_report")
    reply       = parsed.get("reply",       "")
    reasoning   = parsed.get("reasoning",   "")
    # 4e axe : gender_segmentation. "unknown" si non-précisé (Master posera
    # la question si la session en a besoin).
    gender_raw  = parsed.get("gender_segmentation", "unknown")
    if gender_raw not in ("unisex", "by_sex", "unknown"):
        gender_raw = "unknown"
    gender = None if gender_raw == "unknown" else gender_raw
    # Défaut 1.0 (= confiant) si le LLM oublie la clé : on garde le
    # comportement legacy (exécution directe). En pratique le prompt le
    # demande explicitement, donc l'absence est rare.
    try:
        confidence = float(parsed.get("confidence", 1.0))
    except (ValueError, TypeError):
        confidence = 1.0
    # Clamp [0, 1] au cas où le LLM dérape
    confidence = max(0.0, min(1.0, confidence))

    return {
        "kind":                kind,
        "write":               write,
        "report_mode":         report_mode,
        "gender_segmentation": gender,    # None | "unisex" | "by_sex"
        "confidence":          confidence,
        "reasoning":           reasoning,
        "intent":              _derive_legacy_intent(kind, write),
        "reply":               reply,
    }
