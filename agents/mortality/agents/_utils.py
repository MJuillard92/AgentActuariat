"""
_utils.py — Utilitaires partagés entre les nœuds LangGraph.
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


# ── Sérialisation msgpack-safe pour LangGraph MemorySaver ───────────────────
# HOTFIX-pre-refacto-2026-05 (Bug 9) : promu depuis tools_node._msgpack_safe
# au niveau _utils pour pouvoir être appelé par TOUS les nodes au retour,
# pas uniquement par tools_node. Évite le crash :
#   "Dict key must a type serializable with OPT_NON_STR_KEYS"
# quand un tool écrit un dict avec clés int/tuple dans data_store.

def msgpack_safe(obj):
    """Convertit récursivement un objet en types Python natifs sérialisables
    par ormsgpack (utilisé par LangGraph MemorySaver).

    Couvre numpy/pandas scalaires + collections + dicts à clés non-str.
    Ordre des checks important : numpy.bool_/integer/floating sont
    sous-classes de bool/int/float → vérifier numpy AVANT Python natif.
    """
    import numpy as _np
    import pandas as _pd
    if obj is None:
        return None
    if isinstance(obj, _np.bool_):
        return bool(obj)
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        v = float(obj)
        return v if v == v and v not in (float("inf"), float("-inf")) else None
    if isinstance(obj, _np.ndarray):
        return [msgpack_safe(x) for x in obj.tolist()]
    if isinstance(obj, _pd.DataFrame):
        return [msgpack_safe(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, _pd.Series):
        return [msgpack_safe(x) for x in obj.tolist()]
    if isinstance(obj, _pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {str(k): msgpack_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [msgpack_safe(x) for x in obj]
    return obj


def sanitize_node_result(result: dict) -> dict:
    """HOTFIX-pre-refacto-2026-05 (Bug 9) : passe `data_store` et `events`
    d'un retour de node à travers msgpack_safe avant que LangGraph ne
    checkpointe l'état. Garantit l'absence de clés int/tuple/numpy dans
    le state persisté.

    Idempotent : safe à appeler plusieurs fois (les types déjà natifs
    passent à travers sans transformation visible).
    """
    if not isinstance(result, dict):
        return result
    out = dict(result)
    if "data_store" in out and isinstance(out["data_store"], dict):
        out["data_store"] = msgpack_safe(out["data_store"])
    if "events" in out and isinstance(out["events"], list):
        out["events"] = msgpack_safe(out["events"])
    return out

# Délais de retry en secondes. HOTFIX-pre-refacto-2026-05 (Bug 4) :
# réduit de [15, 30, 60] (105s total) à [3, 8, 20] (31s total) pour que
# l'utilisateur reçoive un échec rapide en cas de panne réseau, au lieu
# d'attendre 105s. Sera remplacé par circuit breaker au Lot 11.
_RETRY_WAITS = [3, 8, 20]  # 3 tentatives : 3s, 8s, 20s


def _is_rate_limit(exc: Exception) -> bool:
    """Détecte une erreur de rate limit OpenAI (429 TPM ou RPM)."""
    exc_str = str(exc)
    return (
        "429" in exc_str
        or "rate_limit_exceeded" in exc_str
        or "RateLimitError" in type(exc).__name__
        or getattr(exc, "status_code", None) == 429
    )


def _is_retryable(exc: Exception) -> bool:
    """Détecte les erreurs transitoires retriables (rate limit + timeout + 5xx)."""
    if _is_rate_limit(exc):
        return True
    exc_str = str(exc)
    return (
        "timeout" in exc_str.lower()
        or "connection" in exc_str.lower()
        or getattr(exc, "status_code", None) in (500, 502, 503, 529)
    )


def _adapt_kwargs_for_model(kwargs: dict) -> dict:
    """Adapte les paramètres OpenAI selon la famille du modèle.

    Les modèles GPT-5.x et la série o (raisonnement) exigent
    `max_completion_tokens` au lieu de `max_tokens`. Cette fonction
    fait la traduction transparente — le code applicatif peut continuer
    à utiliser `max_tokens` partout.
    """
    model = (kwargs.get("model") or "").lower()
    needs_completion_tokens = (
        model.startswith("gpt-5") or
        model.startswith("o1") or
        model.startswith("o3") or
        model.startswith("o4") or
        model.startswith("o5")
    )
    if needs_completion_tokens and "max_tokens" in kwargs:
        kwargs = dict(kwargs)
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    return kwargs


def call_with_retry(client, max_attempts: int = 4, **kwargs):
    """
    Appel OpenAI avec retry exponentiel sur les erreurs de rate limit (TPM/RPM)
    et les erreurs transitoires (timeout, 5xx).

    Stratégie d'attente pour TPM 30k/min sur gpt-4o :
      - Tentative 1 : échec → attente 15s
      - Tentative 2 : échec → attente 30s
      - Tentative 3 : échec → attente 60s
      - Tentative 4 : lève l'exception

    Args:
        client      : instance openai.OpenAI()
        max_attempts: nombre max de tentatives (défaut 4)
        **kwargs    : arguments passés à client.chat.completions.create().
                      `max_tokens` est automatiquement traduit en
                      `max_completion_tokens` pour les modèles gpt-5.x et o*.
    """
    kwargs = _adapt_kwargs_for_model(kwargs)
    last_exc: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return client.chat.completions.create(**kwargs)

        except Exception as exc:
            last_exc = exc

            if not _is_retryable(exc) or attempt == max_attempts - 1:
                # Erreur non retriable ou dernière tentative → propager
                log.error(
                    "[call_with_retry] erreur non retriable ou max tentatives atteint "
                    "(tentative %d/%d) : %s",
                    attempt + 1, max_attempts, exc,
                )
                raise

            wait = _RETRY_WAITS[min(attempt, len(_RETRY_WAITS) - 1)]
            rate_limit = _is_rate_limit(exc)
            log.warning(
                "[call_with_retry] %s — attente %ds avant retry (tentative %d/%d) : %s",
                "TPM/RPM 429" if rate_limit else "erreur transitoire",
                wait, attempt + 1, max_attempts, exc,
            )
            time.sleep(wait)

    # Sécurité (ne devrait pas être atteint)
    raise last_exc
