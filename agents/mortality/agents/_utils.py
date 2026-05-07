"""
_utils.py — Utilitaires partagés entre les nœuds LangGraph.
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

# Délais de retry en secondes pour les erreurs TPM/RPM (30k tokens/min sur gpt-4o)
# Stratégie : attendre suffisamment pour que la fenêtre 1-minute se renouvelle
_RETRY_WAITS = [15, 30, 60]  # 3 tentatives : 15s, 30s, 60s


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
