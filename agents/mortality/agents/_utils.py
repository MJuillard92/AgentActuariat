"""
_utils.py — Utilitaires partagés entre les nœuds LangGraph.
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


def call_with_retry(client, max_attempts: int = 3, **kwargs):
    """
    Appel OpenAI avec retry exponentiel sur les erreurs 429 (TPM/RPM rate limit).

    Attente : 1s, 2s, puis lève l'exception.
    """
    for attempt in range(max_attempts):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            is_rate_limit = (
                "429" in str(exc)
                or "rate_limit_exceeded" in str(exc)
                or getattr(exc, "status_code", None) == 429
            )
            if is_rate_limit and attempt < max_attempts - 1:
                wait = 2 ** attempt  # 1s puis 2s
                log.warning(
                    "[retry] TPM 429 — attente %ds (tentative %d/%d) : %s",
                    wait, attempt + 1, max_attempts, exc,
                )
                time.sleep(wait)
                continue
            raise
