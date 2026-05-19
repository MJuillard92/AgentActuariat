"""
HOTFIX-pre-refacto-2026-05 — Bug 4 : délais de retry agressifs (105s) +
timeout client OpenAI absent + warmup retriever double + FutureWarning
sentence-transformers.

Tests :
  - _RETRY_WAITS cumulés sous 35s (vs 105s avant)
  - FutureWarning sentence_transformers silencé au niveau global
"""
from __future__ import annotations

import warnings

import pytest


def test_retry_waits_total_under_35_seconds() -> None:
    """Le cumul des délais de retry doit rester sous 35s (vs 105s avant hotfix).

    Stratégie : 3s + 8s + 20s = 31s. Permet au user de voir l'échec rapidement
    et de retry manuellement, au lieu d'attendre 105s.
    """
    from agents.mortality.agents._utils import _RETRY_WAITS
    total = sum(_RETRY_WAITS)
    assert total <= 35, (
        f"_RETRY_WAITS cumulés = {total}s > 35s. Le plan hotfix exige "
        f"[3, 8, 20] (31s total). Actuel : {_RETRY_WAITS}"
    )


def test_retry_waits_progressive() -> None:
    """Les délais doivent rester progressifs (backoff)."""
    from agents.mortality.agents._utils import _RETRY_WAITS
    assert _RETRY_WAITS == sorted(_RETRY_WAITS), (
        "Les délais doivent être croissants (backoff exponentiel/progressif)"
    )


def test_sentence_transformers_futurewarning_filter_in_source() -> None:
    """Le code source de _pack_embed doit enregistrer un filterwarnings ignore
    ciblant get_sentence_embedding_dimension (FutureWarning).

    Test statique sur la source : robuste aux interactions pytest qui
    réinitialisent warnings.filters entre tests. Le runtime canvas exécute
    cet appel à l'import du module, ce qui silence le warning en prod.
    """
    from pathlib import Path
    src = Path("tools/conversation/_retriever/_pack_embed.py").read_text(encoding="utf-8")
    assert "warnings.filterwarnings" in src, (
        "warnings.filterwarnings absent du module — hotfix Bug 4 manquant"
    )
    assert "get_sentence_embedding_dimension" in src, (
        "Pattern get_sentence_embedding_dimension absent du filtre"
    )
    assert "FutureWarning" in src, (
        "Catégorie FutureWarning absente du filtre"
    )
