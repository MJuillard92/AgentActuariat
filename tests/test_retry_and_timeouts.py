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


def test_sentence_transformers_futurewarning_filter_registered() -> None:
    """Le module _pack_embed doit enregistrer un filtre `ignore` ciblant le
    FutureWarning sur get_sentence_embedding_dimension.

    Test plus pertinent qu'un simple import : on vérifie que le filtre EST
    présent dans la liste warnings.filters après import. Le runtime canvas
    bénéficie du filtre dès que _pack_embed est importé.
    """
    import warnings as _w
    import tools.conversation._retriever._pack_embed  # noqa: F401 — side-effect

    matching = [
        f for f in _w.filters
        if f[0] == "ignore"
        and f[1] is not None
        and "get_sentence_embedding_dimension" in (f[1].pattern if hasattr(f[1], "pattern") else "")
        and f[2] is FutureWarning
    ]
    assert matching, (
        "Aucun filterwarnings('ignore', message='...get_sentence_embedding_dimension...') "
        "enregistré après import de _pack_embed. Vérifier que le hotfix Bug 4 est en place."
    )
