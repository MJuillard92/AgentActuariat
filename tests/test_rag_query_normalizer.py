"""Tests du normaliseur de requêtes RAG.

Correction déterministe des typos actuariels fréquents avant le retrieval.
Python pur, gratuit, isolé — testable sans LLM ni FAISS.
"""
from __future__ import annotations

import pytest


# ──────────────────────────────────────────────────────────────────────
# Whittaker-Henderson — variantes typo
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", [
    "wittaker", "Wittaker", "WITTAKER",
    "whitaker", "wittakker", "whittakker",
])
def test_whittaker_typos_normalized(variant):
    from agents.rag.pipeline.query_normalizer import normalize
    text = f"c'est quoi la méthode de {variant} ?"
    out = normalize(text)
    assert "whittaker" in out.lower(), f"'{variant}' non normalisé → {out!r}"


# ──────────────────────────────────────────────────────────────────────
# Kaplan-Meier — variantes typo
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", [
    "kaplan-meir", "kaplain", "kaplain-meier",
    "kaplan meier", "KM",
])
def test_kaplan_meier_typos_normalized(variant):
    from agents.rag.pipeline.query_normalizer import normalize
    text = f"explique-moi le {variant}"
    out = normalize(text)
    assert "kaplan-meier" in out.lower(), f"'{variant}' non normalisé → {out!r}"


# ──────────────────────────────────────────────────────────────────────
# Autres méthodes — typos plausibles
# ──────────────────────────────────────────────────────────────────────

def test_gompertz_typo_normalized():
    from agents.rag.pipeline.query_normalizer import normalize
    out = normalize("la loi de gompertez c'est quoi ?")
    assert "gompertz" in out.lower()


def test_lee_carter_variants_normalized():
    from agents.rag.pipeline.query_normalizer import normalize
    for v in ["lee carter", "lee-carter", "LEE CARTER"]:
        out = normalize(f"modèle {v}")
        assert "lee-carter" in out.lower(), f"'{v}' non normalisé → {out!r}"


# ──────────────────────────────────────────────────────────────────────
# Acronymes actuariels — expansion
# ──────────────────────────────────────────────────────────────────────

def test_ic_expanded_to_intervalle_confiance():
    from agents.rag.pipeline.query_normalizer import normalize
    out = normalize("comment calculer l'IC à 95% ?")
    assert "intervalle de confiance" in out.lower()


def test_khi2_normalized_to_chi2():
    from agents.rag.pipeline.query_normalizer import normalize
    out = normalize("le test du khi2 sur les tables")
    assert "chi-2" in out.lower() or "chi2" in out.lower()


# ──────────────────────────────────────────────────────────────────────
# Préservation — pas de sur-correction
# ──────────────────────────────────────────────────────────────────────

def test_no_typo_passes_through_unchanged():
    """Un texte sans typo doit être préservé (modulo casse harmonisée)."""
    from agents.rag.pipeline.query_normalizer import normalize
    text = "explique-moi le lissage whittaker-henderson"
    out = normalize(text).lower()
    # Le contenu sémantique doit être préservé
    assert "whittaker-henderson" in out
    assert "lissage" in out


def test_empty_input_returns_empty():
    from agents.rag.pipeline.query_normalizer import normalize
    assert normalize("") == ""


def test_unrelated_text_unchanged():
    """Un texte sans terme actuariel doit être préservé."""
    from agents.rag.pipeline.query_normalizer import normalize
    text = "bonjour comment ça va aujourd'hui"
    out = normalize(text)
    # Pas de modification structurelle (juste lowercase autorisé)
    assert "bonjour" in out.lower()
    assert "aujourd'hui" in out.lower()
