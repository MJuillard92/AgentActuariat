"""Tests du capability registry — self-describe des sous-agents.

Vérifie :
- Chaque sous-agent expose `describe()` retournant un dict valide
- Le registry agrège correctement
- La détection de question méta-capacité catch les phrasings courants
- Le formatter produit un Markdown lisible
"""
from __future__ import annotations

import pytest


# ──────────────────────────────────────────────────────────────────────
# Each agent has describe()
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("agent_name", ["master", "mortality", "report", "rag"])
def test_each_agent_exposes_describe(agent_name):
    mod = __import__(
        f"agents.{agent_name}.describe_capabilities",
        fromlist=["describe"],
    )
    desc = mod.describe()
    assert isinstance(desc, dict)
    assert desc["agent"] == agent_name
    assert desc.get("display")
    assert desc.get("purpose")


def test_mortality_lists_owned_tools_from_catalogue():
    from agents.mortality.describe_capabilities import describe
    d = describe()
    tools = d["tools"]
    # Doit inclure au moins quelques builder.* connus
    names = {t["name"] for t in tools}
    assert "builder.exposure" in names
    assert "builder.smoothing" in names
    assert "builder.validation" in names


def test_report_lists_build_pdf_and_graphs():
    from agents.report.describe_capabilities import describe
    d = describe()
    names = {t["name"] for t in d["tools"]}
    # Au moins un build_pdf.* et un graphs.*
    assert any(n.startswith("build_pdf.") for n in names)
    assert any(n.startswith("graphs.") for n in names)


def test_rag_describe_includes_corpus_stats():
    from agents.rag.describe_capabilities import describe
    d = describe()
    stats = d["corpus_stats"]
    # Le corpus est build : n_chunks > 0
    assert stats["n_chunks"] > 0
    assert stats["n_docs"] > 0


# ──────────────────────────────────────────────────────────────────────
# Registry aggregation
# ──────────────────────────────────────────────────────────────────────

def test_build_registry_aggregates_all_agents():
    from agents.master.capability_registry import build_registry
    r = build_registry()
    assert set(r["agents"].keys()) == {"master", "mortality", "report", "rag"}


def test_get_registry_caches_build():
    """Cache module-level : 2 appels successifs retournent le MÊME objet."""
    from agents.master.capability_registry import get_registry, reset_registry_cache
    reset_registry_cache()
    r1 = get_registry()
    r2 = get_registry()
    assert r1 is r2  # même objet, pas re-build


# ──────────────────────────────────────────────────────────────────────
# Question detection
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    "que sais-tu faire ?",
    "Que peux-tu faire",
    "Tes capacités ?",
    "Quelles sont tes fonctionnalités",
    "liste tes outils",
    "donne-moi la liste de tes méthodes",
    "peux-tu calculer un SMR ?",
    "sais-tu construire une table de mortalité ?",
    "what can you do",
    "list your tools",
])
def test_is_capability_question_catches_meta(q):
    from agents.master.capability_registry import is_capability_question
    assert is_capability_question(q) is True, f"Manqué : {q!r}"


@pytest.mark.parametrize("q", [
    "c'est quoi le lissage Whittaker-Henderson ?",
    "explique-moi le test du chi-2",
    "comment fonctionne l'A132-18 ?",
    "merci",
    "calcule l'exposition pour ce portefeuille",
    "génère un rapport PDF",
])
def test_is_capability_question_does_not_false_positive(q):
    from agents.master.capability_registry import is_capability_question
    assert is_capability_question(q) is False, f"Faux positif : {q!r}"


# ──────────────────────────────────────────────────────────────────────
# Formatter
# ──────────────────────────────────────────────────────────────────────

def test_format_capabilities_produces_markdown():
    from agents.master.capability_registry import format_capabilities_answer
    out = format_capabilities_answer()
    assert "MortalityAgent" in out
    assert "ReportAgent" in out
    assert "RAGAgent" in out
    assert "Whittaker" in out  # une méthode connue doit apparaître
    # Format Markdown
    assert "###" in out


def test_format_with_empty_registry_does_not_crash():
    from agents.master.capability_registry import format_capabilities_answer
    out = format_capabilities_answer({"agents": {}})
    assert isinstance(out, str)
    assert len(out) > 0
