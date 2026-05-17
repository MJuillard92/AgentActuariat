"""Tests du tool conversation.search_doctrine.

Couvre :
  - Scope : tool dans CONVERSATIONAL_TOOLS, pas dans BUILDER_TOOLS
  - Validation des params (query manquant)
  - Comportement quand l'index est absent
  - Sanity end-to-end : top-k cohérent pour requêtes actuarielles types
    (skip si l'index FAISS n'a pas été construit — pour CI sans bge-m3)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────
# Scope
# ──────────────────────────────────────────────────────────────────────

def test_search_doctrine_not_in_builder_tools():
    """Le Builder ne doit pas pouvoir court-circuiter le pipeline normé en
    appelant des recherches de doctrine."""
    from agents.mortality.agents.builder_node import BUILDER_TOOLS
    # search_doctrine vit dans le namespace "conversation"
    assert "conversation" not in BUILDER_TOOLS


def test_search_doctrine_visible_in_conversational_namespace():
    """Le namespace conversation est bien dans CONVERSATIONAL_TOOLS."""
    from agents.master.conversation import CONVERSATIONAL_TOOLS
    assert "conversation" in CONVERSATIONAL_TOOLS


# ──────────────────────────────────────────────────────────────────────
# Validation des params
# ──────────────────────────────────────────────────────────────────────

def test_missing_query_returns_error():
    from tools.conversation.search_doctrine import run
    res = run(None, {})
    assert "erreur" in res
    assert "query" in res["erreur"].lower()


def test_missing_index_returns_error(tmp_path, monkeypatch):
    """Si l'index FAISS n'existe pas, retourne une erreur explicative
    (pas de stacktrace)."""
    from tools.conversation import search_doctrine as sd
    # Redirige vers un dossier vide
    monkeypatch.setattr(sd, "_INDEX_PATH", tmp_path / "nope.bin")
    monkeypatch.setattr(sd, "_META_PATH",  tmp_path / "nope.json")
    monkeypatch.setattr(sd, "_RETRIEVER_CACHE", {})
    res = sd.run(None, {"query": "test"})
    assert "erreur" in res
    assert "index" in res["erreur"].lower()


# ──────────────────────────────────────────────────────────────────────
# End-to-end (skip si l'index n'a pas été construit localement)
# ──────────────────────────────────────────────────────────────────────

def _index_available() -> bool:
    p = Path(__file__).resolve().parent.parent / "knowledge_base" / \
        "rag_doctrine" / "index" / "faiss.bin"
    return p.exists()


@pytest.mark.skipif(not _index_available(),
                    reason="FAISS index non construit (ingest_doctrine.py)")
def test_e2e_query_whittaker_returns_d03():
    """Une question sur Whittaker-Henderson doit ramener des chunks D03.*."""
    from tools.conversation.search_doctrine import run
    res = run(None, {
        "query": "lissage Whittaker-Henderson choix du paramètre h",
        "k": 5,
    })
    assert res.get("n_returned", 0) >= 3
    doc_ids = {r["doc_id"] for r in res["results"]}
    assert "D03" in doc_ids, f"D03 (lissage) absent du top — vu : {doc_ids}"


@pytest.mark.skipif(not _index_available(),
                    reason="FAISS index non construit (ingest_doctrine.py)")
def test_e2e_query_a132_returns_d07():
    """Question réglementaire A132-18 → doit ramener D07.*."""
    from tools.conversation.search_doctrine import run
    res = run(None, {
        "query": "obligation A132-18 Code des assurances certification",
        "k": 5,
    })
    assert res.get("n_returned", 0) >= 1
    doc_ids = {r["doc_id"] for r in res["results"]}
    assert "D07" in doc_ids, f"D07 (réglementaire) absent — vu : {doc_ids}"


@pytest.mark.skipif(not _index_available(),
                    reason="FAISS index non construit (ingest_doctrine.py)")
def test_e2e_filter_by_doc_id():
    """Filtre doc_id=D03 → uniquement chunks du document D03."""
    from tools.conversation.search_doctrine import run
    res = run(None, {
        "query": "lissage",
        "k": 5,
        "filters": {"doc_id": "D03"},
    })
    if res.get("n_returned", 0) > 0:
        doc_ids = {r["doc_id"] for r in res["results"]}
        assert doc_ids == {"D03"}, f"Filtre violé — vu : {doc_ids}"


@pytest.mark.skipif(not _index_available(),
                    reason="FAISS index non construit (ingest_doctrine.py)")
@pytest.mark.skipif(not _index_available(),
                    reason="FAISS index non construit (ingest_doctrine.py)")
def test_e2e_works_without_dataframe_loaded():
    """Régression bug terrain : search_doctrine doit fonctionner sans
    aucun CSV chargé (df=None côté tool_registry). Sans ce test, le
    routing `if df is None: return erreur` du namespace conversation
    bloquait l'appel et le LLM renvoyait 'aucun DataFrame disponible'."""
    from tools.tool_registry import call_tool
    res = call_tool("conversation", "search_doctrine",
                    params={"query": "Whittaker-Henderson", "k": 2},
                    df=None, data={})
    assert "erreur" not in res, f"Bug : df=None bloque search_doctrine : {res}"
    assert res.get("n_returned", 0) >= 1
    assert res["results"][0]["doc_id"]


@pytest.mark.skipif(not _index_available(),
                    reason="FAISS index non construit (ingest_doctrine.py)")
def test_e2e_result_has_citation_fields():
    """Chaque résultat contient les champs nécessaires à la citation
    (doc_id, section_id, section_title)."""
    from tools.conversation.search_doctrine import run
    res = run(None, {"query": "test du chi-2", "k": 3})
    for r in res.get("results", []):
        assert r["doc_id"]
        assert r["section_id"]
        assert r["section_title"]
        assert r["text"]
