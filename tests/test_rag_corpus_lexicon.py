"""Tests du lexique auto-derivé depuis le meta.json du corpus FAISS."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest


def _fake_meta() -> dict:
    return {
        "chunks": [
            {"doc_id": "D03", "section_id": "D03.02",
             "section_title": "Whittaker-Henderson 1D",
             "tags": ["lissage"]},
            {"doc_id": "D02", "section_id": "D02.01",
             "section_title": "Estimateur de Kaplan-Meier",
             "tags": ["estimation"]},
            {"doc_id": "D07", "section_id": "D07.01",
             "section_title": "Article A132-18 Code des assurances",
             "tags": ["réglementaire"]},
        ]
    }


def test_lexicon_extracts_doc_ids_and_section_ids():
    from agents.rag.pipeline import _corpus_lexicon as cl
    with patch.object(cl, "_load_meta", return_value=_fake_meta()):
        cl._LEXICON_CACHE = None  # invalidate
        lexicon = cl.build_lexicon_from_meta()
    assert "d03" in lexicon
    assert "d03.02" in lexicon
    assert "d02.01" in lexicon


def test_lexicon_extracts_section_title_words():
    from agents.rag.pipeline import _corpus_lexicon as cl
    with patch.object(cl, "_load_meta", return_value=_fake_meta()):
        cl._LEXICON_CACHE = None
        lexicon = cl.build_lexicon_from_meta()
    assert "whittaker-henderson" in lexicon
    assert "kaplan-meier" in lexicon
    assert "a132-18" in lexicon


def test_lexicon_extracts_tags():
    from agents.rag.pipeline import _corpus_lexicon as cl
    with patch.object(cl, "_load_meta", return_value=_fake_meta()):
        cl._LEXICON_CACHE = None
        lexicon = cl.build_lexicon_from_meta()
    assert "lissage" in lexicon
    assert "estimation" in lexicon
    assert "réglementaire" in lexicon


def test_lexicon_ignores_stop_words_short_tokens():
    from agents.rag.pipeline import _corpus_lexicon as cl
    with patch.object(cl, "_load_meta", return_value=_fake_meta()):
        cl._LEXICON_CACHE = None
        lexicon = cl.build_lexicon_from_meta()
    # "de", "le", "la", "1d" (court) ne sont pas du lexique technique
    assert "de" not in lexicon
    assert "le" not in lexicon
    assert "la" not in lexicon


def test_get_lexicon_uses_cache():
    from agents.rag.pipeline import _corpus_lexicon as cl
    cl._LEXICON_CACHE = {"cached_term"}
    cl._LEXICON_MTIME = 9999999999.0  # future, jamais invalidé
    lexicon = cl.get_lexicon()
    assert lexicon == {"cached_term"}


def test_get_lexicon_invalidates_on_mtime_change(tmp_path, monkeypatch):
    from agents.rag.pipeline import _corpus_lexicon as cl
    fake_meta_file = tmp_path / "meta.json"
    fake_meta_file.write_text(json.dumps(_fake_meta()))
    monkeypatch.setattr(cl, "_META_PATH", fake_meta_file)
    cl._LEXICON_CACHE = None
    cl._LEXICON_MTIME = 0.0
    lex1 = cl.get_lexicon()
    assert "whittaker-henderson" in lex1


def test_lexicon_excludes_common_french_stop_words():
    """Le lexique ne doit PAS contenir les mots courts génériques français
    (loi, non, vie, exp, cas, avec, ...) qui causent des faux positifs
    de _safety.is_in_scope sur des questions hors-actuariat."""
    from agents.rag.pipeline._corpus_lexicon import get_lexicon
    lex = get_lexicon()
    problematic = {"loi", "lois", "non", "vie", "exp", "cas",
                   "avec", "sans", "dans", "est", "sont", "ont",
                   "plus", "moins", "qui", "que", "tout", "tous",
                   "this", "that", "with", "from", "all", "any"}
    leaked = problematic & lex
    assert not leaked, f"Stop-words leaked into lexicon : {sorted(leaked)}"
