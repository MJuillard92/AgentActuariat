"""Tests du query_rewriter multi-turn (mocks LLM)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.rag.memory.schemas import RAGTurn, RAGSummary


def _mock_resp(text: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = text
    response.choices = [choice]
    return response


# ──────────────────────────────────────────────────────────────────────
# should_rewrite étendu
# ──────────────────────────────────────────────────────────────────────

def test_should_rewrite_forces_on_anaphora_with_buffer():
    """Anaphore + buffer non vide → FORCE rewrite."""
    from agents.rag.pipeline.query_rewriter import should_rewrite
    assert should_rewrite("compare-les", buffer_size=2) is True
    assert should_rewrite("et pour les femmes", buffer_size=1) is True


def test_should_rewrite_skips_anaphora_without_buffer():
    """Anaphore sans contexte → pas de rewrite (rien à résoudre)."""
    from agents.rag.pipeline.query_rewriter import should_rewrite
    # Anaphore mais buffer vide → on laisse passer brut (le retriever fera de son mieux)
    assert should_rewrite("compare-les", buffer_size=0) is False


def test_should_rewrite_skips_declarative_query():
    """Query déclarative (pas de marqueur interrogatif) → skip."""
    from agents.rag.pipeline.query_rewriter import should_rewrite
    from unittest.mock import patch
    with patch("agents.rag.pipeline.query_rewriter.get_lexicon",
               return_value={"whittaker-henderson"}):
        # Pas de "?", "qu'est-ce", "comment", "explique"… → déclarative
        assert should_rewrite("paramètre lissage h Whittaker-Henderson",
                              buffer_size=0) is False


def test_should_rewrite_calls_corpus_lexicon_for_short_queries():
    """Query courte + terme corpus → skip (rewriter inutile)."""
    from agents.rag.pipeline import query_rewriter as qr
    with patch.object(qr, "get_lexicon",
                       return_value={"whittaker-henderson"}):
        assert qr.should_rewrite("c'est quoi whittaker-henderson",
                                  buffer_size=0) is False


def test_should_rewrite_triggers_long_interrogative_query():
    from agents.rag.pipeline import query_rewriter as qr
    with patch.object(qr, "get_lexicon", return_value={"lissage"}):
        assert qr.should_rewrite(
            "comment fait-on pour choisir un bon paramètre de lissage adapté ?",
            buffer_size=0,
        ) is True


# ──────────────────────────────────────────────────────────────────────
# rewrite() avec contexte multi-turn
# ──────────────────────────────────────────────────────────────────────

def test_rewrite_includes_buffer_in_prompt():
    from agents.rag.pipeline import query_rewriter
    buffer = [
        RAGTurn(user_q="qu'est-ce que Whittaker ?",
                rag_answer="...lissage [D03.02]...", sources=[]),
        RAGTurn(user_q="et Kaplan-Meier ?",
                rag_answer="...estimateur [D02.01]...", sources=[]),
    ]
    fake = _mock_resp("comparaison Whittaker Kaplan-Meier")
    with patch("agents.rag.pipeline.query_rewriter.openai.OpenAI"), \
         patch("agents.rag.pipeline.query_rewriter.call_with_retry",
               return_value=fake) as mock_call:
        out = query_rewriter.rewrite("compare-les", buffer=buffer)
    assert out == "comparaison Whittaker Kaplan-Meier"
    messages = mock_call.call_args.kwargs["messages"]
    payload = "\n".join((m.get("content") or "") for m in messages)
    # Les blocs doivent être présents
    assert "[Conversation récente]" in payload
    assert "Whittaker" in payload
    assert "Kaplan-Meier" in payload
    assert "[Nouvelle question]" in payload
    assert "compare-les" in payload


def test_rewrite_includes_summary_in_prompt():
    from agents.rag.pipeline import query_rewriter
    summary = RAGSummary(
        topics_covered=["TH 00-02", "Whittaker"],
        user_focus="tables et lissage",
    )
    fake = _mock_resp("TGH 05 TGF 05 tables réglementaires")
    with patch("agents.rag.pipeline.query_rewriter.openai.OpenAI"), \
         patch("agents.rag.pipeline.query_rewriter.call_with_retry",
               return_value=fake) as mock_call:
        query_rewriter.rewrite("et la version 2005 ?",
                                buffer=[], summary=summary)
    messages = mock_call.call_args.kwargs["messages"]
    payload = "\n".join((m.get("content") or "") for m in messages)
    assert "[Résumé contexte antérieur]" in payload
    assert "TH 00-02" in payload
    assert "tables et lissage" in payload


def test_rewrite_includes_vectorstore_hits_in_prompt():
    from agents.rag.pipeline import query_rewriter
    hits = [
        RAGTurn(user_q="différence taux brut vs lissé ?",
                rag_answer="...estime q_x...", sources=[]),
    ]
    fake = _mock_resp("comparaison taux brut lissé Whittaker-Henderson")
    with patch("agents.rag.pipeline.query_rewriter.openai.OpenAI"), \
         patch("agents.rag.pipeline.query_rewriter.call_with_retry",
               return_value=fake) as mock_call:
        query_rewriter.rewrite("approfondis cette comparaison",
                                buffer=[], summary=None,
                                vectorstore_hits=hits)
    messages = mock_call.call_args.kwargs["messages"]
    payload = "\n".join((m.get("content") or "") for m in messages)
    assert "[Échanges passés pertinents]" in payload
    assert "taux brut vs lissé" in payload


def test_rewrite_omits_empty_blocks():
    """Si tous les contextes sont vides, le prompt n'a pas de bloc inutile."""
    from agents.rag.pipeline import query_rewriter
    fake = _mock_resp("lissage paramètre h")
    with patch("agents.rag.pipeline.query_rewriter.openai.OpenAI"), \
         patch("agents.rag.pipeline.query_rewriter.call_with_retry",
               return_value=fake) as mock_call:
        query_rewriter.rewrite("comment choisir h ?")
    messages = mock_call.call_args.kwargs["messages"]
    payload = "\n".join((m.get("content") or "") for m in messages)
    assert "[Conversation récente]" not in payload
    assert "[Résumé contexte antérieur]" not in payload
    assert "[Échanges passés pertinents]" not in payload
    assert "[Nouvelle question]" in payload
    assert "comment choisir h" in payload


def test_rewrite_backward_compat_with_no_kwargs():
    """rewrite(query) sans args supplémentaires marche comme avant."""
    from agents.rag.pipeline import query_rewriter
    fake = _mock_resp("paramètre lissage h")
    with patch("agents.rag.pipeline.query_rewriter.openai.OpenAI"), \
         patch("agents.rag.pipeline.query_rewriter.call_with_retry",
               return_value=fake):
        out = query_rewriter.rewrite("c'est quoi le truc avec h en lissage ?")
    assert out == "paramètre lissage h"
