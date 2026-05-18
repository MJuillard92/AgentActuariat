"""Tests du query_rewriter RAG.

Le rewriter reformule une question utilisateur en affirmation de recherche
concise pour optimiser le retrieval hybride. LLM nano mocké — on teste
le wiring, pas la qualité de la reformulation.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


# ──────────────────────────────────────────────────────────────────────
# Skip heuristic — les queries courtes et techniques passent through
# ──────────────────────────────────────────────────────────────────────

def test_short_technical_query_skips_llm():
    """Pas d'appel LLM si la query est déjà courte (<40 chars) et contient
    un terme technique connu."""
    from agents.rag.pipeline.query_rewriter import should_rewrite
    assert should_rewrite("whittaker-henderson") is False
    assert should_rewrite("kaplan-meier") is False
    assert should_rewrite("test chi-2") is False
    assert should_rewrite("A132-18") is False


def test_long_query_triggers_rewrite():
    from agents.rag.pipeline.query_rewriter import should_rewrite
    long_q = "c'est quoi le truc avec le paramètre h pour le lissage des taux ?"
    assert should_rewrite(long_q) is True


def test_short_non_technical_query_triggers_rewrite():
    """Une query courte SANS terme technique → rewrite quand même."""
    from agents.rag.pipeline.query_rewriter import should_rewrite
    assert should_rewrite("c'est quoi") is True
    assert should_rewrite("aide") is True


# ──────────────────────────────────────────────────────────────────────
# rewrite() — appel LLM mocké
# ──────────────────────────────────────────────────────────────────────

def _mock_openai_response(text: str) -> MagicMock:
    """Construit une réponse OpenAI mockée."""
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = text
    response.choices = [choice]
    return response


def test_rewrite_calls_llm_with_user_text():
    from agents.rag.pipeline import query_rewriter

    fake = _mock_openai_response("paramètre lissage h méthode Whittaker-Henderson")
    with patch("agents.rag.pipeline.query_rewriter.openai.OpenAI") as mock_client_cls, \
         patch("agents.rag.pipeline.query_rewriter.call_with_retry", return_value=fake) as mock_call:
        out = query_rewriter.rewrite("c'est quoi le truc avec h en lissage des tables ?")

    assert out == "paramètre lissage h méthode Whittaker-Henderson"
    mock_call.assert_called_once()
    # Vérifier que le prompt user contient bien la query
    call_kwargs = mock_call.call_args.kwargs
    messages = call_kwargs["messages"]
    assert any("h en lissage" in (m.get("content") or "") for m in messages)


def test_rewrite_strips_whitespace_and_quotes():
    """Le LLM peut entourer la réponse de guillemets ou retours-ligne — on nettoie."""
    from agents.rag.pipeline import query_rewriter

    fake = _mock_openai_response('  "paramètre h Whittaker-Henderson"\n')
    with patch("agents.rag.pipeline.query_rewriter.openai.OpenAI"), \
         patch("agents.rag.pipeline.query_rewriter.call_with_retry", return_value=fake):
        out = query_rewriter.rewrite("blabla")

    assert out == "paramètre h Whittaker-Henderson"


def test_rewrite_falls_back_to_input_on_llm_error():
    """Si l'appel LLM échoue, on retombe sur la query d'entrée (graceful degradation)."""
    from agents.rag.pipeline import query_rewriter

    with patch("agents.rag.pipeline.query_rewriter.openai.OpenAI"), \
         patch("agents.rag.pipeline.query_rewriter.call_with_retry",
               side_effect=RuntimeError("openai 500")):
        out = query_rewriter.rewrite("ma question originale")

    assert out == "ma question originale"


def test_rewrite_uses_nano_role_config():
    """Le rewriter doit appeler get_llm_config('rag.query_rewriter')."""
    from agents.rag.pipeline import query_rewriter

    fake = _mock_openai_response("ok")
    with patch("agents.rag.pipeline.query_rewriter.openai.OpenAI"), \
         patch("agents.rag.pipeline.query_rewriter.call_with_retry", return_value=fake), \
         patch("agents.rag.pipeline.query_rewriter.get_llm_config",
               return_value={"model": "gpt-5.4-nano", "temperature": 0.0,
                             "max_tokens": 200}) as mock_cfg:
        query_rewriter.rewrite("question quelconque assez longue pour déclencher rewrite")

    mock_cfg.assert_called_with("rag.query_rewriter")
