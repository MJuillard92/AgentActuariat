"""Tests du answer_generator RAG.

Synthèse rédigée à partir de chunks doctrinaux, avec citations `[Dxx.yy]`
inline et section Sources finale. LLM mini mocké — on teste le wiring,
le formatage des chunks injectés, et le pass-through propre.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _mock_openai_response(text: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = text
    response.choices = [choice]
    return response


def _sample_chunks() -> list[dict]:
    """Échantillon de 2 chunks comme retournés par search_doctrine."""
    return [
        {
            "chunk_id":      "ch_001",
            "doc_id":        "D03",
            "section_id":    "D03.02",
            "section_title": "Whittaker-Henderson 1D",
            "text":          "Le lissage de Whittaker-Henderson pénalise "
                             "les différences finies d'ordre k. Le paramètre h "
                             "contrôle l'arbitrage biais-variance.",
            "score":         0.91,
            "tags":          ["lissage"],
        },
        {
            "chunk_id":      "ch_002",
            "doc_id":        "D03",
            "section_id":    "D03.04",
            "section_title": "Sélection du paramètre h (Biessy 2023)",
            "text":          "Le choix optimal de h s'effectue par "
                             "validation croisée ou critère AIC.",
            "score":         0.87,
            "tags":          ["lissage"],
        },
    ]


# ──────────────────────────────────────────────────────────────────────
# generate() — wiring de base
# ──────────────────────────────────────────────────────────────────────

def test_generate_returns_llm_output_unchanged_when_well_formatted():
    """Si le LLM retourne déjà une réponse bien formatée, on la propage."""
    from agents.rag.pipeline import answer_generator

    well_formatted = (
        "Le lissage de Whittaker-Henderson pénalise les différences "
        "finies d'ordre k [D03.02]. Le paramètre h s'optimise par "
        "validation croisée [D03.04].\n\n"
        "Sources :\n"
        "- D03.02 — Whittaker-Henderson 1D\n"
        "- D03.04 — Sélection du paramètre h (Biessy 2023)"
    )
    fake = _mock_openai_response(well_formatted)
    with patch("agents.rag.pipeline.answer_generator.openai.OpenAI"), \
         patch("agents.rag.pipeline.answer_generator.call_with_retry", return_value=fake):
        out = answer_generator.generate("c'est quoi whittaker ?", _sample_chunks())

    assert "[D03.02]" in out
    assert "[D03.04]" in out
    assert "Sources :" in out


def test_generate_passes_chunks_to_prompt():
    """Le texte des chunks doit apparaître dans le prompt user envoyé au LLM."""
    from agents.rag.pipeline import answer_generator

    fake = _mock_openai_response("ok")
    with patch("agents.rag.pipeline.answer_generator.openai.OpenAI"), \
         patch("agents.rag.pipeline.answer_generator.call_with_retry",
               return_value=fake) as mock_call:
        answer_generator.generate("question", _sample_chunks())

    messages = mock_call.call_args.kwargs["messages"]
    full_payload = "\n".join((m.get("content") or "") for m in messages)
    # Les section_id et le texte des chunks doivent être dans le prompt
    assert "D03.02" in full_payload
    assert "D03.04" in full_payload
    assert "Whittaker-Henderson" in full_payload
    assert "validation croisée" in full_payload


def test_generate_includes_original_question_in_prompt():
    from agents.rag.pipeline import answer_generator

    fake = _mock_openai_response("ok")
    with patch("agents.rag.pipeline.answer_generator.openai.OpenAI"), \
         patch("agents.rag.pipeline.answer_generator.call_with_retry",
               return_value=fake) as mock_call:
        answer_generator.generate("c'est quoi le paramètre h ?", _sample_chunks())

    messages = mock_call.call_args.kwargs["messages"]
    payload = "\n".join((m.get("content") or "") for m in messages)
    assert "c'est quoi le paramètre h" in payload


# ──────────────────────────────────────────────────────────────────────
# Cas dégénérés
# ──────────────────────────────────────────────────────────────────────

def test_generate_no_chunks_returns_no_coverage_message_without_llm_call():
    """0 chunk retourné par le retriever → on émet directement le message
    'corpus ne couvre pas' SANS appeler le LLM (économie + déterminisme)."""
    from agents.rag.pipeline import answer_generator

    with patch("agents.rag.pipeline.answer_generator.call_with_retry") as mock_call:
        out = answer_generator.generate("question hors-corpus", [])

    assert "corpus" in out.lower()
    assert "couvre" in out.lower() or "couverture" in out.lower()
    mock_call.assert_not_called()


def test_generate_uses_mini_role_config():
    from agents.rag.pipeline import answer_generator

    fake = _mock_openai_response("réponse")
    with patch("agents.rag.pipeline.answer_generator.openai.OpenAI"), \
         patch("agents.rag.pipeline.answer_generator.call_with_retry", return_value=fake), \
         patch("agents.rag.pipeline.answer_generator.get_llm_config",
               return_value={"model": "gpt-5.4-mini", "temperature": 0.3,
                             "max_tokens": 1500}) as mock_cfg:
        answer_generator.generate("question", _sample_chunks())

    mock_cfg.assert_called_with("rag.answer_generator")


def test_generate_falls_back_on_llm_error():
    """Si l'appel LLM échoue, on retombe sur une réponse dégradée mais utile :
    un message d'erreur explicite + les sources brutes en clair."""
    from agents.rag.pipeline import answer_generator

    with patch("agents.rag.pipeline.answer_generator.openai.OpenAI"), \
         patch("agents.rag.pipeline.answer_generator.call_with_retry",
               side_effect=RuntimeError("openai 500")):
        out = answer_generator.generate("question", _sample_chunks())

    # On doit retourner quelque chose, pas raise
    assert isinstance(out, str)
    assert len(out) > 0
    # Les sources doivent au moins apparaître
    assert "D03.02" in out


# ──────────────────────────────────────────────────────────────────────
# Anti-régressions : doc_id NON dupliqué
# ──────────────────────────────────────────────────────────────────────

def test_chunks_format_does_not_duplicate_doc_id():
    """Bug terrain : section_id contient déjà le préfixe doc_id, donc
    `f'{doc_id}.{section_id}'` produit 'D03.D03.02'. Le formatter doit
    afficher proprement 'D03.02' (= section_id seul)."""
    from agents.rag.pipeline.answer_generator import _format_chunks_for_prompt

    formatted = _format_chunks_for_prompt(_sample_chunks())
    assert "D03.D03" not in formatted, (
        f"Duplication doc_id détectée dans le formatage chunks :\n{formatted}"
    )
    # En revanche on doit bien y retrouver le section_id propre
    assert "D03.02" in formatted
    assert "D03.04" in formatted
