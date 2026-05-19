"""
HOTFIX-pre-refacto-2026-05 — Bug 3 : extract_user_answer doit reconnaître
les aliases FR ("par sexe", "agrégé", ...) AVANT de tomber sur le LLM, pour
les questions de type gender_segmentation.

La liste d'aliases vit déjà dans agents/master/extract_gender.py. Ce fix
branche cette source dans extract_user_answer pour le context_key
'gender_segmentation'.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.master.question_filter import extract_user_answer


GENDER_NEED = {
    "context_key": "gender_segmentation",
    "question": "Voulez-vous une table agrégée (unisex) ou des tables séparées par sexe (H/F) ?",
    "options": ["unisex", "by_sex"],
    "default": "unisex",
}


@pytest.mark.parametrize("user_response", [
    "par sexe",
    "Par sexe",
    "par genre",
    "H/F",
    "h/f",
    "tables séparées",
    "hommes et femmes",
    "homme/femme",
])
def test_by_sex_aliases_resolved_without_llm(user_response: str) -> None:
    """Les aliases by_sex doivent être reconnus sans appeler le LLM."""
    with patch("agents.master.question_filter._call_mini_for_inference") as mock_llm:
        result = extract_user_answer(user_response, GENDER_NEED)
        assert result == "by_sex", f"'{user_response}' devrait mapper à by_sex"
        mock_llm.assert_not_called()


@pytest.mark.parametrize("user_response", [
    "unisex",
    "Unisex",
    "agrégé",
    "agrege",
    "table agrégée",
    "sans distinction de sexe",
    "tous sexes confondus",
])
def test_unisex_aliases_resolved_without_llm(user_response: str) -> None:
    """Les aliases unisex doivent être reconnus sans appeler le LLM."""
    with patch("agents.master.question_filter._call_mini_for_inference") as mock_llm:
        result = extract_user_answer(user_response, GENDER_NEED)
        assert result == "unisex", f"'{user_response}' devrait mapper à unisex"
        mock_llm.assert_not_called()


def test_ambiguous_response_falls_back_to_llm() -> None:
    """Une réponse non couverte par les aliases doit appeler le LLM."""
    with patch("agents.master.question_filter._call_mini_for_inference") as mock_llm:
        mock_llm.return_value = {"answered": True, "value": "by_sex", "confidence": 0.8}
        result = extract_user_answer("quelque chose d'ambigu", GENDER_NEED)
        assert result == "by_sex"
        mock_llm.assert_called_once()


def test_non_gender_context_still_uses_llm() -> None:
    """Pour les questions non-gender, le hotfix ne s'applique pas — LLM appelé."""
    other_need = {
        "context_key": "report_mode",
        "question": "Mode de rapport ?",
        "options": ["full_report", "raw_rates"],
    }
    with patch("agents.master.question_filter._call_mini_for_inference") as mock_llm:
        mock_llm.return_value = {"answered": True, "value": "full_report", "confidence": 0.9}
        result = extract_user_answer("complet", other_need)
        assert result == "full_report"
        mock_llm.assert_called_once()


def test_empty_response_returns_none() -> None:
    """Comportement existant préservé : réponse vide -> None."""
    assert extract_user_answer("", GENDER_NEED) is None
    assert extract_user_answer("   ", GENDER_NEED) is None


def test_no_options_returns_text_stripped() -> None:
    """Comportement existant préservé : si pas d'options, retour texte trimé."""
    need_no_options = {"context_key": "free_text", "question": "?"}
    assert extract_user_answer("  réponse libre  ", need_no_options) == "réponse libre"
