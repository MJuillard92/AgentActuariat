"""Tests — protocole de désambiguïsation LLM ↔ regex.

Le LLM classify_intent et la regex regex_kind_hint travaillent en parallèle.
En cas de désaccord, master_node demande à l'utilisateur de trancher (A/B/
Autre). Pas d'override silencieux.

Plan disambiguation 2026-05-25.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Annotation classify_intent ───────────────────────────────────────────────

def _fake_llm_classify_factory(kind: str, write: str = "ask",
                                report_mode: str = "full_report"):
    """Construit un mock _llm_classify qui retourne un kind donné."""
    def _fake(*args, **kwargs):
        return {
            "kind":                kind,
            "write":               write,
            "report_mode":         report_mode,
            "gender_segmentation": None,
            "confidence":          0.95,
            "reasoning":           "mock",
            "reply":               "mock reply",
        }
    return _fake


def test_classify_intent_annotates_hint_and_disagreement_task_capability():
    """LLM dit task, regex hint = question → _regex_disagrees=True."""
    from agents.master import classify_intent as ci_mod

    with patch.object(ci_mod, "_llm_classify",
                      _fake_llm_classify_factory("task")):
        # "peux-tu calculer ?" → regex hint = question (méta sans objet)
        result = ci_mod.classify_intent("peux-tu calculer ?", has_data=True)

    assert result["kind"] == "task"
    assert result["_regex_kind_hint"] == "question"
    assert result["_regex_disagrees"] is True


def test_classify_intent_annotates_hint_and_disagreement_question_command():
    """LLM dit question, regex hint = task → _regex_disagrees=True."""
    from agents.master import classify_intent as ci_mod

    with patch.object(ci_mod, "_llm_classify",
                      _fake_llm_classify_factory("question")):
        # "calcule les taux bruts" → regex hint = task (impératif + objet)
        result = ci_mod.classify_intent("calcule les taux bruts", has_data=True)

    assert result["kind"] == "question"
    assert result["_regex_kind_hint"] == "task"
    assert result["_regex_disagrees"] is True


def test_classify_intent_no_disagreement_when_aligned():
    """LLM et regex d'accord → _regex_disagrees=False."""
    from agents.master import classify_intent as ci_mod

    with patch.object(ci_mod, "_llm_classify",
                      _fake_llm_classify_factory("task")):
        result = ci_mod.classify_intent("calcule les taux bruts", has_data=True)

    assert result["_regex_kind_hint"] == "task"
    assert result["_regex_disagrees"] is False


def test_classify_intent_no_disagreement_when_hint_none():
    """Regex hint = None (texte ambigu) → pas de désaccord, on suit le LLM."""
    from agents.master import classify_intent as ci_mod

    with patch.object(ci_mod, "_llm_classify",
                      _fake_llm_classify_factory("question")):
        result = ci_mod.classify_intent(
            "c'est quoi le lissage Whittaker-Henderson ?", has_data=True,
        )

    assert result["_regex_kind_hint"] is None
    assert result["_regex_disagrees"] is False


# ── Helper _kind_to_human ────────────────────────────────────────────────────

def test_kind_to_human_task_full_report_with_pdf():
    from agents.mortality.agents.master_node import _kind_to_human
    s = _kind_to_human("task", {"report_mode": "full_report", "write": "yes"})
    assert "complète" in s.lower() or "lissage" in s.lower()
    assert "pdf" in s.lower() or "rapport" in s.lower()


def test_kind_to_human_task_raw_rates_no_pdf():
    from agents.mortality.agents.master_node import _kind_to_human
    s = _kind_to_human("task", {"report_mode": "raw_rates", "write": "no"})
    assert "bruts" in s.lower()
    assert "sans" in s.lower()


def test_kind_to_human_question():
    from agents.mortality.agents.master_node import _kind_to_human
    s = _kind_to_human("question", {})
    assert "question" in s.lower() or "capacit" in s.lower()


def test_kind_to_human_none_returns_unknown():
    from agents.mortality.agents.master_node import _kind_to_human
    s = _kind_to_human(None, {})
    assert "inconnue" in s.lower() or "unknown" in s.lower()


# ── Handler de réponse A/B/Autre (logique data_store) ────────────────────────

def _make_pending_state(llm_kind="task", regex_kind="question",
                        write="yes", report_mode="full_report"):
    """État data_store simulant une désambiguïsation en attente."""
    return {
        "_kind_disambiguation_pending": {
            "original_text":      "peux-tu calculer ?",
            "llm_kind":           llm_kind,
            "regex_kind":         regex_kind,
            "llm_classification": {
                "kind":        llm_kind,
                "write":       write,
                "report_mode": report_mode,
            },
        },
    }


def test_handler_response_A_applies_llm_kind_and_clears_pending():
    """Simule le handler : réponse 'A' → kind LLM appliqué, pending effacé,
    write/scope LLM restaurés."""
    data_store = _make_pending_state(llm_kind="task", regex_kind="question",
                                      write="yes", report_mode="full_report")

    # Simuler le code du handler 1b-ter
    pending = data_store["_kind_disambiguation_pending"]
    answer = "A"
    choice = answer.strip().upper()
    assert choice in ("A", "B")
    chosen_kind = (pending["llm_kind"] if choice == "A" else pending["regex_kind"])
    data_store["_kind_disambiguation_resolved"] = chosen_kind
    data_store.pop("_kind_disambiguation_pending", None)
    saved = pending.get("llm_classification") or {}
    if saved.get("write") in ("yes", "no"):
        data_store["_write"] = saved["write"]
    if saved.get("report_mode") in ("full_report", "raw_rates", "description"):
        data_store["report_mode"] = saved["report_mode"]

    assert data_store["_kind_disambiguation_resolved"] == "task"
    assert "_kind_disambiguation_pending" not in data_store
    assert data_store["_write"] == "yes"
    assert data_store["report_mode"] == "full_report"


def test_handler_response_B_applies_regex_kind_and_clears_pending():
    """Réponse 'B' → kind regex appliqué."""
    data_store = _make_pending_state(llm_kind="task", regex_kind="question")

    pending = data_store["_kind_disambiguation_pending"]
    choice = "B"
    chosen_kind = (pending["llm_kind"] if choice == "A" else pending["regex_kind"])
    data_store["_kind_disambiguation_resolved"] = chosen_kind
    data_store.pop("_kind_disambiguation_pending", None)

    assert data_store["_kind_disambiguation_resolved"] == "question"
    assert "_kind_disambiguation_pending" not in data_store


import re

# Même regex que dans master_node 1b-ter — extraction stricte de A/B
_CHOICE_RE = re.compile(r"^\s*([ABab])(?:\s|[.,;:!?\-)]|$)")


def _extract_choice(answer: str) -> str:
    m = _CHOICE_RE.match(answer or "")
    return m.group(1).upper() if m else ""


@pytest.mark.parametrize("answer,expected", [
    ("A",                   "A"),
    ("a",                   "A"),
    ("A.",                  "A"),
    ("A)",                  "A"),
    ("A car ...",           "A"),
    ("A, je préfère",       "A"),
    ("B",                   "B"),
    ("B.",                  "B"),
    ("Autre",               ""),    # NE doit PAS matcher "A"
    ("Autre, ...",          ""),
    ("autre",               ""),
    ("aucun de ces choix",  ""),
    ("calcule la table",    ""),
    ("",                    ""),
])
def test_choice_extraction_handles_autre_correctly(answer, expected):
    """Régression critique : « Autre » commence par 'A' — ne doit PAS être
    confondu avec le choix « A »."""
    assert _extract_choice(answer) == expected


def test_handler_response_autre_clears_pending_no_resolution():
    """Réponse 'Autre' → on efface pending et resolved, le flux reclasse
    le nouveau texte au prochain passage."""
    data_store = _make_pending_state()
    data_store["_kind_disambiguation_resolved"] = "task"  # résiduel

    answer = "Autre, je voudrais en fait calculer une table H/F"
    choice = _extract_choice(answer)
    assert choice == ""  # ni A ni B

    data_store.pop("_kind_disambiguation_pending", None)
    data_store.pop("_kind_disambiguation_resolved", None)

    assert "_kind_disambiguation_pending" not in data_store
    assert "_kind_disambiguation_resolved" not in data_store


def test_pending_state_serializable_for_langgraph():
    """Le pending state doit être un dict de types primitifs (orjson safe)."""
    import json
    pending = _make_pending_state()
    s = json.dumps(pending)  # ne doit pas lever
    assert "_kind_disambiguation_pending" in s
