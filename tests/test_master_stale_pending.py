"""
HOTFIX-pre-refacto-2026-05 — Bug 20 : garde anti-_pending_need périmé.

Scénario prod : tour 2 résout gender_segmentation=by_sex. Tour 3, l'utilisateur
dit « oui » (continuation). Un _pending_need gender_segmentation résiduel
faisait lire « oui » comme une réponse gender → extract_user_answer=None →
re-pose la question en boucle.

Le fix : si la clé du _pending_need est déjà résolue dans study_plan, le
pending est périmé → ignoré, le message traité normalement.
"""
from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import HumanMessage


def _extract_stage(events, stage_id):
    return next((e for e in (events or [])
                 if e.get("type") == "master_stage" and e.get("stage") == stage_id), None)


def test_stale_pending_discarded_when_key_already_resolved() -> None:
    """_pending_need gender_segmentation + study_plan a déjà gender → pending ignoré."""
    from agents.mortality.agents import master_node as mn

    state = {
        "messages":   [HumanMessage(content="oui")],
        "data_store": {
            "_pending_need": {
                "context_key": "gender_segmentation",
                "question":    "unisex ou by_sex ?",
                "options":     ["unisex", "by_sex"],
            },
            "study_plan":  {"gender_segmentation": "by_sex"},  # DÉJÀ résolu
            "report_mode": "full_report",
        },
        "dataset_ref": "test_session",
    }

    with patch("openai.OpenAI"), \
         patch.object(mn, "_classify_intent",
                      return_value={"intent": "question", "kind": "question",
                                    "write": "ask", "report_mode": "full_report",
                                    "confidence": 0.9, "reply": ""}):
        result = mn.master_node(state)

    # Le pending périmé doit être retiré
    assert result["data_store"].get("_pending_need") is None
    # Stage de traçabilité émis
    assert _extract_stage(result.get("events"), "0.c-stale") is not None
    # PAS de re-pose : aucun message "Je n'ai pas bien compris"
    msgs_text = " ".join(
        str(m.content) for m in (result.get("messages") or []) if hasattr(m, "content")
    )
    assert "n'ai pas bien compris" not in msgs_text


def test_fresh_pending_still_processed_normally() -> None:
    """_pending_need dont la clé n'est PAS résolue → traité normalement."""
    from agents.mortality.agents import master_node as mn

    state = {
        "messages":   [HumanMessage(content="par sexe")],
        "data_store": {
            "_pending_need": {
                "context_key": "gender_segmentation",
                "question":    "unisex ou by_sex ?",
                "options":     ["unisex", "by_sex"],
            },
            "study_plan":  {},          # gender PAS encore résolu
            "report_mode": "full_report",
        },
        "dataset_ref": "test_session",
    }

    with patch("openai.OpenAI"):
        result = mn.master_node(state)

    # La réponse "par sexe" est traitée → gender enregistré
    assert result["data_store"].get("study_plan", {}).get("gender_segmentation") == "by_sex"
    # Le stage normal 0.c (pas 0.c-stale)
    assert _extract_stage(result.get("events"), "0.c") is not None
    assert _extract_stage(result.get("events"), "0.c-stale") is None


def test_stale_guard_ignores_method_pending() -> None:
    """Un pending method_* n'est pas faussement écarté (clé absente de study_plan
    au niveau scalaire — les méthodes sont imbriquées sous study_plan['methods'])."""
    from agents.mortality.agents import master_node as mn

    state = {
        "messages":   [HumanMessage(content="kaplan_meier")],
        "data_store": {
            "_pending_need": {
                "context_key": "method_crude_rates",
                "question":    "Quelle méthode pour les taux bruts ?",
            },
            "study_plan":  {"methods": {}},  # pas de clé scalaire 'method_crude_rates'
            "report_mode": "full_report",
        },
        "dataset_ref": "test_session",
    }

    with patch("openai.OpenAI"):
        result = mn.master_node(state)

    # Le pending method n'est PAS écarté par la garde anti-périmé
    assert _extract_stage(result.get("events"), "0.c-stale") is None
