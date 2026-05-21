"""
Plan « bouton Valider le mapping » — Étapes 3 & 4.

Gate : BuilderAgent/WriterAgent exigent `mapping_validated=True` (clone créé
via le bouton). Sinon refus terminal. L'exploration (intent question) n'est
pas gatée. Le drapeau survit aux tours (persistance cinematic_state).
"""
from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import HumanMessage


def _calc_state(data_store: dict) -> dict:
    return {
        "messages":    [HumanMessage(content="construis une table de mortalité lissée")],
        "data_store":  data_store,
        "dataset_ref": data_store.get("_dataset_ref"),
    }


def _refusal_text(result) -> str:
    return " ".join(str(m.content) for m in (result.get("messages") or [])
                     if hasattr(m, "content"))


# ── Étape 3 : gate ──────────────────────────────────────────────────────────

def test_gate_refuses_calc_when_mapping_not_validated() -> None:
    """Fichier chargé mais mapping non validé → refus terminal, pas de Builder."""
    from agents.mortality.agents import master_node as mn

    data_store = {"_disambiguation_done": True, "_dataset_ref": "sess_x"}
    with patch("openai.OpenAI"), \
         patch.object(mn, "_classify_intent",
                      return_value={"intent": "build_and_write", "kind": "task",
                                    "write": "yes", "report_mode": "full_report",
                                    "confidence": 1.0, "reply": ""}):
        result = mn.master_node(_calc_state(data_store))

    assert result.get("active_agent") not in ("builder", "writer")
    assert "Valider le mapping" in _refusal_text(result)
    # Refus terminal : event done présent
    assert any(e.get("type") == "done" for e in (result.get("events") or []))


def test_gate_allows_calc_when_mapping_validated() -> None:
    """mapping_validated=True → plus de refus base de travail."""
    from agents.mortality.agents import master_node as mn

    data_store = {
        "_disambiguation_done":   True,
        "_methods_question_done": True,
        "_dataset_ref":           "sess_x",
        "mapping_validated":      True,
        "study_plan":             {"gender_segmentation": "unisex"},
    }
    with patch("openai.OpenAI"), \
         patch.object(mn, "_classify_intent",
                      return_value={"intent": "build_only", "kind": "task",
                                    "write": "no", "report_mode": "full_report",
                                    "confidence": 1.0, "reply": ""}):
        result = mn.master_node(_calc_state(data_store))

    # Pas le refus « Valider le mapping »
    assert "Valider le mapping" not in _refusal_text(result)


def test_gate_refuses_when_no_dataset_at_all() -> None:
    """Aucun fichier → refus 'uploadez un fichier' (garde dataset_ref)."""
    from agents.mortality.agents import master_node as mn

    with patch("openai.OpenAI"), \
         patch.object(mn, "_classify_intent",
                      return_value={"intent": "build_and_write", "kind": "task",
                                    "write": "yes", "report_mode": "full_report",
                                    "confidence": 1.0, "reply": ""}):
        result = mn.master_node(_calc_state({"_disambiguation_done": True}))

    txt = _refusal_text(result)
    assert "fichier CSV" in txt or "Uploadez" in txt
    assert result.get("active_agent") not in ("builder", "writer")


def test_exploration_not_gated_by_mapping_validation() -> None:
    """Un intent 'question' (exploration) n'est PAS bloqué par le gate
    mapping_validated — le Master explore librement sur l'original."""
    from agents.mortality.agents import master_node as mn

    data_store = {"_disambiguation_done": True, "_dataset_ref": "sess_x"}
    state = {
        "messages":    [HumanMessage(content="combien de lignes dans mon fichier ?")],
        "data_store":  data_store,
        "dataset_ref": "sess_x",
    }
    with patch("openai.OpenAI"), \
         patch.object(mn, "_classify_intent",
                      return_value={"intent": "question", "kind": "question",
                                    "write": "ask", "report_mode": "full_report",
                                    "confidence": 0.9, "reply": ""}):
        result = mn.master_node(state)

    # Pas de refus « Valider le mapping » pour une exploration
    assert "Valider le mapping" not in _refusal_text(result)


def test_gate_never_emits_disambiguation_required() -> None:
    """Garde anti-régression : le Master ne doit JAMAIS émettre
    disambiguation_required (cause des boucles des 3 tentatives). Il refuse
    seulement — le modal est ouvert par le bouton UI, pas par l'agent."""
    from agents.mortality.agents import master_node as mn

    data_store = {"_disambiguation_done": True, "_dataset_ref": "sess_x"}
    with patch("openai.OpenAI"), \
         patch.object(mn, "_classify_intent",
                      return_value={"intent": "build_and_write", "kind": "task",
                                    "write": "yes", "report_mode": "full_report",
                                    "confidence": 1.0, "reply": ""}):
        result = mn.master_node(_calc_state(data_store))

    assert not any(e.get("type") == "disambiguation_required"
                   for e in (result.get("events") or []))


# ── Étape 4 : persistance ───────────────────────────────────────────────────

def test_mapping_validated_persisted_across_turns() -> None:
    """mapping_validated survit à update_from_data_store → to_data_store."""
    from session.session_state import SessionState

    st = SessionState(session_id="sess_persist")
    st.update_from_data_store({"mapping_validated": True})
    hydrated = st.to_data_store()
    assert hydrated.get("mapping_validated") is True


def test_mapping_validated_removed_when_absent() -> None:
    """Cohérence : si mapping_validated disparaît du data_store, il disparaît
    aussi de la persistance (pas de résurrection)."""
    from session.session_state import SessionState

    st = SessionState(session_id="sess_persist2")
    st.update_from_data_store({"mapping_validated": True})
    assert st.to_data_store().get("mapping_validated") is True
    # Tour suivant sans la clé
    st.update_from_data_store({})
    assert "mapping_validated" not in st.to_data_store()
