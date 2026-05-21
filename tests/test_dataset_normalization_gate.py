"""
HOTFIX-pre-refacto-2026-05 — Bug 22 : gate « base de données utilisable ».

Le Builder ne travaille QUE sur le clone normalisé. Tant qu'un fichier
uploadé n'a pas été mappé+normalisé (pas de clone), le Master refuse tout
calcul et exige le mapping.
"""
from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import HumanMessage

from agents.mortality.agents.master_node import _dataset_state


# ── Helper _dataset_state ───────────────────────────────────────────────────

def test_dataset_state_none() -> None:
    assert _dataset_state({}, None) == "none"
    assert _dataset_state({}, "") == "none"


def test_dataset_state_raw() -> None:
    """Fichier chargé (dataset_ref) mais pas de clone → 'raw'."""
    assert _dataset_state({}, "sess_x") == "raw"
    assert _dataset_state({"_dataset_ref": "sess_x"}, None) == "raw"
    # records_normalized=True mais sans dataset_ref_normalized → toujours raw
    assert _dataset_state({"records_normalized": True}, "sess_x") == "raw"
    # dataset_ref_normalized sans records_normalized → toujours raw
    assert _dataset_state({"dataset_ref_normalized": "/x.parquet"}, "sess_x") == "raw"


def test_dataset_state_normalized() -> None:
    """records_normalized + dataset_ref_normalized → 'normalized'."""
    ds = {
        "_dataset_ref":           "sess_x",
        "records_normalized":     True,
        "dataset_ref_normalized": "/tmp/sess_normalized.parquet",
    }
    assert _dataset_state(ds, "sess_x") == "normalized"


# ── Gate dans master_node ───────────────────────────────────────────────────

def _calc_state(data_store: dict):
    return {
        "messages":   [HumanMessage(content="construis une table de mortalité lissée")],
        "data_store": data_store,
        "dataset_ref": data_store.get("_dataset_ref"),
    }


def test_gate_refuses_calc_when_no_dataset() -> None:
    """Aucun fichier → refus 'uploadez un fichier'."""
    from agents.mortality.agents import master_node as mn

    with patch("openai.OpenAI"), \
         patch.object(mn, "_classify_intent",
                      return_value={"intent": "build_and_write", "kind": "task",
                                    "write": "yes", "report_mode": "full_report",
                                    "confidence": 1.0, "reply": ""}):
        result = mn.master_node(_calc_state({"_disambiguation_done": True}))

    text = " ".join(str(m.content) for m in (result.get("messages") or []))
    assert "besoin d'un fichier" in text or "Uploadez" in text
    assert result.get("active_agent") != "builder"


def test_gate_opens_mapping_modal_when_dataset_raw() -> None:
    """Fichier chargé mais pas de clone → le gate OUVRE le modal de mapping
    (event disambiguation_required) au lieu de refuser à vide."""
    from agents.mortality.agents import master_node as mn

    data_store = {
        "_disambiguation_done": True,
        "_dataset_ref":         "sess_raw",
        # pas de records_normalized / dataset_ref_normalized → état 'raw'
    }
    with patch("openai.OpenAI"), \
         patch.object(mn, "_classify_intent",
                      return_value={"intent": "build_and_write", "kind": "task",
                                    "write": "yes", "report_mode": "full_report",
                                    "confidence": 1.0, "reply": ""}):
        result = mn.master_node(_calc_state(data_store))

    # Pas de routage Builder
    assert result.get("active_agent") != "builder"
    # Un event disambiguation_required est émis (ouverture du modal mapping)
    disam = [e for e in (result.get("events") or [])
             if e.get("type") == "disambiguation_required"]
    assert len(disam) == 1, f"event disambiguation_required manquant : {result.get('events')}"
    assert disam[0]["needs_column_mapping"] is True


def test_gate_allows_calc_when_dataset_normalized() -> None:
    """Clone normalisé présent → le gate laisse passer (pas de refus base)."""
    from agents.mortality.agents import master_node as mn

    data_store = {
        "_disambiguation_done":   True,
        "_methods_question_done": True,
        "_dataset_ref":           "sess_norm",
        "records_normalized":     True,
        "dataset_ref_normalized": "/tmp/sess_norm_normalized.parquet",
        "study_plan":             {"gender_segmentation": "unisex"},
    }
    with patch("openai.OpenAI"), \
         patch.object(mn, "_classify_intent",
                      return_value={"intent": "build_only", "kind": "task",
                                    "write": "no", "report_mode": "full_report",
                                    "confidence": 1.0, "reply": ""}):
        result = mn.master_node(_calc_state(data_store))

    text = " ".join(str(m.content) for m in (result.get("messages") or []))
    # PAS de refus lié à la base
    assert "pas encore été mappé" not in text
    assert "besoin d'un fichier" not in text
