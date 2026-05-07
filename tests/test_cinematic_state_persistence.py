"""Tests : SessionState persiste les flags cinématiques entre les tours.

Les flags `_pending_need`, `_user_messages`, `_master_builder_cycles`,
`_questions_asked_this_cycle`, `_write`, `_kind`, `report_mode`,
`_write_question_asked` doivent survivre à un cycle update→to_data_store.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _round_trip(initial_data_store: dict) -> dict:
    """Simule un cycle complet : update_from_data_store → to_data_store."""
    from session.session_state import SessionState
    state = SessionState(session_id="test-session")
    state.update_from_data_store(initial_data_store)
    return state.to_data_store()


def test_pending_need_survives_round_trip():
    ds = {
        "_pending_need": {
            "context_key": "gender_segmentation",
            "question":    "unisex ou by_sex ?",
            "options":     ["unisex", "by_sex"],
            "default":     "unisex",
        },
    }
    after = _round_trip(ds)
    assert after.get("_pending_need") == ds["_pending_need"]


def test_user_messages_survive_round_trip():
    ds = {"_user_messages": ["construit un rapport", "table unisex"]}
    after = _round_trip(ds)
    assert after.get("_user_messages") == ds["_user_messages"]


def test_master_builder_cycles_survive_round_trip():
    ds = {"_master_builder_cycles": 2, "_questions_asked_this_cycle": 1}
    after = _round_trip(ds)
    assert after.get("_master_builder_cycles") == 2
    assert after.get("_questions_asked_this_cycle") == 1


def test_axes_classification_survive_round_trip():
    ds = {"_kind": "task", "_write": "yes", "report_mode": "raw_rates",
          "_write_question_asked": True}
    after = _round_trip(ds)
    assert after.get("_kind") == "task"
    assert after.get("_write") == "yes"
    assert after.get("report_mode") == "raw_rates"
    assert after.get("_write_question_asked") is True


def test_pending_need_pop_propagates_to_session():
    """Quand le code consomme _pending_need (data_store.pop), le tour suivant
    ne doit PAS le voir réapparaître via la persistance."""
    from session.session_state import SessionState
    state = SessionState(session_id="test-pop-propagation")

    # Tour 1 : Master pose la question, _pending_need est dans le data_store
    state.update_from_data_store({
        "_pending_need": {"context_key": "gender_segmentation",
                          "options": ["unisex", "by_sex"]},
    })
    ds_after_t1 = state.to_data_store()
    assert "_pending_need" in ds_after_t1   # bien persisté

    # Tour 2 : data_store est passé vide pour cette clé (Master a pop)
    # On simule le data_store du tour 2 où _pending_need a été consommé.
    ds_t2 = ds_after_t1.copy()
    ds_t2.pop("_pending_need", None)        # consommé pendant le tour
    state.update_from_data_store(ds_t2)
    ds_after_t2 = state.to_data_store()

    # _pending_need ne doit PAS réapparaître
    assert "_pending_need" not in ds_after_t2, (
        f"_pending_need ressuscité par la persistance : {ds_after_t2.get('_pending_need')!r}"
    )


def test_empty_data_store_no_crash():
    after = _round_trip({})
    # On ne crash pas et on ne pollue pas avec des None partout
    assert "_pending_need" not in after
    assert "_user_messages" not in after
