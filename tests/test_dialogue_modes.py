"""Tests pour agents/master/dialogue_modes.py (US-16)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.master.dialogue_modes import (  # noqa: E402
    DialogueMode,
    choose_mode,
    record_mode,
    get_mode,
)


def test_default_mode_is_proposition_validation():
    assert DialogueMode.DEFAULT == DialogueMode.PROPOSITION_VALIDATION


def test_choose_mode_accepts_shortcut_letters():
    assert choose_mode("a") == DialogueMode.AUTONOMOUS
    assert choose_mode("B") == DialogueMode.USER_FIRST
    assert choose_mode("c") == DialogueMode.PROPOSITION_VALIDATION


def test_choose_mode_accepts_full_names():
    assert choose_mode("autonomous") == DialogueMode.AUTONOMOUS
    assert choose_mode("user_first") == DialogueMode.USER_FIRST


def test_choose_mode_invalid_raises():
    with pytest.raises(ValueError):
        choose_mode("z")


def test_record_and_get_mode_in_data_store():
    ds = {}
    record_mode(ds, phase="data", mode=DialogueMode.AUTONOMOUS)
    assert get_mode(ds, phase="data") == DialogueMode.AUTONOMOUS


def test_audit_log_appended():
    ds = {}
    record_mode(ds, phase="data", mode=DialogueMode.AUTONOMOUS)
    record_mode(ds, phase="modeling", mode=DialogueMode.PROPOSITION_VALIDATION)
    audit = ds["_session"]["mode_audit"]
    assert len(audit) == 2
    assert audit[0]["phase"] == "data"
    assert audit[1]["mode"] == "proposition_validation"


def test_get_mode_returns_default_if_not_set():
    assert get_mode({}, phase="data") == DialogueMode.DEFAULT
