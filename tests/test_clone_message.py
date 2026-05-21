"""
Plan « bouton Valider le mapping » — message de confirmation de création
du clone (canvas_app._format_clone_message).
"""
from __future__ import annotations

import canvas_app


def test_clone_message_lists_modifications() -> None:
    msg = canvas_app._format_clone_message({
        "column_mapping": {"date_entree": "CTREFFET", "date_sortie": "DATE_SORTIE"},
        "value_mapping":  {"cause_sortie": {"D": "deces", "V": "autre"}},
        "rows_in":  530345,
        "rows_out": 528900,
    })
    assert "Base de travail créée" in msg
    assert "CTREFFET → date_entree" in msg
    assert "cause_sortie (D→deces, V→autre)" in msg
    assert "530 345 lignes en entrée" in msg
    assert "528 900 lignes" in msg
    assert "lancer vos calculs" in msg


def test_clone_message_minimal_audit() -> None:
    """Audit minimal (pas de value_mapping) → pas de crash."""
    msg = canvas_app._format_clone_message({
        "column_mapping": {"date_entree": "CTREFFET"},
        "rows_in": 100, "rows_out": 100,
    })
    assert "Base de travail créée" in msg
    assert "CTREFFET → date_entree" in msg


def test_clone_message_empty_audit() -> None:
    """Audit vide → message générique sans crash."""
    msg = canvas_app._format_clone_message({})
    assert "Base de travail créée" in msg
