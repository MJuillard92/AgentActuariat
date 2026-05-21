"""
HOTFIX-pre-refacto-2026-05 — Bug 23 : _format_clone_notice produit un
message visible décrivant la création du clone normalisé et les
modifications appliquées.
"""
from __future__ import annotations

from agents.mortality.agents.master_node import _format_clone_notice


# ── _format_clone_notice ────────────────────────────────────────────────────

def test_clone_notice_lists_modifications() -> None:
    audit = {
        "column_mapping": {
            "date_entree": "CTREFFET", "date_sortie": "DATE_SORTIE",
            "date_naissance": "CLINAISS", "cause_sortie": "STATUT",
        },
        "value_mapping": {"cause_sortie": {"D": "deces", "V": "autre"}},
        "rows_in": 530345,
        "rows_out": 528900,
        "observation_end": "2023-12-31T00:00:00",
    }
    notice = _format_clone_notice(audit)
    assert "Base de données de travail créée" in notice
    assert "CTREFFET → date_entree" in notice
    assert "cause_sortie (D→deces, V→autre)" in notice
    assert "530 345 lignes en entrée" in notice
    assert "528 900 lignes" in notice
    assert "1 445 ligne(s) exclue(s)" in notice
    assert "2023-12-31" in notice


def test_clone_notice_minimal_audit() -> None:
    """Audit minimal (pas de value_mapping, pas d'obs_end) → pas de crash."""
    notice = _format_clone_notice({
        "column_mapping": {"date_entree": "CTREFFET"},
        "rows_in": 100, "rows_out": 100,
    })
    assert "Base de données de travail créée" in notice
    assert "CTREFFET → date_entree" in notice
    # pas de ligne exclue si rows_in == rows_out
    assert "exclue" not in notice


def test_clone_notice_empty_audit() -> None:
    """Audit vide → message générique sans crash."""
    notice = _format_clone_notice({})
    assert "Base de données de travail créée" in notice
