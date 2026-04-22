"""Tests US : gate `decision_required` sur builder.smoothing.

Quand n_non_monotone > 0 après âge 40, le tool doit retourner une clé
`decision_required` avec les options possibles, afin de laisser le
choix à l'utilisateur au lieu de forcer une re-exécution.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.builder.smoothing import _build_decision_required  # noqa: E402


def test_no_decision_required_when_monotone():
    """Si n_non_monotone == 0 ou None, pas de gate."""
    assert _build_decision_required(0, method="whittaker", lambda_used=200) is None
    assert _build_decision_required(None, method="whittaker", lambda_used=200) is None


def test_decision_required_when_violations():
    """Si n_non_monotone > 0, retourne un dict avec reason + 3 options."""
    dr = _build_decision_required(3, method="whittaker", lambda_used=200)

    assert dr is not None
    assert "3" in dr["reason"]  # le nombre de violations est mentionné
    assert "monotoni" in dr["reason"].lower()

    ids = [opt["id"] for opt in dr["options"]]
    assert set(ids) == {"increase_lambda", "change_method", "accept_with_note"}

    # Option "increase_lambda" doit suggérer la valeur doublée
    inc = next(o for o in dr["options"] if o["id"] == "increase_lambda")
    assert "400" in inc["label"]  # lambda actuel = 200 → suggestion = 400


def test_decision_required_change_method_suggestion_depends_on_current():
    """L'option 'change_method' propose Gompertz/spline selon la méthode actuelle."""
    dr = _build_decision_required(2, method="whittaker", lambda_used=200)
    cm = next(o for o in dr["options"] if o["id"] == "change_method")
    # Quand on est en whittaker, on propose les autres méthodes
    assert "gompertz" in cm["label"].lower() or "spline" in cm["label"].lower()


def test_run_output_includes_decision_required_when_violations(monkeypatch):
    """Intégration légère : si le smoother renvoie n_non_monotone > 0,
    le dict final contient decision_required."""
    import tools.builder.smoothing as smod

    class _FakeNb:
        def smooth_whittaker(self, qx_table, lambda_wh, d):
            import numpy as np
            return {
                "ages":                   np.array([30, 31, 32, 41, 42]),
                "qx_smoothed":            np.array([0.001, 0.0011, 0.0012, 0.004, 0.003]),
                "n_non_monotone_after_40": 1,
            }

    monkeypatch.setattr(smod, "load_nb", lambda _: _FakeNb())

    result = smod.run(
        {"qx_table": [{"age": 30, "q_x_brut": 0.001}]},
        {"method": "whittaker", "lambda_wh": 200},
    )

    assert "decision_required" in result
    assert result["decision_required"]["options"]


def test_run_output_no_decision_required_when_monotone(monkeypatch):
    import tools.builder.smoothing as smod

    class _FakeNb:
        def smooth_whittaker(self, qx_table, lambda_wh, d):
            import numpy as np
            return {
                "ages":                   np.array([30, 31, 32, 41, 42]),
                "qx_smoothed":            np.array([0.001, 0.0011, 0.0012, 0.0013, 0.0014]),
                "n_non_monotone_after_40": 0,
            }

    monkeypatch.setattr(smod, "load_nb", lambda _: _FakeNb())

    result = smod.run(
        {"qx_table": [{"age": 30, "q_x_brut": 0.001}]},
        {"method": "whittaker", "lambda_wh": 200},
    )

    assert "decision_required" not in result
