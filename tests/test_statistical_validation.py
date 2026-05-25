"""Tests — builder.statistical_validation

Couvre les 5 tests statistiques (χ², signes, runs, KS, Durbin-Watson),
les métriques de régularité, le tableau déciles avec taux lissés, et
les cas d'erreur.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.builder.statistical_validation import run  # noqa: E402


def _make_qx_table(ages, exposure=1000.0, qx_brut_fn=None):
    """Helper : construit une qx_table par âge avec E_x = exposure constant
    et q_x brut donné par qx_brut_fn(age) ; D_x = E_x × q_x_brut (entier)."""
    qx_brut_fn = qx_brut_fn or (lambda a: 0.001 * (a - 20))
    out = []
    for a in ages:
        q = qx_brut_fn(a)
        out.append({"age": int(a), "E_x": float(exposure),
                    "D_x": int(round(exposure * q)), "qx": float(q)})
    return out


def _make_smoothed_table(ages, qx_brut_fn=None, qx_lisse_fn=None):
    """Helper : smoothed_table avec q_x_brut et q_x_lisse paramétrables."""
    qx_brut_fn  = qx_brut_fn  or (lambda a: 0.001 * (a - 20))
    qx_lisse_fn = qx_lisse_fn or qx_brut_fn
    return [{"age": int(a), "q_x_brut": float(qx_brut_fn(a)),
             "q_x_lisse": float(qx_lisse_fn(a))} for a in ages]


# ── Cas d'erreur ─────────────────────────────────────────────────────────────

def test_missing_smoothed_returns_error():
    res = run({"qx_table": _make_qx_table(range(20, 90))}, {})
    assert "erreur" in res
    assert "smoothed_table" in res["erreur"]


def test_missing_qx_returns_error():
    res = run({"smoothed_table": _make_smoothed_table(range(20, 90))}, {})
    assert "erreur" in res
    assert "qx_table" in res["erreur"]


def test_too_few_common_ages_returns_error():
    qx  = _make_qx_table([20, 21])
    smo = _make_smoothed_table([20, 21])
    res = run({"qx_table": qx, "smoothed_table": smo}, {})
    assert "erreur" in res


# ── Lissage parfait (q_lisse = q_brut) → tous tests acceptés ─────────────────

def test_perfect_smoothing_all_tests_accepted():
    ages = list(range(20, 90))
    qx  = _make_qx_table(ages)
    smo = _make_smoothed_table(ages)  # q_brut = q_lisse
    res = run({"qx_table": qx, "smoothed_table": smo}, {})

    assert "erreur" not in res
    tests = {t["test"]: t for t in res["validation_tests_table"]}
    # χ² doit être ~0 → p très haute → accepted
    assert tests["chi_square"]["decision"] == "accepted"
    # Pas d'écart non nul → sign et runs sont "non calculable" (≥ 0 vs trop peu)
    # Mais ks_residuals et durbin_watson sur résidus nuls doivent passer
    assert tests["durbin_watson"]["decision"] == "accepted"
    # Summary global
    assert res["validation_summary"]["global_assessment"] in ("acceptable", "indéterminé")


# ── Lissage trop rigide (constante) → χ² rejeté ──────────────────────────────

def test_rigid_smoothing_rejects_chi_square():
    ages = list(range(20, 90))
    qx  = _make_qx_table(ages, qx_brut_fn=lambda a: 0.0005 * (a - 19) ** 1.3)
    # q_lisse plat → mauvaise adéquation aux taux qui croissent fortement
    smo = _make_smoothed_table(
        ages,
        qx_brut_fn=lambda a: 0.0005 * (a - 19) ** 1.3,
        qx_lisse_fn=lambda a: 0.05,  # constante
    )
    res = run({"qx_table": qx, "smoothed_table": smo}, {})

    tests = {t["test"]: t for t in res["validation_tests_table"]}
    assert tests["chi_square"]["decision"] == "rejected"
    # Verdict global devrait basculer en questionable ou inadéquat
    assert res["validation_summary"]["global_assessment"] in ("questionable", "inadéquat")


# ── Lissage qui suit le bruit (≈ q_brut + ε) → smoothness élevé ──────────────

def test_noisy_smoothing_has_higher_smoothness_metric():
    import random
    random.seed(42)
    ages = list(range(20, 90))
    qx_brut = lambda a: 0.0005 * (a - 19) ** 1.3
    # Smoothed bien lissé : pente régulière
    qx_clean  = lambda a: 0.0005 * (a - 19) ** 1.3
    # Smoothed qui « suit le bruit » : ajouter une perturbation à chaque âge
    qx_noisy  = lambda a: 0.0005 * (a - 19) ** 1.3 + (random.random() - 0.5) * 0.005

    qx = _make_qx_table(ages, qx_brut_fn=qx_brut)
    res_clean = run({"qx_table": qx,
                     "smoothed_table": _make_smoothed_table(ages, qx_brut, qx_clean)}, {})
    res_noisy = run({"qx_table": qx,
                     "smoothed_table": _make_smoothed_table(ages, qx_brut, qx_noisy)}, {})

    s_clean = res_clean["smoothness_metrics"]["sum_squared_d2"]
    s_noisy = res_noisy["smoothness_metrics"]["sum_squared_d2"]
    assert s_noisy > s_clean * 10, (
        f"smoothness noisy ({s_noisy}) devrait être >> clean ({s_clean})"
    )


# ── smoothed_deciles_table : D_predicted utilise q_lisse, pas q_brut ─────────

def test_smoothed_deciles_uses_q_lisse_for_prediction():
    ages = list(range(20, 90))
    qx  = _make_qx_table(ages, qx_brut_fn=lambda a: 0.001 * (a - 19))
    # q_lisse = 2 × q_brut → D_pred lissé devrait être ~2 × D_pred brut
    smo = _make_smoothed_table(
        ages,
        qx_brut_fn=lambda a: 0.001 * (a - 19),
        qx_lisse_fn=lambda a: 0.002 * (a - 19),
    )
    res = run({"qx_table": qx, "smoothed_table": smo}, {})

    deciles = res["smoothed_deciles_table"]
    assert deciles, "smoothed_deciles_table devrait contenir des buckets"

    # Comparer avec les déciles bruts : D_pred lissé doit être ~ 2× D_pred brut
    from tools.aggregation.exposure_deciles import _aggregate_by_exposure_deciles
    deciles_brut = _aggregate_by_exposure_deciles(qx_records=qx, smoothed_records=None)

    for d_smo, d_brut in zip(deciles, deciles_brut):
        ratio = d_smo["D_x_predicted"] / d_brut["D_x_predicted"]
        assert 1.8 < ratio < 2.2, (
            f"D_pred lissé/brut = {ratio:.2f} ; attendu ~2 sur bucket {d_smo['age_range']}"
        )


# ── Sélection partielle des tests via params ─────────────────────────────────

def test_subset_of_tests_via_params():
    ages = list(range(20, 90))
    qx  = _make_qx_table(ages)
    smo = _make_smoothed_table(ages)
    res = run({"qx_table": qx, "smoothed_table": smo},
              {"tests": ["chi_square", "smoothness"]})

    test_names = [t["test"] for t in res["validation_tests_table"]]
    assert test_names == ["chi_square"]  # seul test "statistique" demandé
    assert res["smoothness_metrics"]  # smoothness séparé, présent
