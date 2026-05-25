"""
TOOL CONTRACT — builder.statistical_validation
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : builder.statistical_validation
domain        : mortality_experience
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-05-24

DESCRIPTION
-----------
Évalue la qualité d'un lissage de taux de mortalité par une batterie de
tests statistiques d'adéquation (fidélité aux observations brutes), de
structure des résidus, et de régularité de la courbe lissée. Produit en
plus le tableau « décès observés vs prédits par déciles d'exposition »
recalculé avec les taux lissés (à comparer avec celui des taux bruts
produit par aggregation.exposure_deciles).

Le tool RAPPORTE — il ne ré-optimise pas le lissage. L'arbitrage est
laissé à l'utilisateur (cf. plan qualité-rapport 2026-05-24).

WHEN TO USE
-----------
Après builder.smoothing, en mode full_report. Appelé automatiquement par
le Builder via une branche déterministe dès que smoothed_table existe et
report_mode == "full_report" — pas de décision LLM.

WHEN NOT TO USE
---------------
Pas en mode raw_rates (rien à valider — le « lissé » est égal au brut).
Pas si smoothed_table ou qx_table absents.

PREREQUISITES
-------------
required_tools:
  - builder.crude_rates → provides qx_table
  - builder.smoothing   → provides smoothed_table
required_data_store_keys:
  - smoothed_table : list[dict] {age, q_x_brut, q_x_lisse}
  - qx_table       : list[dict] {age, E_x, D_x, qx}

INPUTS
------
params:
  alpha:
    type    : float
    default : 0.05
    note    : Niveau des tests d'hypothèse.
  tests:
    type    : list[str]
    default : tous
    values  : chi_square | sign | runs | ks_residuals | durbin_watson | smoothness
    note    : Sous-ensemble de tests à exécuter. Défaut = tous.

OUTPUTS
-------
data_store_keys_written:
  - smoothed_deciles_table : list[dict] — équivalent du qx_deciles_table
                              mais D_x_predicted = E_x × q_x_LISSÉ.
                              Colonnes : age_range, E_x_sum, proportion,
                              D_x_observed, D_x_predicted, ecart, ecart_pct,
                              ci_lower, ci_upper.
  - validation_tests_table : list[dict] — une ligne par test exécuté :
                              {test, statistic, p_value, df, decision,
                               interpretation}.
  - smoothness_metrics     : dict — {sum_squared_d2, sum_squared_d3,
                              mean_abs_d2}. Mesure la régularité de la
                              courbe lissée.
  - validation_summary     : dict — {n_tests_accepted, n_tests_rejected,
                              global_assessment} où global_assessment ∈
                              {"acceptable", "questionable", "inadéquat"}.
return_payload:
  identique aux clés ci-dessus.

QUALITY GATES
-------------
NON-BLOCKING:
  - Pas de gate decision_required. Si un test échoue à se calculer (ex.
    moins de 3 âges), il apparaît avec statistic=None et interpretation
    explicative. La batterie continue.

ERROR HANDLING
--------------
error: "smoothed_table manquant. Appeler builder.smoothing d'abord."
  → cause  : smoothed_table absent du data_store.
error: "qx_table manquant. Appeler builder.crude_rates d'abord."
  → cause  : qx_table absent du data_store.

CATALOGUE METADATA
------------------
display_name      : Validation statistique du lissage
short_description : Tests d'adéquation chi², signes, runs, KS, Durbin-Watson, smoothness + tableau décès obs/prédits lissés.
domain            : mortality_experience
capability_group  : table_construction
depends_on        : [builder.crude_rates, builder.smoothing]
required_by       : [build_pdf.assemble_sections]
client_visible    : true
"""
from __future__ import annotations

import math
from typing import Any


# ── Implémentations des tests ─────────────────────────────────────────────────

def _chi_square_test(observed: list[float], expected: list[float], alpha: float) -> dict:
    """χ² d'adéquation entre décès observés et décès prédits par les taux lissés.
    H₀ : le lissage reproduit les observations à un niveau acceptable.
    """
    from scipy.stats import chi2
    pairs = [(o, e) for o, e in zip(observed, expected) if e and e > 0]
    if len(pairs) < 2:
        return {
            "test": "chi_square", "statistic": None, "p_value": None,
            "df": None, "decision": "non calculable",
            "interpretation": "moins de 2 âges avec E_pred > 0",
        }
    stat = sum((o - e) ** 2 / e for o, e in pairs)
    df = len(pairs) - 1
    p = float(chi2.sf(stat, df))
    decision = "accepted" if p > alpha else "rejected"
    interp = (
        f"χ² = {stat:.2f} sur {df} degrés de liberté ; p = {p:.4f}. "
        + ("Le lissage est compatible avec les observations brutes."
           if decision == "accepted"
           else "Le lissage s'écarte significativement des observations — "
                "compromis fidélité/régularité trop déplacé vers la régularité.")
    )
    return {"test": "chi_square", "statistic": round(stat, 4), "p_value": round(p, 4),
            "df": df, "decision": decision, "interpretation": interp}


def _sign_test(q_brut: list[float], q_lisse: list[float], alpha: float) -> dict:
    """Test des signes : nb d'âges où q_lisse > q_brut suit Binomial(n, 0.5)
    sous H₀ (lissage non biaisé). Détecte un décalage systématique haut/bas.
    """
    from scipy.stats import binomtest
    diffs = [(qa - qb) for qb, qa in zip(q_brut, q_lisse) if qa != qb]
    n = len(diffs)
    if n < 5:
        return {
            "test": "sign", "statistic": None, "p_value": None,
            "df": None, "decision": "non calculable",
            "interpretation": f"trop peu d'écarts non nuls ({n})",
        }
    k_pos = sum(1 for d in diffs if d > 0)
    p = float(binomtest(k_pos, n, p=0.5).pvalue)
    decision = "accepted" if p > alpha else "rejected"
    direction = "supérieurs" if k_pos > n / 2 else "inférieurs"
    interp = (
        f"{k_pos}/{n} âges où taux lissé > taux brut ; p = {p:.4f}. "
        + ("Pas de biais systématique détecté."
           if decision == "accepted"
           else f"Biais systématique : les taux lissés sont majoritairement "
                f"{direction} aux taux bruts.")
    )
    return {"test": "sign", "statistic": k_pos, "p_value": round(p, 4),
            "df": n, "decision": decision, "interpretation": interp}


def _runs_test(q_brut: list[float], q_lisse: list[float], alpha: float) -> dict:
    """Wald-Wolfowitz sur la séquence de signes (q_lisse - q_brut). Trop peu
    de runs → lissage trop rigide ; trop de runs → lissage qui suit le bruit.
    """
    from scipy.stats import norm
    signs = [1 if (a - b) > 0 else (-1 if (a - b) < 0 else 0)
             for b, a in zip(q_brut, q_lisse)]
    signs = [s for s in signs if s != 0]
    n = len(signs)
    if n < 10:
        return {
            "test": "runs", "statistic": None, "p_value": None,
            "df": None, "decision": "non calculable",
            "interpretation": f"trop peu d'écarts non nuls ({n})",
        }
    n1 = sum(1 for s in signs if s > 0)
    n2 = n - n1
    if n1 == 0 or n2 == 0:
        return {"test": "runs", "statistic": None, "p_value": None,
                "df": None, "decision": "rejected",
                "interpretation": "lissage purement biaisé d'un côté (aucun changement de signe)"}
    runs = 1 + sum(1 for i in range(1, n) if signs[i] != signs[i - 1])
    mu = 1 + 2 * n1 * n2 / n
    var = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n * n * (n - 1))
    if var <= 0:
        return {"test": "runs", "statistic": runs, "p_value": None,
                "df": n, "decision": "non calculable",
                "interpretation": "variance nulle"}
    z = (runs - mu) / math.sqrt(var)
    p = float(2 * (1 - norm.cdf(abs(z))))
    decision = "accepted" if p > alpha else "rejected"
    interp = (
        f"{runs} runs observés vs {mu:.1f} attendus (Z = {z:.2f}, p = {p:.4f}). "
        + ("Le lissage présente une alternance normale des écarts de signe."
           if decision == "accepted"
           else ("Trop peu de runs : lissage probablement trop rigide."
                 if runs < mu else
                 "Trop de runs : lissage qui suit le bruit aléatoire."))
    )
    return {"test": "runs", "statistic": runs, "p_value": round(p, 4),
            "df": n, "decision": decision, "interpretation": interp}


def _ks_residuals_test(observed: list[float], expected: list[float], alpha: float) -> dict:
    """Kolmogorov-Smirnov : les résidus standardisés (O - E)/√E suivent-ils N(0,1) ?"""
    from scipy.stats import kstest
    res = [(o - e) / math.sqrt(e) for o, e in zip(observed, expected) if e and e > 0]
    if len(res) < 5:
        return {"test": "ks_residuals", "statistic": None, "p_value": None,
                "df": None, "decision": "non calculable",
                "interpretation": f"trop peu de résidus exploitables ({len(res)})"}
    # Cas dégénéré : résidus tous nuls (lissage = observations exactement).
    # kstest rejetterait N(0,1) à tort — sémantiquement c'est un fit parfait.
    if max(abs(r) for r in res) < 1e-12:
        return {"test": "ks_residuals", "statistic": 0.0, "p_value": 1.0,
                "df": len(res), "decision": "accepted",
                "interpretation": "résidus tous nuls — fit parfait."}
    stat, p = kstest(res, "norm")
    p = float(p)
    decision = "accepted" if p > alpha else "rejected"
    interp = (
        f"KS = {stat:.4f}, p = {p:.4f}. "
        + ("Les résidus standardisés sont compatibles avec N(0,1)."
           if decision == "accepted"
           else "Les résidus s'écartent significativement de la loi normale — "
                "le modèle de bruit Poisson est probablement mal spécifié à certains âges.")
    )
    return {"test": "ks_residuals", "statistic": round(float(stat), 4),
            "p_value": round(p, 4), "df": len(res),
            "decision": decision, "interpretation": interp}


def _durbin_watson_test(observed: list[float], expected: list[float], alpha: float) -> dict:
    """Durbin-Watson sur les résidus ordonnés par âge. DW ∈ [0, 4] ;
    DW ≈ 2 → pas d'autocorrélation ; < 2 → positive ; > 2 → négative.
    On considère acceptable la zone [1.5, 2.5] (heuristique standard).
    """
    res = [(o - e) for o, e in zip(observed, expected)]
    if len(res) < 5:
        return {"test": "durbin_watson", "statistic": None, "p_value": None,
                "df": None, "decision": "non calculable",
                "interpretation": f"trop peu de résidus exploitables ({len(res)})"}
    num = sum((res[i] - res[i - 1]) ** 2 for i in range(1, len(res)))
    den = sum(r ** 2 for r in res)
    if den == 0:
        return {"test": "durbin_watson", "statistic": None, "p_value": None,
                "df": None, "decision": "accepted",
                "interpretation": "résidus tous nuls — lissage parfait"}
    dw = num / den
    in_band = 1.5 <= dw <= 2.5
    decision = "accepted" if in_band else "rejected"
    if dw < 1.5:
        sense = "positive (résidus consécutifs corrélés positivement)"
    elif dw > 2.5:
        sense = "négative (résidus alternés)"
    else:
        sense = "non détectée"
    interp = (
        f"DW = {dw:.3f} (zone d'acceptation [1.5 ; 2.5]). Autocorrélation {sense}."
        + ("" if decision == "accepted"
           else " — un défaut de spécification du lissage est probable.")
    )
    return {"test": "durbin_watson", "statistic": round(dw, 4),
            "p_value": None, "df": len(res),
            "decision": decision, "interpretation": interp}


def _smoothness_metrics(q_lisse: list[float]) -> dict:
    """Différences finies d'ordre 2 et 3 de la courbe lissée. Plus c'est
    petit, plus la courbe est régulière. Permet de juger le compromis
    fidélité/régularité conjointement avec le χ².
    """
    n = len(q_lisse)
    d2 = [q_lisse[i + 1] - 2 * q_lisse[i] + q_lisse[i - 1] for i in range(1, n - 1)]
    d3 = [q_lisse[i + 2] - 3 * q_lisse[i + 1] + 3 * q_lisse[i] - q_lisse[i - 1]
          for i in range(1, n - 2)] if n >= 4 else []
    return {
        "sum_squared_d2": round(sum(x * x for x in d2), 8) if d2 else None,
        "sum_squared_d3": round(sum(x * x for x in d3), 8) if d3 else None,
        "mean_abs_d2":    round(sum(abs(x) for x in d2) / len(d2), 8) if d2 else None,
    }


def _global_assessment(test_results: list[dict]) -> str:
    """Synthèse : combien de tests acceptés/rejetés, et verdict global."""
    countable = [t for t in test_results if t["decision"] in ("accepted", "rejected")]
    if not countable:
        return "indéterminé"
    n_acc = sum(1 for t in countable if t["decision"] == "accepted")
    ratio = n_acc / len(countable)
    if ratio >= 0.8:
        return "acceptable"
    if ratio >= 0.5:
        return "questionable"
    return "inadéquat"


# ── Point d'entrée ────────────────────────────────────────────────────────────

_DEFAULT_TESTS = ("chi_square", "sign", "runs", "ks_residuals", "durbin_watson", "smoothness")


def run(data: dict | None, params: dict | None = None) -> dict:
    data = data or {}
    params = params or {}

    smoothed_records = data.get("smoothed_table")
    if not smoothed_records:
        return {"erreur": "smoothed_table manquant. Appeler builder.smoothing d'abord."}
    qx_records = data.get("qx_table")
    if not qx_records:
        return {"erreur": "qx_table manquant. Appeler builder.crude_rates d'abord."}

    alpha = float(params.get("alpha", 0.05))
    requested = tuple(params.get("tests") or _DEFAULT_TESTS)

    # Indexer par âge pour aligner brut/lissé et observed/exposure
    qx_by_age = {int(r["age"]): r for r in qx_records if r.get("age") is not None}
    smo_by_age = {int(r["age"]): r for r in smoothed_records if r.get("age") is not None}
    common_ages = sorted(set(qx_by_age) & set(smo_by_age))

    if len(common_ages) < 3:
        return {"erreur": "Moins de 3 âges communs entre qx_table et smoothed_table."}

    # Séries alignées sur common_ages
    observed = [float(qx_by_age[a].get("D_x") or 0) for a in common_ages]
    exposure = [float(qx_by_age[a].get("E_x") or 0) for a in common_ages]
    q_lisse  = [float(smo_by_age[a].get("q_x_lisse") or 0) for a in common_ages]
    q_brut   = [float(smo_by_age[a].get("q_x_brut")
                      or qx_by_age[a].get("qx") or 0) for a in common_ages]
    expected_d = [e * q for e, q in zip(exposure, q_lisse)]

    # Batterie de tests
    tests_table: list[dict] = []
    if "chi_square" in requested:
        tests_table.append(_chi_square_test(observed, expected_d, alpha))
    if "sign" in requested:
        tests_table.append(_sign_test(q_brut, q_lisse, alpha))
    if "runs" in requested:
        tests_table.append(_runs_test(q_brut, q_lisse, alpha))
    if "ks_residuals" in requested:
        tests_table.append(_ks_residuals_test(observed, expected_d, alpha))
    if "durbin_watson" in requested:
        tests_table.append(_durbin_watson_test(observed, expected_d, alpha))

    smoothness = _smoothness_metrics(q_lisse) if "smoothness" in requested else {}

    # Tableau déciles avec taux lissés (réutilise la logique existante)
    from tools.aggregation.exposure_deciles import _aggregate_by_exposure_deciles
    smoothed_deciles = _aggregate_by_exposure_deciles(
        qx_records=qx_records,
        smoothed_records=smoothed_records,
        n_buckets=int(params.get("n_buckets", 10)),
    )

    n_acc = sum(1 for t in tests_table if t["decision"] == "accepted")
    n_rej = sum(1 for t in tests_table if t["decision"] == "rejected")
    summary = {
        "n_tests_accepted":  n_acc,
        "n_tests_rejected":  n_rej,
        "alpha":             alpha,
        "global_assessment": _global_assessment(tests_table),
    }

    return {
        "smoothed_deciles_table": smoothed_deciles,
        "validation_tests_table": tests_table,
        "smoothness_metrics":     smoothness,
        "validation_summary":     summary,
    }
