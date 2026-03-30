"""
smoothing_selector.py
Sélection automatique du modèle de lissage actuariel.

Orchestre les fonctions existantes de notebooks/04_smoothing.py et
notebooks/05_diagnostics.py — aucun calcul nouveau.

Logique de décision (trois résultats possibles) :
  1. "clear"   → un modèle est clairement meilleur (écart AIC ≥ seuil, pas de
                  violation de monotonicité) : l'agent peut continuer sans demander.
  2. "close"   → deux modèles sont statistiquement indiscernables (écart AIC <
                  seuil) : l'agent s'arrête et soumet le choix à l'actuaire.
  3. "escalate"→ le meilleur modèle présente des violations de monotonicité
                  inacceptables, ou aucun modèle n'a convergé : intervention
                  humaine requise avant de poursuivre.

Le choix des candidats dépend d'abord du diagnostic de crédibilité des données :
  - "non-parametric" → données suffisamment denses pour Whittaker/spline
  - "mixed"          → données mi-denses : on compare Whittaker vs paramétrique
  - "parametric"     → trop peu de données : on force Gompertz/Makeham

Usage typique (dans le kernel de l'agent) :
    result = smoothing_selector.auto_select_smoother(df_qx, df_exposure)
    print(result["status"])          # 'clear' | 'close' | 'escalate'
    print(result["best_method"])     # ex. 'whittaker'
    print(result["comparison_df"])   # tableau AIC/BIC/MSE/monotonicité
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des modules actuariels (compatibles kernel et import direct)
# ─────────────────────────────────────────────────────────────────────────────

def _load_module(name: str):
    """Retourne le module actuariel s'il est déjà dans sys.modules, sinon le charge.

    La vérification de sys.modules en premier évite de recharger un module déjà
    présent dans le kernel de l'agent, ce qui provoquerait une perte des variables
    calculées (df_exposure, etc.) attachées à ce module lors de la session.
    """
    if name in sys.modules:
        return sys.modules[name]
    notebooks_dir = Path(__file__).parent / "notebooks"
    # Correspondance alias → nom de fichier
    _file_map = {
        "smoothing":   "04_smoothing",
        "diagnostics": "05_diagnostics",
    }
    file_stem = _file_map.get(name, name)
    mod_path = notebooks_dir / f"{file_stem}.py"
    if not mod_path.exists():
        raise ImportError(f"Module '{name}' introuvable : {mod_path}")
    spec = importlib.util.spec_from_file_location(name, str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch : nom du candidat → appel de la bonne fonction
# ─────────────────────────────────────────────────────────────────────────────

def _run_smoother(name: str, qx_table: pd.DataFrame, p: dict) -> dict:
    """Appelle le smoother désigné avec les paramètres de p (sous-dict "smoothing").

    Centralise la correspondance nom → fonction pour que auto_select_smoother
    n'ait pas à connaître l'API de chaque smoother. Les paramètres sont toujours
    lus depuis PARAMS afin que l'actuaire puisse les régler sans modifier le code.
    """
    sm = _load_module("smoothing")
    sp = p.get("smoothing", {})

    if name == "whittaker":
        return sm.smooth_whittaker(
            qx_table,
            lambda_wh=sp.get("lambda_wh", 100),
            d=sp.get("d", 2),
        )
    if name == "gompertz":
        return sm.smooth_gompertz(
            qx_table,
            age_min_fit=sp.get("gompertz_age_min", 40),
            age_max_fit=sp.get("gompertz_age_max", 90),
        )
    if name == "makeham":
        return sm.smooth_makeham(
            qx_table,
            age_min_fit=sp.get("makeham_age_min", 30),
            age_max_fit=sp.get("makeham_age_max", 90),
        )
    if name == "spline":
        return sm.smooth_spline(qx_table)
    if name == "local_poly":
        return sm.smooth_local_polynomial(
            qx_table,
            bandwidth=sp.get("local_poly_bandwidth", 5),
            degree=sp.get("local_poly_degree", 2),
        )
    raise ValueError(f"Modèle de lissage inconnu : '{name}'")


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────────────

def auto_select_smoother(
    qx_table: pd.DataFrame,
    exposure_table: pd.DataFrame,
    params: dict | None = None,
) -> dict:
    """Sélectionne automatiquement le meilleur modèle de lissage.

    Teste les candidats appropriés selon la crédibilité des données,
    les compare sur critères objectifs (AIC Poisson, monotonicité),
    et retourne une décision structurée.

    Args:
        qx_table:       DataFrame avec colonnes age, E_x, D_x, q_x_brut.
                        Sortie typique de crude_rates.crude_rates_central().
        exposure_table: DataFrame avec colonnes age, E_x, D_x, q_x_brut.
                        Sortie de exposure.compute_exposure_by_age().
        params:         Dictionnaire de paramètres (structure de actuarial_params.PARAMS).
                        Si None, lit actuarial_params.PARAMS.

    Returns:
        dict avec les clés suivantes :

        - ``status`` : ``"clear"`` | ``"close"`` | ``"escalate"``
        - ``best_method`` : nom du modèle gagnant (str)
        - ``best_result`` : dict retourné par le smoother gagnant
        - ``comparison_df`` : DataFrame trié par AIC (colonnes : method, AIC_poisson, BIC_poisson, MSE_vs_crude, n_non_monotone, max_reversal)
        - ``reason`` : explication textuelle de la décision
        - ``competing_methods`` : liste des méthodes dans l'écart AIC < seuil
        - ``credibility_info`` : dict retourné par diagnose_credibility()
        - ``params_used`` : snapshot des paramètres utilisés
    """
    # ── Paramètres ──────────────────────────────────────────────────────────
    # On accepte un dict externe pour faciliter les tests unitaires, mais en
    # production les paramètres viennent toujours d'actuarial_params.PARAMS.
    if params is None:
        import sys as _sys, os as _os
        _nb = _os.path.dirname(_os.path.abspath(__file__))
        if _nb not in _sys.path:
            _sys.path.insert(0, _nb)
        from actuarial_params import PARAMS
        params = PARAMS

    ms = params.get("model_selection", {})
    aic_gap   = ms.get("aic_gap_threshold", 2.0)
    mono_max  = ms.get("mono_violations_max", 0)

    # ── Diagnostic de crédibilité ────────────────────────────────────────────
    # Ce diagnostic détermine quelle famille de modèles est pertinente.
    # Si beaucoup d'âges ont peu d'observations, un modèle paramétrique (Gompertz)
    # sera plus robuste qu'une méthode non-paramétrique (Whittaker).
    diag = _load_module("diagnostics")
    cred = diag.diagnose_credibility(
        exposure_table,
        threshold=params.get("credibility", {}).get("threshold_low", 10),
    )
    recommendation = cred.get("recommendation", "non-parametric")

    # ── Sélection des candidats ──────────────────────────────────────────────
    # La liste des candidats est configurable dans PARAMS["model_selection"]
    # pour permettre à l'actuaire d'exclure un modèle sans modifier ce fichier.
    cand_map = {
        "non-parametric": ms.get("candidates_non_parametric", ["whittaker", "spline"]),
        "mixed":          ms.get("candidates_mixed",          ["whittaker", "gompertz"]),
        "parametric":     ms.get("candidates_parametric",     ["gompertz", "makeham"]),
    }
    candidates = cand_map.get(recommendation, ["whittaker"])

    # ── Exécution des smoothers ──────────────────────────────────────────────
    # Les échecs individuels sont collectés mais n'arrêtent pas la boucle :
    # si au moins un modèle converge, on peut quand même produire une décision.
    smoothers_dict: dict = {}
    failed: list[str] = []
    for name in candidates:
        try:
            smoothers_dict[name] = _run_smoother(name, qx_table, params)
        except Exception as exc:
            failed.append(f"{name} : {exc}")

    if not smoothers_dict:
        return {
            "status": "escalate",
            "best_method": None,
            "best_result": None,
            "comparison_df": pd.DataFrame(),
            "reason": (
                f"Aucun modèle n'a convergé. Échecs : {'; '.join(failed)}. "
                "Vérifiez la qualité des données (E_x, D_x) et les bornes d'âge."
            ),
            "competing_methods": [],
            "credibility_info": cred,
            "params_used": params,
        }

    # ── Comparaison ──────────────────────────────────────────────────────────
    # compare_smoothers trie les modèles par AIC Poisson croissant.
    # L'AIC Poisson est préféré au MSE car il pénalise correctement les queues
    # de distribution asymétriques typiques des petits effectifs aux grands âges.
    comparison_df, _rec = diag.compare_smoothers(smoothers_dict, exposure_table)
    # comparison_df columns: method, AIC_poisson, BIC_poisson, MSE_vs_crude,
    #                         n_non_monotone, max_reversal  (trié par AIC_poisson asc)

    best_row  = comparison_df.iloc[0]
    best_name = best_row["method"]
    best_aic  = best_row["AIC_poisson"]
    best_mono = int(best_row.get("n_non_monotone", 0))

    # ── Décision ─────────────────────────────────────────────────────────────
    failed_note = f" (échecs ignorés : {', '.join(failed)})" if failed else ""

    # Cas 1 : monotonicité inacceptable pour le meilleur modèle → escalade
    # Une table de mortalité non monotone après 40 ans est actuariellement
    # inadmissible (elle impliquerait qu'il est "moins risqué" de vieillir).
    if best_mono > mono_max:
        reason = (
            f"Le meilleur modèle ('{best_name}', AIC={best_aic:.1f}) présente "
            f"{best_mono} violation(s) de monotonicité (seuil : {mono_max}). "
            "Options : augmenter lambda_wh, restreindre les bornes d'âge, "
            "ou forcer un modèle paramétrique (Gompertz/Makeham)."
            + failed_note
        )
        return {
            "status": "escalate",
            "best_method": best_name,
            "best_result": smoothers_dict.get(best_name),
            "comparison_df": comparison_df,
            "reason": reason,
            "competing_methods": [],
            "credibility_info": cred,
            "params_used": params,
        }

    # Cas 2 : deux modèles proches → demander confirmation
    # Un écart AIC < 2 est considéré comme «non significatif» en pratique statistique
    # (règle de Burnham & Anderson). Dans ce cas, la décision nécessite un jugement
    # métier que seul l'actuaire peut exercer (connaissance du portefeuille, prudence).
    competing: list[str] = []
    if len(comparison_df) > 1:
        second_aic = comparison_df.iloc[1]["AIC_poisson"]
        gap = abs(best_aic - second_aic)
        if gap < aic_gap:
            competing = list(
                comparison_df[
                    (comparison_df["AIC_poisson"] - best_aic).abs() < aic_gap
                ]["method"]
            )
            reason = (
                f"Deux modèles proches (écart AIC = {gap:.2f} < seuil {aic_gap}) : "
                f"{', '.join(competing)}. "
                f"Modèle favori : '{best_name}' (AIC={best_aic:.1f}). "
                "Veuillez confirmer le choix avant de continuer."
                + failed_note
            )
            return {
                "status": "close",
                "best_method": best_name,
                "best_result": smoothers_dict.get(best_name),
                "comparison_df": comparison_df,
                "reason": reason,
                "competing_methods": competing,
                "credibility_info": cred,
                "params_used": params,
            }

    # Cas 3 : gagnant clair
    second_info = (
        f" (2e : '{comparison_df.iloc[1]['method']}', AIC={comparison_df.iloc[1]['AIC_poisson']:.1f})"
        if len(comparison_df) > 1 else ""
    )
    reason = (
        f"Modèle sélectionné : '{best_name}' "
        f"(AIC={best_aic:.1f}, monotonicité OK){second_info}. "
        f"Crédibilité : {recommendation} "
        f"({cred.get('pct_low', 0):.1f} % âges à faible crédibilité)."
        + failed_note
    )
    return {
        "status": "clear",
        "best_method": best_name,
        "best_result": smoothers_dict.get(best_name),
        "comparison_df": comparison_df,
        "reason": reason,
        "competing_methods": [],
        "credibility_info": cred,
        "params_used": params,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaire : affichage console du résultat (pratique dans execute_python)
# ─────────────────────────────────────────────────────────────────────────────

def print_selection_result(result: dict) -> None:
    """Affiche un résumé lisible du résultat de auto_select_smoother()."""
    status_icon = {"clear": "✓", "close": "⚠", "escalate": "✗"}.get(result["status"], "?")
    print(f"\n{'─'*60}")
    print(f"{status_icon} Sélection du modèle : {result['status'].upper()}")
    print(f"   Meilleur modèle  : {result['best_method']}")
    print(f"   Décision         : {result['reason']}")
    df = result.get("comparison_df")
    if df is not None and not df.empty:
        print(f"\n   Comparatif AIC :")
        print(df.to_string(index=False))
    print(f"{'─'*60}\n")
