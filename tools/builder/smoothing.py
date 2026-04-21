"""
TOOL CONTRACT — builder.smoothing
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : builder.smoothing
domain        : mortality_experience
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Lisse les taux bruts de mortalité q_x pour produire une courbe régulière
et monotone. Quatre méthodes disponibles : Whittaker-Henderson (recommandé),
Gompertz, Makeham, et spline. Produit smoothed_table utilisée par validation
et benchmarking.

WHEN TO USE
-----------
Appeler après builder.crude_rates. Obligatoire avant builder.validation et
builder.benchmarking pour un rapport complet. Choisir la méthode selon la
crédibilité diagnostiquée (résultat de builder.diagnostics).

WHEN NOT TO USE
---------------
Ne pas appeler sans qx_table disponible dans le data_store.
Ne pas choisir Gompertz/Makeham si la plage d'âges est inférieure à 20 ans.

PREREQUISITES
-------------
required_tools:
  - builder.crude_rates → provides qx_table
required_data_store_keys:
  - qx_table

INPUTS
------
params:
  method:
    type    : string
    values  : whittaker | gompertz | makeham | spline
    default : whittaker
    note    : Whittaker-Henderson recommandé par défaut. Voir builder.diagnostics
              pour la recommandation automatique.
  lambda_wh:
    type    : float
    values  : 10–10000
    default : 100
    note    : Pénalité de lissage. Augmenter si n_non_monotone > 0 ou si
              pct_low_credibility > 30%. Valeurs typiques : 50–500.
  d:
    type    : int
    values  : 1 | 2 | 3
    default : 2
    note    : Ordre de différence Whittaker. 2 est standard (lissage quadratique).
  age_min_fit:
    type    : int
    values  : 0–120
    default : 40
    note    : Âge de début pour Gompertz/Makeham. Ignorer pour Whittaker/spline.
  age_max_fit:
    type    : int
    values  : 0–120
    default : 90
    note    : Âge de fin pour Gompertz/Makeham.

OUTPUTS
-------
data_store_keys_written:
  - smoothed_table : list[dict] — age, q_x_brut, q_x_lisse par âge
  - method         : str — méthode utilisée
return_payload:
  smoothed_table : list[dict]
  method         : str
  aic_poisson    : float — critère AIC (si disponible)
  bic_poisson    : float — critère BIC (si disponible)
  n_non_monotone : int — violations de monotonicité après âge 40

QUALITY GATES
-------------
DECISION_REQUIRED:
  - n_non_monotone > 0 après âge 40 → le tool ajoute une clé decision_required
    dans son retour avec 3 options (increase_lambda, change_method,
    accept_with_note). Le jugement revient à l'utilisateur — l'agent ne DOIT
    PAS enchaîner automatiquement sur builder.validation ni relancer
    builder.smoothing sans attendre la réponse humaine (cf. règle
    `decision_required` dans step1_planning.md + garde-fou builder_node).
NON-BLOCKING:
  - AIC/BIC disponibles → les logger pour comparaison si plusieurs méthodes
    sont testées.

ERROR HANDLING
--------------
error: "qx_table manquant. Appeler builder.crude_rates d'abord."
  → cause  : qx_table absent du data_store.
  → action : Appeler builder.crude_rates avant de relancer.
error: "Méthode inconnue : '...'"
  → cause  : Valeur de method incorrecte.
  → action : Utiliser uniquement : whittaker, gompertz, makeham, spline.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Si diagnostics indique recommendation="whittaker" et pct_low > 20%,
  commencer avec lambda_wh=200. Si le retour contient `decision_required`
  (n_non_monotone > 0), ne PAS relancer automatiquement : présenter les
  options à l'utilisateur en texte, attendre son choix, puis exécuter
  l'option choisie au tour suivant.
exemplar_query: >
  Quel lambda Whittaker choisir quand 35% des âges sont sous le seuil de crédibilité ?

CATALOGUE METADATA
------------------
display_name      : Lissage des taux de mortalité
short_description : Produit une courbe q_x régulière et monotone par méthode de lissage.
domain            : mortality_experience
capability_group  : table_construction
depends_on        : [builder.crude_rates]
required_by       : [builder.validation, builder.benchmarking, build_pdf.certification_report]
client_visible    : true
"""
from __future__ import annotations

import pandas as pd
from tools.builder._nb_loader import load_nb


def _build_decision_required(
    n_non_monotone: int | None,
    method:         str,
    lambda_used:    float | None,
) -> dict | None:
    """Construit le bloc `decision_required` si des violations de monotonie
    sont détectées après âge 40.

    Retourne None si `n_non_monotone` est 0 ou absent : le résultat est
    acceptable tel quel.
    """
    if not n_non_monotone:
        return None

    # Option 1 : augmenter le paramètre de lissage (Whittaker)
    if lambda_used:
        suggested_lambda = float(lambda_used) * 2
        inc_label = f"Augmenter le paramètre de lissage (doubler lambda → {suggested_lambda:g})"
    else:
        inc_label = "Augmenter le paramètre de lissage de la méthode actuelle"

    # Option 2 : changer de méthode, en suggérant les autres disponibles
    other_methods = [m for m in ("whittaker", "gompertz", "makeham", "spline") if m != method]
    cm_label = f"Changer de méthode (essayer : {', '.join(other_methods)})"

    return {
        "reason": (
            f"{n_non_monotone} violation(s) de monotonie détectée(s) après l'âge 40 "
            f"sur la méthode '{method}'. Une table non monotone n'est pas acceptable "
            "pour un rapport de certification."
        ),
        "options": [
            {"id": "increase_lambda",   "label": inc_label},
            {"id": "change_method",     "label": cm_label},
            {"id": "accept_with_note",  "label": "Accepter la table et mentionner explicitement les violations dans le rapport"},
        ],
    }


def run(data: dict | None, params: dict | None = None) -> dict:
    data = data or {}
    params = params or {}

    qx_records = data.get("qx_table")
    if not qx_records:
        return {"erreur": "qx_table manquant. Appeler builder.crude_rates d'abord."}

    qx_table = pd.DataFrame(qx_records)
    method = params.get("method", "whittaker")

    nb = load_nb("04_smoothing")

    try:
        if method == "whittaker":
            result = nb.smooth_whittaker(
                qx_table,
                lambda_wh=float(params.get("lambda_wh", 100)),
                d=int(params.get("d", 2)),
            )
        elif method == "gompertz":
            result = nb.smooth_gompertz(
                qx_table,
                age_min_fit=int(params.get("age_min_fit", 40)),
                age_max_fit=int(params.get("age_max_fit", 90)),
            )
        elif method == "makeham":
            result = nb.smooth_makeham(
                qx_table,
                age_min_fit=int(params.get("age_min_fit", 30)),
                age_max_fit=int(params.get("age_max_fit", 90)),
            )
        elif method == "spline":
            result = nb.smooth_spline(qx_table)
        else:
            return {"erreur": f"Méthode inconnue : '{method}'. Valeurs : whittaker, gompertz, makeham, spline"}
    except Exception as exc:
        return {"erreur": f"Erreur lissage ({method}) : {exc}"}

    # Les smoothers retournent {ages, qx_smoothed, ...} (arrays NumPy)
    # ou {smoothed_table, ...} (DataFrame) selon la méthode.
    import numpy as np

    lambda_used = params.get("lambda_wh") if method == "whittaker" else None

    if "ages" in result and "qx_smoothed" in result:
        # Format arrays → convertir en records [{age, q_x_lisse}, ...]
        ages_arr = np.asarray(result["ages"]).astype(int)
        qx_arr = np.asarray(result["qx_smoothed"]).astype(float)
        records = [
            {"age": int(a), "q_x_lisse": float(q) if not np.isnan(q) else None}
            for a, q in zip(ages_arr, qx_arr)
        ]
        n_non_monotone = result.get("n_non_monotone_after_40")
        out: dict = {
            "smoothed_table": records,
            "method":         method,
            "n_non_monotone": n_non_monotone,
        }
        dr = _build_decision_required(n_non_monotone, method, lambda_used)
        if dr:
            out["decision_required"] = dr
        return out

    smoothed_df = result.get("smoothed_table") or result.get("result")
    if smoothed_df is None:
        return {"erreur": f"Le smoother '{method}' n'a pas retourné de table lissée."}

    records = smoothed_df.where(pd.notnull(smoothed_df), None).to_dict(orient="records")
    n_non_monotone = result.get("n_non_monotone")
    out = {
        "smoothed_table": records,
        "method":         method,
        "aic_poisson":    result.get("aic_poisson"),
        "bic_poisson":    result.get("bic_poisson"),
        "n_non_monotone": n_non_monotone,
    }
    dr = _build_decision_required(n_non_monotone, method, lambda_used)
    if dr:
        out["decision_required"] = dr
    return out
