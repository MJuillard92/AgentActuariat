"""
TOOL CONTRACT — aggregation.exposure_deciles
════════════════════════════════════════════

CATALOGUE METADATA
------------------
name              : aggregation.exposure_deciles
display_name      : Agrégation des taux par déciles d'exposition
short_description : Agrège qx_table en buckets d'âges contigus représentant ~10% d'exposition. Produit le tableau type "Tableau 7" du rapport référence (D_obs, D_pred, écart, IC 95%).
domain            : mortality_experience
version           : 1.0.0
author            : Marc Juillard
last_updated      : 2026-05-11
capability_group  : aggregation
depends_on        : [builder.crude_rates]
required_by       : [build_pdf.assemble_sections]
client_visible    : false

DESCRIPTION
-----------
Agrège les taux bruts/lissés par âge en buckets d'âges contigus dont
chacun représente environ 1/n_buckets de l'exposition totale (par défaut
10 buckets → ~10% chacun). Pour chaque bucket :
  - intervalle d'âges (age_range)
  - exposition cumulée (E_x_sum)
  - proportion de l'exposition totale (proportion)
  - décès observés (D_x_observed)
  - décès prédits (D_x_predicted = sum age in bucket : E_x_age × q_x_age)
       où q_x_age vient de smoothed_table si disponible, sinon qx_table
  - écart (observed - predicted)
  - écart relatif (% par rapport au prédit)
  - bornes IC 95% des décès prédits sous l'hypothèse de Poisson :
       ci_lower = D_pred - 1.96 × sqrt(D_pred)
       ci_upper = D_pred + 1.96 × sqrt(D_pred)

Reproduit la structure du Tableau 7 du rapport référence AF8796-TD3 p.9.

WHEN TO USE
-----------
À appeler après builder.crude_rates (et optionnellement builder.smoothing).
Le résultat alimente la section table_construction (mode raw_rates) ou
validation (mode full_report) du YAML.

WHEN NOT TO USE
---------------
Ne pas appeler sans qx_table. Si n_buckets > nombre d'âges distincts,
l'agrégation est triviale (1 âge par bucket).

PREREQUISITES
-------------
required_tools:
  - builder.crude_rates → provides qx_table
required_data_store_keys:
  - qx_table

INPUTS
------
params:
  qx_table:
    type    : list[dict]
    note    : Records {age, E_x, D_x, qx} produits par builder.crude_rates.
  smoothed_table:
    type    : list[dict]
    note    : Records {age, q_x_brut, q_x_lisse} produits par builder.smoothing.
              Si absent, on utilise qx (taux brut) pour calculer D_predicted.
  n_buckets:
    type    : int
    default : 10
    note    : Nombre de buckets cibles. Standard actuariel : 10 (déciles).

OUTPUTS
-------
data_store_keys_written:
  - qx_deciles_table : list[dict] — records {age_range, E_x_sum, proportion, D_x_observed, D_x_predicted, ecart, ecart_pct, ci_lower, ci_upper}
  - n_buckets        : int — nombre effectif de buckets produits
return_payload:
  qx_deciles_table : list[dict]
  n_buckets        : int — nombre effectif de buckets produits

QUALITY GATES
-------------
BLOCKING:
  - qx_table absent → retourne erreur.
NON-BLOCKING:
  - Si certains buckets ont D_predicted=0, ci_lower=ci_upper=0 — à mentionner.

ERROR HANDLING
--------------
error: "qx_table manquant. Appeler builder.crude_rates d'abord."
  → cause  : qx_table absent du data_store.
  → action : Appeler builder.crude_rates avec exposure_table préalable.
"""
from __future__ import annotations

import math
from typing import Any


def _aggregate_by_exposure_deciles(
    qx_records: list[dict],
    smoothed_records: list[dict] | None = None,
    n_buckets: int = 10,
) -> list[dict]:
    """Algorithme :
      1. Trier par âge croissant.
      2. Cumul d'exposition par âge.
      3. Pour k=1..n_buckets, seuil_k = k * total_exposure / n_buckets.
      4. Buckets contigus : on accumule les âges jusqu'à atteindre chaque seuil.
      5. Pour chaque bucket : agréger E_sum, D_obs ; calculer D_pred par
         sommation pondérée des q_x (lissés si dispo, sinon bruts).
      6. IC 95% Poisson sur D_pred.
    """
    if not qx_records:
        return []

    # Trier par âge
    rows = sorted(
        (r for r in qx_records if r.get("age") is not None),
        key=lambda r: int(r["age"]),
    )
    if not rows:
        return []

    # Index q_x_lisse par âge (depuis smoothed_table si fourni)
    q_lisse_by_age: dict[int, float] = {}
    if smoothed_records:
        for r in smoothed_records:
            if r.get("age") is None:
                continue
            qv = r.get("q_x_lisse")
            if qv is not None:
                q_lisse_by_age[int(r["age"])] = float(qv)

    # Exposition totale + seuils cumulés
    total_E = sum(float(r.get("E_x", 0) or 0) for r in rows)
    if total_E <= 0:
        return []

    # Génère les seuils 10%, 20%, ..., 100% de l'exposition cumulée
    thresholds = [k * total_E / n_buckets for k in range(1, n_buckets + 1)]

    buckets: list[dict[str, Any]] = []
    current_ages: list[int] = []
    current_E   = 0.0
    current_D   = 0
    current_Dp  = 0.0
    cum_E       = 0.0
    threshold_idx = 0

    def _close(bucket_data) -> None:
        if not bucket_data["ages"]:
            return
        E = bucket_data["E"]
        Dp = bucket_data["D_pred"]
        D = bucket_data["D"]
        age_lo = min(bucket_data["ages"])
        age_hi = max(bucket_data["ages"])
        proportion = E / total_E if total_E > 0 else 0.0
        ecart = D - Dp
        ecart_pct = (ecart / Dp * 100) if Dp > 0 else 0.0
        # IC 95% Poisson sur les décès prédits
        se = math.sqrt(max(Dp, 0.0))
        ci_lower = max(Dp - 1.96 * se, 0.0)
        ci_upper = Dp + 1.96 * se
        buckets.append({
            "age_range":      f"{age_lo}-{age_hi}" if age_lo != age_hi else str(age_lo),
            "E_x_sum":        round(E, 2),
            "proportion":     round(proportion * 100, 1),   # en pourcentage
            "D_x_observed":   int(round(D)),
            "D_x_predicted":  round(Dp, 2),
            "ecart":          round(ecart, 2),
            "ecart_pct":      round(ecart_pct, 1),
            "ci_lower":       round(ci_lower, 2),
            "ci_upper":       round(ci_upper, 2),
        })

    bd = {"ages": [], "E": 0.0, "D": 0, "D_pred": 0.0}
    for r in rows:
        age = int(r["age"])
        E_x = float(r.get("E_x", 0) or 0)
        D_x = int(r.get("D_x", 0) or 0)
        # q_x prédit : lissé si dispo, sinon brut
        q_pred = q_lisse_by_age.get(age)
        if q_pred is None:
            q_pred = float(r.get("qx") or 0.0)
        D_pred_age = E_x * q_pred

        bd["ages"].append(age)
        bd["E"]      += E_x
        bd["D"]      += D_x
        bd["D_pred"] += D_pred_age
        cum_E        += E_x

        # On clôt le bucket dès qu'on dépasse le seuil courant (sauf le dernier)
        while threshold_idx < n_buckets - 1 and cum_E >= thresholds[threshold_idx]:
            _close(bd)
            bd = {"ages": [], "E": 0.0, "D": 0, "D_pred": 0.0}
            threshold_idx += 1

    # Fermer le dernier bucket (contient tout le reste)
    _close(bd)
    return buckets


def run(data: dict | None = None, params: dict | None = None) -> dict:
    data = data or {}
    params = params or {}

    qx_records = data.get("qx_table") or params.get("qx_table")
    if not qx_records:
        return {"erreur": "qx_table manquant. Appeler builder.crude_rates d'abord."}

    smoothed_records = data.get("smoothed_table") or params.get("smoothed_table")
    n_buckets = int(params.get("n_buckets", 10))

    deciles = _aggregate_by_exposure_deciles(qx_records, smoothed_records, n_buckets)

    return {
        "qx_deciles_table": deciles,
        "n_buckets":        len(deciles),
    }
