"""
TOOL CONTRACT — builder.precedent_comparison
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : builder.precedent_comparison
domain        : mortalite
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-01

DESCRIPTION
-----------
Compare la table de mortalité lissée courante avec une table
précédente (table d'expérience antérieure ou table réglementaire).
Calcule les ratios qx_courant / qx_precedent par âge, la dérive
moyenne et les âges avec des écarts significatifs. Permet de
mesurer l'évolution de la mortalité entre deux périodes
d'observation.

WHEN TO USE
-----------
Quand le client dispose d'une table d'expérience précédente et
souhaite mesurer l'évolution de la mortalité dans le temps.
Typiquement dans §3 du rapport si deux périodes sont comparables.
Optionnel : si la table précédente est absente, retourner warning
sans bloquer.

WHEN NOT TO USE
---------------
Ne pas utiliser pour comparer avec une table réglementaire de
référence (TH00-02, TPRV...) — utiliser builder.benchmarking.
Ne pas appeler si smoothed_table est absente.

PREREQUISITES
-------------
required_tools: [builder.smoothing]
required_data_store_keys: [smoothed_table]

INPUTS
------
params:
  precedent_table:
    type    : list[dict]
    note    : >
      Table précédente sous forme de liste de dicts {age, qx}.
      Si absente, tente de lire data["precedent_table"] du data_store.
      Colonnes acceptées pour qx : "qx", "qx_lisse", "qx_smooth".
  age_min:
    type    : int
    default : 20
    note    : Âge minimum pour le calcul des ratios.
  age_max:
    type    : int
    default : 90
    note    : Âge maximum pour le calcul des ratios.
  seuil_derive:
    type    : float
    default : 0.05
    note    : >
      Seuil d'écart relatif |ratio - 1| pour signaler un âge
      comme ayant une dérive significative. Défaut : 5%.

OUTPUTS
-------
data_store_keys_written: [precedent_comparison]
return_payload:
  comparison_table  : list[dict] {age, qx_courant, qx_precedent,
                                  ratio, derive_pct}
  drift_global      : float   — dérive relative moyenne en %
                                ((qx_courant / qx_precedent) - 1)
  ages_derive_forte : list[int] — âges avec |ratio-1| > seuil_derive
  n_ages_comparables: int
  warning           : string | null

QUALITY GATES
-------------
BLOCKING:
  - smoothed_table absente → erreur "smoothed_table manquante"
NON-BLOCKING:
  - precedent_table absente → warning + comparison_table vide, continuer
  - plage d'âges communes vide → warning + comparison_table vide
  - ratio infini ou NaN pour un âge → ignorer cet âge, signaler dans warning

ERROR HANDLING
--------------
error: "smoothed_table manquante"
  → cause  : pipeline incomplet
  → action : retourner erreur, demander d'exécuter builder.smoothing d'abord

AGENT GUIDANCE
--------------
reasoning_hint: >
  Appeler si et seulement si une table précédente est disponible.
  Le champ drift_global permet de conclure sur la tendance temporelle
  dans §3 du rapport : valeur négative = amélioration de la mortalité.
  Les âges_derive_forte permettent d'identifier des ruptures locales
  (ex : changement de comportement sur un segment d'âge).
exemplar_query: >
  "évolution tendance mortalité entre deux périodes d'observation"

CATALOGUE METADATA
------------------
display_name      : Comparaison table précédente
short_description : Compare la table courante avec une table de mortalité précédente.
domain            : mortalite
capability_group  : builder
depends_on        : [builder.smoothing]
required_by       : []
client_visible    : false
"""
from __future__ import annotations

import math
from typing import Any


def run(data: dict, params: dict | None = None) -> dict:
    params = params or {}

    # Vérifier smoothed_table
    smoothed = data.get("smoothed_table")
    if not smoothed:
        return {"erreur": "smoothed_table manquante dans le data_store. Exécuter builder.smoothing d'abord."}

    # Récupérer la table précédente
    prec_param = params.get("precedent_table")
    prec_store = data.get("precedent_table")
    precedent  = prec_param or prec_store or []

    age_min      = int(params.get("age_min", 20))
    age_max      = int(params.get("age_max", 90))
    seuil_derive = float(params.get("seuil_derive", 0.05))

    if not precedent:
        return {
            "comparison_table":   [],
            "drift_global":       None,
            "ages_derive_forte":  [],
            "n_ages_comparables": 0,
            "warning": (
                "Table précédente absente. "
                "Fournir precedent_table dans params ou dans le data_store."
            ),
        }

    # Construire dicts age → qx pour chaque table
    def _to_qx_dict(table: list[dict]) -> dict[int, float]:
        out = {}
        for row in table:
            age = row.get("age")
            if age is None:
                continue
            qx = (
                row.get("qx_lisse")
                or row.get("qx_smooth")
                or row.get("qx")
            )
            if qx is None:
                continue
            try:
                out[int(float(age))] = float(qx)
            except (ValueError, TypeError):
                continue
        return out

    curr_map = _to_qx_dict(smoothed)
    prec_map = _to_qx_dict(precedent)

    # Plage d'âges communes dans [age_min, age_max]
    ages_communs = sorted(
        a for a in curr_map
        if a in prec_map and age_min <= a <= age_max
    )

    if not ages_communs:
        return {
            "comparison_table":   [],
            "drift_global":       None,
            "ages_derive_forte":  [],
            "n_ages_comparables": 0,
            "warning": (
                f"Aucun âge commun dans la plage [{age_min}, {age_max}] "
                "entre la table courante et la table précédente."
            ),
        }

    comparison_table: list[dict] = []
    ratios_valides: list[float] = []
    ages_derive_forte: list[int] = []
    warnings_list: list[str] = []

    for age in ages_communs:
        qx_c = curr_map[age]
        qx_p = prec_map[age]

        if qx_p == 0 or math.isnan(qx_p) or math.isnan(qx_c):
            warnings_list.append(f"âge {age} ignoré (qx invalide).")
            continue

        ratio    = qx_c / qx_p
        derive   = (ratio - 1.0) * 100.0  # en %

        comparison_table.append({
            "age":           age,
            "qx_courant":    round(qx_c, 6),
            "qx_precedent":  round(qx_p, 6),
            "ratio":         round(ratio, 4),
            "derive_pct":    round(derive, 2),
        })
        ratios_valides.append(ratio)

        if abs(ratio - 1.0) > seuil_derive:
            ages_derive_forte.append(age)

    # Dérive globale = moyenne des ratios - 1
    drift_global = None
    if ratios_valides:
        drift_global = round((sum(ratios_valides) / len(ratios_valides) - 1.0) * 100.0, 2)

    warning = "; ".join(warnings_list) if warnings_list else None

    # Écrire dans le data_store
    result = {
        "comparison_table":   comparison_table,
        "drift_global":       drift_global,
        "ages_derive_forte":  ages_derive_forte,
        "n_ages_comparables": len(comparison_table),
        "warning":            warning,
    }

    data["precedent_comparison"] = result
    return result
