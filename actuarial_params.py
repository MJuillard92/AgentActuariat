"""
actuarial_params.py
Paramètres métier modifiables par un actuaire sans toucher au code.

IMPORTANT : Redémarrer l'application après toute modification.
            Les valeurs sont chargées une seule fois au démarrage.

Plages indicatives mentionnées en commentaire pour aider au calibrage.
"""

PARAMS: dict = {

    # ── Période d'observation ──────────────────────────────────────────────
    "observation": {
        "date_fin": "2023-12-31",   # Format YYYY-MM-DD — fin de la période d'étude
    },

    # ── Plages d'âges ─────────────────────────────────────────────────────
    "ages": {
        "min": 20,   # Âge minimum retenu dans l'exposition et les calculs (entier)
        "max": 90,   # Âge maximum retenu (entier, typiquement 90–100)
    },

    # ── Lissage ───────────────────────────────────────────────────────────
    "smoothing": {
        # Whittaker-Henderson
        "lambda_wh": 100,   # Pénalité de lissage (10 = peu lissé, 500 = très lissé)
        "d": 2,             # Ordre de différence : 2 = pénalise la courbure, 3 = dérivée 3e

        # Gompertz — log(μ_x) = a + b·x
        "gompertz_age_min": 40,   # Âge à partir duquel ajuster le modèle (inclus)
        "gompertz_age_max": 90,   # Âge jusqu'auquel ajuster (inclus)

        # Makeham — μ_x = A + B·exp(c·x)
        "makeham_age_min": 30,    # Capture la bosse accidentelle des jeunes âges
        "makeham_age_max": 90,

        # Polynôme local (LOESS)
        "local_poly_bandwidth": 5,   # Demi-fenêtre en nombre d'âges (3–7)
        "local_poly_degree": 2,      # Degré du polynôme local (1 = linéaire, 2 = quadratique)
    },

    # ── Crédibilité ───────────────────────────────────────────────────────
    "credibility": {
        "threshold_low": 10,    # E_x < seuil → âge à faible crédibilité (5–15 typique)
        "pct_parametric": 30,   # Si ≥ pct_parametric % des âges en faible créd → forcer paramétrique
        "pct_mixed": 10,        # Si ≥ pct_mixed % → méthode mixte (sinon non-paramétrique)
    },

    # ── SMR — Standardized Mortality Ratio ────────────────────────────────
    "smr": {
        "lower": 0.90,   # SMR < lower → sélection favorable (portefeuille plus sain)
        "upper": 1.10,   # SMR > upper → surmortalité par rapport à la table de référence
    },

    # ── Validation statistique ─────────────────────────────────────────────
    "validation": {
        "alpha": 0.05,                   # Niveau de confiance = 1 − alpha (IC 95 % si 0.05)
        "prudence_margin_min": 0.10,     # Marge de prudence ≥ 10 % = table conservatrice
        "chi_square_min_expected": 1.0,  # Nombre attendu minimum pour inclure une cellule dans le χ²
    },

    # ── Diagnostics ───────────────────────────────────────────────────────
    "diagnostics": {
        "age_start_monotonicity": 40,   # Début du contrôle de monotonicité des qx
                                         # (avant 40 ans, non-monotonicité acceptable : bosse accidentelle)
    },

    # ── Agent ─────────────────────────────────────────────────────────────
    "agent": {
        "max_iterations": 40,   # Nombre maximum d'appels d'outils par analyse
                                 # (garde-fou anti-boucle infinie ; 25–50 typique)
    },

    # ── Sélection automatique du modèle de lissage ────────────────────────
    "model_selection": {
        # Écart d'AIC Poisson en dessous duquel deux modèles sont considérés «proches»
        # → l'agent demande confirmation humaine avant de trancher
        "aic_gap_threshold": 2.0,

        # Nombre maximal de violations de monotonicité (qx[x] > qx[x+1]) acceptées
        # pour le meilleur modèle. Si dépassé → escalade.
        "mono_violations_max": 0,

        # Modèles candidats selon le résultat du diagnostic de crédibilité.
        # Valeurs possibles : "whittaker", "gompertz", "makeham", "spline", "local_poly"
        "candidates_non_parametric": ["whittaker", "spline"],
        "candidates_mixed":          ["whittaker", "gompertz"],
        "candidates_parametric":     ["gompertz", "makeham"],
    },
}
