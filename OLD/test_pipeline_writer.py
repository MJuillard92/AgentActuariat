"""
test_pipeline_writer.py
━━━━━━━━━━━━━━━━━━━━━━━
Test intégration complet du pipeline WriterAgent.

Lance les 6 étapes avec un data_store réaliste (valeurs synthétiques
cohérentes avec un portefeuille prévoyance collective 530k lignes).

Usage :
    python test_pipeline_writer.py
"""
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TEST")

# ── Données synthétiques réalistes ────────────────────────────────────────────

AGES = list(range(30, 86))   # 56 âges

def _by_age(base_val, noise=0.02):
    import math, random
    random.seed(42)
    return {a: round(base_val * math.exp(0.08 * (a - 50)) * (1 + random.uniform(-noise, noise)), 4)
            for a in AGES}

def _by_age_list(base_val, noise=0.02):
    import math, random
    random.seed(42)
    return [
        {"age": a,
         "valeur": round(base_val * math.exp(0.08 * (a - 50)) * (1 + random.uniform(-noise, noise)), 4)}
        for a in AGES
    ]


# exposition par âge/sexe
exp_male   = {a: round(4500 - abs(a - 55) * 60 + 200, 0) for a in AGES}
exp_female = {a: round(2800 - abs(a - 55) * 40 + 100, 0) for a in AGES}

# taux bruts et lissés
import math
qx_brut  = {a: round(0.0005 * math.exp(0.085 * (a - 30)), 6) for a in AGES}
qx_lisse = {a: round(qx_brut[a] * 0.95, 6) for a in AGES}   # légèrement plus lisse

# décès observés et modélisés
exp_total = {a: exp_male[a] + exp_female[a] for a in AGES}
obs_deaths   = {a: round(exp_total[a] * qx_brut[a], 2)  for a in AGES}
model_deaths = {a: round(exp_total[a] * qx_lisse[a], 2) for a in AGES}

# référence TH0002
qx_th0002 = {a: round(0.00065 * math.exp(0.088 * (a - 30)), 6) for a in AGES}
discount   = {a: round(qx_lisse[a] / qx_th0002[a], 4) for a in AGES}

# IC à 95%
ci_lower = {a: round(qx_lisse[a] * 0.82, 6) for a in AGES}
ci_upper = {a: round(qx_lisse[a] * 1.18, 6) for a in AGES}

# totaux
total_exp    = sum(exp_total.values())
total_deaths = sum(obs_deaths.values())

# table finale (liste de dicts pour annex)
final_table = [
    {"age": a, "q_x_lisse": qx_lisse[a], "q_x_brut": qx_brut[a],
     "exposition": exp_total[a], "deces_obs": obs_deaths[a], "deces_mod": model_deaths[a]}
    for a in AGES
]

# Données par année
YEARS = [2019, 2020, 2021, 2022, 2023]
exp_by_year    = {y: round(total_exp / len(YEARS) * (1 + 0.02*(y-2021)), 0) for y in YEARS}
deaths_by_year = {y: round(total_deaths / len(YEARS) * (1 + 0.01*(y-2021)), 1) for y in YEARS}

DATA_STORE = {
    # ── study_plan ────────────────────────────────────────────────────────────
    "study_plan": {
        "study_objective":           "prévoyance collective décès",
        "observation_start_date":    "2019-01-01",
        "observation_end_date":      "2023-12-31",
        "observation_period_years":  YEARS,
        "num_observation_years":     len(YEARS),
        "cohort_min_age":            30,
        "cohort_max_age":            85,
        "smoothing_algorithm":       "whittaker_henderson",
        "smoothing_parameters":      "lambda=100",
        "baseline_regulatory_table": "TH0002",
        "product_list":              ["GAR_DC", "PREV_COLL"],
        "crude_rate_method":         "central",
        "confidence_interval_level": 0.95,
        "chi_squared_p_significance":0.05,
        "exclusion_criteria":        "Contrats sans exposition, sinistres non décès",
        "boundary_age_treatment":    "Fermeture aux âges extrêmes (30/85)",
        "discount_jump_tolerance_pct": 0.15,
        "logit_r_squared_minimum":   0.90,
        "death_rate_cv_threshold":   0.30,
        "max_mean_age_change_per_year": 0.5,
    },

    # ── données exposition ────────────────────────────────────────────────────
    "exposure_table": [
        {"age": a, "E_x": exp_total[a], "D_x": obs_deaths[a], "q_x_brut": qx_brut[a]}
        for a in AGES
    ],
    "exposure_by_age_male":   exp_male,
    "exposure_by_age_female": exp_female,
    "exposure_by_year":       exp_by_year,

    # ── décès ─────────────────────────────────────────────────────────────────
    "deaths_by_age_male":     {a: round(exp_male[a] * qx_brut[a], 2)   for a in AGES},
    "deaths_by_age_female":   {a: round(exp_female[a] * qx_brut[a], 2) for a in AGES},
    "observed_deaths_by_age": obs_deaths,
    "modeled_deaths_by_age":  model_deaths,
    "deaths_by_year":         deaths_by_year,

    # ── table lissée ──────────────────────────────────────────────────────────
    "smoothed_table": [
        {"age": a, "q_x_lisse": qx_lisse[a], "q_x_brut": qx_brut[a]}
        for a in AGES
    ],
    "final_mortality_table_by_age": final_table,

    # ── validation ────────────────────────────────────────────────────────────
    "validation": {
        "ci_table": [
            {"age": a, "qx": qx_lisse[a],
             "ci_lower": ci_lower[a], "ci_upper": ci_upper[a]}
            for a in AGES
        ],
        "chi_squared_p": 0.312,
        "p_value":       0.312,
    },
    "ci_lower_by_age":  ci_lower,
    "ci_upper_by_age":  ci_upper,
    "chi_squared_p":    0.312,

    # ── benchmarking ──────────────────────────────────────────────────────────
    "benchmarking": {
        "smr_global":       0.748,
        "reference_name":   "TH0002",
        "abatement_table":  [{"age": a, "abattement": discount[a]} for a in AGES],
    },
    "discount_by_age": discount,
    "avg_prudence_ratio": 0.748,

    # ── statistiques descriptives ─────────────────────────────────────────────
    "summary": {
        "nb_contrats":       12847,
        "age_moyen":         51.3,
        "pct_by_sex":        {"H": 0.637, "F": 0.363},
        "exposition_totale_pa": round(total_exp, 1),
        "nb_deces":          round(total_deaths, 0),
    },
    "total_exposure_years": round(total_exp, 1),
    "total_exposure":       round(total_exp, 1),
    "total_deaths":         round(total_deaths, 0),
    "initial_record_count": 12847,
    "final_record_count":   12847,
    "mean_age_cohort":      51.3,
    "gender_distribution":  {"H": 0.637, "F": 0.363},

    # ── régressions (optionnelles) ────────────────────────────────────────────
    "logit_slope":      0.082,
    "logit_intercept": -5.43,
    "logit_r_squared":  0.968,
    "annual_prediction_ratio": {y: round(1.0 + 0.01*(y-2021), 3) for y in YEARS},
    # cox absent → sections optionnelles skippées

    # ── exposition par classe d'âge (alternative) ─────────────────────────────
    "exposure_by_age_class": {
        "30-39": round(sum(exp_total[a] for a in range(30, 40)), 0),
        "40-49": round(sum(exp_total[a] for a in range(40, 50)), 0),
        "50-59": round(sum(exp_total[a] for a in range(50, 60)), 0),
        "60-69": round(sum(exp_total[a] for a in range(60, 70)), 0),
        "70-85": round(sum(exp_total[a] for a in range(70, 86)), 0),
    },

    # ── diagnostics (facultatif) ──────────────────────────────────────────────
    "diagnostics": {
        "n_low_credibility": 3,
        "pct_low_credibility": 5.4,
        "recommendation": "Table utilisable",
        "n_non_monotone": 0,
        "overall_assessment": "Crédibilité satisfaisante sur 95% des âges",
    },

    # ── session_id ────────────────────────────────────────────────────────────
    "session_id": "test_pipeline_001",
}


# ── Lancement du pipeline ─────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("TEST PIPELINE WRITER — démarrage")
    log.info("  Âges : %d–%d (%d âges)", AGES[0], AGES[-1], len(AGES))
    log.info("  Exposition totale : %.0f années-personnes", total_exp)
    log.info("  Décès totaux : %.0f", total_deaths)
    log.info("  SMR : 0.748  |  chi² p : 0.312  |  R² logit : 0.968")
    log.info("=" * 60)

    t0 = time.time()

    from agents.report.pipeline.run_pipeline import run as run_pipeline

    result = run_pipeline(
        data_store      = DATA_STORE,
        initial_request = (
            "Génère un rapport de certification de table de mortalité "
            "pour notre portefeuille prévoyance collective, en référence TH0002."
        ),
        output_path = "/tmp/test_writer_pipeline.pdf",
    )

    elapsed = time.time() - t0

    log.info("=" * 60)
    log.info("RÉSULTAT PIPELINE")
    log.info("  status       : %s", result.status)
    log.info("  nb_sections  : %d", result.nb_sections)
    log.info("  output_path  : %s", result.output_path)
    log.info("  elapsed      : %.1fs", elapsed)
    log.info("  summary      : %s", result.validation_summary)
    log.info("=" * 60)

    if result.status == "need_data":
        log.error("ÉCHEC — données manquantes : %s", result.need_data)
        sys.exit(1)

    if result.output_path and Path(result.output_path).exists():
        size_kb = Path(result.output_path).stat().st_size // 1024
        log.info("PDF généré : %s (%d KB)", result.output_path, size_kb)
    else:
        log.warning("PDF non trouvé : %s", result.output_path)

    if result.anomalies:
        log.warning("Anomalies (%d) :", len(result.anomalies))
        for a in result.anomalies:
            log.warning("  [%s] %s — %s", a.severity.upper(), a.section_id, a.description)

    log.info("TEST TERMINÉ en %.1fs", elapsed)
    return result


if __name__ == "__main__":
    main()
