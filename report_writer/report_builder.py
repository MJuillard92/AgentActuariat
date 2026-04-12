"""
report_writer/report_builder.py
ReportBuilder — assemblage PURE du rapport PDF.

Pattern : WriterAgent PURE
  - Lit uniquement study_plan et calculation_agent_output
  - Lève WriterError si un champ REQUIS est absent ou vide
  - Ne calcule JAMAIS de valeur manquante
  - Ne contacte JAMAIS le BuilderAgent
"""
from __future__ import annotations

import logging
from typing import Any

from report_writer.errors import WriterError

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED_FIELDS — extraits du YAML mortality_template.yaml v2.0
# Toute valeur absente (None, [], {}, "") dans ces listes lève WriterError.
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FIELDS_STUDY_PLAN = [
    "study_objective",
    "observation_start_date",
    "observation_end_date",
    "observation_period_years",
    "cohort_min_age",
    "cohort_max_age",
    "smoothing_algorithm",
    "baseline_regulatory_table",
]

# Champs requis dans calculation_agent_output, par section YAML.
# La clé est le section_id YAML tel que défini dans processing_sequence.
REQUIRED_FIELDS_DATA_STORE = {
    "preamble": [
        "total_exposure_years",
        "total_deaths",
    ],
    "data_submission": [
        "initial_record_count",
        "final_record_count",
        "exposure_by_age_male",
        "exposure_by_age_female",
        "deaths_by_age_male",
        "deaths_by_age_female",
        "mean_age_cohort",
        "gender_distribution",
    ],
    "obs_vs_modeled": [
        "observed_deaths_by_age",
        "modeled_deaths_by_age",
        "ci_lower_by_age",
        "ci_upper_by_age",
        "chi_squared_p",
    ],
    "regulatory": [
        "discount_by_age",
    ],
    "conclusion": [
        "avg_prudence_ratio",
    ],
    "annex": [
        "final_mortality_table_by_age",
    ],
}

# Champs OPTIONNELS — leur absence ne lève pas d'erreur.
# La section correspondante est skippée ou réduite.
OPTIONAL_FIELDS_DATA_STORE = {
    "data_submission": [
        "cox_hazard_ratio",
        "cox_pvalue",
        "exposure_by_year",
        "deaths_by_year",
    ],
    "prior_comparison": [
        "rate_ratio_current_vs_prior",
        "prior_prudence_ratio",
    ],
    "regulatory": [
        "logit_slope",
        "logit_intercept",
        "logit_r_squared",
    ],
    "obs_vs_modeled": [
        "annual_prediction_ratio",
    ],
}


def _is_empty(value: Any) -> bool:
    """Retourne True si la valeur est considérée comme absente (None, [], {}, "")."""
    return value is None or value == [] or value == {} or value == ""


# ─────────────────────────────────────────────────────────────────────────────

class ReportBuilder:
    """
    Assemble le rapport PDF à partir de study_plan et calculation_agent_output.

    Utilisation :
        builder = ReportBuilder(yaml_path, study_plan, calculation_agent_output)
        result  = builder.run()
        # result = {"status": "success", "report_path": "..."} ou WriterError levée
    """

    def __init__(
        self,
        yaml_path: str,
        study_plan: dict,
        calculation_agent_output: dict,
    ) -> None:
        self.yaml_path = yaml_path
        self.study_plan = study_plan or {}
        self.calculation_agent_output = calculation_agent_output or {}
        self.context: dict = {}
        self.section_outputs: dict = {}

    # ─── Étape 1 : validation et construction du contexte ────────────────────

    def _initialize_context(self) -> None:
        """
        Valide la présence et la non-vacuité de TOUS les champs requis.
        Lève WriterError si un seul champ est absent ou vide.
        Ne calcule aucune valeur — lecture directe uniquement.
        """
        missing_sp = []
        missing_ds = {}

        # — Validation study_plan ─────────────────────────────────────────────
        for field in REQUIRED_FIELDS_STUDY_PLAN:
            if _is_empty(self.study_plan.get(field)):
                missing_sp.append(field)
                log.warning("[ReportBuilder] Champ manquant dans study_plan : %s", field)

        # — Validation calculation_agent_output (par section) ─────────────────
        for section_id, fields in REQUIRED_FIELDS_DATA_STORE.items():
            for field in fields:
                if _is_empty(self.calculation_agent_output.get(field)):
                    missing_ds.setdefault(section_id, []).append(field)
                    log.warning(
                        "[ReportBuilder] Champ manquant dans data_store "
                        "(section '%s') : %s", section_id, field
                    )

        # — Lever WriterError si un champ requis est absent ───────────────────
        all_missing = missing_sp + [
            f for lst in missing_ds.values() for f in lst
        ]
        if all_missing:
            # Priorité : data_store (BuilderAgent responsable) > study_plan
            if missing_ds:
                source = "calculation_agent_output"
                section = next(iter(missing_ds))
            else:
                source = "study_plan"
                section = "global"
            raise WriterError(all_missing, source, section)

        # — Construire le contexte via load_yaml_template (résolution enrichie) ─
        # Tente d'utiliser load_yaml_template pour obtenir les valeurs dérivées
        # (segmentation par sexe, CI bounds, etc.). Bascule sur lecture directe si indisponible.
        _lyt_context = None
        try:
            from tools.build_pdf.load_yaml_template import run as _lyt_run
            combined = dict(self.calculation_agent_output)
            combined["study_plan"] = self.study_plan
            lyt_result = _lyt_run(
                data=combined,
                params={"yaml_path": self.yaml_path, "study_plan": self.study_plan},
            )
            if "erreur" not in lyt_result:
                _lyt_context = lyt_result.get("template_context") or {}
        except Exception as exc:
            log.warning("[ReportBuilder] load_yaml_template indisponible, lecture directe : %s", exc)

        if _lyt_context:
            self.context = _lyt_context
        else:
            # Fallback : lecture directe depuis study_plan + calculation_agent_output
            self.context = {f: self.study_plan[f] for f in REQUIRED_FIELDS_STUDY_PLAN}
            for fields in REQUIRED_FIELDS_DATA_STORE.values():
                for f in fields:
                    self.context[f] = self.calculation_agent_output[f]
            for fields in OPTIONAL_FIELDS_DATA_STORE.values():
                for f in fields:
                    self.context[f] = self.calculation_agent_output.get(f)

        # num_observation_years : garantir la présence (load_yaml_template le calcule aussi)
        if "num_observation_years" not in self.context:
            self.context["num_observation_years"] = len(
                self.study_plan.get("observation_period_years", [])
            )

        log.info(
            "[ReportBuilder] Contexte initialisé : %d champs résolus (%d optionnels à None)",
            len(self.context),
            sum(1 for v in self.context.values() if v is None),
        )

    def run(self) -> dict:
        """
        Point d'entrée principal.
        Retourne {"status": "success", ...} ou lève WriterError.
        """
        self._initialize_context()
        # Les étapes suivantes (sections, PDF) seront ajoutées dans les lacunes #2 et #3.
        log.info("[ReportBuilder] run() — context OK, assembly à implémenter.")
        return {"status": "context_ok", "context_keys": list(self.context.keys())}
