"""
report_writer — module d'assemblage PURE du rapport PDF de mortalité.

Interface publique :
    run_writer(study_plan, calculation_agent_output, yaml_template_path) → dict
    WriterError — levée si un champ requis est absent
"""
from report_writer.errors import WriterError
from report_writer.report_builder import ReportBuilder


def run_writer(
    study_plan: dict,
    calculation_agent_output: dict,
    yaml_template_path: str,
) -> dict:
    """
    Assemble le rapport PDF de certification de mortalité.

    Args:
        study_plan                : paramètres de l'étude (dates, âges, algorithme...)
        calculation_agent_output  : résultats du BuilderAgent (data_store)
        yaml_template_path        : chemin vers mortality_template.yaml

    Returns:
        {"status": "success", "report_path": str, ...}

    Raises:
        WriterError : si un champ requis est absent ou vide dans les inputs
    """
    builder = ReportBuilder(yaml_template_path, study_plan, calculation_agent_output)
    return builder.run()


__all__ = ["run_writer", "WriterError"]
