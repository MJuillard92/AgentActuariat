"""
agents/report/errors.py
Erreurs structurées du WriterAgent — conçues pour être parsées par le MasterAgent.
"""
from __future__ import annotations


class WriterError(Exception):
    """
    Levée par le WriterAgent quand des données requises sont absentes ou vides.

    Attributs (accessibles par le MasterAgent pour router vers le BuilderAgent) :
        missing_inputs   : liste des champs manquants (ex. ["modeled_deaths_by_age"])
        source_should_be : source attendue ("study_plan" | "calculation_agent_output")
        section_id       : section du YAML qui a détecté le manque (ex. "obs_vs_modeled")
    """

    def __init__(
        self,
        missing_inputs,
        source_should_be: str,
        section_id: str,
    ) -> None:
        self.missing_inputs = missing_inputs
        self.source_should_be = source_should_be
        self.section_id = section_id
        summary = ", ".join(missing_inputs)
        super().__init__(
            f"[WriterError] Champs manquants dans {source_should_be} "
            f"(section '{section_id}') : {summary}"
        )
