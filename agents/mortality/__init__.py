"""
report_agent — Génération du rapport PDF de certification
de la table de mortalité d'expérience.

Usage :
    from report_agent.validate_payload import validate
    from report_agent.generate_report import generate_mortality_report

    result = validate(payload)
    if not result.valid:
        raise ValueError(result.refusal_message())
    pdf_path = generate_mortality_report(payload, output_path="rapport.pdf")
"""
__version__ = "1.0.0"
