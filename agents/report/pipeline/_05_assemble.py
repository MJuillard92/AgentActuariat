"""
agents/report/pipeline/05_assemble.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPE 5 — Déterministe, zéro LLM

Wrapper d'appel vers tools/build_pdf/assemble_sections.py.
Prend le data_store avec section_outputs rempli (étape 04)
et produit le PDF final.

Interface publique :
    assemble(data_store, output_path, title) -> AssembleResult
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

_DEFAULT_OUTPUT = Path("/tmp") / f"rapport_mortalite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"


@dataclass
class AssembleResult:
    success:     bool
    output_path: str
    nb_sections: int
    warning:     str


def assemble(
    data_store:  dict,
    output_path: str | None = None,
    title:       str = "",
) -> AssembleResult:
    """
    Appelle assemble_sections pour produire le PDF final.

    Args:
        data_store  : doit contenir section_outputs et template_context
        output_path : chemin de sortie du PDF (défaut : /tmp/rapport_...pdf)
        title       : titre du rapport (déduit du contexte si vide)

    Returns:
        AssembleResult avec success, output_path, nb_sections, warning
    """
    output_path = output_path or str(_DEFAULT_OUTPUT)

    # Titre par défaut depuis le contexte
    if not title:
        ctx = data_store.get("template_context") or {}
        obj = ctx.get("study_objective", "")
        ref = ctx.get("baseline_regulatory_table", "")
        if obj:
            title = f"Table de mortalité d'expérience — {obj}"
            if ref:
                title += f" / réf. {ref}"
        else:
            title = "Rapport de Certification — Table de Mortalité d'Expérience"

    # Infos portefeuille depuis le contexte
    ctx = data_store.get("template_context") or {}
    parts = []
    if ctx.get("observation_start_date") and ctx.get("observation_end_date"):
        parts.append(f"Période : {ctx['observation_start_date']} – {ctx['observation_end_date']}")
    if ctx.get("total_exposure_years"):
        parts.append(f"{ctx['total_exposure_years']:.0f} années-personnes")
    if ctx.get("total_deaths"):
        parts.append(f"{ctx['total_deaths']} décès")
    portfolio_info = "  |  ".join(parts)

    # Vérification préalable
    section_outputs = data_store.get("section_outputs") or {}
    if not section_outputs:
        return AssembleResult(
            success=False, output_path="", nb_sections=0,
            warning="section_outputs vide — lancer 04_redaction.py avant."
        )

    n_done = sum(1 for s in section_outputs.values() if s.get("status") != "skipped")
    log.info("[05_assemble] assemblage de %d section(s) → %s", n_done, output_path)

    # Appel déterministe
    try:
        from tools.build_pdf.assemble_sections import run as _assemble_run
        result = _assemble_run(
            data=data_store,
            params={
                "output_path":    output_path,
                "title":          title,
                "portfolio_info": portfolio_info,
            },
        )
    except Exception as exc:
        log.error("[05_assemble] erreur : %s", exc)
        return AssembleResult(success=False, output_path="", nb_sections=0, warning=str(exc))

    return AssembleResult(
        success     = result.get("succes", False),
        output_path = result.get("output_path", ""),
        nb_sections = result.get("nb_sections", 0),
        warning     = result.get("warning", ""),
    )
