"""
agents.report.describe_capabilities

Self-describe du ReportAgent — appelé au démarrage par capability_registry.

Le ReportAgent génère des PDF (certification, descriptif) à partir du
data_store rempli par le MortalityAgent. Il produit aussi les graphiques
intégrés au rapport.
"""
from __future__ import annotations


_OWNED_NAMESPACES = ("build_pdf", "graphs")


def _list_owned_tools() -> list[dict]:
    """Filtre le catalogue pour ne garder que les tools de cet agent."""
    try:
        from tools.catalogue import get_catalogue
    except Exception:
        return []
    cat = get_catalogue() or {}
    tools_flat = cat.get("tools") or {}
    owned: list[dict] = []
    for qualified_name, info in tools_flat.items():
        if info.get("client_visible") is False:
            continue
        if not qualified_name.startswith(tuple(f"{ns}." for ns in _OWNED_NAMESPACES)):
            continue
        owned.append({
            "name":        qualified_name,
            "display":     info.get("display_name") or qualified_name,
            "description": info.get("short_description") or "",
        })
    return sorted(owned, key=lambda x: x["name"])


def describe() -> dict:
    """Retourne le descriptif des capacités de rédaction PDF du ReportAgent."""
    return {
        "agent":   "report",
        "display": "ReportAgent — Rapports PDF",
        "purpose": (
            "Génère des rapports PDF professionnels à partir des résultats "
            "actuariels calculés. Deux modes : certification (conforme A132-18) "
            "et descriptif (analyse exploratoire). Inclut graphiques intégrés."
        ),
        "tools":           _list_owned_tools(),
        "report_modes": {
            "certification":  "Rapport de certification A132-18 (signature actuariel certifié)",
            "descriptive":    "Rapport descriptif du portefeuille (analyses exploratoires)",
        },
        "graphs_produced": [
            "qx vs âge (taux bruts vs lissés)",
            "SMR par âge avec intervalles de confiance",
            "Pyramide des âges du portefeuille",
            "Comparaison référentielle (vs TH 00-02, TGH/TGF 05)",
            "Distribution exposition par âge",
        ],
        "inputs":  [
            "data_store rempli par MortalityAgent (exposure, qx, smoothed, validation, benchmarking)",
            "informations portefeuille (titre, période, sexe)",
            "commentary rédigé (interprétation actuarielle)",
        ],
        "outputs": [
            "PDF certification (~30-50 pages)",
            "PDF descriptif (~15-25 pages)",
            "Notebook Jupyter exporté (build_pdf.generate_notebook)",
            "Session log (build_pdf.session_log)",
        ],
    }
