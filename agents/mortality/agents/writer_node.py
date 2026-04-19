"""
agents/mortality/agents/writer_node.py
Nœud WriterAgent du graphe LangGraph.

Lance le pipeline de génération de rapport (agents/report/pipeline/run_pipeline.py)
de manière déterministe — pas de LLM dans ce nœud.

Signaux émis :
  <WRITE_DONE: /chemin/rapport.pdf>  → rapport généré, retour au MasterAgent
  <NEED_DATA: field1, field2>        → données insuffisantes, retour au MasterAgent
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from agents.mortality.agents.state import AgentState

log = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _get_initial_request(state: "AgentState") -> str:
    """Extrait le premier message utilisateur comme demande initiale."""
    for msg in state.get("messages", []):
        if isinstance(msg, __import__("langchain_core.messages", fromlist=["HumanMessage"]).HumanMessage):
            return str(msg.content)
    return ""


def writer_node(state: "AgentState") -> dict:
    """
    Nœud WriterAgent : lance run_pipeline et émet le signal approprié.
    Aucun LLM ici — tout est délégué au pipeline.
    """
    new_events: list[dict] = []
    data_store = state.get("data_store") or {}

    new_events.append({
        "type":  "agent_switch",
        "agent": "WriterAgent",
    })
    new_events.append({
        "type":    "message",
        "content": "Lancement du pipeline de génération du rapport...",
    })

    # Demande initiale (pour la validation finale étape 06)
    initial_request = _get_initial_request(state)

    # Chemin de sortie PDF
    session_id  = data_store.get("session_id", "rapport")
    output_path = str(Path("/tmp") / f"rapport_{session_id}.pdf")

    try:
        from agents.report.pipeline.run_pipeline import run as run_pipeline
        result = run_pipeline(
            data_store      = data_store,
            initial_request = initial_request,
            output_path     = output_path,
        )
    except Exception as exc:
        log.error("[WriterAgent] pipeline error : %s", exc)
        new_events.append({"type": "error", "message": f"Erreur pipeline WriterAgent : {exc}"})
        new_events.append({"type": "done"})
        return {"messages": [], "events": new_events}

    # ── Erreur technique (assemblage, reportlab, etc.) ────────────────────────
    if result.status == "error":
        content = f"Erreur lors de la génération du rapport : {result.validation_summary}"
        new_events.append({"type": "error", "message": content})
        new_events.append({"type": "done"})
        lc_msg = AIMessage(content=content)
        return {
            "messages":     [lc_msg],
            "events":       new_events,
            "active_agent": "master",
        }

    # ── Données manquantes → retour au MasterAgent ────────────────────────────
    if result.status == "need_data":
        # need_data vide = validation a bloqué sur des champs sans pouvoir les nommer
        if not result.need_data:
            content = (
                f"Le pipeline de rapport a détecté des données insuffisantes "
                f"mais ne peut pas identifier les champs manquants précisément. "
                f"Détail : {result.validation_summary}"
            )
            new_events.append({"type": "message", "content": content})
            new_events.append({"type": "done"})
            lc_msg = AIMessage(content=content)
            return {"messages": [lc_msg], "events": new_events, "active_agent": "master"}

        fields_str = ", ".join(result.need_data)

        # Garde anti-boucle : si le WriterAgent a déjà signalé ces mêmes champs
        # lors d'un appel précédent, le Builder ne peut pas les produire → mode dégradé.
        prev_need_data = data_store.get("_writer_need_data_prev") or set()
        current_need = set(result.need_data)
        if prev_need_data and (current_need <= prev_need_data):
            log.warning(
                "[WriterAgent] même NEED_DATA qu'au tour précédent (%s) — mode dégradé.",
                fields_str,
            )
            content = (
                f"Certaines données ne sont pas disponibles ({fields_str}), "
                f"mais elles ne peuvent pas être obtenues du Builder. "
                f"Le rapport sera généré avec les sections disponibles.\n\n"
                f"{result.validation_summary}"
            )
            new_events.append({"type": "message", "content": content})
            lc_msg = AIMessage(content=content)
            new_events.append({"type": "done"})
            return {
                "messages":     [lc_msg],
                "events":       new_events,
                "active_agent": "master",
            }

        # Mémoriser pour détecter la boucle au prochain tour
        data_store["_writer_need_data_prev"] = current_need

        content = (
            f"Les données suivantes sont insuffisantes pour générer le rapport : "
            f"<NEED_DATA: {fields_str}>\n\n"
            f"{result.validation_summary}"
        )
        new_events.append({"type": "message", "content": content})
        lc_msg = AIMessage(content=content)
        return {
            "messages":     [lc_msg],
            "events":       new_events,
            "active_agent": "master",
        }

    # ── Succès (avec ou sans warnings) ───────────────────────────────────────
    warnings_text = ""
    if result.status == "success_with_warnings" and result.anomalies:
        anomaly_lines = "\n".join(
            f"  - [{a.severity.upper()}] {a.section_id} : {a.description}"
            for a in result.anomalies
        )
        warnings_text = f"\n\n⚠ Points à noter :\n{anomaly_lines}"

    content = (
        f"Rapport généré avec succès ({result.nb_sections} sections).\n"
        f"Fichier : {result.output_path}\n"
        f"{result.validation_summary}"
        f"{warnings_text}\n\n"
        f"<WRITE_DONE: {result.output_path}>"
    )

    new_events.append({"type": "message", "content": content})
    new_events.append({
        "type":        "report_ready",
        "output_path": result.output_path,
        "nb_sections": result.nb_sections,
        "status":      result.status,
    })
    new_events.append({"type": "done"})

    lc_msg = AIMessage(content=content)
    return {
        "messages":     [lc_msg],
        "events":       new_events,
        "data_store":   data_store,
        "active_agent": "master",
    }
