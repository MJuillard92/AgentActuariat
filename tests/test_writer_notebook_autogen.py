"""
HOTFIX-pre-refacto-2026-05 — Bug 12 : writer_node déclenche automatiquement
generate_notebook après PDF succès. Garantit la livraison du livrable
reproductible sans dépendre du LLM Builder qui pouvait l'oublier.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _stub_pipeline_result(status="success", nb_sections=5, output_path="/tmp/rapport_test.pdf"):
    return SimpleNamespace(
        status=status,
        nb_sections=nb_sections,
        output_path=output_path,
        validation_summary="OK",
        anomalies=[],
        need_data=[],
    )


def test_writer_node_triggers_notebook_on_success(tmp_path) -> None:
    """Après PDF succès, writer_node doit appeler generate_notebook + émettre
    un event notebook_ready."""
    from agents.mortality.agents import writer_node as wn

    pdf_path = str(tmp_path / "rapport_x.pdf")
    nb_expected = str(tmp_path / "rapport_x.ipynb")

    state = {
        "messages": [],
        "data_store": {
            "_call_log":    [{"tool": "builder", "function_name": "exposure", "params": {}}],
            "csv_filename": "portefeuille_test.csv",
            "session_id":   "test_sess",
        },
    }

    nb_run_mock = MagicMock(return_value={
        "succes":      True,
        "output_path": nb_expected,
        "nb_cellules": 12,
    })

    with patch("agents.report.pipeline.run_pipeline.run") as mock_pipeline, \
         patch("tools.export.generate_notebook.run", nb_run_mock):
        mock_pipeline.return_value = _stub_pipeline_result(output_path=pdf_path)
        result = wn.writer_node(state)

    assert nb_run_mock.called, "generate_notebook devrait être appelé"
    notebook_events = [e for e in result["events"] if e.get("type") == "notebook_ready"]
    assert len(notebook_events) == 1, f"event notebook_ready manquant : {result['events']}"
    assert notebook_events[0]["output_path"] == nb_expected
    assert notebook_events[0]["nb_cellules"] == 12


def test_writer_node_notebook_failure_does_not_block_pdf(tmp_path) -> None:
    """Si generate_notebook crashe, le PDF reste livré (best-effort)."""
    from agents.mortality.agents import writer_node as wn

    state = {
        "messages": [],
        "data_store": {"_call_log": [{"tool": "x", "function_name": "y", "params": {}}]},
    }

    with patch("agents.report.pipeline.run_pipeline.run") as mock_pipeline, \
         patch("tools.export.generate_notebook.run", side_effect=RuntimeError("nb fail")):
        mock_pipeline.return_value = _stub_pipeline_result(output_path=str(tmp_path / "r.pdf"))
        result = wn.writer_node(state)

    # PDF event toujours émis
    report_events = [e for e in result["events"] if e.get("type") == "report_ready"]
    assert len(report_events) == 1
    # Pas d'event notebook_ready (échec silencieux)
    notebook_events = [e for e in result["events"] if e.get("type") == "notebook_ready"]
    assert len(notebook_events) == 0


def test_writer_node_no_notebook_on_need_data(tmp_path) -> None:
    """Si pipeline retourne need_data (pas de PDF), pas de notebook non plus."""
    from agents.mortality.agents import writer_node as wn

    state = {
        "messages": [],
        "data_store": {},
    }

    nb_run_mock = MagicMock()
    with patch("agents.report.pipeline.run_pipeline.run") as mock_pipeline, \
         patch("tools.export.generate_notebook.run", nb_run_mock):
        mock_pipeline.return_value = SimpleNamespace(
            status="need_data",
            need_data=["qx_table"],
            validation_summary="champs manquants",
            anomalies=[],
            output_path=None,
            nb_sections=0,
        )
        wn.writer_node(state)

    nb_run_mock.assert_not_called()
