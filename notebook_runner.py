"""
notebook_runner.py
Module de lecture et d'exécution des notebooks blueprint.
Supporte la lecture de notebooks individuels et la concaténation de plusieurs notebooks.
"""

from __future__ import annotations

import io
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import nbformat


def load_notebook(path: str) -> list:
    """Charge un notebook .ipynb et retourne la liste des cellules.

    Retourne une liste de dicts :
        {"id": str, "type": "code"|"markdown", "source": str, "output": str}
    """
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    cells = []
    for i, cell in enumerate(nb.cells):
        cells.append(
            {
                "id": cell.get("id", f"cell-{i}"),
                "type": cell.cell_type,
                "source": cell.source,
                "output": "",
            }
        )
    return cells


def execute_cell(cell_source: str, kernel_state: dict) -> str:
    """Exécute du code Python dans le namespace partagé.

    Capture stdout ET les appels display() (DataFrames, objets IPython).
    En cas d'erreur, retourne le traceback complet.

    Args:
        cell_source:  Code Python à exécuter.
        kernel_state: Namespace partagé entre les cellules (maintient l'état).

    Returns:
        Sortie texte (stdout + repr des objets display) ou message d'erreur.
    """
    stdout_capture = io.StringIO()
    display_outputs = []

    def _display(*objs, **kwargs):
        """Remplace IPython.display.display — capture texte ET HTML."""
        for obj in objs:
            try:
                import pandas as pd
                if isinstance(obj, (pd.DataFrame, pd.Series)):
                    display_outputs.append({
                        "text": obj.to_string(),
                        "html": obj.to_html(border=1),
                    })
                else:
                    display_outputs.append({"text": repr(obj), "html": None})
            except Exception:
                display_outputs.append({"text": repr(obj), "html": None})

    # Injecter display dans le kernel (sans écraser une éventuelle valeur existante)
    _prev_display = kernel_state.get("display")
    kernel_state["display"] = _display

    try:
        with redirect_stdout(stdout_capture):
            exec(cell_source, kernel_state)  # noqa: S102
        parts = []
        if stdout_capture.getvalue().strip():
            parts.append(stdout_capture.getvalue())
        if display_outputs:
            parts.append("\n".join(d["text"] for d in display_outputs))
        output = "\n".join(parts)
        # Stocker les outputs riches pour la génération du notebook
        kernel_state["_last_display_outputs"] = display_outputs
        return output if output.strip() else "✓ Exécution réussie (pas de sortie texte)"
    except Exception:
        return f"❌ Erreur :\n{traceback.format_exc()}"
    finally:
        # Restaurer display si elle existait avant
        if _prev_display is None:
            kernel_state.pop("display", None)
        else:
            kernel_state["display"] = _prev_display


def get_notebook_as_context(path: str) -> str:
    """Retourne le contenu d'un notebook en texte brut pour injection dans le contexte LLM.

    Format :
        [NOTEBOOK: nom_fichier]
        [MARKDOWN]
        {source}

        [CODE]
        {source}
    """
    name = Path(path).name
    cells = load_notebook(path)
    lines = [f"[NOTEBOOK: {name}]"]
    for cell in cells:
        tag = "[MARKDOWN]" if cell["type"] == "markdown" else "[CODE]"
        lines.append(f"{tag}\n{cell['source']}\n")
    return "\n".join(lines)


def get_notebooks_as_context(paths: list[str]) -> str:
    """Concatène plusieurs notebooks en un contexte LLM unique.

    Args:
        paths: Liste ordonnée des chemins de notebooks.

    Returns:
        Texte complet pour injection dans le system prompt.
    """
    sections = []
    for path in paths:
        sections.append(get_notebook_as_context(path))
    return "\n\n" + "=" * 60 + "\n\n".join(sections)
