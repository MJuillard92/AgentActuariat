"""
canvas_app.py
Interface Dash N8N-like pour l'orchestration de notebooks actuariels.

Lancer :  python canvas_app.py
URL :     http://localhost:8050
"""

from __future__ import annotations

import base64
import io
import atexit
import json
import os
import socket
import subprocess
import threading
import time as _time_mod
import uuid
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import nbformat
from dash import Input, Output, State, callback_context, dcc, html
from dash import dash_table
from dotenv import load_dotenv

import config
from notebook_runner import load_notebook, execute_cell
from word_generator import generate_word_report
from workflow import Workflow, WorkflowNode, WorkflowEdge, default_workflow
from workflow_executor import execute_workflow, make_kernel, evaluate_condition
from agent import run_agent_loop, SYSTEM_PROMPT_TEMPLATE
from workflow_executor import capture_figures
from rag import answer_with_rag, answer_with_tools, precompute_index, build_source_chunks, RAG_SYSTEM_PROMPT, RAG_TOOLS_SYSTEM_PROMPT
from actuary_logger import LOGGER as _ACTUARY_LOGGER
from actuary_state import STATE as _ACTUARY_STATE
from analyze_report_template import analyze_report_pdf

# Validator loaded lazily (importlib) to avoid heavy import at startup
_validator_module = None
def _get_validator():
    global _validator_module
    if _validator_module is None:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "pipeline_validator",
            str(Path(__file__).parent / "notebooks" / "00_pipeline_validator.py"),
        )
        _validator_module = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_validator_module)
    return _validator_module

load_dotenv()
cyto.load_extra_layouts()

# ─────────────────────────────────────────────────────────────────────────────
# Rendu HTML des notebooks via nbconvert (pas de serveur Jupyter externe)
# ─────────────────────────────────────────────────────────────────────────────

_ANACONDA_PYTHON = "/opt/anaconda3/bin/python"
_ANACONDA_JUPYTER = "/opt/anaconda3/bin/jupyter"
_jupyter_proc: "subprocess.Popen | None" = None
_jupyter_port: int = 0


def _start_jupyter_server() -> int:
    """Démarre Jupyter Notebook (Anaconda) si nécessaire, retourne le port."""
    global _jupyter_proc, _jupyter_port
    if _jupyter_proc is not None and _jupyter_proc.poll() is None:
        return _jupyter_port
    import socket as _socket
    with _socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    project_dir = Path(__file__).parent
    _jupyter_proc = subprocess.Popen(
        [_ANACONDA_JUPYTER, "notebook", f"--port={port}", "--no-browser",
         "--NotebookApp.token=", "--NotebookApp.password=",
         "--NotebookApp.allow_origin=*", "--NotebookApp.disable_check_xsrf=True",
         f"--notebook-dir={project_dir}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    _jupyter_port = port
    import urllib.request as _u
    deadline = _time_mod.time() + 20
    while _time_mod.time() < deadline:
        try:
            _u.urlopen(f"http://127.0.0.1:{port}/api/kernels", timeout=1)
            break
        except Exception:
            _time_mod.sleep(0.5)
    return port

def _patch_notebook_setup_cell(nb_path: Path, csv_path: str = "", sexe: str = "H") -> None:
    """Met à jour le setup cell : importlib.util + injection FILE_PATH/SEXE.

    - Remplace l'ancien `import data_prep` par la version importlib.util.
    - Injecte csv_path et sexe (priorité : paramètres > valeurs déjà dans le notebook).
    - Sans effet si le notebook est déjà à jour (importlib.util présent ET FILE_PATH correct).
    """
    try:
        import nbformat as _nbf
        nb = _nbf.read(str(nb_path), as_version=4)
        project_dir = str(Path(__file__).parent)
        notebooks_dir = str(Path(__file__).parent / "notebooks")
        _module_map = {
            "01_data_preparation": "data_prep",
            "02_exposure":         "exposure",
            "03_crude_rates":      "crude_rates",
            "04_smoothing":        "smoothing",
            "05_diagnostics":      "diagnostics",
            "06_validation":       "validation",
            "07_benchmarking":     "benchmarking",
            "08_visualization":    "visualization",
        }
        mod_lines = "\n".join(
            f"{v} = _load_module(r'{notebooks_dir}/{k}.py', '{v}')"
            for k, v in _module_map.items()
        )
        loader_block = (
            f"import sys, os, importlib.util\n"
            f"os.chdir(r'{project_dir}')\n"
            f"sys.path.insert(0, r'{project_dir}')\n"
            f"sys.path.insert(0, r'{notebooks_dir}')\n\n"
            "def _load_module(path, alias):\n"
            "    spec = importlib.util.spec_from_file_location(alias, path)\n"
            "    mod = importlib.util.module_from_spec(spec)\n"
            "    spec.loader.exec_module(mod)\n"
            "    sys.modules[alias] = mod\n"
            "    return mod\n\n"
            f"{mod_lines}\n\n"
            "import pandas as pd\nimport numpy as np\n"
            "%matplotlib inline\n"
            "import matplotlib\nimport matplotlib.pyplot as plt\n\n"
            "try:\n"
            "    from actuarial_params import PARAMS\n"
            "except ImportError:\n"
            "    pass\n"
        )
        file_path_line = (
            f"FILE_PATH = r'{csv_path}'" if csv_path
            else "FILE_PATH = None  # définir le chemin CSV avant d'exécuter"
        )
        sexe_line = f"SEXE = '{sexe}'"

        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            src = "".join(cell.source)
            needs_patch = "import data_prep" in src and "importlib.util" not in src
            needs_csv   = "importlib.util" in src and (
                "FILE_PATH = None" in src or "FILE_PATH = r''" in src
            ) and csv_path
            if needs_patch or needs_csv:
                nb.cells[i].source = f"{file_path_line}\n{sexe_line}\n\n{loader_block}"
            break  # seul le 1er code cell est le setup
        _nbf.write(nb, str(nb_path))
    except Exception:
        pass  # ne pas bloquer si le patch échoue


def _notebook_to_html(nb_path: Path) -> str:
    """Convertit un .ipynb en HTML via nbconvert (Python Anaconda).

    Retourne le HTML complet à injecter dans un iframe srcDoc.
    """
    script = (
        "import nbformat, sys;"
        "from nbconvert import HTMLExporter;"
        f"nb=nbformat.read(open(r'{nb_path}','r',encoding='utf-8'),as_version=4);"
        "exp=HTMLExporter(template_name='classic');"
        "body,_=exp.from_notebook_node(nb);"
        "sys.stdout.buffer.write(body.encode('utf-8'))"
    )
    result = subprocess.run(
        [_ANACONDA_PYTHON, "-c", script],
        capture_output=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace")[:300])
    return result.stdout.decode("utf-8")


def _list_notebooks() -> list[dict]:
    """Liste tous les notebooks .ipynb disponibles (générés + projet)."""
    root = Path(__file__).parent
    results = []
    # Notebooks générés par l'agent (les plus récents en premier)
    gen_dir = root / "notebooks_generated"
    if gen_dir.exists():
        for p in sorted(gen_dir.glob("*.ipynb"), reverse=True):
            results.append({
                "label": f"🤖 {p.name}",
                "value": str(p),
                "group": "Agent",
            })
    # Notebooks du projet
    nb_dir = root / "notebooks"
    if nb_dir.exists():
        for p in sorted(nb_dir.glob("*.ipynb")):
            results.append({
                "label": f"📓 {p.name}",
                "value": str(p),
                "group": "Projet",
            })
    return results


atexit.register(lambda: _jupyter_proc.terminate() if _jupyter_proc else None)


def _generate_agent_notebook(steps: list, summary: str = "",
                              csv_path: str = "", sexe: str = "H") -> Path:
    """Crée un vrai fichier .ipynb structuré avec raisonnement et tentatives échouées."""
    from datetime import datetime as _dt
    nb = nbformat.v4.new_notebook()
    cells = []
    ts = _dt.now().strftime("%Y-%m-%d %H:%M")

    n_steps = len(steps)
    n_failed = sum(1 for s in steps if s.get("output", "").startswith("❌"))
    cells.append(nbformat.v4.new_markdown_cell(
        f"# Analyse Actuarielle — Agent\n\n"
        f"*Généré le {ts}*\n\n"
        f"- **{n_steps} étapes** exécutées ({n_failed} tentatives échouées, "
        f"{n_steps - n_failed} réussies)\n"
        f"- Exécutez les cellules **dans l'ordre** pour recréer l'état complet\n"
        f"- Les cellules préfixées `# ❌` sont des tentatives échouées — "
        f"elles sont conservées pour la traçabilité mais ne doivent pas être rejouées\n"
    ))

    project_dir = str(Path(__file__).parent)
    notebooks_dir = str(Path(__file__).parent / "notebooks")
    csv_line = f"FILE_PATH = r'{csv_path}'" if csv_path else "FILE_PATH = None  # à définir"
    _module_map = {
        "01_data_preparation": "data_prep",
        "02_exposure":         "exposure",
        "03_crude_rates":      "crude_rates",
        "04_smoothing":        "smoothing",
        "05_diagnostics":      "diagnostics",
        "06_validation":       "validation",
        "07_benchmarking":     "benchmarking",
        "08_visualization":    "visualization",
    }
    mod_lines = "\n".join(
        f"{v} = _load_module(r'{notebooks_dir}/{k}.py', '{v}')"
        for k, v in _module_map.items()
    )
    setup = (
        f"import sys, os, importlib.util\n"
        f"os.chdir(r'{project_dir}')\n"
        f"sys.path.insert(0, r'{project_dir}')\n"
        f"sys.path.insert(0, r'{notebooks_dir}')\n\n"
        f"# Portefeuille d'entrée\n"
        f"{csv_line}\n"
        f"SEXE = '{sexe}'\n\n"
        "import pandas as pd\nimport numpy as np\n"
        "import matplotlib\nimport matplotlib.pyplot as plt\n"
        "%matplotlib inline\n\n"
        "# Chargement des modules actuariels (alias courts)\n"
        "def _load_module(path, alias):\n"
        "    spec = importlib.util.spec_from_file_location(alias, path)\n"
        "    mod = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(mod)\n"
        "    sys.modules[alias] = mod\n"
        "    return mod\n\n"
        f"{mod_lines}\n\n"
        "try:\n"
        f"    from actuarial_params import PARAMS\n"
        "except ImportError:\n"
        "    pass\n"
    )
    cells.append(nbformat.v4.new_markdown_cell("## 0 — Configuration"))
    cells.append(nbformat.v4.new_code_cell(setup))

    step_num = 0  # compte uniquement les étapes réussies pour la numérotation

    for raw_i, step in enumerate(steps):
        code = (step.get("code") or "").strip()
        desc = (step.get("description") or f"Étape {raw_i + 1}").strip()
        output = (step.get("output") or "").strip()
        success = step.get("success", not output.startswith("❌"))

        if success:
            step_num += 1

        # ── Cellule Markdown : raisonnement de l'agent ────────────────────────
        if not success:
            status_icon = "⚠️"
            step_label = f"Tentative échouée — {desc}"
            err_lines = output.splitlines()[:6]
            err_preview = "\n".join(f"    {l}" for l in err_lines)
            md = (
                f"## {status_icon} {step_label}\n\n"
                f"**Raisonnement agent** : {desc}\n\n"
                f"**Résultat** : ❌ Échec\n\n"
                f"```\n{err_preview}\n```\n"
            )
        else:
            step_label = f"Étape {step_num} — {desc}"
            md = (
                f"## ✅ {step_label}\n\n"
                f"**Raisonnement agent** : {desc}\n"
            )

        cells.append(nbformat.v4.new_markdown_cell(md))

        # ── Cellule code ───────────────────────────────────────────────────────
        if code:
            if not success:
                prefix = (
                    f"# ❌ Tentative échouée — non rejouable\n"
                    f"# Erreur : {output[:120].splitlines()[0] if output else '?'}\n\n"
                )
                cells.append(nbformat.v4.new_code_cell(prefix + code))
            else:
                cells.append(nbformat.v4.new_code_cell(code))

    # ── Synthèse finale ───────────────────────────────────────────────────────
    if summary:
        cells.append(nbformat.v4.new_markdown_cell(
            f"## 📋 Synthèse finale\n\n{summary}"
        ))

    nb.cells = cells
    out_dir = Path(__file__).parent / "notebooks_generated"
    out_dir.mkdir(exist_ok=True)
    ts_file = _dt.now().strftime("%Y%m%d_%H%M%S")
    nb_path = out_dir / f"agent_{ts_file}.ipynb"
    nbformat.write(nb, str(nb_path))
    return nb_path

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.resolve()

WORKFLOWS_DIR = (_ROOT / "workflows")
WORKFLOWS_DIR.mkdir(exist_ok=True)

UPLOADS_DIR = (_ROOT / Path(config.UPLOADS_DIR).name)
UPLOADS_DIR.mkdir(exist_ok=True)

PALETTE = [
    "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#F44336", "#607D8B", "#E91E63",
]

# ─────────────────────────────────────────────────────────────────────────────
# Cytoscape stylesheet
# ─────────────────────────────────────────────────────────────────────────────
CYTO_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "content": "data(label)",
            "background-color": "data(color)",
            "color": "#fff",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "12px",
            "font-weight": "bold",
            "width": "160px",
            "height": "60px",
            "shape": "round-rectangle",
            "border-width": 2,
            "border-color": "#fff",
            "text-wrap": "wrap",
            "text-max-width": "140px",
        },
    },
    {
        "selector": "node:selected",
        "style": {"border-width": 3, "border-color": "#FFD600"},
    },
    {"selector": ".running", "style": {"border-color": "#FFD600", "border-width": 4}},
    {"selector": ".done",    "style": {"border-color": "#4CAF50", "border-width": 3}},
    {"selector": ".error",   "style": {"border-color": "#F44336", "border-width": 4}},
    {"selector": ".skipped", "style": {"opacity": 0.4}},
    {
        "selector": "edge",
        "style": {
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "target-arrow-color": "#90A4AE",
            "line-color": "#90A4AE",
            "width": 2,
            "label": "data(label)",
            "font-size": "10px",
            "color": "#455A64",
            "text-background-color": "#fff",
            "text-background-opacity": 0.8,
            "text-background-padding": "3px",
        },
    },
    {
        "selector": ".conditional-edge",
        "style": {
            "line-style": "dashed",
            "line-color": "#FF9800",
            "target-arrow-color": "#FF9800",
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# App Dash
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Actuarial Canvas",
    suppress_callback_exceptions=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — onglet Canvas
# ─────────────────────────────────────────────────────────────────────────────
def _palette_card(nb_name: str, nb_path: str, color: str) -> html.Div:
    label = nb_name.replace(".ipynb", "").replace("_", " ")
    return html.Div(
        label,
        id={"type": "palette-item", "index": nb_name},
        className="palette-item",
        style={
            "background": color, "color": "#fff",
            "padding": "8px 12px", "borderRadius": "6px",
            "marginBottom": "8px", "cursor": "grab",
            "fontSize": "12px", "fontWeight": "bold",
            "border": "2px solid rgba(255,255,255,0.2)",
            "userSelect": "none",
        },
        **{"data-notebook": nb_path, "data-label": label, "data-color": color},
    )


def _build_palette() -> list:
    nb_dir = Path(config.NOTEBOOKS_DIR)
    items = []
    for i, p in enumerate(sorted(nb_dir.glob("*.ipynb"))):
        color = PALETTE[i % len(PALETTE)]
        items.append(_palette_card(p.name, str(p), color))
    if not items:
        items.append(html.P("Aucun notebook trouvé", style={"color": "#aaa", "fontSize": "12px"}))
    return items


def _sidebar() -> dbc.Col:
    return dbc.Col(
        [
            html.H5("Notebooks", className="text-dark mb-3",
                    style={"fontSize": "14px", "fontWeight": "bold"}),
            html.P("Cliquez pour ajouter au canvas",
                   style={"color": "#777", "fontSize": "11px", "marginBottom": "12px"}),
            html.Div(_build_palette(), id="palette-container"),
            html.Hr(style={"borderColor": "#C5BDB0"}),
            html.H5("Workflow", className="text-dark mb-2",
                    style={"fontSize": "14px", "fontWeight": "bold"}),
            dbc.Button("💾 Sauvegarder", id="btn-save", color="secondary", size="sm",
                       className="w-100 mb-2"),
            dbc.Button("📂 Charger", id="btn-load-open", color="secondary", size="sm",
                       className="w-100 mb-2"),
            dbc.Button("🔄 Réinitialiser", id="btn-reset", color="warning", size="sm",
                       className="w-100 mb-2", outline=True),
            html.Hr(style={"borderColor": "#444"}),
            html.H5("Exécution", className="text-light mb-2",
                    style={"fontSize": "14px", "fontWeight": "bold"}),
            dcc.Upload(
                id="upload-csv",
                children=html.Div(["📁 Charger CSV", html.Br(),
                                   html.Small("(glisser-déposer)", style={"color": "#777"})]),
                style={
                    "border": "2px dashed #A09890", "borderRadius": "8px",
                    "padding": "12px", "textAlign": "center",
                    "color": "#555", "fontSize": "12px",
                    "cursor": "pointer", "marginBottom": "8px",
                },
                multiple=False,
            ),
            html.Div(id="csv-filename",
                     style={"color": "#777", "fontSize": "11px", "marginBottom": "8px"}),
            dbc.Button("▶ Lancer l'analyse", id="btn-run", color="success", size="sm",
                       className="w-100 mb-2", disabled=True),
            html.Div(id="run-status", style={"fontSize": "11px", "color": "#777"}),
            html.Hr(style={"borderColor": "#C5BDB0"}),
            html.H5("Rapport Word", className="text-dark mb-2",
                    style={"fontSize": "14px", "fontWeight": "bold"}),
            dcc.Upload(
                id="upload-template",
                children=html.Div(["📄 Template .docx", html.Br(),
                                   html.Small("(optionnel)", style={"color": "#777"})]),
                style={
                    "border": "2px dashed #A09890", "borderRadius": "8px",
                    "padding": "10px", "textAlign": "center",
                    "color": "#555", "fontSize": "12px",
                    "cursor": "pointer", "marginBottom": "6px",
                },
                accept=".docx",
                multiple=False,
            ),
            html.Div(id="template-filename",
                     style={"color": "#777", "fontSize": "11px", "marginBottom": "6px"}),
            dbc.Button("📄 Générer rapport Word", id="btn-word", color="info", size="sm",
                       className="w-100", disabled=True),
            html.Hr(style={"borderColor": "#C5BDB0"}),
            html.H5("Qualité", className="text-dark mb-2",
                    style={"fontSize": "14px", "fontWeight": "bold"}),
            dbc.Button("🔍 Valider le pipeline", id="btn-validate", color="secondary",
                       size="sm", className="w-100 mb-1", outline=True),
            html.Div(id="validate-status",
                     style={"fontSize": "10px", "color": "#777", "marginTop": "4px"}),
        ],
        width=2,
        style={
            "background": "#F0EDE3", "padding": "16px",
            "height": "100%", "overflowY": "auto",
            "borderRight": "1px solid #C5BDB0",
        },
    )


def _canvas_panel() -> dbc.Col:
    wf = default_workflow(config.NOTEBOOKS_DIR)
    return dbc.Col(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div([
                            dbc.Button("✖ Supprimer nœud", id="btn-delete-node",
                                       color="danger", size="sm", className="me-2", outline=True),
                            dbc.Button("🔗 Connecter", id="btn-connect-hint",
                                       color="info", size="sm", className="me-2", outline=True),
                            dbc.Badge("Layout:", color="secondary", className="me-1"),
                            dcc.Dropdown(
                                id="layout-dropdown",
                                options=[
                                    {"label": "Libre", "value": "preset"},
                                    {"label": "Horizontal (dagre)", "value": "dagre"},
                                    {"label": "Grille", "value": "grid"},
                                    {"label": "Cercle", "value": "circle"},
                                ],
                                value="preset",
                                clearable=False,
                                style={"width": "140px", "display": "inline-block",
                                       "fontSize": "12px"},
                            ),
                        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
                        width=12,
                    ),
                ],
                className="mb-2",
                style={"padding": "8px 16px", "background": "#E6E2D6", "borderRadius": "6px"},
            ),
            cyto.Cytoscape(
                id="canvas",
                layout={"name": "preset"},
                style={"width": "100%", "height": "calc(100vh - 160px)",
                       "background": "#F5F2E7"},
                elements=wf.to_cytoscape_elements(),
                stylesheet=CYTO_STYLESHEET,
                boxSelectionEnabled=True,
                autoungrabify=False,
                userZoomingEnabled=True,
                userPanningEnabled=True,
                minZoom=0.3,
                maxZoom=2.5,
            ),
        ],
        width=7,
        style={"padding": "12px", "background": "#FBF8F1"},
    )


def _properties_panel() -> dbc.Col:
    return dbc.Col(
        [
            html.H5("Propriétés", className="text-dark mb-3",
                    style={"fontSize": "14px", "fontWeight": "bold"}),
            html.Div(id="properties-panel", children=[
                html.P("Sélectionnez un nœud ou une arête.",
                       style={"color": "#777", "fontSize": "12px"}),
            ]),
            html.Hr(style={"borderColor": "#C5BDB0"}),
            html.H5("Résultats", className="text-dark mb-2",
                    style={"fontSize": "14px", "fontWeight": "bold"}),
            html.Div(id="results-panel", style={"overflowY": "auto", "maxHeight": "60vh"}),
        ],
        width=3,
        style={
            "background": "#F0EDE3", "padding": "16px",
            "height": "100%", "overflowY": "auto",
            "borderLeft": "1px solid #C5BDB0",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — onglet Notebooks
# ─────────────────────────────────────────────────────────────────────────────
def _build_nb_list() -> list:
    """Construit la liste des fichiers affichables dans l'onglet Notebooks.

    Inclut :
      - les notebooks .ipynb du répertoire NOTEBOOKS_DIR
      - les modules actuariels .py du répertoire NOTEBOOKS_DIR
      - les fichiers de configuration .py à la racine du projet
        (actuarial_params.py, smoothing_selector.py)

    L'index de chaque item est le chemin relatif depuis la racine du projet
    (ex : «notebooks/01_data_preparation.py»). Le callback load_and_render_notebook
    reconstruit le chemin absolu via Path(__file__).parent / index.
    """
    _ROOT = Path(__file__).parent
    items = []
    color_idx = 0

    def _item(label: str, rel_path: str) -> html.Div:
        nonlocal color_idx
        color = PALETTE[color_idx % len(PALETTE)]
        color_idx += 1
        return html.Div(
            label,
            id={"type": "nb-list-item", "index": rel_path},
            style={
                "background": color, "color": "#fff",
                "padding": "8px 10px", "borderRadius": "6px",
                "marginBottom": "6px", "cursor": "pointer",
                "fontSize": "11px", "fontWeight": "bold",
                "border": "2px solid rgba(255,255,255,0.15)",
                "userSelect": "none",
            },
        )

    # ── Notebooks .ipynb ──────────────────────────────────────────────────────
    nb_dir = _ROOT / config.NOTEBOOKS_DIR
    ipynb_files = sorted(nb_dir.glob("*.ipynb"))
    if ipynb_files:
        items.append(
            html.P("Notebooks", style={"color": "#888", "fontSize": "10px",
                                       "fontWeight": "bold", "marginTop": "4px",
                                       "marginBottom": "4px", "textTransform": "uppercase"})
        )
        for p in ipynb_files:
            label = p.name.replace(".ipynb", "").replace("_", " ")
            rel = str(Path(config.NOTEBOOKS_DIR) / p.name)
            items.append(_item(label, rel))

    # ── Modules actuariels .py (notebooks/) ───────────────────────────────────
    py_nb_files = sorted(nb_dir.glob("[0-9][0-9]_*.py"))
    if py_nb_files:
        items.append(
            html.P("Modules actuariels", style={"color": "#888", "fontSize": "10px",
                                                "fontWeight": "bold", "marginTop": "10px",
                                                "marginBottom": "4px", "textTransform": "uppercase"})
        )
        for p in py_nb_files:
            label = p.name.replace(".py", "").replace("_", " ")
            rel = str(Path(config.NOTEBOOKS_DIR) / p.name)
            items.append(_item(label, rel))

    # ── Fichiers de configuration .py (racine) ────────────────────────────────
    _ROOT_PY_FILES = ["actuarial_params.py", "smoothing_selector.py",
                      "config.py", "actuary_logger.py"]
    root_py = [_ROOT / name for name in _ROOT_PY_FILES if (_ROOT / name).exists()]
    if root_py:
        items.append(
            html.P("Configuration", style={"color": "#888", "fontSize": "10px",
                                           "fontWeight": "bold", "marginTop": "10px",
                                           "marginBottom": "4px", "textTransform": "uppercase"})
        )
        for p in root_py:
            items.append(_item(p.name.replace(".py", ""), p.name))

    if not items:
        items.append(html.P("Aucun fichier", style={"color": "#aaa", "fontSize": "12px"}))
    return items


def _build_notebook_view(cells: list) -> list:
    """Convertit une liste de cellules en composants Dash."""
    components = []
    for i, cell in enumerate(cells):
        ctype = cell.get("type", "code")
        source = cell.get("source", "")

        if ctype == "markdown":
            components.append(
                html.Div(
                    dcc.Markdown(source or "*cellule vide*",
                                 style={"color": "#333", "fontSize": "13px"}),
                    style={
                        "background": "#EDE9DE",
                        "borderLeft": "3px solid #A09890",
                        "padding": "10px 16px",
                        "marginBottom": "10px",
                        "borderRadius": "4px",
                    },
                )
            )
        elif ctype == "code":
            n_lines = max(3, source.count("\n") + 2)
            height_px = min(400, n_lines * 18 + 24)
            components.append(
                dbc.Card(
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Badge(f"[{i}]", color="primary",
                                                  style={"fontFamily": "monospace",
                                                         "fontSize": "11px"}),
                                        width="auto", className="align-self-center",
                                    ),
                                    dbc.Col(width=True),
                                    dbc.Col(
                                        dbc.Button(
                                            "▶ Exécuter",
                                            id={"type": "btn-run-cell", "index": i},
                                            color="success", size="sm", outline=True,
                                            style={"fontSize": "11px", "padding": "2px 10px"},
                                        ),
                                        width="auto",
                                    ),
                                ],
                                className="mb-2", align="center",
                            ),
                            dcc.Textarea(
                                id={"type": "cell-textarea", "index": i},
                                value=source,
                                style={
                                    "width": "100%",
                                    "fontFamily": "'Courier New', monospace",
                                    "fontSize": "12px",
                                    "background": "#F0EDE3",
                                    "color": "#1A1A1A",
                                    "border": "1px solid #C5BDB0",
                                    "borderRadius": "4px",
                                    "minHeight": f"{height_px}px",
                                    "resize": "vertical",
                                    "padding": "8px",
                                    "lineHeight": "1.5",
                                },
                            ),
                            html.Pre(
                                "",
                                id={"type": "cell-output-pre", "index": i},
                                style={
                                    "background": "#EDEAE0",
                                    "color": "#555",
                                    "fontSize": "11px",
                                    "padding": "6px 8px",
                                    "borderRadius": "3px",
                                    "maxHeight": "200px",
                                    "overflow": "auto",
                                    "marginTop": "6px",
                                    "display": "none",
                                    "whiteSpace": "pre-wrap",
                                    "border": "1px solid #C8C0A8",
                                },
                            ),
                        ],
                        style={"padding": "10px"},
                    ),
                    style={
                        "marginBottom": "10px",
                        "background": "#F5F2E7",
                        "border": "1px solid #C5BDB0",
                    },
                )
            )
    return components


def _nb_sidebar_col() -> dbc.Col:
    return dbc.Col(
        [
            html.H5("Notebooks", className="text-dark mb-2",
                    style={"fontSize": "13px", "fontWeight": "bold"}),
            html.Div(_build_nb_list(), id="nb-file-list"),
            html.Hr(style={"borderColor": "#C5BDB0"}),
            dbc.Button("▶ Tout exécuter", id="btn-run-all-cells", color="success", size="sm",
                       className="w-100 mb-2", disabled=True),
            dbc.Button("💾 Sauvegarder", id="btn-save-notebook", color="secondary", size="sm",
                       className="w-100 mb-2", disabled=True),
            dbc.Button("🔄 Reset kernel", id="btn-reset-kernel", color="warning", size="sm",
                       className="w-100 mb-2", outline=True, disabled=True),
            html.Div(id="nb-save-status",
                     style={"fontSize": "11px", "color": "#777", "marginTop": "6px"}),
        ],
        width=2,
        style={
            "background": "#F0EDE3", "padding": "14px",
            "height": "100%", "overflowY": "auto",
            "borderRight": "1px solid #C5BDB0",
        },
    )


def _nb_viewer_col() -> dbc.Col:
    return dbc.Col(
        [
            html.Div(
                id="nb-viewer-area",
                children=[
                    html.P(
                        "← Sélectionnez un notebook dans la liste pour le visualiser et l'éditer.",
                        style={"color": "#888", "fontSize": "13px",
                               "textAlign": "center", "marginTop": "60px"},
                    )
                ],
                style={"height": "calc(100vh - 100px)", "overflowY": "auto", "padding": "12px"},
            ),
        ],
        width=10,
        style={"background": "#FBF8F1", "padding": "8px"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — onglet RAG
# ─────────────────────────────────────────────────────────────────────────────
def _rag_tab(h_offset: int = 0) -> html.Div:
    total_h = f"calc(100vh - {88 + h_offset}px)"
    return html.Div(
        [
            # Éléments cachés — gardés pour les callbacks
            html.Div(
                [
                    html.Div(id="rag-prompt-save-status"),
                    dcc.Textarea(id="rag-system-prompt-textarea", value=RAG_SYSTEM_PROMPT),
                    dbc.Button(id="btn-rag-apply-prompt", n_clicks=0),
                    dbc.Button(id="btn-rag-reset-prompt", n_clicks=0),
                ],
                id="rag-sub-tabs",
                style={"display": "none"},
            ),
            # ── Chat RAG — flex column pour que l'input reste toujours visible ──
            html.Div(
                [
                    # En-tête
                    dbc.Row([
                        dbc.Col(
                            html.Span("💬 RAG",
                                      style={"fontSize": "13px", "fontWeight": "bold",
                                             "color": "#2D2D2D"}),
                            width=True, className="align-self-center",
                        ),
                        dbc.Col(
                            dbc.ButtonGroup([
                                dbc.Button("⟺", id="btn-expand-chat",
                                           color="secondary", size="sm", outline=True,
                                           title="Élargir/réduire"),
                                dbc.Button("🗑", id="btn-rag-clear",
                                           color="secondary", size="sm", outline=True,
                                           title="Effacer la conversation"),
                            ]),
                            width="auto",
                        ),
                    ], className="mb-1", align="center", style={"flexShrink": "0"}),
                    # Upload PDF
                    dcc.Upload(
                        id="upload-rag-pdf",
                        children=html.Div([
                            "📄 PDF ",
                            html.Small("(contexte RAG)", style={"color": "#888"}),
                        ]),
                        multiple=True,
                        accept=".pdf",
                        style={
                            "border": "1px dashed #A09890", "borderRadius": "6px",
                            "padding": "4px 8px", "textAlign": "center",
                            "color": "#555", "fontSize": "11px",
                            "cursor": "pointer", "marginBottom": "3px",
                            "background": "#F5F2E7", "flexShrink": "0",
                        },
                    ),
                    html.Div(id="rag-pdf-status",
                             style={"fontSize": "10px", "color": "#888",
                                    "marginBottom": "3px", "minHeight": "14px",
                                    "flexShrink": "0"}),
                    # Messages — flex:1 pour remplir l'espace disponible
                    dcc.Loading(
                        html.Div(
                            id="rag-chat-messages",
                            children=[
                                html.Div(
                                    "Lancez une analyse puis posez vos questions.",
                                    style={"color": "#999", "fontSize": "12px",
                                           "textAlign": "center", "marginTop": "40px"},
                                )
                            ],
                            style={
                                "overflowY": "auto",
                                "flex": "1",
                                "minHeight": "80px",
                                "padding": "4px 2px",
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "8px",
                            },
                        ),
                        type="circle",
                        color="#6C757D",
                    ),
                    # Barre de saisie — toujours en bas
                    html.Hr(style={"margin": "4px 0", "borderColor": "#C5BDB0",
                                   "flexShrink": "0"}),
                    dbc.Row([
                        dbc.Col(
                            dcc.Input(
                                id="rag-chat-input",
                                type="text",
                                placeholder="Posez une question… (Entrée)",
                                debounce=False,
                                n_submit=0,
                                style={
                                    "width": "100%", "height": "38px",
                                    "fontSize": "12px", "fontFamily": "inherit",
                                    "background": "#F5F2E7", "color": "#2D2D2D",
                                    "border": "1px solid #C5BDB0", "borderRadius": "6px",
                                    "padding": "6px 10px",
                                },
                            ),
                            width=True,
                        ),
                        dbc.Col(
                            dbc.Button("↵", id="btn-rag-send",
                                       color="primary", size="sm",
                                       style={"height": "38px", "width": "44px",
                                              "fontSize": "15px"}),
                            width="auto", className="align-self-center",
                        ),
                    ], className="g-1", style={"flexShrink": "0"}),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "height": total_h,
                    "padding": "10px 12px",
                    "overflow": "hidden",
                },
            ),
        ],
        style={"height": total_h, "background": "#FBF8F1"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — onglet Analyse Rapport
# ─────────────────────────────────────────────────────────────────────────────
def _report_template_tab() -> dbc.Row:
    """Onglet d'analyse offline d'un rapport de référence."""
    _card_style = {
        "background": "#FBF8F1", "border": "1px solid #D8D0C4",
        "borderRadius": "8px", "padding": "14px", "marginBottom": "10px",
    }
    _label_style = {"fontSize": "11px", "color": "#777", "marginBottom": "4px"}

    left_col = dbc.Col(
        [
            html.H5("📋 Analyse de rapport",
                    style={"fontSize": "14px", "fontWeight": "bold",
                           "color": "#2D2D2D", "marginBottom": "12px"}),
            # Upload PDF
            html.Div([
                html.P("Rapport PDF de référence", style=_label_style),
                dcc.Upload(
                    id="upload-report-pdf",
                    children=html.Div([
                        "📄 Charger un PDF",
                        html.Br(),
                        html.Small("(glisser-déposer)", style={"color": "#888"}),
                    ]),
                    multiple=False,
                    accept=".pdf",
                    style={
                        "border": "2px dashed #A09890", "borderRadius": "8px",
                        "padding": "12px", "textAlign": "center",
                        "color": "#555", "fontSize": "12px",
                        "cursor": "pointer", "marginBottom": "6px",
                    },
                ),
                html.Div(id="report-pdf-filename",
                         style={"color": "#4CAF50", "fontSize": "11px",
                                "marginBottom": "8px"}),
            ], style=_card_style),
            # Bouton analyser
            dbc.Button(
                "🔍 Analyser le rapport",
                id="btn-analyze-report",
                color="primary", size="sm",
                className="w-100 mb-2",
                disabled=True,
            ),
            # Statut / progression
            html.Div(
                id="template-analysis-status",
                style={"fontSize": "11px", "color": "#777",
                       "marginTop": "6px", "minHeight": "40px"},
            ),
            html.Hr(style={"borderColor": "#C5BDB0", "marginTop": "16px"}),
            # Charger un JSON existant
            html.Div([
                html.P("Ou charger un template existant (.json)", style=_label_style),
                dcc.Upload(
                    id="upload-report-template-json",
                    children=html.Div([
                        "📂 Charger JSON",
                        html.Br(),
                        html.Small("(template déjà analysé)", style={"color": "#888"}),
                    ]),
                    multiple=False,
                    accept=".json",
                    style={
                        "border": "1px dashed #A09890", "borderRadius": "6px",
                        "padding": "8px", "textAlign": "center",
                        "color": "#555", "fontSize": "12px",
                        "cursor": "pointer",
                    },
                ),
            ], style=_card_style),
        ],
        width=3,
        style={
            "background": "#F0EDE3", "padding": "16px",
            "height": "calc(100vh - 88px)", "overflowY": "auto",
            "borderRight": "1px solid #C5BDB0",
        },
    )

    right_col = dbc.Col(
        [
            dbc.Row([
                dbc.Col(
                    html.H5("Structure extraite",
                            style={"fontSize": "14px", "fontWeight": "bold",
                                   "color": "#2D2D2D", "marginBottom": "0"}),
                    width=True, className="align-self-center",
                ),
                dbc.Col(
                    dbc.ButtonGroup([
                        dbc.Button("💾 Sauvegarder JSON",
                                   id="btn-download-template",
                                   color="success", size="sm", outline=True,
                                   disabled=True),
                        dbc.Button("📤 Envoyer à l'Agent",
                                   id="btn-send-template-to-agent",
                                   color="primary", size="sm",
                                   disabled=True,
                                   title="Pré-remplit le system prompt de l'Agent"),
                    ]),
                    width="auto",
                ),
            ], className="mb-3", align="center"),
            html.Div(
                id="template-analysis-result",
                children=[
                    html.P(
                        "Chargez un rapport PDF et cliquez sur 'Analyser' "
                        "pour extraire la structure du rapport.",
                        style={"color": "#999", "fontSize": "13px",
                               "textAlign": "center", "marginTop": "60px"},
                    )
                ],
                style={
                    "overflowY": "auto",
                    "height": "calc(100vh - 165px)",
                    "padding": "4px 8px",
                },
            ),
        ],
        width=9,
        style={
            "background": "#FBF8F1", "padding": "16px",
            "height": "calc(100vh - 88px)", "overflowY": "hidden",
        },
    )

    return dbc.Row(
        [left_col, right_col],
        className="g-0",
        style={"height": "calc(100vh - 88px)"},
    )




def _notebook_tab(h_offset: int = 0) -> html.Div:
    """Onglet Notebook — Jupyter Notebook intégré dans l'app."""
    h = 88 + h_offset
    _initial_nb_options = [{"label": d["label"], "value": d["value"]} for d in _list_notebooks()]
    return html.Div(
        [
            # ── Barre d'actions ───────────────────────────────────────────────
            dbc.Row([
                dbc.Col(
                    dbc.Button("📓 Générer depuis agent", id="btn-nb-generate",
                               size="sm", color="warning", outline=False,
                               title="Génère le notebook .ipynb depuis les étapes de l'agent"),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button("⬇ Télécharger", id="btn-nb-download",
                               size="sm", color="secondary", outline=True,
                               style={"display": "none"}),
                    width="auto",
                ),
            ], className="g-1 px-2 pt-2 mb-1", align="center"),
            html.Div(id="nb-gen-status",
                     style={"fontSize": "10px", "color": "#777",
                            "padding": "0 8px 4px 8px", "minHeight": "14px"}),
            dcc.Download(id="nb-download-ipynb"),
            # ── Sélecteur de notebook ─────────────────────────────────────────
            html.Div([
                dcc.Dropdown(
                    id="nb-picker",
                    options=_initial_nb_options,
                    placeholder="Ouvrir un notebook...",
                    clearable=True,
                    style={"fontSize": "11px", "flex": "1"},
                ),
                dbc.Button("▶ Aperçu", id="btn-nb-open",
                           size="sm", color="primary", outline=True,
                           style={"marginLeft": "4px", "whiteSpace": "nowrap",
                                  "fontSize": "11px"}),
                dbc.Button("🚀 Jupyter", id="btn-nb-launch",
                           size="sm", color="success", outline=True,
                           style={"marginLeft": "4px", "whiteSpace": "nowrap",
                                  "fontSize": "11px"}),
            ], style={"padding": "0 8px 4px 8px", "display": "flex",
                      "alignItems": "center"}),
            # ── iframe Jupyter (HTML statique via nbconvert) ──────────────────
            html.Iframe(
                id="nb-jupyter-iframe",
                srcDoc="",
                style={
                    "width": "100%",
                    "height": f"calc(100vh - {h + 112}px)",
                    "border": "none",
                    "background": "#F5F2E7",
                    "display": "block",
                },
            ),
            # ── Éléments cachés (compatibilité callbacks) ─────────────────────
            html.Div([
                html.A(id="nb-jupyter-link", href="#", target="_blank"),
                dcc.Store(id="nb-jupyter-url-store", data=""),
                dcc.Store(id="nb-agent-cells-data", data=[]),
                html.Div(id="nb-agent-cells-area"),
                html.Div(id="nb-inner-tabs"),
                dbc.Button(id="btn-nb-load", n_clicks=0),
                dbc.Button(id="btn-nb-replay", n_clicks=0),
                dbc.Button(id="btn-nb-add", n_clicks=0),
            ], style={"display": "none"}),
        ],
        style={"height": f"calc(100vh - {h}px)", "overflowY": "hidden",
               "background": "#FBF8F1"},
    )


def _agent_tab() -> html.Div:
    """Onglet Agent — layout chat-centric 3 colonnes."""
    _card_style = {
        "background": "#FBF8F1", "border": "1px solid #D8D0C4",
        "borderRadius": "8px", "padding": "10px", "marginBottom": "8px",
    }
    _label_style = {"fontSize": "11px", "color": "#777", "marginBottom": "3px"}

    # ── Colonne gauche : Config ───────────────────────────────────────────────
    left_col = html.Div(
        [
            html.H5("Agent", style={"fontSize": "13px", "fontWeight": "bold",
                                    "color": "#2D2D2D", "marginBottom": "8px"}),
            # Upload CSV
            html.P("Fichier CSV", style=_label_style),
            dcc.Upload(
                id="upload-csv-agent",
                children=html.Div(["📁 CSV", html.Br(),
                                   html.Small("(glisser)", style={"color": "#888"})]),
                style={
                    "border": "2px dashed #A09890", "borderRadius": "8px",
                    "padding": "8px", "textAlign": "center",
                    "color": "#555", "fontSize": "11px",
                    "cursor": "pointer", "marginBottom": "4px",
                },
                multiple=False,
            ),
            html.Div(id="csv-filename-agent",
                     style={"color": "#4CAF50", "fontSize": "10px",
                            "marginBottom": "6px"}),
            # Sexe
            html.P("Sexe", style=_label_style),
            dcc.Dropdown(
                id="agent-sexe-select",
                options=[{"label": "Hommes (H)", "value": "H"},
                         {"label": "Femmes (F)", "value": "F"}],
                value="H", clearable=False,
                style={"fontSize": "11px", "marginBottom": "8px"},
            ),
            # Instruction
            html.P("Instruction", style=_label_style),
            dcc.Textarea(
                id="agent-user-message",
                value="Construis la table de mortalité d'expérience pour le fichier fourni.",
                style={
                    "width": "100%", "height": "80px",
                    "fontSize": "11px", "fontFamily": "inherit",
                    "background": "#F5F2E7", "color": "#2D2D2D",
                    "border": "1px solid #C5BDB0", "borderRadius": "4px",
                    "padding": "6px", "resize": "vertical",
                    "marginBottom": "8px",
                },
            ),
            # Pas-à-pas
            dbc.Switch(
                id="toggle-stepbystep",
                label="Pas-à-pas",
                value=False,
                style={"fontSize": "11px", "marginBottom": "6px"},
            ),
            # Boutons exécution
            dbc.Button("🤖 Lancer", id="btn-run-agent",
                       color="primary", size="sm", className="w-100 mb-1",
                       disabled=True),
            dbc.Button("⏹ Arrêter", id="btn-stop-agent",
                       color="danger", size="sm", className="w-100 mb-1",
                       outline=True, disabled=True),
            dbc.Button("🗓 Plan + Agent", id="btn-plan-execute",
                       color="info", size="sm", className="w-100 mb-1",
                       outline=True, disabled=True,
                       title="Exécute l'agent en suivant le plan du Canvas"),
            dbc.Button("⚙ System Prompt", id="btn-show-prompt",
                       color="secondary", size="sm", className="w-100 mb-1",
                       outline=True),
            html.Div(id="agent-run-status",
                     style={"fontSize": "10px", "color": "#777",
                            "marginTop": "4px", "marginBottom": "6px"}),
            html.Hr(style={"borderColor": "#C5BDB0", "margin": "6px 0"}),
            # Template rapport (upload .json)
            html.P("Template rapport (.json)", style=_label_style),
            dcc.Upload(
                id="upload-agent-template",
                children=html.Div([
                    "📋 Charger .json",
                    html.Br(),
                    html.Small("(pré-remplit le prompt)", style={"color": "#888"}),
                ]),
                multiple=False,
                accept=".json",
                style={
                    "border": "1px dashed #A09890", "borderRadius": "6px",
                    "padding": "6px", "textAlign": "center",
                    "color": "#555", "fontSize": "10px",
                    "cursor": "pointer", "marginBottom": "3px",
                },
            ),
            html.Div(id="agent-template-status",
                     style={"color": "#4CAF50", "fontSize": "10px",
                            "marginBottom": "3px", "minHeight": "14px"}),
            dbc.Button("✕ Effacer template", id="btn-clear-agent-template",
                       color="secondary", size="sm", outline=True,
                       className="w-100 mb-1",
                       style={"fontSize": "10px"}, disabled=True),
            html.Hr(style={"borderColor": "#C5BDB0", "margin": "6px 0"}),
            # Section Analyse Rapport dépliable
            dbc.Button(
                "📋 Analyse rapport ▾",
                id="btn-toggle-report-section",
                color="secondary", size="sm", outline=True,
                className="w-100 mb-1",
                style={"fontSize": "10px"},
            ),
            dbc.Collapse(
                html.Div([
                    html.P("Rapport PDF", style={**_label_style, "marginTop": "6px"}),
                    dcc.Upload(
                        id="upload-report-pdf",
                        children=html.Div([
                            "📄 Charger PDF",
                            html.Br(),
                            html.Small("(glisser)", style={"color": "#888"}),
                        ]),
                        multiple=False,
                        accept=".pdf",
                        style={
                            "border": "1px dashed #A09890", "borderRadius": "6px",
                            "padding": "6px", "textAlign": "center",
                            "color": "#555", "fontSize": "10px",
                            "cursor": "pointer", "marginBottom": "3px",
                        },
                    ),
                    html.Div(id="report-pdf-filename",
                             style={"color": "#4CAF50", "fontSize": "10px",
                                    "marginBottom": "4px"}),
                    dbc.Button(
                        "🔍 Analyser",
                        id="btn-analyze-report",
                        color="primary", size="sm",
                        className="w-100 mb-1",
                        disabled=True,
                    ),
                    html.Div(
                        id="template-analysis-status",
                        style={"fontSize": "10px", "color": "#777",
                               "marginBottom": "4px", "minHeight": "30px"},
                    ),
                    html.P("Ou charger JSON existant", style=_label_style),
                    dcc.Upload(
                        id="upload-report-template-json",
                        children=html.Div([
                            "📂 JSON",
                            html.Br(),
                            html.Small("(template analysé)", style={"color": "#888"}),
                        ]),
                        multiple=False,
                        accept=".json",
                        style={
                            "border": "1px dashed #A09890", "borderRadius": "6px",
                            "padding": "6px", "textAlign": "center",
                            "color": "#555", "fontSize": "10px",
                            "cursor": "pointer", "marginBottom": "3px",
                        },
                    ),
                    dbc.ButtonGroup([
                        dbc.Button(
                            "📤 → Agent",
                            id="btn-send-template-to-agent",
                            color="success", size="sm",
                            disabled=True,
                        ),
                        dbc.Button(
                            "💾 JSON",
                            id="btn-download-template",
                            color="secondary", size="sm", outline=True,
                            disabled=True,
                        ),
                    ], className="w-100 mb-1"),
                    html.Div(
                        id="template-analysis-result",
                        style={
                            "maxHeight": "200px", "overflowY": "auto",
                            "fontSize": "10px", "background": "#F5F2E7",
                            "border": "1px solid #D8D0C4", "borderRadius": "4px",
                            "padding": "6px", "marginTop": "4px",
                        },
                    ),
                ]),
                id="collapse-report-section",
                is_open=False,
            ),
        ],
        id="agent-config-col",
        style={
            "background": "#F0EDE3", "padding": "12px",
            "height": "calc(100vh - 88px)", "overflowY": "auto",
            "borderRight": "1px solid #C5BDB0",
            "width": "16.666%",
            "minWidth": "160px",
            "flexShrink": "0",
        },
    )

    # ── Colonne centrale : Conversation ─────────────────────────────────────
    center_col = html.Div(
        [
            html.H5("Résultats Agent",
                    style={"fontSize": "13px", "fontWeight": "bold",
                           "color": "#2D2D2D", "marginBottom": "6px",
                           "paddingTop": "6px", "paddingLeft": "8px"}),
            html.Div(
                id="agent-results-panel",
                children=[
                    html.P(
                        "Configurez et lancez l'agent.",
                        style={"color": "#888", "fontSize": "12px",
                               "textAlign": "center", "marginTop": "60px"},
                    )
                ],
                style={
                    "overflowY": "auto",
                    "height": "calc(100vh - 145px)",
                    "padding": "0 8px 8px 8px",
                },
            ),
        ],
        id="agent-conv-col",
        style={
            "background": "#FBF8F1", "padding": "0",
            "borderRight": "1px solid #C5BDB0",
            "flex": "1",
            "minWidth": "300px",
            "overflow": "hidden",
            "height": "calc(100vh - 88px)",
        },
    )

    # ── Colonne droite : Outils RAG + Notebook (w=3) ─────────────────────────
    right_col = html.Div(
        dbc.Tabs(
            [
                dbc.Tab(
                    _rag_tab(h_offset=42),
                    label="💬 RAG",
                    tab_id="tools-rag",
                ),
                dbc.Tab(
                    _notebook_tab(h_offset=42),
                    label="📓 Notebook",
                    tab_id="tools-notebook",
                ),
            ],
            id="tools-sub-tabs",
            active_tab="tools-rag",
            style={"background": "#FBF8F1"},
        ),
        id="agent-tools-col",
        style={
            "background": "#FBF8F1",
            "width": "25%",
            "minWidth": "200px",
            "maxWidth": "60%",
            "flexShrink": "0",
            "height": "calc(100vh - 88px)",
            "overflow": "hidden",
        },
    )

    # Drag handle entre colonne centre et colonne outils
    drag_handle = html.Div(
        id="agent-resize-handle",
        style={
            "width": "5px",
            "cursor": "col-resize",
            "background": "#C5BDB0",
            "flexShrink": "0",
            "height": "calc(100vh - 88px)",
            "transition": "background 0.2s",
        },
    )

    return html.Div(
        [left_col, center_col, drag_handle, right_col],
        style={
            "display": "flex",
            "flexDirection": "row",
            "height": "calc(100vh - 88px)",
            "background": "#FBF8F1",
            "overflow": "hidden",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Layout principal
# ─────────────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    [
        # Header
        dbc.Navbar(
            dbc.Container(
                [
                    html.Span("⚙ Actuarial Canvas", className="navbar-brand",
                              style={"fontWeight": "bold", "fontSize": "16px"}),
                    html.Span("Orchestration visuelle de notebooks actuariels",
                              style={"color": "#aaa", "fontSize": "12px"}),
                ],
                fluid=True,
            ),
            color="light",
            dark=False,
            style={"borderBottom": "1px solid #C5BDB0", "background": "#F0EDE3", "padding": "6px 16px"},
        ),
        # Corps — onglets
        dbc.Tabs(
            [
                dbc.Tab(
                    dbc.Row(
                        [_sidebar(), _canvas_panel(), _properties_panel()],
                        className="g-0",
                        style={"height": "calc(100vh - 88px)"},
                    ),
                    label="⚙ Canvas",
                    tab_id="tab-canvas",
                ),
                dbc.Tab(
                    dbc.Row(
                        [_nb_sidebar_col(), _nb_viewer_col()],
                        className="g-0",
                        style={"height": "calc(100vh - 88px)"},
                    ),
                    label="📓 Dev",
                    tab_id="tab-notebooks",
                ),
                dbc.Tab(
                    _agent_tab(),
                    label="🤖 Agent",
                    tab_id="tab-agent",
                ),
            ],
            id="main-tabs",
            active_tab="tab-canvas",
            style={"background": "#FBF8F1"},
        ),
        # Stores
        dcc.Store(id="workflow-store", data=default_workflow(config.NOTEBOOKS_DIR).to_dict()),
        dcc.Store(id="csv-path-store", data=None),
        dcc.Store(id="execution-results-store", data={}),
        dcc.Store(id="selected-element-store", data=None),
        dcc.Store(id="template-path-store", data=None),
        dcc.Store(id="nb-current-path-store", data=None),
        dcc.Store(id="nb-cells-store", data=[]),
        dcc.Store(id="system-prompt-store", data=SYSTEM_PROMPT_TEMPLATE),
        dcc.Store(id="csv-path-agent-store", data=None),
        dcc.Store(id="rag-history-store", data=[]),
        dcc.Store(id="rag-system-prompt-store", data=RAG_SYSTEM_PROMPT),
        dcc.Store(id="rag-expanded-store", data=False),
        dcc.Store(id="report-template-store", data=None),
        dcc.Store(id="nb-gen-path-store", data=None),
        dcc.Store(id="resize-init-store", data=0),
        dcc.Download(id="download-word"),
        dcc.Download(id="download-template-json"),
        dcc.Interval(id="agent-interval", interval=800, disabled=True, n_intervals=0),
        # Interval pour rafraîchissement asynchrone
        dcc.Interval(id="exec-interval", interval=1000, disabled=True, n_intervals=0),
        dcc.Interval(id="template-analysis-interval", interval=1000, disabled=True, n_intervals=0),
        # Modal connexion
        dbc.Modal(
            [
                dbc.ModalHeader("Ajouter une connexion"),
                dbc.ModalBody([
                    dbc.Label("Nœud source"),
                    dcc.Dropdown(id="edge-source-select", placeholder="Source…"),
                    html.Br(),
                    dbc.Label("Nœud cible"),
                    dcc.Dropdown(id="edge-target-select", placeholder="Cible…"),
                    html.Br(),
                    dbc.Label("Condition (optionnelle)", id="condition-help"),
                    dbc.Input(id="edge-condition-input", placeholder='ex: SMR > 1.2',
                              style={"fontFamily": "monospace"}),
                    dbc.FormText("Laissez vide pour une connexion inconditionnelle. "
                                 "Variables disponibles : SMR, n_vides, non_mono…",
                                 color="muted"),
                ]),
                dbc.ModalFooter([
                    dbc.Button("Annuler", id="btn-edge-cancel", color="secondary",
                               className="me-2"),
                    dbc.Button("Connecter", id="btn-edge-confirm", color="primary"),
                ]),
            ],
            id="edge-modal",
            is_open=False,
        ),
        # Modal propriétés arête
        dbc.Modal(
            [
                dbc.ModalHeader("Modifier la connexion"),
                dbc.ModalBody([
                    dbc.Label("Condition"),
                    dbc.Input(id="edit-condition-input", placeholder="SMR > 1.2",
                              style={"fontFamily": "monospace"}),
                    dbc.FormText("Laissez vide pour une arête inconditionnelle."),
                ]),
                dbc.ModalFooter([
                    dbc.Button("Supprimer la connexion", id="btn-delete-edge",
                               color="danger", className="me-2"),
                    dbc.Button("Annuler", id="btn-edge-edit-cancel", color="secondary",
                               className="me-2"),
                    dbc.Button("Enregistrer", id="btn-edge-edit-save", color="primary"),
                ]),
            ],
            id="edge-edit-modal",
            is_open=False,
        ),
        # Modal System Prompt
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("⚙ System Prompt")),
                dbc.ModalBody([
                    dbc.Row([
                        dbc.Col(width=True),
                        dbc.Col(
                            dbc.ButtonGroup([
                                dbc.Button("✓ Appliquer", id="btn-apply-prompt",
                                           color="success", size="sm"),
                                dbc.Button("↺ Réinitialiser", id="btn-reset-prompt",
                                           color="warning", size="sm", outline=True),
                            ]),
                            width="auto",
                        ),
                    ], className="mb-2", align="center"),
                    html.Div(id="prompt-save-status",
                             style={"fontSize": "11px", "color": "#777",
                                    "marginBottom": "6px"}),
                    dcc.Textarea(
                        id="system-prompt-textarea",
                        value=SYSTEM_PROMPT_TEMPLATE,
                        style={
                            "width": "100%",
                            "height": "60vh",
                            "fontFamily": "'Courier New', monospace",
                            "fontSize": "11px",
                            "background": "#F5F2E7",
                            "color": "#1A1A1A",
                            "border": "1px solid #C5BDB0",
                            "borderRadius": "4px",
                            "padding": "10px",
                            "resize": "vertical",
                            "lineHeight": "1.5",
                        },
                    ),
                ]),
                dbc.ModalFooter(
                    dbc.Button("Fermer", id="btn-close-prompt-modal",
                               color="secondary", size="sm")
                ),
            ],
            id="modal-system-prompt",
            is_open=False,
            size="xl",
            scrollable=True,
        ),
        # Modal validation pipeline
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("🔍 Validation du pipeline")),
                dbc.ModalBody(
                    html.Div(id="validate-results-body",
                             style={"fontFamily": "monospace", "fontSize": "12px"}),
                ),
                dbc.ModalFooter(
                    dbc.Button("Fermer", id="btn-validate-close", color="secondary")
                ),
            ],
            id="validate-modal",
            is_open=False,
            size="lg",
            scrollable=True,
        ),
        dcc.Interval(id="validate-interval", interval=1500, disabled=True, n_intervals=0),
        # Toasts
        html.Div(id="toast-container",
                 style={"position": "fixed", "top": 60, "right": 20, "zIndex": 9999}),
    ],
    style={"background": "#FBF8F1", "minHeight": "100vh"},
)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Canvas : ajout de nœud
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("canvas", "elements"),
    Output("workflow-store", "data"),
    Output("edge-edit-modal", "is_open", allow_duplicate=True),
    Input({"type": "palette-item", "index": dash.ALL}, "n_clicks"),
    Input("btn-delete-node", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    Input("btn-edge-confirm", "n_clicks"),
    Input("btn-delete-edge", "n_clicks"),
    Input("btn-edge-edit-save", "n_clicks"),
    Input("layout-dropdown", "value"),
    State("canvas", "elements"),
    State("canvas", "selectedNodeData"),
    State("canvas", "selectedEdgeData"),
    State("workflow-store", "data"),
    State({"type": "palette-item", "index": dash.ALL}, "id"),
    State({"type": "palette-item", "index": dash.ALL}, "children"),
    State({"type": "palette-item", "index": dash.ALL}, "style"),
    State("edge-source-select", "value"),
    State("edge-target-select", "value"),
    State("edge-condition-input", "value"),
    State("selected-element-store", "data"),
    State("edit-condition-input", "value"),
    prevent_initial_call=True,
)
def update_canvas(
    palette_clicks, del_clicks, reset_clicks, edge_confirm, del_edge, edge_edit_save,
    layout_val,
    current_elements, selected_nodes, selected_edges, wf_data,
    palette_ids, palette_labels, palette_styles,
    edge_src, edge_tgt, edge_cond,
    selected_el, edit_cond,
):
    ctx = callback_context
    if not ctx.triggered:
        return current_elements, wf_data, dash.no_update

    # triggered_id est un dict pour les IDs pattern-matched, une str pour les IDs simples
    triggered_id = ctx.triggered_id

    if triggered_id == "btn-reset":
        wf = default_workflow(config.NOTEBOOKS_DIR)
        return wf.to_cytoscape_elements(), wf.to_dict(), dash.no_update

    if isinstance(triggered_id, dict) and triggered_id.get("type") == "palette-item":
        triggered_index = triggered_id["index"]

        nb_path_found, label_found, color_found = None, triggered_index, PALETTE[0]
        for i, pid in enumerate(palette_ids):
            if pid["index"] == triggered_index:
                nb_path_found = Path(config.NOTEBOOKS_DIR) / triggered_index
                label_found = palette_labels[i] if palette_labels[i] else triggered_index
                color_found = (palette_styles[i].get("background", PALETTE[0])
                               if palette_styles[i] else PALETTE[0])
                break

        if not nb_path_found:
            return current_elements, wf_data, dash.no_update

        n_nodes = sum(1 for el in current_elements if "source" not in el.get("data", {}))
        x = 100 + (n_nodes % 5) * 200
        y = 100 + (n_nodes // 5) * 150
        new_id = f"nb_{uuid.uuid4().hex[:6]}"
        new_el = {
            "data": {
                "id": new_id, "label": label_found, "description": "",
                "notebook_path": str(nb_path_found), "color": color_found,
            },
            "position": {"x": x, "y": y},
            "classes": "notebook-node",
        }
        wf = Workflow.from_cytoscape_elements(current_elements)
        wf.nodes.append(WorkflowNode(
            id=new_id, notebook_path=str(nb_path_found),
            label=label_found, color=color_found, x=x, y=y,
        ))
        return current_elements + [new_el], wf.to_dict(), dash.no_update

    if triggered_id == "btn-delete-node" and selected_nodes:
        ids_to_del = {n["id"] for n in selected_nodes}
        new_els = [
            el for el in current_elements
            if el["data"].get("id") not in ids_to_del
            and el["data"].get("source") not in ids_to_del
            and el["data"].get("target") not in ids_to_del
        ]
        wf = Workflow.from_cytoscape_elements(new_els)
        return new_els, wf.to_dict(), dash.no_update

    if triggered_id == "btn-edge-confirm" and edge_src and edge_tgt:
        edge_id = f"e{edge_src}-{edge_tgt}-{uuid.uuid4().hex[:4]}"
        cond = edge_cond or ""
        new_edge = {
            "data": {
                "id": edge_id, "source": edge_src, "target": edge_tgt,
                "condition": cond, "label": cond,
            },
            "classes": "conditional-edge" if cond else "default-edge",
        }
        wf = Workflow.from_cytoscape_elements(current_elements + [new_edge])
        return current_elements + [new_edge], wf.to_dict(), dash.no_update

    if triggered_id == "btn-delete-edge" and selected_el and selected_el.get("type") == "edge":
        edge_id = selected_el["id"]
        new_els = [el for el in current_elements if el["data"].get("id") != edge_id]
        wf = Workflow.from_cytoscape_elements(new_els)
        return new_els, wf.to_dict(), False

    if triggered_id == "btn-edge-edit-save" and selected_el and selected_el.get("type") == "edge":
        edge_id = selected_el["id"]
        cond = edit_cond or ""
        new_els = []
        for el in current_elements:
            if el["data"].get("id") == edge_id:
                el = dict(el)
                el["data"] = dict(el["data"])
                el["data"]["condition"] = cond
                el["data"]["label"] = cond
                el["classes"] = "conditional-edge" if cond else "default-edge"
            new_els.append(el)
        wf = Workflow.from_cytoscape_elements(new_els)
        return new_els, wf.to_dict(), False

    return current_elements, wf_data, dash.no_update


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Canvas : propriétés
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("properties-panel", "children"),
    Output("selected-element-store", "data"),
    Output("edge-edit-modal", "is_open"),
    Output("edit-condition-input", "value"),
    Input("canvas", "selectedNodeData"),
    Input("canvas", "selectedEdgeData"),
    State("edge-edit-modal", "is_open"),
    prevent_initial_call=True,
)
def show_properties(node_data, edge_data, modal_open):
    if edge_data:
        e = edge_data[-1]
        cond = e.get("condition", "")
        panel = [
            html.P(f"Arête : {e.get('source','?')} → {e.get('target','?')}",
                   style={"color": "#aaa", "fontSize": "12px"}),
            dbc.Badge("Condition" if cond else "Inconditionnelle",
                      color="warning" if cond else "success"),
            html.P(cond or "Toujours exécutée", className="mt-2",
                   style={"fontFamily": "monospace", "fontSize": "12px", "color": "#fff"}),
            dbc.Button("✏ Modifier la condition", id="btn-open-edge-edit",
                       color="warning", size="sm", outline=True, className="mt-2"),
        ]
        return panel, {"type": "edge", "id": e["id"]}, True, cond

    if node_data:
        n = node_data[-1]
        path = n.get("notebook_path", "")
        panel = [
            html.H6(n.get("label", "?"), style={"color": "#fff", "fontWeight": "bold"}),
            html.P(Path(path).name if path else "—",
                   style={"color": "#aaa", "fontSize": "11px", "fontFamily": "monospace"}),
            dbc.Badge("Notebook", color="primary"),
        ]
        return panel, {"type": "node", "id": n["id"]}, False, ""

    return [html.P("Sélectionnez un élément.",
                   style={"color": "#aaa", "fontSize": "12px"})], None, False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Canvas : modals connexion
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("edge-modal", "is_open"),
    Output("edge-source-select", "options"),
    Output("edge-target-select", "options"),
    Input("btn-connect-hint", "n_clicks"),
    Input("btn-edge-cancel", "n_clicks"),
    Input("btn-edge-confirm", "n_clicks"),
    State("canvas", "elements"),
    State("edge-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_edge_modal(open_clicks, cancel, confirm, elements, is_open):
    trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
    if trigger == "btn-connect-hint":
        nodes = [el for el in (elements or []) if "source" not in el.get("data", {})]
        opts = [{"label": el["data"].get("label", el["data"]["id"]),
                 "value": el["data"]["id"]} for el in nodes]
        return True, opts, opts
    return False, [], []


@app.callback(
    Output("edge-edit-modal", "is_open", allow_duplicate=True),
    Input("btn-edge-edit-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def close_edge_edit(_):
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Canvas : upload CSV
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("csv-path-store", "data"),
    Output("csv-filename", "children"),
    Output("btn-run", "disabled"),
    Input("upload-csv", "contents"),
    State("upload-csv", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if contents is None:
        return None, "", True
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    save_path = str((UPLOADS_DIR / filename).resolve())
    with open(save_path, "wb") as f:
        f.write(decoded)
    return save_path, f"✓ {filename}", False


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Canvas : exécution du workflow
# ─────────────────────────────────────────────────────────────────────────────
_execution_results: dict = {}
_execution_lock = threading.Lock()

# Kernels persistants pour l'éditeur de notebooks
_nb_kernels: dict = {}
_nb_kernels_lock = threading.Lock()


# Agent results store
_agent_results: dict = {}
_agent_lock = threading.Lock()

# Template analysis results store
_tpl_results: dict = {}
_tpl_lock = threading.Lock()


def _make_execute_fn(kernel: dict):
    """Crée un execute_fn compatible avec run_agent_loop."""
    def execute_fn(code: str) -> tuple:
        from notebook_runner import execute_cell as _exec_cell
        output = _exec_cell(code, kernel)
        figs = capture_figures(kernel)
        return output, figs
    return execute_fn


def _run_agent_in_thread(csv_path: str, sexe: str, user_message: str,
                          system_prompt_template: str, max_steps: int = None) -> None:
    """Exécute run_agent_loop dans un thread background."""
    try:
        from inspector.kernel_snapshot import save_snapshot as _save_snapshot
        _inspector_available = True
    except ImportError:
        _inspector_available = False

    try:
        kernel = make_kernel()
    except Exception as exc:
        with _agent_lock:
            _agent_results["summary"] = f"Erreur initialisation kernel : {exc}"
            _agent_results["status"] = "error"
        return

    kernel["FILE_PATH"] = csv_path
    kernel["SEXE"] = sexe
    # Paramètres métier lus depuis actuarial_params (déjà injectés par make_kernel,
    # mais réaffectés ici pour garantir la cohérence avec FILE_PATH et SEXE)
    from actuarial_params import PARAMS as _ap
    import pandas as pd
    kernel["DATE_FIN_OBSERVATION"] = pd.Timestamp(_ap["observation"]["date_fin"])
    kernel["LAMBDA_WH"] = _ap["smoothing"]["lambda_wh"]
    kernel["PARAMS"] = _ap

    # Partager le kernel avec le RAG
    _ACTUARY_STATE.set_kernel(kernel)

    execute_fn = _make_execute_fn(kernel)

    # Injecter les paramètres concrets dans le message pour que l'agent les connaisse
    full_message = (
        f"{user_message}\n\n"
        f"Paramètres déjà définis dans le kernel :\n"
        f"- FILE_PATH = '{csv_path}'\n"
        f"- SEXE = '{sexe}'\n"
        f"- DATE_FIN_OBSERVATION = '2023-12-31'\n"
        f"- LAMBDA_WH = 100\n\n"
        f"Ces variables sont accessibles directement dans le kernel. "
        f"Commence immédiatement l'analyse sans demander de confirmation."
    )

    _ACTUARY_LOGGER.clear()
    _inspector_session_id = None  # identifiant de session partagé entre toutes les étapes

    try:
        for event in run_agent_loop(
            user_message=full_message,
            notebook_context="",
            conversation_history=[],
            execute_fn=execute_fn,
            system_prompt_template=system_prompt_template,
            max_steps=max_steps,
        ):
            with _agent_lock:
                if event["type"] == "step":
                    figs_b64 = [base64.b64encode(f).decode()
                                for f in event.get("figures", [])]
                    _agent_results["steps"].append({
                        "description": event.get("description", ""),
                        "code": event.get("code", ""),
                        "output": event.get("output", ""),
                        "figures": figs_b64,
                        "success": not event.get("output", "").startswith("❌"),
                    })
                    # Snapshot pour l'inspecteur (best-effort)
                    if _inspector_available:
                        try:
                            _inspector_session_id = _save_snapshot(
                                kernel,
                                step_info={
                                    "description": event.get("description", ""),
                                    "code": event.get("code", ""),
                                    "output": event.get("output", ""),
                                    "success": "❌" not in event.get("output", ""),
                                },
                                session_id=_inspector_session_id,
                            )
                        except Exception:
                            pass
                elif event["type"] == "summary":
                    _agent_results["summary"] = event.get("content", "")
                    _agent_results["status"] = "done"
                elif event["type"] == "error":
                    _agent_results["summary"] = event.get("content", "")
                    _agent_results["status"] = "error"
    except Exception as exc:
        with _agent_lock:
            _agent_results["summary"] = f"Erreur agent : {exc}"
            _agent_results["status"] = "error"

    # Pré-calculer les embeddings RAG en arrière-plan
    def _precompute():
        with _agent_lock:
            ag_steps = list(_agent_results.get("steps", []))
        log_chunks = _ACTUARY_LOGGER.to_chunks()
        log_steps = [{"label": c["label"], "output": c["text"]} for c in log_chunks]
        precompute_index(ag_steps + log_steps, _ACTUARY_STATE)

    threading.Thread(target=_precompute, daemon=True).start()


def _run_template_analysis_in_thread(pdf_bytes: bytes, filename: str) -> None:
    """Analyse un PDF de rapport de référence dans un thread background."""
    def _progress(msg: str) -> None:
        with _tpl_lock:
            _tpl_results["progress"] = msg

    try:
        template = analyze_report_pdf(
            pdf_bytes=pdf_bytes,
            filename=filename,
            progress_fn=_progress,
        )
        _ACTUARY_STATE.set_template(template)
        with _tpl_lock:
            _tpl_results["template"] = template
            _tpl_results["status"] = "done"
            _tpl_results["progress"] = f"✓ Analyse terminée : {template.get('report_title', filename)}"
    except Exception as exc:
        with _tpl_lock:
            _tpl_results["status"] = "error"
            _tpl_results["progress"] = f"❌ Erreur : {exc}"


def _run_in_thread(workflow_dict: dict, csv_path: str) -> None:
    wf = Workflow.from_dict(workflow_dict)
    try:
        kernel = make_kernel()
    except Exception as exc:
        with _execution_lock:
            _execution_results["summary"] = f"Erreur initialisation kernel : {exc}"
            _execution_results["status"] = "error"
        return
    kernel["FILE_PATH"] = csv_path
    kernel["SEXE"] = "H"
    from actuarial_params import PARAMS as _ap_wf
    kernel["DATE_FIN_OBSERVATION"] = __import__("pandas").Timestamp(_ap_wf["observation"]["date_fin"])
    kernel["LAMBDA_WH"] = _ap_wf["smoothing"]["lambda_wh"]
    kernel["PARAMS"] = _ap_wf

    # Partager le kernel avec le RAG (référence vivante — mis à jour à chaque step)
    _ACTUARY_STATE.set_kernel(kernel)

    _ACTUARY_LOGGER.clear()

    for event in execute_workflow(wf, kernel):
        with _execution_lock:
            if event["type"] == "step_start":
                _execution_results["steps"][event["node_id"]] = {
                    "status": "running", "label": event["label"],
                    "output": "", "figures": [],
                }
            elif event["type"] == "step_done":
                figs_b64 = [base64.b64encode(f).decode()
                            for f in event.get("figures", [])]
                _execution_results["steps"][event["node_id"]] = {
                    "status": "skipped" if event["skipped"] else "done",
                    "label": event["label"],
                    "output": event["output"],
                    "figures": figs_b64,
                }
            elif event["type"] == "error":
                _execution_results["steps"][event["node_id"]] = {
                    "status": "error",
                    "label": event.get("node_id", "?"),
                    "output": event["message"],
                    "figures": [],
                }
                _execution_results["status"] = "error"
            elif event["type"] == "done":
                _execution_results["status"] = "done"

    # Pré-calculer les embeddings RAG en arrière-plan dès que l'analyse est terminée
    def _precompute():
        with _execution_lock:
            all_steps = list(_execution_results.get("steps", {}).values())
        log_chunks = _ACTUARY_LOGGER.to_chunks()
        log_steps = [{"label": c["label"], "output": c["text"]} for c in log_chunks]
        precompute_index(all_steps + log_steps, _ACTUARY_STATE)

    threading.Thread(target=_precompute, daemon=True).start()


@app.callback(
    Output("exec-interval", "disabled"),
    Output("run-status", "children"),
    Input("btn-run", "n_clicks"),
    State("workflow-store", "data"),
    State("csv-path-store", "data"),
    prevent_initial_call=True,
)
def start_execution(n_clicks, wf_data, csv_path):
    if not csv_path or not wf_data:
        return True, "Chargez d'abord un fichier CSV."
    # Réinitialiser les résultats AVANT d'activer l'intervalle (même protection que l'agent).
    with _execution_lock:
        _execution_results.clear()
        _execution_results["status"] = "running"
        _execution_results["steps"] = {}
        _execution_results["summary"] = ""
    t = threading.Thread(target=_run_in_thread, args=(wf_data, csv_path), daemon=True)
    t.start()
    return False, "⏳ Exécution en cours…"


@app.callback(
    Output("results-panel", "children"),
    Output("exec-interval", "disabled", allow_duplicate=True),
    Output("run-status", "children", allow_duplicate=True),
    Output("canvas", "elements", allow_duplicate=True),
    Output("btn-word", "disabled"),
    Input("exec-interval", "n_intervals"),
    State("canvas", "elements"),
    prevent_initial_call=True,
)
def refresh_results(n, canvas_elements):
    with _execution_lock:
        results = dict(_execution_results)

    if not results:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    steps = results.get("steps", {})
    status = results.get("status", "running")

    cards = []
    for node_id, step in steps.items():
        st = step["status"]
        color_map = {"done": "success", "error": "danger",
                     "skipped": "secondary", "running": "warning"}
        badge_color = color_map.get(st, "secondary")
        icon = {"done": "✓", "error": "✗", "skipped": "⊘", "running": "⏳"}.get(st, "?")
        output_text = step["output"] or ""
        figs_html = [
            html.Img(src=f"data:image/png;base64,{fig_b64}",
                     style={"width": "100%", "borderRadius": "4px", "marginTop": "8px"})
            for fig_b64 in step.get("figures", [])
        ]
        cards.append(
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Strong(step["label"],
                                            style={"fontSize": "12px", "color": "#2D2D2D"}), width=10),
                        dbc.Col(dbc.Badge(f"{icon} {st}", color=badge_color,
                                          style={"fontSize": "10px"}), width=2),
                    ]),
                    html.Pre(
                        output_text[:500] + ("…" if len(output_text) > 500 else ""),
                        style={"fontSize": "10px", "color": "#555", "marginTop": "6px",
                               "maxHeight": "80px", "overflow": "hidden",
                               "background": "#EDEAE0", "padding": "4px",
                               "borderRadius": "3px"}
                    ) if output_text else None,
                    *figs_html,
                ], style={"padding": "8px"}),
                style={"marginBottom": "8px", "background": "#F5F2E7",
                       "border": "1px solid #C5BDB0"},
            )
        )

    new_elements = []
    for el in (canvas_elements or []):
        d = el.get("data", {})
        if "source" not in d:
            node_id = d.get("id", "")
            st = steps.get(node_id, {}).get("status", "")
            cls = {"running": "running", "done": "done",
                   "error": "error", "skipped": "skipped"}.get(st, "notebook-node")
            el = dict(el)
            el["classes"] = cls
        new_elements.append(el)

    done = status in ("done", "error")
    status_msg = ("✓ Terminé" if status == "done"
                  else "✗ Erreur" if status == "error" else "⏳ En cours…")
    return cards, done, status_msg, new_elements, not (status == "done")


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Canvas : sauvegarde / chargement workflow
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("toast-container", "children"),
    Input("btn-save", "n_clicks"),
    State("workflow-store", "data"),
    prevent_initial_call=True,
)
def save_workflow(n, wf_data):
    if not wf_data:
        return []
    wf = Workflow.from_dict(wf_data)
    path = WORKFLOWS_DIR / "workflow.json"
    wf.save(str(path))
    return dbc.Toast(
        f"Workflow sauvegardé : {path}",
        header="💾 Sauvegarde",
        is_open=True,
        duration=3000,
        style={"minWidth": "280px"},
        color="success",
    )


@app.callback(
    Output("canvas", "elements", allow_duplicate=True),
    Output("workflow-store", "data", allow_duplicate=True),
    Input("btn-load-open", "n_clicks"),
    prevent_initial_call=True,
)
def load_workflow(n):
    path = WORKFLOWS_DIR / "workflow.json"
    if not path.exists():
        return dash.no_update, dash.no_update
    wf = Workflow.load(str(path))
    return wf.to_cytoscape_elements(), wf.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Canvas : rapport Word
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("template-path-store", "data"),
    Output("template-filename", "children"),
    Input("upload-template", "contents"),
    State("upload-template", "filename"),
    prevent_initial_call=True,
)
def handle_template_upload(contents, filename):
    if contents is None:
        return None, ""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    save_path = str((UPLOADS_DIR / filename).resolve())
    with open(save_path, "wb") as f:
        f.write(decoded)
    return save_path, f"✓ {filename}"


@app.callback(
    Output("download-word", "data"),
    Input("btn-word", "n_clicks"),
    State("template-path-store", "data"),
    State("csv-path-store", "data"),
    prevent_initial_call=True,
)
def download_word_report(n_clicks, template_path, csv_path):
    with _execution_lock:
        steps_data = dict(_execution_results.get("steps", {}))
        summary = _execution_results.get("summary", "")

    steps = [
        {
            "label": s.get("label", ""),
            "output": s.get("output", ""),
            "figures": [base64.b64decode(f) for f in s.get("figures", [])],
            "status": s.get("status", ""),
        }
        for s in steps_data.values()
    ]
    # Enrichir chaque étape avec le log structuré correspondant
    log_report = _ACTUARY_LOGGER.to_report()
    if log_report:
        steps.append({"label": "Logs détaillés des fonctions", "output": log_report, "figures": []})
    doc_bytes = generate_word_report(
        steps=steps, summary=summary,
        template_path=template_path, file_path=csv_path or "",
    )
    return dcc.send_bytes(doc_bytes, "rapport_mortalite.docx")


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Notebooks : sélection et rendu
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("nb-viewer-area", "children"),
    Output("nb-current-path-store", "data"),
    Output("nb-cells-store", "data"),
    Output("btn-run-all-cells", "disabled"),
    Output("btn-save-notebook", "disabled"),
    Output("btn-reset-kernel", "disabled"),
    Input({"type": "nb-list-item", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def load_and_render_notebook(all_clicks):
    ctx = callback_context
    if not ctx.triggered or not any(c for c in (all_clicks or []) if c):
        return dash.no_update, dash.no_update, dash.no_update, True, True, True

    # triggered_id renvoie directement le dict — pas de parsing manuel du prop_id
    triggered_id = ctx.triggered_id
    if not isinstance(triggered_id, dict) or triggered_id.get("type") != "nb-list-item":
        return dash.no_update, dash.no_update, dash.no_update, True, True, True
    triggered_index = triggered_id["index"]

    # triggered_index est un chemin relatif depuis la racine du projet
    nb_path = str(Path(__file__).parent / triggered_index)
    try:
        if triggered_index.endswith(".py"):
            # Fichier .py : lire comme une seule cellule de code
            with open(nb_path, "r", encoding="utf-8") as fh:
                source = fh.read()
            cells = [{"id": "cell-0", "type": "code", "source": source, "output": ""}]
        else:
            cells = load_notebook(nb_path)
    except Exception as exc:
        return ([html.P(f"Erreur : {exc}", style={"color": "#F44336"})],
                None, [], True, True, True)

    header = html.H6(
        triggered_index,
        style={"color": "#777", "fontSize": "12px", "fontFamily": "monospace",
               "marginBottom": "12px"},
    )
    return [header] + _build_notebook_view(cells), nb_path, cells, False, False, False


@app.callback(
    Output("nb-viewer-area", "children", allow_duplicate=True),
    Input("main-tabs", "active_tab"),
    State("nb-current-path-store", "data"),
    State("nb-cells-store", "data"),
    prevent_initial_call=True,
)
def restore_nb_view_on_tab_switch(active_tab, nb_path, cells_data):
    """Restaure la vue notebook au retour sur l'onglet Notebooks."""
    if active_tab != "tab-notebooks" or not nb_path or not cells_data:
        return dash.no_update
    header = html.H6(
        Path(nb_path).name,
        style={"color": "#777", "fontSize": "12px", "fontFamily": "monospace",
               "marginBottom": "12px"},
    )
    return [header] + _build_notebook_view(cells_data)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Notebooks : exécution
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output({"type": "cell-output-pre", "index": dash.MATCH}, "children"),
    Output({"type": "cell-output-pre", "index": dash.MATCH}, "style"),
    Input({"type": "btn-run-cell", "index": dash.MATCH}, "n_clicks"),
    State({"type": "cell-textarea", "index": dash.MATCH}, "value"),
    State("nb-current-path-store", "data"),
    prevent_initial_call=True,
)
def run_single_cell(n_clicks, source, nb_path):
    if not n_clicks or not nb_path:
        return dash.no_update, dash.no_update
    with _nb_kernels_lock:
        if nb_path not in _nb_kernels:
            _nb_kernels[nb_path] = make_kernel()
        kernel = _nb_kernels[nb_path]
    output = execute_cell(source or "", kernel)
    color = "#F44336" if output.startswith("❌") else "#4CAF50" if "✓" in output else "#555"
    style = {
        "background": "#EDEAE0", "color": color,
        "fontSize": "11px", "padding": "6px 8px",
        "borderRadius": "3px", "maxHeight": "200px",
        "overflow": "auto", "marginTop": "6px",
        "display": "block", "whiteSpace": "pre-wrap",
        "border": "1px solid #C8C0A8",
    }
    return output, style


@app.callback(
    Output({"type": "cell-output-pre", "index": dash.ALL}, "children",
           allow_duplicate=True),
    Output({"type": "cell-output-pre", "index": dash.ALL}, "style",
           allow_duplicate=True),
    Output("nb-save-status", "children"),
    Input("btn-run-all-cells", "n_clicks"),
    State({"type": "cell-textarea", "index": dash.ALL}, "value"),
    State("nb-current-path-store", "data"),
    prevent_initial_call=True,
)
def run_all_cells(n_clicks, all_sources, nb_path):
    if not n_clicks or not nb_path:
        return [], [], dash.no_update
    # Kernel frais
    new_kernel = make_kernel()
    with _nb_kernels_lock:
        _nb_kernels[nb_path] = new_kernel

    outputs, styles = [], []
    for source in (all_sources or []):
        out = execute_cell(source or "", new_kernel)
        color = "#F44336" if out.startswith("❌") else "#4CAF50" if "✓" in out else "#555"
        outputs.append(out)
        styles.append({
            "background": "#EDEAE0", "color": color,
            "fontSize": "11px", "padding": "6px 8px",
            "borderRadius": "3px", "maxHeight": "200px",
            "overflow": "auto", "marginTop": "6px",
            "display": "block", "whiteSpace": "pre-wrap",
            "border": "1px solid #C8C0A8",
        })

    errors = sum(1 for o in outputs if o.startswith("❌"))
    msg = (f"✗ {errors} erreur(s) sur {len(outputs)} cellules" if errors
           else f"✓ {len(outputs)} cellule(s) exécutée(s)")
    return outputs, styles, msg


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Notebooks : sauvegarde et reset kernel
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("nb-save-status", "children", allow_duplicate=True),
    Input("btn-save-notebook", "n_clicks"),
    State({"type": "cell-textarea", "index": dash.ALL}, "value"),
    State({"type": "cell-textarea", "index": dash.ALL}, "id"),
    State("nb-current-path-store", "data"),
    prevent_initial_call=True,
)
def save_notebook_cb(n_clicks, all_sources, all_ids, nb_path):
    if not n_clicks or not nb_path:
        return dash.no_update
    try:
        if nb_path.endswith(".py"):
            # Fichier .py : la première (et seule) cellule contient tout le code
            source = (all_sources or [""])[0] or ""
            with open(nb_path, "w", encoding="utf-8") as f:
                f.write(source)
        else:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
            for id_dict, source in zip(all_ids or [], all_sources or []):
                cell_idx = id_dict["index"]
                if cell_idx < len(nb.cells):
                    nb.cells[cell_idx].source = source or ""
            with open(nb_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
        return f"✓ {Path(nb_path).name} sauvegardé"
    except Exception as exc:
        return f"✗ Erreur sauvegarde : {exc}"


@app.callback(
    Output("nb-save-status", "children", allow_duplicate=True),
    Input("btn-reset-kernel", "n_clicks"),
    State("nb-current-path-store", "data"),
    prevent_initial_call=True,
)
def reset_kernel_cb(n_clicks, nb_path):
    if not n_clicks or not nb_path:
        return dash.no_update
    with _nb_kernels_lock:
        if nb_path in _nb_kernels:
            del _nb_kernels[nb_path]
    return "🔄 Kernel réinitialisé"


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Agent : upload CSV
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("csv-path-agent-store", "data"),
    Output("csv-filename-agent", "children"),
    Output("btn-run-agent", "disabled"),
    Output("btn-plan-execute", "disabled"),
    Input("upload-csv-agent", "contents"),
    State("upload-csv-agent", "filename"),
    prevent_initial_call=True,
)
def handle_agent_upload(contents, filename):
    if contents is None:
        return None, "", True, True
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    save_path = str((UPLOADS_DIR / filename).resolve())
    with open(save_path, "wb") as f:
        f.write(decoded)
    return save_path, f"✓ {filename}", False, False


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Agent : system prompt apply / reset
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("system-prompt-store", "data"),
    Output("prompt-save-status", "children"),
    Input("btn-apply-prompt", "n_clicks"),
    State("system-prompt-textarea", "value"),
    prevent_initial_call=True,
)
def apply_system_prompt(n_clicks, textarea_value):
    if not n_clicks:
        return dash.no_update, dash.no_update
    return textarea_value, "✓ Prompt appliqué"


@app.callback(
    Output("system-prompt-textarea", "value"),
    Output("system-prompt-store", "data", allow_duplicate=True),
    Output("prompt-save-status", "children", allow_duplicate=True),
    Input("btn-reset-prompt", "n_clicks"),
    prevent_initial_call=True,
)
def reset_system_prompt(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    return SYSTEM_PROMPT_TEMPLATE, SYSTEM_PROMPT_TEMPLATE, "↺ Prompt réinitialisé"


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Agent : lancer l'agent
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("agent-interval", "disabled"),
    Output("agent-run-status", "children"),
    Output("btn-run-agent", "disabled", allow_duplicate=True),
    Output("btn-stop-agent", "disabled"),
    Output("btn-plan-execute", "disabled", allow_duplicate=True),
    Input("btn-run-agent", "n_clicks"),
    State("csv-path-agent-store", "data"),
    State("agent-user-message", "value"),
    State("agent-sexe-select", "value"),
    State("system-prompt-store", "data"),
    State("toggle-stepbystep", "value"),
    prevent_initial_call=True,
)
def start_agent(n_clicks, csv_path, user_message, sexe, system_prompt, stepbystep):
    if not csv_path:
        return True, "⚠ Chargez d'abord un fichier CSV.", False, True, False
    sp = system_prompt if system_prompt else SYSTEM_PROMPT_TEMPLATE
    msg = user_message or "Construis la table de mortalité d'expérience pour le fichier fourni."
    max_steps = 1 if stepbystep else None
    with _agent_lock:
        _agent_results.clear()
        _agent_results["status"] = "running"
        _agent_results["steps"] = []
        _agent_results["summary"] = ""
        _agent_results["csv_path"] = csv_path
        _agent_results["sexe"] = sexe or "H"
    t = threading.Thread(
        target=_run_agent_in_thread,
        args=(csv_path, sexe or "H", msg, sp, max_steps),
        daemon=True,
    )
    t.start()
    return False, "⏳ Agent en cours…", True, False, True


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Agent : rafraîchissement résultats
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("agent-results-panel", "children"),
    Output("agent-interval", "disabled", allow_duplicate=True),
    Output("agent-run-status", "children", allow_duplicate=True),
    Output("btn-run-agent", "disabled", allow_duplicate=True),
    Output("btn-stop-agent", "disabled", allow_duplicate=True),
    Output("btn-plan-execute", "disabled", allow_duplicate=True),
    Input("agent-interval", "n_intervals"),
    prevent_initial_call=True,
)
def refresh_agent_results(n):
    with _agent_lock:
        results = dict(_agent_results)

    if not results:
        return (dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update)

    steps = results.get("steps", [])
    status = results.get("status", "running")
    summary = results.get("summary", "")

    cards = []
    for i, step in enumerate(steps):
        figs_html = [
            html.Img(
                src=f"data:image/png;base64,{fig_b64}",
                style={"width": "100%", "borderRadius": "4px", "marginTop": "8px"},
            )
            for fig_b64 in step.get("figures", [])
        ]
        code = step.get("code", "")
        output = step.get("output", "")
        success = "❌" not in output
        step_id = f"step-code-{i}"
        cards.append(
            dbc.Card(
                dbc.CardBody([
                    html.P(f"Étape {i+1}  {'✅' if success else '❌'}",
                           style={"fontSize": "10px", "color": "#999", "marginBottom": "2px"}),
                    html.Strong(step["description"],
                                style={"fontSize": "12px", "color": "#2D2D2D"}),
                    # Sortie (tronquée)
                    html.Pre(
                        (output)[:600] + ("…" if len(output) > 600 else ""),
                        style={
                            "fontSize": "10px", "color": "#555" if success else "#c0392b",
                            "marginTop": "6px", "maxHeight": "80px", "overflow": "hidden",
                            "background": "#EDEAE0", "padding": "4px", "borderRadius": "3px",
                        },
                    ) if output else None,
                    # Bouton + code collapsible
                    html.Div([
                        dbc.Button(
                            "{ } Voir le code",
                            id={"type": "btn-toggle-code", "index": i},
                            size="sm", color="link",
                            style={"fontSize": "10px", "padding": "0", "color": "#888"},
                        ),
                        dbc.Collapse(
                            html.Pre(
                                code,
                                style={
                                    "fontSize": "10px", "background": "#2b2b2b",
                                    "color": "#f8f8f2", "padding": "8px",
                                    "borderRadius": "3px", "marginTop": "4px",
                                    "overflowX": "auto", "maxHeight": "200px",
                                },
                            ),
                            id={"type": "collapse-code", "index": i},
                            is_open=False,
                        ),
                    ]) if code else None,
                    *figs_html,
                ], style={"padding": "8px"}),
                style={"marginBottom": "8px", "background": "#F5F2E7",
                       "border": f"1px solid {'#C5BDB0' if success else '#F44336'}"},
            )
        )

    if summary:
        cards.append(
            dbc.Card(
                dbc.CardBody([
                    html.H6("Synthèse finale", style={"color": "#2D2D2D", "fontWeight": "bold"}),
                    dcc.Markdown(summary, style={"fontSize": "12px", "color": "#333"}),
                ], style={"padding": "12px"}),
                style={"marginBottom": "8px",
                       "background": "#E8F5E9" if status == "done" else "#FFEBEE",
                       "border": f"1px solid {'#4CAF50' if status == 'done' else '#F44336'}"},
            )
        )

    done = status in ("done", "error")
    status_msg = ("✓ Terminé" if status == "done"
                  else "✗ Erreur" if status == "error" else "⏳ En cours…")

    return cards, done, status_msg, not done, done, not done


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — construction des bulles de chat
# ─────────────────────────────────────────────────────────────────────────────
def _build_chat_messages(history: list[dict]) -> list:
    bubbles = []
    for msg in history:
        is_user = msg["role"] == "user"
        figures = msg.get("figures", [])
        bubble_children = [
            dcc.Markdown(msg["content"],
                         style={"margin": "0", "fontSize": "13px",
                                "lineHeight": "1.6"}),
        ]
        # Figures inline pour les réponses RAG
        for fig_b64 in figures:
            bubble_children.append(
                html.Img(
                    src=f"data:image/png;base64,{fig_b64}",
                    style={"maxWidth": "100%", "borderRadius": "6px",
                           "marginTop": "8px", "display": "block"},
                )
            )
        bubble = html.Div(
            [
                html.Div(
                    "Vous" if is_user else "RAG",
                    style={"fontSize": "10px", "color": "#999",
                           "marginBottom": "3px",
                           "textAlign": "right" if is_user else "left"},
                ),
                html.Div(
                    bubble_children,
                    style={
                        "background": "#D4EDDA" if is_user else "#FFFFFF",
                        "border": "1px solid " + ("#B8DACC" if is_user else "#C5BDB0"),
                        "borderRadius": "12px " + ("4px 12px 12px" if is_user
                                                    else "12px 12px 4px"),
                        "padding": "10px 14px",
                        "maxWidth": "90%",
                        "alignSelf": "flex-end" if is_user else "flex-start",
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
                    },
                ),
            ],
            style={"display": "flex", "flexDirection": "column",
                   "alignItems": "flex-end" if is_user else "flex-start"},
        )
        bubbles.append(bubble)
    return bubbles


def _get_rag_context() -> tuple[list[dict], str]:
    """Construit les steps et le summary depuis les résultats disponibles + logs."""
    with _execution_lock:
        wf_steps = list(_execution_results.get("steps", {}).values())
        summary = _execution_results.get("summary", "")
    with _agent_lock:
        ag_steps = list(_agent_results.get("steps", []))
        ag_summary = _agent_results.get("summary", "")
    for s in ag_steps:
        if "description" in s and "label" not in s:
            s["label"] = s["description"]
    log_chunks = _ACTUARY_LOGGER.to_chunks()
    log_steps = [{"label": c["label"], "output": c["text"]} for c in log_chunks]
    pdf_chunks = _ACTUARY_STATE.get_pdf_chunks()
    pdf_steps = [{"label": c["label"], "output": c["text"]} for c in pdf_chunks]
    # Code source des modules actuariels (.py) — permet au RAG de répondre
    # à des questions sur la méthodologie et les paramètres des fonctions.
    src_chunks = build_source_chunks()
    src_steps = [{"label": c["label"], "output": c["text"]} for c in src_chunks]
    # Injecter le résumé RAG du template comme contexte additionnel
    tpl = _ACTUARY_STATE.get_template()
    tpl_summary = ""
    if tpl:
        tpl_summary = tpl.get("rag_summary", "")
    return wf_steps + ag_steps + log_steps + pdf_steps + src_steps, summary or ag_summary or tpl_summary


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — RAG : prompt système
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("rag-system-prompt-store", "data"),
    Output("rag-prompt-save-status", "children"),
    Input("btn-rag-apply-prompt", "n_clicks"),
    State("rag-system-prompt-textarea", "value"),
    prevent_initial_call=True,
)
def rag_apply_prompt(n, value):
    return value or RAG_SYSTEM_PROMPT, "✓ Prompt appliqué"


@app.callback(
    Output("rag-system-prompt-textarea", "value"),
    Output("rag-system-prompt-store", "data", allow_duplicate=True),
    Output("rag-prompt-save-status", "children", allow_duplicate=True),
    Input("btn-rag-reset-prompt", "n_clicks"),
    prevent_initial_call=True,
)
def rag_reset_prompt(_):
    return RAG_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT, "↺ Prompt réinitialisé"


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — RAG : upload PDF
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("rag-pdf-status", "children"),
    Input("upload-rag-pdf", "contents"),
    State("upload-rag-pdf", "filename"),
    prevent_initial_call=True,
)
def handle_rag_pdf_upload(contents_list, filenames):
    if not contents_list:
        return dash.no_update
    try:
        import base64
        import fitz  # PyMuPDF
    except ImportError:
        return "❌ PyMuPDF non installé (pip install pymupdf)"

    _MAX_PDF_CHUNK = 1200  # caractères par chunk de page

    added = 0
    errors = []
    for content, filename in zip(contents_list, filenames or []):
        if not filename.lower().endswith(".pdf"):
            errors.append(f"{filename} : pas un PDF")
            continue
        try:
            _, b64 = content.split(",", 1)
            pdf_bytes = base64.b64decode(b64)
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            chunks = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text").strip()
                if not text:
                    continue
                # Découper les longues pages en sous-chunks
                for i in range(0, len(text), _MAX_PDF_CHUNK):
                    chunk_text = text[i : i + _MAX_PDF_CHUNK].strip()
                    if chunk_text:
                        chunks.append({
                            "text": chunk_text,
                            "label": f"PDF:{filename} p.{page_num + 1}",
                        })
            doc.close()
            _ACTUARY_STATE.add_pdf_chunks(chunks)
            added += len(chunks)
        except Exception as exc:
            errors.append(f"{filename} : {exc}")

    parts = []
    if added:
        parts.append(f"✓ {added} chunks PDF ajoutés au contexte RAG")
    parts.extend(errors)
    return " | ".join(parts) if parts else dash.no_update


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — RAG : chat
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("rag-chat-messages", "children"),
    Output("rag-history-store", "data"),
    Output("rag-chat-input", "value"),
    Input("btn-rag-send", "n_clicks"),
    Input("rag-chat-input", "n_submit"),
    State("rag-chat-input", "value"),
    State("rag-history-store", "data"),
    State("rag-system-prompt-store", "data"),
    prevent_initial_call=True,
)
def rag_send_message(n_clicks, n_submit, question, history, system_prompt):
    if not question or not question.strip():
        return dash.no_update, dash.no_update, dash.no_update

    history = history or []
    history.append({"role": "user", "content": question.strip()})

    try:
        all_steps, summary = _get_rag_context()
        exec_ns = _ACTUARY_STATE.get_exec_namespace()
        answer, figures = answer_with_tools(
            question=question.strip(),
            steps=all_steps,
            exec_ns=exec_ns,
            state=_ACTUARY_STATE,
            summary=summary,
            system_prompt=system_prompt or RAG_TOOLS_SYSTEM_PROMPT,
            conversation_history=history,
        )
        _ACTUARY_STATE.update_rag_ns(exec_ns)
        # Convertir les bytes en base64 pour le stockage JSON
        figs_b64 = [base64.b64encode(f).decode() for f in figures]
    except Exception as exc:
        import traceback
        answer = f"❌ Erreur RAG : {exc}\n\n```\n{traceback.format_exc()}\n```"
        figs_b64 = []

    history.append({"role": "assistant", "content": answer, "figures": figs_b64})
    return _build_chat_messages(history), history, ""


@app.callback(
    Output("rag-chat-messages", "children", allow_duplicate=True),
    Output("rag-history-store", "data", allow_duplicate=True),
    Input("btn-rag-clear", "n_clicks"),
    prevent_initial_call=True,
)
def rag_clear_chat(_):
    _ACTUARY_STATE.reset_rag_ns()   # libère aussi les variables calculées par le RAG
    placeholder = html.Div(
        "Conversation effacée. Posez votre prochaine question.",
        style={"color": "#999", "fontSize": "13px",
               "textAlign": "center", "marginTop": "60px"},
    )
    return [placeholder], []


# ─────────────────────────────────────────────────────────────────────────────
# Callback — Expand/collapse RAG chat
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG_NORMAL = {
    "background": "#F0EDE3", "padding": "12px",
    "height": "calc(100vh - 88px)", "overflowY": "auto",
    "borderRight": "1px solid #C5BDB0",
    "width": "16.666%", "minWidth": "160px", "flexShrink": "0",
}
_CONV_NORMAL = {
    "background": "#FBF8F1", "padding": "0",
    "borderRight": "1px solid #C5BDB0",
    "flex": "1", "minWidth": "300px", "overflow": "hidden",
    "height": "calc(100vh - 88px)",
}
_CONFIG_HIDDEN = {
    "display": "none",
}
_CONV_EXPANDED = {
    "background": "#FBF8F1", "padding": "0",
    "borderRight": "1px solid #C5BDB0",
    "flex": "1", "minWidth": "300px", "overflow": "hidden",
    "height": "calc(100vh - 88px)",
}


@app.callback(
    Output("agent-config-col", "style"),
    Output("agent-conv-col", "style"),
    Output("rag-expanded-store", "data"),
    Input("btn-expand-chat", "n_clicks"),
    State("rag-expanded-store", "data"),
    prevent_initial_call=True,
)
def toggle_chat_expand(_n, is_expanded):
    new_expanded = not (is_expanded or False)
    if new_expanded:
        return _CONFIG_HIDDEN, _CONV_EXPANDED, True
    return _CONFIG_NORMAL, _CONV_NORMAL, False


# ─────────────────────────────────────────────────────────────────────────────
# Callback — Modal System Prompt
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("modal-system-prompt", "is_open"),
    Input("btn-show-prompt", "n_clicks"),
    Input("btn-close-prompt-modal", "n_clicks"),
    State("modal-system-prompt", "is_open"),
    prevent_initial_call=True,
)
def toggle_prompt_modal(open_clicks, close_clicks, is_open):
    return not is_open


# ─────────────────────────────────────────────────────────────────────────────
# Callback — Collapse section Analyse Rapport
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("collapse-report-section", "is_open"),
    Input("btn-toggle-report-section", "n_clicks"),
    State("collapse-report-section", "is_open"),
    prevent_initial_call=True,
)
def toggle_report_section(_n, is_open):
    return not is_open


# ─────────────────────────────────────────────────────────────────────────────
# Callback — Plan-and-Execute : Canvas → Agent
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("agent-user-message", "value"),
    Output("btn-run-agent", "n_clicks"),
    Input("btn-plan-execute", "n_clicks"),
    State("workflow-store", "data"),
    State("agent-user-message", "value"),
    prevent_initial_call=True,
)
def plan_and_execute(n, wf_data, current_msg):
    if not wf_data:
        return dash.no_update, dash.no_update
    try:
        wf = Workflow.from_dict(wf_data)
        order = wf.execution_order()
        steps_labels = []
        for node_id in order:
            node = wf.get_node(node_id)
            if node:
                steps_labels.append(node.label)
        if not steps_labels:
            return dash.no_update, dash.no_update
        plan_prefix = (
            "Voici le plan d'analyse à suivre (dans cet ordre) :\n"
            + "\n".join(f"  {i+1}. {label}" for i, label in enumerate(steps_labels))
            + "\n\nRéalise chaque étape dans l'ordre en utilisant les fonctions "
            "disponibles dans ta bibliothèque actuarielle. "
            "Tu gardes la main sur la façon de réaliser chaque étape.\n\n"
        )
        base_msg = (current_msg or "").strip()
        if base_msg:
            new_msg = plan_prefix + base_msg
        else:
            new_msg = plan_prefix + "Construis la table de mortalité d'expérience pour le fichier fourni."
    except Exception:
        return dash.no_update, dash.no_update
    # Mettre à jour le message puis déclencher le bouton run
    return new_msg, 1


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Validation pipeline
# ─────────────────────────────────────────────────────────────────────────────
_validation_results: dict = {}
_validation_lock = threading.Lock()


def _run_validation_in_thread() -> None:
    with _validation_lock:
        _validation_results.clear()
        _validation_results["status"] = "running"
        _validation_results["steps"] = []

    try:
        validator = _get_validator()
        steps = validator.validate_pipeline(verbose=False)
        with _validation_lock:
            _validation_results["steps"] = [
                {"name": s.name, "status": s.status,
                 "detail": s.detail, "warning": s.warning}
                for s in steps
            ]
            n_ok = sum(1 for s in steps if s.status == "ok")
            n_warn = sum(1 for s in steps if s.status == "warning")
            n_fail = sum(1 for s in steps if s.status == "fail")
            _validation_results["summary"] = (
                f"{n_ok + n_warn}/{len(steps)} OK  |  "
                f"{n_warn} avertissements  |  {n_fail} échecs"
            )
            _validation_results["status"] = "done" if n_fail == 0 else "error"
    except Exception as exc:
        with _validation_lock:
            _validation_results["summary"] = f"Erreur : {exc}"
            _validation_results["status"] = "error"


@app.callback(
    Output("validate-interval", "disabled"),
    Output("validate-status", "children"),
    Input("btn-validate", "n_clicks"),
    prevent_initial_call=True,
)
def start_validation(n_clicks):
    t = threading.Thread(target=_run_validation_in_thread, daemon=True)
    t.start()
    return False, "⏳ Validation en cours…"


@app.callback(
    Output("validate-modal", "is_open"),
    Output("validate-results-body", "children"),
    Output("validate-interval", "disabled", allow_duplicate=True),
    Output("validate-status", "children", allow_duplicate=True),
    Input("validate-interval", "n_intervals"),
    Input("btn-validate-close", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_validation(n_intervals, close_clicks):
    ctx = callback_context
    if ctx.triggered_id == "btn-validate-close":
        return False, dash.no_update, True, dash.no_update

    with _validation_lock:
        results = dict(_validation_results)

    if not results or results.get("status") == "running":
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    steps = results.get("steps", [])
    summary_txt = results.get("summary", "")
    status = results.get("status", "done")

    color_map = {"ok": "#4CAF50", "warning": "#FF9800", "fail": "#F44336"}
    icon_map  = {"ok": "✓", "warning": "⚠", "fail": "✗"}

    rows = []
    for s in steps:
        icon  = icon_map.get(s["status"], "?")
        color = color_map.get(s["status"], "#777")
        warn  = f"  ⚠ {s['warning']}" if s.get("warning") else ""
        rows.append(
            html.Div(
                [
                    html.Span(f"{icon} ", style={"color": color, "fontWeight": "bold"}),
                    html.Span(f"{s['name']:<45}", style={"color": "#2D2D2D"}),
                    html.Span(s["detail"], style={"color": "#666"}),
                    html.Span(warn, style={"color": "#FF9800"}),
                ],
                style={"padding": "2px 0", "borderBottom": "1px solid #EEE",
                       "whiteSpace": "pre"},
            )
        )

    summary_color = "#4CAF50" if status == "done" else "#F44336"
    body = [
        html.Div(
            summary_txt,
            style={"fontWeight": "bold", "color": summary_color,
                   "marginBottom": "12px", "fontSize": "13px"},
        ),
        html.Div(rows),
    ]

    status_label = ("✓ " + summary_txt if status == "done"
                    else "✗ Échecs détectés")
    return True, body, True, status_label


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Onglet Analyse Rapport
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("report-pdf-filename", "children"),
    Output("btn-analyze-report", "disabled"),
    Input("upload-report-pdf", "filename"),
    prevent_initial_call=True,
)
def show_report_pdf_name(filename):
    if not filename:
        return "", True
    return f"📄 {filename}", False


@app.callback(
    Output("template-analysis-interval", "disabled"),
    Output("template-analysis-status", "children"),
    Output("btn-analyze-report", "disabled", allow_duplicate=True),
    Input("btn-analyze-report", "n_clicks"),
    State("upload-report-pdf", "contents"),
    State("upload-report-pdf", "filename"),
    prevent_initial_call=True,
)
def start_template_analysis(n_clicks, contents, filename):
    if not contents or not filename:
        return True, "⚠ Chargez d'abord un PDF.", False
    try:
        _, b64 = contents.split(",", 1)
        pdf_bytes = base64.b64decode(b64)
    except Exception as exc:
        return True, f"❌ Erreur lecture PDF : {exc}", False

    with _tpl_lock:
        _tpl_results.clear()
        _tpl_results["status"] = "running"
        _tpl_results["progress"] = "Extraction du texte…"
        _tpl_results["template"] = None

    t = threading.Thread(
        target=_run_template_analysis_in_thread,
        args=(pdf_bytes, filename),
        daemon=True,
    )
    t.start()
    return False, "⏳ Extraction du texte…", True


@app.callback(
    Output("template-analysis-status", "children", allow_duplicate=True),
    Output("template-analysis-result", "children"),
    Output("report-template-store", "data"),
    Output("btn-download-template", "disabled"),
    Output("btn-send-template-to-agent", "disabled"),
    Output("template-analysis-interval", "disabled", allow_duplicate=True),
    Output("btn-analyze-report", "disabled", allow_duplicate=True),
    Input("template-analysis-interval", "n_intervals"),
    prevent_initial_call=True,
)
def poll_template_analysis(n_intervals):
    with _tpl_lock:
        results = dict(_tpl_results)

    status = results.get("status", "")
    progress = results.get("progress", "")

    if status == "running":
        return progress, dash.no_update, dash.no_update, True, True, False, True

    if status == "error":
        return progress, dash.no_update, dash.no_update, True, True, True, False

    if status != "done":
        return dash.no_update, dash.no_update, dash.no_update, True, True, True, False

    template = results.get("template") or {}
    result_children = _render_template_result(template)
    return progress, result_children, template, False, False, True, False


@app.callback(
    Output("template-analysis-result", "children", allow_duplicate=True),
    Output("report-template-store", "data", allow_duplicate=True),
    Output("btn-download-template", "disabled", allow_duplicate=True),
    Output("btn-send-template-to-agent", "disabled", allow_duplicate=True),
    Output("template-analysis-status", "children", allow_duplicate=True),
    Input("upload-report-template-json", "contents"),
    State("upload-report-template-json", "filename"),
    prevent_initial_call=True,
)
def load_existing_template_json(contents, filename):
    if not contents:
        return dash.no_update, dash.no_update, True, True, dash.no_update
    try:
        _, b64 = contents.split(",", 1)
        template = json.loads(base64.b64decode(b64).decode("utf-8"))
    except Exception as exc:
        return dash.no_update, dash.no_update, True, True, f"❌ {exc}"
    _ACTUARY_STATE.set_template(template)
    result_children = _render_template_result(template)
    title = template.get("report_title", filename)
    return result_children, template, False, False, f"✓ Template chargé : {title}"


@app.callback(
    Output("download-template-json", "data"),
    Input("btn-download-template", "n_clicks"),
    State("report-template-store", "data"),
    State("upload-report-pdf", "filename"),
    prevent_initial_call=True,
)
def download_template(n_clicks, template, pdf_filename):
    if not template:
        return dash.no_update
    stem = Path(pdf_filename or "rapport").stem if pdf_filename else "rapport"
    return dcc.send_string(
        json.dumps(template, ensure_ascii=False, indent=2),
        filename=f"{stem}_template.json",
        type="application/json",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Chargement template dans l'onglet Agent
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("system-prompt-textarea", "value", allow_duplicate=True),
    Output("system-prompt-store", "data", allow_duplicate=True),
    Output("agent-template-status", "children"),
    Output("btn-clear-agent-template", "disabled"),
    Input("upload-agent-template", "contents"),
    State("upload-agent-template", "filename"),
    prevent_initial_call=True,
)
def load_agent_template(contents, filename):
    if not contents:
        return dash.no_update, dash.no_update, dash.no_update, True
    try:
        _, b64 = contents.split(",", 1)
        template = json.loads(base64.b64decode(b64).decode("utf-8"))
    except Exception as exc:
        return dash.no_update, dash.no_update, f"❌ {exc}", True
    prompt = template.get("agent_system_prompt", "")
    if not prompt:
        return dash.no_update, dash.no_update, "⚠ Pas de prompt dans ce template", True
    _ACTUARY_STATE.set_template(template)
    title = template.get("report_title", filename)
    return prompt, prompt, f"✓ {title}", False


@app.callback(
    Output("system-prompt-textarea", "value", allow_duplicate=True),
    Output("system-prompt-store", "data", allow_duplicate=True),
    Output("agent-template-status", "children", allow_duplicate=True),
    Output("btn-clear-agent-template", "disabled", allow_duplicate=True),
    Input("btn-clear-agent-template", "n_clicks"),
    prevent_initial_call=True,
)
def clear_agent_template(_):
    _ACTUARY_STATE.clear_template()
    return SYSTEM_PROMPT_TEMPLATE, SYSTEM_PROMPT_TEMPLATE, "", True


@app.callback(
    Output("main-tabs", "active_tab"),
    Output("system-prompt-textarea", "value", allow_duplicate=True),
    Output("system-prompt-store", "data", allow_duplicate=True),
    Output("agent-template-status", "children", allow_duplicate=True),
    Output("btn-clear-agent-template", "disabled", allow_duplicate=True),
    Input("btn-send-template-to-agent", "n_clicks"),
    State("report-template-store", "data"),
    prevent_initial_call=True,
)
def send_template_to_agent(n_clicks, template):
    if not template:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, True
    prompt = template.get("agent_system_prompt", "")
    if not prompt:
        return dash.no_update, dash.no_update, dash.no_update, "⚠ Pas de prompt", True
    _ACTUARY_STATE.set_template(template)
    title = template.get("report_title", "?")
    return "tab-agent", prompt, prompt, f"✓ {title}", False


# ─────────────────────────────────────────────────────────────────────────────
# Helper — rendu du résultat d'analyse
# ─────────────────────────────────────────────────────────────────────────────

def _render_template_result(template: dict) -> list:
    """Construit le contenu HTML du panneau de résultat d'analyse."""
    _section_style = {
        "background": "#FBF8F1", "border": "1px solid #D8D0C4",
        "borderRadius": "8px", "padding": "12px", "marginBottom": "10px",
    }
    _h_style = {"fontSize": "12px", "fontWeight": "bold",
                "color": "#5A4A3A", "marginBottom": "6px"}
    _item_style = {"fontSize": "11px", "color": "#333",
                   "padding": "3px 0", "borderBottom": "1px solid #EEE"}

    children = []

    # Titre
    title = template.get("report_title", "Rapport sans titre")
    children.append(
        html.Div(title, style={"fontSize": "15px", "fontWeight": "bold",
                               "color": "#2D2D2D", "marginBottom": "14px"})
    )

    # Résumé RAG
    rag_sum = template.get("rag_summary", "")
    if rag_sum:
        children.append(html.Div([
            html.P("Résumé", style=_h_style),
            html.P(rag_sum, style={"fontSize": "11px", "color": "#555",
                                   "lineHeight": "1.5"}),
        ], style=_section_style))

    # Sections
    sections = template.get("sections", [])
    if sections:
        children.append(html.Div([
            html.P(f"Sections ({len(sections)})", style=_h_style),
            html.Div([
                html.Div(
                    f"{s.get('id', '')} — {s.get('title', '')}",
                    title=s.get("description", ""),
                    style=_item_style,
                )
                for s in sections
            ]),
        ], style=_section_style))

    # Tableaux
    tables = template.get("tables", [])
    if tables:
        children.append(html.Div([
            html.P(f"Tableaux attendus ({len(tables)})", style=_h_style),
            html.Div([
                html.Div([
                    html.Span(f"{t.get('id', '')} — {t.get('name', '')}  ",
                              style={"fontWeight": "500"}),
                    html.Span(", ".join(t.get("columns", [])),
                              style={"color": "#777", "fontStyle": "italic"}),
                ], title=t.get("description", ""), style=_item_style)
                for t in tables
            ]),
        ], style=_section_style))

    # Graphiques
    figures = template.get("figures", [])
    if figures:
        children.append(html.Div([
            html.P(f"Graphiques attendus ({len(figures)})", style=_h_style),
            html.Div([
                html.Div([
                    html.Span(f"{f.get('id', '')} — {f.get('title', '')}  ",
                              style={"fontWeight": "500"}),
                    html.Span(
                        f"[{f.get('type', '')}] {f.get('x_axis', '')} → {f.get('y_axis', '')}",
                        style={"color": "#777", "fontStyle": "italic"},
                    ),
                ], title=f.get("description", ""), style=_item_style)
                for f in figures
            ]),
        ], style=_section_style))

    # Métriques clés
    metrics = template.get("key_metrics", [])
    if metrics:
        children.append(html.Div([
            html.P("Métriques clés", style=_h_style),
            html.P(", ".join(metrics),
                   style={"fontSize": "11px", "color": "#555"}),
        ], style=_section_style))

    # Prompt agent (textarea éditable)
    agent_prompt = template.get("agent_system_prompt", "")
    if agent_prompt:
        children.append(html.Div([
            html.P("System Prompt généré pour l'Agent", style=_h_style),
            dcc.Textarea(
                value=agent_prompt,
                id="template-result-prompt-textarea",
                style={
                    "width": "100%", "height": "220px",
                    "fontSize": "11px", "fontFamily": "'Courier New', monospace",
                    "background": "#F0EDE3", "color": "#1A1A1A",
                    "border": "1px solid #C5BDB0", "borderRadius": "4px",
                    "padding": "8px", "resize": "vertical",
                },
            ),
        ], style=_section_style))

    # Notes d'analyse
    notes = template.get("analysis_notes", "")
    if notes:
        children.append(html.Div([
            html.P("Notes d'analyse", style=_h_style),
            html.P(notes, style={"fontSize": "11px", "color": "#888",
                                 "lineHeight": "1.5"}),
        ], style=_section_style))

    return children


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Onglet Notebook
# ─────────────────────────────────────────────────────────────────────────────

def _nb_exec(code: str, kernel: dict) -> tuple[str, bool]:
    """Exécute du code dans le kernel et retourne (texte_sortie, is_error)."""
    import io as _io, traceback as _tb, numpy as _np
    from contextlib import redirect_stdout as _rs
    buf = _io.StringIO()
    try:
        with _rs(buf):
            exec(code, kernel)  # noqa: S102
        text = buf.getvalue()
        # Valeur de la dernière expression
        try:
            last = [l.strip() for l in code.strip().splitlines()
                    if l.strip() and not l.strip().startswith("#")][-1]
            if not any(last.startswith(kw) for kw in
                       ("import", "from", "=", "print", "if", "for",
                        "while", "def", "class", "return", "try", "with")):
                val = eval(last, kernel)  # noqa: S307
                if val is not None:
                    if isinstance(val, pd.DataFrame):
                        text += "\n" + val.head(20).to_string()
                    elif isinstance(val, _np.ndarray):
                        text += "\n" + repr(val)
                    else:
                        text += "\n" + repr(val)
        except Exception:
            pass
        return (text or "✓ Exécuté."), False
    except Exception:
        return _tb.format_exc(), True


def _render_nb_cell(cell: dict) -> html.Div:
    """Génère le rendu HTML d'une cellule notebook."""
    uid = cell["uid"]
    code = cell.get("code", "")
    output = cell.get("output")
    error = cell.get("error", False)
    out_style = {
        "fontSize": "11px", "fontFamily": "monospace", "padding": "6px",
        "borderRadius": "3px", "whiteSpace": "pre-wrap", "marginTop": "4px",
        "background": "#fff5f5" if error else "#F5F2E7",
        "color": "#c0392b" if error else "#333",
        "border": f"1px solid {'#f5c6c6' if error else '#D5CEC0'}",
    }
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Small(f"In [{uid}]:",
                                   style={"color": "#888", "fontFamily": "monospace",
                                          "fontSize": "10px"}),
                        width="auto", className="align-self-start pe-1",
                    ),
                    dbc.Col(
                        dcc.Textarea(
                            id={"type": "nbc-code", "index": uid},
                            value=code,
                            style={
                                "width": "100%", "fontFamily": "monospace",
                                "fontSize": "11px", "background": "#FAFAF8",
                                "border": "1px solid #D5CEC0", "borderRadius": "3px",
                                "padding": "4px", "resize": "vertical",
                                "minHeight": "60px",
                            },
                        ),
                        width=True,
                    ),
                    dbc.Col(
                        html.Div([
                            dbc.Button("▶", id={"type": "nbc-run", "index": uid},
                                       color="success", size="sm",
                                       style={"display": "block", "marginBottom": "3px",
                                              "padding": "2px 8px"}),
                            dbc.Button("✕", id={"type": "nbc-del", "index": uid},
                                       color="light", size="sm",
                                       style={"display": "block", "padding": "2px 8px",
                                              "fontSize": "10px"}),
                        ]),
                        width="auto", className="align-self-start ps-1",
                    ),
                ],
                className="g-0",
            ),
            html.Div(output, style=out_style) if output is not None else None,
            html.Hr(style={"margin": "6px 0", "borderColor": "#E0DBD0"}),
        ],
        style={"marginBottom": "2px"},
    )


@app.callback(
    Output("nb-agent-cells-area", "children"),
    Input("nb-agent-cells-data", "data"),
)
def render_nb_cells(cells_data):
    if not cells_data:
        return html.Div(
            "Cliquez '📥 Charger étapes agent' pour importer les étapes "
            "de l'agent, ou '+ Cellule' pour ajouter une cellule vide.",
            style={"color": "#aaa", "fontSize": "12px",
                   "padding": "20px", "textAlign": "center"},
        )
    return [_render_nb_cell(c) for c in cells_data]


@app.callback(
    Output("nb-agent-cells-data", "data"),
    Input({"type": "nbc-run", "index": dash.ALL}, "n_clicks"),
    Input({"type": "nbc-del", "index": dash.ALL}, "n_clicks"),
    Input("btn-nb-add", "n_clicks"),
    Input("btn-nb-load", "n_clicks"),
    Input("btn-nb-replay", "n_clicks"),
    State({"type": "nbc-code", "index": dash.ALL}, "value"),
    State("nb-agent-cells-data", "data"),
    prevent_initial_call=True,
)
def update_nb_cells(run_clicks, del_clicks, _add, _load, _replay,
                    codes_all, cells_data):
    # Sync textarea values back to store (preserve edits)
    for i, cell in enumerate(cells_data):
        if i < len(codes_all) and codes_all[i] is not None:
            cell["code"] = codes_all[i]

    triggered = dash.ctx.triggered_id

    if triggered == "btn-nb-add":
        new_uid = max((c["uid"] for c in cells_data), default=-1) + 1
        cells_data.append({"uid": new_uid, "code": "", "output": None, "error": False})

    elif triggered == "btn-nb-load":
        with _agent_lock:
            steps = list(_agent_results.get("steps", []))
        new_cells = []
        uid = 0
        for step in steps:
            code = (step.get("code") or "").strip()
            if code:
                new_cells.append({"uid": uid, "code": code,
                                  "output": None, "error": False})
                uid += 1
        cells_data = new_cells or cells_data

    elif triggered == "btn-nb-replay":
        kernel = _ACTUARY_STATE.get_exec_namespace()
        for cell in cells_data:
            out, err = _nb_exec(cell["code"], kernel)
            cell["output"] = out
            cell["error"] = err

    elif isinstance(triggered, dict) and triggered.get("type") == "nbc-run":
        uid = triggered["index"]
        cell = next((c for c in cells_data if c["uid"] == uid), None)
        if cell:
            kernel = _ACTUARY_STATE.get_exec_namespace()
            out, err = _nb_exec(cell["code"], kernel)
            cell["output"] = out
            cell["error"] = err

    elif isinstance(triggered, dict) and triggered.get("type") == "nbc-del":
        uid = triggered["index"]
        cells_data = [c for c in cells_data if c["uid"] != uid]

    return cells_data


@app.callback(
    Output("nb-jupyter-link", "href"),
    Output("btn-nb-download", "style"),
    Output("nb-gen-status", "children"),
    Output("nb-gen-path-store", "data"),
    Output("nb-jupyter-iframe", "srcDoc"),
    Output("nb-picker", "options"),
    Output("nb-picker", "value"),
    Input("btn-nb-generate", "n_clicks"),
    prevent_initial_call=True,
)
def generate_jupyter_nb(_n):
    with _agent_lock:
        steps = list(_agent_results.get("steps", []))
        summary = _agent_results.get("summary", "")
        csv_path = _agent_results.get("csv_path", "")
        sexe = _agent_results.get("sexe", "H")

    visible = {"display": "inline-block"}
    hidden = {"display": "none"}

    if not steps:
        opts = [{"label": d["label"], "value": d["value"]} for d in _list_notebooks()]
        return "#", hidden, "⚠ Aucune étape — lancez d'abord une analyse.", None, "", opts, dash.no_update

    try:
        nb_path = _generate_agent_notebook(steps, summary=summary,
                                            csv_path=csv_path, sexe=sexe)
    except Exception as exc:
        opts = [{"label": d["label"], "value": d["value"]} for d in _list_notebooks()]
        return "#", hidden, f"❌ Erreur : {exc}", None, "", opts, dash.no_update

    nb_str = str(nb_path)
    opts = [{"label": d["label"], "value": d["value"]} for d in _list_notebooks()]

    try:
        html_content = _notebook_to_html(nb_path)
        status = f"✓ {nb_path.name}"
        return nb_str, visible, status, nb_str, html_content, opts, nb_str
    except Exception as exc:
        return "#", visible, f"❌ Erreur rendu : {exc}", nb_str, "", opts, nb_str


@app.callback(
    Output("nb-jupyter-iframe", "srcDoc", allow_duplicate=True),
    Output("nb-gen-status", "children", allow_duplicate=True),
    Output("nb-gen-path-store", "data", allow_duplicate=True),
    Output("btn-nb-download", "style", allow_duplicate=True),
    Input("nb-picker", "value"),
    Input("btn-nb-open", "n_clicks"),
    State("nb-picker", "value"),
    prevent_initial_call=True,
)
def open_notebook_from_picker(nb_value_input, _n_open, nb_value_state):
    """Affiche un notebook existant dans l'iframe via nbconvert HTML."""
    triggered = dash.ctx.triggered_id
    nb_value = nb_value_state if triggered == "btn-nb-open" else nb_value_input
    if not nb_value:
        raise dash.exceptions.PreventUpdate
    nb_path = Path(nb_value)
    if not nb_path.exists():
        return "", f"❌ Fichier introuvable : {nb_path.name}", None, {"display": "none"}
    try:
        html_content = _notebook_to_html(nb_path)
        status = f"✓ {nb_path.name}"
        return html_content, status, nb_value, {"display": "inline-block"}
    except Exception as exc:
        return "", f"❌ Erreur rendu : {exc}", None, {"display": "none"}


@app.callback(
    Output("nb-picker", "options", allow_duplicate=True),
    Input("tools-sub-tabs", "active_tab"),
    prevent_initial_call=True,
)
def refresh_nb_picker(active_tab):
    """Rafraîchit la liste des notebooks quand l'onglet Notebook devient actif."""
    if active_tab != "tools-notebook":
        raise dash.exceptions.PreventUpdate
    return [{"label": d["label"], "value": d["value"]} for d in _list_notebooks()]


@app.callback(
    Output("nb-jupyter-url-store", "data"),
    Output("nb-gen-status", "children", allow_duplicate=True),
    Input("btn-nb-launch", "n_clicks"),
    State("nb-picker", "value"),
    State("nb-gen-path-store", "data"),
    State("csv-path-agent-store", "data"),
    State("agent-sexe-select", "value"),
    prevent_initial_call=True,
)
def launch_in_jupyter(_n, picker_val, store_val, csv_store, sexe_val):
    """Pré-exécute le notebook puis l'ouvre dans Jupyter (nouvel onglet, kernel prêt)."""
    nb_value = picker_val or store_val
    if not nb_value:
        return "", "⚠ Sélectionnez d'abord un notebook"
    nb_path = Path(nb_value)
    if not nb_path.exists():
        return "", f"❌ Fichier introuvable : {nb_path.name}"
    try:
        # Priorité : store UI > session agent en mémoire
        csv_path = csv_store or _agent_results.get("csv_path", "")
        sexe = sexe_val or _agent_results.get("sexe", "H") or "H"
        _patch_notebook_setup_cell(nb_path, csv_path=csv_path, sexe=sexe)
        # Pré-exécuter le notebook
        result = subprocess.run(
            [_ANACONDA_JUPYTER, "nbconvert", "--to", "notebook",
             "--execute", "--inplace", "--allow-errors",
             "--ExecutePreprocessor.timeout=180",
             "--ExecutePreprocessor.kernel_name=python3",
             str(nb_path)],
            capture_output=True, timeout=200,
            cwd=str(Path(__file__).parent),
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")[:400]
            return "", f"❌ Exécution : {err}"
        # Ouvrir dans Jupyter
        port = _start_jupyter_server()
        rel = nb_path.relative_to(Path(__file__).parent)
        url = f"http://127.0.0.1:{port}/notebooks/{rel}"
        return url, "✓ Notebook exécuté — ouverture dans un nouvel onglet"
    except Exception as exc:
        return "", f"❌ {exc}"


app.clientside_callback(
    """
    function(url) {
        if (url && url !== "") {
            window.open(url, "_blank");
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("nb-jupyter-url-store", "data", allow_duplicate=True),
    Input("nb-jupyter-url-store", "data"),
    prevent_initial_call=True,
)


@app.callback(
    Output("nb-download-ipynb", "data"),
    Input("btn-nb-download", "n_clicks"),
    State("nb-gen-path-store", "data"),
    prevent_initial_call=True,
)
def download_ipynb(_n, nb_str):
    if not nb_str:
        raise dash.exceptions.PreventUpdate
    nb_path = Path(nb_str)
    if not nb_path.exists():
        raise dash.exceptions.PreventUpdate
    return dcc.send_file(str(nb_path))


# Callback : toggle affichage code dans les étapes agent
@app.callback(
    Output({"type": "collapse-code", "index": dash.MATCH}, "is_open"),
    Input({"type": "btn-toggle-code", "index": dash.MATCH}, "n_clicks"),
    State({"type": "collapse-code", "index": dash.MATCH}, "is_open"),
    prevent_initial_call=True,
)
def toggle_step_code(n, is_open):
    if n:
        return not is_open
    return is_open


# ─────────────────────────────────────────────────────────────────────────────
# Drag-to-resize : poignée entre colonne Résultats Agent et colonne Outils
# ─────────────────────────────────────────────────────────────────────────────
app.clientside_callback(
    """
    function(active_tab) {
        if (active_tab !== 'tab-agent') return window.dash_clientside.no_update;
        if (window._agentResizeReady) return window.dash_clientside.no_update;

        function _setup() {
            var handle  = document.getElementById('agent-resize-handle');
            var convCol = document.getElementById('agent-conv-col');
            var toolCol = document.getElementById('agent-tools-col');
            if (!handle || !convCol || !toolCol) {
                setTimeout(_setup, 300);
                return;
            }
            window._agentResizeReady = true;

            var isResizing = false;
            var startX     = 0;
            var startW     = 0;

            handle.addEventListener('mousedown', function(e) {
                isResizing = true;
                startX     = e.clientX;
                startW     = toolCol.getBoundingClientRect().width;
                document.body.style.cursor    = 'col-resize';
                document.body.style.userSelect = 'none';
                e.preventDefault();
            });

            document.addEventListener('mousemove', function(e) {
                if (!isResizing) return;
                var delta   = startX - e.clientX;   // vers la gauche = plus large
                var newW    = Math.max(200, Math.min(800, startW + delta));
                toolCol.style.width    = newW + 'px';
                toolCol.style.minWidth = newW + 'px';
                handle.style.background = '#A09890';
            });

            document.addEventListener('mouseup', function() {
                if (!isResizing) return;
                isResizing = false;
                document.body.style.cursor    = '';
                document.body.style.userSelect = '';
                handle.style.background = '#C5BDB0';
            });
        }
        _setup();
        return window.dash_clientside.no_update;
    }
    """,
    Output("resize-init-store", "data"),
    Input("main-tabs", "active_tab"),
    prevent_initial_call=False,
)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Canvas Actuarial — http://localhost:8050")
    app.run(debug=False, port=8050, host="::", use_reloader=False)
