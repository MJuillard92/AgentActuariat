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
import nbformat
from dash import Input, Output, State, callback_context, dcc, html
from dash import dash_table
from dotenv import load_dotenv

import config
from notebook_runner import load_notebook, execute_cell
from workflow_executor import make_kernel, capture_figures
from agent import run_agent_loop, SYSTEM_PROMPT_TEMPLATE, load_knowledge_base_context, plan_agent
from report_generator import generate_pdf_report, generate_reasoning_trace, generate_final_notebook
from report_agent import generate_narrative_report
from domain_config import list_domains, load_system_prompt as _load_domain_prompt, load_kb_context as _load_domain_kb, get_default_message as _get_domain_default_msg
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


_NOTEBOOK_TOGGLE_CSS_JS = """
<style>
  .input { display: none; }
  .toggle-code-btn {
    font-size: 11px; cursor: pointer; color: #888; background: none;
    border: 1px solid #ddd; border-radius: 3px; padding: 1px 7px;
    margin-bottom: 3px; float: right;
  }
  .toggle-code-btn:hover { color: #333; border-color: #999; }
  .cell { position: relative; }
</style>
<script>
document.addEventListener("DOMContentLoaded", function() {
  document.querySelectorAll(".code_cell").forEach(function(cell) {
    var inp = cell.querySelector(".input");
    if (!inp) return;
    var btn = document.createElement("button");
    btn.className = "toggle-code-btn";
    btn.textContent = "{ } afficher le code";
    btn.onclick = function() {
      var hidden = inp.style.display === "none" || inp.style.display === "";
      inp.style.display = hidden ? "block" : "none";
      btn.textContent = hidden ? "{ } masquer le code" : "{ } afficher le code";
    };
    cell.insertBefore(btn, cell.firstChild);
  });
});
</script>
"""

def _notebook_to_html(nb_path: Path) -> str:
    """Convertit un .ipynb en HTML via nbconvert.

    Post-processing : injecte un toggle JS/CSS pour masquer/afficher le code.
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
    html = result.stdout.decode("utf-8")
    # Injecter le toggle juste avant </head>
    return html.replace("</head>", _NOTEBOOK_TOGGLE_CSS_JS + "</head>", 1)


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


# ─────────────────────────────────────────────────────────────────────────────
# Cellules de méthodologie injectées avant les étapes clés du notebook
# Clé = sous-chaîne cherchée dans le code de l'étape ; valeur = texte Markdown
# ─────────────────────────────────────────────────────────────────────────────
_NOTEBOOK_METHODOLOGY: list[tuple[str, str]] = [
    ("data_prep.load_data", """\
### Méthodologie — Chargement des données

Le fichier de données brutes (CSV / TXT / Excel) est importé et normalisé.
Les noms de colonnes sont standardisés vers les noms canoniques du pipeline :
`date_naissance`, `date_entree`, `date_sortie`, `sexe`, `cause_sortie`.
Un mapping utilisateur permet de gérer les colonnes et valeurs non-standards."""),

    ("data_prep.clean_data", """\
### Méthodologie — Nettoyage et validation des données

**Critères d'inclusion :**
- Cohérence temporelle : $t_{naissance} < t_{entrée} < t_{sortie}$
- Âge d'entrée dans l'intervalle admissible $[x_{min},\\ x_{max}]$
- `cause_sortie` ∈ {`deces`, `autre`}
- `sexe` ∈ {`H`, `F`}

Les individus ne satisfaisant pas ces critères sont exclus et tracés dans le rapport de nettoyage."""),

    ("data_prep.compute_ages", """\
### Méthodologie — Calcul des âges et durées d'observation

Âge en **années révolues** (anniversaire exact) :

$$x_i = \\lfloor t_{obs} - t_{naissance,i} \\rfloor$$

Durée d'exposition individuelle :

$$\\delta_i = \\min(t_{sortie,i},\\ \\tau) - t_{entrée,i}$$

où $\\tau$ est la date de fin d'observation."""),

    ("data_prep.detect_anomalies", """\
### Méthodologie — Détection d'anomalies structurelles

Contrôles effectués :
- **Doublons** : individus avec le même identifiant ou le même triplet (date naissance, date entrée, sexe)
- **Valeurs manquantes** sur les colonnes clés
- **Incohérences temporelles** : $t_{entrée} > t_{sortie}$, âge négatif
- **Outliers** : taux brut par âge hors de $[\\mu_x^{ref}/10;\\ 10 \\cdot \\mu_x^{ref}]$"""),

    ("exposure.compute_exposure_by_age", """\
### Méthodologie — Calcul de l'exposition en années-personnes

**Exposition centrale** à l'âge entier $x$ :

$$E_x^c = \\sum_{i} \\bigl[\\min(t_{sortie,i},\\ x+1) - \\max(t_{entrée,i},\\ x)\\bigr]^+$$

**Exposition initiale** (base du taux $q_x$) :

$$E_x = E_x^c + \\tfrac{1}{2}\\, d_x$$

où $d_x$ est le nombre de décès observés à l'âge $x$."""),

    ("crude_rates.crude_ra", """\
### Méthodologie — Taux bruts de mortalité

**Estimateur de Kaplan-Meier** (adapté aux petits effectifs) :

$$\\hat{S}(t) = \\prod_{t_i \\leq t} \\Bigl(1 - \\frac{d_i}{n_i}\\Bigr)$$

**Taux central brut** et passage au taux initial :

$$\\hat{\\mu}_x = \\frac{d_x}{E_x^c} \\qquad \\hat{q}_x = 1 - e^{-\\hat{\\mu}_x} \\approx \\frac{d_x}{E_x}$$

Les âges avec $E_x^c < E_{min}$ sont signalés comme peu crédibles."""),

    ("diagnostics.diagnose_credibility", """\
### Méthodologie — Diagnostic de crédibilité

Seuil de Bühlmann : $E_x^c \\geq E_{min}$ (typiquement 5–10 années-personnes).

Le coefficient de variation des taux bruts détecte l'instabilité locale.
Les âges sous-représentés reçoivent un traitement différencié lors du lissage."""),

    ("smoothing_selector.auto_select", """\
### Méthodologie — Sélection automatique du lisseur

| Méthode | Paramètres | Critère |
|---------|-----------|---------|
| **Whittaker-Henderson** | $\\lambda$, ordre $n$ | AIC, monotonie |
| **Gompertz** | $a$, $b$ | Log-vraisemblance |
| **Makeham** | $a$, $b$, $c$ | Log-vraisemblance |

Critère de sélection :
$$AIC = -2\\,\\ell(\\hat{\\theta}) + 2k$$
Le lisseur retenu minimise l'AIC tout en présentant le minimum de violations de monotonie."""),

    ("smoothing.smooth_whittaker", """\
### Méthodologie — Lissage de Whittaker-Henderson

Minimisation de la fonction pénalisée :

$$S(\\mathbf{z}) = \\sum_x w_x (z_x - y_x)^2 + \\lambda \\sum_x (\\Delta^n z_x)^2$$

où $y_x = \\hat{q}_x^{brut}$, $w_x = E_x^c$, $\\Delta^n$ = différence finie d'ordre $n$.

Solution analytique :

$$\\mathbf{z}^* = (W + \\lambda D^T D)^{-1} W\\, \\mathbf{y}$$

$\\lambda$ faible → fidélité aux données ; $\\lambda$ élevé → lissé mais risque d'effacer les tendances réelles."""),

    ("validation.confiden", """\
### Méthodologie — Intervalles de confiance à 95 %

Variance asymptotique (approximation binomiale) :

$$\\widehat{\\mathrm{Var}}(\\hat{q}_x) = \\frac{\\hat{q}_x\\,(1-\\hat{q}_x)}{E_x}$$

Intervalle de confiance normal :

$$IC_{95\\%}(q_x) = \\hat{q}_x \\pm 1{,}96\\,\\sqrt{\\widehat{\\mathrm{Var}}(\\hat{q}_x)}$$"""),

    ("validation.chi_square_tes", """\
### Méthodologie — Test du $\\chi^2$ d'adéquation

Décès attendus sous la table de référence : $A_x = E_x^c \\cdot \\mu_x^{ref}$

Statistique de test :

$$\\chi^2 = \\sum_x \\frac{(O_x - A_x)^2}{A_x} \\sim \\chi^2(k)$$

$k$ = nombre de classes d'âge avec $A_x \\geq 5$.
La p-value teste $H_0 : q_x = q_x^{ref}$."""),

    ("diagnostics.compute_smr", """\
### Méthodologie — Ratio Standardisé de Mortalité (SMR)

$$SMR = \\frac{\\displaystyle\\sum_x d_x}{\\displaystyle\\sum_x E_x^c \\cdot \\mu_x^{ref}}$$

**Interprétation :** $SMR < 1$ → sélection favorable ; $SMR > 1$ → surmortalité relative.

Intervalle de confiance à 95 % (approximation de Poisson) :

$$IC_{95\\%}(SMR) = SMR \\pm 1{,}96 \\cdot \\frac{SMR}{\\sqrt{\\sum_x d_x}}$$"""),
]


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

        # ── Cellule méthodologie (injectée avant le code pour les étapes clés) ─
        if code and success:
            for snippet, meth_text in _NOTEBOOK_METHODOLOGY:
                if snippet in code:
                    cells.append(nbformat.v4.new_markdown_cell(meth_text))
                    break

        # ── Cellule code avec outputs embarqués ────────────────────────────────
        if code:
            if not success:
                prefix = (
                    f"# ❌ Tentative échouée — non rejouable\n"
                    f"# Erreur : {output[:120].splitlines()[0] if output else '?'}\n\n"
                )
                code_cell = nbformat.v4.new_code_cell(prefix + code)
                if output:
                    code_cell.outputs = [nbformat.v4.new_output(
                        output_type="stream", name="stderr", text=output)]
            else:
                code_cell = nbformat.v4.new_code_cell(code)
                nb_outputs = []
                # Outputs stream (stdout)
                stream_text = output
                display_outputs = step.get("display_outputs", [])
                # Retirer du stream_text la partie déjà dans display_outputs
                if stream_text:
                    nb_outputs.append(nbformat.v4.new_output(
                        output_type="stream", name="stdout", text=stream_text))
                # Outputs display_data (DataFrames HTML)
                for d in display_outputs:
                    if d.get("html"):
                        nb_outputs.append(nbformat.v4.new_output(
                            output_type="display_data",
                            data={"text/html": d["html"],
                                  "text/plain": d["text"]},
                            metadata={}))
                    elif d.get("text"):
                        nb_outputs.append(nbformat.v4.new_output(
                            output_type="stream", name="stdout",
                            text=d["text"] + "\n"))
                code_cell.outputs = nb_outputs
            cells.append(code_cell)

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

UPLOADS_DIR = (_ROOT / Path(config.UPLOADS_DIR).name)
UPLOADS_DIR.mkdir(exist_ok=True)

PALETTE = [
    "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#F44336", "#607D8B", "#E91E63",
]

# ─────────────────────────────────────────────────────────────────────────────
# App Dash
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Agent Actuariel",
    suppress_callback_exceptions=True,
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
    """Onglet Agent — interface conversationnelle générique 2 colonnes."""
    _label_style = {"fontSize": "11px", "color": "#777", "marginBottom": "3px"}

    # ── Barre supérieure (topbar) ──────────────────────────────────────────
    topbar = html.Div(
        [
            # Sélecteur de domaine
            html.Div([
                html.Span("Domaine :", style={"fontSize": "11px", "color": "#777",
                                              "marginRight": "6px", "whiteSpace": "nowrap"}),
                dcc.Dropdown(
                    id="agent-domain-select",
                    options=list_domains(),
                    value="mortality",
                    clearable=False,
                    style={"fontSize": "11px", "minWidth": "200px", "maxWidth": "280px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginRight": "12px"}),
            # Sexe (domaine mortalité)
            html.Div([
                html.Span("Sexe :", style={"fontSize": "11px", "color": "#777",
                                           "marginRight": "6px", "whiteSpace": "nowrap"}),
                dcc.Dropdown(
                    id="agent-sexe-select",
                    options=[{"label": "H", "value": "H"}, {"label": "F", "value": "F"}],
                    value="H", clearable=False,
                    style={"fontSize": "11px", "width": "70px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginRight": "12px"}),
            # Séparateur
            html.Div(style={"flex": "1"}),
            # Pas-à-pas
            dbc.Switch(id="toggle-stepbystep", label="Pas-à-pas", value=False,
                       style={"fontSize": "11px"}, className="me-2"),
            # Bouton Plan+Agent
            dbc.Button("🗓 Plan+Agent", id="btn-plan-execute", color="info", size="sm",
                       outline=True, disabled=True, className="me-1",
                       title="Exécute l'agent en suivant le plan du Canvas"),
            # System Prompt
            dbc.Button("⚙ Prompt", id="btn-show-prompt", color="secondary", size="sm",
                       outline=True, className="me-1"),
            # Config (template rapport, etc.)
            dbc.Button("📋 Config", id="btn-toggle-agent-config", color="secondary",
                       size="sm", outline=True, className="me-1"),
            # Stop
            dbc.Button("⏹ Stop", id="btn-stop-agent", color="danger", size="sm",
                       outline=True, disabled=True, className="me-1"),
            # Status
            html.Div(id="agent-run-status",
                     style={"fontSize": "11px", "color": "#777",
                            "whiteSpace": "nowrap", "marginRight": "8px"}),
            # Toggle notebook panel
            dbc.Button("◧ Notebook", id="btn-collapse-notebook", color="secondary",
                       size="sm", outline=True),
        ],
        id="chat-topbar",
        style={
            "display": "flex", "alignItems": "center", "flexWrap": "wrap",
            "gap": "4px", "padding": "6px 10px",
            "background": "#F0EDE3", "borderBottom": "1px solid #C5BDB0",
            "flexShrink": "0",
        },
    )

    # ── Panneau config dépliable (template rapport + options avancées) ─────
    config_panel = dbc.Collapse(
        html.Div(
            [
                html.Hr(style={"borderColor": "#C5BDB0", "margin": "4px 0"}),
                dbc.Row([
                    # Template rapport JSON
                    dbc.Col([
                        html.P("Template rapport (.json)", style=_label_style),
                        dcc.Upload(
                            id="upload-agent-template",
                            children=html.Div(["📋 Charger .json",
                                               html.Small(" (pré-remplit le prompt)",
                                                          style={"color": "#888"})]),
                            multiple=False, accept=".json",
                            style={"border": "1px dashed #A09890", "borderRadius": "6px",
                                   "padding": "5px", "textAlign": "center",
                                   "color": "#555", "fontSize": "11px",
                                   "cursor": "pointer"},
                        ),
                        html.Div(id="agent-template-status",
                                 style={"color": "#4CAF50", "fontSize": "10px",
                                        "minHeight": "14px"}),
                        dbc.Button("✕ Effacer", id="btn-clear-agent-template",
                                   color="secondary", size="sm", outline=True,
                                   style={"fontSize": "10px"}, disabled=True),
                    ], width=4),
                    # Analyse rapport PDF
                    dbc.Col([
                        html.P("Analyse rapport PDF", style=_label_style),
                        dcc.Upload(
                            id="upload-report-pdf",
                            children=html.Div(["📄 Charger PDF",
                                               html.Small(" (glisser)", style={"color": "#888"})]),
                            multiple=False, accept=".pdf",
                            style={"border": "1px dashed #A09890", "borderRadius": "6px",
                                   "padding": "5px", "textAlign": "center",
                                   "color": "#555", "fontSize": "11px",
                                   "cursor": "pointer"},
                        ),
                        html.Div(id="report-pdf-filename",
                                 style={"color": "#4CAF50", "fontSize": "10px"}),
                        dbc.Button("🔍 Analyser", id="btn-analyze-report",
                                   color="primary", size="sm",
                                   className="me-1", disabled=True),
                    ], width=4),
                    # Template JSON / export
                    dbc.Col([
                        html.P("Ou JSON analysé", style=_label_style),
                        dcc.Upload(
                            id="upload-report-template-json",
                            children=html.Div(["📂 JSON",
                                               html.Small(" (template)", style={"color": "#888"})]),
                            multiple=False, accept=".json",
                            style={"border": "1px dashed #A09890", "borderRadius": "6px",
                                   "padding": "5px", "textAlign": "center",
                                   "color": "#555", "fontSize": "11px",
                                   "cursor": "pointer"},
                        ),
                        dbc.ButtonGroup([
                            dbc.Button("📤 → Agent", id="btn-send-template-to-agent",
                                       color="success", size="sm", disabled=True),
                            dbc.Button("💾 JSON", id="btn-download-template",
                                       color="secondary", size="sm", outline=True, disabled=True),
                        ], className="mt-1"),
                        html.Div(id="template-analysis-status",
                                 style={"fontSize": "10px", "color": "#777", "minHeight": "20px"}),
                        html.Div(id="template-analysis-result",
                                 style={"maxHeight": "100px", "overflowY": "auto",
                                        "fontSize": "10px", "background": "#F5F2E7",
                                        "border": "1px solid #D8D0C4", "borderRadius": "4px",
                                        "padding": "4px"}),
                    ], width=4),
                ], className="g-2"),
                html.Hr(style={"borderColor": "#C5BDB0", "margin": "4px 0"}),
            ],
            style={"padding": "4px 10px", "background": "#F5F2E7"},
        ),
        id="collapse-agent-config",
        is_open=False,
    )

    # ── Zone de messages unifiée ───────────────────────────────────────────
    messages_area = html.Div(
        id="chat-messages-area",
        children=[
            html.P(
                "Joignez un fichier (📎) et décrivez votre analyse, ou posez une question.",
                style={"color": "#AAA", "fontSize": "13px",
                       "textAlign": "center", "marginTop": "80px"},
            )
        ],
        style={
            "flex": "1",
            "overflowY": "auto",
            "display": "flex",
            "flexDirection": "column",
            "gap": "10px",
            "padding": "12px 16px",
        },
    )

    # ── Zone de saisie du chat ─────────────────────────────────────────────
    input_area = html.Div(
        [
            # Chip fichier joint
            html.Div(id="chat-file-chip",
                     style={"minHeight": "20px", "marginBottom": "4px"}),
            # Rangée input
            dbc.Row(
                [
                    # Bouton 📎 intégré dans dcc.Upload pour déclencher le sélecteur de fichier
                    dbc.Col(
                        dcc.Upload(
                            id="upload-chat-file",
                            children=html.Span(
                                "📎",
                                title="Joindre un fichier CSV",
                                style={"cursor": "pointer", "fontSize": "20px",
                                       "lineHeight": "56px", "display": "block",
                                       "padding": "0 4px"},
                            ),
                            multiple=False,
                            style={"display": "inline-block"},
                        ),
                        width="auto",
                    ),
                    # Textarea
                    dbc.Col(
                        dcc.Textarea(
                            id="chat-text-input",
                            placeholder="Écrivez votre message… (Entrée pour envoyer, Shift+Entrée pour nouvelle ligne)",
                            style={
                                "width": "100%", "height": "56px",
                                "resize": "none", "borderRadius": "8px",
                                "border": "1px solid #C5BDB0",
                                "padding": "8px 12px", "fontSize": "13px",
                                "fontFamily": "inherit", "background": "#FAFAF5",
                                "outline": "none",
                            },
                        ),
                        width=True,
                    ),
                    # Bouton Envoyer
                    dbc.Col(
                        dbc.Button(
                            "↵",
                            id="btn-chat-send",
                            color="primary",
                            style={"height": "56px", "width": "52px",
                                   "fontSize": "18px", "borderRadius": "8px"},
                        ),
                        width="auto",
                    ),
                ],
                className="g-1 align-items-center",
            ),
        ],
        id="chat-input-area",
        style={
            "flexShrink": "0",
            "padding": "8px 16px 12px 16px",
            "borderTop": "1px solid #C5BDB0",
            "background": "#FAFAF5",
        },
    )

    # ── Colonne principale chat ────────────────────────────────────────────
    chat_col = html.Div(
        [messages_area, input_area],
        id="agent-chat-col",
        style={
            "flex": "1",
            "display": "flex",
            "flexDirection": "column",
            "overflow": "hidden",
            "minWidth": "300px",
            "background": "#FBF8F1",
        },
    )

    # ── Drag handle ────────────────────────────────────────────────────────
    drag_handle = html.Div(
        id="agent-resize-handle",
        style={
            "width": "5px",
            "cursor": "col-resize",
            "background": "#C5BDB0",
            "flexShrink": "0",
            "height": "100%",
            "transition": "background 0.2s",
        },
    )

    # ── Panneau Notebook (collapsible) ─────────────────────────────────────
    notebook_panel = html.Div(
        _notebook_tab(h_offset=88),
        id="agent-notebook-panel",
        style={
            "width": "30%",
            "minWidth": "200px",
            "maxWidth": "55%",
            "flexShrink": "0",
            "height": "100%",
            "overflow": "hidden",
            "background": "#FBF8F1",
        },
    )

    # ── Corps (chat + handle + notebook) ──────────────────────────────────
    body = html.Div(
        [chat_col, drag_handle, notebook_panel],
        style={
            "display": "flex",
            "flexDirection": "row",
            "flex": "1",
            "overflow": "hidden",
        },
    )

    return html.Div(
        [topbar, config_panel, body],
        id="agent-tab-root",
        style={
            "display": "flex",
            "flexDirection": "column",
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
                    html.Span("Agent Actuariel", className="navbar-brand",
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
            active_tab="tab-agent",
            style={"background": "#FBF8F1"},
        ),
        # Stores
        dcc.Store(id="nb-current-path-store", data=None),
        dcc.Store(id="nb-cells-store", data=[]),
        dcc.Store(id="system-prompt-store", data=SYSTEM_PROMPT_TEMPLATE),
        dcc.Store(id="csv-path-agent-store", data=None),
        dcc.Store(id="column-mapping-store", data=None),
        dcc.Store(id="mapping-validated-store", data=False),
        dcc.Store(id="rag-history-store", data=[]),
        dcc.Store(id="rag-system-prompt-store", data=RAG_SYSTEM_PROMPT),
        dcc.Store(id="rag-expanded-store", data=False),
        dcc.Store(id="unified-chat-store", data=[]),
        dcc.Store(id="domain-store", data="mortality"),
        dcc.Store(id="report-template-store", data=None),
        dcc.Store(id="nb-gen-path-store", data=None),
        dcc.Store(id="resize-init-store", data=0),
        dcc.Store(id="enter-bind-store", data=0),
        dcc.Store(id="agent-plan-store", data=[]),
        dcc.Store(id="agent-outputs-store", data={}),
        dcc.Download(id="download-agent-pdf"),
        dcc.Download(id="download-agent-trace"),
        dcc.Download(id="download-agent-notebook"),
        dcc.Download(id="download-template-json"),
        dcc.Interval(id="agent-interval", interval=800, disabled=True, n_intervals=0),
        dcc.Interval(id="template-analysis-interval", interval=1000, disabled=True, n_intervals=0),
        # Modal mapping des colonnes (remplace le panneau inline collapse)
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("⚠ Mapping des colonnes requis")),
                dbc.ModalBody(html.Div(id="column-mapping-modal-body", children=[])),
                dbc.ModalFooter([
                    dbc.Button("✓ Valider", id="btn-validate-mapping",
                               color="warning", className="me-2"),
                    dbc.Button("Ignorer", id="btn-skip-mapping",
                               color="secondary", outline=True),
                ]),
            ],
            id="modal-column-mapping",
            is_open=False,
            size="lg",
            backdrop="static",
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


# Kernels persistants pour l'éditeur de notebooks
_nb_kernels: dict = {}
_nb_kernels_lock = threading.Lock()


# Agent results store
_agent_results: dict = {}
_agent_lock = threading.Lock()

# Synchronisation pour la Q&A bidirectionnelle Agent ↔ Utilisateur
_agent_reply_event: threading.Event = threading.Event()
_agent_reply_value: str = ""
_AGENT_REPLY_TIMEOUT: int = 300  # secondes avant auto-reprise (5 min)

# RAG async results store
_rag_state: dict = {"status": "idle", "answer": "", "figures": [], "msg_idx": -1}
_rag_lock = threading.Lock()

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


def _make_wait_for_user_fn():
    """Crée un callable bloquant pour la Q&A bidirectionnelle Agent ↔ Utilisateur.

    Lorsque l'agent appelle ask_user() :
    1. Le statut passe à "waiting" et la question est stockée dans _agent_results.
    2. Le thread se bloque sur _agent_reply_event (jusqu'à réponse ou timeout).
    3. rag_send_message() détecte le statut "waiting", écrit la réponse et lève l'event.
    4. Le thread reprend avec la réponse et continue le run_agent_loop.
    """
    def wait_for_user(question: str, options: list) -> str:
        global _agent_reply_value
        with _agent_lock:
            _agent_results["status"] = "waiting"
            _agent_results["pending_question"] = question
            _agent_results["pending_options"] = options
            _agent_results["thinking"] = ""
        _agent_reply_event.clear()
        with _agent_lock:
            _agent_reply_value = ""
        # Bloquer jusqu'à réponse ou timeout
        replied = _agent_reply_event.wait(timeout=_AGENT_REPLY_TIMEOUT)
        with _agent_lock:
            reply = _agent_reply_value if replied else "continuer"
            _agent_results["status"] = "running"
            _agent_results.pop("pending_question", None)
            _agent_results.pop("pending_options", None)
        return reply
    return wait_for_user


def _run_agent_in_thread(csv_path: str, sexe: str, user_message: str,
                          system_prompt_template: str, max_steps: int = None,
                          column_mapping: dict = None,
                          value_mapping: dict = None,
                          domain_id: str = "mortality") -> None:
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
    kernel["COLUMN_MAPPING"] = column_mapping or {}
    kernel["VALUE_MAPPING"] = value_mapping or {}
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
        f"- LAMBDA_WH = 100\n"
        f"- COLUMN_MAPPING = {json.dumps(column_mapping or {})}\n"
        f"- VALUE_MAPPING = {json.dumps(value_mapping or {})}\n\n"
        f"Ces variables sont accessibles directement dans le kernel. "
        f"Commence immédiatement l'analyse sans demander de confirmation."
    )
    if column_mapping or value_mapping:
        full_message += (
            "\n\nATTENTION CRITIQUE — ÉTAPE 1 OBLIGATOIRE :\n"
            "Les colonnes du fichier ne correspondent PAS aux noms standards du pipeline.\n"
            "Tu DOIS appeler load_data EXACTEMENT ainsi :\n\n"
            f"  df_raw, summary = data_prep.load_data(\n"
            f"      FILE_PATH,\n"
            f"      column_mapping=COLUMN_MAPPING,\n"
            f"      value_mapping=VALUE_MAPPING\n"
            f"  )\n\n"
            "Sans ces paramètres, clean_data échouera avec 'Missing required columns'.\n"
            "PUIS appelle clean_data EXACTEMENT ainsi :\n\n"
            f"  df_clean, report = data_prep.clean_data(\n"
            f"      df_raw,\n"
            f"      date_fin_observation=DATE_FIN_OBSERVATION\n"
            f"  )\n"
        )

    _ACTUARY_LOGGER.clear()
    _inspector_session_id = None  # identifiant de session partagé entre toutes les étapes

    # Charger le contexte KB et le prompt selon le domaine
    from pathlib import Path as _Path
    from domain_config import get_domain as _get_domain
    _domain_cfg = _get_domain(domain_id)
    _kb_dir = _Path(__file__).parent / _domain_cfg["kb_dir"]
    _kb_context = load_knowledge_base_context(kb_dir=_kb_dir)
    _sp = system_prompt_template  # utilise le prompt passé par l'appelant (déjà résolu par send_chat_message)

    # ── Phase 1 : Planning ────────────────────────────────────────────────
    # Construire le contexte data pour le planner
    data_context = (
        f"Fichier CSV : {csv_path}\n"
        f"Sexe : {sexe}\n"
        f"Domaine : {domain_id}\n"
    )

    with _agent_lock:
        _agent_results["status"] = "planning"
        _agent_results["steps"] = []
        _agent_results["thinking"] = ""

    try:
        plan = plan_agent(
            user_message=full_message,
            kb_context=_kb_context[:3000],  # tronquer pour le planner
            data_context=data_context,
        )
    except Exception:
        plan = []

    if not plan:
        # Fallback : lancer directement sans plan
        pass
    else:
        # ── Phase 2 : Demande de confirmation du plan ─────────────────────
        with _agent_lock:
            _agent_results["status"] = "waiting"
            _agent_results["pending_question"] = "Voici le plan d'analyse proposé. Cochez les étapes à exécuter :"
            _agent_results["pending_options"] = [
                f"**{s['titre']}** — {s['description']}" for s in plan
            ]
            _agent_results["pending_question_type"] = "checklist"
            _agent_results["pending_plan"] = plan

        # Attendre la réponse de l'utilisateur
        _agent_reply_event.wait(timeout=_AGENT_REPLY_TIMEOUT)
        _agent_reply_event.clear()

        with _agent_lock:
            reply = _agent_reply_value
            _agent_results["status"] = "running"

        # ── Boucle de replan si l'utilisateur demande un affinement ─────────
        _replan_attempts = 0
        while reply.startswith("REPLAN:") and _replan_attempts < 3:
            _replan_attempts += 1
            extra_context = reply[len("REPLAN:"):].strip()
            with _agent_lock:
                _agent_results["status"] = "planning"
                _agent_results["status_detail"] = "🔄 Régénération du plan avec vos compléments…"
            enriched_data_context = data_context + (
                f"\n\nCompléments utilisateur : {extra_context}" if extra_context else ""
            )
            try:
                plan = plan_agent(
                    user_message=full_message,
                    kb_context=_kb_context[:3000],
                    data_context=enriched_data_context,
                )
            except Exception:
                pass
            with _agent_lock:
                _agent_results["status"] = "waiting"
                _agent_results["pending_question"] = (
                    "Plan révisé. Cochez les étapes à exécuter :"
                )
                _agent_results["pending_options"] = [
                    f"**{s['titre']}** — {s['description']}" for s in plan
                ]
                _agent_results["pending_question_type"] = "checklist"
                _agent_results["pending_plan"] = plan
                _agent_results["status_detail"] = ""
            _agent_reply_event.wait(timeout=_AGENT_REPLY_TIMEOUT)
            _agent_reply_event.clear()
            with _agent_lock:
                reply = _agent_reply_value
                _agent_results["status"] = "running"

        # Parser les étapes sélectionnées (reply = "1,2,3[|COMMENT:...]" ou "all")
        extra_comment = ""
        if "|COMMENT:" in reply:
            reply, extra_comment = reply.split("|COMMENT:", 1)
            extra_comment = extra_comment.strip()
        if reply and reply.strip().lower() != "all":
            try:
                selected_ids = set(int(x.strip()) for x in reply.split(",") if x.strip().isdigit())
                plan = [s for s in plan if s["id"] in selected_ids]
            except Exception:
                pass  # garder le plan complet si parse échoue

        # Injecter le plan dans le system prompt
        if plan:
            plan_context = "\n\nPLAN D'ANALYSE APPROUVÉ PAR L'UTILISATEUR :\n" + "\n".join(
                f"{s['id']}. {s['titre']} ({s.get('methode','')}) : {s['description']}"
                for s in plan
            )
            plan_context += (
                "\n\nEXÉCUTE CE PLAN DANS L'ORDRE. Ne saute aucune étape approuvée."
                " Commence directement par l'étape 1 sans re-planifier."
            )
            if extra_comment:
                plan_context += f"\n\nINSTRUCTION ADDITIONNELLE DE L'UTILISATEUR : {extra_comment}"
                plan_context += "\nAdapte le code custom si nécessaire (ex: table unisexe, recodage de variables)."
                plan_context += " Marque les steps de code custom avec un commentaire # CUSTOM dans le code."
            _sp = _sp + plan_context

        # Stocker le plan approuvé dans les résultats
        with _agent_lock:
            _agent_results["approved_plan"] = plan

    try:
        for event in run_agent_loop(
            user_message=full_message,
            notebook_context=_kb_context,
            conversation_history=[],
            execute_fn=execute_fn,
            system_prompt_template=_sp,
            max_steps=max_steps,
            wait_for_user_fn=_make_wait_for_user_fn(),
            kb_dir=_kb_dir,
        ):
            with _agent_lock:
                if event["type"] == "step":
                    figs_b64 = [base64.b64encode(f).decode()
                                for f in event.get("figures", [])]
                    desc = event.get("description", "")
                    is_custom = "# CUSTOM" in event.get("code", "")
                    _agent_results["steps"].append({
                        "description": desc,
                        "code": event.get("code", ""),
                        "output": event.get("output", ""),
                        "figures": figs_b64,
                        "success": not event.get("output", "").startswith("❌"),
                        "display_outputs": kernel.pop("_last_display_outputs", []),
                        "custom": is_custom,
                    })
                    step_num = len(_agent_results["steps"])
                    _agent_results["status_detail"] = f"⚙ Étape {step_num} terminée : {desc[:60]}"
                    _agent_results["thinking"] = ""  # effacer bulle "thinking"
                elif event["type"] == "thinking":
                    _msg = event.get("message", "Réflexion…")
                    _agent_results["status_detail"] = f"💭 {_msg[:80]}"
                    _agent_results["thinking"] = _msg
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
                    _agent_results["status"] = "generating_report"
                    _agent_results["thinking"] = ""
                elif event["type"] == "error":
                    _agent_results["summary"] = event.get("content", "")
                    _agent_results["status"] = "error"
                    _agent_results["thinking"] = ""
    except Exception as exc:
        with _agent_lock:
            _agent_results["summary"] = f"Erreur agent : {exc}"
            _agent_results["status"] = "error"

    # ── Phase 4 : Génération des outputs ─────────────────────────────────
    try:
        from pathlib import Path as _OutPath
        from datetime import datetime as _dt
        _outputs_dir = _OutPath(__file__).parent / "outputs"
        _outputs_dir.mkdir(exist_ok=True)
        _ts = _dt.now().strftime("%Y%m%d_%H%M%S")

        with _agent_lock:
            _steps = list(_agent_results.get("steps", []))
            _summary = _agent_results.get("summary", "")
            _approved_plan = _agent_results.get("approved_plan", [])

        _pdf_path = str(_outputs_dir / f"rapport_{_ts}.pdf")
        _trace_path = str(_outputs_dir / f"trace_{_ts}.md")
        _nb_path = str(_outputs_dir / f"notebook_{_ts}.ipynb")

        # Récupérer le prompt rédacteur issu de l'encodeur (peut être None)
        _tpl = _ACTUARY_STATE.get_template()
        _writer_prompt = (_tpl.get("agent_system_prompt") or None) if _tpl else None

        generate_narrative_report(
            _steps, _summary, full_message, domain_id, _pdf_path,
            study_ref=f"Analyse {_ts}",
            writer_prompt=_writer_prompt,
            template_sections=_tpl.get("sections", []) if _tpl else None,
            methodology=_tpl.get("methodology") if _tpl else None,
        )
        generate_reasoning_trace(_steps, _summary, full_message, _approved_plan, _trace_path)
        generate_final_notebook(_steps, full_message, _nb_path)

        with _agent_lock:
            _agent_results["pdf_path"] = _pdf_path
            _agent_results["trace_path"] = _trace_path
            _agent_results["notebook_path"] = _nb_path
            _agent_results["status"] = "done"
    except Exception as _e:
        import traceback as _tb2
        print(f"[outputs] Erreur génération : {_e}\n{_tb2.format_exc()}", flush=True)
        with _agent_lock:
            _agent_results["status"] = "done"

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


def _run_rag_in_thread(question: str, all_steps: list, exec_ns: dict,
                       sp_rag: str, rag_history: list, summary: str,
                       msg_idx: int) -> None:
    """Exécute le RAG dans un thread background et stocke le résultat."""
    with _rag_lock:
        _rag_state["status"] = "running"
        _rag_state["msg_idx"] = msg_idx
        _rag_state["answer"] = ""
        _rag_state["figures"] = []
    try:
        has_data = any(k in exec_ns for k in ("df", "df_clean", "df_exposure", "df_qx", "df_smooth"))
        if has_data:
            answer, figures = answer_with_tools(
                question=question,
                steps=all_steps,
                exec_ns=exec_ns,
                state=_ACTUARY_STATE,
                summary=summary,
                system_prompt=sp_rag,
                conversation_history=rag_history,
            )
            figs_b64 = [base64.b64encode(f).decode() for f in figures]
        else:
            answer = answer_with_rag(
                question=question,
                steps=all_steps,
                summary=summary,
                system_prompt=RAG_SYSTEM_PROMPT,
                conversation_history=rag_history,
                state=_ACTUARY_STATE,
            )
            figs_b64 = []
        _ACTUARY_STATE.update_rag_ns(exec_ns)
    except Exception as exc:
        import traceback as _tb
        answer = f"❌ Erreur RAG : {exc}\n\n```\n{_tb.format_exc()}\n```"
        figs_b64 = []
    with _rag_lock:
        _rag_state["status"] = "done"
        _rag_state["answer"] = answer
        _rag_state["figures"] = figs_b64


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
# Utilitaire — détection des problèmes de mapping à l'import
# ─────────────────────────────────────────────────────────────────────────────

def _detect_mapping_issues(path: str) -> dict:
    """Lit les 200 premières lignes du CSV et détecte les colonnes/valeurs non-standards.

    Retourne un dict vide ({}) si tout est OK, sinon un dict décrivant les problèmes.
    """
    import pandas as _pd_det
    import importlib.util as _ilu_det

    REQUIRED = ["date_naissance", "date_entree", "date_sortie", "cause_sortie", "sexe"]
    VALID_SEXE = {"H", "F"}
    VALID_CAUSE = {"deces", "autre"}

    # Chargement du dictionnaire de synonymes depuis 01_data_preparation.py
    try:
        _dp_spec = _ilu_det.spec_from_file_location(
            "_dp_det", str(_ROOT / "notebooks" / "01_data_preparation.py")
        )
        _dp_mod = _ilu_det.module_from_spec(_dp_spec)
        _dp_spec.loader.exec_module(_dp_mod)
        _SYNONYMS = _dp_mod._COLUMN_SYNONYMS
    except Exception:
        _SYNONYMS = {}

    # Lecture d'un échantillon — détection automatique du séparateur et de l'encodage
    import csv as _csv_det
    sample = None
    for _enc in ("utf-8", "latin-1", "cp1252"):
        for _sep in (None, ";", ",", "\t"):  # None = laisser Sniffer choisir
            try:
                if _sep is None:
                    try:
                        with open(path, encoding=_enc, errors="replace") as _fh:
                            _dialect = _csv_det.Sniffer().sniff(_fh.read(4096), delimiters=";,\t|")
                        _sep = _dialect.delimiter
                    except Exception:
                        continue
                _df_try = _pd_det.read_csv(path, sep=_sep, nrows=200, encoding=_enc, engine="python")
                if len(_df_try.columns) > 1:
                    sample = _df_try
                    break
            except Exception:
                continue
        if sample is not None:
            break
    if sample is None:
        return {}

    # Normalisation des noms de colonnes (même logique que normalize_column_names)
    normalized: dict[str, str | None] = {}
    for col in sample.columns:
        norm = col.strip().lower().replace(" ", "_").replace("-", "_")
        canonical = _SYNONYMS.get(norm)
        normalized[col] = canonical  # None si pas de correspondance

    # Canoniques déjà présents directement
    direct_canonicals = {
        c.strip().lower().replace(" ", "_").replace("-", "_")
        for c in sample.columns
    }
    available = (
        {v for v in normalized.values() if v}
        | {c for c in REQUIRED if c in direct_canonicals}
    )
    unmapped_required = [r for r in REQUIRED if r not in available]

    issues: dict = {}
    needs_mapping = False

    if unmapped_required:
        issues["unmapped_required"] = unmapped_required
        issues["all_cols"] = list(sample.columns)
        needs_mapping = True

    # Suggestions automatiques par similarité de nom pour les colonnes requises non trouvées
    _NAME_HINTS = {
        "date_naissance": {"naiss", "birth", "dob", "naissance", "born", "nee", "bdate"},
        "date_entree":    {"entree", "entry", "effet", "start", "debut", "adhesion",
                           "souscript", "contrat", "effect", "ouverture"},
        "date_sortie":    {"sortie", "exit", "fin", "end", "death", "deces", "clot"},
        "cause_sortie":   {"statut", "status", "cause", "reason", "exit_type", "motif"},
        "sexe":           {"sexe", "sex", "gender", "genre", "ref"},
    }
    suggested: dict[str, str] = {}
    unmapped_cols = [c for c in sample.columns if normalized.get(c) is None]
    for req in unmapped_required:
        hints = _NAME_HINTS.get(req, set())
        for col in unmapped_cols:
            col_lower = col.lower()
            if any(h in col_lower for h in hints):
                suggested[req] = col
                break
    if suggested:
        issues["suggested_mapping"] = suggested

    # Colonne sexe détectée (via synonymes puis heuristique sur le contenu)
    sexe_col_raw = next(
        (orig for orig, can in normalized.items() if can == "sexe"),
        "sexe" if "sexe" in sample.columns else None,
    )
    # Heuristique : si sexe non trouvé via synonymes, scanner les colonnes non reconnues
    if sexe_col_raw is None and "sexe" in unmapped_required:
        _SEXE_HINTS = {"1", "2", "m", "f", "h", "homme", "femme", "male", "female", "man", "woman"}
        for _col in sample.columns:
            if normalized.get(_col) is not None:
                continue
            _vals = {str(v).strip().lower() for v in sample[_col].dropna().unique()}
            if _vals and _vals <= _SEXE_HINTS and 1 <= len(_vals) <= 3:
                sexe_col_raw = _col
                break

    if sexe_col_raw and sexe_col_raw in sample.columns:
        # Alimenter la suggestion de colonne avec ce qu'on vient de détecter par contenu
        if "sexe" in unmapped_required:
            issues.setdefault("suggested_mapping", {}). \
                setdefault("sexe", sexe_col_raw)
        unique_sexe = [str(v) for v in sample[sexe_col_raw].dropna().unique()]
        if not set(unique_sexe).issubset(VALID_SEXE):
            issues["sexe_col_raw"] = sexe_col_raw
            issues["sexe_values"] = sorted(unique_sexe)
            issues["sexe_needs_mapping"] = True
            needs_mapping = True

    # Colonne cause_sortie détectée (via synonymes puis heuristique sur le contenu)
    cause_col_raw = next(
        (orig for orig, can in normalized.items() if can == "cause_sortie"),
        "cause_sortie" if "cause_sortie" in sample.columns else None,
    )
    # Heuristique : si cause_sortie non trouvée via synonymes, scanner les colonnes non reconnues
    if cause_col_raw is None and "cause_sortie" in unmapped_required:
        _CAUSE_HINTS = {"actif", "sorti", "decede", "decedes", "deces", "décédé",
                        "autre", "death", "alive", "active", "lapse", "lapsed",
                        "sorti", "0", "1", "2", "d", "a"}
        for _col in sample.columns:
            if normalized.get(_col) is not None or _col == sexe_col_raw:
                continue
            _vals = {str(v).strip().lower() for v in sample[_col].dropna().unique()}
            if _vals and _vals <= _CAUSE_HINTS and 1 <= len(_vals) <= 6:
                cause_col_raw = _col
                break

    if cause_col_raw and cause_col_raw in sample.columns:
        if "cause_sortie" in unmapped_required:
            issues.setdefault("suggested_mapping", {}). \
                setdefault("cause_sortie", cause_col_raw)
        unique_cause = [str(v) for v in sample[cause_col_raw].dropna().unique()]
        if not set(unique_cause).issubset(VALID_CAUSE):
            issues["cause_col_raw"] = cause_col_raw
            issues["cause_values"] = sorted(unique_cause)
            issues["cause_needs_mapping"] = True
            needs_mapping = True

    if not needs_mapping:
        return {}

    issues.setdefault("all_cols", list(sample.columns))
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Agent : upload fichier via le chat
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("csv-path-agent-store", "data"),
    Output("chat-file-chip", "children"),
    Output("column-mapping-store", "data"),
    Output("mapping-validated-store", "data"),
    Output("modal-column-mapping", "is_open"),
    Input("upload-chat-file", "contents"),
    State("upload-chat-file", "filename"),
    prevent_initial_call=True,
)
def handle_chat_upload(contents, filename):
    if contents is None:
        return None, "", None, False, False
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    save_path = str((UPLOADS_DIR / filename).resolve())
    with open(save_path, "wb") as f:
        f.write(decoded)
    issues = _detect_mapping_issues(save_path)
    chip = dbc.Badge(
        [filename, " ",
         dbc.Button("✕", id="btn-clear-chat-file", color="link", size="sm",
                    style={"padding": "0 4px", "fontSize": "10px", "color": "#fff",
                           "fontWeight": "bold", "lineHeight": "1"})],
        color="success",
        style={"fontSize": "11px", "padding": "4px 8px", "cursor": "default"},
    )
    if issues:
        return save_path, chip, issues, False, True
    return save_path, chip, {}, True, False


@app.callback(
    Output("csv-path-agent-store", "data", allow_duplicate=True),
    Output("chat-file-chip", "children", allow_duplicate=True),
    Output("mapping-validated-store", "data", allow_duplicate=True),
    Input("btn-clear-chat-file", "n_clicks"),
    prevent_initial_call=True,
)
def clear_chat_file(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    return None, html.Span(), False


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Modal de mapping des colonnes
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("column-mapping-modal-body", "children"),
    Input("column-mapping-store", "data"),
    prevent_initial_call=True,
)
def populate_mapping_modal(mapping_data):
    if not mapping_data:
        return []
    # Store already holds the validated form — don't re-populate
    if "column_mapping" in mapping_data or "value_mapping" in mapping_data:
        return []

    children = [
        html.Div("Certaines colonnes ou valeurs du fichier ne correspondent pas aux noms standards du pipeline.",
                 style={"marginBottom": "12px", "fontSize": "12px", "color": "#856404"}),
    ]

    all_cols = mapping_data.get("all_cols", [])
    col_options = [{"label": c, "value": c} for c in all_cols]
    suggested = mapping_data.get("suggested_mapping", {})

    # Colonnes requises non trouvées
    for req_col in mapping_data.get("unmapped_required", []):
        suggestion = suggested.get(req_col)
        children.append(html.Div([
            html.Label(f"Colonne « {req_col} » → quelle colonne du fichier ?",
                       style={"fontSize": "12px", "marginBottom": "4px"}),
            dcc.Dropdown(
                id={"type": "col-map-dd", "index": req_col},
                options=col_options,
                value=suggestion,
                placeholder="Sélectionner…" if not suggestion else None,
                style={"fontSize": "12px", "marginBottom": "8px",
                       "border": "1px solid #28a745" if suggestion else None},
            ),
            html.Div(f"✓ Suggestion automatique : {suggestion}",
                     style={"fontSize": "11px", "color": "#28a745",
                            "marginTop": "-6px", "marginBottom": "8px"}) if suggestion else None,
        ]))

    # Valeurs sexe non-standard
    if mapping_data.get("sexe_needs_mapping"):
        children.append(html.P(
            f"Colonne « {mapping_data.get('sexe_col_raw', 'sexe')} » — mapper les valeurs de sexe :",
            style={"fontSize": "12px", "fontWeight": "bold", "marginTop": "8px", "marginBottom": "6px"},
        ))
        for val in mapping_data.get("sexe_values", []):
            children.append(dbc.Row([
                dbc.Col(html.Label(f"{val!r} →", style={"fontSize": "12px"}), width=3),
                dbc.Col(dcc.Dropdown(
                    id={"type": "val-map-dd", "index": f"sexe||{val}"},
                    options=[{"label": "H (Homme)", "value": "H"},
                             {"label": "F (Femme)", "value": "F"}],
                    value="H" if val.upper() in ("M", "H", "HOMME", "MALE", "1") else "F",
                    clearable=False, style={"fontSize": "12px"},
                ), width=4),
            ], className="mb-2 align-items-center"))

    # Valeurs cause_sortie non-standard
    if mapping_data.get("cause_needs_mapping"):
        children.append(html.P(
            f"Colonne « {mapping_data.get('cause_col_raw', 'cause_sortie')} » — mapper les causes de sortie :",
            style={"fontSize": "12px", "fontWeight": "bold", "marginTop": "8px", "marginBottom": "6px"},
        ))
        for val in mapping_data.get("cause_values", []):
            children.append(dbc.Row([
                dbc.Col(html.Label(f"{val!r} →", style={"fontSize": "12px"}), width=3),
                dbc.Col(dcc.Dropdown(
                    id={"type": "val-map-dd", "index": f"cause_sortie||{val}"},
                    options=[{"label": "deces", "value": "deces"},
                             {"label": "autre", "value": "autre"}],
                    value="deces" if val.lower() in ("deces", "decede", "decedes", "death", "décès", "décédé", "mort", "1", "d") else "autre",
                    clearable=False, style={"fontSize": "12px"},
                ), width=4),
            ], className="mb-2 align-items-center"))

    return children


@app.callback(
    Output("column-mapping-store", "data", allow_duplicate=True),
    Output("mapping-validated-store", "data", allow_duplicate=True),
    Output("modal-column-mapping", "is_open", allow_duplicate=True),
    Input("btn-validate-mapping", "n_clicks"),
    Input("btn-skip-mapping", "n_clicks"),
    State({"type": "col-map-dd", "index": dash.ALL}, "value"),
    State({"type": "col-map-dd", "index": dash.ALL}, "id"),
    State({"type": "val-map-dd", "index": dash.ALL}, "value"),
    State({"type": "val-map-dd", "index": dash.ALL}, "id"),
    State("column-mapping-store", "data"),
    prevent_initial_call=True,
)
def validate_or_skip_mapping(validate_clicks, skip_clicks,
                              col_values, col_ids, val_values, val_ids, current_data):
    from dash import callback_context as _ctx
    triggered = _ctx.triggered[0]["prop_id"] if _ctx.triggered else ""

    if "btn-skip-mapping" in triggered:
        return {}, True, False

    if "btn-validate-mapping" not in triggered:
        return dash.no_update, dash.no_update, dash.no_update

    current_data = current_data or {}

    # Assemblage du mapping de colonnes
    column_mapping: dict = {}
    for v, id_dict in zip(col_values, col_ids):
        if v:
            req_col = id_dict["index"]
            column_mapping[v] = req_col   # raw_col → canonical

    # Assemblage du mapping de valeurs
    value_mapping: dict = {}
    for v, id_dict in zip(val_values, val_ids):
        if v is None:
            continue
        idx = id_dict["index"]          # e.g. "sexe||M" or "cause_sortie||death"
        parts = idx.split("||", 1)
        if len(parts) != 2:
            continue
        col_canon, raw_val = parts
        value_mapping.setdefault(col_canon, {})[raw_val] = v

    current_data["column_mapping"] = column_mapping
    current_data["value_mapping"] = value_mapping
    return current_data, True, False


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
# Callbacks — Chat : envoi de message (unifié agent + RAG)
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("unified-chat-store", "data"),
    Output("chat-messages-area", "children"),
    Output("chat-text-input", "value"),
    Output("agent-interval", "disabled"),
    Output("agent-run-status", "children"),
    Output("btn-chat-send", "disabled"),
    Output("btn-stop-agent", "disabled"),
    Input("btn-chat-send", "n_clicks"),
    State("chat-text-input", "value"),
    State("unified-chat-store", "data"),
    State("csv-path-agent-store", "data"),
    State("agent-domain-select", "value"),
    State("agent-sexe-select", "value"),
    State("system-prompt-store", "data"),
    State("toggle-stepbystep", "value"),
    State("column-mapping-store", "data"),
    prevent_initial_call=True,
)
def send_chat_message(n_clicks, text, history, csv_path,
                      domain_id, sexe, system_prompt, stepbystep, mapping_data):
    global _agent_reply_value

    if not text or not text.strip():
        return (dash.no_update,) * 7

    history = list(history or [])
    text = text.strip()

    # ── Cas 1 : agent en attente d'une réponse (ask_user) ─────────────────
    with _agent_lock:
        agent_status = _agent_results.get("status", "")

    if agent_status == "waiting":
        with _agent_lock:
            _agent_reply_value = text
        _agent_reply_event.set()
        history.append({"role": "user", "content": text, "figures": [], "options": []})
        history.append({"role": "assistant_rag",
                        "content": f"*(Réponse transmise à l'agent : « {text} »)*",
                        "figures": [], "options": []})
        return (history, _build_unified_chat_messages(history),
                "", dash.no_update, dash.no_update, dash.no_update, dash.no_update)

    # ── Cas 2 : pas de CSV → question RAG (async) ─────────────────────────
    if not csv_path and agent_status not in ("running", "waiting", "generating_report"):
        history.append({"role": "user", "content": text, "figures": [], "options": []})
        # Placeholder immédiat — remplacé par refresh_agent_results quand le thread finit
        history.append({"role": "assistant_rag", "content": "⏳ *Réflexion en cours…*",
                        "figures": [], "options": []})
        msg_idx = len(history) - 1
        all_steps, summary = _get_rag_context()
        exec_ns = _ACTUARY_STATE.get_exec_namespace()
        sp_rag = system_prompt or RAG_TOOLS_SYSTEM_PROMPT
        rag_history_fmt = [
            {"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]}
            for m in history[:-1]  # exclure le placeholder
        ]
        threading.Thread(
            target=_run_rag_in_thread,
            args=(text, all_steps, exec_ns, sp_rag, rag_history_fmt, summary, msg_idx),
            daemon=True,
        ).start()
        return (history, _build_unified_chat_messages(history), "", False,
                "💬 RAG en cours…", False, True)

    # ── Cas 3 : CSV présent + agent idle → lancer une nouvelle analyse ────
    if agent_status not in ("running", "waiting", "generating_report"):
        history.append({"role": "user", "content": text, "figures": [], "options": []})
        # Charger le prompt selon le domaine
        domain_id = domain_id or "mortality"
        sp = _load_domain_prompt(domain_id) or (system_prompt or SYSTEM_PROMPT_TEMPLATE)
        max_steps = 1 if stepbystep else None
        col_mapping = (mapping_data or {}).get("column_mapping", {})
        val_mapping = (mapping_data or {}).get("value_mapping", {})

        # Si un template encodeur est chargé, injecter la liste des éléments requis
        # dans le message pour que l'agent de calcul sache quoi produire.
        _tpl = _ACTUARY_STATE.get_template()
        _enriched_text = text
        if _tpl:
            _required = _build_required_elements_note(_tpl)
            if _required:
                _enriched_text = text + "\n\n" + _required

        with _agent_lock:
            _agent_results.clear()
            _agent_results["status"] = "running"
            _agent_results["steps"] = []
            _agent_results["summary"] = ""
            _agent_results["csv_path"] = csv_path
            _agent_results["sexe"] = sexe or "H"
        threading.Thread(
            target=_run_agent_in_thread,
            args=(csv_path, sexe or "H", _enriched_text, sp, max_steps, col_mapping, val_mapping),
            kwargs={"domain_id": domain_id},
            daemon=True,
        ).start()
        return (history, _build_unified_chat_messages(history),
                "", False, "⏳ Agent en cours…", True, False)

    # Agent déjà en cours → ignorer (l'utilisateur peut attendre ou stop)
    return (dash.no_update,) * 7


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Agent : rafraîchissement résultats (polling)
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("unified-chat-store", "data", allow_duplicate=True),
    Output("chat-messages-area", "children", allow_duplicate=True),
    Output("agent-interval", "disabled", allow_duplicate=True),
    Output("agent-run-status", "children", allow_duplicate=True),
    Output("btn-chat-send", "disabled", allow_duplicate=True),
    Output("btn-stop-agent", "disabled", allow_duplicate=True),
    Input("agent-interval", "n_intervals"),
    State("unified-chat-store", "data"),
    prevent_initial_call=True,
)
def refresh_agent_results(n, history):
    import time as _time_mod

    # ── Cas RAG async : remplacer le placeholder quand le thread a fini ───────
    with _rag_lock:
        rag_status = _rag_state["status"]
        rag_answer = _rag_state["answer"]
        rag_figures = list(_rag_state["figures"])
        rag_msg_idx = _rag_state["msg_idx"]

    if rag_status == "done" and rag_msg_idx >= 0:
        history = list(history or [])
        if rag_msg_idx < len(history):
            history[rag_msg_idx] = {
                "role": "assistant_rag",
                "content": rag_answer,
                "figures": rag_figures,
                "options": [],
            }
        with _rag_lock:
            _rag_state["status"] = "idle"
            _rag_state["msg_idx"] = -1
        with _agent_lock:
            ag_status = _agent_results.get("status", "")
        agent_running = ag_status in ("running", "waiting", "generating_report")
        return (history, _build_unified_chat_messages(history),
                not agent_running, "" if not agent_running else "⏳ Agent en cours…",
                agent_running, not agent_running)

    with _agent_lock:
        results = dict(_agent_results)

    if not results:
        return (dash.no_update,) * 6

    history = list(history or [])
    steps = results.get("steps", [])
    status = results.get("status", "running")
    summary = results.get("summary", "")

    # Indices déjà présents dans l'historique
    existing_step_indices = {
        e.get("step_index")
        for e in history
        if e.get("role") == "agent_step" and e.get("step_index") is not None
    }
    new_entries = []

    # Nouveaux steps
    for i, step in enumerate(steps):
        if i not in existing_step_indices:
            output = step.get("output", "")
            new_entries.append({
                "role": "agent_step",
                "step_index": i,
                "content": step.get("description", ""),
                "code": step.get("code", ""),
                "output": output,
                "figures": step.get("figures", []),
                "display_outputs": step.get("display_outputs", []),
                "success": "❌" not in output,
                "options": [],
                "timestamp": _time_mod.time(),
            })

    # Question de l'agent si status == "waiting"
    if status == "waiting":
        pending_q = results.get("pending_question", "")
        pending_opts = results.get("pending_options", [])
        pending_q_type = results.get("pending_question_type", "choice")
        pending_plan = results.get("pending_plan", [])
        already_injected = any(
            e.get("role") == "agent_question" and e.get("content") == pending_q
            for e in history
        )
        if pending_q and not already_injected:
            # Utiliser step_index unique basé sur la longueur de l'historique courant
            _q_step_idx = len(history) + len(new_entries)
            entry = {
                "role": "agent_question",
                "content": pending_q,
                "options": pending_opts,
                "question_type": pending_q_type,
                "figures": [],
                "timestamp": _time_mod.time(),
                "step_index": _q_step_idx,
            }
            if pending_q_type == "checklist" and pending_plan:
                entry["plan"] = pending_plan
            new_entries.append(entry)

    # Résumé final — uniquement quand les chemins sont disponibles (après Phase 4)
    if summary and status in ("done", "error"):
        already_has_summary = any(e.get("role") == "agent_summary" for e in history)
        if not already_has_summary:
            new_entries.append({
                "role": "agent_summary",
                "content": summary,
                "success": status == "done",
                "figures": [],
                "options": [],
                "downloads": {
                    "pdf": results.get("pdf_path", ""),
                    "notebook": results.get("notebook_path", ""),
                    "trace": results.get("trace_path", ""),
                },
                "timestamp": _time_mod.time(),
            })

    history = history + new_entries
    done = status in ("done", "error")
    status_detail = results.get("status_detail", "")
    status_msg = {
        "done": "✓ Terminé",
        "error": "✗ Erreur",
        "waiting": "⏸ En attente de votre validation…",
        "planning": "🗺 Génération du plan…",
        "generating_report": "📝 Rédaction du rapport…",
    }.get(status, status_detail or "⏳ En cours…")

    thinking_msg = results.get("thinking", "")

    if status == "waiting":
        # Ne pas re-rendre : l'utilisateur interagit avec la checklist
        chat_children = (
            _build_unified_chat_messages(history) if new_entries
            else dash.no_update
        )
    elif new_entries or (thinking_msg and status == "running") or status == "generating_report":
        bubbles = _build_unified_chat_messages(history)
        if thinking_msg and status == "running":
            bubbles = list(bubbles) + [_build_thinking_bubble(thinking_msg)]
        elif status == "generating_report":
            bubbles = list(bubbles) + [_build_thinking_bubble("Rédaction narrative du rapport en cours…")]
        chat_children = bubbles
    else:
        chat_children = dash.no_update

    return (history, chat_children, done, status_msg, not done, done)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — construction des bulles de chat
# ─────────────────────────────────────────────────────────────────────────────
def _render_display_outputs(display_outputs: list) -> list:
    """Convertit les display_outputs (DataFrames capturés) en composants Dash."""
    items = []
    for do in display_outputs:
        html_content = do.get("html")
        text_content = do.get("text", "")
        if html_content:
            items.append(html.Div(
                dash_dangerously_set_inner_html.DangerouslySetInnerHTML(  # type: ignore[attr-defined]
                    __html=html_content
                ) if False else  # pragma: no cover
                # Fallback: rendu HTML via iframe ou dcc.Markdown
                dcc.Markdown(
                    f"```\n{text_content[:2000]}\n```" if text_content else "",
                    style={"fontSize": "11px", "overflowX": "auto"},
                ),
                style={"overflowX": "auto", "marginTop": "6px"},
            ))
        elif text_content:
            items.append(html.Pre(
                text_content[:2000],
                style={"fontSize": "11px", "background": "#F5F2E7",
                       "borderRadius": "4px", "padding": "6px",
                       "overflowX": "auto", "marginTop": "6px"},
            ))
    return items


def _build_thinking_bubble(message: str):
    """Bulle animée 'thinking' affichée pendant que l'agent réfléchit."""
    return html.Div(
        [
            html.Div("🤖 Agent", style={"fontSize": "10px", "color": "#999", "marginBottom": "3px"}),
            html.Div(
                [
                    html.Span("💭 ", style={"fontSize": "14px"}),
                    html.Em(message, style={"color": "#666", "fontSize": "12px"}),
                    html.Span(" ●●●", style={
                        "color": "#AAA",
                        "fontSize": "12px",
                        "marginLeft": "4px",
                        "animation": "thinking-dots 1.2s infinite",
                    }),
                ],
                style={
                    "background": "#F8F8F8",
                    "border": "1px dashed #CCC",
                    "borderRadius": "4px 12px 12px 12px",
                    "padding": "8px 12px",
                    "maxWidth": "75%",
                    "alignSelf": "flex-start",
                    "fontStyle": "italic",
                },
            ),
        ],
        style={"display": "flex", "flexDirection": "column", "alignItems": "flex-start",
               "marginBottom": "8px"},
    )


def _build_unified_chat_messages(history: list[dict]) -> list:
    """Construit les bulles de la conversation unifiée (agent + RAG).

    Gère 6 rôles :
      user          → bulle verte droite
      assistant_rag → bulle blanche gauche (réponse RAG)
      agent_step    → carte beige/rouge avec toggle code + figures + DataFrames
      agent_question→ bulle jaune + boutons d'options
      agent_summary → carte verte/rouge finale
      system        → texte gris centré (info système)
    """
    bubbles = []
    for msg in history:
        role = msg.get("role", "assistant_rag")
        content = msg.get("content", "")
        figures = msg.get("figures", [])
        options = msg.get("options", [])

        # ── user ──────────────────────────────────────────────────────────────
        if role == "user":
            bubbles.append(html.Div(
                [
                    html.Div("Vous", style={"fontSize": "10px", "color": "#999",
                                            "marginBottom": "3px", "textAlign": "right"}),
                    html.Div(
                        dcc.Markdown(content, style={"margin": "0", "fontSize": "13px",
                                                     "lineHeight": "1.6"}),
                        style={"background": "#D4EDDA", "border": "1px solid #B8DACC",
                               "borderRadius": "12px 4px 12px 12px",
                               "padding": "10px 14px", "maxWidth": "85%",
                               "alignSelf": "flex-end",
                               "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"},
                    ),
                ],
                style={"display": "flex", "flexDirection": "column", "alignItems": "flex-end"},
            ))

        # ── agent_question ────────────────────────────────────────────────────
        elif role == "agent_question":
            question_type = msg.get("question_type", "choice")
            step_idx = msg.get("step_index", 0)
            plan = msg.get("plan", [])

            if question_type == "checklist" and plan:
                # Rendu checklist pour confirmation du plan
                checklist_options = []
                for s in plan:
                    methode = s.get("methode", "")
                    formule = s.get("formule", "")
                    alternatives = s.get("alternatives", "")
                    is_custom = s.get("custom_code", False)
                    detail_parts = []
                    if methode:
                        detail_parts.append(html.Span(f"⚙ {methode}", style={"color": "#555", "fontSize": "11px"}))
                    if formule:
                        detail_parts.append(html.Code(f"  {formule}", style={"fontSize": "11px", "background": "#F0EDE0", "padding": "1px 4px", "borderRadius": "3px"}))
                    if alternatives:
                        detail_parts.append(html.Span(f"  (alt: {alternatives})", style={"color": "#888", "fontSize": "10px", "fontStyle": "italic"}))
                    label = html.Div([
                        html.Span([
                            html.Strong(f"{s['id']}. {s['titre']}"),
                            html.Span(" 🔧 code custom" if is_custom else "", style={"color": "#0066CC", "fontSize": "10px", "marginLeft": "4px"}),
                            html.Span(" (obligatoire)" if s.get("obligatoire") else "", style={"color": "#B05010", "fontSize": "10px", "marginLeft": "4px"}),
                        ]),
                        html.Div(s.get("description", ""), style={"color": "#555", "fontSize": "11px", "marginTop": "2px"}),
                        html.Div(detail_parts, style={"marginTop": "3px"}) if detail_parts else None,
                    ])
                    checklist_options.append({"label": label, "value": s["id"]})

                all_ids = [s["id"] for s in plan]

                checklist_card = html.Div([
                    dcc.Markdown(content, style={"marginBottom": "8px"}),
                    dbc.Checklist(
                        id={"type": "plan-checklist", "index": step_idx},
                        options=checklist_options,
                        value=all_ids,
                        style={"marginBottom": "12px"},
                    ),
                    # Zone de texte libre pour compléments / ajustements
                    html.Div([
                        html.Label("Compléments ou ajustements (optionnel) :",
                                   style={"fontSize": "11px", "color": "#666", "marginBottom": "4px"}),
                        dcc.Textarea(
                            id={"type": "plan-comment", "index": step_idx},
                            placeholder="Ex : table unisexe souhaitable, exclure les âges > 80, appliquer une correction de sélection…",
                            style={"width": "100%", "height": "60px", "fontSize": "12px",
                                   "resize": "vertical", "borderRadius": "6px",
                                   "border": "1px solid #D4C89A", "padding": "6px 8px",
                                   "background": "#FFFDF5", "fontFamily": "inherit"},
                        ),
                    ], style={"marginBottom": "10px"}),
                    html.Div([
                        dbc.Button(
                            "▶ Lancer l'analyse",
                            id={"type": "btn-confirm-plan", "index": step_idx},
                            color="warning", size="sm", className="me-2",
                        ),
                        dbc.Button(
                            "🔄 Affiner le plan",
                            id={"type": "btn-replan", "index": step_idx},
                            color="secondary", size="sm", outline=True,
                        ),
                    ]),
                ], style={
                    "background": "#FFF8E8",
                    "border": "1px solid #E8B84B",
                    "borderRadius": "8px",
                    "padding": "12px 16px",
                    "maxWidth": "740px",
                    "marginBottom": "8px",
                })

                bubbles.append(html.Div(
                    [
                        html.Div("🤖 Agent", style={"fontSize": "10px", "color": "#999",
                                                    "marginBottom": "3px"}),
                        checklist_card,
                    ],
                    style={"display": "flex", "flexDirection": "column", "alignItems": "flex-start"},
                ))
            else:
                # Rendu standard avec boutons de choix
                btn_children = [
                    dbc.Button(opt, id={"type": "agent-option-btn", "index": i},
                               size="sm", color="warning", outline=True,
                               className="me-1 mt-1", style={"fontSize": "11px"})
                    for i, opt in enumerate(options)
                ]
                bubbles.append(html.Div(
                    [
                        html.Div("🤖 Agent", style={"fontSize": "10px", "color": "#999",
                                                    "marginBottom": "3px"}),
                        html.Div(
                            [
                                dcc.Markdown(content, style={"margin": "0", "fontSize": "13px",
                                                             "lineHeight": "1.6"}),
                                html.Div(btn_children, style={"marginTop": "6px"}) if btn_children else None,
                            ],
                            style={"background": "#FFF3CD", "border": "1px solid #F0C040",
                                   "borderRadius": "4px 12px 12px 12px",
                                   "padding": "10px 14px", "maxWidth": "90%",
                                   "alignSelf": "flex-start",
                                   "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"},
                        ),
                    ],
                    style={"display": "flex", "flexDirection": "column", "alignItems": "flex-start"},
                ))

        # ── assistant_rag ─────────────────────────────────────────────────────
        elif role == "assistant_rag":
            fig_elems = [
                html.Img(src=f"data:image/png;base64,{f}",
                         style={"maxWidth": "100%", "borderRadius": "6px",
                                "marginTop": "8px", "display": "block"})
                for f in figures
            ]
            bubbles.append(html.Div(
                [
                    html.Div("RAG", style={"fontSize": "10px", "color": "#999",
                                           "marginBottom": "3px"}),
                    html.Div(
                        [dcc.Markdown(content, style={"margin": "0", "fontSize": "13px",
                                                      "lineHeight": "1.6"})] + fig_elems,
                        style={"background": "#FFFFFF", "border": "1px solid #C5BDB0",
                               "borderRadius": "4px 12px 12px 12px",
                               "padding": "10px 14px", "maxWidth": "90%",
                               "alignSelf": "flex-start",
                               "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"},
                    ),
                ],
                style={"display": "flex", "flexDirection": "column", "alignItems": "flex-start"},
            ))

        # ── agent_step ────────────────────────────────────────────────────────
        elif role == "agent_step":
            step_idx = msg.get("step_index", 0)
            success = msg.get("success", True)
            code = msg.get("code", "")
            output = msg.get("output", "")
            disp_outputs = msg.get("display_outputs", [])
            is_custom = msg.get("custom", False)
            step_figs = [
                html.Img(src=f"data:image/png;base64,{f}",
                         style={"maxWidth": "100%", "borderRadius": "6px",
                                "marginTop": "8px", "display": "block"})
                for f in figures
            ]
            card_bg = "#EEF4FF" if is_custom else ("#F5F2E8" if success else "#FFF0F0")
            card_children = [
                # Header
                html.Div([
                    html.Span(f"Étape {step_idx + 1}  ",
                              style={"fontSize": "10px", "color": "#999"}),
                    html.Span("✅" if success else "❌",
                              style={"marginRight": "6px"}),
                    html.Span("🔧 CODE CUSTOM  " if is_custom else "",
                              style={"fontSize": "10px", "color": "#0066CC",
                                     "fontWeight": "bold", "marginRight": "4px"}),
                    html.Strong(content[:120],
                                style={"fontSize": "12px", "color": "#2D2D2D"}),
                ], style={"marginBottom": "4px"}),
            ]
            # Output preview
            if output:
                card_children.append(html.Pre(
                    output[:800] + ("…" if len(output) > 800 else ""),
                    style={"fontSize": "10px", "background": "#FAFAFA",
                           "border": "1px solid #E0E0E0", "borderRadius": "4px",
                           "padding": "4px 8px", "overflowX": "auto",
                           "maxHeight": "120px", "overflowY": "auto",
                           "marginBottom": "4px"},
                ))
            # Code toggle
            if code:
                card_children.append(html.Div([
                    dbc.Button(
                        "{ } afficher le code", size="sm", color="link",
                        id={"type": "btn-toggle-code", "index": step_idx},
                        style={"fontSize": "10px", "padding": "0", "marginBottom": "2px"},
                    ),
                    dbc.Collapse(
                        html.Pre(code,
                                 style={"background": "#2b2b2b", "color": "#f8f8f2",
                                        "fontSize": "10px", "borderRadius": "4px",
                                        "padding": "8px", "overflowX": "auto",
                                        "maxHeight": "300px", "overflowY": "auto"}),
                        id={"type": "collapse-code", "index": step_idx},
                        is_open=False,
                    ),
                ]))
            # DataFrames
            card_children.extend(_render_display_outputs(disp_outputs))
            # Figures
            card_children.extend(step_figs)

            bubbles.append(html.Div(
                [
                    html.Div("🤖 Agent", style={"fontSize": "10px", "color": "#999",
                                                "marginBottom": "3px"}),
                    html.Div(
                        card_children,
                        style={
                            "background": card_bg,
                            "border": f"1px solid {'#A8C4F9' if is_custom else ('#C5BDB0' if success else '#F9A8A8')}",
                            "borderRadius": "4px 12px 12px 12px",
                            "padding": "10px 14px", "maxWidth": "95%",
                            "alignSelf": "flex-start",
                            "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
                        },
                    ),
                ],
                style={"display": "flex", "flexDirection": "column", "alignItems": "flex-start"},
            ))

        # ── agent_summary ─────────────────────────────────────────────────────
        elif role == "agent_summary":
            success = msg.get("success", True)
            downloads = msg.get("downloads", {})
            summary_children = [
                dcc.Markdown(content, style={"margin": "0", "fontSize": "13px",
                                             "lineHeight": "1.6"}),
            ]
            # Boutons de téléchargement si des outputs ont été générés
            if downloads and any(downloads.values()):
                dl_buttons = []
                if downloads.get("pdf"):
                    dl_buttons.append(
                        dbc.Button("Télécharger le rapport PDF",
                                   id="btn-dl-pdf", color="success", size="sm",
                                   className="me-2")
                    )
                if downloads.get("notebook"):
                    dl_buttons.append(
                        dbc.Button("Télécharger le notebook",
                                   id="btn-dl-nb", color="primary", size="sm",
                                   className="me-2")
                    )
                if downloads.get("trace"):
                    dl_buttons.append(
                        dbc.Button("Télécharger la trace",
                                   id="btn-dl-trace", color="secondary", size="sm")
                    )
                if dl_buttons:
                    summary_children.append(
                        html.Div(dl_buttons, style={"marginTop": "8px"})
                    )

            bubbles.append(html.Div(
                [
                    html.Div("Synthèse", style={"fontSize": "10px", "color": "#999",
                                                "marginBottom": "3px"}),
                    html.Div(
                        summary_children,
                        style={
                            "background": "#E8F5E9" if success else "#FFEBEE",
                            "border": f"1px solid {'#A5D6A7' if success else '#FFCDD2'}",
                            "borderRadius": "4px 12px 12px 12px",
                            "padding": "14px", "maxWidth": "95%",
                            "alignSelf": "flex-start",
                            "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
                        },
                    ),
                ],
                style={"display": "flex", "flexDirection": "column", "alignItems": "flex-start"},
            ))

        # ── system ────────────────────────────────────────────────────────────
        elif role == "system":
            bubbles.append(html.Div(
                content,
                style={"textAlign": "center", "color": "#AAA",
                       "fontSize": "11px", "padding": "4px 0"},
            ))

    return bubbles


def _build_chat_messages(history: list[dict]) -> list:
    bubbles = []
    for msg in history:
        role = msg.get("role", "assistant")
        is_user = role == "user"
        is_agent = role == "agent"   # question posée par l'agent via ask_user
        figures = msg.get("figures", [])
        options = msg.get("options", [])

        # Couleur et alignement selon le rôle
        if is_user:
            label, bg, border, radius, align = (
                "Vous", "#D4EDDA", "#B8DACC", "12px 4px 12px 12px", "flex-end"
            )
        elif is_agent:
            label, bg, border, radius, align = (
                "🤖 Agent", "#FFF3CD", "#F0C040", "4px 12px 12px 12px", "flex-start"
            )
        else:
            label, bg, border, radius, align = (
                "RAG", "#FFFFFF", "#C5BDB0", "4px 12px 12px 12px", "flex-start"
            )

        bubble_children = [
            dcc.Markdown(msg["content"],
                         style={"margin": "0", "fontSize": "13px",
                                "lineHeight": "1.6"}),
        ]
        # Boutons d'options pour les questions de l'agent
        if is_agent and options:
            bubble_children.append(html.Div(
                [
                    dbc.Button(
                        opt,
                        id={"type": "agent-option-btn", "index": i},
                        size="sm", color="warning", outline=True,
                        className="me-1 mt-1",
                        style={"fontSize": "11px"},
                    )
                    for i, opt in enumerate(options)
                ],
                style={"marginTop": "6px"},
            ))
        # Figures inline
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
                    label,
                    style={"fontSize": "10px", "color": "#999",
                           "marginBottom": "3px",
                           "textAlign": "right" if is_user else "left"},
                ),
                html.Div(
                    bubble_children,
                    style={
                        "background": bg,
                        "border": f"1px solid {border}",
                        "borderRadius": radius,
                        "padding": "10px 14px",
                        "maxWidth": "90%",
                        "alignSelf": align,
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
                    },
                ),
            ],
            style={"display": "flex", "flexDirection": "column",
                   "alignItems": align},
        )
        bubbles.append(bubble)
    return bubbles


def _get_rag_context() -> tuple[list[dict], str]:
    """Construit les steps et le summary depuis les résultats disponibles + logs."""
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
    return ag_steps + log_steps + pdf_steps + src_steps, ag_summary or tpl_summary


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
# Callback — Options de l'agent : envoi direct de la réponse
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("unified-chat-store", "data", allow_duplicate=True),
    Output("chat-messages-area", "children", allow_duplicate=True),
    Input({"type": "agent-option-btn", "index": dash.ALL}, "n_clicks"),
    State({"type": "agent-option-btn", "index": dash.ALL}, "children"),
    State("unified-chat-store", "data"),
    prevent_initial_call=True,
)
def select_agent_option(n_clicks_list, option_labels, history):
    global _agent_reply_value
    if not any(n for n in n_clicks_list if n):
        return dash.no_update, dash.no_update
    from dash import callback_context as _ctx
    if not _ctx.triggered:
        return dash.no_update, dash.no_update
    triggered_prop = _ctx.triggered[0]["prop_id"]
    import json as _json
    try:
        triggered_id = _json.loads(triggered_prop.split(".")[0])
        idx = triggered_id["index"]
        chosen = option_labels[idx]
    except Exception:
        return dash.no_update, dash.no_update

    history = list(history or [])
    with _agent_lock:
        agent_status = _agent_results.get("status", "")
    if agent_status == "waiting":
        with _agent_lock:
            _agent_reply_value = chosen
        _agent_reply_event.set()
        import time as _time_mod
        history.append({"role": "user", "content": chosen, "figures": [], "options": [],
                        "timestamp": _time_mod.time()})
    return history, _build_unified_chat_messages(history)


# ─────────────────────────────────────────────────────────────────────────────
# Callback — Confirmation du plan d'analyse (checklist) + Affiner le plan
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("unified-chat-store", "data", allow_duplicate=True),
    Output("chat-messages-area", "children", allow_duplicate=True),
    Input({"type": "btn-confirm-plan", "index": dash.ALL}, "n_clicks"),
    Input({"type": "btn-replan", "index": dash.ALL}, "n_clicks"),
    State({"type": "plan-checklist", "index": dash.ALL}, "value"),
    State({"type": "plan-comment", "index": dash.ALL}, "value"),
    State("unified-chat-store", "data"),
    prevent_initial_call=True,
)
def confirm_plan(confirm_clicks, replan_clicks, selected_ids_list, comments_list, history):
    global _agent_reply_value
    import time as _time_plan

    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    triggered_id = ctx.triggered[0]["prop_id"]
    is_replan = "btn-replan" in triggered_id
    is_confirm = "btn-confirm-plan" in triggered_id
    if not is_replan and not is_confirm:
        return dash.no_update, dash.no_update

    # Index du bouton cliqué
    clicks_list = replan_clicks if is_replan else confirm_clicks
    if not any(clicks_list):
        return dash.no_update, dash.no_update
    triggered_idx = next((i for i, n in enumerate(clicks_list) if n), 0)

    selected_ids = selected_ids_list[triggered_idx] if triggered_idx < len(selected_ids_list) else []
    comment = (comments_list[triggered_idx] or "").strip() if triggered_idx < len(comments_list) else ""

    history = list(history or [])

    if is_replan:
        # Envoyer "REPLAN:<commentaire>" pour que l'agent régénère un plan enrichi
        reply = f"REPLAN:{comment}" if comment else "REPLAN:"
        with _agent_lock:
            _agent_reply_value = reply
        _agent_reply_event.set()
        history.append({
            "role": "user",
            "content": f"🔄 Demande d'affinement du plan" + (f" : *{comment}*" if comment else ""),
            "figures": [], "options": [], "timestamp": _time_plan.time(),
        })
    else:
        # Lancer l'analyse — transmettre IDs + commentaire éventuel
        ids_str = ",".join(str(i) for i in sorted(selected_ids)) if selected_ids else "all"
        reply = f"{ids_str}|COMMENT:{comment}" if comment else ids_str
        with _agent_lock:
            _agent_reply_value = reply
        _agent_reply_event.set()
        n_selected = len(selected_ids)
        msg = f"▶ Plan validé — {n_selected} étape(s) sélectionnée(s)"
        if comment:
            msg += f"\n\n*Instruction additionnelle : {comment}*"
        history.append({
            "role": "user",
            "content": msg,
            "figures": [], "options": [], "timestamp": _time_plan.time(),
        })

    return history, _build_unified_chat_messages(history)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Téléchargement des outputs générés
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("download-agent-pdf", "data"),
    Input("btn-dl-pdf", "n_clicks"),
    prevent_initial_call=True,
)
def download_pdf(n):
    with _agent_lock:
        path = _agent_results.get("pdf_path", "")
    if not path or not Path(path).exists():
        return dash.no_update
    return dcc.send_file(path)


@app.callback(
    Output("download-agent-trace", "data"),
    Input("btn-dl-trace", "n_clicks"),
    prevent_initial_call=True,
)
def download_trace(n):
    with _agent_lock:
        path = _agent_results.get("trace_path", "")
    if not path or not Path(path).exists():
        return dash.no_update
    return dcc.send_file(path)


@app.callback(
    Output("download-agent-notebook", "data"),
    Input("btn-dl-nb", "n_clicks"),
    prevent_initial_call=True,
)
def download_notebook(n):
    with _agent_lock:
        path = _agent_results.get("notebook_path", "")
    if not path or not Path(path).exists():
        return dash.no_update
    return dcc.send_file(path)


# ─────────────────────────────────────────────────────────────────────────────
# Callback — Effacer la conversation
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("unified-chat-store", "data", allow_duplicate=True),
    Output("chat-messages-area", "children", allow_duplicate=True),
    Input("btn-rag-clear", "n_clicks"),
    prevent_initial_call=True,
)
def clear_unified_chat(_):
    _ACTUARY_STATE.reset_rag_ns()
    placeholder = html.Div(
        "Conversation effacée.",
        style={"color": "#AAA", "fontSize": "13px",
               "textAlign": "center", "marginTop": "60px"},
    )
    return [], [placeholder]


# ─────────────────────────────────────────────────────────────────────────────
# Callback — Collapse/expand panneau notebook
# ─────────────────────────────────────────────────────────────────────────────
_NOTEBOOK_PANEL_VISIBLE = {
    "width": "30%", "minWidth": "200px", "maxWidth": "55%",
    "flexShrink": "0", "height": "100%", "overflow": "hidden",
    "background": "#FBF8F1",
}
_NOTEBOOK_PANEL_HIDDEN = {"display": "none"}


@app.callback(
    Output("agent-notebook-panel", "style"),
    Output("btn-collapse-notebook", "children"),
    Input("btn-collapse-notebook", "n_clicks"),
    State("agent-notebook-panel", "style"),
    prevent_initial_call=True,
)
def toggle_notebook_panel(_n, current_style):
    if current_style and current_style.get("display") == "none":
        return _NOTEBOOK_PANEL_VISIBLE, "◧ Notebook"
    return _NOTEBOOK_PANEL_HIDDEN, "□ Notebook"


# Callback — Toggle panneau config agent
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("collapse-agent-config", "is_open"),
    Input("btn-toggle-agent-config", "n_clicks"),
    State("collapse-agent-config", "is_open"),
    prevent_initial_call=True,
)
def toggle_agent_config(_n, is_open):
    return not (is_open or False)


# Callback — Chargement du domaine → mise à jour du system prompt
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("system-prompt-store", "data", allow_duplicate=True),
    Output("system-prompt-textarea", "value", allow_duplicate=True),
    Input("agent-domain-select", "value"),
    prevent_initial_call=True,
)
def load_domain_config(domain_id):
    sp = _load_domain_prompt(domain_id or "mortality")
    if sp is None:
        sp = SYSTEM_PROMPT_TEMPLATE
    return sp, sp


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
# Helpers — template encodeur
# ─────────────────────────────────────────────────────────────────────────────

def _build_required_elements_note(template: dict) -> str:
    """Génère une note listant les éléments requis par le template encodeur.

    Cette note est injectée dans le message utilisateur envoyé à l'agent de calcul
    pour qu'il sache quels tableaux et graphiques produire avant de conclure.
    """
    lines: list[str] = []
    title = template.get("report_title", "")
    if title:
        lines.append(f"RAPPORT CIBLE : {title}")
        lines.append("─" * 40)

    tables = template.get("tables", [])
    figures = template.get("figures", [])

    if tables:
        lines.append("TABLEAUX OBLIGATOIRES à produire avant de conclure :")
        for t in tables:
            cols = ", ".join(t.get("columns", [])[:6])
            lines.append(f"  □ {t.get('id', '?')} — {t.get('name', '?')}"
                         + (f" (colonnes : {cols})" if cols else ""))

    if figures:
        lines.append("GRAPHIQUES OBLIGATOIRES à produire avant de conclure :")
        for f in figures:
            lines.append(f"  □ {f.get('id', '?')} — {f.get('title', '?')}"
                         + f" (x={f.get('x_axis', '?')}, y={f.get('y_axis', '?')})")

    meth = template.get("methodology", {})
    if meth:
        parts = []
        if meth.get("smoother"):
            parts.append(f"lissage {meth['smoother']}"
                         + (f" λ={meth['lambda']}" if meth.get("lambda") else ""))
        if meth.get("reference_table"):
            parts.append(f"référence {meth['reference_table']}")
        if meth.get("age_min") and meth.get("age_max"):
            parts.append(f"âges {meth['age_min']}–{meth['age_max']} ans")
        if parts:
            lines.append("MÉTHODE IMPOSÉE : " + ", ".join(parts))

    return "\n".join(lines) if lines else ""




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
    """Charge un template.json exporté par l'encodeur.

    Le champ agent_system_prompt est destiné au sous-agent RÉDACTEUR (report_agent),
    pas à l'agent de calcul. On le stocke via _ACTUARY_STATE sans toucher au
    system-prompt-store de l'agent de calcul.
    """
    if not contents:
        return dash.no_update, dash.no_update, dash.no_update, True
    try:
        _, b64 = contents.split(",", 1)
        template = json.loads(base64.b64decode(b64).decode("utf-8"))
    except Exception as exc:
        return dash.no_update, dash.no_update, f"❌ {exc}", True
    writer_prompt = template.get("agent_system_prompt", "")
    if not writer_prompt:
        return dash.no_update, dash.no_update, "⚠ Pas de prompt rédacteur dans ce template", True
    _ACTUARY_STATE.set_template(template)
    title = template.get("report_title", filename)
    # Le system-prompt-store de l'agent de calcul reste inchangé.
    return dash.no_update, dash.no_update, f"✓ {title} — prompt rédacteur chargé", False


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
    """Envoie un template analysé vers l'onglet Agent.

    Le prompt rédacteur (agent_system_prompt) est stocké dans _ACTUARY_STATE
    pour être transmis au report_agent en fin d'analyse.
    L'agent de calcul garde son prompt standard (SYSTEM_PROMPT_TEMPLATE).
    """
    if not template:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, True
    writer_prompt = template.get("agent_system_prompt", "")
    if not writer_prompt:
        return dash.no_update, dash.no_update, dash.no_update, "⚠ Pas de prompt rédacteur", True
    _ACTUARY_STATE.set_template(template)
    title = template.get("report_title", "?")
    # Navigation vers tab-agent ; system-prompt-store inchangé.
    return "tab-agent", dash.no_update, dash.no_update, f"✓ {title} — prompt rédacteur chargé", False


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
    Input("main-tabs", "active_tab"),
    prevent_initial_call=True,
)
def refresh_nb_picker(active_tab):
    """Rafraîchit la liste des notebooks quand l'onglet Agent devient actif."""
    if active_tab != "tab-agent":
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
# Drag-to-resize + auto-scroll + drag-and-drop sur la textarea
# ─────────────────────────────────────────────────────────────────────────────
app.clientside_callback(
    """
    function(active_tab) {
        if (active_tab !== 'tab-agent') return window.dash_clientside.no_update;
        if (window._agentResizeReady) return window.dash_clientside.no_update;

        function _setup() {
            var handle    = document.getElementById('agent-resize-handle');
            var chatCol   = document.getElementById('agent-chat-col');
            var notebookPanel = document.getElementById('agent-notebook-panel');
            if (!handle || !chatCol || !notebookPanel) {
                setTimeout(_setup, 300);
                return;
            }
            window._agentResizeReady = true;

            // ── Drag-to-resize ─────────────────────────────────────────────
            var isResizing = false, startX = 0, startW = 0;

            handle.addEventListener('mousedown', function(e) {
                isResizing = true;
                startX = e.clientX;
                startW = notebookPanel.getBoundingClientRect().width;
                document.body.style.cursor = 'col-resize';
                document.body.style.userSelect = 'none';
                e.preventDefault();
            });
            document.addEventListener('mousemove', function(e) {
                if (!isResizing) return;
                var delta = startX - e.clientX;
                var newW = Math.max(200, Math.min(800, startW + delta));
                notebookPanel.style.width = newW + 'px';
                notebookPanel.style.minWidth = newW + 'px';
                handle.style.background = '#A09890';
            });
            document.addEventListener('mouseup', function() {
                if (!isResizing) return;
                isResizing = false;
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
                handle.style.background = '#C5BDB0';
            });

            // ── Auto-scroll chat-messages-area ─────────────────────────────
            var observer = new MutationObserver(function() {
                var el = document.getElementById('chat-messages-area');
                if (el) el.scrollTop = el.scrollHeight;
            });
            var messagesEl = document.getElementById('chat-messages-area');
            if (messagesEl) observer.observe(messagesEl, {childList: true, subtree: true});

            // ── Drag-and-drop sur la textarea ──────────────────────────────
            function _setupDrop() {
                var ta = document.getElementById('chat-text-input');
                if (!ta || ta._ddReady) return;
                ta._ddReady = true;
                ta.addEventListener('dragover', function(e) {
                    e.preventDefault(); e.stopPropagation();
                    ta.style.borderColor = '#4CAF50';
                });
                ta.addEventListener('dragleave', function(e) {
                    ta.style.borderColor = '#C5BDB0';
                });
                ta.addEventListener('drop', function(e) {
                    e.preventDefault(); e.stopPropagation();
                    ta.style.borderColor = '#C5BDB0';
                    var files = e.dataTransfer.files;
                    if (!files.length) return;
                    var uploadInput = document.querySelector('#upload-chat-file input[type=file]');
                    if (uploadInput) {
                        try {
                            var dt = new DataTransfer();
                            dt.items.add(files[0]);
                            uploadInput.files = dt.files;
                            uploadInput.dispatchEvent(new Event('change', {bubbles: true}));
                        } catch(err) { console.warn('drop failed', err); }
                    }
                });
            }
            _setupDrop();
            // Re-try after any navigation (the textarea may be re-rendered)
            setTimeout(_setupDrop, 1000);
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
# Enter pour envoyer (Shift+Enter = nouvelle ligne)
# ─────────────────────────────────────────────────────────────────────────────
app.clientside_callback(
    """
    function(dummy) {
        if (window._enterSendBound) return window.dash_clientside.no_update;
        window._enterSendBound = true;
        document.addEventListener('keydown', function(e) {
            if (e.key !== 'Enter' || e.shiftKey) return;
            var container = document.getElementById('chat-text-input');
            if (!container) return;
            var target = e.target;
            var inside = (target === container || container.contains(target));
            if (!inside) return;
            e.preventDefault();
            var btn = document.getElementById('btn-chat-send');
            if (btn && !btn.disabled) btn.click();
        }, true);
        return window.dash_clientside.no_update;
    }
    """,
    Output("enter-bind-store", "data"),
    Input("main-tabs", "active_tab"),
    prevent_initial_call=False,
)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Canvas Actuarial — http://localhost:8050")
    app.run(debug=True, port=8050, host="::", use_reloader=False)
