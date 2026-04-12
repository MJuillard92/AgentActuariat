"""
TOOL CONTRACT — build_pdf.generate_notebook
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : build_pdf.generate_notebook
domain        : descriptive
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Génère un Jupyter notebook Python reproduisant toute la session d'analyse.
Chaque appel de tool produit une cellule Python exécutable. Le notebook
est autonome et peut être lancé depuis le dossier racine du projet. Permet
au client de reproduire et d'adapter l'analyse indépendamment.

WHEN TO USE
-----------
Proposer systématiquement à la fin d'une analyse complète comme livrable
reproductible. Appeler après que tous les outils d'analyse ont été exécutés
et leurs résultats enregistrés dans _call_log.

WHEN NOT TO USE
---------------
Ne pas appeler si _call_log est vide (aucun appel de tool effectué).
Ne pas appeler en cours d'analyse — attendre la fin.

PREREQUISITES
-------------
required_tools: [any tools called during the session]
required_data_store_keys:
  - _call_log (requis — liste des appels de session)

INPUTS
------
params:
  output_path:
    type    : string
    values  : chemin de fichier .ipynb
    default : /tmp/analyse_actuarielle.ipynb
    note    : L'interface gère le téléchargement. Ne pas exposer au client.
  portfolio_info:
    type    : string
    values  : texte court
    default : ""
    note    : Description courte du portefeuille (ex: "45 231 lignes, 2010-2023").
  csv_filename:
    type    : string
    values  : nom de fichier CSV
    default : portefeuille.csv
    note    : Nom du fichier CSV que le client devra fournir pour relancer le notebook.

OUTPUTS
-------
data_store_keys_written: []
return_payload:
  succes      : bool
  output_path : str
  nb_cellules : int — nombre de cellules générées
  nb_etapes   : int — nombre d'étapes du pipeline reproduites

QUALITY GATES
-------------
BLOCKING:
  - _call_log vide → notebook généré avec seulement les imports (pas d'étapes).
    Informer le client que le notebook ne contient pas d'étapes reproductibles.
NON-BLOCKING: []

ERROR HANDLING
--------------
error: "[exception lors de l'écriture du fichier]"
  → cause  : Erreur système lors de l'écriture du fichier .ipynb.
  → action : Vérifier les droits d'accès au répertoire /tmp/.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Proposer à la fin d'une analyse : "Souhaitez-vous un notebook Python
  reproductible ?" Inclure le nom du fichier CSV dans csv_filename.
  Ne jamais mentionner le chemin output_path dans la réponse au client.
exemplar_query: >
  Comment générer un notebook reproductible à la fin d'une analyse actuarielle ?

CATALOGUE METADATA
------------------
display_name      : Notebook Python reproductible
short_description : Génère un Jupyter notebook reproduisant toute la session d'analyse.
domain            : descriptive
capability_group  : reporting
depends_on        : []
required_by       : []
client_visible    : true
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code_cell(source: str) -> dict:
    return {
        "cell_type":       "code",
        "execution_count": None,
        "metadata":        {},
        "outputs":         [],
        "source":          source,
    }


# Code Python par (tool, function_name)
# Utiliser .replace("{params}", ...) et non .format() car repr() peut contenir {
_TEMPLATES = {
    ("statistical_analysis", "portfolio_summary"): """\
from tools.statistical_analysis.portfolio_summary import run
result = run(df, {params})
for k, v in result.items():
    print(f"  {k}: {v}")""",

    ("statistical_analysis", "age_distribution"): """\
from tools.statistical_analysis.age_distribution import run
result = run(df, {params})
data_store["ages"] = result
print(result)""",

    ("statistical_analysis", "time_series"): """\
from tools.statistical_analysis.time_series import run
result = run(df, {params})
data_store["series"] = result
print(result)""",

    ("statistical_analysis", "segmentation"): """\
from tools.statistical_analysis.segmentation import run
result = run(df, {params})
data_store["segmentation"] = result
print(result)""",

    ("builder", "exposure"): """\
from tools.builder.exposure import run
result = run(df, {params})
data_store["exposure_table"] = result.get("exposure_table", [])
print(f"Exposition: {len(data_store['exposure_table'])} ages")""",

    ("builder", "crude_rates"): """\
from tools.builder.crude_rates import run
result = run(data_store, {params})
data_store["qx_table"] = result.get("qx_table", [])
print(f"Taux bruts: {len(data_store['qx_table'])} ages")""",

    ("builder", "smoothing"): """\
from tools.builder.smoothing import run
result = run(data_store, {params})
data_store["smoothed_table"] = result.get("smoothed_table", [])
print(f"Lissage: {result.get('method')} — {len(data_store['smoothed_table'])} ages")""",

    ("builder", "diagnostics"): """\
from tools.builder.diagnostics import run
result = run(data_store, {params})
data_store["diagnostics"] = result
print(result)""",

    ("builder", "validation"): """\
from tools.builder.validation import run
result = run(data_store, {params})
data_store["validation"] = result
print(result)""",

    ("builder", "benchmarking"): """\
from tools.builder.benchmarking import run
result = run(data_store, {params})
data_store["benchmarking"] = result
print(result)""",

    ("graphs", "analysis_plots"): """\
import base64
from IPython.display import Image, display
from tools.graphs.analysis_plots import run
result = run(data_store, {params})
if "image_b64" in result:
    display(Image(base64.b64decode(result["image_b64"])))""",

    ("graphs", "builder_plots"): """\
import base64
from IPython.display import Image, display
from tools.graphs.builder_plots import run
result = run(data_store, {params})
if "image_b64" in result:
    display(Image(base64.b64decode(result["image_b64"])))""",

    ("build_pdf", "descriptive_report"): """\
from tools.build_pdf.descriptive_report import run
result = run(data_store, {params})
print(f"PDF: {result.get('output_path')}")""",
}


def run(data: dict | None, params: dict | None = None) -> dict:
    data   = data   or {}
    params = params or {}

    output_path    = params.get("output_path", "/tmp/analyse_actuarielle.ipynb")
    portfolio_info = params.get("portfolio_info", "")
    csv_filename   = params.get("csv_filename", "portefeuille.csv")

    call_log      = data.get("_call_log", [])
    reasoning_log = data.get("_reasoning_log", [])

    cells = []

    # ── Titre ─────────────────────────────────────────────────────────────────
    cells.append(_md_cell(
        f"# Analyse Actuarielle — Notebook généré automatiquement\n\n"
        f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
        f"**Données** : `{portfolio_info or csv_filename}`\n\n"
        "Ce notebook reproduit la session Agent Actuariat v2.0.  \n"
        "> **Prérequis** : exécuter depuis le dossier racine `Agent actuariat/`"
    ))

    # ── Imports + data_store ──────────────────────────────────────────────────
    cells.append(_code_cell(
        "import sys, os, pandas as pd\n"
        "sys.path.insert(0, os.path.abspath('.'))\n\n"
        "data_store = {}  # accumulateur inter-étapes"
    ))

    # ── Chargement CSV ────────────────────────────────────────────────────────
    cells.append(_md_cell("## Chargement des données"))
    cells.append(_code_cell(
        f'CSV_FILE = "{csv_filename}"\n\n'
        "for sep in (\";\", \",\", \"\\t\", \"|\"):\n"
        "    try:\n"
        "        df = pd.read_csv(CSV_FILE, sep=sep, encoding=\"utf-8\", engine=\"python\")\n"
        "        if len(df.columns) > 1:\n"
        "            break\n"
        "    except Exception:\n"
        "        pass\n\n"
        "print(f\"Chargé: {len(df):,} lignes, {len(df.columns)} colonnes\")\n"
        "df.head()"
    ))

    # ── Raisonnement ──────────────────────────────────────────────────────────
    if reasoning_log:
        text = "\n\n---\n\n".join(reasoning_log)
        cells.append(_md_cell(f"## Raisonnement de l'agent\n\n{text}"))

    # ── Pipeline ──────────────────────────────────────────────────────────────
    valid_steps = [e for e in call_log if not e.get("has_error")]
    if valid_steps:
        cells.append(_md_cell("## Pipeline d'analyse"))

    for entry in valid_steps:
        tool        = entry.get("tool", "")
        fn          = entry.get("function_name", "")
        call_params = entry.get("params", {})
        step        = entry.get("step", "?")
        summary     = entry.get("result_summary", {})

        # Header avec résumé
        header = [f"### Étape {step} — `{tool}.{fn}`"]
        for k, v in summary.items():
            if k not in ("erreur", "traceback", "image_b64"):
                header.append(f"- **{k}** : {v}")
        cells.append(_md_cell("\n".join(header)))

        # Code cell
        template = _TEMPLATES.get((tool, fn))
        params_repr = repr(call_params) if call_params else "{}"
        if template:
            code = template.replace("{params}", params_repr)
        else:
            code = (
                f"# {tool}.{fn}\n"
                f"from tools.{tool}.{fn} import run\n"
                f"result = run(data_store, {params_repr})\n"
                "print(result)"
            )
        cells.append(_code_cell(code))

    cells.append(_md_cell("---\n\n*Notebook généré par Agent Actuariat v2.0*"))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.8.0"},
        },
        "cells": cells,
    }

    try:
        Path(output_path).write_text(
            json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return {
            "succes":      True,
            "output_path": output_path,
            "nb_cellules": len(cells),
            "nb_etapes":   len(valid_steps),
        }
    except Exception as exc:
        return {"erreur": str(exc), "succes": False}
