"""
workflow_executor.py
Exécute un Workflow en évaluant les conditions métier sur les arêtes.
Réutilise execute_cell / notebook_runner sans modification.

Un workflow est un graphe orienté acyclique (DAG) dont :
  - les nœuds (WorkflowNode) correspondent à des notebooks actuariels à exécuter,
  - les arêtes (WorkflowEdge) portent des conditions Python évaluées sur le kernel.

Le kernel (dict Python) est l'espace de noms partagé entre tous les nœuds :
  - il est initialisé une seule fois par make_kernel(),
  - chaque nœud peut lire ET modifier les variables des nœuds précédents,
  - les conditions des arêtes sont évaluées sur ce même namespace après chaque nœud.

Exemple de flux conditionnel :
  nœud A → (condition: "SMR > 1.2") → nœud B (alerte surmortalité)
          → (condition: "SMR <= 1.2") → nœud C (flux normal)
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
from pathlib import Path
from typing import Callable, Generator

from notebook_runner import load_notebook, execute_cell
from workflow import Workflow, WorkflowNode


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation des conditions
# ─────────────────────────────────────────────────────────────────────────────

_SAFE_BUILTINS = {"abs", "round", "min", "max", "len", "int", "float", "bool", "str"}
# On restreint les builtins disponibles dans eval() pour éviter qu'une condition
# malformée puisse exécuter du code arbitraire (import, open, exec, etc.).

def evaluate_condition(condition: str, kernel_state: dict) -> bool:
    """Évalue une condition Python simple dans le contexte du kernel.

    Exemples de conditions valides :
        "SMR > 1.2"
        "SMR < 0.8"
        "n_vides > 5"
        "non_mono > 0"
        "True"   (toujours exécuter)

    Retourne True si la condition est satisfaite (branche à prendre).
    Retourne True aussi si la condition est vide/None (arête inconditionnelle).
    """
    if not condition or not condition.strip():
        return True
    try:
        # Namespace limité : variables du kernel + builtins sûrs.
        # Les callables (fonctions, modules) sont exclus car une condition
        # ne doit comparer que des scalaires (SMR, n_violations, etc.).
        safe_ns = {k: v for k, v in kernel_state.items()
                   if not k.startswith("_") and not callable(v)}
        result = eval(condition.strip(), {"__builtins__": {}}, safe_ns)  # noqa: S307
        return bool(result)
    except Exception as exc:
        # En cas d'erreur d'évaluation, on prend la branche par défaut (fail-open)
        # pour éviter de bloquer un workflow sur une variable non encore calculée.
        # L'utilisateur verra l'avertissement dans les logs.
        print(f"[workflow] Condition '{condition}' non évaluable ({exc}) → branche prise par défaut")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Capture des figures matplotlib
# ─────────────────────────────────────────────────────────────────────────────

def _capture_figures(kernel: dict) -> list[bytes]:
    plt = kernel.get("plt")
    if plt is None:
        return []
    figs = []
    for fn in plt.get_fignums():
        fig = plt.figure(fn)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        figs.append(buf.read())
    plt.close("all")
    return figs


def capture_figures(kernel: dict) -> list[bytes]:
    """Public alias for _capture_figures — use this in external modules."""
    return _capture_figures(kernel)


# ─────────────────────────────────────────────────────────────────────────────
# Exécuteur principal
# ─────────────────────────────────────────────────────────────────────────────

def execute_workflow(
    workflow: Workflow,
    kernel_state: dict,
    on_step: Callable[[str, str, str, list[bytes]], None] = None,
) -> Generator[dict, None, None]:
    """Exécute le workflow nœud par nœud selon l'ordre topologique et les conditions.

    Args:
        workflow: Workflow à exécuter.
        kernel_state: Namespace Python partagé (modifié en place).
        on_step: Callback optionnel(node_id, label, output, figures).

    Yields:
        dict avec la clé ``type`` valant l'une des valeurs suivantes :

        - ``"step_start"`` : nœud en cours de démarrage (node_id, label)
        - ``"step_done"``  : nœud terminé (node_id, label, output, figures, skipped)
        - ``"error"``      : erreur bloquante (node_id, message)
        - ``"done"``       : workflow terminé normalement
    """
    order = workflow.execution_order()
    executed: set[str] = set()
    skipped: set[str] = set()

    for node_id in order:
        node = workflow.get_node(node_id)
        if node is None:
            continue

        # Vérifier si toutes les arêtes entrantes dont la source est exécutée
        # ont leur condition satisfaite (logique OR : si au moins une arête
        # entrante exécutée mène ici, on continue)
        incoming = [e for e in workflow.edges if e.target == node_id]
        if incoming:
            # Sources déjà exécutées qui pointent vers ce nœud
            active_sources = [e for e in incoming if e.source in executed]
            if not active_sources:
                # Toutes les sources ont été sautées → sauter ce nœud aussi
                skipped.add(node_id)
                yield {"type": "step_done", "node_id": node_id, "label": node.label,
                       "output": "", "figures": [], "skipped": True}
                continue

            # Évaluer les conditions des arêtes actives
            any_condition_met = any(
                evaluate_condition(e.condition, kernel_state)
                for e in active_sources
            )
            if not any_condition_met:
                skipped.add(node_id)
                yield {"type": "step_done", "node_id": node_id, "label": node.label,
                       "output": f"Condition non satisfaite — nœud ignoré.",
                       "figures": [], "skipped": True}
                continue

        yield {"type": "step_start", "node_id": node_id, "label": node.label}

        # Charger et exécuter toutes les cellules code du notebook
        try:
            cells = load_notebook(node.notebook_path)
        except FileNotFoundError:
            yield {"type": "error", "node_id": node_id,
                   "message": f"Notebook introuvable : {node.notebook_path}"}
            continue

        combined_output = []
        for cell in cells:
            if cell["type"] != "code" or not cell["source"].strip():
                continue
            output = execute_cell(cell["source"], kernel_state)
            combined_output.append(output)

        full_output = "\n".join(combined_output)
        figures = _capture_figures(kernel_state)
        executed.add(node_id)

        if on_step:
            on_step(node_id, node.label, full_output, figures)

        yield {
            "type": "step_done",
            "node_id": node_id,
            "label": node.label,
            "output": full_output,
            "figures": figures,
            "skipped": False,
        }

        if "❌ Erreur" in full_output:
            yield {"type": "error", "node_id": node_id,
                   "message": f"Erreur dans {node.label} — exécution arrêtée."}
            return

    yield {"type": "done"}


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation du kernel
# ─────────────────────────────────────────────────────────────────────────────

def make_kernel() -> dict:
    """Crée un kernel Python frais avec les imports standards et la bibliothèque actuarielle.

    Le kernel est un dictionnaire Python ordinaire utilisé comme espace de noms
    global pour exec(). Tous les modules actuariels y sont pré-chargés pour que
    l'agent puisse appeler data_prep.load_data(), smoothing.smooth_whittaker(),
    etc. directement, sans import.

    Les modules sont aussi enregistrés dans sys.modules (via leur alias court)
    pour que les instructions "import data_prep" dans execute_python() fonctionnent
    même si l'utilisateur les écrit dans ses cellules de code.

    Les paramètres PARAMS et les variables de configuration dérivées (DATE_FIN,
    LAMBDA_WH) sont injectés pour que le code de l'agent n'ait jamais à hardcoder
    de valeurs numériques — toute modification passe par actuarial_params.py.
    """
    # Répertoire racine du projet (contient canvas_app.py, workflow_executor.py, notebooks/)
    _project_dir = Path(__file__).parent.resolve()
    os.chdir(str(_project_dir))

    ns: dict = {}
    exec(  # noqa: S102
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "try:\n    import seaborn as sns\n    try:\n        sns.set_theme(style='whitegrid')\n    except AttributeError:\n        sns.set(style='whitegrid')\nexcept ImportError:\n    pass\n"
        "import matplotlib.pyplot as plt\n"
        "from scipy.linalg import solve_banded\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n",
        ns,
    )

    # ── Bibliothèque actuarielle — modules appelables par l'agent ────────────
    _notebooks_dir = Path(__file__).parent / "notebooks"
    _actuarial_modules = {
        "01_data_preparation": "data_prep",
        "02_exposure":         "exposure",
        "03_crude_rates":      "crude_rates",
        "04_smoothing":        "smoothing",
        "05_diagnostics":      "diagnostics",
        "06_validation":       "validation",
        "07_benchmarking":     "benchmarking",
        "08_visualization":    "visualization",
    }
    for mod_file, alias in _actuarial_modules.items():
        mod_path = _notebooks_dir / f"{mod_file}.py"
        if not mod_path.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location(mod_file, str(mod_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            ns[alias] = mod
            sys.modules[alias] = mod   # rend "import data_prep" utilisable dans exec()
        except Exception as exc:  # noqa: BLE001
            print(f"[make_kernel] Impossible de charger {mod_file}: {exc}")

    # ── Sélecteur de modèle de lissage ───────────────────────────────────────
    _selector_path = Path(__file__).parent / "smoothing_selector.py"
    if _selector_path.exists():
        try:
            _spec = importlib.util.spec_from_file_location("smoothing_selector", str(_selector_path))
            _sel_mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_sel_mod)
            ns["smoothing_selector"] = _sel_mod
            sys.modules["smoothing_selector"] = _sel_mod
        except Exception as exc:  # noqa: BLE001
            print(f"[make_kernel] Impossible de charger smoothing_selector: {exc}")

    # ── Paramètres métier (accessibles via PARAMS dans execute_python) ────────
    try:
        from actuarial_params import PARAMS as _AP
        import pandas as _pd_tmp
        ns["PARAMS"] = _AP
        ns["DATE_FIN_OBSERVATION"] = _pd_tmp.Timestamp(_AP["observation"]["date_fin"])
        ns["LAMBDA_WH"] = _AP["smoothing"]["lambda_wh"]
    except Exception as exc:  # noqa: BLE001
        print(f"[make_kernel] Impossible de charger actuarial_params: {exc}")

    return ns
