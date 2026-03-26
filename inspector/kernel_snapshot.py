"""
kernel_snapshot.py
==================
Sérialise l'état du kernel Python de l'agent après chaque étape.

Chaque snapshot est un répertoire horodaté dans inspector/snapshots/ :
  session_YYYYMMDD_HHMMSS/
    _manifest.json     ← liste des variables avec métadonnées
    _steps.jsonl       ← log des étapes (code + output + description)
    df_exposure.pkl    ← DataFrames sérialisés (pickle)
    figure_1.png       ← figures matplotlib
    SMR.json           ← scalaires et dicts

Usage :
    from inspector.kernel_snapshot import save_snapshot, list_sessions, load_variable
    session_id = save_snapshot(kernel, step_info)
"""

from __future__ import annotations

import io
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SNAPSHOT_DIR = Path(__file__).parent / "snapshots"

# Noms à ignorer : modules, bibliothèques, fonctions internes
_SKIP_NAMES = frozenset({
    "pd", "np", "plt", "matplotlib", "sns", "scipy", "io",
    "warnings", "json", "pathlib", "datetime", "sys", "os",
    "data_prep", "exposure", "crude_rates", "smoothing",
    "diagnostics", "validation", "benchmarking", "visualization",
    "smoothing_selector", "solve_banded", "redirect_stdout",
})

# Icônes affichées dans le navigateur de variables
TYPE_ICONS = {
    "DataFrame": "📊",
    "ndarray":   "🔢",
    "Figure":    "📈",
    "dict":      "📋",
    "scalar":    "🔢",
    "str":       "📝",
    "list":      "📌",
}


def _classify(val: Any):
    """Retourne la catégorie de la variable, ou None si à ignorer."""
    if isinstance(val, pd.DataFrame):
        return "DataFrame"
    if isinstance(val, np.ndarray):
        return "ndarray"
    if isinstance(val, dict):
        return "dict"
    if isinstance(val, bool):
        return "scalar"
    if isinstance(val, (int, float)):
        return "scalar"
    if isinstance(val, str):
        return "str"
    if isinstance(val, (list, tuple)):
        return "list"
    try:
        import matplotlib.figure
        if isinstance(val, matplotlib.figure.Figure):
            return "Figure"
    except ImportError:
        pass
    return None


def save_snapshot(
    kernel: dict,
    step_info: dict | None = None,
    session_id: str | None = None,
) -> str:
    """
    Sérialise les variables intéressantes du kernel.

    Args:
        kernel:     Namespace Python de l'agent (dict).
        step_info:  {"description": str, "code": str, "output": str, "success": bool}
        session_id: Identifiant de session (créé automatiquement si None).

    Returns:
        session_id utilisé (str, format YYYYMMDD_HHMMSS).
    """
    if session_id is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    snap_dir = _SNAPSHOT_DIR / session_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    # ── Sérialisation des variables ──────────────────────────────────────────
    manifest = []
    for name, val in kernel.items():
        if name.startswith("_") or name in _SKIP_NAMES:
            continue
        kind = _classify(val)
        if kind is None:
            continue

        entry: dict = {"name": name, "type": kind, "icon": TYPE_ICONS.get(kind, "📦")}

        try:
            if kind == "DataFrame":
                fpath = snap_dir / f"{name}.pkl"
                with fpath.open("wb") as fh:
                    pickle.dump(val, fh, protocol=4)
                entry["shape"] = list(val.shape)
                entry["columns"] = list(val.columns)
                entry["dtypes"] = {c: str(t) for c, t in val.dtypes.items()}
                entry["file"] = str(fpath)

            elif kind == "ndarray":
                fpath = snap_dir / f"{name}.npy"
                np.save(str(fpath), val)
                entry["shape"] = list(val.shape)
                entry["dtype"] = str(val.dtype)
                entry["file"] = str(fpath)

            elif kind == "Figure":
                fpath = snap_dir / f"{name}.png"
                val.savefig(str(fpath), dpi=100, bbox_inches="tight")
                entry["file"] = str(fpath)

            elif kind in ("dict", "scalar", "str", "list"):
                fpath = snap_dir / f"{name}.json"
                fpath.write_text(
                    json.dumps(val, ensure_ascii=False, default=str, indent=2),
                    encoding="utf-8",
                )
                entry["file"] = str(fpath)
                if kind == "scalar":
                    entry["value"] = round(val, 6) if isinstance(val, float) else val

            manifest.append(entry)

        except Exception:
            continue  # best-effort : une variable non sérialisable ne bloque pas

    (snap_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Ajout de l'étape au log des steps ───────────────────────────────────
    if step_info:
        steps_path = snap_dir / "_steps.jsonl"
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "description": step_info.get("description", ""),
            "code": step_info.get("code", ""),
            "output": step_info.get("output", ""),
            "success": step_info.get("success", True),
        }
        with steps_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return session_id


# ─────────────────────────────────────────────────────────────────────────────
# Lecture
# ─────────────────────────────────────────────────────────────────────────────

def list_sessions() -> list:
    """Liste les sessions disponibles, triées de la plus récente à la plus ancienne."""
    if not _SNAPSHOT_DIR.exists():
        return []
    sessions = [
        d.name for d in _SNAPSHOT_DIR.iterdir()
        if d.is_dir() and (d / "_manifest.json").exists()
    ]
    return sorted(sessions, reverse=True)


def load_manifest(session_id: str) -> list:
    """Charge le manifest d'une session."""
    path = _SNAPSHOT_DIR / session_id / "_manifest.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_steps(session_id: str) -> list:
    """Charge les étapes enregistrées pour une session."""
    path = _SNAPSHOT_DIR / session_id / "_steps.jsonl"
    if not path.exists():
        return []
    steps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                steps.append(json.loads(line))
            except Exception:
                pass
    return steps


def load_variable(session_id: str, var_name: str) -> Any:
    """Charge une variable depuis un snapshot."""
    manifest = load_manifest(session_id)
    entry = next((e for e in manifest if e["name"] == var_name), None)
    if entry is None:
        return None

    fpath = Path(entry["file"])
    if not fpath.exists():
        return None

    kind = entry["type"]
    if kind == "DataFrame":
        with fpath.open("rb") as fh:
            return pickle.load(fh)
    elif kind == "ndarray":
        return np.load(str(fpath))
    elif kind in ("dict", "scalar", "str", "list"):
        return json.loads(fpath.read_text(encoding="utf-8"))
    elif kind == "Figure":
        return fpath  # chemin vers le PNG
    return None


def build_exec_namespace(session_id: str) -> dict:
    """
    Construit un namespace Python avec toutes les variables du snapshot.
    Utilisé pour exécuter du code ad-hoc dans le code cell de l'inspecteur.
    """
    ns = {}
    exec("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt", ns)
    for entry in load_manifest(session_id):
        name = entry["name"]
        if entry["type"] in ("DataFrame", "ndarray", "dict", "scalar", "str", "list"):
            val = load_variable(session_id, name)
            if val is not None:
                ns[name] = val
    return ns
