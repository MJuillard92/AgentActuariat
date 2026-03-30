"""
_nb_loader.py
Utilitaire partagé : chargement des modules notebooks/*.py via importlib.

Les notebooks ne sont pas des packages Python (pas de __init__.py),
donc on les charge avec spec_from_file_location.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_NOTEBOOKS_DIR = Path(__file__).parent.parent.parent.parent / "notebooks"


def load_nb(stem: str):
    """
    Charge un module notebook (ex. '02_exposure') depuis notebooks/.

    Résultat mis en cache dans sys.modules pour éviter de recharger.
    """
    if stem in sys.modules:
        return sys.modules[stem]
    path = _NOTEBOOKS_DIR / f"{stem}.py"
    if not path.exists():
        raise ImportError(f"Module notebook introuvable : {path}")
    spec = importlib.util.spec_from_file_location(stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[stem] = mod
    return mod
