"""
tool_registry.py — projection minimale des TOOL CONTRACTs pour le validator.

Adapter mince au-dessus de `tools.catalogue.scan_contracts` (source unique de
parsing des docstrings). Projette chaque contrat en :

    {qualified_name: {inputs: {name: type}, outputs: {name: type}, path: str}}

où `qualified_name` vient de `IDENTITY.name` (ex. "builder.exposure").

Ce format est celui consommé par le validator (Phase 0 preflight) pour vérifier
les `produced_by` du YAML contre les signatures réelles des tools.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "tools"
if str(_TOOLS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR.parent))

from tools.catalogue import scan_contracts  # noqa: E402


class ToolSpec(TypedDict):
    inputs: dict[str, str]
    outputs: dict[str, str]
    path: str


def _project(contract: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Extrait inputs (params.type) et outputs (return_payload) du contrat parsé."""
    inputs: dict[str, str] = {}
    for param_name, info in (contract.get("params") or {}).items():
        if isinstance(info, dict):
            inputs[param_name] = str(info.get("type", "unknown"))
        else:
            inputs[param_name] = "unknown"

    outputs: dict[str, str] = {}
    outputs_raw = contract.get("outputs") or {}
    return_payload = outputs_raw.get("return_payload") or {}
    for field, type_str in return_payload.items():
        outputs[field] = str(type_str)
    return inputs, outputs


def build_registry(tools_root: Path) -> dict[str, ToolSpec]:
    """Construit le registry à partir d'un répertoire de tools.

    Lève ValueError si deux fichiers déclarent le même `name`.
    """
    registry: dict[str, ToolSpec] = {}
    for py_path, contract in scan_contracts(Path(tools_root)):
        name = contract.get("name")
        if not name:
            continue
        if name in registry:
            raise ValueError(
                f"collision de nom dans le registry : '{name}' déclaré par "
                f"{registry[name]['path']} et {py_path}"
            )
        inputs, outputs = _project(contract)
        registry[name] = {"inputs": inputs, "outputs": outputs, "path": str(py_path)}
    return registry
