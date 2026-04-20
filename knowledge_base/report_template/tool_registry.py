"""
tool_registry.py — scanne un répertoire de tools et construit un registry
{qualified_name: ToolSpec} où ToolSpec = {inputs, outputs, path}.

Parse les docstrings TOOL CONTRACT de chaque module .py sous `tools_root`.
Source de vérité pour le validator (Phase 0 preflight) et les checks de contrat.

Seul point d'entrée public : build_registry(tools_root).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TypedDict

_EXCLUDED = {
    "__init__.py",
    "TOOL_CONTRACT_TEMPLATE.py",
    "validate_contracts.py",
    "generate_catalogue.py",
    "catalogue.py",
    "tool_registry.py",
    "_nb_loader.py",
    "example.py",
}


class ToolSpec(TypedDict):
    inputs: dict[str, str]
    outputs: dict[str, str]
    path: str


# ── Section regex (TOOL CONTRACT docstrings, cf. tools/catalogue.py) ─────────
_SECTION_RE = re.compile(
    r"^([A-Z][A-Z\s]+?)\s*\n[-─═]+\s*\n(.*?)(?=\n[A-Z][A-Z\s]+?\s*\n[-─═]+|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _extract_sections(docstring: str) -> dict[str, str]:
    return {m.group(1).strip(): m.group(2) for m in _SECTION_RE.finditer(docstring)}


def _parse_name(identity_text: str) -> str | None:
    for line in identity_text.splitlines():
        m = re.match(r"^\s*name\s*:\s*(.+)$", line)
        if m:
            return m.group(1).strip()
    return None


def _parse_inputs(text: str) -> dict[str, str]:
    """Parse INPUTS.params block → {param_name: type_str}."""
    params: dict[str, str] = {}
    current: str | None = None
    in_params = False
    for line in text.splitlines():
        stripped = line.rstrip()
        indent = len(stripped) - len(stripped.lstrip())
        content = stripped.strip()
        if not content or content.startswith("#"):
            continue
        if content == "params:":
            in_params = True
            continue
        if not in_params:
            continue
        if indent == 2 and content.endswith(":"):
            current = content[:-1].strip()
            params[current] = "unknown"
        elif indent == 4 and current:
            m = re.match(r"^type\s*:\s*(.+)$", content)
            if m:
                params[current] = m.group(1).strip()
    return params


def _parse_outputs(text: str) -> dict[str, str]:
    """Parse OUTPUTS.return_payload → {field: type_str}. Types souvent implicites."""
    outputs: dict[str, str] = {}
    in_return = False
    for line in text.splitlines():
        content = line.strip()
        if not content or content.startswith("#"):
            continue
        if re.match(r"^return_payload\s*:", content):
            in_return = True
            continue
        if re.match(r"^data_store_keys_written\s*:", content):
            in_return = False
            continue
        if not in_return:
            continue
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_\[\]]*)\s*:\s*(.+)$", content)
        if m:
            field = m.group(1).strip()
            type_str = m.group(2).strip()
            outputs[field] = type_str
    return outputs


def _extract_spec(py_path: Path) -> tuple[str, ToolSpec] | None:
    try:
        source = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return None
    doc = ast.get_docstring(tree)
    if not doc:
        return None
    sections = _extract_sections(doc)
    if "IDENTITY" not in sections:
        return None
    name = _parse_name(sections["IDENTITY"])
    if not name:
        return None
    inputs = _parse_inputs(sections.get("INPUTS", ""))
    outputs = _parse_outputs(sections.get("OUTPUTS", ""))
    return name, {"inputs": inputs, "outputs": outputs, "path": str(py_path)}


def build_registry(tools_root: Path) -> dict[str, ToolSpec]:
    """Scanne `tools_root` récursivement, parse les TOOL CONTRACT, retourne le registry.

    Lève ValueError si deux fichiers déclarent le même `name`.
    """
    tools_root = Path(tools_root)
    registry: dict[str, ToolSpec] = {}
    for py_file in sorted(tools_root.rglob("*.py")):
        if py_file.name in _EXCLUDED:
            continue
        result = _extract_spec(py_file)
        if not result:
            continue
        name, spec = result
        if name in registry:
            raise ValueError(
                f"collision de nom dans le registry : '{name}' déclaré par "
                f"{registry[name]['path']} et {spec['path']}"
            )
        registry[name] = spec
    return registry
