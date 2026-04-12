"""
catalogue.py
Catalogue dynamique enrichi des tools actuariels.

Lit les TOOL CONTRACT docstrings de chaque tool .py et extrait :
  - CATALOGUE METADATA  : identité, domaine, dépendances
  - INPUTS              : params avec type/values/default/note
  - QUALITY GATES       : conditions bloquantes et non-bloquantes
  - ERROR HANDLING      : messages d'erreur + actions correctives
  - AGENT GUIDANCE      : reasoning_hint pour le LLM

Détection automatique de péremption : si un tool .py est plus récent que
catalogue.yaml, le catalogue est régénéré avant d'être retourné.

Interface publique
------------------
    from catalogue import get_catalogue, regenerate

    # Retourne le catalogue (régénère si périmé)
    cat = get_catalogue()

    # Force la régénération
    regenerate()

Standalone :
    python report_agent/tools/catalogue.py
    python report_agent/tools/catalogue.py --force
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────
_TOOLS_ROOT   = Path(__file__).parent
_PROJECT_ROOT = _TOOLS_ROOT.parent.parent
YAML_PATH     = _TOOLS_ROOT / "catalogue.yaml"

EXCLUDED_FILES = {
    "__init__.py", "TOOL_CONTRACT_TEMPLATE.py", "validate_contracts.py",
    "generate_catalogue.py", "catalogue.py", "_nb_loader.py",
    "tool_registry.py", "example.py",
}

# ── Section boundary regex ────────────────────────────────────────────────────
# Matches "SECTION NAME\n---..." up to the next all-caps section or end
_SECTION_RE = re.compile(
    r"^([A-Z][A-Z\s]+?)\s*\n[-─═]+\s*\n(.*?)(?=\n[A-Z][A-Z\s]+?\s*\n[-─═]+|\Z)",
    re.DOTALL | re.MULTILINE,
)


# ── Section parsers ───────────────────────────────────────────────────────────

def _parse_kv(text: str) -> dict:
    """Parse simple key : value pairs (CATALOGUE METADATA, IDENTITY)."""
    result: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_]+)\s*:\s*(.+)$", line)
        if not m:
            continue
        key = m.group(1).strip()
        val = _coerce(m.group(2).strip())
        result[key] = val
    return result


def _coerce(raw: str) -> Any:
    """Coerce a raw string value to bool, list, or str."""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        inner = inner.strip("[]")
        return [s.strip().strip("'\"") for s in inner.split(",") if s.strip()]
    return raw


def _parse_inputs(text: str) -> dict:
    """
    Parse INPUTS section.
    Format:
        params:
          param_name:
            type    : string
            values  : a | b | c
            default : a
            note    : ...
    """
    params: dict[str, dict] = {}
    current_param: str | None = None
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

        # indent 2 = param name
        if indent == 2 and content.endswith(":"):
            current_param = content[:-1].strip()
            params[current_param] = {}
        # indent 4 = param field
        elif indent == 4 and current_param:
            m = re.match(r"^([a-zA-Z_]+)\s*:\s*(.+)$", content)
            if m:
                field = m.group(1).strip()
                val = m.group(2).strip()
                # Strip trailing note continuation lines
                val = re.sub(r"\s+", " ", val)
                params[current_param][field] = val

    return {"params": params} if params else {}


def _parse_gates(text: str) -> dict:
    """
    Parse QUALITY GATES section.
    Format:
        BLOCKING:
          - condition → action
        NON-BLOCKING:
          - condition → action
    """
    gates: dict[str, list[str]] = {"blocking": [], "non_blocking": []}
    current: str | None = None

    for line in text.splitlines():
        content = line.strip()
        if not content or content.startswith("#"):
            continue
        if content.upper().startswith("BLOCKING:"):
            current = "blocking"
        elif content.upper().startswith("NON-BLOCKING:") or content.upper().startswith("NON_BLOCKING:"):
            current = "non_blocking"
        elif content.startswith("-") and current:
            item = content[1:].strip()
            # Collapse multi-line items (indented continuation)
            if item:
                gates[current].append(item)

    return {"quality_gates": {k: v for k, v in gates.items() if v}}


def _parse_errors(text: str) -> dict:
    """
    Parse ERROR HANDLING section.
    Format:
        error: "message"
          → cause  : ...
          → action : ...
    """
    errors: list[dict] = []
    current: dict | None = None

    for line in text.splitlines():
        content = line.strip()
        if not content or content.startswith("#"):
            continue

        # Match: error: "..." or error: '...'
        m_err = re.match(r'^error\s*:\s*["\']?(.*?)["\']?\s*$', content, re.IGNORECASE)
        if m_err:
            msg = m_err.group(1).strip().strip('"\'')
            if msg:
                current = {"error": msg}
                errors.append(current)
            continue

        if current is not None:
            # Match: → cause : ... or → action : ...
            m_arrow = re.match(r"^[→>]\s*(cause|action)\s*:\s*(.+)$", content)
            if m_arrow:
                current[m_arrow.group(1).strip()] = m_arrow.group(2).strip()

    return {"errors": errors} if errors else {}


def _parse_guidance(text: str) -> dict:
    """Parse AGENT GUIDANCE section — extract reasoning_hint and exemplar_query."""
    result: dict[str, str] = {}

    hint_lines: list[str] = []
    query_lines: list[str] = []
    current: str | None = None

    for line in text.splitlines():
        content = line.strip()
        if re.match(r"^reasoning_hint\s*:\s*>?\s*$", content):
            current = "hint"
            continue
        if re.match(r"^exemplar_query\s*:\s*>?\s*$", content):
            current = "query"
            continue
        if content:
            if current == "hint":
                hint_lines.append(content)
            elif current == "query":
                query_lines.append(content)

    if hint_lines:
        result["reasoning_hint"] = " ".join(hint_lines).strip()
    if query_lines:
        result["exemplar_query"] = " ".join(query_lines).strip()
    return result


def _parse_freetext(text: str) -> str:
    """Return trimmed free-text from a section body (DESCRIPTION, WHEN TO USE, etc.)."""
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    return " ".join(lines).strip()


def _parse_prerequisites(text: str) -> dict:
    """
    Parse PREREQUISITES section.
    Format:
        required_tools:
          - tool_name → provides key
        required_data_store_keys:
          - key_name
        Note: free text note
    """
    result: dict[str, Any] = {}
    current_list: str | None = None

    for line in text.splitlines():
        content = line.strip()
        if not content or content.startswith("#"):
            continue

        if re.match(r"^required_tools\s*:", content):
            current_list = "required_tools"
            result.setdefault("required_tools", [])
        elif re.match(r"^required_data_store_keys\s*:", content):
            current_list = "required_data_store_keys"
            result.setdefault("required_data_store_keys", [])
        elif re.match(r"^[Nn]ote\s*:", content):
            current_list = None
            note = re.sub(r"^[Nn]ote\s*:\s*", "", content).strip()
            if note:
                result["note"] = note
        elif content.startswith("-") and current_list:
            item = content[1:].strip()
            if item:
                result[current_list].append(item)

    return {"prerequisites": result} if result else {}


def _parse_outputs(text: str) -> dict:
    """
    Parse OUTPUTS section.
    Format:
        data_store_keys_written:
          - key : description
        return_payload:
          field : description
    """
    result: dict[str, Any] = {}
    current: str | None = None

    for line in text.splitlines():
        content = line.strip()
        if not content or content.startswith("#"):
            continue

        if re.match(r"^data_store_keys_written\s*:", content):
            current = "data_store"
            result.setdefault("data_store_keys_written", [])
        elif re.match(r"^return_payload\s*:", content):
            current = "return"
            result.setdefault("return_payload", {})
        elif content.startswith("-") and current == "data_store":
            item = content[1:].strip()
            if item:
                result["data_store_keys_written"].append(item)
        elif current == "return" and ":" in content:
            m = re.match(r"^([a-zA-Z_\[\]]+)\s*:\s*(.+)$", content)
            if m:
                result["return_payload"][m.group(1).strip()] = m.group(2).strip()

    return {"outputs": result} if result else {}


# ── Main extractor ────────────────────────────────────────────────────────────

def _extract_contract(content: str) -> dict | None:
    """Extract FULL tool contract (all 11 sections) from a file's module docstring."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    docstring = ast.get_docstring(tree)
    if not docstring:
        return None

    # Find all sections
    sections: dict[str, str] = {}
    for m in _SECTION_RE.finditer(docstring):
        name = m.group(1).strip()
        body = m.group(2)
        sections[name] = body

    if "CATALOGUE METADATA" not in sections:
        return None

    # Base: CATALOGUE METADATA
    tool: dict[str, Any] = _parse_kv(sections.get("CATALOGUE METADATA", ""))

    # Tool name from IDENTITY if not in CATALOGUE METADATA
    if "name" not in tool and "IDENTITY" in sections:
        identity = _parse_kv(sections["IDENTITY"])
        if "name" in identity:
            tool["name"] = identity["name"]

    if not tool.get("name"):
        return None

    # DESCRIPTION
    if "DESCRIPTION" in sections:
        desc = _parse_freetext(sections["DESCRIPTION"])
        if desc:
            tool["description"] = desc

    # WHEN TO USE
    if "WHEN TO USE" in sections:
        val = _parse_freetext(sections["WHEN TO USE"])
        if val:
            tool["when_to_use"] = val

    # WHEN NOT TO USE
    if "WHEN NOT TO USE" in sections:
        val = _parse_freetext(sections["WHEN NOT TO USE"])
        if val:
            tool["when_not_to_use"] = val

    # PREREQUISITES
    if "PREREQUISITES" in sections:
        tool.update(_parse_prerequisites(sections["PREREQUISITES"]))

    # INPUTS
    if "INPUTS" in sections:
        tool.update(_parse_inputs(sections["INPUTS"]))

    # OUTPUTS
    if "OUTPUTS" in sections:
        tool.update(_parse_outputs(sections["OUTPUTS"]))

    # QUALITY GATES
    if "QUALITY GATES" in sections:
        tool.update(_parse_gates(sections["QUALITY GATES"]))

    # ERROR HANDLING
    if "ERROR HANDLING" in sections:
        tool.update(_parse_errors(sections["ERROR HANDLING"]))

    # AGENT GUIDANCE
    if "AGENT GUIDANCE" in sections:
        tool.update(_parse_guidance(sections["AGENT GUIDANCE"]))

    return tool


# ── YAML serializer ───────────────────────────────────────────────────────────

def _yaml_str(v: Any, indent: int = 0) -> str:
    """Minimal YAML serializer — avoids dependency on PyYAML for writing."""
    pad = "  " * indent
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, list):
        if not v:
            return "[]"
        if all(isinstance(i, str) and "\n" not in i for i in v):
            items = ", ".join(f'"{i}"' if any(c in i for c in ": #[]{}") else i for i in v)
            return f"[{items}]"
        lines = [f"{pad}- {_yaml_str(i, 0)}" for i in v]
        return "\n" + "\n".join(lines)
    if isinstance(v, dict):
        if not v:
            return "{}"
        lines = []
        for k, val in v.items():
            serialized = _yaml_str(val, indent + 1)
            if serialized.startswith("\n"):
                lines.append(f"{pad}  {k}:{serialized}")
            else:
                lines.append(f"{pad}  {k}: {serialized}")
        return "\n" + "\n".join(lines)
    if isinstance(v, str):
        if any(c in v for c in (':', '#', '{', '}', '[', ']', '&', '*', '?', '|', '>', '!', "'", '"', '\n')):
            escaped = v.replace('"', '\\"')
            return f'"{escaped}"'
        return v
    return str(v)


def _write_yaml(tools: list[dict], output_path: Path) -> None:
    """Write enriched catalogue to YAML using PyYAML for correctness."""
    try:
        import yaml as _yaml

        # Dependency-aware sort
        def _sort_key(t: dict) -> tuple:
            deps = t.get("depends_on", [])
            n_deps = len(deps) if isinstance(deps, list) else 0
            return (n_deps, t.get("domain", ""), t.get("name", ""))

        FIELD_ORDER = [
            "display_name", "short_description", "description",
            "domain", "capability_group", "depends_on", "required_by", "client_visible",
            "when_to_use", "when_not_to_use",
            "prerequisites", "params", "outputs",
            "quality_gates", "errors",
            "reasoning_hint", "exemplar_query",
        ]

        # Build ordered dict per tool
        catalogue: dict[str, dict] = {}
        for tool in sorted(tools, key=_sort_key):
            name = tool.pop("name", "[unknown]")
            ordered: dict = {}
            for field in FIELD_ORDER:
                if field in tool:
                    ordered[field] = tool[field]
            for k, v in tool.items():
                if k not in FIELD_ORDER:
                    ordered[k] = v
            catalogue[name] = ordered

        header = (
            "# Catalogue des tools disponibles — Agent Actuariat\n"
            "# Generated by catalogue.py — do not edit manually.\n"
            "# Re-generate with: python tools/catalogue.py --force\n\n"
        )
        body = _yaml.dump(
            {"tools": catalogue},
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=120,
        )
        output_path.write_text(header + body, encoding="utf-8")

    except ImportError:
        # Fallback: minimal hand-written YAML (less pretty but functional)
        lines = [
            "# Catalogue des tools disponibles — Agent Actuariat",
            "# Generated by catalogue.py",
            "", "tools:",
        ]
        for tool in sorted(tools, key=lambda t: t.get("name", "")):
            name = tool.get("name", "[unknown]")
            lines.append(f"  - name: {name}")
            for k, v in tool.items():
                if k != "name":
                    lines.append(f"    {k}: {v!r}")
            lines.append("")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Staleness check ───────────────────────────────────────────────────────────

def _is_stale() -> bool:
    """Return True if any tool .py is newer than catalogue.yaml."""
    if not YAML_PATH.exists():
        return True
    yaml_mtime = YAML_PATH.stat().st_mtime
    return any(
        f.stat().st_mtime > yaml_mtime
        for f in _TOOLS_ROOT.rglob("*.py")
        if f.name not in EXCLUDED_FILES and "__pycache__" not in f.parts
    )


# ── Public interface ──────────────────────────────────────────────────────────

def regenerate() -> int:
    """
    Scan all tool .py files, extract contracts, write catalogue.yaml.
    Returns the number of tools found.
    """
    py_files = sorted(
        f for f in _TOOLS_ROOT.rglob("*.py")
        if f.name not in EXCLUDED_FILES and "__pycache__" not in f.parts
    )

    tools: list[dict] = []
    domains: set[str] = set()

    for py_file in py_files:
        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"  [WARN] Cannot read {py_file.name}: {exc}", file=sys.stderr)
            continue

        contract = _extract_contract(content)
        if contract is None:
            rel = py_file.relative_to(_PROJECT_ROOT)
            print(f"  [SKIP] No CATALOGUE METADATA: {rel}", file=sys.stderr)
            continue

        tools.append(contract)
        if "domain" in contract:
            domains.add(str(contract["domain"]))

    _write_yaml(tools, YAML_PATH)
    print(
        f"Catalogue regenerated: {len(tools)} tools across {len(domains)} domain(s) → {YAML_PATH}",
        file=sys.stderr,
    )
    return len(tools)


def get_catalogue() -> dict:
    """
    Return the enriched catalogue as a Python dict.
    Automatically regenerates catalogue.yaml if any tool .py is newer.
    """
    if _is_stale():
        regenerate()

    # Parse YAML — try PyYAML first, fallback to manual parse
    try:
        import yaml as _yaml
        return _yaml.safe_load(YAML_PATH.read_text(encoding="utf-8")) or {}
    except ImportError:
        # Minimal fallback: return raw text wrapped in a dict
        return {"_raw_yaml": YAML_PATH.read_text(encoding="utf-8")}


def get_catalogue_as_yaml() -> str:
    """Return the catalogue as a YAML string (triggers regeneration if stale)."""
    if _is_stale():
        regenerate()
    return YAML_PATH.read_text(encoding="utf-8")


# ── 3-level catalogue ─────────────────────────────────────────────────────────

# Champs retenus par niveau (correspondent aux clés du YAML généré)
_MIDDLE_FIELDS = {"short_description", "when_not_to_use", "prerequisites"}
_FULL_FIELDS   = {
    "short_description", "depends_on", "when_to_use", "when_not_to_use",
    "prerequisites", "params", "outputs", "quality_gates", "reasoning_hint",
}
_LIGHT_FIELDS  = {"short_description", "depends_on", "quality_gates"}


def _filter_tool(tool_data: dict, fields: set) -> dict:
    """Retourne un sous-ensemble des champs d'un tool selon le niveau."""
    result: dict = {}
    for k, v in tool_data.items():
        if k not in fields:
            continue
        # Pour outputs : ne garder que data_store_keys_written
        if k == "outputs" and isinstance(v, dict):
            dsk = v.get("data_store_keys_written")
            if dsk:
                result["outputs_data_store_keys"] = dsk
        # Pour quality_gates : ne garder que blocking
        elif k == "quality_gates" and isinstance(v, dict):
            blocking = v.get("blocking")
            if blocking:
                result["quality_gates_blocking"] = blocking
        else:
            result[k] = v
    return result


def _build_filtered_yaml(fields: set) -> str:
    """Construit un YAML filtré à partir du catalogue complet."""
    import yaml as _yaml
    cat = get_catalogue()
    tools_raw = cat.get("tools", {})
    tools_filtered = {
        name: _filter_tool(data, fields)
        for name, data in tools_raw.items()
    }
    return _yaml.dump(
        {"tools": tools_filtered},
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )


def get_catalogue_middle_yaml() -> str:
    """Catalogue MIDDLE (~1k tokens) : qualification initiale.
    Champs : short_description, when_not_to_use, prerequisites."""
    return _build_filtered_yaml(_MIDDLE_FIELDS)


def get_catalogue_full_yaml() -> str:
    """Catalogue FULL (~6k tokens) : planification complète.
    Champs : short_description, depends_on, when_to_use, when_not_to_use,
    prerequisites, params, outputs_data_store_keys, quality_gates_blocking,
    reasoning_hint."""
    return _build_filtered_yaml(_FULL_FIELDS)


def get_catalogue_light_yaml() -> str:
    """Catalogue LIGHT (~1.5k tokens) : exécution pas à pas.
    Champs : short_description, depends_on, quality_gates_blocking."""
    return _build_filtered_yaml(_LIGHT_FIELDS)


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    force = "--force" in sys.argv
    if force or _is_stale():
        n = regenerate()
        sys.exit(0 if n > 0 else 1)
    else:
        print(f"Catalogue is up to date ({YAML_PATH})", file=sys.stderr)
        sys.exit(0)
