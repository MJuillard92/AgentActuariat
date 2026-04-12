"""
validate_contracts.py
Scans all tool .py files and verifies they contain all required TOOL CONTRACT sections.
Exit code 1 if any file fails (usable as pre-commit hook).
Usage: python report_agent/tools/validate_contracts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── Required sections ─────────────────────────────────────────────────────────
REQUIRED_SECTIONS = [
    "IDENTITY",
    "DESCRIPTION",
    "WHEN TO USE",
    "WHEN NOT TO USE",
    "PREREQUISITES",
    "INPUTS",
    "OUTPUTS",
    "QUALITY GATES",
    "ERROR HANDLING",
    "AGENT GUIDANCE",
    "CATALOGUE METADATA",
]

# ── Files to exclude ──────────────────────────────────────────────────────────
EXCLUDED_FILES = {
    "__init__.py",
    "TOOL_CONTRACT_TEMPLATE.py",
    "validate_contracts.py",
    "generate_catalogue.py",
    "_nb_loader.py",
    "tool_registry.py",
    "example.py",
}


def find_tool_files(tools_root: Path) -> list[Path]:
    """Collect all .py tool files, excluding the listed exclusions."""
    files = []
    for py_file in sorted(tools_root.rglob("*.py")):
        if py_file.name in EXCLUDED_FILES:
            continue
        if "__pycache__" in py_file.parts:
            continue
        files.append(py_file)
    return files


def check_file(py_file: Path) -> list[str]:
    """Return list of missing sections for a given file."""
    content = py_file.read_text(encoding="utf-8")
    missing = []
    for section in REQUIRED_SECTIONS:
        # Accept either "SECTION\n---" or "SECTION\n════" style headers
        if section not in content:
            missing.append(section)
    return missing


def main() -> int:
    tools_root = Path(__file__).parent
    files = find_tool_files(tools_root)

    if not files:
        print("No tool files found.")
        return 0

    failures: dict[Path, list[str]] = {}
    for f in files:
        missing = check_file(f)
        if missing:
            failures[f] = missing

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"Scanned {len(files)} tool file(s).")
    if not failures:
        print(f"All {len(files)} files pass contract validation.")
        return 0

    print(f"\nFAILED: {len(failures)} file(s) missing required TOOL CONTRACT sections:\n")
    for path, missing_sections in sorted(failures.items()):
        rel = path.relative_to(tools_root.parent.parent)
        print(f"  {rel}")
        for sec in missing_sections:
            print(f"    - Missing section: {sec}")
    print(
        f"\nFix: add a complete TOOL CONTRACT docstring at the top of each failed file.\n"
        f"See TOOL_CONTRACT_TEMPLATE.py for the required format."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
