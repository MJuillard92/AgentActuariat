#!/usr/bin/env python3
"""
check_template.py — gardien pré-commit / CI.

Usage :
    python scripts/check_template.py
    python scripts/check_template.py --yaml path/to/template.yaml --tools-root tools/
    python scripts/check_template.py --no-color

Exit code :
    0 : YAML valide contre le registry des tools, aucune erreur bloquante.
    1 : au moins une erreur de validation.

Les warnings (clés déclarées non consommées, etc.) sont affichés mais
ne font pas échouer le script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from knowledge_base.report_template.tool_registry import build_registry  # noqa: E402
from knowledge_base.report_template.validator import validate_template    # noqa: E402


DEFAULT_YAML = _PROJECT_ROOT / "knowledge_base" / "report_template" / "mortality_template.yaml"
DEFAULT_TOOLS = _PROJECT_ROOT / "tools"


# ───────────────── Coloration ─────────────────

_ANSI = {
    "red":    "\033[31m",
    "yellow": "\033[33m",
    "green":  "\033[32m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}


def _colorize(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{_ANSI[color]}{text}{_ANSI['reset']}"


# ───────────────── Rendu rapport ─────────────────

def _render_report(report, color: bool) -> str:
    lines: list[str] = []
    if report.errors:
        lines.append(_colorize(f"✗ {len(report.errors)} erreur(s)", "red", color))
        for issue in report.errors:
            lines.append(f"  [{issue.location}] {issue.message}")
    if report.warnings:
        lines.append(_colorize(f"⚠ {len(report.warnings)} warning(s)", "yellow", color))
        for issue in report.warnings:
            lines.append(f"  [{issue.location}] {issue.message}")
    if report.ok and not report.warnings:
        lines.append(_colorize("✓ template valide", "green", color))
    elif report.ok:
        lines.append(_colorize("✓ template valide (warnings non bloquants)", "green", color))
    return "\n".join(lines)


# ───────────────── Entry point ─────────────────

def main(
    yaml_path: Path = DEFAULT_YAML,
    tools_root: Path = DEFAULT_TOOLS,
    color: bool = True,
    check_columns: bool = False,
) -> int:
    """Point d'entrée testable. Retourne l'exit code (0 ou 1).

    Args:
        check_columns : si True, lance en plus `check_table_columns` qui
            vérifie que chaque colonne d'un visual_spec table existe dans
            le schéma déclaré par la docstring OUTPUTS du tool source.
    """
    try:
        registry = build_registry(Path(tools_root))
    except Exception as exc:
        print(_colorize(f"✗ erreur de construction du registry : {exc}", "red", color))
        return 1

    report = validate_template(Path(yaml_path), registry)
    print(_render_report(report, color))
    rc = 0 if report.ok else 1

    if check_columns:
        from knowledge_base.report_template.validator import check_table_columns
        col_report = check_table_columns(Path(yaml_path), registry)
        print(_colorize("\n── Vérification colonnes table ↔ tool OUTPUTS ──", "bold", color))
        print(_render_report(col_report, color))
        if not col_report.ok:
            rc = 1

    return rc


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Valide un template YAML contre le registry des tools.")
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML, help="Chemin du template YAML.")
    parser.add_argument("--tools-root", type=Path, default=DEFAULT_TOOLS, help="Racine des tools à scanner.")
    parser.add_argument("--no-color", action="store_true", help="Désactive la coloration ANSI.")
    parser.add_argument(
        "--check-columns", action="store_true",
        help="Vérifie que les columns[].key des visual_specs table existent "
             "bien dans le schéma déclaré par la docstring OUTPUTS du tool.",
    )
    args = parser.parse_args()
    return main(
        yaml_path=args.yaml,
        tools_root=args.tools_root,
        color=not args.no_color,
        check_columns=args.check_columns,
    )


if __name__ == "__main__":
    sys.exit(_cli())
