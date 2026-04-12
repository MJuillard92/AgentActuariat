"""
loader.py
Assembles the agent system prompt from system_prompt files + agent_instructions/*.md + catalogue.yaml.

Usage:
    from loader import get_system_prompt
    prompt = get_system_prompt()                         # mortality agent, full catalogue
    prompt = get_system_prompt(agent_name="report")      # report agent
    prompt = get_system_prompt(level="middle")           # mortality agent, middle catalogue

    python loader.py  # prints assembled prompt to stdout
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Project root is the directory containing this file
_PROJECT_ROOT = Path(__file__).parent

# Agent system prompt paths
_AGENT_SYSTEM_PROMPTS: dict[str, Path] = {
    "mortality": _PROJECT_ROOT / "agents" / "mortality" / "system_prompt_level1.md",
    "report":    _PROJECT_ROOT / "agents" / "report" / "system_prompt.md",
    "master":    _PROJECT_ROOT / "agents" / "master" / "system_prompt.md",
}

# Pattern for inject directives: ## [INJECT] Title\nsource: path\n[format: format]
_INJECT_RE = re.compile(
    r"^## \[INJECT\]\s+(.+?)\nsource:\s*(\S+)(?:\nformat:\s*(\S+))?",
    re.MULTILINE,
)


def _read_source(source_path: Path, fmt: str | None = None, catalogue_level: str = "full") -> str:
    """Read a source file and optionally format it."""
    if fmt == "yaml_block":
        # Use catalogue.py to get catalogue at the requested level
        try:
            import importlib.util as _ilu
            _cat_path = _PROJECT_ROOT / "tools" / "catalogue.py"
            spec = _ilu.spec_from_file_location("catalogue", _cat_path)
            _cat_mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(_cat_mod)
            _level_fn = {
                "middle": _cat_mod.get_catalogue_middle_yaml,
                "light":  _cat_mod.get_catalogue_light_yaml,
                "full":   _cat_mod.get_catalogue_full_yaml,
            }.get(catalogue_level, _cat_mod.get_catalogue_full_yaml)
            yaml_text = _level_fn()
        except Exception:
            # Fallback: read catalogue.yaml directly
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Source file not found: {source_path}\n"
                    f"  (referenced in system prompt)"
                )
            yaml_text = source_path.read_text(encoding="utf-8").strip()
        return f"# Catalogue des tools disponibles\n\n```yaml\n{yaml_text}\n```"

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source file not found: {source_path}\n"
            f"  (referenced in system prompt)"
        )

    content = source_path.read_text(encoding="utf-8").strip()
    return content


def get_system_prompt(level: str = "full", agent_name: str = "mortality") -> str:
    """
    Assembles and returns the agent system prompt.

    Args:
        level: catalogue level — "middle" | "full" | "light"
            - "middle" (~2.5k tokens) : qualification initiale
            - "full"   (~8k tokens)   : planification complète
            - "light"  (~1.3k tokens) : exécution pas à pas
        agent_name: which agent's prompt to load — "mortality" | "report" | "master"

    Returns:
        str — the fully assembled system prompt
    """
    level1_path = _AGENT_SYSTEM_PROMPTS.get(agent_name)
    if level1_path is None:
        raise ValueError(f"Unknown agent_name: {agent_name!r}. Valid: {list(_AGENT_SYSTEM_PROMPTS)}")

    if not level1_path.exists():
        raise FileNotFoundError(
            f"System prompt not found: {level1_path}"
        )

    agent_dir = level1_path.parent
    level1_content = level1_path.read_text(encoding="utf-8")

    sections: list[str] = []
    last_end = 0
    n_sections = 0

    for match in _INJECT_RE.finditer(level1_content):
        section_title = match.group(1).strip()
        source_rel = match.group(2).strip()
        fmt = match.group(3)

        # Resolve source path:
        # - "tools/..." → relative to project root
        # - "agents/..." → relative to project root
        # - "agent_instructions/..." → relative to agent's directory
        # - "catalogue/..." → relative to agent's directory (master agent identity cards)
        # - anything else → try agent_dir first, then project root
        if source_rel.startswith("tools/") or source_rel.startswith("agents/"):
            source_path = _PROJECT_ROOT / source_rel
        elif source_rel.startswith("agent_instructions/") or source_rel.startswith("catalogue/"):
            source_path = agent_dir / source_rel
        else:
            candidate = agent_dir / source_rel
            source_path = candidate if candidate.exists() else _PROJECT_ROOT / source_rel

        content = _read_source(source_path, fmt=fmt, catalogue_level=level)

        sections.append(f"---\n\n## {section_title}\n\n{content}")
        n_sections += 1
        last_end = match.end()

    if not sections:
        return level1_content

    assembled = "\n\n".join(sections)

    tokens_est = len(assembled) // 4
    print(
        f"System prompt loaded [{agent_name}/{level}]: {n_sections} sections, ~{tokens_est:,} tokens (estimated)",
        file=sys.stderr,
    )

    return assembled


if __name__ == "__main__":
    level_arg = "full"
    agent_arg = "mortality"
    for arg in sys.argv[1:]:
        if arg.startswith("--level="):
            level_arg = arg.split("=", 1)[1]
        elif arg.startswith("--agent="):
            agent_arg = arg.split("=", 1)[1]
        elif arg in ("--middle", "--full", "--light"):
            level_arg = arg.lstrip("-")
        elif arg in ("--mortality", "--report", "--master"):
            agent_arg = arg.lstrip("-")
    try:
        prompt = get_system_prompt(level=level_arg, agent_name=agent_arg)
        print(prompt)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
