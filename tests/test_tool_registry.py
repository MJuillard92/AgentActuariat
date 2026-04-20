"""
Tests pour knowledge_base/report_template/tool_registry.py (US-1).

Registry attendu : {qualified_name: ToolSpec}
ToolSpec = {inputs: dict[name, type_str], outputs: dict[name, type_str], path: str}
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from knowledge_base.report_template.tool_registry import build_registry  # noqa: E402


# ───────────────── Fixture : faux répertoire de tools ─────────────────

_FAKE_TOOL = dedent('''
    """
    TOOL CONTRACT — fake_domain.fake_tool
    ════════════════════════════════════

    IDENTITY
    --------
    name          : fake_domain.fake_tool
    version       : 1.0.0

    DESCRIPTION
    -----------
    Outil factice pour tests.

    INPUTS
    ------
    params:
      foo:
        type    : string
        note    : description
      bar:
        type    : int
        default : 0

    OUTPUTS
    -------
    data_store_keys_written: []
    return_payload:
      result : string
      count  : int

    CATALOGUE METADATA
    ------------------
    display_name      : Fake tool
    short_description : Outil de test
    domain            : test
    capability_group  : test
    depends_on        : []
    required_by       : []
    client_visible    : false
    """
    def run(data, params=None):
        return {"result": "ok", "count": 1}
''').strip()


@pytest.fixture
def fake_tools_root(tmp_path: Path) -> Path:
    root = tmp_path / "tools"
    pkg = root / "fake_domain"
    pkg.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "fake_tool.py").write_text(_FAKE_TOOL)
    return root


# ───────────────── Tests ─────────────────

def test_build_registry_returns_dict(fake_tools_root):
    reg = build_registry(fake_tools_root)
    assert isinstance(reg, dict)


def test_build_registry_discovers_fake_tool(fake_tools_root):
    reg = build_registry(fake_tools_root)
    assert "fake_domain.fake_tool" in reg


def test_tool_spec_has_required_fields(fake_tools_root):
    reg = build_registry(fake_tools_root)
    spec = reg["fake_domain.fake_tool"]
    assert "inputs" in spec
    assert "outputs" in spec
    assert "path" in spec


def test_inputs_parsed_with_types(fake_tools_root):
    reg = build_registry(fake_tools_root)
    spec = reg["fake_domain.fake_tool"]
    assert "foo" in spec["inputs"]
    assert "bar" in spec["inputs"]
    assert spec["inputs"]["foo"] == "string"
    assert spec["inputs"]["bar"] == "int"


def test_outputs_parsed_from_return_payload(fake_tools_root):
    reg = build_registry(fake_tools_root)
    spec = reg["fake_domain.fake_tool"]
    assert "result" in spec["outputs"]
    assert "count" in spec["outputs"]


def test_path_points_to_file(fake_tools_root):
    reg = build_registry(fake_tools_root)
    spec = reg["fake_domain.fake_tool"]
    assert Path(spec["path"]).exists()
    assert spec["path"].endswith("fake_tool.py")


def test_collision_raises(tmp_path: Path):
    """Deux tools déclarant le même name dans CATALOGUE METADATA → ValueError."""
    root = tmp_path / "tools"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "a" / "__init__.py").write_text("")
    (root / "b" / "__init__.py").write_text("")
    # Même qualified name dans deux fichiers distincts
    (root / "a" / "dup.py").write_text(_FAKE_TOOL)
    (root / "b" / "dup.py").write_text(_FAKE_TOOL)
    with pytest.raises(ValueError, match="collision"):
        build_registry(root)


def test_real_repo_has_multiple_tools():
    """Smoke test : sur le vrai repo, au moins 5 tools découverts."""
    real_root = _PROJECT_ROOT / "tools"
    reg = build_registry(real_root)
    assert len(reg) >= 5, f"Attendu ≥ 5 tools, trouvé {len(reg)}: {list(reg.keys())}"
