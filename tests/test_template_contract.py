"""
Tests d'intégration pour scripts/check_template.py (US-3).

Wrapper pytest du script CLI : vérifie que le script retourne bien 0/1
selon la validité d'un template + registry, et produit un rapport lisible.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.check_template import main  # noqa: E402


# ───────────────── Fixtures ─────────────────

_FAKE_TOOL = '''"""
TOOL CONTRACT — fake.classify
═════════════════════════════

CATALOGUE METADATA
------------------
name          : fake.classify
version       : 1.0.0

INPUTS
------
params:
  request:
    type    : string
    note    : The request to classify.

OUTPUTS
-------
return_payload:
  objective : string
"""

def run(data, params):
    return {"objective": "x"}
'''

_VALID_YAML = textwrap.dedent("""
    session_inputs:
      - key: raw_user_request
        type: string
        required: true

    data_contract:
      master_from_modeling:
        - key: study_objective
          type: string
          produced_by:
            tool: fake.classify
            inputs: {request: raw_user_request}
            output_mapping: {objective: study_objective}

    sections:
      - id: preamble
        label: "Préambule"
        required: true
        dependencies: []
        narrative:
          text: |
            Objectif : {{ study_objective }}.
        llm_directives: {tone: neutre}
        visual_specs: []
""").strip()


@pytest.fixture
def fake_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Crée un mini-repo (tools/ + yaml) dans tmp_path."""
    tools_root = tmp_path / "tools"
    tools_root.mkdir()
    (tools_root / "classify.py").write_text(_FAKE_TOOL, encoding="utf-8")

    yaml_path = tmp_path / "template.yaml"
    yaml_path.write_text(_VALID_YAML, encoding="utf-8")
    return yaml_path, tools_root


# ───────────────── Tests via main() direct ─────────────────

def test_valid_template_exits_zero(fake_repo, capsys):
    yaml_path, tools_root = fake_repo
    code = main(yaml_path=yaml_path, tools_root=tools_root)
    assert code == 0


def test_invalid_template_exits_one(fake_repo, capsys):
    yaml_path, tools_root = fake_repo
    bad = _VALID_YAML.replace("fake.classify", "fake.inexistant")
    yaml_path.write_text(bad, encoding="utf-8")
    code = main(yaml_path=yaml_path, tools_root=tools_root)
    assert code == 1


def test_invalid_template_prints_error_message(fake_repo, capsys):
    yaml_path, tools_root = fake_repo
    bad = _VALID_YAML.replace("fake.classify", "fake.inexistant")
    yaml_path.write_text(bad, encoding="utf-8")
    main(yaml_path=yaml_path, tools_root=tools_root)
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "inexistant" in out or "fake.inexistant" in (capsys.readouterr().out)


# ───────────────── Tests via subprocess (CLI réel) ─────────────────

def test_cli_exits_zero_on_valid(fake_repo):
    yaml_path, tools_root = fake_repo
    script = _PROJECT_ROOT / "scripts" / "check_template.py"
    result = subprocess.run(
        [sys.executable, str(script),
         "--yaml", str(yaml_path),
         "--tools-root", str(tools_root),
         "--no-color"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_exits_one_on_invalid(fake_repo):
    yaml_path, tools_root = fake_repo
    bad = _VALID_YAML.replace("fake.classify", "fake.inexistant")
    yaml_path.write_text(bad, encoding="utf-8")
    script = _PROJECT_ROOT / "scripts" / "check_template.py"
    result = subprocess.run(
        [sys.executable, str(script),
         "--yaml", str(yaml_path),
         "--tools-root", str(tools_root),
         "--no-color"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "inexistant" in result.stdout or "inexistant" in result.stderr


# ───────────────── US-38 : section data_preprocessing ─────────────────

def test_data_preprocessing_section_exists():
    import yaml
    tpl = yaml.safe_load(open("knowledge_base/report_template/mortality_template.yaml"))
    ids = [s["id"] for s in tpl["sections"]]
    assert "data_preprocessing" in ids


def test_data_preprocessing_has_exclusion_table():
    import yaml
    tpl = yaml.safe_load(open("knowledge_base/report_template/mortality_template.yaml"))
    section = next(s for s in tpl["sections"] if s["id"] == "data_preprocessing")
    vs_ids = [v["id"] for v in section["visual_specs"]]
    assert "exclusion_table" in vs_ids


# ───────────────── US-39 : sections data_analysis_unisex + data_analysis_by_sex ─────────────────

def test_data_analysis_unisex_and_by_sex_exist():
    import yaml
    tpl = yaml.safe_load(open("knowledge_base/report_template/mortality_template.yaml"))
    ids = [s["id"] for s in tpl["sections"]]
    assert "data_analysis_unisex" in ids
    assert "data_analysis_by_sex" in ids


def test_data_analysis_sections_have_activation():
    import yaml
    tpl = yaml.safe_load(open("knowledge_base/report_template/mortality_template.yaml"))
    for sid in ("data_analysis_unisex", "data_analysis_by_sex"):
        section = next(s for s in tpl["sections"] if s["id"] == sid)
        assert section["activation"]["key"] == "gender_segmentation"
