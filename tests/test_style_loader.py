"""Tests pour load_style() (US-8)."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from knowledge_base.report_template.template_loader import load_style  # noqa: E402


def test_load_style_from_default():
    style = load_style()
    assert "colors" in style
    assert style["colors"]["primary"].startswith("#")
    assert "table" in style


def test_load_style_from_explicit_path(tmp_path):
    p = tmp_path / "style.yaml"
    p.write_text("colors:\n  primary: '#FF0000'\n", encoding="utf-8")
    style = load_style(p)
    assert style["colors"]["primary"] == "#FF0000"


def test_load_style_missing_file_returns_defaults(tmp_path):
    style = load_style(tmp_path / "missing.yaml")
    # defaults minimaux
    assert "colors" in style
    assert "primary" in style["colors"]
