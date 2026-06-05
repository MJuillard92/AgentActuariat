"""Tests — Post-processor LaTeX dans _04_redaction._wrap_bare_latex.

Le LLM Writer produit parfois des commandes LaTeX sans délimiteurs
`$...$`, notamment dans la section lissage. Sans wrapping, le PDF
affiche les commandes en clair. Le post-processor les détecte et les
entoure. Plan refonte PDF 2026-06-03 (Lot 6).
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.report.pipeline._04_redaction import _wrap_bare_latex


def test_wraps_widehat_command():
    """`\\widehat{q}_x` nu → `$\\widehat{q}_x$`."""
    src = r"Le taux \widehat{q}_x est lissé."
    out = _wrap_bare_latex(src)
    assert r"$\widehat{q}_x$" in out


def test_wraps_frac_command():
    """`\\frac{D_x}{E_x}` nu → entouré de `$…$`."""
    src = r"q_x = \frac{D_x}{E_x}"
    out = _wrap_bare_latex(src)
    assert r"$\frac{D_x}{E_x}$" in out


def test_idempotent_on_already_wrapped():
    """Une formule déjà délimitée n'est pas re-wrappée."""
    src = r"Le taux $\widehat{q}_x$ vaut $0{,}012$."
    out = _wrap_bare_latex(src)
    assert "$$" not in out  # pas de double wrap


def test_no_change_when_no_latex():
    """Texte sans LaTeX → identique."""
    src = "Le portefeuille compte 1 234 contrats et 56 décès."
    assert _wrap_bare_latex(src) == src


def test_handles_subscript_chain():
    """`\\sum_x D_x` → `$\\sum_x D_x$` (capture la chaîne subscript)."""
    src = r"On a \sum_{x=0}^{N} D_x au total."
    out = _wrap_bare_latex(src)
    assert r"$\sum_{x=0}^{N}$" in out  # \sum + ses limites capturées


def test_preserves_display_math():
    """Display math `$$…$$` reste en display."""
    src = r"$$\sum_x w_x q_x$$ illustre la formule."
    out = _wrap_bare_latex(src)
    assert r"$$\sum_x w_x q_x$$" in out
