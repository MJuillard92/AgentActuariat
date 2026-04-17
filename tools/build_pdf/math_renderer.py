"""
tools/build_pdf/math_renderer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rendu de formules LaTeX en PNG via matplotlib + moteur TeX natif.

Utilise matplotlib usetex=True (LaTeX installé obligatoire) pour
une qualité identique au rendu LaTeX documentaire.

API publique :
    render_formula(latex_str, display=False, fontsize=11) -> str
        latex_str : expression LaTeX sans les $ (ex: r"q_x = \frac{D_x}{E_x}")
        display   : True = formule centrée (mode display $$), False = inline ($)
        fontsize  : taille de police en pt (doit correspondre au corps du texte)
        → retourne le chemin du PNG (cache disque /tmp/math_cache/)

    split_math(text) -> list[tuple[str, bool]]
        Découpe un texte en alternant texte brut et formules LaTeX.
        Chaque tuple : (contenu, is_formula)
        is_formula=True + contenu = latex sans $
        Exemples :
            "Le taux $q_x = D_x / E_x$ est ..." →
            [("Le taux ", False), ("q_x = D_x / E_x", True), (" est ...", False)]

Cache :
    /tmp/math_cache/  — PNG nommés par hash du contenu
    Ne pas nettoyer pendant une session (réutilisés lors des retry de section)
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_CACHE_DIR = Path("/tmp/math_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Paquets LaTeX chargés pour toutes les formules actuarielles
_LATEX_PREAMBLE = "\n".join([
    r"\usepackage{amsmath}",
    r"\usepackage{amssymb}",
    r"\usepackage{bm}",          # bold math
])

# Regex pour détecter les formules : $$ ... $$ (display) et $ ... $ (inline)
# Ordre : display d'abord pour éviter la collision
_DISPLAY_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_INLINE_RE  = re.compile(r"\$(.+?)\$",     re.DOTALL)


def _cache_path(latex_str: str, display: bool, fontsize: float) -> Path:
    """Calcule le chemin cache du PNG pour une formule donnée."""
    key = hashlib.md5(f"{latex_str}|{display}|{fontsize}".encode()).hexdigest()[:16]
    return _CACHE_DIR / f"math_{key}.png"


def render_formula(
    latex_str: str,
    display:   bool  = False,
    fontsize:  float = 11.0,
    dpi:       int   = 200,
) -> str | None:
    """
    Rend une formule LaTeX en PNG haute résolution.

    Args:
        latex_str : expression LaTeX sans délimiteurs $ (ex: r"\frac{D_x}{E_x}")
        display   : True = mode display (grande formule centrée), False = inline
        fontsize  : taille en pt (9 pour corps de texte ReportLab → utiliser 11 pour compenser dpi)
        dpi       : résolution du PNG (200 par défaut — bon compromis taille/qualité)

    Returns:
        Chemin absolu vers le PNG, ou None en cas d'erreur.
    """
    latex_str = latex_str.strip()
    if not latex_str:
        return None

    out_path = _cache_path(latex_str, display, fontsize)
    if out_path.exists():
        return str(out_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "text.usetex":         True,
            "text.latex.preamble": _LATEX_PREAMBLE,
            "font.family":         "serif",
            "font.size":           fontsize,
        })

        # Délimiteurs selon le mode
        delim_l = r"\[" if display else r"\("
        delim_r = r"\]" if display else r"\)"
        full_expr = f"{delim_l}{latex_str}{delim_r}"

        fig, ax = plt.subplots(figsize=(0.5, 0.3))
        ax.set_axis_off()
        ax.set_position([0, 0, 1, 1])

        text_obj = ax.text(
            0.5, 0.5, full_expr,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=fontsize,
            color="black",
        )

        # Ajuster la figure à la taille réelle du texte rendu
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox     = text_obj.get_window_extent(renderer=renderer)
        pad_x    = 0.06   # padding horizontal en pouces
        pad_y    = 0.04   # padding vertical

        w_in = max(bbox.width  / dpi + 2 * pad_x, 0.4)
        h_in = max(bbox.height / dpi + 2 * pad_y, 0.15)
        fig.set_size_inches(w_in, h_in)

        fig.savefig(
            str(out_path),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=pad_y,
            transparent=True,
            facecolor="none",
        )
        plt.close(fig)

        log.debug("[math_renderer] rendu '%s' → %s", latex_str[:40], out_path.name)
        return str(out_path)

    except Exception as exc:
        log.warning("[math_renderer] échec rendu '%s' : %s", latex_str[:60], exc)
        # Nettoyage fichier partiel
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        return None


def split_math(text: str) -> list[tuple[str, bool, bool]]:
    """
    Découpe un texte en segments alternant texte brut et formules LaTeX.

    Returns:
        list of (content, is_formula, is_display)
        - is_formula  : True si le segment est une formule LaTeX
        - is_display  : True si formule en mode display ($$...$$)
    """
    if not text:
        return [(text, False, False)]

    segments: list[tuple[str, bool, bool]] = []
    pos = 0

    # On cherche $$ ... $$ et $ ... $ en même temps, dans l'ordre d'apparition
    # Regex combinée : group 1 = display, group 2 = inline
    combined = re.compile(r"\$\$(.*?)\$\$|\$(.+?)\$", re.DOTALL)

    for m in combined.finditer(text):
        # Texte brut avant la formule
        if m.start() > pos:
            segments.append((text[pos:m.start()], False, False))

        if m.group(1) is not None:
            # Match display $$ ... $$
            segments.append((m.group(1).strip(), True, True))
        else:
            # Match inline $ ... $
            segments.append((m.group(2).strip(), True, False))

        pos = m.end()

    # Texte restant
    if pos < len(text):
        segments.append((text[pos:], False, False))

    return segments


def has_math(text: str) -> bool:
    """Retourne True si le texte contient au moins une formule LaTeX."""
    return bool(_DISPLAY_RE.search(text) or _INLINE_RE.search(text))
