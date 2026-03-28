"""
report_agent/graph_template_builder.py
Génère les figures du rapport en s'appuyant sur un PDF de référence.

Pattern :
  1. Extraire les pages graphiques du PDF de référence via pdftoppm
  2. Analyser chaque graphique via OpenAI Vision → template matplotlib (avec cache)
  3. Exécuter chaque template avec les vraies données → bytes PNG
  4. Retourner dict{"key": bytes} pour prebuilt_figures dans generate_narrative_report()
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Any

import openai
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph_to_template import analyser_graphique, sauvegarder_template


# Mapping page PDF de référence → clé rapport
# À adapter selon les pages réelles de ton PDF de référence
_DEFAULT_PAGE_MAPPING = {
    6:  "exposure",
    9:  "rates",
    11: "smr",
    12: "comparison",
}

_KEY_TO_NAME = {
    "exposure":   "exposition_par_age",
    "rates":      "taux_bruts_lisses",
    "smr":        "smr_par_groupe",
    "oa":         "obs_vs_attendus",
    "comparison": "comparaison_reference",
}

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates_matplotlib"
_TMP_PAGES_DIR = Path(__file__).parent.parent / ".tmp_pdf_pages"


def _extraire_page_pdf(pdf_path: str, page_num: int) -> str | None:
    _TMP_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    prefix = str(_TMP_PAGES_DIR / f"page_{page_num:03d}")
    cmd = ["pdftoppm", "-jpeg", "-r", "150",
           "-f", str(page_num), "-l", str(page_num),
           pdf_path, prefix]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return None
    candidates = sorted(_TMP_PAGES_DIR.glob(f"page_{page_num:03d}*.jpg"))
    return str(candidates[0]) if candidates else None


def _obtenir_template(
    client: openai.OpenAI,
    image_path: str,
    fig_key: str,
    model: str = "gpt-4o",
) -> str | None:
    """Retourne le code du template depuis le cache ou via Vision API."""
    nom = _KEY_TO_NAME.get(fig_key, fig_key)
    template_path = _TEMPLATES_DIR / f"{nom}.py"

    if template_path.exists():
        print(f"  [cache] Template '{nom}' déjà disponible.")
        return template_path.read_text(encoding="utf-8")

    print(f"  [vision] Analyse de la page pour '{fig_key}'...")
    try:
        code = analyser_graphique(client, image_path, nom, model=model)
        sauvegarder_template(code, nom, str(_TEMPLATES_DIR))
        return code
    except Exception as e:
        print(f"  [erreur] graph_to_template pour '{fig_key}': {e}")
        return None


def _executer_template(code: str, data: dict[str, Any]) -> bytes | None:
    """Exécute le template matplotlib avec les données réelles.

    Injecte les variables du dict `data` pour remplacer les données synthétiques.
    Intercepte plt.savefig() pour capturer les bytes PNG au lieu d'écrire un fichier.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    buf = io.BytesIO()
    namespace = {"plt": plt, "np": np, "io": io, "__buf__": buf, **data}

    original_savefig = plt.savefig

    def _capture_savefig(*args, **kwargs):
        kwargs["fname"] = buf
        kwargs.setdefault("format", "png")
        kwargs.setdefault("dpi", 150)
        kwargs.setdefault("bbox_inches", "tight")
        original_savefig(**kwargs)

    namespace["plt"].savefig = _capture_savefig

    try:
        exec(code, namespace)  # noqa: S102
        buf.seek(0)
        result = buf.read()
        return result if len(result) > 100 else None
    except Exception as e:
        print(f"  [erreur] Exécution template : {e}")
        return None
    finally:
        plt.savefig = original_savefig
        plt.close("all")


def build_figures_from_reference(
    pdf_reference_path: str,
    exposure_df: Any | None = None,
    page_mapping: dict[int, str] | None = None,
    model: str = "gpt-4o",
) -> dict[str, bytes]:
    """Génère les figures du rapport à partir du PDF de référence + données réelles.

    Args:
        pdf_reference_path: Chemin vers le PDF de rapport de référence
        exposure_df:        DataFrame avec colonnes age, E_x, D_x, q_brut, q_lisse, etc.
        page_mapping:       Dict {numéro_page: clé_rapport} — si None, utilise le défaut
        model:              Modèle OpenAI Vision (défaut: gpt-4o)

    Returns:
        Dict {"exposure": bytes_png, ...} compatible avec prebuilt_figures
    """
    if not Path(pdf_reference_path).exists():
        return {}

    client = openai.OpenAI()
    mapping = page_mapping or _DEFAULT_PAGE_MAPPING
    figures: dict[str, bytes] = {}

    # Préparer les données réelles pour injection dans les templates
    data_injection: dict[str, Any] = {}
    if exposure_df is not None:
        try:
            data_injection["ages"] = exposure_df["age"].values.tolist()
            for col, aliases in [
                ("E_x",    ["E_x", "ex"]),
                ("D_x",    ["D_x"]),
                ("q_brut", ["q_brut"]),
                ("q_lisse",["q_lisse"]),
                ("IC_inf", ["IC_inf"]),
                ("IC_sup", ["IC_sup"]),
                ("D_exp",  ["D_exp"]),
            ]:
                if col in exposure_df.columns:
                    vals = exposure_df[col].values.tolist()
                    for alias in aliases:
                        data_injection[alias] = vals
                    # Variantes en ‰
                    if col in ("q_brut", "q_lisse"):
                        data_injection[f"{col}_permille"] = (
                            (exposure_df[col] * 1000).values.tolist()
                        )
        except Exception as e:
            print(f"  [avertissement] Injection données : {e}")

    for page_num, fig_key in sorted(mapping.items()):
        print(f"\n► Page {page_num} → '{fig_key}'")

        image_path = _extraire_page_pdf(pdf_reference_path, page_num)
        if not image_path:
            print(f"  [skip] Impossible d'extraire la page {page_num}")
            continue

        code = _obtenir_template(client, image_path, fig_key, model=model)
        if not code:
            continue

        png_bytes = _executer_template(code, data_injection)
        if png_bytes:
            figures[fig_key] = png_bytes
            print(f"  ✓ Figure '{fig_key}' générée ({len(png_bytes)//1024} Ko)")
        else:
            print(f"  ✗ Échec génération '{fig_key}'")

    return figures
