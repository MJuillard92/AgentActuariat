#!/usr/bin/env python3
"""
analyze_report_template.py
Analyse offline d'un rapport actuariel de référence (PDF).

Produit un fichier JSON "template" contenant :
  - La structure du rapport (sections, tableaux, graphiques)
  - Un system prompt optimisé pour l'agent
  - Un résumé court pour le contexte RAG

Utilisation CLI :
    python analyze_report_template.py path/to/report.pdf
    python analyze_report_template.py path/to/report.pdf -o my_template.json

Utilisation programmatique :
    from analyze_report_template import analyze_report_pdf
    template = analyze_report_pdf("report.pdf")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Extraction du texte PDF
# ─────────────────────────────────────────────────────────────────────────────

_MAX_PDF_CHARS = 80_000  # limite pour l'envoi à l'API (~ 20k tokens)


def extract_pdf_text(pdf_path: str | Path, max_chars: int = _MAX_PDF_CHARS) -> str:
    """Extrait le texte du PDF page par page, limité à max_chars."""
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF requis : pip install pymupdf")

    doc = fitz.open(str(pdf_path))
    pages_text: list[str] = []
    total = 0
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text").strip()
        if not text:
            continue
        header = f"\n--- Page {i + 1} / {len(doc)} ---\n"
        block = header + text
        pages_text.append(block)
        total += len(block)
        if total >= max_chars:
            pages_text.append(
                f"\n[... texte tronqué à {max_chars:,} caractères "
                f"({len(doc) - i - 1} pages restantes non analysées) ...]"
            )
            break
    doc.close()
    return "\n".join(pages_text)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt système pour l'analyse structurée
# ─────────────────────────────────────────────────────────────────────────────

_ANALYSIS_SYSTEM_PROMPT = """\
Tu es un expert actuariel senior. On te fournit le texte complet (ou partiel) \
d'un rapport de synthèse actuariel portant sur une table de mortalité d'expérience.

Tu dois analyser ce rapport et retourner un JSON STRICT (sans markdown, sans commentaires) \
avec exactement la structure suivante :

{
  "report_title": "titre exact du rapport",
  "sections": [
    {"id": "S1", "title": "...", "description": "contenu résumé en 1 phrase"},
    ...
  ],
  "tables": [
    {
      "id": "T1",
      "name": "nom du tableau",
      "columns": ["col1", "col2", ...],
      "description": "ce que représente ce tableau et son rôle dans l'analyse"
    },
    ...
  ],
  "figures": [
    {
      "id": "F1",
      "type": "line|bar|scatter|heatmap|boxplot|autre",
      "title": "titre exact ou déduit du graphique",
      "x_axis": "variable ou axe en abscisse",
      "y_axis": "variable ou axe en ordonnée",
      "description": "ce que montre ce graphique et pourquoi il est important"
    },
    ...
  ],
  "key_metrics": ["liste des métriques clés mentionnées : SMR, qx, Ex, Dx, lx, ..."],
  "agent_system_prompt": "voir instructions ci-dessous",
  "rag_summary": "résumé de 4-6 phrases décrivant ce rapport (pour le contexte RAG)",
  "analysis_notes": "observations importantes sur la méthodologie, les données ou la structure"
}

═══════════════════════════════════════════════════════
INSTRUCTIONS POUR agent_system_prompt :
═══════════════════════════════════════════════════════
Ce champ doit contenir un prompt COMPLET et AUTO-SUFFISANT destiné à un agent IA \
qui va construire ce type de rapport actuariel. Il ne doit PAS référencer le rapport PDF.

Structure requise du agent_system_prompt :
1. Rôle et objectif (2-3 phrases)
2. LIVRABLES ATTENDUS — liste exhaustive des tableaux ET graphiques à produire,
   avec pour chaque élément :
   - Nom exact
   - Colonnes ou axes requis
   - Calculs ou transformations à réaliser
3. CRITÈRES DE VALIDATION (SMR cible, plages d'âge, comparaison TH/TF, etc.)
4. INSTRUCTIONS TECHNIQUES (utiliser les modules data_prep, exposure, crude_rates, etc.)
5. FORMAT DE SORTIE (résumé final structuré en français)
Longueur cible : 500-700 mots. Terminer par "Réponds en français avec des valeurs numériques précises."

IMPORTANT : Retourner UNIQUEMENT le JSON, sans aucun texte avant ou après.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Analyse LLM
# ─────────────────────────────────────────────────────────────────────────────

def analyze_with_llm(pdf_text: str, report_filename: str,
                     progress_fn=None) -> dict:
    """Envoie le texte du PDF à GPT-4o pour analyse structurée.

    Args:
        pdf_text: Texte extrait du PDF.
        report_filename: Nom du fichier pour contexte.
        progress_fn: Callable(str) optionnel pour afficher la progression.
    """
    import config

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY manquante dans .env")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    user_content = (
        f"Fichier source : {report_filename}\n\n"
        f"Texte du rapport :\n\n{pdf_text}"
    )

    if progress_fn:
        progress_fn(
            f"Envoi à {config.ANALYSIS_MODEL}… "
            f"({len(pdf_text):,} caractères / ~{len(pdf_text) // 4:,} tokens)"
        )

    response = client.chat.completions.create(
        model=config.ANALYSIS_MODEL,
        messages=[
            {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=4_096,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw = (response.choices[0].message.content or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Réponse LLM non JSON : {exc}\n\n{raw[:800]}")


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale — utilisée par le CLI et l'interface web
# ─────────────────────────────────────────────────────────────────────────────

def analyze_report_pdf(
    pdf_path: str | Path | None = None,
    pdf_bytes: bytes | None = None,
    filename: str = "rapport.pdf",
    progress_fn=None,
) -> dict:
    """Pipeline complet : PDF → dict template structuré.

    Accepte soit un chemin fichier (pdf_path) soit des bytes (pdf_bytes).

    Returns:
        dict avec les clés : report_title, sections, tables, figures,
        key_metrics, agent_system_prompt, rag_summary, analysis_notes, source_pdf.
    """
    if pdf_path is not None:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF introuvable : {pdf_path}")
        filename = pdf_path.name
        if progress_fn:
            progress_fn(f"Extraction du texte de {filename}…")
        pdf_text = extract_pdf_text(pdf_path)
    elif pdf_bytes is not None:
        if progress_fn:
            progress_fn(f"Extraction du texte de {filename}…")
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF requis : pip install pymupdf")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text: list[str] = []
        total = 0
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text("text").strip()
            if not text:
                continue
            header = f"\n--- Page {i + 1} / {len(doc)} ---\n"
            block = header + text
            pages_text.append(block)
            total += len(block)
            if total >= _MAX_PDF_CHARS:
                pages_text.append(
                    f"\n[... tronqué à {_MAX_PDF_CHARS:,} chars ...]"
                )
                break
        doc.close()
        pdf_text = "\n".join(pages_text)
    else:
        raise ValueError("Fournir pdf_path ou pdf_bytes")

    if progress_fn:
        progress_fn(f"{len(pdf_text):,} caractères extraits — analyse LLM en cours…")

    template = analyze_with_llm(pdf_text, filename, progress_fn)
    template["source_pdf"] = filename
    return template


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse un rapport actuariel PDF et génère un template JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python analyze_report_template.py rapport.pdf\n"
            "  python analyze_report_template.py rapport.pdf -o mon_template.json\n"
        ),
    )
    parser.add_argument("pdf", help="Chemin vers le rapport PDF de référence")
    parser.add_argument(
        "-o", "--output",
        help="Fichier de sortie JSON (défaut : <nom_pdf>_template.json)",
        default=None,
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    output_path = (
        Path(args.output)
        if args.output
        else pdf_path.with_name(pdf_path.stem + "_template.json")
    )

    try:
        template = analyze_report_pdf(pdf_path, progress_fn=print)
    except Exception as exc:
        print(f"\n[ERREUR] {exc}", file=sys.stderr)
        sys.exit(1)

    output_path.write_text(
        json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n✓ Template sauvegardé : {output_path}")
    print(f"  Titre      : {template.get('report_title', '?')}")
    print(f"  Sections   : {len(template.get('sections', []))}")
    print(f"  Tableaux   : {len(template.get('tables', []))}")
    print(f"  Graphiques : {len(template.get('figures', []))}")
    print(
        f"\nChargez ce fichier dans l'app → onglet '📋 Analyse Rapport' "
        f"ou onglet '🤖 Agent' → 'Charger template'."
    )


if __name__ == "__main__":
    main()
