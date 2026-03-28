"""
graph_to_template.py
--------------------
Outil offline : analyse un graphique (image) via OpenAI Vision
et génère un template matplotlib prêt à l'emploi.

Usage :
    python graph_to_template.py image.png
    python graph_to_template.py image.png --nom exposition_par_age
    python graph_to_template.py --dossier ./graphiques_rapport/
    python graph_to_template.py --pdf rapport.pdf --pages 6,9,11,12

Sortie :
    Un fichier .py par graphique dans ./templates_matplotlib/
"""

from __future__ import annotations

import openai
import base64
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime


SYSTEM_PROMPT = """Tu es un expert en visualisation de données actuarielles.
Ta mission est d'analyser précisément un graphique fourni en image
et de générer un template Python matplotlib complet permettant de reproduire
ce graphique avec de nouvelles données.

Tu dois produire UNIQUEMENT du code Python, sans aucun texte avant ou après,
sans blocs markdown (pas de ```python), juste le code brut.

Le code doit :
1. Définir des variables DONNÉES au début (à remplacer par les vraies données)
2. Reproduire fidèlement : type de graphique, couleurs, styles de lignes,
   marqueurs, épaisseurs, grille, légende, titres, labels d'axes
3. Être fonctionnel et exécutable immédiatement avec des données synthétiques
4. Sauvegarder le graphique avec plt.savefig() à la fin
5. Appeler plt.close() pour libérer la mémoire

Structure attendue du code :
    # === DONNÉES (à remplacer par les vraies valeurs) ===
    ages = [...]
    valeurs_h = [...]
    ...

    # === GRAPHIQUE ===
    fig, ax = plt.subplots(figsize=(...))
    ...
    plt.savefig('NOM_GRAPHIQUE.png', dpi=150, bbox_inches='tight')
    plt.close()

Sois précis sur :
- Les couleurs exactes (hex si possible)
- Les styles de lignes (solid, dashed, dotted, dashdot)
- Les marqueurs (o, x, *, s, ^, etc.)
- L'épaisseur des lignes (linewidth)
- La transparence si zone remplie (alpha)
- La position de la légende
- Les formats des axes (%, logarithmique, etc.)
- Les annotations ou textes dans le graphique
"""

USER_PROMPT_TEMPLATE = """Analyse ce graphique issu d'un rapport actuariel.

Nom suggéré pour le fichier de sortie : {nom}

Génère le template matplotlib Python complet pour reproduire ce graphique
avec de nouvelles données, en suivant exactement les instructions du system prompt.
"""


def image_to_base64(path: str) -> tuple[str, str]:
    path = Path(path)
    ext = path.suffix.lower()
    media_types = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png', '.gif': 'image/gif', '.webp': 'image/webp',
    }
    media_type = media_types.get(ext, 'image/jpeg')
    with open(path, 'rb') as f:
        data = base64.standard_b64encode(f.read()).decode('utf-8')
    return data, media_type


def extraire_pages_pdf(pdf_path: str, pages: list[int], output_dir: str) -> list[str]:
    import subprocess
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for page_num in pages:
        output_prefix = str(output_dir / f"page_{page_num:03d}")
        cmd = ['pdftoppm', '-jpeg', '-r', '150',
               '-f', str(page_num), '-l', str(page_num),
               pdf_path, output_prefix]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            generated = sorted(output_dir.glob(f"page_{page_num:03d}*.jpg"))
            if generated:
                images.append(str(generated[0]))
                print(f"  Page {page_num} extraite → {generated[0].name}")
        else:
            print(f"  ⚠ Impossible d'extraire la page {page_num}")
    return images


def analyser_graphique(
    client: openai.OpenAI,
    image_path: str,
    nom: str,
    model: str = "gpt-4o"
) -> str:
    print(f"  Analyse de {Path(image_path).name} via {model}...")
    image_data, media_type = image_to_base64(image_path)
    data_url = f"data:{media_type};base64,{image_data}"
    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"}
                    },
                    {
                        "type": "text",
                        "text": USER_PROMPT_TEMPLATE.format(nom=nom)
                    }
                ]
            }
        ]
    )
    code = response.choices[0].message.content
    if code.startswith("```"):
        lines = code.split('\n')
        code = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
    return code


def sauvegarder_template(code: str, nom: str, output_dir: str) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = f'''"""
Template matplotlib — {nom}
Généré automatiquement par graph_to_template.py le {timestamp}

UTILISATION :
1. Remplacez les variables dans la section DONNÉES par vos vraies valeurs
2. Exécutez : python {nom}.py
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11

'''
    lines = code.split('\n')
    lines_filtered = [
        l for l in lines
        if not l.strip().startswith('import matplotlib')
        and not l.strip().startswith('import numpy')
        and l.strip() not in ('import matplotlib.pyplot as plt', 'import numpy as np')
    ]
    full_code = header + '\n'.join(lines_filtered)
    output_path = output_dir / f"{nom}.py"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_code)
    return str(output_path)


def noms_rapport_allianz() -> dict[int, str]:
    return {
        6:  "exposition_et_deces_par_age",
        9:  "obs_vs_pred_par_age",
        10: "deces_predits_vs_observes_3d",
        11: "abattements_tables_reglementaires",
        12: "regression_logits",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse des graphiques et génère des templates matplotlib"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('image', nargs='?', help="Chemin vers une image de graphique")
    group.add_argument('--dossier', help="Dossier contenant plusieurs images")
    group.add_argument('--pdf', help="Fichier PDF source")
    parser.add_argument('--nom', default=None)
    parser.add_argument('--pages', default=None,
                        help="Pages du PDF à extraire (ex: 6,9,11,12)")
    parser.add_argument('--output', default='./templates_matplotlib')
    parser.add_argument('--model', default='gpt-4o',
                        choices=['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-5'])
    return parser.parse_args()


def main():
    args = parse_args()
    client = openai.OpenAI()
    output_dir = args.output
    model = args.model
    templates_produits = []

    print(f"\n{'='*60}")
    print("  graph_to_template.py — Générateur de templates matplotlib")
    print(f"  Modèle : {args.model}")
    print(f"{'='*60}\n")

    if args.image:
        images = [(args.image, args.nom or Path(args.image).stem)]
    elif args.dossier:
        extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        dossier = Path(args.dossier)
        images = [
            (str(f), f.stem)
            for f in sorted(dossier.iterdir())
            if f.suffix.lower() in extensions
        ]
        print(f"  {len(images)} images trouvées dans {args.dossier}\n")
    elif args.pdf:
        if not args.pages:
            print("❌ --pages requis avec --pdf (ex: --pages 6,9,11,12)")
            sys.exit(1)
        pages = [int(p.strip()) for p in args.pages.split(',')]
        noms_suggeres = noms_rapport_allianz()
        print(f"  Extraction des pages {pages} depuis {args.pdf}...\n")
        tmp_dir = Path(output_dir) / "_pages_extraites"
        chemins = extraire_pages_pdf(args.pdf, pages, str(tmp_dir))
        images = [
            (chemin, noms_suggeres.get(pages[i], f"graphique_page_{pages[i]}"))
            for i, chemin in enumerate(chemins)
        ]

    for image_path, nom in images:
        print(f"► Graphique : {nom}")
        try:
            code = analyser_graphique(client, image_path, nom, model=model)
            output_path = sauvegarder_template(code, nom, output_dir)
            templates_produits.append({'nom': nom, 'source': image_path, 'output': output_path})
            print(f"  ✓ Template sauvegardé → {output_path}\n")
        except Exception as e:
            print(f"  ✗ Erreur : {e}\n")

    print(f"\n{'='*60}")
    print(f"  {len(templates_produits)} template(s) générés dans {output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
