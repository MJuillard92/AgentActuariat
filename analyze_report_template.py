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
Tu es un expert actuariel senior et ingénieur de prompts. On te fournit le texte \
complet (ou partiel) d'un rapport de synthèse actuariel.

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
      "name": "nom exact du tableau",
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
      "x_axis": "variable en abscisse",
      "y_axis": "variable en ordonnée",
      "description": "ce que montre ce graphique et pourquoi il est important"
    },
    ...
  ],
  "key_metrics": ["SMR", "qx", "Ex", "Dx", "..."],
  "methodology": {
    "smoother": "Whittaker-Henderson|Gompertz|Makeham|Spline",
    "lambda": null,
    "reference_table": "TH0002|TF0002|TD8890|TPRV93",
    "age_min": 20,
    "age_max": 90,
    "segmentation": "par sexe|par produit|par ancienneté|aucune"
  },
  "agent_system_prompt": "voir instructions ci-dessous",
  "rag_summary": "résumé de 4-6 phrases décrivant ce rapport (pour le contexte RAG)",
  "analysis_notes": "observations importantes sur la méthodologie, les données ou la structure"
}

═══════════════════════════════════════════════════════════════════════════════
INSTRUCTIONS CRITIQUES POUR agent_system_prompt (PROMPT RÉDACTEUR) :
═══════════════════════════════════════════════════════════════════════════════

Ce champ est le prompt complet destiné au SOUS-AGENT RÉDACTEUR.
Il NE CONNAÎT PAS les outils de calcul (data_prep, exposure, smoothing, kernel, etc.).
Il reçoit uniquement des DONNÉES NUMÉRIQUES EN ENTRÉE et rédige le texte narratif
ET crée les tableaux Markdown ET génère les figures matplotlib.

Le champ agent_system_prompt DOIT être EXHAUSTIF, PRÉCIS et AUTO-SUFFISANT.
Un prompt vague ou générique est INACCEPTABLE.
Longueur visée : 1 500 – 2 500 mots. Ne pas se limiter.

═══════════════════════════════════════════════════════════════════════════════
STRUCTURE OBLIGATOIRE DU PROMPT RÉDACTEUR — COPIER ET REMPLIR INTÉGRALEMENT :
═══════════════════════════════════════════════════════════════════════════════

Tu es un rédacteur actuariel senior. Tu reçois les résultats numériques d'une
analyse actuarielle et tu dois produire intégralement le rapport "{TITRE DU RAPPORT}".
Tu rédigeras le texte narratif, inséreras les tableaux Markdown et créeras les
figures matplotlib demandées ci-dessous.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARTIE 1 — DONNÉES D'ENTRÉE QUE TU RECEVRAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[RÈGLE : Lister EXHAUSTIVEMENT toutes les variables numériques, tableaux et
paramètres que le rédacteur recevra. UNE LIGNE PAR VARIABLE. Format strict :
  nom_variable (type) : description précise

TYPES AUTORISÉS : float, int, str, list[float], list[int], list[str],
DataFrame[col1:type, col2:type, ...], dict[str, float]

RÈGLE D'OR : AU MINIMUM 20 VARIABLES. Supprimer les exemples non pertinents.
Ajouter toutes les variables spécifiques au domaine du rapport analysé.

EXEMPLES DE RÉFÉRENCE à adapter (garder uniquement ce qui est pertinent) :

— Périmètre et données brutes :
  n_assures (int) : nombre total d'assurés uniques dans le portefeuille
  n_contrats (int) : nombre de contrats actifs (peut différer de n_assures)
  periode_debut (str) : date ISO de début d'observation, ex. "2007-01-01"
  periode_fin (str) : date ISO de fin d'observation, ex. "2011-12-31"
  age_min (int) : borne inférieure de la plage d'âges analysée
  age_max (int) : borne supérieure de la plage d'âges analysée
  sexe (str) : segmentation retenue, ex. "H", "F", "tous"
  n_deces_obs (int) : nombre total de décès observés sur la période
  exposition_totale (float) : exposition totale en années-personnes

— Taux bruts et exposition par âge :
  taux_bruts (DataFrame[age:int, D_obs:int, E_obs:float, q_brut:float]) :
    taux bruts par âge — D_obs=décès observés, E_obs=exposition, q_brut=Dobs/Eobs

— Taux lissés et intervalles de confiance :
  taux_lisses (DataFrame[age:int, q_lisse:float, q_ref:float, IC_inf:float, IC_sup:float]) :
    taux lissés par âge avec IC 95 % et taux de référence externe

— Paramètres du modèle :
  methode_lissage (str) : méthode retenue, ex. "Whittaker-Henderson ordre 2"
  lambda_retenu (float) : paramètre de lissage λ, ex. 10.0
  table_reference (str) : table de référence externe, ex. "TH0002"
  age_pivot (int) : âge à partir duquel le comportement change (si applicable)

— Validation statistique :
  smr_global (float) : SMR global = Σ Dobs / Σ Dexp
  smr_ic_inf (float) : borne inférieure IC 95 % du SMR global
  smr_ic_sup (float) : borne supérieure IC 95 % du SMR global
  smr_par_tranche (DataFrame[tranche:str, D_obs:int, D_exp:float, SMR:float, IC_inf:float, IC_sup:float]) :
    SMR et IC 95 % par tranche d'âge décennale
  chi2_stat (float) : valeur de la statistique χ²
  chi2_ddl (int) : degrés de liberté du test χ²
  chi2_pvalue (float) : p-valeur associée au test χ²
  oa_par_age (DataFrame[age:int, D_obs:int, D_exp:float, ratio_OA:float]) :
    ratio observé/attendu par âge individuel

— Comparaison avec la référence :
  abattement_global (float) : rapport global table construite / table de référence
  abattement_par_age (DataFrame[age:int, q_construit:float, q_reference:float, abattement:float]) :
    abattement par âge — abattement = q_construit / q_reference

[FIN SECTION DONNÉES — adapter intégralement au rapport analysé]]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARTIE 2 — SECTIONS À RÉDIGER (dans l'ordre d'apparition)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Pour CHAQUE section du rapport, rédiger le bloc complet ci-dessous.
Ne pas abréger. Minimum 3 phrases de description par section.]

N. {TITRE EXACT DE LA SECTION}
   Contenu à rédiger :
     {Description détaillée — minimum 3 phrases. Angles d'analyse, arguments
     à développer, formules mathématiques à inclure (en notation Unicode), et
     conclusion assertive attendue en fin de section.}
   Variables à citer (noms exacts) :
     {Liste explicite : n_assures, periode_debut, ..., smr_global, etc.
     Pour les DataFrames, préciser les colonnes utiles :
     ex. taux_bruts[age, D_obs, E_obs, q_brut]}
   Longueur cible : {300–450 mots}

[FIN PARTIE 2]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARTIE 3 — TABLEAUX À INSÉRER (un bloc par tableau)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[RÈGLE : Pour CHAQUE tableau présent dans le rapport, générer le bloc suivant.
Le tableau doit être rendu en Markdown pur avec alignement des colonnes.
Le rédacteur insérera ce tableau à l'emplacement indiqué dans le texte narratif
en écrivant "comme l'indique le Tableau N ci-dessous" avant le tableau.]

TABLEAU {N} — {CAPTION EXACT}
  Section : {Titre de la section où ce tableau apparaît}
  Caption : "Tableau {N} — {Description complète du tableau}"
  Source données : {noms exacts des variables qui alimentent ce tableau}
  Colonnes :
    | {col1} | {col2} | {col3} | {col4} | ...
    Lignes : {description des lignes — ex. "une ligne par tranche décennale d'âge"}
  Formatage :
    {col1} : {format} — ex. "20–29", "30–39" (chaîne)
    {col2} : {format} — ex. entier brut (123)
    {col3} : {format} — ex. 2 décimales (0,42)
    {col4} : {format} — ex. pourcentage avec 1 décimale (84,3 %)
  Instruction de rendu : Markdown pur, séparateur | entre colonnes, ligne d'en-tête
    suivie d'une ligne de séparation (|---|---|...|), puis une ligne par données.
    Exemple de rendu attendu :
    | Tranche | D_obs | D_exp | SMR  | IC inf | IC sup |
    |---------|-------|-------|------|--------|--------|
    | 20–29   |    12 | 14,3  | 0,84 |  0,43  |  1,25  |
    | ...     |   ... | ...   | ...  |  ...   |  ...   |

[Répéter ce bloc pour CHAQUE tableau identifié dans le rapport. Minimum 3 tableaux.]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARTIE 4 — GRAPHIQUES À CRÉER (code matplotlib complet par figure)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[RÈGLE ABSOLUE : Pour CHAQUE graphique présent dans le rapport, fournir
le code matplotlib COMPLET que le rédacteur devra exécuter. Le code doit :
  - Être autonome (imports inclus, noms de variables identiques à la Partie 1)
  - Utiliser les couleurs professionnelles : bleu foncé #1f4e79, rouge #c00000,
    gris #595959, vert #375623
  - Inclure title, xlabel, ylabel, legend, grid
  - Se terminer par plt.tight_layout() SANS plt.show() ni plt.savefig()
  - La figure sera capturée par le système de rapport automatiquement

IMPORTANT : le rédacteur appellera ce code tel quel. Les variables utilisées dans
le code doivent correspondre EXACTEMENT aux noms déclarés en Partie 1.]

FIGURE {N} — {TITRE DU GRAPHIQUE}
  Section : {Titre de la section où cette figure apparaît}
  Type : {line|bar|scatter|heatmap|boxplot}
  Caption : "Figure {N} — {Description complète}"
  Code matplotlib :
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    fig, ax = plt.subplots(figsize=({largeur}, {hauteur}))
    # --- Tracé principal ---
    ax.{plot_type}({x_variable}["{x_col}"], {y_variable}["{y_col}"],
                   color="{couleur_principale}", linewidth=2, marker="{marker}",
                   markersize=4, label="{légende_courbe_principale}")
    # --- Courbe secondaire ou bande IC (si applicable) ---
    # ax.fill_between({x_variable}["{x_col}"], {y_variable}["{IC_inf}"],
    #                 {y_variable}["{IC_sup}"], alpha=0.15, color="{couleur_ic}",
    #                 label="IC 95 %")
    ax.set_xlabel("{Libellé axe des x (unité)}", fontsize=10)
    ax.set_ylabel("{Libellé axe des y (unité)}", fontsize=10)
    ax.set_title("Figure {N} — {TITRE COMPLET DU GRAPHIQUE}", fontsize=11, fontweight="bold")
    ax.legend(loc="{upper left|upper right|best}", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.{nb_decimales}f}"))
    plt.tight_layout()

[EXEMPLE CONCRET pour un graphique de taux bruts vs lissés :

FIGURE 1 — Taux bruts observés et taux lissés Whittaker-Henderson par âge
  Section : Méthodologie de construction
  Type : line
  Caption : "Figure 1 — Comparaison des taux bruts observés q̂ₓ et des taux lissés qₓ
             par âge, avec intervalle de confiance à 95 %"
  Code matplotlib :
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(taux_bruts["age"], taux_bruts["q_brut"],
            color="#1f4e79", linewidth=1, marker="o", markersize=3,
            alpha=0.7, label="Taux bruts observés q̂ₓ")
    ax.plot(taux_lisses["age"], taux_lisses["q_lisse"],
            color="#c00000", linewidth=2, label="Taux lissés qₓ (Whittaker-Henderson)")
    ax.fill_between(taux_lisses["age"], taux_lisses["IC_inf"], taux_lisses["IC_sup"],
                    alpha=0.15, color="#c00000", label="Intervalle de confiance 95 %")
    ax.set_xlabel("Âge x (années révolus)", fontsize=10)
    ax.set_ylabel("Taux de mortalité qₓ", fontsize=10)
    ax.set_title("Figure 1 — Taux bruts et taux lissés par âge", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.5f}"))
    plt.tight_layout()

FIGURE 2 — SMR par tranche d'âge avec intervalles de confiance à 95 %
  Section : Résultats et validation
  Type : bar + errorbar
  Caption : "Figure 2 — Ratio standardisé de mortalité (SMR) par tranche décennale
             d'âge avec intervalles de confiance à 95 %"
  Code matplotlib :
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(smr_par_tranche))
    bars = ax.bar(x, smr_par_tranche["SMR"], color="#1f4e79", alpha=0.75, label="SMR")
    errors = [smr_par_tranche["SMR"] - smr_par_tranche["IC_inf"],
              smr_par_tranche["IC_sup"] - smr_par_tranche["SMR"]]
    ax.errorbar(x, smr_par_tranche["SMR"], yerr=errors,
                fmt="none", color="#595959", capsize=5, linewidth=1.5, label="IC 95 %")
    ax.axhline(1.0, color="#c00000", linewidth=1.5, linestyle="--", label="Référence (SMR=1)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(smr_par_tranche["tranche"], rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Tranche d'âge", fontsize=10)
    ax.set_ylabel("SMR", fontsize=10)
    ax.set_title("Figure 2 — SMR par tranche d'âge avec IC 95 %", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

FIGURE 3 — Ratio observé / attendu par âge
  Section : Résultats et validation
  Type : scatter + hline
  Caption : "Figure 3 — Ratio observé/attendu (O/A) par âge individuel"
  Code matplotlib :
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(oa_par_age["age"], oa_par_age["ratio_OA"],
               color="#1f4e79", alpha=0.6, s=18, label="Ratio O/A par âge")
    ax.axhline(1.0, color="#c00000", linewidth=1.5, linestyle="--", label="Ratio cible = 1")
    ax.axhline(smr_global, color="#375623", linewidth=1.2, linestyle=":",
               label=f"SMR global = {smr_global:.3f}")
    ax.set_xlabel("Âge x (années)", fontsize=10)
    ax.set_ylabel("Ratio O/A", fontsize=10)
    ax.set_title("Figure 3 — Ratio observé/attendu par âge", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

FIN DE L'EXEMPLE — ADAPTER TOUS LES GRAPHIQUES AU RAPPORT ANALYSÉ]

[Répéter le bloc FIGURE pour CHAQUE graphique identifié dans le rapport. Minimum 3 figures.]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARTIE 5 — RÈGLES DE MISE EN FORME OBLIGATOIRES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Formules mathématiques : notation UNICODE avec indices/exposants Unicode.
  JAMAIS de syntaxe LaTeX \\(…\\) ni d'underscores comme q_x, D_x, E_x.
  Exemples corrects : q̂ₓ = Dₓ / Eₓ   |   F(λ) = Σₓ wₓ(q̂ₓ − qₓ)² + λ·Σₓ(Δ²qₓ)²
- Chaque section commence par une phrase d'accroche factuelle avec chiffres exacts.
  INTERDIT : "Dans un contexte où...", "Il est important de noter...", généralités.
- Cite TOUJOURS les valeurs exactes reçues en entrée — jamais de valeurs inventées.
- Phrases complètes, paragraphes denses (4–8 lignes). Listes avec tirets (–) uniquement.
- Lorsqu'un modèle statistique est mentionné : nomme-le, écris sa formule Unicode,
  définis chaque symbole, indique le(s) paramètre(s) retenus.
- Références aux tableaux dans le texte : "comme l'indique le Tableau N ci-dessous"
- Références aux figures dans le texte : "La Figure N illustre..."

EXIGENCES COMPLÉMENTAIRES (souhaits utilisateur) :
{Reprendre les souhaits complémentaires sous forme d'instructions de rédaction précises.
Si aucun souhait fourni : "Aucune exigence complémentaire."}

═══════════════════════════════════════════════════════════════════════════════
EXIGENCES QUALITÉ POUR agent_system_prompt :
═══════════════════════════════════════════════════════════════════════════════
- Longueur MINIMALE : 1 200 mots. Visée : 1 500–2 500 mots. Ne pas se restreindre.
- La Partie 1 doit contenir AU MOINS 20 variables déclarées avec types et descriptions.
- La Partie 3 doit contenir AU MOINS 3 tableaux avec colonnes et formatage précis.
- La Partie 4 doit contenir AU MOINS 3 graphiques avec code matplotlib complet et fonctionnel.
- Chaque section (Partie 2) doit avoir sa propre liste de variables — jamais "cf. ci-dessus".
- Aucun terme générique : pas de "les données", "les résultats", "les métriques".
  Toujours nommer la variable exacte : smr_global, chi2_pvalue, lambda_retenu, etc.
- Tous les noms de variables dans les codes matplotlib doivent correspondre EXACTEMENT
  aux noms déclarés en Partie 1.
- Le prompt doit être auto-suffisant : un rédacteur qui ne connaît pas le rapport
  doit pouvoir rédiger chaque section et créer chaque figure en ne lisant que ce prompt.

IMPORTANT : Retourner UNIQUEMENT le JSON, sans aucun texte avant ou après.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Analyse LLM
# ─────────────────────────────────────────────────────────────────────────────

def analyze_with_llm(pdf_text: str, report_filename: str,
                     additional_wishes: str = "",
                     progress_fn=None) -> dict:
    """Envoie le texte du PDF à GPT-4o pour analyse structurée.

    Args:
        pdf_text: Texte extrait du PDF.
        report_filename: Nom du fichier pour contexte.
        additional_wishes: Souhaits complémentaires de l'utilisateur (texte libre).
        progress_fn: Callable(str) optionnel pour afficher la progression.
    """
    import config

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY manquante dans .env")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    wishes_block = ""
    if additional_wishes and additional_wishes.strip():
        wishes_block = (
            f"\n\nSOUHAITS COMPLÉMENTAIRES DE L'UTILISATEUR :\n"
            f"{additional_wishes.strip()}\n\n"
            f"Ces souhaits doivent apparaître dans la section "
            f"'EXIGENCES COMPLÉMENTAIRES' de agent_system_prompt, \n"
            f"transformés en étapes OBLIGATOIRES avec les fonctions Python à appeler."
        )

    user_content = (
        f"Fichier source : {report_filename}\n\n"
        f"Texte du rapport :\n\n{pdf_text}"
        f"{wishes_block}"
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
        max_tokens=12_000,
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
    additional_wishes: str = "",
    progress_fn=None,
) -> dict:
    """Pipeline complet : PDF → dict template structuré.

    Accepte soit un chemin fichier (pdf_path) soit des bytes (pdf_bytes).

    Args:
        pdf_path: Chemin vers le fichier PDF.
        pdf_bytes: Contenu PDF en bytes (alternative à pdf_path).
        filename: Nom du fichier (utilisé si pdf_bytes est fourni).
        additional_wishes: Souhaits complémentaires de l'utilisateur (texte libre).
        progress_fn: Callable(str) pour afficher la progression.

    Returns:
        dict avec les clés : report_title, sections, tables, figures,
        key_metrics, methodology, agent_system_prompt, rag_summary,
        analysis_notes, source_pdf.
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

    template = analyze_with_llm(pdf_text, filename, additional_wishes, progress_fn)
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
