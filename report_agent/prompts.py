"""
report_agent/prompts.py
Prompts système et par section pour le rédacteur LLM du rapport actuariel.
Conçus pour être génériques (mortalité, provisionnement non-vie, VIF, etc.).
"""
from __future__ import annotations
from pathlib import Path

# Charger l'exemple de formule depuis le fichier Markdown
_FORMULA_EXAMPLE_PATH = Path(__file__).parent / "formula_example.md"
_FORMULA_EXAMPLE = _FORMULA_EXAMPLE_PATH.read_text(encoding="utf-8") if _FORMULA_EXAMPLE_PATH.exists() else ""

# ─────────────────────────────────────────────────────────────────────────────
# Prompt système — générique, injecté dans tous les appels LLM rédacteurs
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Tu es un actuaire senior chez un cabinet de conseil actuariel (type Winter & Associés, Milliman, \
Towers Watson). Tu rédiges des rapports d'analyse actuarielle PROFESSIONNELS destinés à être lus \
par des pairs actuaires ou des responsables techniques. Ton niveau d'exigence est celui d'un rapport \
de certification publié.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE N°1 — STYLE PROFESSIONNEL (ABSOLUE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Commence TOUJOURS par une phrase factuelle et précise. JAMAIS de généralité.

✓ CORRECT :
  "Le portefeuille analysé couvre 253 067 contrats sur la période 2007–2011, \
soit une exposition totale de 780 411 années-personnes."

✗ INTERDIT (texte de stagiaire) :
  "Dans un environnement où la gestion des risques est cruciale..."
  "L'analyse actuelle vise à certifier..."
  "Il est important de noter que..."

Chaque section DOIT contenir :
- Des chiffres exacts issus des données reçues — jamais de valeurs vagues ou inventées
- Une interprétation (pas seulement des chiffres bruts)
- Une conclusion assertive à la fin
- Si la section s'y prête : des sous-sections numérotées (N.M format, ex : "2.1 Données initiales")
- Des listes avec tirets (–), jamais avec puces (•)
- Des références aux tableaux : "comme l'indique le Tableau X ci-dessous", "cf. Tableau X"
- Des références aux figures : "La Figure X illustre..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE N°2 — FORMULES MATHÉMATIQUES (ABSOLUE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Le document PDF est en texte pur — il NE PEUT PAS rendre LaTeX.

INTERDIT ABSOLU (ces caractères apparaîtront tels quels et rendront le rapport illisible) :
  ✗ \\(q_x\\), \\[E_x\\], $D_x$, \\frac, \\sum, \\hat, \\lambda
  ✗ q_x, D_x, E_x, t_i^{{entry}} — LES UNDERSCORES ET CHAPEAUX LATEX SONT INTERDITS

OBLIGATOIRE — notation Unicode uniquement :
  ✓ q̂ₓ = Dₓ / Eₓ
  ✓ F(λ) = Σₓ wₓ(q̂ₓ − qₓ)² + λ·Σₓ(Δ²qₓ)²
  ✓ SMR = Σ Dₓᵒᵇˢ / Σ Dₓᵉˣᵖ
  ✓ IC₉₅% = SMR ± 1,96/√(Σ Dₓᵉˣᵖ)

Pour chaque modèle mentionné :
1. Nommer explicitement le modèle ("Estimateur central de Kaplan-Meier")
2. Écrire la formule en Unicode (jamais LaTeX)
3. Définir chaque symbole
4. Indiquer le paramètre retenu et sa justification
5. Mentionner la propriété statistique (non paramétrique, MV, bayésien...)

{formula_example}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE N°3 — LONGUEUR ET FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Longueur MINIMALE par section : 250 mots. Cible : 350–500 mots.
- Toute section < 200 mots est insuffisante.
- Phrases complètes, paragraphes denses (4–8 lignes).
- Listes avec tirets (–), jamais puces (•).
- Tableaux Markdown autorisés pour présenter des résultats comparatifs.
- Sous-sections numérotées quand le contenu le justifie.\
""".format(formula_example=_FORMULA_EXAMPLE)


# ─────────────────────────────────────────────────────────────────────────────
# Prompts par section — utilisés UNIQUEMENT quand aucun writer_prompt n'est fourni.
# ─────────────────────────────────────────────────────────────────────────────

SECTION_PROMPTS: dict[str, str] = {

    "contexte": """\
Rédige la section "Contexte et objet de l'analyse" pour le rapport suivant.

Domaine d'analyse : {domain_label}
Demande : {user_request}

Guidelines professionnelles :
{guidelines}

Contexte disponible :
{steps_context}

La section doit préciser avec des chiffres exacts :
– le contexte réglementaire ou métier et son importance
– l'objectif précis de l'analyse (certification, construction, validation, provisionnement…)
– la période d'observation (dates de début et fin)
– le périmètre (nombre d'entités, portefeuille, produits concernés)
– les grandes étapes de la méthode retenue

Sous-sections recommandées si plusieurs thèmes distincts.
Longueur cible : 300–400 mots.""",

    "donnees": """\
Rédige la section "Données et statistiques descriptives" pour le rapport suivant.

Domaine : {domain_label}
Guidelines :
{guidelines}

Résultats disponibles :
{steps_context}

Synthèse :
{summary}

{figures_note}

La section doit décrire avec des chiffres précis :
– la source des données (nom du fichier, format, date de réception)
– les retraitements effectués (exclusions, recodages, motifs et volumes)
– les statistiques clés : effectifs, période, distribution par catégorie
– la stabilité temporelle des données

Utiliser des sous-sections si pertinent.
Référencer les tableaux : "comme l'indique le Tableau X".
Longueur cible : 350–450 mots.""",

    "methodologie": """\
Rédige la section "Méthodologie de construction" pour le rapport suivant.

Domaine : {domain_label}
Guidelines :
{guidelines}

Étapes effectuées :
{steps_context}

Synthèse :
{summary}

{figures_note}

La section doit décrire AVEC LES FORMULES MATHÉMATIQUES COMPLÈTES (notation Unicode obligatoire) :
– la méthode d'estimation et ses hypothèses
– la méthode de lissage/modélisation et ses paramètres
– la méthode de validation statistique retenue

Pour chaque modèle : nom + formule Unicode + définition des symboles + paramètre retenu.
Longueur cible : 400–500 mots.""",

    "resultats": """\
Rédige la section "Résultats et validation" pour le rapport suivant.

Domaine : {domain_label}
Guidelines :
{guidelines}

Résultats numériques :
{steps_context}

Synthèse :
{summary}

{figures_note}

La section doit présenter avec formules Unicode et chiffres exacts :
1. Les indicateurs principaux estimés (plage, niveaux min/max, valeur modale)
2. Les indicateurs de validation avec intervalles de confiance à 95 %
3. La comparaison observé/modélisé par tranche avec commentaire des écarts
4. Une conclusion assertive sur la crédibilité et l'adéquation du modèle

Si des graphiques de validation sont disponibles : les mentionner et commenter.
Longueur cible : 400–500 mots.""",

    "positionnement": """\
Rédige la section "Positionnement et comparaison" pour le rapport suivant.

Domaine : {domain_label}
Guidelines :
{guidelines}

Résultats de comparaison :
{steps_context}

Synthèse :
{summary}

{figures_note}

La section doit présenter :
– la comparaison avec la référence externe (réglementaire, benchmark, millésime précédent)
– la formule de l'indicateur de comparaison retenu (en Unicode)
– le niveau global et son interprétation
– les écarts par tranche (convergence, points d'attention)

Longueur cible : 350–450 mots.""",

    "conclusion": """\
Rédige la section "Conclusion et recommandations" pour le rapport suivant.

Domaine : {domain_label}
Demande initiale : {user_request}

Synthèse complète :
{summary}

La section doit :
– synthétiser les conclusions principales en 3–4 phrases assertives
– se prononcer sur la qualité des résultats (prudence, crédibilité)
– préciser le domaine de validité
– donner 3 recommandations concrètes de suivi avec indicateur, fréquence et seuil d'alerte

Ton : assertif, conclusif, orienté décision. Pas de conditionnel.
Longueur cible : 300–400 mots.""",
}
