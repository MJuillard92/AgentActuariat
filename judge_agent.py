"""
judge_agent.py
Évaluateur hybride (rubric automatisée + LLM-as-judge) pour la boucle encodeur-décodeur.

Évalue la qualité structurelle d'un rapport généré par rapport au template de référence,
sur 5 dimensions :
  - complétude   (25%) : sections, tableaux et graphiques attendus présents
  - methodologie (30%) : séquence correcte, appels de fonctions actuariels
  - metriques    (20%) : SMR, chi², IC, p-valeur présents dans le résumé
  - narrative    (15%) : qualité rédactionnelle et cohérence (LLM)
  - visuels      (10%) : graphiques effectivement produits

Score global = moyenne pondérée des 5 dimensions.

Usage :
    from judge_agent import evaluate_report_structure
    result = evaluate_report_structure(template, steps, summary)
    # result["score_global"] → float 0–1
    # result["ecarts"]       → list[str]
    # result["suggestions"]  → list[str]
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

# Mots-clés attendus dans les descriptions des steps pour chaque phase de l'analyse.
# Utilisés par la rubric automatisée pour vérifier la séquence.
_SEQUENCE_KEYWORDS = {
    "chargement":   ["load_data", "chargement", "charger", "données"],
    "nettoyage":    ["clean_data", "nettoyage", "nettoyer", "exclusion"],
    "exposition":   ["exposure", "exposition", "expositions", "compute_exposure"],
    "taux_bruts":   ["crude_rates", "taux bruts", "bruts"],
    "lissage":      ["smooth", "lissage", "whittaker", "gompertz", "makeham", "spline"],
    "validation":   ["chi", "smr", "validation", "intervalles", "confidence"],
    "benchmark":    ["benchmark", "référence", "abattement", "th0002", "tf0002"],
}

# Regex pour détecter les métriques clés dans le résumé final.
_METRICS_PATTERNS = {
    "smr":        re.compile(r"\bsmr\b|ratio\s+standardis", re.I),
    "chi2":       re.compile(r"chi[\s\-_²2]|khi[\s\-_²2]|\bχ²\b|\bchi_carre\b", re.I),
    "ic95":       re.compile(r"ic\s*9[05]|intervalle\s+de\s+confiance|confidence|ic_inf|ic_sup", re.I),
    "p_value":    re.compile(r"p[-_\s]?val|p-value|probabilit[ée]\s+cr[ée]dibilit[ée]", re.I),
    "rapport_oa": re.compile(r"rapport\s+o[\s/]a|o/a|observ[ée].+att[ée]ndu|d_obs|d_exp", re.I),
}

# Poids des dimensions dans le score global
_WEIGHTS = {
    "completude":   0.25,
    "methodologie": 0.30,
    "metriques":    0.20,
    "narrative":    0.15,
    "visuels":      0.10,
}

# ─────────────────────────────────────────────────────────────────────────────
# Rubric automatisée (60 % du score)
# ─────────────────────────────────────────────────────────────────────────────

def _score_completude(template: dict, steps: list[dict]) -> tuple[float, list[str]]:
    """Vérifie la présence de chaque tableau et graphique attendus.

    Pour les tableaux : cherche les colonnes attendues dans les display_outputs
    des steps (sorties textuelles — résultats de display()).
    Pour les graphiques : vérifie que les steps contiennent des figures.
    """
    tables = template.get("tables", [])
    figures = template.get("figures", [])
    ecarts: list[str] = []
    found = 0
    total = len(tables) + len(figures)

    if total == 0:
        return 1.0, []

    # Agréger toutes les sorties textuelles
    all_outputs = " ".join(
        (s.get("output", "") or "") + " " + " ".join(
            (d.get("content", "") or "") if isinstance(d, dict) else str(d)
            for d in (s.get("display_outputs") or [])
        )
        for s in steps
    ).lower()

    # Tableaux — chercher les colonnes dans les outputs
    for t in tables:
        cols = [c.lower() for c in t.get("columns", [])]
        if not cols:
            found += 1
            continue
        # Au moins la moitié des colonnes doivent être présentes
        n_found = sum(1 for c in cols if c in all_outputs)
        if n_found >= max(1, len(cols) // 2):
            found += 1
        else:
            missing = [c for c in cols if c not in all_outputs]
            ecarts.append(
                f"Tableau '{t.get('name', t.get('id', '?'))}' : "
                f"colonnes manquantes dans les sorties : {missing[:5]}"
            )

    # Figures — au moins un step avec des figures
    total_figs_produced = sum(len(s.get("figures", [])) for s in steps)
    expected_figs = len(figures)
    if expected_figs > 0:
        ratio = min(1.0, total_figs_produced / expected_figs)
        found += ratio * expected_figs
        if total_figs_produced < expected_figs:
            ecarts.append(
                f"Graphiques : {total_figs_produced} produits, {expected_figs} attendus "
                f"d'après le template."
            )

    score = found / total if total > 0 else 1.0
    return min(1.0, max(0.0, score)), ecarts


def _score_methodologie(steps: list[dict]) -> tuple[float, list[str]]:
    """Vérifie la présence des étapes clés dans la séquence de l'analyse."""
    all_text = " ".join(
        (s.get("description", "") or "") + " " + (s.get("output", "") or "")
        for s in steps
    ).lower()

    ecarts: list[str] = []
    found = 0
    for phase, kws in _SEQUENCE_KEYWORDS.items():
        if any(kw in all_text for kw in kws):
            found += 1
        else:
            ecarts.append(f"Étape '{phase}' absente ou non détectée dans les descriptions de steps.")

    score = found / len(_SEQUENCE_KEYWORDS)
    return score, ecarts


def _score_metriques(summary: str) -> tuple[float, list[str]]:
    """Vérifie la présence des métriques clés dans le résumé final."""
    ecarts: list[str] = []
    found = 0
    for name, pattern in _METRICS_PATTERNS.items():
        if pattern.search(summary):
            found += 1
        else:
            ecarts.append(f"Métrique '{name}' absente du résumé final.")

    score = found / len(_METRICS_PATTERNS)
    return score, ecarts


def _score_visuels(steps: list[dict]) -> tuple[float, list[str]]:
    """Vérifie que les étapes ont produit des figures."""
    total_figs = sum(len(s.get("figures", [])) for s in steps)
    ecarts: list[str] = []
    if total_figs == 0:
        ecarts.append("Aucune figure produite dans les steps.")
        return 0.0, ecarts
    if total_figs < 3:
        ecarts.append(f"Seulement {total_figs} figure(s) produite(s) — attendu ≥ 3.")
        return 0.5, ecarts
    return 1.0, []


# ─────────────────────────────────────────────────────────────────────────────
# LLM-as-judge (40 % du score, 2 appels gpt-4o-mini, moyenne)
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = """\
Tu es un évaluateur expert en qualité de rapports actuariels.
On te fournit :
  - Un résumé final généré par un agent actuariel.
  - La liste des sections attendues du rapport de référence.
  - Les écarts structurels déjà détectés par une rubric automatisée.

Évalue sur 2 dimensions (score 0.0 à 1.0 chacun) :
  1. narrative   : qualité rédactionnelle, ton professionnel, présence de formules
                   mathématiques définies, citations de valeurs numériques précises.
  2. coherence   : cohérence entre les étapes décrites et les conclusions du résumé,
                   adéquation avec les sections attendues.

Retourne UNIQUEMENT un JSON :
{"narrative": float, "coherence": float, "suggestions": [str, ...]}
(suggestions : 2-4 améliorations concrètes à apporter au prompt pour corriger les écarts)
"""


def _llm_judge(
    summary: str,
    expected_sections: list[str],
    ecarts: list[str],
) -> dict[str, Any]:
    """Lance 2 appels LLM-as-judge et retourne la moyenne des scores."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"narrative": 0.5, "coherence": 0.5, "suggestions": []}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        return {"narrative": 0.5, "coherence": 0.5, "suggestions": []}

    sections_str = "\n".join(f"- {s}" for s in expected_sections)
    ecarts_str = "\n".join(f"- {e}" for e in ecarts[:10]) if ecarts else "Aucun écart majeur."

    user_content = (
        f"RÉSUMÉ FINAL DE L'AGENT :\n{summary[:3000]}\n\n"
        f"SECTIONS ATTENDUES :\n{sections_str}\n\n"
        f"ÉCARTS STRUCTURELS DÉTECTÉS :\n{ecarts_str}"
    )

    scores_list: list[dict] = []
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=512,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = (resp.choices[0].message.content or "{}").strip()
            scores_list.append(json.loads(raw))
        except Exception:
            scores_list.append({"narrative": 0.5, "coherence": 0.5, "suggestions": []})

    # Moyenne des deux appels
    narrative = sum(s.get("narrative", 0.5) for s in scores_list) / len(scores_list)
    coherence = sum(s.get("coherence", 0.5) for s in scores_list) / len(scores_list)
    # Fusionner les suggestions (déduplication)
    all_suggestions: list[str] = []
    seen: set[str] = set()
    for s in scores_list:
        for sug in s.get("suggestions", []):
            key = sug[:60].lower()
            if key not in seen:
                seen.add(key)
                all_suggestions.append(sug)

    return {
        "narrative": round(narrative, 3),
        "coherence": round(coherence, 3),
        "suggestions": all_suggestions[:4],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_report_structure(
    template: dict,
    steps: list[dict],
    summary: str,
) -> dict:
    """Évalue la qualité structurelle d'un rapport généré par l'agent actuariel.

    Rubric hybride :
      60 % automatisée (déterministe) :
        - complétude   (25%) — tableaux et graphiques attendus présents
        - methodologie (30%) — séquence des étapes clés
        - metriques    (20%) — métriques clés dans le résumé
        - visuels      (10%) — figures produites
      40 % LLM-as-judge (gpt-4o-mini, 2 appels, moyenne) :
        - narrative + cohérence → remplace la dimension "narrative" du plan

    Args:
        template : dict issu de analyze_report_pdf (sections, tables, figures, ...)
        steps    : liste des events "step" retournés par run_agent_loop
                   (champs: description, output, figures, display_outputs)
        summary  : synthèse finale de l'agent (string markdown)

    Returns:
        {
          "score_global": float (0–1),
          "scores": {
            "completude":   float,
            "methodologie": float,
            "metriques":    float,
            "narrative":    float,  # LLM
            "visuels":      float,
          },
          "ecarts":      [str, ...],
          "suggestions": [str, ...],
          "verdict":     str,
        }
    """
    ecarts_all: list[str] = []

    # ── Rubric automatisée ────────────────────────────────────────────────
    sc_completude,   e1 = _score_completude(template, steps)
    sc_methodologie, e2 = _score_methodologie(steps)
    sc_metriques,    e3 = _score_metriques(summary)
    sc_visuels,      e4 = _score_visuels(steps)
    ecarts_all.extend(e1 + e2 + e3 + e4)

    # ── LLM-as-judge ──────────────────────────────────────────────────────
    expected_sections = [s.get("title", "") for s in template.get("sections", [])]
    llm_result = _llm_judge(summary, expected_sections, ecarts_all)
    sc_narrative = (llm_result["narrative"] + llm_result["coherence"]) / 2

    scores = {
        "completude":   round(sc_completude,   3),
        "methodologie": round(sc_methodologie, 3),
        "metriques":    round(sc_metriques,    3),
        "narrative":    round(sc_narrative,    3),
        "visuels":      round(sc_visuels,      3),
    }

    score_global = sum(scores[k] * _WEIGHTS[k] for k in _WEIGHTS)
    score_global = round(min(1.0, max(0.0, score_global)), 3)

    if score_global >= 0.85:
        verdict = "Excellent — rapport conforme au template de référence."
    elif score_global >= 0.70:
        verdict = "Satisfaisant — quelques écarts mineurs à corriger."
    elif score_global >= 0.50:
        verdict = "Insuffisant — écarts significatifs, prompt à améliorer."
    else:
        verdict = "Très insuffisant — revoir la section MISSION entièrement."

    return {
        "score_global": score_global,
        "scores":       scores,
        "ecarts":       ecarts_all,
        "suggestions":  llm_result["suggestions"],
        "verdict":      verdict,
    }
