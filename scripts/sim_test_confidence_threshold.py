"""Mesure le score de confidence sur des phrases plus ou moins claires.

But : valider que :
  - phrases claires → confidence ≥ 0.80 → exécution directe
  - phrases ambiguës → confidence < 0.80 → reformulation demandée
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

for line in (_PROJECT_ROOT / ".env").read_text().splitlines():
    if line.startswith("OPENAI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')


def main() -> int:
    from agents.master.classify_intent import (
        classify_intent, is_confident, confidence_threshold,
    )

    threshold = confidence_threshold()
    print(f"Seuil de confiance configuré : {threshold:.2f}\n")
    print(f"{'─' * 90}")
    print(f"{'PHRASE':60} {'CONFIDENCE':>10} {'write':>7} {'mode':>14} {'≥thresh':>8}")
    print(f"{'─' * 90}")

    cases = [
        # Phrases CLAIRES (attendu : confidence ≥ 0.80)
        ("construit un rapport d'analyse descriptive unisex",       "CLEAR"),
        ("fais-moi le rapport descriptif H/F",                       "CLEAR"),
        ("calcule sans rapport",                                     "CLEAR"),
        ("rédige un rapport avec les taux bruts",                    "CLEAR"),
        ("c'est quoi le lissage Whittaker ?",                        "CLEAR"),

        # Phrases AMBIGUËS (attendu : confidence < 0.80)
        ("vas-y",                                                    "AMBIG"),
        ("ok",                                                       "AMBIG"),
        ("fais-le",                                                  "AMBIG"),
        ("quelque chose",                                            "AMBIG"),
        ("démarre",                                                  "AMBIG"),

        # Cas intermédiaires
        ("construis une table de mortalité",                         "MEDIUM"),
        ("fais une analyse",                                         "MEDIUM"),
        ("traite mon portefeuille",                                  "MEDIUM"),
    ]

    n_ok = 0
    for phrase, expected in cases:
        result = classify_intent(phrase, has_data=True, has_calcs=False)
        conf = result.get("confidence", 0.0)
        confident = is_confident(result, threshold)
        flag = "✓" if confident else "demander reformulation"
        # Validation
        if expected == "CLEAR" and confident:
            n_ok += 1
        elif expected == "AMBIG" and not confident:
            n_ok += 1
        elif expected == "MEDIUM":
            n_ok += 1   # médian, on accepte les deux
        mark = "✓" if (expected == "CLEAR" and confident) or \
                      (expected == "AMBIG" and not confident) or \
                      (expected == "MEDIUM") else "✗"
        print(f"{mark} {phrase:58} {conf:>10.2f} {result['write']:>7} {result['report_mode']:>14} {flag}")

    print(f"{'─' * 90}")
    print(f"\n{n_ok}/{len(cases)} cases comportent comme attendu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
