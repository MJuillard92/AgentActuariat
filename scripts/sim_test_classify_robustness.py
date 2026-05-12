"""Mesure empirique de la robustesse de _classify_intent.

On lance la classification N fois sur la même phrase pour évaluer la
variance du LLM. On compare aussi gpt-5.4-mini vs gpt-5.4 vs gpt-4o-mini."""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

for line in (_PROJECT_ROOT / ".env").read_text().splitlines():
    if line.startswith("OPENAI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')


def _classify_with_model(prompt: str, model: str) -> dict:
    """Réimplémente _classify_intent en variant le modèle, sans filet."""
    import openai, json
    from agents.mortality.agents._utils import call_with_retry

    client = openai.OpenAI()
    full_prompt = (
        "Tu es un routeur pour un système actuariel. Classifie la demande en 3 axes :\n\n"
        "Axe 1 — kind :\n"
        "  - task      : calculs / rapport\n"
        "  - question  : explication, conversation hors calculs et hors rapport\n\n"
        "Axe 2 — write (uniquement si kind=task) — RÈGLE STRICTE :\n"
        "  - yes : le mot 'rapport', 'PDF' ou 'document' apparaît explicitement dans la demande\n"
        "          (ex: 'fais-moi le rapport', 'génère un rapport', 'je veux un PDF')\n"
        "  - no  : refus explicite du rapport ('sans rapport', 'pas de PDF', 'juste les calculs')\n"
        "  - ask : AUCUN mot-clé explicite (défaut).\n"
        "Axe 3 — report_mode (uniquement si kind=task) :\n"
        "  - full_report : pipeline complet avec lissage (défaut)\n"
        "  - raw_rates   : 'taux bruts', 'sans lissage', 'brut'\n"
        "  - description : 'description', 'analyse descriptive', 'résumé du portefeuille'\n\n"
        f"Contexte : Fichier CSV chargé : oui. Calculs complets : non.\n"
        f"Demande : {prompt}\n\n"
        'Réponds UNIQUEMENT en JSON :\n'
        '{"kind": "...", "write": "...", "report_mode": "...", "reply": "..."}'
    )
    resp = call_with_retry(
        client, model=model,
        messages=[{"role": "user", "content": full_prompt}],
        response_format={"type": "json_object"},
        max_tokens=200, temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content or "{}")


def stress(prompt: str, model: str, n: int = 5):
    """Lance N classifications, compte les write et report_mode obtenus."""
    writes, modes, kinds, replies = [], [], [], []
    for _ in range(n):
        r = _classify_with_model(prompt, model)
        writes.append(r.get("write"))
        modes.append(r.get("report_mode"))
        kinds.append(r.get("kind"))
        replies.append(r.get("reply", "")[:80])
    return {
        "writes":  Counter(writes),
        "modes":   Counter(modes),
        "kinds":   Counter(kinds),
        "replies": replies,
    }


def main() -> int:
    cases = [
        "construit un rapport d'analyse descriptive unisex",
        "fais-moi le rapport descriptif de mon portefeuille",
        "construis une table de mortalité",
        "fais-moi une analyse descriptive",
        "rédige un rapport avec les taux bruts",
        "calcule sans rapport",
    ]
    models = ["gpt-5.4-mini", "gpt-5.4"]
    n_runs = 5

    for prompt in cases:
        print(f"\n{'═' * 78}")
        print(f"PHRASE : {prompt!r}")
        print('═' * 78)
        for model in models:
            result = stress(prompt, model, n_runs)
            print(f"\n  Modèle : {model}  ({n_runs} runs, temp=0)")
            print(f"    kind   : {dict(result['kinds'])}")
            print(f"    write  : {dict(result['writes'])}")
            print(f"    mode   : {dict(result['modes'])}")
            print(f"    replies (3 premiers) :")
            for r in result["replies"][:3]:
                print(f"       • {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
