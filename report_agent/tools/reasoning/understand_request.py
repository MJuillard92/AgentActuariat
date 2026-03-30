"""
report_agent/tools/reasoning/understand_request.py
Compréhension de la demande métier — appelé en tout premier par le WriterAgent.

════════════════════════════════════════════════════════════════
INPUTS
════════════════════════════════════════════════════════════════
context dict :
    user_message  : str        — dernier message de l'utilisateur
    history       : list[dict] — historique [{role, content}]
    csv_columns   : list[str]  — colonnes détectées dans le CSV (optionnel)

params dict : {} (aucun paramètre requis)

════════════════════════════════════════════════════════════════
OUTPUT  (dict)
════════════════════════════════════════════════════════════════
    intent           : str    — "analyse_descriptive" | "table_mortalite"
                                | "rapport_pdf" | "graphique" | "autre"
    entities         : dict   — {portefeuille, sexe, periode, produit, ...}
                                éléments métier identifiés dans le message
    clarifications   : list[str] — questions à poser si intent ambigu
    confidence       : float  — 0.0 à 1.0 (certitude de l'intent)
    reasoning        : str    — explication courte (1–2 phrases)
════════════════════════════════════════════════════════════════

Interface : run(context, params) -> dict
"""
from __future__ import annotations

import json
import openai


_SYSTEM = """Tu es un assistant actuariel. Ton rôle est d'analyser une demande métier
et de la classifier avant tout traitement.

Réponds uniquement en JSON valide avec les clés :
  intent         : string parmi ["analyse_descriptive", "table_mortalite", "rapport_pdf", "graphique", "autre"]
  entities       : dict  — éléments métier mentionnés (sexe, période, produit, portefeuille…)
  clarifications : list  — questions à poser si la demande est ambiguë (max 3)
  confidence     : float — entre 0.0 et 1.0
  reasoning      : string — explication courte (1-2 phrases)

Définitions des intents :
  analyse_descriptive : résumé, statistiques descriptives, pyramide des âges, séries temporelles
  table_mortalite     : construction d'une table d'expérience (exposition, taux bruts, lissage, SMR)
  rapport_pdf         : génération d'un document PDF (rapport descriptif ou de certification)
  graphique           : demande explicite d'un graphique ou visualisation
  autre               : hors périmètre (IBNR, tarification, etc.)
"""


def run(context: dict | None, params: dict | None = None) -> dict:
    context = context or {}
    user_message = context.get("user_message", "")
    history = context.get("history", [])
    csv_columns = context.get("csv_columns", [])

    if not user_message:
        return {
            "intent": "autre",
            "entities": {},
            "clarifications": ["Quelle est votre demande ?"],
            "confidence": 0.0,
            "reasoning": "Message utilisateur vide.",
        }

    # Construire le contexte pour le LLM
    context_lines = [f"Colonnes CSV disponibles : {csv_columns}"] if csv_columns else []
    if history:
        recent = history[-4:]  # 2 derniers échanges
        context_lines.append("Historique récent :")
        for h in recent:
            context_lines.append(f"  {h.get('role', 'user')}: {str(h.get('content', ''))[:200]}")

    user_prompt = "\n".join(context_lines + [f"\nDemande : {user_message}"])

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        # S'assurer que toutes les clés sont présentes
        result.setdefault("intent", "autre")
        result.setdefault("entities", {})
        result.setdefault("clarifications", [])
        result.setdefault("confidence", 0.5)
        result.setdefault("reasoning", "")
        return result
    except Exception as exc:
        return {
            "intent": "autre",
            "entities": {},
            "clarifications": [],
            "confidence": 0.0,
            "reasoning": f"Erreur lors de l'analyse : {exc}",
        }
