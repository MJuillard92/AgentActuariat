"""
TOOL CONTRACT — reasoning.understand_request
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : reasoning.understand_request
domain        : descriptive
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Analyse et classifie la demande métier de l'utilisateur via un appel
à l'API OpenAI (GPT-4o-mini). Identifie l'intent (analyse descriptive,
construction de table, rapport PDF, graphique, autre), extrait les entités
métier, et génère des questions de clarification si nécessaire.

WHEN TO USE
-----------
Appeler optionnellement en début de session pour classifier une demande
ambiguë ou complexe. Utile si l'intent n'est pas évident depuis le message
de l'utilisateur.

WHEN NOT TO USE
---------------
Ne pas appeler si l'intent est clair depuis le message (ex: "construis ma
table de mortalité" → intent évident). Ne pas appeler à chaque tour de
conversation — uniquement au premier message ambigu.

PREREQUISITES
-------------
required_tools: []
required_data_store_keys: []
Note: reçoit context dict directement (pas df, pas data_store).

INPUTS
------
params: {}
context:
  user_message:
    type    : string
    note    : Dernier message de l'utilisateur.
  history:
    type    : list[dict]
    note    : Historique [{role, content}]. Optionnel — les 4 derniers messages
              sont utilisés pour le contexte.
  csv_columns:
    type    : list[string]
    note    : Colonnes détectées dans le CSV. Aide à classifier l'intent.

OUTPUTS
-------
data_store_keys_written: []
return_payload:
  intent         : str — analyse_descriptive | table_mortalite | rapport_pdf | graphique | autre
  entities       : dict — {portefeuille, sexe, periode, produit, ...}
  clarifications : list[str] — questions à poser (max 3)
  confidence     : float — 0.0 à 1.0
  reasoning      : str — explication courte

QUALITY GATES
-------------
BLOCKING: []
NON-BLOCKING:
  - confidence < 0.5 → poser les clarifications retournées au client avant
    de lancer un pipeline.
  - intent = "autre" → informer le client que la demande est hors périmètre.

ERROR HANDLING
--------------
error: "Erreur lors de l'analyse : ..."
  → cause  : Erreur API OpenAI ou message vide.
  → action : Ne pas relancer. Procéder avec l'intent par défaut ("autre")
             et demander une clarification directe au client.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Utiliser uniquement pour les demandes vraiment ambiguës. Si confidence > 0.7,
  procéder directement avec l'intent sans demander de confirmation.
  Si clarifications est non vide, poser la première question uniquement
  (ne pas surcharger le client de questions).
  Note : utilise OpenAI API — nécessite une clé API valide configurée.
exemplar_query: >
  Comment classifier la demande "fais-moi une étude complète du portefeuille" ?

CATALOGUE METADATA
------------------
display_name      : Compréhension de la demande
short_description : Classifie l'intent de la demande client via OpenAI et extrait les entités métier.
domain            : descriptive
capability_group  : descriptive
depends_on        : []
required_by       : []
client_visible    : false
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
        from agents.mortality.agents._utils import call_with_retry
        client = openai.OpenAI()
        response = call_with_retry(
            client,
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
