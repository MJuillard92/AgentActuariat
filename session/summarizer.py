"""
session/summarizer.py
Compaction structurée de l'historique de conversation.

Déclencheur : len(messages) > COMPACT_THRESHOLD
Résultat    : ContextSummary JSON + les N derniers messages verbatim

Le résumé est produit par GPT-4o en mode JSON.
Il remplace l'historique ancien dans le system prompt, jamais dans les messages.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from session.session_state import ContextSummary

if TYPE_CHECKING:
    from langchain_core.messages import AnyMessage

log = logging.getLogger(__name__)

COMPACT_THRESHOLD  = 15   # déclenche la compaction au-delà de ce nombre
KEEP_VERBATIM      = 5    # messages récents conservés en verbatim après compaction


class Summarizer:
    """
    Produit un ContextSummary depuis un historique de messages LangChain.
    """

    def should_compact(self, messages: list) -> bool:
        return len(messages) > COMPACT_THRESHOLD

    def compact(self, messages: list, data_store: dict | None = None) -> ContextSummary:
        """
        Résume les messages anciens en un ContextSummary structuré.

        Args:
            messages   : liste de AnyMessage LangChain
            data_store : data_store courant pour enrichir le contexte

        Returns:
            ContextSummary rempli
        """
        import datetime
        import openai
        from agents.mortality.agents._utils import call_with_retry

        # Construire le texte de conversation à résumer
        lines = []
        for m in messages[:-KEEP_VERBATIM]:  # tous sauf les N derniers
            role    = getattr(m, "type", "")
            content = getattr(m, "content", "") or ""
            if role == "human":
                lines.append(f"USER: {str(content)[:400]}")
            elif role == "ai":
                lines.append(f"AGENT: {str(content)[:400]}")

        if not lines:
            return ContextSummary()

        # Contexte data_store
        ds = data_store or {}
        computed = [k for k in ("exposure_table", "qx_table", "smoothed_table",
                                "validation", "benchmarking") if ds.get(k)]

        prompt = (
            "Tu analyses une conversation entre un utilisateur et un agent actuariel.\n"
            "Produis un JSON structuré résumant l'état de la session.\n\n"
            "Format JSON attendu (toutes les listes peuvent être vides []) :\n"
            "{\n"
            '  "decisions_prises": ["..."],\n'
            '  "ambiguites_levees": ["..."],\n'
            '  "hypotheses_actives": ["..."],\n'
            '  "objets_construits": ["..."],\n'
            '  "donnees_manquantes": ["..."],\n'
            '  "prochaine_etape": "..."\n'
            "}\n\n"
            f"Calculs déjà effectués (data_store) : {computed or 'aucun'}\n\n"
            "Conversation à résumer :\n"
            + "\n".join(lines)
            + "\n\nJSON uniquement :"
        )

        try:
            client = openai.OpenAI()
            response = call_with_retry(
                client,
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=600,
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            summary = ContextSummary(
                decisions_prises   = data.get("decisions_prises", []),
                ambiguites_levees  = data.get("ambiguites_levees", []),
                hypotheses_actives = data.get("hypotheses_actives", []),
                objets_construits  = data.get("objets_construits", []) or computed,
                donnees_manquantes = data.get("donnees_manquantes", []),
                prochaine_etape    = data.get("prochaine_etape", ""),
                compacted_at       = datetime.datetime.now().isoformat(),
                messages_since     = KEEP_VERBATIM,
            )
            log.info("[Summarizer] compaction OK — %d décisions, prochaine étape: %s",
                     len(summary.decisions_prises), summary.prochaine_etape[:80])
            return summary

        except Exception as exc:
            log.warning("[Summarizer] échec compaction GPT-4o : %s — résumé vide", exc)
            # Fallback minimal : lister les objets construits depuis data_store
            return ContextSummary(
                objets_construits = computed,
                compacted_at      = datetime.datetime.now().isoformat(),
                messages_since    = KEEP_VERBATIM,
            )

    def trim_messages(self, messages: list) -> list:
        """
        Retourne les N derniers messages verbatim après compaction.
        À utiliser conjointement avec compact().
        """
        return messages[-KEEP_VERBATIM:] if len(messages) > KEEP_VERBATIM else messages
