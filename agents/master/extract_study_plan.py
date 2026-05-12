"""
agents/master/extract_study_plan.py
Extraction des paramètres d'étude actuarielle depuis l'historique
conversationnel (LLM mini, JSON mode).

Utilisé par le Master pour pré-remplir le study_plan à partir des
20 derniers messages de la conversation, sans devoir poser un formulaire
explicite à l'utilisateur s'il a déjà mentionné les dates / le produit /
la table de référence / etc.
"""
from __future__ import annotations

import json


def extract_study_plan_from_history(messages: list[dict]) -> dict:
    """Extrait les paramètres d'étude depuis la conversation.

    Args:
        messages: liste de dicts {"role": "user"|"assistant", "content": str}.
                  Seuls les 20 derniers messages sont envoyés au LLM.

    Returns:
        dict des paramètres détectés (clés présentes uniquement) :
          observation_start_date, observation_end_date,
          observation_period_years, study_objective, product_list,
          smoothing_algorithm, baseline_regulatory_table,
          cohort_min_age, cohort_max_age,
          confidence_interval_level, chi_squared_p_significance
        Retourne {} si rien n'est extractible ou si l'appel échoue.
    """
    import openai
    from agents.mortality.agents._utils import call_with_retry
    from agents.mortality.agents.llm_config import get_llm_config

    conv_lines: list[str] = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            conv_lines.append(f"{role.upper()}: {str(content)[:300]}")
    if not conv_lines:
        return {}

    conversation_text = "\n".join(conv_lines[-20:])

    prompt = (
        "Extrait les paramètres d'étude actuarielle mentionnés dans cette conversation.\n"
        "Retourne UNIQUEMENT un JSON avec les clés présentes (ignore les absentes).\n\n"
        "Clés possibles :\n"
        "  observation_start_date (YYYY-MM-DD)\n"
        "  observation_end_date (YYYY-MM-DD)\n"
        "  observation_period_years (liste d'années ex: [2019,2020,2021])\n"
        "  study_objective (ex: 'prévoyance collective décès')\n"
        "  product_list (liste de codes produits)\n"
        "  smoothing_algorithm (ex: 'whittaker_henderson')\n"
        "  baseline_regulatory_table (ex: 'TH0002')\n"
        "  cohort_min_age (entier)\n"
        "  cohort_max_age (entier)\n"
        "  confidence_interval_level (ex: 0.95)\n"
        "  chi_squared_p_significance (ex: 0.05)\n\n"
        f"Conversation :\n{conversation_text}\n\n"
        "JSON uniquement, sans markdown :"
    )

    try:
        cfg = get_llm_config("master.extract_study_plan")
        client = openai.OpenAI()
        response = call_with_retry(
            client,
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=cfg.get("max_tokens", 400),
            temperature=cfg.get("temperature", 0.0),
        )
        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)
    except Exception:
        return {}
