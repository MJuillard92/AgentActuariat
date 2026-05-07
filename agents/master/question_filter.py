"""
question_filter.py — Résolution en 3 niveaux des questions du Builder.

Quand le Builder émet un AIMessage avec un marqueur
`additional_kwargs["need_user_input"]`, ce module :
  Niveau 1 : lookup déterministe dans data_store + study_plan (Python pur).
  Niveau 2 : inférence LLM mini sur l'historique conversationnel.
  Niveau 3 : retour "forward" — le Master doit poser la question à l'user.

API publique :
    QuestionResolution                                    — dataclass de retour
    resolve_builder_question(need, data_store, user_msgs) — orchestrateur 3-niveaux
    detect_need_in_message(msg) -> dict | None             — parse l'AIMessage Builder
    extract_user_answer(response_text, need) -> Any        — mini-call extraction
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class QuestionResolution:
    """Résultat de la résolution d'une question Builder."""
    decision:   Literal["answered", "forward", "use_default"]
    value:      Any
    source:     str          # "study_plan" | "data_store" | "llm_inference" | "user_response" | "default"
    confidence: float
    reasoning:  str = ""


# ──────────────────────────────────────────────────────────────────────────
# Niveau 1 — Lookup déterministe Python
# ──────────────────────────────────────────────────────────────────────────

def _try_resolve_from_data_store(
    need:       dict,
    data_store: dict,
) -> tuple[Any, str | None]:
    """Niveau 1 — lookup déterministe Python pur. Aucun appel LLM.

    Cherche la `context_key` :
      1. Dans data_store["study_plan"] (paramètres confirmés par le user)
      2. Au top-level de data_store (clés produites par les tools)

    Retourne (valeur, source) ou (None, None) si non trouvé.
    """
    key = need.get("context_key")
    if not key:
        return None, None
    sp = data_store.get("study_plan") or {}
    if key in sp and sp[key] is not None:
        return sp[key], "study_plan"
    if key in data_store and data_store[key] is not None:
        return data_store[key], "data_store"
    return None, None


# ──────────────────────────────────────────────────────────────────────────
# Détection du marqueur dans un AIMessage Builder
# ──────────────────────────────────────────────────────────────────────────

def detect_need_in_message(msg) -> dict | None:
    """Retourne le dict need_user_input si présent dans additional_kwargs.

    Ne s'applique qu'aux AIMessage. Pour les autres types (HumanMessage,
    ToolMessage), retourne None.
    """
    from langchain_core.messages import AIMessage
    if not isinstance(msg, AIMessage):
        return None
    kwargs = getattr(msg, "additional_kwargs", None) or {}
    need = kwargs.get("need_user_input")
    return need if isinstance(need, dict) else None


# ──────────────────────────────────────────────────────────────────────────
# Niveau 2 — LLM mini inference
# ──────────────────────────────────────────────────────────────────────────

def _call_mini_for_inference(prompt: str) -> dict:
    """Appelle gpt-5.4-mini en mode JSON. Retourne un dict {answered, value,
    confidence, reasoning} ou {} en cas d'erreur."""
    import openai
    from agents.mortality.agents._utils import call_with_retry
    from agents.mortality.agents.llm_config import get_llm_config

    cfg = get_llm_config("master.classify_intent")  # même profil mini
    try:
        client = openai.OpenAI()
        resp = call_with_retry(
            client,
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=200,
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return {}


def _llm_infer_from_history(need: dict, user_messages: list[str]) -> dict:
    """Niveau 2 — interroge gpt-5.4-mini pour savoir si la question Builder
    a été répondue implicitement par l'utilisateur dans son historique.

    Retourne un dict {answered: bool, value: Any, confidence: float, reasoning: str}.
    Si user_messages est vide, retourne immédiatement {answered: False} sans
    appel LLM (évite un round-trip inutile).
    """
    if not user_messages:
        return {"answered": False, "value": None, "confidence": 0.0, "reasoning": "no user messages"}

    options_str = need.get("options")
    options_hint = f"Options possibles : {options_str}\n" if options_str else ""
    history = "\n".join(f"- {m}" for m in user_messages[-10:])

    prompt = (
        "Tu analyses si l'utilisateur a déjà répondu, même implicitement, "
        "à une question technique posée par un agent actuariel.\n\n"
        f"Question posée par l'agent : {need.get('question', '?')}\n"
        f"{options_hint}"
        f"Messages utilisateur récents :\n{history}\n\n"
        "Si la réponse est claire (mots-clés, formulation explicite) → "
        "answered=true, value=la valeur déduite, confidence=0.7-1.0.\n"
        "Si ambigu → answered=false, confidence < 0.7.\n\n"
        "Réponds UNIQUEMENT en JSON :\n"
        '{"answered": true|false, "value": <valeur ou null>, '
        '"confidence": 0.0-1.0, "reasoning": "courte explication"}'
    )
    result = _call_mini_for_inference(prompt)
    return {
        "answered":   bool(result.get("answered", False)),
        "value":      result.get("value"),
        "confidence": float(result.get("confidence", 0.0)),
        "reasoning":  str(result.get("reasoning", "")),
    }


# ──────────────────────────────────────────────────────────────────────────
# Orchestrateur 3-niveaux
# ──────────────────────────────────────────────────────────────────────────

def resolve_builder_question(
    need:                 dict,
    data_store:           dict,
    user_messages:        list[str],
    confidence_threshold: float = 0.7,
) -> QuestionResolution:
    """Orchestrateur 3-niveaux : Python lookup → LLM mini → forward."""
    # Niveau 1
    val, source = _try_resolve_from_data_store(need, data_store)
    if val is not None:
        return QuestionResolution(
            decision="answered", value=val, source=source, confidence=1.0,
            reasoning="found in data_store",
        )

    # Niveau 2
    inf = _llm_infer_from_history(need, user_messages)
    if inf.get("answered") and inf.get("confidence", 0.0) >= confidence_threshold:
        return QuestionResolution(
            decision="answered",
            value=inf["value"],
            source="llm_inference",
            confidence=inf["confidence"],
            reasoning=inf.get("reasoning", ""),
        )

    # Niveau 3 — forward au user
    return QuestionResolution(
        decision="forward", value=None, source="user", confidence=0.0,
        reasoning="no signal in history",
    )


# ──────────────────────────────────────────────────────────────────────────
# Extraction structurée d'une réponse libre user
# ──────────────────────────────────────────────────────────────────────────

def extract_user_answer(response_text: str, need: dict) -> Any:
    """Extrait la valeur structurée d'une réponse libre de l'utilisateur."""
    if not response_text or not response_text.strip():
        return None

    options = need.get("options")
    if not options:
        return response_text.strip()

    prompt = (
        "Tu mappes la réponse d'un utilisateur vers l'une des options proposées.\n"
        f"Question initiale : {need.get('question', '?')}\n"
        f"Options : {options}\n"
        f"Réponse utilisateur : {response_text}\n\n"
        "Réponds UNIQUEMENT en JSON :\n"
        '{"answered": true|false, "value": <option choisie ou null>, '
        '"confidence": 0.0-1.0}'
    )
    result = _call_mini_for_inference(prompt)
    if result.get("answered") and result.get("confidence", 0) >= 0.6:
        return result.get("value")
    return None
