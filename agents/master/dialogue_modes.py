"""
dialogue_modes.py — helpers pour le dialogue 3-modes Master ↔ user (US-16).

Trois modes d'interaction proposés en début de chaque phase
(data puis modeling) :

    (a) AUTONOMOUS            — Master décide, pas de validation user.
    (b) USER_FIRST            — user fournit les valeurs avant les tools.
    (c) PROPOSITION_VALIDATION — Master propose, user valide en bloc (défaut).

Le mode choisi est journalisé dans data_store["_session"]["mode_audit"]
pour la traçabilité du rapport.
"""
from __future__ import annotations

import enum
from typing import Any


class DialogueMode(str, enum.Enum):
    AUTONOMOUS = "autonomous"
    USER_FIRST = "user_first"
    PROPOSITION_VALIDATION = "proposition_validation"

    @classmethod
    @property
    def DEFAULT(cls) -> "DialogueMode":  # type: ignore[override]
        return cls.PROPOSITION_VALIDATION


_SHORTCUTS = {
    "a": DialogueMode.AUTONOMOUS,
    "b": DialogueMode.USER_FIRST,
    "c": DialogueMode.PROPOSITION_VALIDATION,
}


def choose_mode(raw: str) -> DialogueMode:
    """Parse une saisie utilisateur (a/b/c ou nom complet) en DialogueMode."""
    key = (raw or "").strip().lower()
    if key in _SHORTCUTS:
        return _SHORTCUTS[key]
    try:
        return DialogueMode(key)
    except ValueError as exc:
        raise ValueError(
            f"mode inconnu : {raw!r}. Attendu : a/b/c ou "
            f"{[m.value for m in DialogueMode]}"
        ) from exc


def record_mode(data_store: dict[str, Any], *, phase: str, mode: DialogueMode) -> None:
    """Trace le mode retenu pour une phase dans _session.mode_audit + mode_<phase>."""
    session = data_store.setdefault("_session", {})
    session[f"mode_{phase}"] = mode.value
    audit = session.setdefault("mode_audit", [])
    audit.append({"phase": phase, "mode": mode.value})


def get_mode(data_store: dict[str, Any], *, phase: str) -> DialogueMode:
    """Retourne le mode retenu pour une phase, ou DEFAULT si absent."""
    session = data_store.get("_session") or {}
    raw = session.get(f"mode_{phase}")
    if raw is None:
        return DialogueMode.DEFAULT
    return DialogueMode(raw)
