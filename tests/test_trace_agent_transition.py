"""
HOTFIX-pre-refacto-2026-05 — Bug 18 (A3) : trigger générique
'agent_transition' rendu `📍 [Switch model] from → to — reason`.

Les wrappers de graph.py émettent cet event et tracent l'agent de départ
via data_store["_last_agent"].
"""
from __future__ import annotations

from agents.mortality.agents.graph import _emit_transition


def test_transition_emitted_when_agent_changes() -> None:
    """from != to → event agent_transition émis avec from/to/reason."""
    state = {"data_store": {"_last_agent": "master"}}
    result: dict = {"events": []}
    out = _emit_transition(state, result, "builder")

    trans = [e for e in out["events"] if e.get("type") == "agent_transition"]
    assert len(trans) == 1
    assert trans[0]["from"] == "MasterAgent"
    assert trans[0]["to"] == "BuilderAgent"
    assert trans[0]["reason"] == "calculs à exécuter"


def test_no_transition_on_first_node_of_turn() -> None:
    """Aucun _last_agent (1er nœud du tour) → pas d'event transition."""
    state = {"data_store": {}}
    out = _emit_transition(state, {"events": []}, "master")
    trans = [e for e in out["events"] if e.get("type") == "agent_transition"]
    assert len(trans) == 0


def test_no_transition_when_same_agent() -> None:
    """from == to (ex. builder → tools → builder) → pas de transition."""
    state = {"data_store": {"_last_agent": "builder"}}
    out = _emit_transition(state, {"events": []}, "builder")
    trans = [e for e in out["events"] if e.get("type") == "agent_transition"]
    assert len(trans) == 0


def test_last_agent_persisted_for_next_wrapper() -> None:
    """_emit_transition met à jour data_store['_last_agent']."""
    state = {"data_store": {"_last_agent": "master"}}
    out = _emit_transition(state, {"events": []}, "writer")
    assert out["data_store"]["_last_agent"] == "writer"


def test_transition_reason_per_target() -> None:
    """Chaque agent cible a sa raison."""
    for to_key, expected in [
        ("builder", "calculs à exécuter"),
        ("writer",  "rédaction du rapport PDF"),
        ("rag",     "question doctrinale"),
        ("master",  "retour au superviseur"),
    ]:
        state = {"data_store": {"_last_agent": "master" if to_key != "master" else "builder"}}
        out = _emit_transition(state, {"events": []}, to_key)
        trans = [e for e in out["events"] if e.get("type") == "agent_transition"]
        assert trans and trans[0]["reason"] == expected


def test_data_store_preserved_when_node_returned_none() -> None:
    """Si le result n'a pas de data_store, il est créé depuis l'entrée."""
    state = {"data_store": {"_last_agent": "master", "study_plan": {"x": 1}}}
    out = _emit_transition(state, {"events": []}, "builder")
    assert out["data_store"]["study_plan"] == {"x": 1}
    assert out["data_store"]["_last_agent"] == "builder"
