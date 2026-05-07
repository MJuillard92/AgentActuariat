# Builder ↔ Master Question Delegation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Le Builder s'auto-vérifie avant de poser une question à l'utilisateur. S'il a besoin malgré tout d'une réponse, il émet un marqueur `need_user_input` ; le Master joue alors le rôle de filtre intelligent en 3 niveaux (lookup `study_plan` → inférence LLM mini → forward à l'utilisateur), avec mise en cache automatique pour ne plus jamais reposer la même question.

**Architecture:** Nouveau module `agents/master/question_filter.py` qui expose une API pure (`resolve_builder_question(need, data_store, user_messages) -> QuestionResolution`). Master l'invoque dans son routage quand le dernier `AIMessage` du Builder contient un `additional_kwargs["need_user_input"]`. Le Builder est instruit via `step3_client_communication.md` de s'auto-vérifier avant d'émettre le marqueur. Garde-fous : compteur de questions par cycle, détection heuristique des questions hors-marqueur, nettoyage sur `<WRITE_DONE>`.

**Tech Stack:** Python 3.11, pytest, LangGraph, openai (gpt-5.4-mini via le helper `get_llm_config`), pandas (uniquement pour types).

---

## File Structure

| Fichier | Création / Modification | Responsabilité |
|---|---|---|
| `agents/master/question_filter.py` | **Créer** | Logique pure 3-niveaux : Python lookup, LLM mini inference, dispatch. ~150 lignes. |
| `agents/mortality/agents/master_node.py` | Modifier | Détection du `need_user_input`, branchement vers le filtre, gestion de la réponse utilisateur, marquage des HumanMessage synthétiques. ~50 lignes ajoutées. |
| `agents/mortality/agent_instructions/step3_client_communication.md` | Modifier | Règle d'auto-vérification avant question + protocole `need_user_input`. ~30 lignes texte. |
| `agents/mortality/agent_instructions/step1_planning.md` | Modifier | Règle `decision_required` → auto-check `study_plan` avant de poser la question. ~10 lignes texte. |
| `tests/test_question_filter.py` | **Créer** | Tests unitaires des 3 niveaux + dataclass + détection marker + extraction réponse. ~12 tests. |
| `tests/test_master_question_filter_integration.py` | **Créer** | Tests d'intégration master_node avec un Builder mocké qui émet `need_user_input`. ~6 tests. |
| `tests/test_user_messages_accumulation.py` | **Créer** | Tests sur `data_store["_user_messages"]` (Option B) et marquage des synthétiques (Option A). ~4 tests. |

**Convention** : tout le code de la logique métier de filtre vit dans `agents/master/question_filter.py`. Master n'a que les 50 lignes d'orchestration LangGraph (détection, branchement, mise à jour du state). Réutilisable par d'autres agents (ex: Writer si un jour il pose des questions).

---

## Task 1: Module foundation + Niveau 1 (Python lookup)

**Files:**
- Create: `agents/master/question_filter.py`
- Test: `tests/test_question_filter.py`

- [ ] **Step 1.1: Créer le squelette du test (RED)**

Créer `tests/test_question_filter.py` avec ce contenu :

```python
"""Tests pour agents.master.question_filter — résolution des questions Builder."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_resolution_dataclass_has_required_fields():
    """QuestionResolution doit exposer decision, value, source, confidence."""
    from agents.master.question_filter import QuestionResolution
    r = QuestionResolution(
        decision="answered",
        value=200,
        source="study_plan",
        confidence=1.0,
    )
    assert r.decision == "answered"
    assert r.value == 200
    assert r.source == "study_plan"
    assert r.confidence == 1.0


def test_level1_finds_in_study_plan():
    """Niveau 1 : si la clé existe dans study_plan, on retourne sa valeur."""
    from agents.master.question_filter import _try_resolve_from_data_store
    need = {"context_key": "smoothing_lambda"}
    data_store = {"study_plan": {"smoothing_lambda": 200}}
    val, source = _try_resolve_from_data_store(need, data_store)
    assert val == 200
    assert source == "study_plan"


def test_level1_finds_at_top_level_data_store():
    """Niveau 1 : fallback sur la clé top-level du data_store."""
    from agents.master.question_filter import _try_resolve_from_data_store
    need = {"context_key": "report_mode"}
    data_store = {"report_mode": "raw_rates"}
    val, source = _try_resolve_from_data_store(need, data_store)
    assert val == "raw_rates"
    assert source == "data_store"


def test_level1_returns_none_when_not_found():
    """Niveau 1 : ni study_plan ni top-level ne contient la clé."""
    from agents.master.question_filter import _try_resolve_from_data_store
    need = {"context_key": "lambda_inconnu"}
    val, source = _try_resolve_from_data_store(need, {"study_plan": {}})
    assert val is None
    assert source is None
```

- [ ] **Step 1.2: Lancer le test, vérifier qu'il échoue**

Run: `python -m pytest tests/test_question_filter.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'agents.master.question_filter'`

- [ ] **Step 1.3: Créer le module avec dataclass + Niveau 1**

Créer `agents/master/question_filter.py` :

```python
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

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class QuestionResolution:
    """Résultat de la résolution d'une question Builder."""
    decision:   Literal["answered", "forward", "use_default"]
    value:      Any
    source:     str          # "study_plan" | "data_store" | "llm_inference" | "user_response" | "default"
    confidence: float        # 0.0 à 1.0
    reasoning:  str = ""


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
```

- [ ] **Step 1.4: Lancer les tests, vérifier qu'ils passent**

Run: `python -m pytest tests/test_question_filter.py -v`
Expected: PASS pour les 4 tests.

- [ ] **Step 1.5: Commit**

```bash
git add agents/master/question_filter.py tests/test_question_filter.py
git commit -m "feat(question_filter): module foundation + Niveau 1 lookup"
```

---

## Task 2: detect_need_in_message + extract_user_answer

**Files:**
- Modify: `agents/master/question_filter.py`
- Test: `tests/test_question_filter.py`

- [ ] **Step 2.1: Ajouter les tests pour détection du marker (RED)**

Ajouter dans `tests/test_question_filter.py` :

```python
def test_detect_need_in_message_returns_dict():
    """Si l'AIMessage a additional_kwargs.need_user_input, on le retourne."""
    from agents.master.question_filter import detect_need_in_message
    from langchain_core.messages import AIMessage
    need = {
        "context_key": "smoothing_lambda",
        "question":    "Lambda 100, 200 ou 500 ?",
        "options":     [100, 200, 500],
    }
    msg = AIMessage(content="...", additional_kwargs={"need_user_input": need})
    assert detect_need_in_message(msg) == need


def test_detect_need_in_message_returns_none_when_absent():
    """Sans le marqueur, on retourne None."""
    from agents.master.question_filter import detect_need_in_message
    from langchain_core.messages import AIMessage
    msg = AIMessage(content="Plan d'analyse...")
    assert detect_need_in_message(msg) is None


def test_detect_need_in_message_returns_none_for_non_ai_message():
    """Pour un HumanMessage ou ToolMessage, retourne None."""
    from agents.master.question_filter import detect_need_in_message
    from langchain_core.messages import HumanMessage
    msg = HumanMessage(content="ok")
    assert detect_need_in_message(msg) is None
```

- [ ] **Step 2.2: Lancer les tests, vérifier qu'ils échouent**

Run: `python -m pytest tests/test_question_filter.py::test_detect_need_in_message_returns_dict -v`
Expected: FAIL avec `ImportError: cannot import name 'detect_need_in_message'`

- [ ] **Step 2.3: Implémenter detect_need_in_message**

Ajouter dans `agents/master/question_filter.py` :

```python
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
```

- [ ] **Step 2.4: Lancer les tests, vérifier qu'ils passent**

Run: `python -m pytest tests/test_question_filter.py -v`
Expected: PASS pour les 3 nouveaux tests.

- [ ] **Step 2.5: Commit**

```bash
git add agents/master/question_filter.py tests/test_question_filter.py
git commit -m "feat(question_filter): detect_need_in_message"
```

---

## Task 3: Niveau 2 — LLM mini inference

**Files:**
- Modify: `agents/master/question_filter.py`
- Test: `tests/test_question_filter.py`

- [ ] **Step 3.1: Ajouter le test du Niveau 2 avec mock LLM (RED)**

Ajouter dans `tests/test_question_filter.py` :

```python
def test_level2_llm_infers_lambda_from_smooth_keyword(monkeypatch):
    """Niveau 2 : le user a dit 'lissage doux' → mini doit déduire lambda=100."""
    from agents.master import question_filter as qf

    fake_response = {
        "answered":   True,
        "value":      100,
        "confidence": 0.85,
        "reasoning":  "User a explicitement dit 'lissage doux' = lambda faible.",
    }

    def _fake_mini_call(prompt: str) -> dict:
        return fake_response

    monkeypatch.setattr(qf, "_call_mini_for_inference", _fake_mini_call)

    need = {
        "context_key": "smoothing_lambda",
        "question":    "Lambda 100, 200 ou 500 ?",
        "options":     [100, 200, 500],
    }
    user_msgs = ["Construis-moi une table avec un lissage doux et progressif"]

    inf = qf._llm_infer_from_history(need, user_msgs)
    assert inf["answered"] is True
    assert inf["value"] == 100
    assert inf["confidence"] >= 0.7


def test_level2_returns_no_answer_when_user_silent(monkeypatch):
    """Niveau 2 : si l'historique user ne mentionne pas le sujet, answered=False."""
    from agents.master import question_filter as qf

    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": False, "value": None, "confidence": 0.2})

    need = {
        "context_key": "smoothing_lambda",
        "question":    "Lambda 100, 200 ou 500 ?",
        "options":     [100, 200, 500],
    }
    user_msgs = ["Bonjour", "Construis-moi une table"]
    inf = qf._llm_infer_from_history(need, user_msgs)
    assert inf["answered"] is False


def test_level2_handles_empty_user_messages(monkeypatch):
    """Niveau 2 : si user_messages est vide, retourne sans appeler le LLM."""
    from agents.master import question_filter as qf
    called = []
    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: called.append(p) or {"answered": False})
    inf = qf._llm_infer_from_history({"context_key": "x", "question": "y"}, [])
    assert inf["answered"] is False
    assert called == []  # pas d'appel LLM si rien à inférer
```

- [ ] **Step 3.2: Lancer les tests, vérifier qu'ils échouent**

Run: `python -m pytest tests/test_question_filter.py -v -k level2`
Expected: FAIL avec `AttributeError: module ... has no attribute '_llm_infer_from_history'`

- [ ] **Step 3.3: Implémenter _llm_infer_from_history**

Ajouter dans `agents/master/question_filter.py` :

```python
import json


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
    # Normaliser : valeurs par défaut si LLM renvoie un JSON incomplet
    return {
        "answered":   bool(result.get("answered", False)),
        "value":      result.get("value"),
        "confidence": float(result.get("confidence", 0.0)),
        "reasoning":  str(result.get("reasoning", "")),
    }
```

- [ ] **Step 3.4: Lancer les tests, vérifier qu'ils passent**

Run: `python -m pytest tests/test_question_filter.py -v -k level2`
Expected: PASS pour les 3 tests.

- [ ] **Step 3.5: Commit**

```bash
git add agents/master/question_filter.py tests/test_question_filter.py
git commit -m "feat(question_filter): Niveau 2 LLM inference"
```

---

## Task 4: resolve_builder_question (orchestrateur 3-niveaux)

**Files:**
- Modify: `agents/master/question_filter.py`
- Test: `tests/test_question_filter.py`

- [ ] **Step 4.1: Tests d'intégration des 3 niveaux (RED)**

Ajouter dans `tests/test_question_filter.py` :

```python
def test_resolve_uses_level1_when_study_plan_match():
    """L'orchestrateur retourne immédiatement la valeur du study_plan."""
    from agents.master.question_filter import resolve_builder_question
    need = {"context_key": "smoothing_lambda", "question": "?", "options": [100, 200]}
    data_store = {"study_plan": {"smoothing_lambda": 200}}
    res = resolve_builder_question(need, data_store, ["bonjour"])
    assert res.decision == "answered"
    assert res.value == 200
    assert res.source == "study_plan"
    assert res.confidence == 1.0


def test_resolve_uses_level2_when_level1_misses(monkeypatch):
    """Niveau 1 ne match pas, Niveau 2 trouve."""
    from agents.master import question_filter as qf
    from agents.master.question_filter import resolve_builder_question
    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": True, "value": 100, "confidence": 0.85})
    need = {"context_key": "smoothing_lambda", "question": "?", "options": [100, 200]}
    res = resolve_builder_question(need, {}, ["lissage doux"])
    assert res.decision == "answered"
    assert res.value == 100
    assert res.source == "llm_inference"
    assert res.confidence == 0.85


def test_resolve_forwards_when_no_level_matches(monkeypatch):
    """Aucun niveau ne match → forward."""
    from agents.master import question_filter as qf
    from agents.master.question_filter import resolve_builder_question
    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": False, "confidence": 0.2})
    need = {"context_key": "smoothing_lambda", "question": "?", "options": [100, 200]}
    res = resolve_builder_question(need, {}, ["bonjour"])
    assert res.decision == "forward"
    assert res.value is None


def test_resolve_forwards_when_confidence_below_threshold(monkeypatch):
    """Niveau 2 trouve mais confidence < seuil → forward."""
    from agents.master import question_filter as qf
    from agents.master.question_filter import resolve_builder_question
    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": True, "value": 100, "confidence": 0.5})
    need = {"context_key": "smoothing_lambda", "question": "?", "options": [100, 200]}
    res = resolve_builder_question(need, {}, ["lissage"], confidence_threshold=0.7)
    assert res.decision == "forward"
```

- [ ] **Step 4.2: Lancer, vérifier qu'ils échouent**

Run: `python -m pytest tests/test_question_filter.py -v -k resolve`
Expected: FAIL avec `ImportError: cannot import name 'resolve_builder_question'`

- [ ] **Step 4.3: Implémenter l'orchestrateur**

Ajouter dans `agents/master/question_filter.py` :

```python
def resolve_builder_question(
    need:                 dict,
    data_store:           dict,
    user_messages:        list[str],
    confidence_threshold: float = 0.7,
) -> QuestionResolution:
    """Orchestrateur 3-niveaux : Python lookup → LLM mini → forward.

    Args:
        need: dict avec au minimum `context_key` et `question`. Peut aussi
              contenir `options`, `default`.
        data_store: l'état partagé du LangGraph.
        user_messages: liste des messages utilisateur (pas les synthétiques
                       du Master, voir Task 6).
        confidence_threshold: en dessous de ce seuil au Niveau 2, on forward
                              au lieu d'injecter une réponse LLM-inférée.

    Returns:
        QuestionResolution avec `decision` ∈ {"answered", "forward", "use_default"}.
    """
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
```

- [ ] **Step 4.4: Lancer tous les tests du module**

Run: `python -m pytest tests/test_question_filter.py -v`
Expected: PASS pour 11 tests.

- [ ] **Step 4.5: Commit**

```bash
git add agents/master/question_filter.py tests/test_question_filter.py
git commit -m "feat(question_filter): orchestrateur resolve_builder_question"
```

---

## Task 5: extract_user_answer (parsing réponse libre user)

**Files:**
- Modify: `agents/master/question_filter.py`
- Test: `tests/test_question_filter.py`

- [ ] **Step 5.1: Tests d'extraction (RED)**

Ajouter dans `tests/test_question_filter.py` :

```python
def test_extract_user_answer_uses_default_when_options_unspecified(monkeypatch):
    """Si pas d'options, on retourne le texte brut comme valeur."""
    from agents.master.question_filter import extract_user_answer
    need = {"context_key": "objectif", "question": "Quel objectif ?"}
    val = extract_user_answer("certifier la table", need)
    assert val == "certifier la table"


def test_extract_user_answer_matches_option_when_explicit(monkeypatch):
    """Si options=[100,200,500] et user dit '200', extraction directe."""
    from agents.master import question_filter as qf
    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": True, "value": 200, "confidence": 0.95})
    need = {"context_key": "lambda", "question": "?", "options": [100, 200, 500]}
    val = qf.extract_user_answer("200 ça me va", need)
    assert val == 200


def test_extract_user_answer_returns_none_when_unparseable(monkeypatch):
    """Si la réponse user n'est pas mappable aux options, retourne None."""
    from agents.master import question_filter as qf
    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": False, "confidence": 0.1})
    need = {"context_key": "lambda", "question": "?", "options": [100, 200]}
    val = qf.extract_user_answer("euh je sais pas", need)
    assert val is None
```

- [ ] **Step 5.2: Lancer, vérifier l'échec**

Run: `python -m pytest tests/test_question_filter.py -v -k extract_user`
Expected: FAIL avec `ImportError: cannot import name 'extract_user_answer'`

- [ ] **Step 5.3: Implémenter extract_user_answer**

Ajouter dans `agents/master/question_filter.py` :

```python
def extract_user_answer(response_text: str, need: dict) -> Any:
    """Extrait la valeur structurée d'une réponse libre de l'utilisateur.

    Si `need.options` est défini : utilise gpt-5.4-mini pour mapper la réponse
    libre vers une option (ex: "ça me va, 200" → 200). Retourne None si rien
    ne match.

    Si `need.options` est absent : retourne `response_text.strip()` tel quel.
    """
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
```

- [ ] **Step 5.4: Lancer les tests**

Run: `python -m pytest tests/test_question_filter.py -v`
Expected: PASS pour 14 tests.

- [ ] **Step 5.5: Commit**

```bash
git add agents/master/question_filter.py tests/test_question_filter.py
git commit -m "feat(question_filter): extract_user_answer pour parsing structuré"
```

---

## Task 6: User messages accumulator + synthetic marker (Master state)

**Files:**
- Modify: `agents/mortality/agents/master_node.py`
- Test: `tests/test_user_messages_accumulation.py`

**Contexte** : actuellement les `HumanMessage` sont ambigus — Master ne distingue pas un message vrai user d'une instruction synthétique qu'il génère. Pour Niveau 2 du filtre (qui passe `user_messages` au LLM), il faut une liste propre.

- [ ] **Step 6.1: Test d'accumulation _user_messages (RED)**

Créer `tests/test_user_messages_accumulation.py` :

```python
"""Tests : Master maintient `data_store["_user_messages"]` avec uniquement
les messages user (pas les synthétiques)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import HumanMessage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _fake_classify(*a, **kw):
    return {
        "kind": "task", "write": "yes", "report_mode": "full_report",
        "intent": "build_and_write", "reply": "",
    }


def test_master_stores_first_user_message():
    """Le 1er HumanMessage user doit apparaître dans data_store['_user_messages']."""
    from agents.mortality.agents import master_node as mn
    state = {
        "messages":    [HumanMessage(content="construis-moi une table")],
        "data_store":  {
            "_disambiguation_done": True,
            "study_plan":           {"gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
    }
    with patch.object(mn, "_classify_intent", _fake_classify):
        out = mn.master_node(state)
    user_msgs = out["data_store"].get("_user_messages") or []
    assert "construis-moi une table" in user_msgs


def test_master_does_not_store_synthetic_messages():
    """Les HumanMessages synthétiques (instructions Master→Builder) ne doivent
    PAS apparaître dans _user_messages."""
    from agents.mortality.agents import master_node as mn
    synthetic = HumanMessage(
        content="Mode de rapport : full_report\nSections actives : [...]",
        additional_kwargs={"source": "master_synthetic"},
    )
    real = HumanMessage(content="construit avec un lissage doux")
    state = {
        "messages":    [synthetic, real],
        "data_store":  {
            "_disambiguation_done": True,
            "study_plan":           {"gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
    }
    with patch.object(mn, "_classify_intent", _fake_classify):
        out = mn.master_node(state)
    user_msgs = out["data_store"].get("_user_messages") or []
    assert "construit avec un lissage doux" in user_msgs
    assert all("Mode de rapport" not in m for m in user_msgs)


def test_master_accumulates_multiple_user_messages():
    """Plusieurs tours user → liste cumulative."""
    from agents.mortality.agents import master_node as mn
    state = {
        "messages":    [HumanMessage(content="bonjour")],
        "data_store":  {
            "_disambiguation_done": True,
            "_user_messages":       ["fais-moi un rapport"],  # message d'un tour précédent
            "study_plan":           {"gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
    }
    with patch.object(mn, "_classify_intent", _fake_classify):
        out = mn.master_node(state)
    user_msgs = out["data_store"].get("_user_messages") or []
    assert "fais-moi un rapport" in user_msgs
    assert "bonjour" in user_msgs
```

- [ ] **Step 6.2: Lancer, vérifier l'échec**

Run: `python -m pytest tests/test_user_messages_accumulation.py -v`
Expected: FAIL — `_user_messages` n'existe pas dans le data_store.

- [ ] **Step 6.3: Implémenter l'accumulateur dans master_node**

Modifier `agents/mortality/agents/master_node.py`. Localiser la section qui extrait `last_human` (autour de la ligne 460) et ajouter, juste après que `last_human` est calculé, l'accumulation :

```python
# Accumuler le message user (filtré : pas les synthétiques) dans data_store.
# Sert de source de vérité pour le filtre question_filter.
if last_human:
    last_msg = next(
        (m for m in reversed(messages_list)
         if getattr(m, "type", "") == "human"),
        None,
    )
    is_synthetic = (
        last_msg is not None
        and (getattr(last_msg, "additional_kwargs", None) or {}).get("source") == "master_synthetic"
    )
    if not is_synthetic:
        history = data_store.setdefault("_user_messages", [])
        if not history or history[-1] != last_human:
            history.append(last_human)
```

- [ ] **Step 6.4: Marquer les HumanMessage synthétiques émis par Master**

Repérer dans `master_node.py` les endroits où Master émet un `HumanMessage` synthétique (instruction Builder, redirections NEED_DATA, etc.) — environ 2-3 occurrences. Ajouter `additional_kwargs={"source": "master_synthetic"}` à chaque construction. Exemple :

```python
# Avant
instr = HumanMessage(content=f"Mode de rapport : {report_mode}\n...")
# Après
instr = HumanMessage(
    content=f"Mode de rapport : {report_mode}\n...",
    additional_kwargs={"source": "master_synthetic"},
)
```

- [ ] **Step 6.5: Lancer les tests, vérifier qu'ils passent**

Run: `python -m pytest tests/test_user_messages_accumulation.py -v`
Expected: PASS pour 3 tests.

Run: `python -m pytest tests/ -q` (suite complète)
Expected: tous verts (pas de régression).

- [ ] **Step 6.6: Commit**

```bash
git add agents/mortality/agents/master_node.py tests/test_user_messages_accumulation.py
git commit -m "feat(master): _user_messages accumulator + synthetic marker"
```

---

## Task 7: Master integration — détection need + branchement filtre

**Files:**
- Modify: `agents/mortality/agents/master_node.py`
- Test: `tests/test_master_question_filter_integration.py`

- [ ] **Step 7.1: Tests d'intégration master_node + need_user_input (RED)**

Créer `tests/test_master_question_filter_integration.py` :

```python
"""Tests d'intégration : master_node détecte le marqueur need_user_input
émis par le Builder et applique la résolution 3-niveaux."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _builder_emits_need(question="Lambda 100, 200 ou 500 ?",
                        context_key="smoothing_lambda",
                        options=None):
    return AIMessage(
        content="J'ai besoin d'une précision pour le lissage.",
        additional_kwargs={
            "need_user_input": {
                "context_key": context_key,
                "question":    question,
                "options":     options or [100, 200, 500],
            }
        }
    )


def test_master_resolves_via_study_plan_and_routes_back_to_builder():
    """study_plan contient déjà la réponse → Master injecte et route Builder."""
    from agents.mortality.agents import master_node as mn

    state = {
        "messages":    [
            HumanMessage(content="construit la table"),
            _builder_emits_need(),
        ],
        "data_store":  {
            "_disambiguation_done":   True,
            "_master_builder_cycles": 1,
            "study_plan":             {"smoothing_lambda": 200, "gender_segmentation": "unisex"},
            "_user_messages":         ["construit la table"],
        },
        "dataset_ref": None,
        "active_agent": "master",
    }
    out = mn.master_node(state)

    # Master a routé Builder
    assert out.get("active_agent") == "builder"
    # Master a injecté un HumanMessage synthétique avec la réponse
    msgs = out.get("messages") or []
    injection = next((m for m in msgs if isinstance(m, HumanMessage)), None)
    assert injection is not None
    assert "200" in injection.content
    # Marquage source synthetic
    src = (injection.additional_kwargs or {}).get("source")
    assert src == "master_synthetic"


def test_master_forwards_to_user_when_no_signal(monkeypatch):
    """study_plan vide + LLM ne trouve rien → Master pose la question à l'user."""
    from agents.mortality.agents import master_node as mn
    from agents.master import question_filter as qf

    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": False, "confidence": 0.1})

    state = {
        "messages":    [
            HumanMessage(content="construit la table"),
            _builder_emits_need(),
        ],
        "data_store":  {
            "_disambiguation_done":   True,
            "_master_builder_cycles": 1,
            "study_plan":             {"gender_segmentation": "unisex"},
            "_user_messages":         ["construit la table"],
        },
        "dataset_ref": None,
        "active_agent": "master",
    }
    out = mn.master_node(state)

    # Master n'a PAS routé vers Builder (pause pour réponse user)
    assert out.get("active_agent") != "builder"
    # Un AIMessage est posé (la question)
    msgs = out.get("messages") or []
    assert any(isinstance(m, AIMessage) and "lambda" in (m.content or "").lower() for m in msgs)
    # Le need est mémorisé pour capter la réponse au prochain tour
    assert out["data_store"].get("_pending_need") is not None


def test_master_extracts_user_response_and_routes_back(monkeypatch):
    """Quand _pending_need existe et user répond, Master extrait et inject Builder."""
    from agents.mortality.agents import master_node as mn
    from agents.master import question_filter as qf

    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": True, "value": 200, "confidence": 0.95})

    state = {
        "messages":    [HumanMessage(content="200 ça me va")],
        "data_store":  {
            "_disambiguation_done":   True,
            "_master_builder_cycles": 1,
            "study_plan":             {"gender_segmentation": "unisex"},
            "_user_messages":         ["construit la table", "200 ça me va"],
            "_pending_need":          {
                "context_key": "smoothing_lambda",
                "question":    "Lambda 100, 200 ou 500 ?",
                "options":     [100, 200, 500],
            },
        },
        "dataset_ref": None,
        "active_agent": "master",
    }
    out = mn.master_node(state)

    # Cache : study_plan["smoothing_lambda"] doit avoir 200
    assert out["data_store"]["study_plan"]["smoothing_lambda"] == 200
    # _pending_need doit être nettoyé
    assert "_pending_need" not in out["data_store"]
    # Builder relancé
    assert out.get("active_agent") == "builder"
```

- [ ] **Step 7.2: Lancer, vérifier l'échec**

Run: `python -m pytest tests/test_master_question_filter_integration.py -v`
Expected: FAIL sur les 3 tests (le filtre n'est pas branché).

- [ ] **Step 7.3: Brancher le filtre dans master_node**

Modifier `agents/mortality/agents/master_node.py`. Au tout début de `master_node(state)`, après l'extraction du `data_store` et la branche WRITE_DONE, **avant les autres branches**, ajouter :

```python
    # ── Branche : résolution d'un need_user_input émis par le Builder ────────
    # Si le dernier AIMessage du Builder contient un marqueur need_user_input,
    # on délègue la résolution au filtre 3-niveaux de question_filter.
    from langchain_core.messages import AIMessage as _AIMessage
    last_ai = next(
        (m for m in reversed(messages_list) if isinstance(m, _AIMessage)),
        None,
    )
    if last_ai is not None:
        from agents.master.question_filter import (
            detect_need_in_message, resolve_builder_question,
        )
        need = detect_need_in_message(last_ai)
        if need:
            user_msgs = data_store.get("_user_messages") or []
            resolution = resolve_builder_question(need, data_store, user_msgs)

            if resolution.decision == "answered":
                # Cacher dans study_plan + injecter HumanMessage synthétique
                sp = data_store.setdefault("study_plan", {})
                sp[need["context_key"]] = resolution.value
                from langchain_core.messages import HumanMessage as _HMsg
                instr = _HMsg(
                    content=(
                        f"[Master] Réponse à ta question '{need.get('context_key')}' : "
                        f"{resolution.value} (source: {resolution.source})."
                    ),
                    additional_kwargs={"source": "master_synthetic"},
                )
                return {
                    "messages":     [instr],
                    "events":       [{"type": "agent_switch", "agent": "MasterAgent"},
                                     {"type": "message",
                                      "content": f"Question '{need.get('context_key')}' "
                                                 f"résolue automatiquement (source: {resolution.source})."}],
                    "active_agent": "builder",
                    "data_store":   data_store,
                }
            else:  # forward
                data_store["_pending_need"] = need
                from langchain_core.messages import AIMessage as _AIM
                question_msg = _AIM(content=need.get("question", "Précision nécessaire."))
                return {
                    "messages":     [question_msg],
                    "events":       [{"type": "agent_switch", "agent": "MasterAgent"},
                                     {"type": "message", "content": need.get("question", "")}],
                    "data_store":   data_store,
                }
```

- [ ] **Step 7.4: Gérer la réponse user quand `_pending_need` existe**

Toujours dans `master_node.py`, avant la branche `_classify_intent` habituelle, intercepter le cas "réponse à une question pendante" :

```python
    # Si on a une question pendante et que le dernier message est human,
    # on extrait la réponse et on relance le Builder.
    pending = data_store.get("_pending_need")
    if pending and last_human:
        from agents.master.question_filter import extract_user_answer
        value = extract_user_answer(last_human, pending)
        if value is not None:
            sp = data_store.setdefault("study_plan", {})
            sp[pending["context_key"]] = value
            data_store.pop("_pending_need", None)
            from langchain_core.messages import HumanMessage as _HMsg
            instr = _HMsg(
                content=(f"[Master] L'utilisateur a répondu '{pending.get('context_key')}' "
                         f"= {value}."),
                additional_kwargs={"source": "master_synthetic"},
            )
            return {
                "messages":     [instr],
                "events":       [{"type": "agent_switch", "agent": "MasterAgent"},
                                 {"type": "message",
                                  "content": f"Réponse '{pending.get('context_key')}' enregistrée : {value}."}],
                "active_agent": "builder",
                "data_store":   data_store,
            }
```

**Important** : ce bloc doit être **après** l'accumulation des `_user_messages` (Task 6) pour que la réponse soit bien dans l'historique.

- [ ] **Step 7.5: Lancer les tests**

Run: `python -m pytest tests/test_master_question_filter_integration.py -v`
Expected: PASS pour 3 tests.

Run: `python -m pytest tests/ -q`
Expected: tous verts.

- [ ] **Step 7.6: Commit**

```bash
git add agents/mortality/agents/master_node.py tests/test_master_question_filter_integration.py
git commit -m "feat(master): branchement question_filter + gestion réponse pending"
```

---

## Task 8: Garde-fou compteur de questions par cycle

**Files:**
- Modify: `agents/master/question_filter.py`
- Modify: `agents/mortality/agents/master_node.py`
- Test: `tests/test_master_question_filter_integration.py`

- [ ] **Step 8.1: Test du garde-fou (RED)**

Ajouter dans `tests/test_master_question_filter_integration.py` :

```python
def test_master_uses_default_after_max_questions_in_cycle(monkeypatch):
    """Au-delà de 3 questions dans un cycle, Master force use_default."""
    from agents.mortality.agents import master_node as mn
    from agents.master import question_filter as qf

    # Forcer le LLM à dire "answered=false" (ne match jamais)
    monkeypatch.setattr(qf, "_call_mini_for_inference",
                        lambda p: {"answered": False, "confidence": 0.1})

    builder_msg = AIMessage(
        content="J'ai besoin d'une précision.",
        additional_kwargs={
            "need_user_input": {
                "context_key": "lambda_3rd",
                "question":    "Quel paramètre ?",
                "options":     [100, 200],
                "default":     100,
            }
        }
    )
    state = {
        "messages":    [
            HumanMessage(content="bonjour"),
            builder_msg,
        ],
        "data_store":  {
            "_disambiguation_done":             True,
            "_master_builder_cycles":           1,
            "_questions_asked_this_cycle":      3,    # déjà 3 questions posées
            "study_plan":                       {"gender_segmentation": "unisex"},
            "_user_messages":                   ["bonjour"],
        },
        "dataset_ref": None,
        "active_agent": "master",
    }
    out = mn.master_node(state)

    # Master a forcé le default sans poser la question
    assert out.get("active_agent") == "builder"
    sp = out["data_store"].get("study_plan", {})
    assert sp.get("lambda_3rd") == 100   # default appliqué
    # Pas de _pending_need (pas de forward)
    assert "_pending_need" not in out["data_store"]
```

- [ ] **Step 8.2: Lancer, vérifier l'échec**

Run: `python -m pytest tests/test_master_question_filter_integration.py::test_master_uses_default_after_max_questions_in_cycle -v`
Expected: FAIL.

- [ ] **Step 8.3: Implémenter le compteur dans master_node**

Modifier la branche du filtre dans `master_node.py` (Task 7). Au début, juste après `if need:`, ajouter le check du compteur :

```python
        if need:
            # Garde-fou : limiter le nombre de questions par cycle
            asked = data_store.get("_questions_asked_this_cycle", 0)
            MAX_QUESTIONS_PER_CYCLE = 3
            if asked >= MAX_QUESTIONS_PER_CYCLE:
                # Forcer le default sans poser la question
                default_val = need.get("default")
                sp = data_store.setdefault("study_plan", {})
                sp[need["context_key"]] = default_val
                from langchain_core.messages import HumanMessage as _HMsg
                instr = _HMsg(
                    content=(
                        f"[Master] Trop de questions dans ce cycle ({asked}). "
                        f"Application du default pour '{need.get('context_key')}' : {default_val}."
                    ),
                    additional_kwargs={"source": "master_synthetic"},
                )
                return {
                    "messages":     [instr],
                    "events":       [{"type": "agent_switch", "agent": "MasterAgent"},
                                     {"type": "message",
                                      "content": f"Question '{need.get('context_key')}' "
                                                 f"forcée au default ({default_val}) — "
                                                 f"limite de {MAX_QUESTIONS_PER_CYCLE} questions atteinte."}],
                    "active_agent": "builder",
                    "data_store":   data_store,
                }

            # ... suite (resolve_builder_question + branchement)
            user_msgs = data_store.get("_user_messages") or []
            resolution = resolve_builder_question(need, data_store, user_msgs)
            # Incrémenter compteur uniquement si on traite la question (LLM call ou forward)
            data_store["_questions_asked_this_cycle"] = asked + 1
            # ... le reste du code de branchement reste identique
```

- [ ] **Step 8.4: Réinitialiser le compteur sur BUILD_DONE et WRITE_DONE**

Repérer dans `master_node.py` les blocs WRITE_DONE et BUILD_DONE (où `_master_builder_cycles` est déjà nettoyé) et ajouter `data_store.pop("_questions_asked_this_cycle", None)` au même endroit.

- [ ] **Step 8.5: Lancer les tests**

Run: `python -m pytest tests/test_master_question_filter_integration.py -v`
Expected: PASS pour 4 tests.

Run: `python -m pytest tests/ -q`
Expected: tous verts.

- [ ] **Step 8.6: Commit**

```bash
git add agents/mortality/agents/master_node.py tests/test_master_question_filter_integration.py
git commit -m "feat(master): garde-fou _questions_asked_this_cycle"
```

---

## Task 9: Update Builder instructions — auto-check before asking

**Files:**
- Modify: `agents/mortality/agent_instructions/step3_client_communication.md`
- Modify: `agents/mortality/agent_instructions/step1_planning.md`

**Note** : pas de tests Python — c'est du contenu textuel pour le LLM.

- [ ] **Step 9.1: Modifier step3_client_communication.md**

Remplacer le contenu de `agents/mortality/agent_instructions/step3_client_communication.md` par :

```markdown
## Communication avec l'utilisateur — règle d'auto-vérification

**AVANT de poser une question à l'utilisateur, vérifie systématiquement :**

1. **Le user a-t-il déjà répondu explicitement dans son message initial ?**
   Mots-clés à reconnaître : "rapport"/"PDF" → `write=yes` ; "sans rapport" → `write=no` ;
   "taux bruts" → `report_mode=raw_rates` ; "descriptive" → `report_mode=description` ;
   "lissage doux/standard/fort" → choix lambda implicite.

2. **Le `study_plan` contient-il déjà la valeur ?**
   Si oui (`study_plan.smoothing_algorithm`, `study_plan.observation_end_date`…),
   utilise-la directement sans re-demander.

3. **Sinon (vraiment ambigu) → utiliser le protocole need_user_input.**

## Plan d'analyse — communication au client

Après planification interne, présente une version synthétique :

> **Plan d'analyse :**
> - Séquence de tools prévus
> - Choix techniques (lissage, plage d'âge, table de référence)
> - Livrables prévus (graphiques, tableaux)
>
> Je commence.

**Pas de demande de confirmation si l'intent est explicite.** Lance directement les tools.

## Protocole `need_user_input` — quand tu dois VRAIMENT demander

Si après auto-vérification tu as besoin d'une réponse utilisateur, n'écris PAS la question dans ton message texte. À la place, émets un AIMessage **avec ce marqueur structuré** :

```json
additional_kwargs:
  need_user_input:
    context_key: "smoothing_lambda"        # clé canonique pour le cache
    question:    "Lambda 100, 200 ou 500 ?"
    options:     [100, 200, 500]
    default:     100                        # fallback si rien trouvé
```

Le Master interprétera ce marqueur, vérifiera le contexte, et soit te répondra
directement (cache, inférence), soit demandera à l'utilisateur. Tu recevras
ensuite un HumanMessage `[Master] Réponse à ta question '<key>' : <value>`.

**Ne demande JAMAIS la même question deux fois** — si tu vois un HumanMessage
synthétique du Master mentionnant la `context_key`, considère que la réponse
est déjà dans `study_plan`.
```

- [ ] **Step 9.2: Modifier step1_planning.md (decision_required)**

Localiser dans `step1_planning.md` la section qui parle de `decision_required` (autour de la ligne 35-45). Remplacer :

```markdown
### Tool retournant `decision_required` — quoi faire

Quand un tool retourne un dict avec `decision_required`, **vérifie d'abord si
le contexte permet de trancher seul** :

1. `study_plan[<context_key>]` existe-t-il ? → utilise cette valeur, ignore
   `decision_required`.
2. L'utilisateur a-t-il exprimé une préférence dans son message initial ?
   → applique-la.
3. Sinon, émets un AIMessage avec un marqueur `need_user_input`
   (cf. step3_client_communication.md). Le Master se chargera de filtrer
   la question ou de la forwarder à l'utilisateur.
```

- [ ] **Step 9.3: Vérifier que la suite de tests passe toujours**

Run: `python -m pytest tests/ -q`
Expected: tous verts. Aucun test ne dépend de ces fichiers MD.

- [ ] **Step 9.4: Commit**

```bash
git add agents/mortality/agent_instructions/step3_client_communication.md \
        agents/mortality/agent_instructions/step1_planning.md
git commit -m "docs(builder): instructions auto-check + protocole need_user_input"
```

---

## Task 10: End-to-end manual simulation

**Files:**
- Create: `scripts/sim_test_question_delegation.py`

**But** : prouver le pattern complet en environnement réel avec OpenAI.

- [ ] **Step 10.1: Créer le script de simulation**

Créer `scripts/sim_test_question_delegation.py` :

```python
"""Simulation end-to-end du pattern Builder→Master question delegation.

3 scénarios :
  A. study_plan déjà rempli → Niveau 1, 0 LLM call.
  B. Signal implicite dans le message user → Niveau 2 (LLM mini).
  C. Pas de signal → Niveau 3 (forward), simulation user répond, Master extrait.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
load_dotenv(_PROJECT_ROOT / ".env")


def _builder_emits_lambda_question():
    return AIMessage(
        content="J'ai besoin d'une précision pour le lissage.",
        additional_kwargs={
            "need_user_input": {
                "context_key": "smoothing_lambda",
                "question":    "Quel paramètre lambda Whittaker ? (100=souple, 200=standard, 500=fort)",
                "options":     [100, 200, 500],
                "default":     200,
            }
        }
    )


def _print_outcome(label, out, data_store):
    print(f"\n  ── {label} ──")
    print(f"     active_agent      : {out.get('active_agent')}")
    msgs = out.get("messages") or []
    for m in msgs:
        cls = type(m).__name__
        c = (getattr(m, "content", "") or "")[:200]
        print(f"     {cls}: {c}")
    sp = (out.get("data_store") or {}).get("study_plan") or {}
    if sp:
        print(f"     study_plan        : {sp}")


def scenario_a_level1():
    """study_plan a déjà la valeur → 0 LLM call."""
    from agents.mortality.agents import master_node as mn
    print("\n" + "█" * 70)
    print("  SCÉNARIO A — Niveau 1 (study_plan rempli)")
    print("█" * 70)
    state = {
        "messages":    [HumanMessage(content="construit la table"),
                        _builder_emits_lambda_question()],
        "data_store":  {
            "_disambiguation_done":   True,
            "_master_builder_cycles": 1,
            "_user_messages":         ["construit la table"],
            "study_plan":             {"smoothing_lambda": 300, "gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
        "active_agent": "master",
    }
    out = mn.master_node(state)
    _print_outcome("Master", out, state["data_store"])
    assert out["data_store"]["study_plan"]["smoothing_lambda"] == 300
    print("  ✓ Niveau 1 OK — réponse 300 injectée sans appel LLM.")


def scenario_b_level2():
    """User a dit 'lissage doux' → mini doit déduire 100."""
    from agents.mortality.agents import master_node as mn
    print("\n" + "█" * 70)
    print("  SCÉNARIO B — Niveau 2 (LLM infère depuis 'lissage doux')")
    print("█" * 70)
    state = {
        "messages":    [HumanMessage(content="construit la table avec un lissage doux"),
                        _builder_emits_lambda_question()],
        "data_store":  {
            "_disambiguation_done":   True,
            "_master_builder_cycles": 1,
            "_user_messages":         ["construit la table avec un lissage doux"],
            "study_plan":             {"gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
        "active_agent": "master",
    }
    out = mn.master_node(state)
    _print_outcome("Master", out, state["data_store"])
    sp_value = out["data_store"]["study_plan"].get("smoothing_lambda")
    print(f"  → Mini a inféré : {sp_value}")
    print("  ✓ Niveau 2 OK si valeur cohérente avec 'doux' (100 attendu).")


def scenario_c_level3():
    """Pas de signal → forward au user, user répond, Master extrait."""
    from agents.mortality.agents import master_node as mn
    print("\n" + "█" * 70)
    print("  SCÉNARIO C — Niveau 3 (forward + extract)")
    print("█" * 70)

    # Tour 1 : Builder émet need, Master forward
    state = {
        "messages":    [HumanMessage(content="construit la table"),
                        _builder_emits_lambda_question()],
        "data_store":  {
            "_disambiguation_done":   True,
            "_master_builder_cycles": 1,
            "_user_messages":         ["construit la table"],
            "study_plan":             {"gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
        "active_agent": "master",
    }
    out = mn.master_node(state)
    _print_outcome("Master tour 1 (forward)", out, state["data_store"])
    assert "_pending_need" in out["data_store"]
    state["data_store"] = out["data_store"]
    state["messages"].extend(out.get("messages") or [])

    # Tour 2 : user répond
    print("\n  → User répond : '500 ça me va'")
    state["messages"].append(HumanMessage(content="500 ça me va"))
    out2 = mn.master_node(state)
    _print_outcome("Master tour 2 (extract + route)", out2, state["data_store"])
    assert out2["data_store"]["study_plan"].get("smoothing_lambda") == 500
    assert "_pending_need" not in out2["data_store"]
    assert out2.get("active_agent") == "builder"
    print("  ✓ Niveau 3 OK — réponse 500 capturée et cachée.")


def main():
    print("┌" + "─" * 68 + "┐")
    print("│  Simulation : Builder→Master question delegation                  │")
    print("│  Coût estimé : ~0,005 €                                            │")
    print("└" + "─" * 68 + "┘")
    scenario_a_level1()
    scenario_b_level2()
    scenario_c_level3()
    print("\n" + "═" * 70)
    print("  3 scénarios passés.")
    print("═" * 70)


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.2: Lancer la simulation**

Run: `python scripts/sim_test_question_delegation.py`
Expected: les 3 scénarios passent, 1-2 LLM calls totaux (Niveaux 2 et 3).

- [ ] **Step 10.3: Commit**

```bash
git add scripts/sim_test_question_delegation.py
git commit -m "test(question_filter): simulation end-to-end 3 scénarios"
```

---

## Vérification finale

- [ ] **Step F.1: Suite complète**

Run: `python -m pytest tests/ -q && python scripts/check_template.py`
Expected: tous tests verts (≥ 200 attendus après ajout des 22 nouveaux tests), template valide.

- [ ] **Step F.2: Critère d'acceptation**

- ≥ 14 tests dans `tests/test_question_filter.py` (Niveaux 1/2 + dataclass + detect/extract).
- ≥ 4 tests dans `tests/test_master_question_filter_integration.py` (resolve study_plan, forward, extract+route, garde-fou compteur).
- ≥ 3 tests dans `tests/test_user_messages_accumulation.py`.
- Simulation `sim_test_question_delegation.py` joue les 3 scénarios sans crash.
- Suite globale : pas de régression vs baseline (177 verts avant ce plan).
- `check_template.py` : ✓ template valide.

---

## Ordre d'exécution recommandé

1. **Tasks 1-5** (foundation `question_filter.py`) — peuvent être enchaînées rapidement, peu de dépendances.
2. **Task 6** (`_user_messages` accumulator + synthetic marker) — pré-requis avant Task 7.
3. **Task 7** (intégration master_node) — gros morceau, à faire avec attention.
4. **Task 8** (garde-fou compteur) — simple ajout sur la branche existante.
5. **Task 9** (instructions Builder MD) — sans code, peut être fait en parallèle.
6. **Task 10** (simulation E2E) — validation finale avec vrai LLM.
