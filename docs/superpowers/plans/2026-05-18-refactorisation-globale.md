# Refactorisation Globale AgentActuariat — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabiliser, factoriser et professionnaliser le codebase AgentActuariat tout en préservant l'architecture master/sub-agent LangGraph, et préparer le terrain pour l'ajout de nouveaux agents de calcul et de nouveaux formats de rapport.

**Architecture:** LangGraph StateGraph existant conservé (master → builder/writer/rag + tools_node). Refacto par lots indépendants : (1) hygiène code mort + magic strings → enums, (2) Pydantic v2 + reducers explicites + exceptions custom, (3) tool registry metadata-driven + AgentNode abstrait + renderer registry, (4) observability + retry agent-level + checkpointer persistant, (5) tests d'intégration + failure injection.

**Tech Stack:** Python 3.10+, LangGraph 1.x, LangChain Core, OpenAI/Anthropic SDKs, Pydantic v2, pandas, reportlab, Dash (canvas), pytest, structlog (à introduire).

---

## Contexte audit (faits vérifiés)

| Constat | Preuve (file:line) |
|---|---|
| Code mort | [agents/mortality/agents/mortality_node.py](agents/mortality/agents/mortality_node.py), [agents/mortality/agents/report_node.py](agents/mortality/agents/report_node.py), [agents/mortality/writer_agent.py](agents/mortality/writer_agent.py), [OLD/](OLD/) |
| Hardcoding LLM | [config.py:7](config.py#L7) `WRITER_MODEL = "gpt-4o"` |
| Magic strings signaux | `<GO_BUILD>`, `<BUILD_DONE>`, `<HANDOFF_WRITER>` dans graph.py, master_node.py, builder_node.py |
| Bug substring | [agents/mortality/agents/graph.py:142](agents/mortality/agents/graph.py#L142) `"<WRITE_DONE" in content` sans `:` final (filets aussi `<WRITE_DONE_FOO>`) |
| State TypedDict | [agents/mortality/agents/state.py:17](agents/mortality/agents/state.py#L17) — seul `messages` a un reducer |
| Tool dispatch if/elif | [tools/tool_registry.py:340-376](tools/tool_registry.py#L340-L376) |
| Mapping result keys hardcodé | [agents/mortality/agents/tools_node.py:94-105](agents/mortality/agents/tools_node.py#L94-L105) `_RESULT_KEYS` |
| MemorySaver RAM-only | [agents/mortality/agents/graph.py:48](agents/mortality/agents/graph.py#L48) |
| Exception trop large | [agents/mortality/agents/graph.py:372-375](agents/mortality/agents/graph.py#L372-L375) |
| Coverage tests | ~3.3 % (49 fichiers tests / ~197k lignes Python) |

## Non-goals (volontairement hors scope)

- Réécriture du canvas (canvas_app.py — 2150 lignes Dash) : chantier UI séparé.
- Passage à LangGraph Send API pour parallélisation multi-agents : surdimensionné.
- Externalisation RAG (Qdrant/Pinecone) : embeddings locaux suffisent.
- Factorisation `mortality.compute_*` en `describe()` générique : dette tracée séparément.
- Migration providers LLM (OpenAI → Anthropic ou inverse) : préserver la flexibilité, pas changer de provider par défaut.

---

## File Structure — vue d'ensemble des changements

### Nouveaux fichiers à créer

```
agents/mortality/agents/
├── signals.py                  # NEW — enum RoutingSignal (Lot 2)
├── data_store_keys.py          # NEW — enum DataStoreKey (Lot 3)
├── exceptions.py               # NEW — hiérarchie AgentError (Lot 6)
├── base_agent.py               # NEW — BaseAgentNode (Lot 8)
└── observability.py            # NEW — structured logging + correlation_id (Lot 10)

tools/
├── registry/                   # NEW — package (Lot 7)
│   ├── __init__.py
│   ├── spec.py                 # ToolSpec dataclass + decorator @register_tool
│   ├── dispatcher.py           # metadata-driven dispatch
│   └── validators.py           # Pydantic input/output validation

agents/report/renderers/        # NEW — package (Lot 9)
├── __init__.py
├── base.py                     # ReportRenderer abstract
├── pdf_renderer.py             # déménagement _05_assemble.py
└── html_renderer.py            # stub avec test placeholder

tests/
├── conftest.py                 # NEW — fixtures partagées (Lot 13)
├── integration/                # NEW — E2E graph tests (Lot 14)
│   ├── __init__.py
│   ├── test_master_to_builder_e2e.py
│   ├── test_master_to_writer_e2e.py
│   └── test_rag_flow_e2e.py
└── failure_injection/          # NEW — failure tests (Lot 15)
    ├── __init__.py
    ├── test_llm_malformed_json.py
    └── test_tool_exceptions.py
```

### Fichiers à modifier (résumé)

- `agents/mortality/agents/state.py` — TypedDict → Pydantic v2 BaseModel (Lot 5)
- `agents/mortality/agents/graph.py` — bug `<WRITE_DONE` (Lot 2), routeurs propres (Lot 5), SqliteSaver (Lot 12)
- `agents/mortality/agents/master_node.py`, `builder_node.py`, `writer_node.py`, `rag_node.py` — hériter de BaseAgentNode (Lot 8)
- `agents/mortality/agents/tools_node.py` — _RESULT_KEYS dans registry (Lot 7)
- `agents/mortality/agents/_utils.py` — retry policy étendue (Lot 11)
- `agents/report/pipeline/_05_assemble.py` — vidé au profit de renderers/pdf_renderer.py (Lot 9)
- `agents/report/pipeline/run_pipeline.py` — sélection renderer via registry (Lot 9)
- `config.py` — suppression `WRITER_MODEL` (Lot 4)
- `config/llm_models.yaml` — ajout rôle `writer` (Lot 4)
- `tools/tool_registry.py` — vidé au profit de tools/registry/ (Lot 7)
- ~15 sites d'accès `data_store["..."]` → `data_store[DataStoreKey.X]` (Lot 3)

### Fichiers à supprimer

- `agents/mortality/agents/mortality_node.py` (Lot 1)
- `agents/mortality/agents/report_node.py` (Lot 1)
- `agents/mortality/writer_agent.py` (Lot 1)
- `OLD/` tout le dossier (Lot 1)

---

## Vue d'ensemble des 5 phases et 15 lots

| Phase | Lot | Titre | Risque | Effort |
|---|---|---|---|---|
| **1. Hygiène** | 1 | Suppression code mort | Très faible | 0.5 j |
|  | 2 | Enums signaux agentiques | Faible | 1 j |
|  | 3 | Enum clés data_store | Faible | 1 j |
| **2. Config & schémas** | 4 | Config LLM unifiée | Très faible | 0.5 j |
|  | 5 | AgentState → Pydantic v2 BaseModel | **Élevé** (pivot) | 3 j |
|  | 6 | Hiérarchie exceptions custom | Faible | 1 j |
| **3. Extensibilité** | 7 | Tool registry metadata-driven | Élevé | 4 j |
|  | 8 | BaseAgentNode abstrait + factory | Moyen | 3 j |
|  | 9 | Renderer registry pour rapports | Moyen | 3 j |
| **4. Robustesse** | 10 | Observability (structlog + correlation_id) | Faible | 2 j |
|  | 11 | Retry/circuit breaker agent-level | Moyen | 2 j |
|  | 12 | Checkpointer SqliteSaver persistant | Moyen | 1.5 j |
| **5. Tests** | 13 | conftest.py + fixtures | Très faible | 1 j |
|  | 14 | Tests intégration graph E2E | Moyen | 3 j |
|  | 15 | Failure injection | Faible | 2 j |

**Dépendances critiques** : Lots 1-4 indépendants → Lot 5 (pivot) → débloque 6, 7, 8 → Lot 8 débloque 11 → Lots 12, 13, 14, 15 parallélisables.

**Convention de commits** : Conventional Commits (`feat:`, `refactor:`, `fix:`, `chore:`, `test:`, `docs:`). Suffixe optionnel `(lot-N)` pour traçabilité.

---

# PHASE 1 — HYGIÈNE

## Lot 1 : Suppression du code mort

**Goal:** Retirer les 3 modules Python jamais importés et le dossier `OLD/` pour réduire la surface du codebase de ~370 lignes.

**Dependencies:** aucune.

**Files:**
- Delete: `agents/mortality/agents/mortality_node.py`
- Delete: `agents/mortality/agents/report_node.py`
- Delete: `agents/mortality/writer_agent.py`
- Delete: `OLD/` (récursif)
- Test: `tests/test_dead_code_removal.py` (NEW)

### Tasks

- [ ] **Step 1.1 : Écrire le test de non-importabilité**

Créer `tests/test_dead_code_removal.py` :

```python
"""Garantit que les modules supprimés en Lot 1 ne sont jamais réintroduits."""
import importlib
import pytest


@pytest.mark.parametrize("module_path", [
    "agents.mortality.agents.mortality_node",
    "agents.mortality.agents.report_node",
    "agents.mortality.writer_agent",
])
def test_dead_module_not_importable(module_path: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_path)


def test_old_directory_does_not_exist() -> None:
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    assert not (project_root / "OLD").exists(), "OLD/ doit avoir été supprimé"
```

- [ ] **Step 1.2 : Exécuter le test pour vérifier qu'il échoue**

```bash
pytest tests/test_dead_code_removal.py -v
```

Attendu : 4 FAILED (3 imports réussissent encore + OLD/ existe).

- [ ] **Step 1.3 : Vérifier qu'aucun fichier vivant n'importe ces modules**

```bash
grep -rn "mortality_node\|report_node\|writer_agent\b" \
  --include="*.py" \
  --exclude-dir=__pycache__ \
  --exclude-dir=.worktrees \
  --exclude-dir=OLD \
  --exclude-dir=tests \
  /Users/macbook14/Python_projects/AgentActuariat
```

Attendu : aucune occurrence d'import dans `agents/`, `tools/`, `session/`, `canvas_app.py`, `loader.py`, `run_pipeline.py`.

Si une occurrence est trouvée : **STOP**, ne pas supprimer, créer un sous-ticket pour traiter la dépendance.

- [ ] **Step 1.4 : Supprimer les fichiers**

```bash
rm /Users/macbook14/Python_projects/AgentActuariat/agents/mortality/agents/mortality_node.py
rm /Users/macbook14/Python_projects/AgentActuariat/agents/mortality/agents/report_node.py
rm /Users/macbook14/Python_projects/AgentActuariat/agents/mortality/writer_agent.py
rm -rf /Users/macbook14/Python_projects/AgentActuariat/OLD
```

- [ ] **Step 1.5 : Vérifier que le test passe et qu'aucune régression n'est introduite**

```bash
pytest tests/test_dead_code_removal.py -v
pytest tests/ -q
```

Attendu : `test_dead_code_removal.py` PASS, suite globale au même statut qu'avant.

- [ ] **Step 1.6 : Commit**

```bash
git add tests/test_dead_code_removal.py
git rm agents/mortality/agents/mortality_node.py
git rm agents/mortality/agents/report_node.py
git rm agents/mortality/writer_agent.py
git rm -r OLD/
git commit -m "chore(lot-1): remove dead code (mortality_node, report_node, writer_agent, OLD/)"
```

**Acceptance criteria:**
- 3 fichiers + dossier OLD/ supprimés.
- Test `test_dead_code_removal.py` PASS.
- Suite pytest globale sans régression.

---

## Lot 2 : Enums pour signaux agentiques

**Goal:** Remplacer toutes les strings XML de routing (`<GO_BUILD>`, `<BUILD_DONE>`, `<WRITE_DONE>`, etc.) par un enum centralisé typé. Corrige aussi le bug substring `"<WRITE_DONE"` (sans `:`) à graph.py:142 qui matche faussement.

**Dependencies:** Lot 1 (les modules morts pouvaient référencer ces signaux).

**Files:**
- Create: `agents/mortality/agents/signals.py`
- Modify: `agents/mortality/agents/graph.py` (lignes 98, 100, 142)
- Modify: `agents/mortality/agents/master_node.py` (toutes occurrences `<GO_*>`, `<*_DONE>`)
- Modify: `agents/mortality/agents/builder_node.py` (toutes occurrences `<BUILD_DONE>`, `<HANDOFF_WRITER>`, `<NEED_DATA>`)
- Modify: `agents/mortality/agents/writer_node.py` (toutes occurrences `<WRITE_DONE>`, `<NEED_DATA>`)
- Modify: `agents/mortality/agents/rag_node.py` (toutes occurrences `<RAG_DONE>`)
- Test: `tests/test_signals.py` (NEW)

### Tasks

- [ ] **Step 2.1 : Inventaire exhaustif des signaux existants**

```bash
grep -rn "<[A-Z_]\+\(:[^>]*\)\?>" \
  --include="*.py" \
  --exclude-dir=__pycache__ \
  /Users/macbook14/Python_projects/AgentActuariat/agents/ \
  > /tmp/signals_inventory.txt
cat /tmp/signals_inventory.txt
```

Lister tous les tokens `<XXX>` ou `<XXX:...>` trouvés. Confirmer qu'on a au moins : `GO_BUILD`, `GO_WRITE`, `GO_RAG`, `BUILD_DONE`, `WRITE_DONE`, `NEED_DATA`, `RAG_DONE`, `HANDOFF_WRITER`, `MODEL_CHOICE_CHECKPOINT`, `ROUTE:`. Ajouter à l'enum toute string XML trouvée.

- [ ] **Step 2.2 : Écrire le test du module signals**

Créer `tests/test_signals.py` :

```python
"""Test du module agents.mortality.agents.signals."""
import pytest

from agents.mortality.agents.signals import RoutingSignal, has_signal, extract_payload


def test_routing_signal_enum_values() -> None:
    """Vérifie que tous les signaux historiques sont présents."""
    assert RoutingSignal.GO_BUILD.value == "<GO_BUILD>"
    assert RoutingSignal.GO_WRITE.value == "<GO_WRITE>"
    assert RoutingSignal.GO_RAG.value == "<GO_RAG>"
    assert RoutingSignal.BUILD_DONE.value == "<BUILD_DONE>"
    assert RoutingSignal.WRITE_DONE.value == "<WRITE_DONE>"
    assert RoutingSignal.RAG_DONE.value == "<RAG_DONE>"
    assert RoutingSignal.NEED_DATA.value == "<NEED_DATA>"
    assert RoutingSignal.HANDOFF_WRITER.value == "<HANDOFF_WRITER>"
    assert RoutingSignal.MODEL_CHOICE_CHECKPOINT.value == "<MODEL_CHOICE_CHECKPOINT>"


def test_has_signal_exact_match() -> None:
    """has_signal doit reconnaître un signal sans payload."""
    assert has_signal("Texte avant <BUILD_DONE> texte après", RoutingSignal.BUILD_DONE)


def test_has_signal_with_payload() -> None:
    """has_signal doit reconnaître un signal avec payload (notation <SIGNAL: ...>)."""
    assert has_signal("Rapport ok <WRITE_DONE: /tmp/r.pdf>", RoutingSignal.WRITE_DONE)


def test_has_signal_does_not_match_substring() -> None:
    """Le bug historique graph.py:142 (`<WRITE_DONE` sans `:`) matchait à tort
    `<WRITE_DONE_FOO>`. has_signal NE doit PAS matcher ça."""
    assert not has_signal("Texte <WRITE_DONE_FOO>", RoutingSignal.WRITE_DONE)


def test_has_signal_not_found() -> None:
    assert not has_signal("Aucun signal ici", RoutingSignal.BUILD_DONE)


def test_extract_payload_present() -> None:
    """extract_payload retourne la portion après ':' jusqu'à '>'."""
    assert extract_payload(
        "Texte <WRITE_DONE: /tmp/rapport.pdf>",
        RoutingSignal.WRITE_DONE,
    ) == "/tmp/rapport.pdf"


def test_extract_payload_absent() -> None:
    """Si le signal est présent sans payload, extract_payload retourne ''."""
    assert extract_payload("Texte <BUILD_DONE>", RoutingSignal.BUILD_DONE) == ""


def test_extract_payload_signal_missing() -> None:
    """Si le signal est totalement absent, extract_payload retourne None."""
    assert extract_payload("Rien ici", RoutingSignal.WRITE_DONE) is None
```

- [ ] **Step 2.3 : Exécuter le test pour vérifier qu'il échoue**

```bash
pytest tests/test_signals.py -v
```

Attendu : ImportError (`agents.mortality.agents.signals` n'existe pas).

- [ ] **Step 2.4 : Implémenter `signals.py`**

Créer `agents/mortality/agents/signals.py` :

```python
"""
agents/mortality/agents/signals.py
Énumération centralisée des signaux XML de routing entre agents.

Pourquoi : avant ce module, chaque nœud comparait du texte par substring
(`"<BUILD_DONE>" in content`), ce qui (a) duplique les strings dans 6+ fichiers
et (b) introduit des faux positifs (cf. bug graph.py:142 où `"<WRITE_DONE" in c`
matchait `<WRITE_DONE_FOO>`).

Utilisation :
    from agents.mortality.agents.signals import RoutingSignal, has_signal, extract_payload

    if has_signal(text, RoutingSignal.WRITE_DONE):
        path = extract_payload(text, RoutingSignal.WRITE_DONE)
"""
from __future__ import annotations

import re
from enum import StrEnum


class RoutingSignal(StrEnum):
    """Signaux XML émis par les nodes pour piloter le routing du graph."""
    GO_BUILD = "<GO_BUILD>"
    GO_WRITE = "<GO_WRITE>"
    GO_RAG = "<GO_RAG>"
    BUILD_DONE = "<BUILD_DONE>"
    WRITE_DONE = "<WRITE_DONE>"
    RAG_DONE = "<RAG_DONE>"
    NEED_DATA = "<NEED_DATA>"
    HANDOFF_WRITER = "<HANDOFF_WRITER>"
    MODEL_CHOICE_CHECKPOINT = "<MODEL_CHOICE_CHECKPOINT>"


def _pattern_for(signal: RoutingSignal) -> re.Pattern[str]:
    """Construit un pattern qui matche soit <SIGNAL>, soit <SIGNAL: payload>.

    Important : on échappe `<` et `>` pour éviter les faux positifs sur
    `<SIGNAL_FOO>` (pattern strict sur le nom du signal).
    """
    name = signal.value.strip("<>")
    return re.compile(rf"<{re.escape(name)}(?::\s*([^>]*))?>")


def has_signal(text: str, signal: RoutingSignal) -> bool:
    """Retourne True si le signal exact est présent dans le texte."""
    return _pattern_for(signal).search(text) is not None


def extract_payload(text: str, signal: RoutingSignal) -> str | None:
    """Extrait la portion après `:` dans `<SIGNAL: payload>`.

    Retourne :
        - None si le signal n'est pas présent
        - "" si le signal est présent sans payload (`<SIGNAL>`)
        - la string payload trimée sinon
    """
    match = _pattern_for(signal).search(text)
    if match is None:
        return None
    payload = match.group(1)
    return (payload or "").strip()
```

- [ ] **Step 2.5 : Exécuter le test du module signals**

```bash
pytest tests/test_signals.py -v
```

Attendu : 8 PASS.

- [ ] **Step 2.6 : Remplacer les usages dans `graph.py`**

Modifier `agents/mortality/agents/graph.py` :

- Importer en tête : `from agents.mortality.agents.signals import RoutingSignal, has_signal`
- Ligne 98 : `if "<BUILD_DONE>" in content or "<HANDOFF_WRITER>" in content:` →
  ```python
  if has_signal(content, RoutingSignal.BUILD_DONE) or has_signal(content, RoutingSignal.HANDOFF_WRITER):
  ```
- Ligne 100 : `if "<MODEL_CHOICE_CHECKPOINT>" in content:` →
  ```python
  if has_signal(content, RoutingSignal.MODEL_CHOICE_CHECKPOINT):
  ```
- Ligne 142 (**BUG FIX**) : `if "<WRITE_DONE" in content or "<NEED_DATA" in content:` →
  ```python
  if has_signal(content, RoutingSignal.WRITE_DONE) or has_signal(content, RoutingSignal.NEED_DATA):
  ```

- [ ] **Step 2.7 : Remplacer les usages dans `master_node.py`, `builder_node.py`, `writer_node.py`, `rag_node.py`**

Pour chaque fichier :

1. Ajouter `from agents.mortality.agents.signals import RoutingSignal, has_signal, extract_payload` en tête.
2. Remplacer toutes les strings littérales `"<XXX>"` et toutes les comparaisons substring `"<XXX>" in content` par les helpers du module.
3. Quand un signal est **émis** par le LLM (instruction dans un system prompt), conserver la string littérale dans le prompt — l'enum sert à la **détection côté Python**, pas à la génération côté LLM.

Exemple `builder_node.py` (suppose une ligne du type `if "<BUILD_DONE>" in last.content:`) :

```python
# avant
if "<BUILD_DONE>" in last.content:
    ...
# après
if has_signal(last.content, RoutingSignal.BUILD_DONE):
    ...
```

- [ ] **Step 2.8 : Lancer la suite complète et vérifier l'absence de régression**

```bash
pytest tests/ -q
```

Attendu : aucun nouveau FAIL. Le bug `<WRITE_DONE` doit être corrigé (les tests qui passaient continuent de passer ; aucun test ne s'appuyait sur le faux positif).

- [ ] **Step 2.9 : Vérification grep finale**

```bash
grep -rn '"<\(GO_\|BUILD_DONE\|WRITE_DONE\|RAG_DONE\|NEED_DATA\|HANDOFF_WRITER\|MODEL_CHOICE_CHECKPOINT\)' \
  --include="*.py" \
  --exclude-dir=__pycache__ \
  --exclude-dir=tests \
  /Users/macbook14/Python_projects/AgentActuariat/agents/mortality/agents/
```

Attendu : aucune occurrence (tous les signaux sont désormais consommés via l'enum).

Cas autorisés (ne pas modifier) :
- Strings dans les **system prompts** servant à instruire le LLM à émettre le signal (souvent dans `loader.py` ou les `.md` de prompt).
- Strings dans les **commentaires** et docstrings.

- [ ] **Step 2.10 : Commit**

```bash
git add agents/mortality/agents/signals.py
git add agents/mortality/agents/graph.py
git add agents/mortality/agents/master_node.py
git add agents/mortality/agents/builder_node.py
git add agents/mortality/agents/writer_node.py
git add agents/mortality/agents/rag_node.py
git add tests/test_signals.py
git commit -m "refactor(lot-2): centralize routing signals in RoutingSignal enum

- Add agents/mortality/agents/signals.py with strict regex matching
- Fix substring bug at graph.py:142 ('<WRITE_DONE' matched '<WRITE_DONE_FOO>')
- Replace all literal signal strings with RoutingSignal + has_signal/extract_payload"
```

**Acceptance criteria:**
- `tests/test_signals.py` : 8 PASS.
- Suite pytest globale sans régression.
- Aucune string `"<BUILD_DONE>"` (ou autres signaux) consommée par `in` dans le code Python applicatif (hors prompts/docstrings).
- Bug `<WRITE_DONE` substring corrigé.

---

## Lot 3 : Enum pour clés `data_store`

**Goal:** Remplacer les ~15 clés magiques accédées comme strings dans `data_store` par un enum typé pour éviter les typos silencieuses et faciliter le refactoring futur.

**Dependencies:** Lot 1.

**Files:**
- Create: `agents/mortality/agents/data_store_keys.py`
- Modify: ~15 sites d'accès à identifier via grep (cf. Step 3.1)
- Test: `tests/test_data_store_keys.py` (NEW)

### Tasks

- [ ] **Step 3.1 : Inventaire des clés `data_store`**

```bash
grep -rn 'data_store\(\[\|\.get(\|\.pop(\|\.setdefault(\)' \
  --include="*.py" \
  --exclude-dir=__pycache__ \
  --exclude-dir=tests \
  /Users/macbook14/Python_projects/AgentActuariat/agents/ \
  /Users/macbook14/Python_projects/AgentActuariat/canvas_app.py \
  > /tmp/data_store_inventory.txt

grep -rn 'data\[\|data\.get(\|data\.setdefault(' \
  --include="*.py" \
  --exclude-dir=__pycache__ \
  --exclude-dir=tests \
  /Users/macbook14/Python_projects/AgentActuariat/tools/ \
  >> /tmp/data_store_inventory.txt

cat /tmp/data_store_inventory.txt
```

Lister chaque clé string distincte trouvée. Confirmer au minimum la présence de : `_builder_turns`, `_call_log`, `_initial_active_agent`, `_stage_buffer`, `study_plan`, `csv_filename`, `session_id`, `_writer_need_data_prev`, `summary`, `ages`, `series`, `segmentation`, `exposure_table`, `qx_table`, `smoothed_table`, `diagnostics`, `validation`, `benchmarking`. Tout autre clé trouvée doit être ajoutée à l'enum.

- [ ] **Step 3.2 : Écrire le test**

Créer `tests/test_data_store_keys.py` :

```python
"""Test du module agents.mortality.agents.data_store_keys."""
from agents.mortality.agents.data_store_keys import DataStoreKey


def test_metadata_keys_present() -> None:
    """Clés techniques (préfixées `_`) utilisées par les nodes pour leur logique."""
    assert DataStoreKey.BUILDER_TURNS.value == "_builder_turns"
    assert DataStoreKey.CALL_LOG.value == "_call_log"
    assert DataStoreKey.INITIAL_ACTIVE_AGENT.value == "_initial_active_agent"
    assert DataStoreKey.STAGE_BUFFER.value == "_stage_buffer"
    assert DataStoreKey.WRITER_NEED_DATA_PREV.value == "_writer_need_data_prev"


def test_business_keys_present() -> None:
    """Clés métier exposées au validator/template."""
    assert DataStoreKey.STUDY_PLAN.value == "study_plan"
    assert DataStoreKey.CSV_FILENAME.value == "csv_filename"
    assert DataStoreKey.SESSION_ID.value == "session_id"


def test_tool_result_keys_present() -> None:
    """Clés de résultats produits par les tools (cf. tools_node._RESULT_KEYS)."""
    expected = {
        "summary", "ages", "series", "segmentation",
        "exposure_table", "qx_table", "smoothed_table",
        "diagnostics", "validation", "benchmarking",
    }
    actual = {key.value for key in DataStoreKey}
    assert expected.issubset(actual), f"Clés manquantes : {expected - actual}"


def test_str_enum_compat() -> None:
    """DataStoreKey doit être un StrEnum pour permettre data_store[KEY] direct."""
    d: dict = {}
    d[DataStoreKey.STUDY_PLAN] = {"foo": "bar"}
    # Accès via la string brute doit aussi marcher (compat backward)
    assert d["study_plan"] == {"foo": "bar"}
```

- [ ] **Step 3.3 : Run pour vérifier failure**

```bash
pytest tests/test_data_store_keys.py -v
```

Attendu : ImportError.

- [ ] **Step 3.4 : Implémenter `data_store_keys.py`**

Créer `agents/mortality/agents/data_store_keys.py` :

```python
"""
agents/mortality/agents/data_store_keys.py
Énumération centralisée des clés autorisées dans le `data_store` partagé
entre les nodes du graphe LangGraph.

StrEnum : permet `data_store[DataStoreKey.STUDY_PLAN]` ET `data_store["study_plan"]`
(rétro-compatibilité durant la migration).
"""
from __future__ import annotations

from enum import StrEnum


class DataStoreKey(StrEnum):
    # ── Clés techniques (préfixées `_`) ───────────────────────────────────────
    BUILDER_TURNS = "_builder_turns"
    CALL_LOG = "_call_log"
    INITIAL_ACTIVE_AGENT = "_initial_active_agent"
    STAGE_BUFFER = "_stage_buffer"
    WRITER_NEED_DATA_PREV = "_writer_need_data_prev"

    # ── Clés métier ───────────────────────────────────────────────────────────
    STUDY_PLAN = "study_plan"
    CSV_FILENAME = "csv_filename"
    SESSION_ID = "session_id"

    # ── Résultats de tools (cf. tools_node._RESULT_KEYS) ──────────────────────
    SUMMARY = "summary"
    AGES = "ages"
    SERIES = "series"
    SEGMENTATION = "segmentation"
    EXPOSURE_TABLE = "exposure_table"
    QX_TABLE = "qx_table"
    SMOOTHED_TABLE = "smoothed_table"
    DIAGNOSTICS = "diagnostics"
    VALIDATION = "validation"
    BENCHMARKING = "benchmarking"

    # Ajouter ici toute nouvelle clé identifiée par Step 3.1
```

- [ ] **Step 3.5 : Run test enum**

```bash
pytest tests/test_data_store_keys.py -v
```

Attendu : 4 PASS.

- [ ] **Step 3.6 : Migration progressive des sites d'accès**

Pour chaque fichier listé dans `/tmp/data_store_inventory.txt` :

1. Ajouter `from agents.mortality.agents.data_store_keys import DataStoreKey` en tête.
2. Remplacer `data_store["foo"]` par `data_store[DataStoreKey.FOO]`.
3. Conserver les strings littérales pour les accès qui pourraient être étendus par des tools non encore listés (ex : keys préfixées par tool_name). Si ambigu, **laisser la string** et créer un sous-ticket.

**Règle de migration** : on ne touche **PAS** :
- Les serializations JSON/Parquet (ex : `mm.to_data_store()` qui retourne un dict avec des strings keys).
- Les LLM prompts qui décrivent le `data_store` (les strings doivent rester lisibles humainement).

- [ ] **Step 3.7 : Test de régression**

```bash
pytest tests/ -q
```

Attendu : aucun nouveau FAIL.

- [ ] **Step 3.8 : Commit**

```bash
git add agents/mortality/agents/data_store_keys.py
git add tests/test_data_store_keys.py
# + tous les fichiers modifiés
git commit -m "refactor(lot-3): centralize data_store keys in DataStoreKey enum"
```

**Acceptance criteria:**
- Enum `DataStoreKey` couvre toutes les clés identifiées dans Step 3.1.
- Sites d'accès migrés vers l'enum (sauf serialization et prompts).
- Tests sans régression.

---

# PHASE 2 — CONFIG & SCHÉMAS

## Lot 4 : Config LLM unifiée — suppression de `WRITER_MODEL` hardcodé

**Goal:** Faire passer le WriterAgent par le même mécanisme que master/builder/rag (`config/llm_models.yaml` + env override) en supprimant `WRITER_MODEL` hardcodé dans `config.py:7`.

**Dependencies:** aucune (peut paralléliser avec Lots 1-3).

**Files:**
- Modify: `config.py` (suppression de `WRITER_MODEL`)
- Modify: `config/llm_models.yaml` (ajout rôle `writer`)
- Modify: tous fichiers qui importent `WRITER_MODEL` (à identifier via grep)
- Modify: `agents/mortality/agents/llm_config.py` si nécessaire pour exposer le rôle `writer`
- Test: `tests/test_llm_config.py` (étendre)

### Tasks

- [ ] **Step 4.1 : Identifier les consommateurs de `WRITER_MODEL`**

```bash
grep -rn "WRITER_MODEL\|from config import\|import config" \
  --include="*.py" \
  --exclude-dir=__pycache__ \
  /Users/macbook14/Python_projects/AgentActuariat
```

Lister chaque site (typiquement dans `agents/report/pipeline/_04_redaction.py` et/ou `_03_completion_plan.py`).

- [ ] **Step 4.2 : Lire la structure actuelle de `config/llm_models.yaml`**

```bash
cat /Users/macbook14/Python_projects/AgentActuariat/config/llm_models.yaml
```

Identifier le format (probablement `roles: {master: {...}, builder: {...}, rag: {...}}`).

- [ ] **Step 4.3 : Écrire le test**

Étendre `tests/test_llm_config.py` :

```python
def test_writer_role_config_loaded() -> None:
    """Le rôle 'writer' doit être chargé depuis llm_models.yaml."""
    from agents.mortality.agents.llm_config import get_llm_config
    cfg = get_llm_config("writer")
    assert cfg is not None
    assert cfg.get("model") is not None, "writer role doit avoir un champ 'model'"


def test_writer_env_override() -> None:
    """LLM_MODEL_WRITER doit override la valeur YAML."""
    import os
    from agents.mortality.agents.llm_config import get_llm_config
    os.environ["LLM_MODEL_WRITER"] = "gpt-test-override"
    try:
        cfg = get_llm_config("writer")
        assert cfg["model"] == "gpt-test-override"
    finally:
        del os.environ["LLM_MODEL_WRITER"]


def test_writer_model_constant_removed() -> None:
    """config.py ne doit plus exporter WRITER_MODEL."""
    import config
    assert not hasattr(config, "WRITER_MODEL"), \
        "WRITER_MODEL doit être supprimé de config.py (migration vers llm_models.yaml)"
```

- [ ] **Step 4.4 : Run pour vérifier failure**

```bash
pytest tests/test_llm_config.py -v
```

Attendu : 3 nouveaux FAIL (les helpers writer n'existent pas + WRITER_MODEL existe encore).

- [ ] **Step 4.5 : Ajouter le rôle `writer` dans `config/llm_models.yaml`**

Ajouter sous `roles:` (en respectant le format existant) :

```yaml
roles:
  # ... existant ...
  writer:
    model: gpt-4o
    temperature: 0.2
    max_tokens: 4096
    # Override env : LLM_MODEL_WRITER, LLM_TEMPERATURE_WRITER, etc.
```

La valeur `gpt-4o` est conservée pour iso-fonctionnel.

- [ ] **Step 4.6 : Vérifier que `get_llm_config("writer")` fonctionne déjà**

Lire `agents/mortality/agents/llm_config.py` pour s'assurer que la fonction est générique (lit n'importe quel rôle YAML). Si ce n'est pas le cas, ajuster pour que le rôle `writer` soit traité comme master/builder.

- [ ] **Step 4.7 : Migrer les consommateurs de `WRITER_MODEL`**

Pour chaque site identifié en Step 4.1 :

```python
# avant
from config import WRITER_MODEL
model = WRITER_MODEL

# après
from agents.mortality.agents.llm_config import get_llm_config
model = get_llm_config("writer")["model"]
```

Si le consommateur a besoin de plus (temperature, max_tokens), utiliser le dict complet retourné par `get_llm_config`.

- [ ] **Step 4.8 : Supprimer `WRITER_MODEL` de `config.py`**

Édition de `config.py` :

```python
"""
config.py
Configuration de l'agent actuariel.
"""

# Chemins
UPLOADS_DIR = "./uploads"
NOTEBOOKS_DIR = "./notebooks"
```

(La ligne `WRITER_MODEL = "gpt-4o"` est retirée.)

- [ ] **Step 4.9 : Run tests**

```bash
pytest tests/test_llm_config.py -v
pytest tests/ -q
```

Attendu : 3 nouveaux PASS, aucune régression.

- [ ] **Step 4.10 : Commit**

```bash
git add config.py config/llm_models.yaml
git add tests/test_llm_config.py
# + sites consommateurs modifiés
git commit -m "refactor(lot-4): unify writer LLM config via llm_models.yaml

- Add 'writer' role to config/llm_models.yaml
- Remove hardcoded WRITER_MODEL from config.py
- Migrate consumers to get_llm_config('writer')
- Support env override LLM_MODEL_WRITER"
```

**Acceptance criteria:**
- `WRITER_MODEL` absent de `config.py`.
- `get_llm_config("writer")` retourne `gpt-4o` par défaut.
- `LLM_MODEL_WRITER=foo` override fonctionne.
- Tests sans régression.

---

## Lot 5 : AgentState → Pydantic v2 BaseModel (PIVOT)

**Goal:** Migrer `AgentState` de TypedDict vers Pydantic v2 BaseModel avec reducers explicites sur tous les champs mutables, validators sur `active_agent` (Literal), invariants sur `plan_established` vs `study_plan`. Lot pivot : débloque Lots 6, 7, 8, 11.

**Dependencies:** Lots 1, 2, 3.

**⚠️ Risque élevé** : touche le cœur du graph. Procéder en 2 sous-temps :
- 5A : nouveau modèle Pydantic en parallèle, conversion bidirectionnelle TypedDict↔BaseModel
- 5B : migration des nodes pour consommer le BaseModel directement

**Files:**
- Modify: `agents/mortality/agents/state.py` (réécriture)
- Create: `tests/test_state.py` (NEW)
- Modify: `agents/mortality/agents/graph.py` (signature `StateGraph(AgentState)`)
- Modify: `agents/mortality/agents/master_node.py`, `builder_node.py`, `writer_node.py`, `rag_node.py`, `tools_node.py` (accès attribute au lieu de dict)

### Tasks (résumé — détails à expanser avant exécution)

- [ ] **Step 5.1 : Vérifier compatibilité LangGraph 1.x avec Pydantic BaseModel**

Consulter la doc LangGraph 1.x : `StateGraph(AgentState)` supporte les BaseModel Pydantic v2 nativement depuis LangGraph 0.2.40+. Confirmer la version dans `requirements.txt`.

```bash
grep -i "langgraph" /Users/macbook14/Python_projects/AgentActuariat/requirements.txt
python -c "import langgraph; print(langgraph.__version__)"
```

Si version < 0.2.40 : **STOP**, créer un sous-ticket "bump langgraph" en pré-requis.

- [ ] **Step 5.2 : Écrire les tests de spec du nouveau `AgentState`**

Créer `tests/test_state.py` :

```python
"""Test du nouveau AgentState Pydantic v2."""
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import ValidationError

from agents.mortality.agents.state import AgentState


def test_state_default_construction() -> None:
    """AgentState doit pouvoir être instancié avec valeurs par défaut."""
    state = AgentState()
    assert state.messages == []
    assert state.data_store == {}
    assert state.context_docs == []
    assert state.plan_established is False
    assert state.active_agent == "master"
    assert state.events == []
    assert state.step_by_step is False
    assert state.pending_tool_call is None
    assert state.dataset_ref is None


def test_active_agent_validates_enum() -> None:
    """active_agent doit être une valeur autorisée."""
    AgentState(active_agent="master")
    AgentState(active_agent="builder")
    AgentState(active_agent="writer")
    AgentState(active_agent="rag")
    with pytest.raises(ValidationError):
        AgentState(active_agent="hacker")


def test_messages_reducer_appends() -> None:
    """Le reducer add_messages doit accumuler les messages."""
    from langgraph.graph.message import add_messages
    s1 = AgentState(messages=[HumanMessage(content="bonjour")])
    delta = {"messages": [AIMessage(content="salut")]}
    merged = add_messages(s1.messages, delta["messages"])
    assert len(merged) == 2


def test_data_store_arbitrary_types() -> None:
    """data_store doit accepter des valeurs hétérogènes (pandas DataFrame, dicts...)."""
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2]})
    state = AgentState(data_store={"foo": df, "bar": {"nested": True}})
    assert isinstance(state.data_store["foo"], pd.DataFrame)


def test_plan_established_invariant() -> None:
    """Si plan_established=True, study_plan doit être présent dans data_store.

    (validator model_validator(mode='after'))
    """
    # OK : plan_established=False, pas de contrainte
    AgentState(plan_established=False, data_store={})
    # OK : plan_established=True ET study_plan présent
    AgentState(plan_established=True, data_store={"study_plan": {"foo": "bar"}})
    # KO : plan_established=True SANS study_plan
    with pytest.raises(ValidationError):
        AgentState(plan_established=True, data_store={})


def test_serializable_for_checkpointer() -> None:
    """State doit pouvoir être sérialisé par LangGraph MemorySaver (msgpack)."""
    state = AgentState(
        data_store={"foo": "bar"},
        active_agent="builder",
        events=[{"type": "agent_switch", "agent": "BuilderAgent"}],
    )
    dumped = state.model_dump()
    assert dumped["active_agent"] == "builder"
    assert dumped["data_store"] == {"foo": "bar"}
```

- [ ] **Step 5.3 : Run test pour failure**

```bash
pytest tests/test_state.py -v
```

Attendu : les tests échouent (AgentState est encore TypedDict).

- [ ] **Step 5.4 : Réécrire `state.py` en BaseModel**

Remplacer le contenu de `agents/mortality/agents/state.py` :

```python
"""
agents/mortality/agents/state.py
État partagé entre les nœuds du graphe LangGraph.

Pydantic v2 BaseModel — fournit validation runtime, reducers explicites,
invariants déclaratifs. Compatible LangGraph StateGraph depuis 0.2.40+.

La mémoire de l'agent est gérée par MemorySaver (checkpointer) via thread_id.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, model_validator


ActiveAgent = Literal["master", "builder", "writer", "rag", "calculation"]


class AgentState(BaseModel):
    """État partagé entre nœuds. Reducers explicites pour parallélisation future."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # accepte pd.DataFrame, etc.
        extra="forbid",                # rejette les clés inconnues
    )

    # ── Conversation ──────────────────────────────────────────────────────────
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)

    # ── Données portefeuille ──────────────────────────────────────────────────
    dataset_ref: Optional[str] = None
    data_store: Dict[str, Any] = Field(default_factory=dict)
    context_docs: List[Any] = Field(default_factory=list)

    # ── État de la session ────────────────────────────────────────────────────
    plan_established: bool = False
    active_agent: ActiveAgent = "master"

    # ── Interface canvas (events streaming) ───────────────────────────────────
    events: List[Any] = Field(default_factory=list)
    step_by_step: bool = False
    pending_tool_call: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _check_plan_invariant(self) -> "AgentState":
        """Si plan_established=True, study_plan doit exister dans data_store."""
        if self.plan_established and "study_plan" not in self.data_store:
            raise ValueError(
                "Invariant violé : plan_established=True requiert "
                "data_store['study_plan'] présent."
            )
        return self
```

- [ ] **Step 5.5 : Run tests state**

```bash
pytest tests/test_state.py -v
```

Attendu : 6 PASS.

- [ ] **Step 5.6 : Lancer la suite globale pour détecter les ruptures**

```bash
pytest tests/ -q
```

Attendu : **probablement plusieurs FAIL** liés à des accès `state["messages"]` (style dict) qui ne marchent plus avec BaseModel.

- [ ] **Step 5.7 : Migration des nodes — accès par attribut**

Pour chaque node (`master_node.py`, `builder_node.py`, `writer_node.py`, `rag_node.py`, `tools_node.py`) :

1. Remplacer `state["messages"]` par `state.messages`, `state["data_store"]` par `state.data_store`, etc.
2. Remplacer `state.get("foo", default)` par `state.foo if state.foo else default` ou utiliser `getattr` avec default.
3. **Pas** changer le format de retour : les nodes continuent de retourner un `dict` (LangGraph applique les reducers sur le dict retourné). LangGraph accepte un dict comme delta même quand le state est BaseModel.

Exemple `builder_node.py` :

```python
# avant
messages = state.get("messages") or []
data_store = state.get("data_store") or {}

# après
messages = state.messages
data_store = state.data_store
```

- [ ] **Step 5.8 : Migration des routers dans `graph.py`**

Les fonctions `_router`, `_should_continue_master`, `_should_continue_builder`, `_should_continue_writer`, `_should_continue_rag` reçoivent désormais un `AgentState` BaseModel. Migrer :

```python
# avant
def _router(state: AgentState) -> str:
    agent = state.get("active_agent", "master")
    ...

# après
def _router(state: AgentState) -> str:
    agent = state.active_agent
    ...
```

- [ ] **Step 5.9 : Migration des wrappers `_master_node_w` etc.**

Vérifier que les wrappers n'utilisent pas d'accès style dict.

- [ ] **Step 5.10 : Migration de `stream_agent` (input_state)**

Dans `stream_agent`, l'`input_state` est actuellement un `dict`. Le passer à `graph.stream(input_state, ...)` continue de fonctionner (LangGraph convertit dict → BaseModel à l'entrée). Aucun changement nécessaire ici, mais vérifier par test.

- [ ] **Step 5.11 : Run suite complète**

```bash
pytest tests/ -q
```

Attendu : tous les tests qui passaient avant doivent repasser. Nouveaux PASS sur `test_state.py`.

- [ ] **Step 5.12 : Commit**

```bash
git add agents/mortality/agents/state.py
git add agents/mortality/agents/graph.py
git add agents/mortality/agents/*_node.py
git add tests/test_state.py
git commit -m "refactor(lot-5): migrate AgentState TypedDict -> Pydantic v2 BaseModel

- Add runtime validation (active_agent Literal, plan_established invariant)
- Migrate nodes from state.get() to attribute access
- Compatible with LangGraph StateGraph since 0.2.40+
- Reducer add_messages preserved for messages field"
```

**Acceptance criteria:**
- `AgentState` est un Pydantic BaseModel avec `model_config` adéquat.
- Validators : `active_agent` enum, invariant `plan_established` ⇒ `study_plan`.
- Tous les nodes accèdent à l'état via attribute (`state.messages`), pas dict (`state["messages"]`).
- Suite pytest globale verte.

---

## Lot 6 : Hiérarchie d'exceptions custom

**Goal:** Remplacer les `except Exception` larges par une hiérarchie typée permettant le retry ciblé (Lot 11) et la traçabilité.

**Dependencies:** Lot 5 (utilise des champs Pydantic dans les exceptions).

**Files:**
- Create: `agents/mortality/agents/exceptions.py`
- Modify: `agents/mortality/agents/graph.py:372-375` (except dans stream_agent)
- Modify: `agents/mortality/agents/_utils.py` (retry)
- Modify: `tools/tool_registry.py:378+` (except sur run())
- Modify: `agents/report/pipeline/_04_redaction.py:41-46`
- Test: `tests/test_exceptions.py` (NEW)

### Tasks

- [ ] **Step 6.1 : Écrire le test**

Créer `tests/test_exceptions.py` :

```python
"""Test de la hiérarchie d'exceptions custom."""
import pytest

from agents.mortality.agents.exceptions import (
    AgentError,
    ToolError,
    LLMError,
    StateError,
    LLMRateLimitError,
    LLMTimeoutError,
    ToolValidationError,
    ToolExecutionError,
)


def test_hierarchy() -> None:
    assert issubclass(ToolError, AgentError)
    assert issubclass(LLMError, AgentError)
    assert issubclass(StateError, AgentError)
    assert issubclass(LLMRateLimitError, LLMError)
    assert issubclass(LLMTimeoutError, LLMError)
    assert issubclass(ToolValidationError, ToolError)
    assert issubclass(ToolExecutionError, ToolError)


def test_agent_error_carries_context() -> None:
    """Toute AgentError porte un contexte structuré."""
    try:
        raise ToolExecutionError(
            "builder.exposure failed",
            tool_name="builder.exposure",
            params={"df_size": 1000},
        )
    except AgentError as exc:
        assert exc.context["tool_name"] == "builder.exposure"
        assert exc.context["params"]["df_size"] == 1000
        assert "builder.exposure failed" in str(exc)


def test_llm_rate_limit_is_retryable() -> None:
    exc = LLMRateLimitError("429 received", provider="openai")
    assert exc.is_retryable is True


def test_tool_validation_error_is_not_retryable() -> None:
    exc = ToolValidationError("Invalid params", tool_name="builder.foo")
    assert exc.is_retryable is False
```

- [ ] **Step 6.2 : Run pour failure**

```bash
pytest tests/test_exceptions.py -v
```

Attendu : ImportError.

- [ ] **Step 6.3 : Implémenter `exceptions.py`**

Créer `agents/mortality/agents/exceptions.py` :

```python
"""
agents/mortality/agents/exceptions.py
Hiérarchie d'exceptions custom pour AgentActuariat.

Pourquoi : permet le retry ciblé (Lot 11), la traçabilité (correlation_id, Lot 10),
et la distinction entre erreurs transitoires (rate limit, timeout) vs erreurs
fatales (validation, état corrompu).
"""
from __future__ import annotations

from typing import Any


class AgentError(Exception):
    """Base de toutes les exceptions levées par les agents.

    Attributs:
        context: dict structuré attaché à l'erreur (logguable en JSON).
        is_retryable: True si l'opération peut être réessayée (rate limit,
            timeout réseau). False si l'erreur est déterministe (validation,
            état corrompu).
    """
    is_retryable: bool = False

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = context


# ── Famille LLM ──────────────────────────────────────────────────────────────

class LLMError(AgentError):
    """Erreur provenant d'un appel à un provider LLM (OpenAI/Anthropic)."""


class LLMRateLimitError(LLMError):
    """429 / quota dépassé. Retryable avec backoff."""
    is_retryable = True


class LLMTimeoutError(LLMError):
    """Timeout réseau côté LLM. Retryable."""
    is_retryable = True


class LLMMalformedResponseError(LLMError):
    """JSON malformé ou structure inattendue dans la réponse LLM. Non retryable
    (déterministe si même prompt)."""
    is_retryable = False


# ── Famille Tools ────────────────────────────────────────────────────────────

class ToolError(AgentError):
    """Erreur dans l'exécution d'un tool."""


class ToolValidationError(ToolError):
    """Params invalides en entrée d'un tool. Non retryable."""
    is_retryable = False


class ToolExecutionError(ToolError):
    """Erreur runtime pendant l'exécution d'un tool. Non retryable par défaut
    (souvent dû à des données invalides ; le retry ne change rien)."""
    is_retryable = False


# ── Famille State ────────────────────────────────────────────────────────────

class StateError(AgentError):
    """Invariant d'état violé (ex : plan_established=True sans study_plan)."""
    is_retryable = False
```

- [ ] **Step 6.4 : Run test exceptions**

```bash
pytest tests/test_exceptions.py -v
```

Attendu : 4 PASS.

- [ ] **Step 6.5 : Remplacer les `except Exception` larges**

Pour chaque site identifié dans l'audit :

**`agents/mortality/agents/graph.py:372-375`** — actuellement :

```python
except Exception as exc:
    import traceback
    yield {"type": "error", "message": str(exc)}
    yield {"type": "error", "message": traceback.format_exc()}
```

Remplacer par :

```python
except (LLMError, ToolError, StateError) as exc:
    # Erreur agentique connue — message utilisateur clair
    yield {"type": "error", "message": str(exc), "context": exc.context}
except Exception as exc:
    # Erreur inattendue — log full + message générique à l'utilisateur
    import traceback
    yield {"type": "error", "message": f"Erreur interne : {exc}"}
    yield {"type": "error", "message": traceback.format_exc()}
```

**`tools/tool_registry.py:378+`** : wrapper l'appel `mod.run(...)` dans try/except :

```python
try:
    if tool_name in _DF_TOOLS:
        ...
except (KeyError, ValueError, TypeError) as exc:
    raise ToolValidationError(
        f"{tool_name}.{function_name} : params invalides",
        tool_name=f"{tool_name}.{function_name}",
        params=params,
    ) from exc
except Exception as exc:
    raise ToolExecutionError(
        f"{tool_name}.{function_name} : erreur runtime",
        tool_name=f"{tool_name}.{function_name}",
    ) from exc
```

**`agents/report/pipeline/_04_redaction.py:41-46`** : préciser quels types sont attendus dans le `except Exception` actuel ; remplacer par les types spécifiques (probablement `(ImportError, AttributeError)` pour `_get_formats`).

- [ ] **Step 6.6 : Run suite globale**

```bash
pytest tests/ -q
```

- [ ] **Step 6.7 : Commit**

```bash
git add agents/mortality/agents/exceptions.py
git add agents/mortality/agents/graph.py
git add tools/tool_registry.py
git add agents/report/pipeline/_04_redaction.py
git add tests/test_exceptions.py
git commit -m "refactor(lot-6): introduce custom AgentError hierarchy

- Add AgentError, LLMError (RateLimit/Timeout/Malformed), ToolError, StateError
- Replace broad 'except Exception' with typed catches
- is_retryable flag enables targeted retry (lot 11)"
```

**Acceptance criteria:**
- Hiérarchie `AgentError` créée avec sous-classes typées.
- `except Exception` remplacés par catches spécifiques aux endroits identifiés.
- Tests de la hiérarchie PASS.

---

# PHASE 3 — EXTENSIBILITÉ AGENTIQUE

## Lot 7 : Tool registry metadata-driven

**Goal:** Remplacer le dispatch if/elif de `tools/tool_registry.py:340-376` par un registre décoré où chaque tool déclare ses besoins (df, data, validation Pydantic des inputs/outputs). Élimine la dette « ajouter un nouveau tool = modifier le dispatch ».

**Dependencies:** Lots 5 (Pydantic), 6 (exceptions).

**Files:**
- Create: `tools/registry/__init__.py`
- Create: `tools/registry/spec.py` (ToolSpec + decorator)
- Create: `tools/registry/dispatcher.py` (call_tool refait)
- Create: `tools/registry/validators.py` (helpers Pydantic)
- Modify: `tools/tool_registry.py` (devient un shim qui ré-exporte depuis registry/)
- Modify: chaque tool dans `tools/builder/`, `tools/statistical_analysis/`, `tools/graphs/`, `tools/build_pdf/`, `tools/preprocessing/`, `tools/conversation/`, `tools/master/`, `tools/reasoning/` (ajout du décorateur `@register_tool`)
- Test: `tests/test_tool_registry_v2.py` (NEW)

### Spec détaillée

#### `ToolSpec` (dataclass)

```python
@dataclass(frozen=True)
class ToolSpec:
    namespace: str               # "builder", "statistical_analysis", ...
    name: str                    # "exposure", "crude_rates", ...
    function: Callable           # run() du module
    requires_df: bool            # True si signature inclut df: pd.DataFrame
    requires_data: bool          # True si signature inclut data: dict
    requires_context: bool       # True si signature inclut context: dict (reasoning)
    input_model: type[BaseModel] | None    # Pydantic pour valider params
    output_model: type[BaseModel] | None   # Pydantic pour valider le retour
    result_key: str | None       # clé data_store où stocker le résultat (cf. _RESULT_KEYS)
    catalogue_metadata: dict     # extrait de la docstring TOOL CONTRACT
```

#### Decorator `@register_tool`

Usage dans chaque tool :

```python
# tools/builder/exposure.py
from tools.registry.spec import register_tool
from pydantic import BaseModel

class ExposureInput(BaseModel):
    rule: Literal["midpoint", "exact"] = "midpoint"

@register_tool(
    namespace="builder",
    name="exposure",
    requires_df=True,
    requires_data=False,
    input_model=ExposureInput,
    result_key="exposure_table",
)
def run(df, params):
    ...
```

#### `dispatcher.call_tool(tool_name, function_name, params, df, data, context)`

Implémentation :
1. Lookup `ToolSpec` dans le registry (KeyError → `ToolValidationError`).
2. Valider `params` via `spec.input_model.model_validate(params)`.
3. Préparer les args selon `requires_df/data/context`.
4. Invoquer `spec.function(...)`.
5. Si `output_model` défini : valider le retour.
6. Stocker dans `data[spec.result_key]` si défini.
7. Retourner le résultat.

### Tasks (résumé — à expanser avant exécution)

- [ ] **Step 7.1 : Cartographier les 66 tools et leur signature actuelle**

```bash
grep -rn "^def run(" /Users/macbook14/Python_projects/AgentActuariat/tools/ --include="*.py" > /tmp/tools_signatures.txt
cat /tmp/tools_signatures.txt
```

Produire un tableau (markdown ou CSV) : namespace × function × signature × result_key actuel (cf. `_RESULT_KEYS` dans tools_node.py:94-105). Sauvegarder dans `docs/superpowers/notes/2026-05-18-tools-inventory.md` pour traçabilité.

- [ ] **Step 7.2 : Écrire les tests du registry (spec + dispatcher)**

Créer `tests/test_tool_registry_v2.py` avec au minimum :
- Test : un tool décoré apparaît dans `REGISTRY`.
- Test : `dispatcher.call_tool` lève `ToolValidationError` si params manquants.
- Test : `dispatcher.call_tool` valide les outputs si `output_model` défini.
- Test : `dispatcher.call_tool` injecte df/data/context selon `ToolSpec`.
- Test : compat backward — les tests existants utilisant `tool_registry.call_tool` continuent de passer.

(Détail TDD à écrire avant exécution — chaque assertion = un test paramétré sur 2-3 tools représentatifs.)

- [ ] **Step 7.3 : Implémenter `tools/registry/spec.py`**

(Implémentation complète à fournir lors de l'expansion du lot — squelette dans la spec ci-dessus.)

- [ ] **Step 7.4 : Implémenter `tools/registry/dispatcher.py`**

(Implémentation : reproduit la logique de `tool_registry.call_tool` actuel mais data-driven via `ToolSpec`.)

- [ ] **Step 7.5 : Migrer les 66 tools (ajouter `@register_tool` à chaque `run`)**

Par paquets de 5-10 tools, en validant les tests après chaque paquet :
1. `tools/builder/` (≈12 tools)
2. `tools/statistical_analysis/` (≈7)
3. `tools/graphs/` (≈6)
4. `tools/build_pdf/` (≈8)
5. `tools/preprocessing/` (≈1)
6. `tools/conversation/` (≈3)
7. `tools/master/` (≈3)
8. `tools/reasoning/` (≈1)

Chaque migration de tool = 1 commit séparé : `refactor(lot-7): register {tool_name} in metadata-driven registry`.

- [ ] **Step 7.6 : Faire de `tools/tool_registry.py` un shim**

```python
# tools/tool_registry.py — shim de compat backward
from tools.registry.dispatcher import call_tool
from tools.registry.spec import get_capabilities

__all__ = ["call_tool", "get_capabilities"]
```

- [ ] **Step 7.7 : Migrer `_RESULT_KEYS` de tools_node.py vers les ToolSpec**

Dans `agents/mortality/agents/tools_node.py`, remplacer le dict `_RESULT_KEYS` par un lookup dans le registry :

```python
from tools.registry.spec import REGISTRY

def _result_key_for(tool_name: str, function_name: str) -> str | None:
    spec = REGISTRY.get(f"{tool_name}.{function_name}")
    return spec.result_key if spec else None
```

- [ ] **Step 7.8 : Run suite globale**

```bash
pytest tests/ -q
```

Attendu : tous les tests passent. Les nouveaux tests `test_tool_registry_v2.py` PASS.

- [ ] **Step 7.9 : Commit final lot 7**

```bash
git commit -m "refactor(lot-7): finalize metadata-driven tool registry

- All 66 tools migrated to @register_tool decorator
- tool_registry.py reduced to shim re-exporting from tools/registry/
- _RESULT_KEYS map removed (now declared in each ToolSpec)
- Adding a new tool now = decorate run() + Pydantic input/output models"
```

**Acceptance criteria:**
- Les 66 tools ont un `ToolSpec` enregistré.
- `call_tool` dispatch via le registry, sans if/elif chain.
- Ajouter un nouveau tool ne nécessite **aucun** changement dans `dispatcher.py` ni `tools_node.py`.
- Suite tests globale verte.

---

## Lot 8 : `BaseAgentNode` abstrait + factory

**Goal:** Extraire la logique commune (chargement system prompt, appel LLM, gestion tool_calls, émission events) dans une classe abstraite. Permet d'ajouter un nouvel agent de calcul = créer une sous-classe + l'enregistrer dans le graph.

**Dependencies:** Lots 5, 6, 7.

**Files:**
- Create: `agents/mortality/agents/base_agent.py`
- Modify: `master_node.py`, `builder_node.py`, `writer_node.py`, `rag_node.py` → hériter de `BaseAgentNode`
- Modify: `graph.py` (utiliser une factory)
- Test: `tests/test_base_agent.py` (NEW)

### Spec détaillée

#### `BaseAgentNode` (ABC)

```python
class BaseAgentNode(ABC):
    """Base class pour tous les nodes d'agent (Master, Builder, Writer, Rag, futurs)."""

    name: str                       # identifiant (master, builder, ...)
    llm_role: str                   # rôle dans llm_models.yaml
    system_prompt_path: Path        # chemin vers le .md de prompt

    @abstractmethod
    def allowed_tools(self) -> list[str]:
        """Liste des `tool_name.function_name` autorisés pour cet agent."""

    @abstractmethod
    def on_tool_result(self, state: AgentState, tool_msgs: list) -> dict:
        """Hook : traitement custom des résultats de tools (ex : update study_plan)."""

    def load_system_prompt(self, level: str) -> str:
        """Implémentation par défaut via loader.py."""

    def call_llm(self, messages: list, tools: list | None) -> AIMessage:
        """Implémentation par défaut via _utils.call_with_retry."""

    def emit_agent_switch(self, events: list) -> None:
        """Insère un event 'agent_switch' au début de la liste."""

    def __call__(self, state: AgentState) -> dict:
        """Entrée du node — orchestration standard. Override possible pour custom."""
```

### Tasks (résumé)

- [ ] **Step 8.1 : Identifier la logique commune dans les 4 nodes existants**

Diff manuel entre `master_node.py`, `builder_node.py`, `writer_node.py`, `rag_node.py`. Lister les blocs identiques (chargement prompt, ajout system_message, call LLM, event injection).

- [ ] **Step 8.2 : Écrire les tests de `BaseAgentNode`**

`tests/test_base_agent.py` : test une sous-classe fictive (`DummyAgentNode`) avec mock LLM.

- [ ] **Step 8.3 : Implémenter `BaseAgentNode`**

- [ ] **Step 8.4 : Migrer les 4 nodes existants en sous-classes**

Chaque migration = 1 commit séparé. Vérifier après chaque que `pytest tests/` reste vert.

- [ ] **Step 8.5 : Adapter `graph.py` pour utiliser les instances**

```python
master = MasterAgentNode()
builder = BuilderAgentNode()
writer = WriterAgentNode()
rag = RagAgentNode()

g.add_node("master", master)
g.add_node("builder", builder)
# etc.
```

- [ ] **Step 8.6 : Documenter le pattern « ajouter un nouvel agent de calcul »**

Créer `docs/superpowers/notes/2026-05-18-adding-new-agent.md` avec un exemple complet : créer `MyCalcAgentNode(BaseAgentNode)`, l'enregistrer dans `graph.py`, ajouter au routeur. Sert de guide pour les futurs ajouts d'agents.

- [ ] **Step 8.7 : Commit final**

**Acceptance criteria:**
- `BaseAgentNode` abstrait avec hooks `allowed_tools`, `on_tool_result`.
- Les 4 nodes existants héritent de `BaseAgentNode`.
- ~60 lignes de duplication system prompt loading éliminées.
- Doc « adding new agent » écrite.

---

## Lot 9 : Renderer registry pour rapports

**Goal:** Permettre l'ajout de nouveaux formats de rapport (HTML, Markdown, Docx, JSON) sans toucher au pipeline existant. Le pipeline `_05_assemble` devient un sélecteur.

**Dependencies:** Lot 6.

**Files:**
- Create: `agents/report/renderers/__init__.py`
- Create: `agents/report/renderers/base.py` (`ReportRenderer` ABC)
- Create: `agents/report/renderers/pdf_renderer.py` (déménagement reportlab)
- Create: `agents/report/renderers/html_renderer.py` (stub)
- Modify: `agents/report/pipeline/_05_assemble.py` (devient un dispatch)
- Modify: `agents/report/pipeline/run_pipeline.py` (accepte un paramètre `output_format`)
- Test: `tests/test_renderer_registry.py` (NEW)

### Spec détaillée

#### `ReportRenderer` (ABC)

```python
class ReportRenderer(ABC):
    """Renderer abstrait : transforme un ReportPlan + sections rédigées en bytes."""

    format_id: str            # "pdf", "html", "markdown", "docx", ...
    mime_type: str            # "application/pdf", ...
    file_extension: str       # ".pdf"

    @abstractmethod
    def render(self, plan: ReportPlan, sections: dict[str, RenderedSection]) -> bytes:
        ...

    @abstractmethod
    def validate_environment(self) -> None:
        """Vérifie que les dépendances (reportlab, weasyprint, ...) sont dispo."""
```

#### Registry

```python
RENDERERS: dict[str, type[ReportRenderer]] = {}

def register_renderer(cls: type[ReportRenderer]) -> type[ReportRenderer]:
    RENDERERS[cls.format_id] = cls
    return cls
```

#### Dispatch dans `_05_assemble.py`

```python
def assemble(plan, sections, output_path: str, format_id: str = "pdf") -> bytes:
    renderer_cls = RENDERERS.get(format_id)
    if renderer_cls is None:
        raise ValueError(f"Format inconnu : {format_id}. Disponibles : {list(RENDERERS)}")
    renderer = renderer_cls()
    renderer.validate_environment()
    return renderer.render(plan, sections)
```

### Tasks (résumé)

- [ ] **Step 9.1 : Lire `_05_assemble.py` actuel et identifier la frontière reportlab**

Identifier les fonctions qui touchent reportlab (création canvas, paragraphs, tables). Tout le reste reste dans `_05_assemble.py`.

- [ ] **Step 9.2 : Écrire les tests du renderer registry**

- [ ] **Step 9.3 : Implémenter `ReportRenderer` ABC + decorator**

- [ ] **Step 9.4 : Déménager la logique reportlab dans `pdf_renderer.py`**

- [ ] **Step 9.5 : Créer `html_renderer.py` stub**

```python
@register_renderer
class HtmlRenderer(ReportRenderer):
    format_id = "html"
    mime_type = "text/html"
    file_extension = ".html"

    def render(self, plan, sections):
        # TODO Lot futur : implémentation Jinja2
        raise NotImplementedError("HTML renderer to be implemented in lot 9-bis")

    def validate_environment(self) -> None:
        pass
```

- [ ] **Step 9.6 : Modifier `run_pipeline.py` pour accepter `output_format`**

Signature mise à jour : `run(data_store, initial_request, output_path, output_format="pdf")`.

- [ ] **Step 9.7 : Modifier `WriterAgent` (writer_node.py) pour propager `output_format`**

Lire le format demandé depuis `data_store["report_format"]` (clé optionnelle, défaut `"pdf"`). Documenter la clé dans `DataStoreKey` (Lot 3 — ajout REPORT_FORMAT = "report_format").

- [ ] **Step 9.8 : Documenter « adding a new report format »**

`docs/superpowers/notes/2026-05-18-adding-new-renderer.md`.

- [ ] **Step 9.9 : Commit final**

**Acceptance criteria:**
- `ReportRenderer` ABC en place.
- `pdf_renderer.py` produit le même output bit-à-bit que l'ancien `_05_assemble.py` (test golden file).
- `html_renderer.py` stub présent, lève `NotImplementedError`.
- `run_pipeline(output_format="pdf")` fonctionne ; `run_pipeline(output_format="html")` lève `NotImplementedError` proprement.

---

# PHASE 4 — ROBUSTESSE PRODUCTION

## Lot 10 : Observability (structured logging + correlation_id)

**Goal:** Logs structurés JSON avec `correlation_id` par session, métriques basiques (tool latency, LLM tokens) via décorateur.

**Dependencies:** Lots 6 (exceptions à logguer).

**Files:**
- Create: `agents/mortality/agents/observability.py`
- Modify: `requirements.txt` (ajout `structlog>=24.0`)
- Modify: tous les nodes pour utiliser `log = get_logger(__name__)`
- Modify: `tools/registry/dispatcher.py` (décorateur `@measure_latency` sur `call_tool`)
- Test: `tests/test_observability.py` (NEW)

### Spec

- `correlation_id` = `thread_id` propagé via context var `contextvars.ContextVar`.
- Logger structlog configuré pour sortir en JSON sur stderr.
- Décorateur `@measure_latency(metric_name: str)` qui logge `{event: metric, name: ..., latency_ms: ..., correlation_id: ...}`.
- Helper `log_with_context(**extra)` qui injecte automatiquement `correlation_id`.

### Tasks (résumé)

- [ ] 10.1 : Tests observability (correlation_id propagation, JSON format)
- [ ] 10.2 : Implémenter `observability.py` (configure_structlog, get_logger, correlation_id context var, @measure_latency)
- [ ] 10.3 : Injecter `correlation_id` au début de `stream_agent` (graph.py)
- [ ] 10.4 : Migrer les `logging.getLogger(__name__)` vers `get_logger(__name__)` dans les nodes
- [ ] 10.5 : Décorer `dispatcher.call_tool` avec `@measure_latency("tool_call")`
- [ ] 10.6 : Documenter le format de log dans `docs/superpowers/notes/2026-05-18-observability.md`
- [ ] 10.7 : Commit

**Acceptance criteria:**
- Tous les logs sont JSON-structurés.
- Chaque log porte `correlation_id` = thread_id.
- `tool_call` events loggés avec `latency_ms`.

---

## Lot 11 : Retry/circuit breaker agent-level

**Goal:** Étendre `call_with_retry` de `_utils.py` au niveau node : si un agent échoue 3× (errors marquées `is_retryable=True` via Lot 6) → fallback message clair user. Circuit breaker sur provider LLM en cas de rate limit prolongé.

**Dependencies:** Lots 6 (exceptions typées), 8 (BaseAgentNode pour décorer).

**Files:**
- Modify: `agents/mortality/agents/_utils.py` (étendre `call_with_retry`)
- Modify: `agents/mortality/agents/base_agent.py` (décorer `call_llm`)
- Create: `agents/mortality/agents/circuit_breaker.py`
- Test: `tests/test_retry_circuit_breaker.py` (NEW)

### Spec

- Retry exponentiel : 1s, 2s, 4s, max 3 tentatives.
- Seulement sur `AgentError.is_retryable=True`.
- Circuit breaker : si `LLMRateLimitError` survient ≥ 5 fois en 60s sur un provider, ouvrir le circuit pendant 30s (toutes les requêtes lèvent `LLMError("circuit open")` sans appel réseau).

### Tasks (résumé)

- [ ] 11.1 : Tests retry (3 tentatives, backoff exponentiel, non-retryable lève immédiatement)
- [ ] 11.2 : Tests circuit breaker (state machine open/half-open/closed)
- [ ] 11.3 : Étendre `call_with_retry`
- [ ] 11.4 : Implémenter `CircuitBreaker` class
- [ ] 11.5 : Intégrer dans `BaseAgentNode.call_llm`
- [ ] 11.6 : Commit

**Acceptance criteria:**
- Retry uniquement sur `is_retryable=True`.
- Circuit breaker testé sur scénario fail × 5 / wait / succès.

---

## Lot 12 : Checkpointer SqliteSaver persistant

**Goal:** Remplacer `MemorySaver` (RAM-only) par `SqliteSaver` pour survivre aux redémarrages. Élimine la sync manuelle canvas ↔ disk.

**Dependencies:** Lot 5 (state Pydantic doit être sérialisable).

**Files:**
- Modify: `agents/mortality/agents/graph.py:48` (remplacer `MemorySaver()` par `SqliteSaver`)
- Modify: `requirements.txt` (ajout `langgraph-checkpoint-sqlite` si pas déjà inclus)
- Create: `sessions/checkpoints.db` (auto-créé au runtime, ajouter à `.gitignore`)
- Test: `tests/test_checkpointer_persistence.py` (NEW)

### Tasks (résumé)

- [ ] 12.1 : Vérifier dispo `langgraph-checkpoint-sqlite` ou équivalent
- [ ] 12.2 : Test persistence (créer state, redémarrer process, recharger même thread_id)
- [ ] 12.3 : Implémenter SqliteSaver wrapper
- [ ] 12.4 : Migration : si `sessions/{thread_id}_audit.json` existe, importer dans Sqlite au démarrage (one-shot)
- [ ] 12.5 : Mettre à jour `.gitignore` (`sessions/checkpoints.db`, `sessions/*.db-wal`, `sessions/*.db-shm`)
- [ ] 12.6 : Commit

**Acceptance criteria:**
- State survit à `python` restart sur même `thread_id`.
- Pas d'impact perf (SQLite local < 5ms par checkpoint).
- Migration depuis JSON legacy fonctionne.

---

# PHASE 5 — TESTS

## Lot 13 : `conftest.py` + fixtures partagées

**Goal:** Centraliser les fixtures réutilisables, éliminer les paths hardcodés `/tmp`, accélérer les nouveaux tests.

**Dependencies:** Lots 5 (AgentState Pydantic), 7 (registry).

**Files:**
- Create: `tests/conftest.py`
- Modify: tests existants pour utiliser les fixtures (refactor opportuniste, pas obligatoire)

### Fixtures à fournir

- `tmp_session_dir` (tmp_path scope=function)
- `sample_dataset` : pd.DataFrame minimal (50 lignes : id, sexe, age, date_entree, date_sortie, cause_sortie)
- `sample_dataset_h_f` : variante avec mix homme/femme
- `fresh_agent_state` : `AgentState()` clean
- `mock_llm` : `unittest.mock.MagicMock` pour `ChatOpenAI` qui retourne une réponse paramétrable
- `mock_llm_with_tool_call` : mock qui simule un tool_call structuré
- `temp_data_store` : dict pré-rempli avec quelques clés DataStoreKey
- `compiled_graph` : `build_graph()` compilé avec MemorySaver pour tests

### Tasks

- [ ] 13.1 : Implémenter `conftest.py`
- [ ] 13.2 : Refactor 5 tests existants pour utiliser les fixtures (PR exemple)
- [ ] 13.3 : Commit

**Acceptance criteria:**
- `tests/conftest.py` créé.
- ≥ 5 tests refactorisés en exemple.
- Doc d'utilisation dans le docstring du conftest.

---

## Lot 14 : Tests d'intégration graph E2E

**Goal:** Suite `tests/integration/` qui exécute des conversations complètes (upload → builder → writer → PDF) avec LLM mocké. Filet de sécurité contre les régressions de routing.

**Dependencies:** Lots 5, 7, 8, 13.

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_master_to_builder_e2e.py`
- Create: `tests/integration/test_master_to_writer_e2e.py`
- Create: `tests/integration/test_rag_flow_e2e.py`
- Create: `tests/integration/test_step_by_step_approval.py`
- Create: `tests/integration/test_writer_need_data_loop_prevention.py`

### Scénarios à couvrir

1. **Master → Builder → Master** : user demande « calcule l'exposition », master route vers builder, builder appelle `builder.exposure`, retour à master.
2. **Master → Builder → Writer → PDF** : user demande « génère un rapport », flow complet jusqu'au PDF généré.
3. **RAG flow** : user pose une question doctrinale, route vers rag, retour 1 cycle.
4. **Step-by-step approval** : `step_by_step=True`, vérifier que `pending_tool_call` est posé et que `approval_event` débloque l'exécution.
5. **NEED_DATA loop prevention** : writer signale NEED_DATA, master tente builder, builder ne peut pas, writer redéploie NEED_DATA avec mêmes champs → mode dégradé (cf. writer_node.py:103-123).

### Tasks (résumé)

- [ ] 14.1 : Test scénario 1
- [ ] 14.2 : Test scénario 2
- [ ] 14.3 : Test scénario 3
- [ ] 14.4 : Test scénario 4
- [ ] 14.5 : Test scénario 5
- [ ] 14.6 : Commit

**Acceptance criteria:**
- ≥ 5 scénarios E2E PASS.
- LLM 100% mocké (pas d'appel réseau, pas de coût).
- Durée totale < 30 s.

---

## Lot 15 : Failure injection

**Goal:** Tests qui injectent des pannes (LLM JSON malformé, tool exception, timeout, dataset corrompu) pour valider que les exceptions custom (Lot 6) + retry (Lot 11) gèrent gracieusement.

**Dependencies:** Lots 6, 11, 13, 14.

**Files:**
- Create: `tests/failure_injection/__init__.py`
- Create: `tests/failure_injection/test_llm_malformed_json.py`
- Create: `tests/failure_injection/test_llm_rate_limit_retry.py`
- Create: `tests/failure_injection/test_tool_exceptions.py`
- Create: `tests/failure_injection/test_dataset_corrupted.py`

### Scénarios

1. LLM retourne `{"tool_calls": "not_a_list"}` → `LLMMalformedResponseError` propre, message user clair.
2. LLM retourne `429` 2× puis 200 → retry réussit après backoff.
3. LLM retourne `429` 5× en 60s → circuit breaker ouvre.
4. Tool `builder.exposure` lève `KeyError` → `ToolExecutionError`, message user, pas de crash graph.
5. CSV uploadé avec colonnes manquantes → `ToolValidationError` au preprocessing.

### Tasks (résumé)

- [ ] 15.1 : Scénario 1
- [ ] 15.2 : Scénario 2
- [ ] 15.3 : Scénario 3
- [ ] 15.4 : Scénario 4
- [ ] 15.5 : Scénario 5
- [ ] 15.6 : Commit

**Acceptance criteria:**
- 5 scénarios PASS.
- Vérification que les messages utilisateurs sont compréhensibles (pas de stack trace brute).

---

# ANNEXE A — Ordre d'exécution recommandé

```
JOUR 1-2  : Phase 1 (Lots 1, 2, 3) — 3 commits, faible risque
JOUR 3    : Lot 4 (config LLM unifiée)
JOUR 4-6  : Lot 5 (PIVOT Pydantic) — review approfondie obligatoire avant merge
JOUR 7    : Lot 6 (exceptions custom)
JOUR 8-11 : Lot 7 (tool registry) — parallélisable avec Lot 8
JOUR 8-10 : Lot 8 (BaseAgentNode) — parallélisable avec Lot 7
JOUR 12-14: Lot 9 (renderer registry)
JOUR 15-16: Lot 10 (observability)
JOUR 17-18: Lot 11 (retry/circuit breaker)
JOUR 19-20: Lot 12 (SqliteSaver)
JOUR 21   : Lot 13 (conftest)
JOUR 22-24: Lot 14 (tests E2E)
JOUR 25-26: Lot 15 (failure injection)
```

**Total estimé** : 26 jours-homme (~5 semaines avec slack), dont 4 jours sur le lot pivot 5.

# ANNEXE B — Registre des risques

| Risque | Impact | Mitigation |
|---|---|---|
| Lot 5 (Pydantic) introduit régression silencieuse sur sérialisation MemorySaver | Élevé | Test `test_serializable_for_checkpointer` + tests E2E (Lot 14) avant merge |
| Lot 7 (registry) cassé sur un tool non détecté lors de l'inventaire | Moyen | Step 7.1 produit l'inventaire exhaustif, audit manuel obligatoire |
| Lot 8 (BaseAgentNode) introduit subtilité d'héritage incompatible avec un node existant | Moyen | Migration node-par-node, suite tests verte après chaque |
| Lot 12 (SqliteSaver) cassé sur la migration JSON → SQLite | Moyen | Garder le code MemorySaver derrière un feature flag `USE_SQLITE_CHECKPOINTER` pendant 1 semaine |
| Augmentation du temps de tests à cause des fixtures (Lot 13) | Faible | Fixtures scope=session pour les coûteuses, scope=function pour le reste |

# ANNEXE C — Critères de done global

- [ ] Tous les lots 1-15 mergés sur main.
- [ ] Coverage tests ≥ 25 % (vs 3.3 % initial).
- [ ] Aucun `WRITER_MODEL` ni autre hardcoding LLM dans le code.
- [ ] Aucune string `<XXX>` signal consommée par `in` (tous via `has_signal`).
- [ ] Aucun `except Exception` sans qualification (sauf points d'entrée stream_agent).
- [ ] `docs/superpowers/notes/2026-05-18-adding-new-agent.md` rédigé.
- [ ] `docs/superpowers/notes/2026-05-18-adding-new-renderer.md` rédigé.
- [ ] CI verte sur la branche.

# ANNEXE D — Convention de branches

- Une branche par lot : `feat/lot-N-short-name`
- PRs ouvertes en séquence pour les lots dépendants
- PRs parallélisables (cf. graphe de dépendances) peuvent être ouvertes simultanément
- Merge dans l'ordre : 1 → 2 → 3, 4 → 5 → 6 → (7 || 8) → 9 → 10 → 11 → 12 → 13 → 14 → 15
