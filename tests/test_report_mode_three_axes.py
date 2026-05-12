"""Tests du modèle à 3 axes (kind, write, report_mode) + filtrage YAML.

Couvre :
  - activation multi-clés dans _is_active
  - load_section variante narrative selon report_mode
  - _sections_for_mode / _keys_for_sections
  - _classify_intent fallback (mode offline)
  - Master : désambiguation write=ask avant Builder
  - Master : compteur cumulatif _master_builder_cycles
  - Master : BUILD_DONE suit _write
  - Builder : bloc capabilities généré depuis YAML
  - Builder : branche déterministe raw_rates assimile qx→smoothed
  - Builder : garde-fou decision_required écrase tool_calls même sans content
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────
# 1. _is_active : activation multi-clés (nouveau format)
# ──────────────────────────────────────────────────────────────────────────

def test_is_active_old_format_scalar():
    from knowledge_base.report_template.template_loader import _is_active
    sec = {"activation": {"key": "gender_segmentation", "equals": "unisex"}}
    assert _is_active(sec, {"gender_segmentation": "unisex"}) is True
    assert _is_active(sec, {"gender_segmentation": "by_sex"}) is False


def test_is_active_new_format_list_match():
    from knowledge_base.report_template.template_loader import _is_active
    sec = {"activation": {"report_mode": ["full_report", "raw_rates"]}}
    assert _is_active(sec, {"report_mode": "full_report"}) is True
    assert _is_active(sec, {"report_mode": "raw_rates"}) is True
    assert _is_active(sec, {"report_mode": "description"}) is False


def test_is_active_new_format_multi_key_and():
    from knowledge_base.report_template.template_loader import _is_active
    sec = {"activation": {
        "report_mode": ["full_report", "raw_rates", "description"],
        "gender_segmentation": ["unisex"],
    }}
    # Les deux matchent
    assert _is_active(sec, {"report_mode": "full_report", "gender_segmentation": "unisex"}) is True
    # Une seule matche
    assert _is_active(sec, {"report_mode": "full_report", "gender_segmentation": "by_sex"}) is False


def test_is_active_missing_context_key_is_tolerant():
    """Si une clé d'activation n'est pas fournie par le contexte, on skippe
    la contrainte (au lieu de tomber False)."""
    from knowledge_base.report_template.template_loader import _is_active
    sec = {"activation": {
        "report_mode": ["full_report"],
        "gender_segmentation": ["unisex"],
    }}
    # gender_segmentation absent du contexte : on ne bloque pas
    assert _is_active(sec, {"report_mode": "full_report"}) is True


# ──────────────────────────────────────────────────────────────────────────
# 2. load_section : variante de narrative selon report_mode
# ──────────────────────────────────────────────────────────────────────────

def test_load_section_default_variant(tmp_path):
    import yaml as _yaml
    from knowledge_base.report_template.template_loader import load_section
    tpl = {
        "session_inputs": [],
        "data_contract": {
            "master_from_data": [],
            "master_from_modeling": [],
            "builder_outputs": [],
        },
        "sections": [{
            "id": "preamble", "label": "P", "required": True, "dependencies": [],
            "narrative": {
                "text_default": "Rapport avec lissage",
                "text_raw_rates": "Rapport avec taux bruts assimilés",
            },
            "llm_directives": {"tone": "", "length_words": [1, 2], "rag_query": ""},
            "visual_specs": [],
        }],
    }
    path = tmp_path / "t.yaml"
    path.write_text(_yaml.safe_dump(tpl))

    # Sans context → text_default
    sec = load_section("preamble", path)
    assert "lissage" in sec.narrative["text"]

    # Avec report_mode=raw_rates → text_raw_rates
    sec = load_section("preamble", path, context={"report_mode": "raw_rates"})
    assert "assimilés" in sec.narrative["text"]

    # Avec autre mode → text_default
    sec = load_section("preamble", path, context={"report_mode": "full_report"})
    assert "lissage" in sec.narrative["text"]


def test_load_section_single_text_backward_compat(tmp_path):
    import yaml as _yaml
    from knowledge_base.report_template.template_loader import load_section
    tpl = {
        "session_inputs": [],
        "data_contract": {
            "master_from_data": [], "master_from_modeling": [], "builder_outputs": [],
        },
        "sections": [{
            "id": "p", "label": "P", "required": True, "dependencies": [],
            "narrative": {"text": "texte unique"},
            "llm_directives": {"tone": "", "length_words": [1, 2], "rag_query": ""},
            "visual_specs": [],
        }],
    }
    path = tmp_path / "t.yaml"
    path.write_text(_yaml.safe_dump(tpl))
    sec = load_section("p", path, context={"report_mode": "raw_rates"})
    assert sec.narrative["text"] == "texte unique"


# ──────────────────────────────────────────────────────────────────────────
# 3. _sections_for_mode et _keys_for_sections
# ──────────────────────────────────────────────────────────────────────────

def test_sections_for_mode_full_report():
    from agents.mortality.agents.master_node import _sections_for_mode
    sections = _sections_for_mode("full_report", gender_segmentation="unisex")
    # Le YAML actuel contient au minimum preamble, data_preprocessing, data_analysis_unisex
    assert "preamble" in sections
    assert "data_preprocessing" in sections
    assert "data_analysis_unisex" in sections
    # Pas data_analysis_by_sex car gender=unisex
    assert "data_analysis_by_sex" not in sections


def test_sections_for_mode_by_sex():
    from agents.mortality.agents.master_node import _sections_for_mode
    sections = _sections_for_mode("full_report", gender_segmentation="by_sex")
    assert "data_analysis_by_sex" in sections
    assert "data_analysis_unisex" not in sections


def test_keys_for_sections_subset_of_builder_outputs():
    """Les clés retournées doivent être un sous-ensemble des builder_outputs."""
    from agents.mortality.agents.master_node import _sections_for_mode, _keys_for_sections, _get_builder_keys
    sections = _sections_for_mode("full_report", gender_segmentation="unisex")
    keys = _keys_for_sections(sections)
    all_keys = set(_get_builder_keys())
    assert set(keys).issubset(all_keys)
    assert len(keys) >= 1  # au moins 1 clé consommée


# ──────────────────────────────────────────────────────────────────────────
# 4. Master — désambiguation write=ask avant Builder
# ──────────────────────────────────────────────────────────────────────────

def test_master_asks_write_question_before_builder(monkeypatch):
    """Quand classify_intent renvoie write=ask, Master émet un AIMessage
    question et ne route pas vers le Builder."""
    from langchain_core.messages import HumanMessage
    from agents.mortality.agents import master_node as mn

    def _fake_classify(*args, **kwargs):
        return {
            "kind": "task",
            "write": "ask",
            "report_mode": "full_report",
            "intent": "build_and_write",
            "reply": "",
        }
    monkeypatch.setattr(mn, "_classify_intent", _fake_classify)

    state = {
        "messages": [HumanMessage(content="construis une table de mortalité")],
        "data_store": {"_disambiguation_done": True, "study_plan": {"gender_segmentation": "unisex"}},
        "dataset_ref": None,
    }
    out = mn.master_node(state)

    # Master a émis un AIMessage avec la question, pas de route Builder
    assert "active_agent" not in out or out.get("active_agent") != "builder"
    assert out["data_store"].get("_write_question_asked") is True
    msgs = out.get("messages") or []
    assert any("rapport" in (m.content or "").lower() for m in msgs)


def test_master_routes_to_builder_when_write_yes(monkeypatch):
    """Si classify_intent retourne write=yes, Master route direct vers Builder."""
    from langchain_core.messages import HumanMessage
    from agents.mortality.agents import master_node as mn

    def _fake_classify(*args, **kwargs):
        return {
            "kind": "task",
            "write": "yes",
            "report_mode": "full_report",
            "intent": "build_and_write",
            "reply": "",
        }
    monkeypatch.setattr(mn, "_classify_intent", _fake_classify)

    state = {
        "messages": [HumanMessage(content="fais-moi le rapport")],
        "data_store": {
            "_disambiguation_done":   True,
            "_methods_question_done": True,
            "study_plan":             {"gender_segmentation": "unisex",
                                       "methods_auto":        True},
        },
        "dataset_ref": None,
    }
    out = mn.master_node(state)
    assert out.get("active_agent") == "builder"
    # Instruction Builder contient "Sections actives" et "Reste à produire"
    instr = (out.get("messages") or [None])[0]
    assert instr is not None
    assert "Sections actives" in instr.content
    assert "Reste à produire" in instr.content


# ──────────────────────────────────────────────────────────────────────────
# 5. Master — compteur cumulatif _master_builder_cycles
# ──────────────────────────────────────────────────────────────────────────

def test_master_cumulative_cycle_limit(monkeypatch):
    """Après 6 cycles sans convergence, Master doit s'arrêter.

    La limite a été portée de 3 à 6 pour absorber le pipeline full_report
    (~5 batchs : descriptifs → crude_rates → smoothing → validation →
    aggregation_deciles).
    """
    from langchain_core.messages import HumanMessage
    from agents.mortality.agents import master_node as mn

    def _fake_classify(*args, **kwargs):
        return {
            "kind": "task",
            "write": "yes",
            "report_mode": "full_report",
            "intent": "build_and_write",
            "reply": "",
        }
    monkeypatch.setattr(mn, "_classify_intent", _fake_classify)

    state = {
        "messages": [HumanMessage(content="fais-moi le rapport")],
        "data_store": {
            "_disambiguation_done":     True,
            "_methods_question_done":   True,
            "_master_builder_cycles":   6,  # déjà 6 cycles → 7 > 6 → stop
            "study_plan":               {"gender_segmentation": "unisex",
                                         "methods_auto":        True},
        },
        "dataset_ref": None,
    }
    out = mn.master_node(state)

    events = out.get("events") or []
    has_done = any(e.get("type") == "done" for e in events if isinstance(e, dict))
    assert has_done
    # Pas de route Builder
    assert out.get("active_agent") != "builder"


# ──────────────────────────────────────────────────────────────────────────
# 6. Builder — bloc capabilities depuis YAML
# ──────────────────────────────────────────────────────────────────────────

def test_capabilities_block_non_empty_and_references_sections():
    from agents.mortality.agents.builder_node import _capabilities_block
    block = _capabilities_block()
    assert "Capacités disponibles" in block
    assert "preamble" in block
    assert "data_preprocessing" in block
    assert "Clés à produire" in block
    assert "Tools à appeler" in block


# ──────────────────────────────────────────────────────────────────────────
# 7. Builder — branche raw_rates assimile qx→smoothed
# ──────────────────────────────────────────────────────────────────────────

def test_builder_raw_rates_assimilation_is_deterministic(monkeypatch):
    """Avec report_mode=raw_rates et qx_table présent, smoothed_table doit être
    auto-produit SANS appel LLM."""
    from unittest.mock import MagicMock
    from langchain_core.messages import HumanMessage
    from agents.mortality.agents import builder_node as bn

    # Fake le client OpenAI + call_with_retry pour éviter tout vrai appel réseau
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "go ahead"
    mock_response.choices[0].message.tool_calls = []
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    class _FakeClient:
        def __init__(self, *a, **kw): pass
    monkeypatch.setattr("openai.OpenAI", _FakeClient)
    monkeypatch.setattr(
        "agents.mortality.agents._utils.call_with_retry",
        lambda *a, **kw: mock_response,
    )

    data_store = {
        "report_mode": "raw_rates",
        "qx_table": [
            {"age": 30, "q_x_brut": 0.0012},
            {"age": 31, "q_x_brut": 0.0013},
        ],
    }
    state = {"messages": [HumanMessage(content="go")], "data_store": data_store}

    out = bn.builder_node(state)

    # smoothed_table doit être dans data_store, produit par assimilation.
    # Avec l'enrichissement Lot 1, on inclut aussi q_x_brut pour le
    # tableau YAML smoothing_table.
    ds = out.get("data_store", {})
    assert ds.get("smoothed_table") == [
        {"age": 30, "q_x_brut": 0.0012, "q_x_lisse": 0.0012},
        {"age": 31, "q_x_brut": 0.0013, "q_x_lisse": 0.0013},
    ]


# ──────────────────────────────────────────────────────────────────────────
# 8. Builder — garde-fou decision_required écrase tool_calls même sans content
# ──────────────────────────────────────────────────────────────────────────

def test_decision_gate_erases_tool_calls_even_without_content(monkeypatch):
    """Si decision_required est pending ET LLM émet tool_calls (avec ou sans
    content), on écrase les tool_calls et on force un message de rappel."""
    from unittest.mock import MagicMock
    from langchain_core.messages import HumanMessage, ToolMessage
    from agents.mortality.agents import builder_node as bn

    tm = ToolMessage(
        content='{"smoothed_table": [...], "decision_required": {"options": []}}',
        tool_call_id="1",
    )
    hm = HumanMessage(content="continue")
    state = {"messages": [hm, tm], "data_store": {}}

    # LLM réponse : tool_calls uniquement, AUCUN content
    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_tool_call = MagicMock()
    mock_tool_call.id = "new-1"
    mock_tool_call.function.name = "builder.validation"
    mock_tool_call.function.arguments = "{}"
    mock_choice.message.tool_calls = [mock_tool_call]
    mock_choice.finish_reason = "tool_calls"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    class _FakeClient:
        def __init__(self, *a, **kw): pass
    monkeypatch.setattr("openai.OpenAI", _FakeClient)
    monkeypatch.setattr(
        "agents.mortality.agents._utils.call_with_retry",
        lambda *a, **kw: mock_response,
    )

    out = bn.builder_node(state)

    # Le message retourné doit avoir tool_calls=[] (écrasés par le garde-fou)
    msgs = out.get("messages") or []
    assert msgs, "Builder doit retourner au moins un message"
    last = msgs[-1]
    assert not getattr(last, "tool_calls", None), "Les tool_calls doivent être écrasés"
    # Content forcé (non vide) pour que l'UI affiche quelque chose
    assert getattr(last, "content", None), "Le content doit être forcé par le garde-fou"
