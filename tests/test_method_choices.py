"""Tests : flux de désambiguation des méthodes (Master ↔ user) + injection
des méthodes choisies dans les params au moment de l'appel des tools.

Couvre :
  - `agents.master.method_choices.method_choices_for_mode` (filtrage + résolution).
  - `agents.master.method_choices.resolve_user_answer_to_method` (matching tolérant).
  - Master pose la méta-question "auto / préciser" quand des choix existent.
  - Master enchaîne les questions par tool si l'utilisateur dit "préciser".
  - `tools_node._inject_user_method` écrit la méthode dans `params` avant exécution.
"""
from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────
# 1. method_choices : découverte par mode
# ──────────────────────────────────────────────────────────────────────────

def test_raw_rates_mode_only_proposes_crude_rates():
    from agents.master.method_choices import all_choices_for_mode
    choices = all_choices_for_mode("raw_rates", "unisex")
    tools = [c.tool for c in choices]
    assert tools == ["builder.crude_rates"]
    spec = choices[0]
    assert "central"      in spec.choices
    assert "binomial"     in spec.choices
    assert "kaplan_meier" in spec.choices


def test_description_mode_no_method_choice():
    from agents.master.method_choices import all_choices_for_mode
    assert all_choices_for_mode("description", "unisex") == []


def test_full_report_proposes_crude_smoothing_validation():
    from agents.master.method_choices import all_choices_for_mode
    tools = {c.tool for c in all_choices_for_mode("full_report", "unisex")}
    assert tools >= {"builder.crude_rates", "builder.smoothing", "builder.validation"}


def test_method_choices_skip_when_already_picked():
    from agents.master.method_choices import method_choices_for_mode
    sp = {"methods": {"builder.crude_rates": "binomial"}}
    remaining = method_choices_for_mode("full_report", "unisex", sp)
    tools = {c.tool for c in remaining}
    assert "builder.crude_rates" not in tools


def test_method_choices_skip_all_when_methods_auto():
    from agents.master.method_choices import method_choices_for_mode
    sp = {"methods_auto": True}
    assert method_choices_for_mode("full_report", "unisex", sp) == []


def test_resolve_user_answer_aliases():
    from agents.master.method_choices import resolve_user_answer_to_method
    choices = ["central", "binomial", "kaplan_meier"]
    assert resolve_user_answer_to_method("KM",       choices) == "kaplan_meier"
    assert resolve_user_answer_to_method("kaplan",   choices) == "kaplan_meier"
    assert resolve_user_answer_to_method("Binomial", choices) == "binomial"
    assert resolve_user_answer_to_method("xyz",      choices) is None


def test_resolve_user_answer_handles_typos_and_sentences():
    """Cas vu en prod : l'utilisateur tape la méthode dans une phrase
    libre, avec typos et ponctuation. On doit retrouver les noms canon."""
    from agents.master.method_choices import resolve_user_answer_to_method
    ans = ("pour les taux bruts, sélectionne kaplan_meir. pour le lissage "
           "applique Whittaker, validation via confidence_intervals")
    assert resolve_user_answer_to_method(
        ans, ["central", "binomial", "kaplan_meier"],
    ) == "kaplan_meier"
    assert resolve_user_answer_to_method(
        ans, ["whittaker", "gompertz", "makeham", "spline"],
    ) == "whittaker"
    assert resolve_user_answer_to_method(
        ans, ["confidence_intervals", "chi_square"],
    ) == "confidence_intervals"


def test_llm_fallback_picks_methods_when_regex_fails(monkeypatch):
    """Fallback LLM appelé uniquement quand regex+alias échoue.
    Validation : on rejette toute valeur LLM hors liste autorisée."""
    from agents.master import method_choices as mc

    # Stub openai.OpenAI : faux client retournant un JSON valide
    class _FakeMsg:
        def __init__(self, content): self.content = content
    class _FakeChoice:
        def __init__(self, content): self.message = _FakeMsg(content)
    class _FakeResp:
        def __init__(self, content): self.choices = [_FakeChoice(content)]
    class _FakeCompletions:
        def create(self, **kw):
            return _FakeResp(
                '{"builder.crude_rates": "kaplan_meier", '
                '"builder.smoothing": "whittaker", '
                '"builder.validation": "INVALID_METHOD"}'  # → filtré
            )
    class _FakeChat:
        completions = _FakeCompletions()
    class _FakeClient:
        chat = _FakeChat()
    import openai as _real_openai
    monkeypatch.setattr(_real_openai, "OpenAI", lambda *a, **k: _FakeClient())

    specs = mc.all_choices_for_mode("full_report", "unisex")
    out = mc.llm_fallback_resolve_methods(
        "Je veux un estimateur non-paramétrique pour les bruts et "
        "un lissage type Henderson", specs,
    )
    assert out.get("builder.crude_rates") == "kaplan_meier"
    assert out.get("builder.smoothing")   == "whittaker"
    # La valeur invalide doit être filtrée (pas dans la liste autorisée)
    assert "builder.validation" not in out


def test_llm_fallback_returns_empty_on_network_error(monkeypatch):
    """Si le LLM lève une exception, retourner {} sans crasher."""
    import openai as _real_openai
    from agents.master import method_choices as mc

    def _boom(*a, **kw): raise RuntimeError("network down")
    monkeypatch.setattr(_real_openai, "OpenAI", _boom)

    specs = mc.all_choices_for_mode("raw_rates", "unisex")
    assert mc.llm_fallback_resolve_methods("blah blah", specs) == {}


def test_master_skips_meta_question_when_methods_already_picked(monkeypatch):
    """Garde-fou : même si `_methods_question_done` a été perdu entre cycles,
    la présence de `study_plan.methods` non vide doit suffire à skipper la
    méta-question. Bug observé en prod : la question se reposait après le
    column mapping."""
    from agents.mortality.agents import master_node as mn
    monkeypatch.setattr(mn, "_classify_intent", _fake_classify("yes", "full_report"))
    state = {
        "messages":   [HumanMessage(content="continue")],
        "data_store": {
            "_disambiguation_done": True,
            # _methods_question_done volontairement absent
            "study_plan": {
                "gender_segmentation": "unisex",
                "methods": {
                    "builder.crude_rates": "kaplan_meier",
                    "builder.smoothing":   "whittaker",
                    "builder.validation":  "confidence_intervals",
                },
            },
        },
        "dataset_ref": None,
    }
    out = mn.master_node(state)
    pending = (out["data_store"].get("_pending_need") or {})
    assert pending.get("context_key") != "methods_choice_mode", (
        "Master a re-posé la méta-question alors que study_plan.methods est rempli"
    )


def test_study_plan_persists_methods_through_to_data_store():
    """Régression : sans `methods` et `methods_auto` dans le schema Pydantic
    StudyPlan, ces champs sont silencieusement droppés à chaque cycle, ce
    qui fait re-poser la méta-question méthodes. Vu en prod."""
    from session.session_state import SessionState, StudyPlan
    state = SessionState(
        session_id="test",
        study_plan=StudyPlan(
            gender_segmentation="unisex",
            methods={
                "builder.crude_rates": "kaplan_meier",
                "builder.smoothing":   "whittaker",
                "builder.validation":  "confidence_intervals",
            },
        ),
    )
    ds = state.to_data_store()
    sp = ds.get("study_plan") or {}
    assert sp.get("methods", {}).get("builder.crude_rates") == "kaplan_meier"
    assert sp.get("methods", {}).get("builder.smoothing")   == "whittaker"
    assert sp.get("methods", {}).get("builder.validation")  == "confidence_intervals"


def test_msgpack_safe_converts_numpy_scalars():
    """Régression : numpy.float64 dans result de tool fait crasher
    LangGraph MemorySaver (msgpack). Doit être converti en float natif."""
    import numpy as np
    from agents.mortality.agents.tools_node import _msgpack_safe
    out = _msgpack_safe({
        "scalar_float":  np.float64(1.5),
        "scalar_int":    np.int64(42),
        "scalar_bool":   np.bool_(True),
        "nested": {"x": np.float64(2.5), "y": [np.int64(1), np.float64(3.14)]},
        "nan_value": np.float64("nan"),
    })
    assert type(out["scalar_float"]).__name__ == "float"
    assert type(out["scalar_int"]).__name__   == "int"
    assert type(out["scalar_bool"]).__name__  == "bool"
    assert type(out["nested"]["x"]).__name__  == "float"
    assert type(out["nested"]["y"][1]).__name__ == "float"
    assert out["nan_value"] is None  # NaN → None pour msgpack-safety
    # Round-trip msgpack pour confirmer la sérialisation
    import ormsgpack
    ormsgpack.packb(out)  # ne doit pas lever


def test_maybe_normalize_records_writes_parquet_with_dates(tmp_path, monkeypatch):
    """E2E : après validation des mappings, on doit avoir un Parquet
    sur disque avec colonnes canoniques, valeurs mappées, dates en
    datetime64, sentinelles 2999 clippées."""
    import pandas as pd
    from session import dataset_store as _ds
    monkeypatch.setattr(_ds, "_ARTIFACTS_DIR", tmp_path)
    from agents.master.disambiguation import maybe_normalize_records

    df = pd.DataFrame({
        "CLINAISS":   ["01/01/1950", "01/01/1960", "01/01/1970"],
        "CTREFFET":   ["01/01/2000", "01/01/2005", "01/01/2010"],
        "DATE_SORTIE": ["31/12/2010", "31/12/2999", "31/12/2999"],
        "STATUT":     ["Decede",     "Vivant",     "Vivant"],
    })
    ds = {
        "column_mapping": {
            "date_naissance": "CLINAISS",
            "date_entree":    "CTREFFET",
            "date_sortie":    "DATE_SORTIE",
            "cause_sortie":   "STATUT",
        },
        "column_mapping_confirmed": True,
        "value_mapping": {
            "cause_sortie": {"Decede": "deces", "Vivant": "autre"},
        },
        "value_mapping_confirmed": True,
    }
    out = maybe_normalize_records(
        ds, df.to_json(orient="split"), dataset_ref="test_e2e",
    )
    assert out is not None
    norm = pd.read_parquet(out["dataset_ref_normalized"])
    # Colonnes canoniques
    assert set(norm.columns) == {"date_naissance", "date_entree",
                                  "date_sortie", "cause_sortie"}
    # Valeurs enum mappées
    assert list(norm["cause_sortie"]) == ["deces", "autre", "autre"]
    # Dates en datetime64
    assert pd.api.types.is_datetime64_any_dtype(norm["date_naissance"])
    assert pd.api.types.is_datetime64_any_dtype(norm["date_sortie"])
    # Sentinelles 2999 clippées au dernier décès (31/12/2010)
    assert (norm["date_sortie"].dt.year < 2100).all()
    assert norm["date_sortie"].max().year == 2010


def test_time_series_clips_sentinel_dates():
    """Régression bug 1 : 31/12/2999 (contrats actifs) ne doit pas
    générer une période d'observation de 1017 ans."""
    import pandas as pd
    from tools.statistical_analysis.time_series import run as _ts_run
    df = pd.DataFrame({
        "date_entree":   ["2000-01-01", "2005-06-15", "2010-03-20"],
        "date_sortie":   ["2010-12-31", "2999-12-31", "2999-12-31"],
        "cause_sortie":  ["deces",      "vivant",     "vivant"],
        "date_naissance": ["1950-01-01", "1960-01-01", "1970-01-01"],
    })
    res = _ts_run(df, {})
    assert "erreur" not in res
    assert res["annee_max"] <= 2100, (
        f"annee_max = {res['annee_max']} : sentinelle 2999 non clippée"
    )
    # Période d'observation raisonnable (≤ 100 ans)
    assert res["nb_annees"] <= 100


def test_format_cell_delegates_to_fmt_no_desync():
    """Régression : avant le fix, _format_cell de _04_redaction et _fmt
    de table_renderer étaient deux fonctions distinctes. _format_cell
    ne connaissait pas pct2 → tombait dans str(value) → '16.886858051057914'
    affiché en raw dans le PDF. Maintenant les deux convergent."""
    from agents.report.pipeline._04_redaction import _format_cell
    from tools.build_pdf.table_renderer import _fmt
    # Toute valeur formatée par _fmt doit donner le même résultat via _format_cell
    for fmt in ("int", "float1", "float2", "float4", "pct1", "pct2", "sci"):
        val = 16.886858051057914
        assert _format_cell(val, fmt) == _fmt(val, fmt), (
            f"Désync entre _format_cell et _fmt pour format={fmt}"
        )
    # Cas spécifique du bug : pct2 doit retourner "16.89 %" pas la valeur raw
    assert _format_cell(16.886858051057914, "pct2") == "16.89 %"
    assert "16.886858" not in _format_cell(16.886858051057914, "float1")


def test_format_placeholder_value_no_scientific_notation():
    """Régression bug 4 : les nombres dans la narrative ne doivent
    jamais sortir en notation scientifique."""
    from knowledge_base.report_template.template_loader import (
        _format_placeholder_value, resolve_placeholders,
    )
    # Grand entier-float → séparateur milliers
    assert _format_placeholder_value(6082714.0) == "6 082 714"
    assert _format_placeholder_value(94282) == "94 282"
    # Très grand nombre → toujours pas de notation sci
    out = _format_placeholder_value(1.5e9)
    assert "e+" not in out and "E+" not in out
    # Float « normal »
    assert _format_placeholder_value(3.14159) == "3.14"
    # Intégration template
    text = "Total : {{ total_exposure }} années-personne, {{ n_deaths }} décès."
    ds = {"total_exposure": 6082714.05, "n_deaths": 94282}
    out = resolve_placeholders(text, ds)
    assert "6 082 714" in out
    assert "94 282" in out
    assert "e+" not in out


def test_value_mapping_synonyms_cover_numeric_sexe():
    """Régression bug 2 : 1/2 doivent être reconnus comme H/F par le
    matching synonyme (convention INSEE)."""
    from tools.master.suggest_value_mapping import _match_canonical
    assert _match_canonical("1", ["H", "F"]) == "H"
    assert _match_canonical("2", ["H", "F"]) == "F"
    # Sanity : labels canon eux-mêmes
    assert _match_canonical("H", ["H", "F"]) == "H"
    assert _match_canonical("homme", ["H", "F"]) == "H"


def test_column_mapping_autodetect_renames_csv_cols():
    """Régression : sans column_mapping confirmé via UI, les colonnes CSV
    brutes (CLINAISS, STATUT, …) doivent quand même être renommées en
    canoniques (date_naissance, cause_sortie, …) via auto-détection
    depuis COLUMN_SCHEMA. Sinon clean_records crashe."""
    import pandas as pd
    from agents.mortality.dictionary.column_schema import COLUMN_SCHEMA, find_col

    df = pd.DataFrame({
        "CTREFFET":   ["2020-01-01"],
        "DATE_SORTIE": ["2999-12-31"],
        "CLINAISS":   ["1950-06-15"],
        "STATUT":     ["ACTIF"],
        "SEXEREF":    ["H"],
        "CDPROD":     ["P1"],
        "CTRNUM":     ["C1"],
    })
    mapping = {
        role: find_col(df, info["candidates"])
        for role, info in COLUMN_SCHEMA.items()
    }
    mapping = {k: v for k, v in mapping.items() if v}
    rename_map = {v: k for k, v in mapping.items()}
    df = df.rename(columns=rename_map)
    assert "date_naissance" in df.columns
    assert "date_sortie"    in df.columns
    assert "cause_sortie"   in df.columns
    assert "sexe"           in df.columns


def test_msgpack_safe_handles_dataframe():
    import pandas as pd
    from agents.mortality.agents.tools_node import _msgpack_safe
    df = pd.DataFrame({"age": [20, 21], "qx": [0.001, 0.002]})
    out = _msgpack_safe(df)
    assert out == [{"age": 20, "qx": 0.001}, {"age": 21, "qx": 0.002}]


def test_classify_intent_accepts_known_context(monkeypatch):
    """Régression : si gender est déjà connu d'un tour précédent, le
    prompt doit l'indiquer au LLM pour éviter qu'il dise 'inconnu'."""
    from agents.master import classify_intent as ci

    captured_prompts = []

    class _FakeMsg:
        def __init__(self, content): self.content = content
    class _FakeChoice:
        def __init__(self, content): self.message = _FakeMsg(content)
    class _FakeResp:
        def __init__(self, content): self.choices = [_FakeChoice(content)]
    def _fake_call_with_retry(client, **kw):
        captured_prompts.append(kw["messages"][0]["content"])
        return _FakeResp(
            '{"kind":"task","write":"yes","report_mode":"full_report",'
            '"gender_segmentation":"unknown","confidence":0.9,'
            '"reasoning":"","reply":"ok"}'
        )
    monkeypatch.setattr(ci, "call_with_retry", _fake_call_with_retry,
                        raising=False)

    import openai as _real_openai
    class _FakeClient: ...
    monkeypatch.setattr(_real_openai, "OpenAI", lambda *a, **k: _FakeClient())
    # Patch _llm_classify path : la fonction utilise call_with_retry importé
    # localement dans le module classify_intent.
    monkeypatch.setattr(
        "agents.mortality.agents._utils.call_with_retry",
        _fake_call_with_retry,
    )

    ci.classify_intent(
        "oui", has_data=True, has_calcs=False,
        known_context={"gender_segmentation": "unisex", "report_mode": "full_report"},
    )
    assert captured_prompts, "Le LLM n'a pas été appelé"
    prompt = captured_prompts[0]
    assert "gender_segmentation déjà connu : unisex" in prompt
    assert "report_mode déjà connu : full_report" in prompt


def test_resolve_user_answer_typos_space_separated():
    """Le user tape les noms de méthodes avec typos et espaces (vu en prod)."""
    from agents.master.method_choices import resolve_user_answer_to_method
    ans = "kaplain meier, whittaker, confidence_intervals"
    assert resolve_user_answer_to_method(
        ans, ["central", "binomial", "kaplan_meier"],
    ) == "kaplan_meier"


def test_master_parses_inline_methods_in_meta_question(monkeypatch):
    """Quand l'utilisateur répond à la méta-question 'auto/préciser' en
    donnant directement les méthodes, le Master doit les enregistrer et
    router vers Builder sans re-questionner."""
    from agents.mortality.agents import master_node as mn
    monkeypatch.setattr(mn, "_classify_intent", _fake_classify("yes", "full_report"))

    state = {
        "messages": [
            HumanMessage(content=(
                "pour les taux bruts, sélectionne kaplan_meir. "
                "pour le lissage applique Whittaker, "
                "validation via confidence_intervals"
            )),
        ],
        "data_store": {
            "_disambiguation_done": True,
            "study_plan":           {"gender_segmentation": "unisex"},
            "_pending_need": {
                "context_key": "methods_choice_mode",
                "question":    "...",
                "options":     ["preciser", "auto"],
                "default":     "auto",
            },
        },
        "dataset_ref": None,
    }
    out = mn.master_node(state)
    methods = (out["data_store"].get("study_plan") or {}).get("methods") or {}
    assert methods.get("builder.crude_rates") == "kaplan_meier"
    assert methods.get("builder.smoothing")   == "whittaker"
    assert methods.get("builder.validation")  == "confidence_intervals"
    assert out.get("active_agent") == "builder"


# ──────────────────────────────────────────────────────────────────────────
# 2. Master : pose la méta-question quand pertinent
# ──────────────────────────────────────────────────────────────────────────

def _fake_classify(write="yes", report_mode="full_report"):
    def _f(*a, **kw):
        return {"kind": "task", "write": write, "report_mode": report_mode,
                "intent": "build_and_write", "reply": ""}
    return _f


def test_master_asks_method_meta_question_when_choices_pending(monkeypatch):
    from agents.mortality.agents import master_node as mn
    monkeypatch.setattr(mn, "_classify_intent", _fake_classify("yes", "raw_rates"))
    state = {
        "messages":    [HumanMessage(content="taux bruts svp")],
        "data_store":  {
            "_disambiguation_done": True,
            "study_plan":           {"gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
    }
    out = mn.master_node(state)
    pending = out["data_store"].get("_pending_need") or {}
    assert pending.get("context_key") == "methods_choice_mode"
    assert "préciser" in pending.get("question", "").lower()


def test_master_skips_method_question_when_methods_auto_already_set(monkeypatch):
    from agents.mortality.agents import master_node as mn
    monkeypatch.setattr(mn, "_classify_intent", _fake_classify("yes", "raw_rates"))
    state = {
        "messages":    [HumanMessage(content="taux bruts svp")],
        "data_store":  {
            "_disambiguation_done":   True,
            "_methods_question_done": True,
            "study_plan":             {"gender_segmentation": "unisex",
                                       "methods_auto":        True},
        },
        "dataset_ref": None,
    }
    out = mn.master_node(state)
    # méta-question NON posée → route vers Builder
    assert out.get("active_agent") == "builder"


# ──────────────────────────────────────────────────────────────────────────
# 3. tools_node._inject_user_method : injection au moment de l'appel
# ──────────────────────────────────────────────────────────────────────────

def test_inject_user_method_writes_chosen_method_into_params():
    from agents.mortality.agents.tools_node import _inject_user_method
    params = {}
    study_plan = {"methods": {"builder.crude_rates": "kaplan_meier"}}
    _inject_user_method("builder", "crude_rates", params, study_plan)
    assert params.get("method") == "kaplan_meier"


def test_inject_user_method_skips_when_auto():
    from agents.mortality.agents.tools_node import _inject_user_method
    params = {}
    study_plan = {"methods": {"builder.crude_rates": "auto"}}
    _inject_user_method("builder", "crude_rates", params, study_plan)
    assert "method" not in params


def test_inject_user_method_skips_when_no_choice_for_tool():
    from agents.mortality.agents.tools_node import _inject_user_method
    params = {}
    study_plan = {"methods": {"builder.smoothing": "whittaker_henderson"}}
    _inject_user_method("builder", "crude_rates", params, study_plan)
    assert "method" not in params


def test_inject_user_method_overrides_llm_choice():
    """Le choix utilisateur prime sur ce que le LLM a placé dans params."""
    from agents.mortality.agents.tools_node import _inject_user_method
    params = {"method": "central"}
    study_plan = {"methods": {"builder.crude_rates": "binomial"}}
    _inject_user_method("builder", "crude_rates", params, study_plan)
    assert params["method"] == "binomial"


def test_inject_user_method_uses_function_name_param_for_validation():
    """Pour builder.validation, le param-méthode s'appelle `function_name`."""
    from agents.mortality.agents.tools_node import _inject_user_method
    params = {}
    study_plan = {"methods": {"builder.validation": "chi_square"}}
    _inject_user_method("builder", "validation", params, study_plan)
    assert params.get("function_name") == "chi_square"


# ──────────────────────────────────────────────────────────────────────
# Échappe-question pendant pending_need méthode
# ──────────────────────────────────────────────────────────────────────

def test_is_meta_question_detects_questions():
    """Régression : 'rappelle moi les méthodes' doit être détecté comme
    question (vs choix de méthode silencieusement rejeté en re-ask)."""
    from agents.master.method_choices import _is_meta_question
    assert _is_meta_question("rappelle moi les méthodes")
    assert _is_meta_question("c'est quoi le central ?")
    assert _is_meta_question("explique-moi kaplan")
    assert _is_meta_question("comment ça marche ?")
    # Faux positifs à éviter
    assert not _is_meta_question("central")
    assert not _is_meta_question("kaplan_meier")
    assert not _is_meta_question("auto")
    assert not _is_meta_question("préciser")


def test_question_without_pending_routes_to_doctrine(monkeypatch):
    """Régression : une question SANS pending doit être traitée
    EXACTEMENT comme une question AVEC pending → search_doctrine direct,
    pas de LLM nano. Uniformité du traitement."""
    from agents.master import method_choices as mc
    fake_calls = []
    def _fake_search(df, params):
        fake_calls.append(params)
        return {"results": [{
            "doc_id": "D03", "section_id": "D03.02",
            "section_title": "Whittaker-Henderson 1D",
            "text": "Méthode de lissage non-paramétrique...",
        }], "n_returned": 1, "query_used": params.get("query", "")}
    import sys
    fake_mod = type(sys)("tools.conversation.search_doctrine")
    fake_mod.run = _fake_search
    monkeypatch.setitem(sys.modules, "tools.conversation.search_doctrine", fake_mod)

    # Sans pending → answer_question_via_doctrine(..., pending=None)
    data_store = {"_stage_buffer": []}
    out = mc.answer_question_via_doctrine(
        "C'est quoi le lissage Whittaker ?", data_store, pending=None,
    )
    assert len(fake_calls) == 1
    msg = out["messages"][0].content
    assert "D03.02" in msg
    # Sans pending : pas de "Reprenons" à la fin
    assert "Reprenons" not in msg
    # Stage 0.e-q tracé (pas 0.c-q)
    stages = [e for e in (data_store.get("_stage_buffer") or [])
              if e.get("type") == "master_stage"]
    # Buffer consommé par _ask_user → _prepend_stages — donc absent ici
    # On vérifie au moins que ça n'a pas planté
    assert "events" in out


def test_method_pending_with_question_routes_to_doctrine(monkeypatch):
    """Régression bug terrain : pendant le pending_need méthode, si l'user
    pose une question, on ne doit PAS re-asker en boucle 'Je n'ai pas
    compris' — on doit appeler search_doctrine et re-poser la question
    méthode après."""
    # Mock search_doctrine pour éviter d'invoquer le retriever réel
    from agents.master import method_choices as mc
    fake_calls = []
    def _fake_search(df, params):
        fake_calls.append(params)
        return {"results": [{
            "doc_id": "D02", "section_id": "D02.05",
            "section_title": "Kaplan-Meier",
            "text": "Estimateur non-paramétrique...",
        }], "n_returned": 1, "query_used": params.get("query", "")}
    # Patch via import dynamique (le tool est importé à l'intérieur de
    # _answer_meta_question_keeping_pending)
    import sys
    fake_mod = type(sys)("tools.conversation.search_doctrine")
    fake_mod.run = _fake_search
    monkeypatch.setitem(sys.modules, "tools.conversation.search_doctrine", fake_mod)

    pending = {
        "context_key": "methods_choice_mode",
        "question": "Voulez-vous (a) préciser ou (b) auto ?",
        "options": ["preciser", "auto"],
    }
    data_store = {}
    out = mc.handle_methods_choice_response(
        pending, "rappelle moi les méthodes", data_store, report_mode="full_report",
    )
    # search_doctrine doit avoir été appelé
    assert len(fake_calls) == 1
    assert "méthodes" in fake_calls[0].get("query", "").lower()
    # Pas de routing Builder, pas de pop du pending → on attend toujours
    assert out.get("active_agent") != "builder"
    # Le message doit contenir l'extrait du chunk + la question méthode
    msg_content = out["messages"][0].content if out.get("messages") else ""
    assert "D02" in msg_content
    assert "Reprenons" in msg_content or "préciser" in msg_content
