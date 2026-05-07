"""Bug : `_classify_intent` crash quand un builder_output (ex: cleaned_records)
est un DataFrame, à cause de `all(data_store.get(k) for k in builder_keys)`
qui appelle `bool(df)` → ValueError pandas."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_classify_intent_does_not_crash_on_dataframe_in_data_store(monkeypatch):
    """Reproduit la trace : has_calcs = bool(builder_keys) and all(data_store.get(k) ...).
    Si l'une des clés est un DataFrame, all() fait bool(df) → ValueError."""
    from agents.mortality.agents import master_node as mn

    # Mock LLM pour ne pas faire de vrai appel
    def _fake_call(*a, **kw):
        class _R:
            class _C:
                class _M:
                    content = '{"kind":"task","write":"yes","report_mode":"full_report","reply":""}'
                message = _M()
            choices = [_C()]
        return _R()

    monkeypatch.setattr(
        "agents.mortality.agents._utils.call_with_retry", _fake_call,
    )
    # Bypasser la création du client OpenAI
    class _FakeClient:
        def __init__(self, *a, **kw): pass
    monkeypatch.setattr("openai.OpenAI", _FakeClient)

    df = pd.DataFrame({"id": [1, 2, 3], "sexe": ["H", "F", "H"]})
    # data_store contient un DataFrame sous une clé builder_output
    data_store = {
        "cleaned_records": df,        # ← potentiellement piégeur
        "total_exposure":  1234.5,
        "total_deaths":    42,
    }
    # Ne doit PAS lever ValueError pandas
    result = mn._classify_intent("construit un rapport", data_store, dataset_ref=None)
    assert "kind" in result
    assert result["kind"] == "task"
