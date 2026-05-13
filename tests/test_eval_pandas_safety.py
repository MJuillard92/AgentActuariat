"""Tests sécurité du tool conversation.eval_pandas.

Couvre :
  - Cas autorisés : pandas, numpy, scipy.stats, matplotlib avec capture, lifelines.
  - Cas refusés (10+) : chacun doit lever ExpressionRejected AVANT exécution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _df():
    return pd.DataFrame({
        "age":          [20, 30, 40, 50, 60, 70, 80],
        "sexe":         ["H", "F", "H", "F", "H", "F", "H"],
        "duree_obs":    [10.0, 5.0, 8.0, 3.0, 12.0, 6.0, 9.0],
        "deces":        [0, 1, 0, 1, 0, 1, 1],
        "taux":         [0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08],
    })


# ──────────────────────────────────────────────────────────────────────
# Cas autorisés
# ──────────────────────────────────────────────────────────────────────

def test_allowed_df_head():
    from tools.conversation.eval_pandas import run
    res = run(_df(), {"expression": "df.head(3)"})
    assert "erreur" not in res
    assert len(res["value"]) == 3


def test_allowed_df_columns():
    from tools.conversation.eval_pandas import run
    res = run(_df(), {"expression": "df.columns.tolist()"})
    assert "erreur" not in res
    assert "age" in res["value"]


def test_allowed_query_groupby():
    from tools.conversation.eval_pandas import run
    res = run(_df(), {"expression": "df.query('age > 30').groupby('sexe').size().to_dict()"})
    assert "erreur" not in res
    assert isinstance(res["value"], dict)


def test_allowed_numpy_percentile():
    from tools.conversation.eval_pandas import run
    res = run(_df(), {"expression": "np.percentile(df['taux'], [25, 50, 75]).tolist()"})
    assert "erreur" not in res
    assert len(res["value"]) == 3


def test_allowed_scipy_stats():
    """scipy.stats.chi2_contingency sur un crosstab."""
    from tools.conversation.eval_pandas import run
    res = run(_df(), {
        "expression": "stats.chi2_contingency(pd.crosstab(df['sexe'], df['deces']))[1]",
    })
    assert "erreur" not in res
    # Le 1er élément du tuple chi2_contingency est p_value (float)
    assert isinstance(res["value"], (int, float))


def test_allowed_matplotlib_with_capture(tmp_path, monkeypatch):
    """plt.hist appelé via df → un PNG capturé dans plots."""
    import os
    monkeypatch.chdir(tmp_path)
    from tools.conversation.eval_pandas import run
    res = run(_df(), {"expression": "df['age'].hist(bins=5)"})
    assert "erreur" not in res
    assert len(res["plots"]) >= 1
    assert os.path.exists(res["plots"][0])


def test_allowed_lifelines_kaplan_meier():
    """ll.KaplanMeierFitter — disponible si lifelines installé."""
    pytest.importorskip("lifelines")
    from tools.conversation.eval_pandas import run
    res = run(_df(), {
        "expression": "ll.KaplanMeierFitter().fit(df['duree_obs'], df['deces']).median_survival_time_",
    })
    assert "erreur" not in res, f"erreur inattendue : {res.get('erreur')}"


# ──────────────────────────────────────────────────────────────────────
# Cas refusés — validation AST avant toute exécution
# ──────────────────────────────────────────────────────────────────────

_FORBIDDEN_EXPRESSIONS = [
    # Imports
    ("import os",                          "import"),
    ("from os import path",                "import"),
    # Builtins dangereux
    ("open('/etc/passwd').read()",         "open"),
    ("exec('import os')",                  "exec"),
    ("eval('1+1')",                        "eval"),
    ("compile('1+1', '<x>', 'eval')",      "compile"),
    ("__import__('os')",                   "__import__"),
    ("getattr(df, '__class__')",           "getattr"),
    ("setattr(df, 'x', 1)",                "setattr"),
    ("globals()",                          "globals"),
    ("locals()",                           "locals"),
    ("vars(df)",                           "vars"),
    # Dunders
    ("df.__class__.__bases__",             "dunder"),
    ("df.__dict__",                        "dunder"),
    ("df.__getattribute__('values')",      "dunder"),
    # I/O filesystem
    ("df.to_csv('/tmp/leak.csv')",         "to_csv"),
    ("df.to_pickle('/tmp/d.pkl')",         "to_pickle"),
    ("df.to_parquet('/tmp/d.parquet')",    "to_parquet"),
    ("np.save('/tmp/d', df.values)",       "save"),
    ("plt.savefig('/tmp/p.png')",          "savefig"),
    ("pd.read_csv('/etc/passwd')",         "read_csv"),
    ("np.load('/etc/passwd')",             "load"),
    # Statements
    ("x = 1",                              "Assign"),
    ("for i in range(10): pass",           "For"),
    ("while True: pass",                   "While"),
    ("def f(): pass",                      "FunctionDef"),
    ("class C: pass",                      "ClassDef"),
    # Lambda contenant un nom interdit
    ("(lambda: __import__('os'))()",       "__import__"),
]


@pytest.mark.parametrize("expression,reason_hint", _FORBIDDEN_EXPRESSIONS)
def test_forbidden_expression_rejected(expression, reason_hint):
    """Chaque expression dangereuse doit retourner {erreur} SANS exécution."""
    from tools.conversation.eval_pandas import run
    res = run(_df(), {"expression": expression})
    assert "erreur" in res, (
        f"Expression non refusée : {expression!r} → {res}"
    )
    # Le message d'erreur doit pointer vers la raison (utile pour debug)
    err_lower = res["erreur"].lower()
    hint_lower = reason_hint.lower()
    # Tolérant : le hint apparaît dans le message OU le message indique "refusée"
    assert hint_lower in err_lower or "refus" in err_lower or "interdit" in err_lower, (
        f"Erreur peu claire pour {expression!r} : {res['erreur']}"
    )


def test_no_filesystem_write_before_rejection(tmp_path, monkeypatch):
    """Sanity : df.to_csv refusé NE doit PAS écrire le fichier."""
    import os
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "leak.csv"
    from tools.conversation.eval_pandas import run
    res = run(_df(), {"expression": f"df.to_csv('{target}')"})
    assert "erreur" in res
    assert not target.exists(), "Le fichier a été écrit malgré le refus !"
