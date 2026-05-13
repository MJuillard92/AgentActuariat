"""
TOOL CONTRACT — conversation.eval_pandas
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : conversation.eval_pandas
domain        : conversation
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-05-13

DESCRIPTION
-----------
Escape hatch d'analyse libre : exécute une expression Python générée
par le LLM, restreinte à pandas / numpy / scipy.stats / matplotlib /
seaborn / lifelines, après validation AST stricte (pas d'import,
pas de filesystem, pas de réseau, pas de dunders).

WHEN TO USE
-----------
Quand aucun tool structuré ne répond exactement au besoin :
  - Filtrage + agrégation ad hoc
  - Test statistique non listé (chi2, log-rank, Cox quick fit)
  - Plot non-standard (KM superposés, heatmap…)
La capture matplotlib auto retourne le PNG produit.

WHEN NOT TO USE
---------------
Si un tool structuré existe (data_inspect, plot_basic, statistical_analysis.*).
Pas d'écriture filesystem : `to_csv`, `savefig`, `read_*` sont bloqués.

INPUTS
------
params:
  expression:
    type    : string
    note    : Expression Python pure (pas de statement, pas d'assignation).
              Variables disponibles : df, pd, np, stats, plt, sns, ll,
              datetime, date, timedelta.

OUTPUTS
-------
return_payload:
  value : any         — résultat sérialisable (scalaire / list / dict / records)
  plots : list[str]   — paths des PNG capturés (si plt utilisé)
  expression : str    — expression évaluée (echo)

AGENT GUIDANCE
--------------
reasoning_hint: >
  Préférer une expression unique compacte. Pour plots, utiliser
  `df['age'].hist()` ou `ll.KaplanMeierFitter().fit(...).plot()` :
  les figures matplotlib sont capturées automatiquement.

CATALOGUE METADATA
------------------
display_name      : Évaluation pandas libre (sandboxée)
short_description : Expression pandas/numpy/lifelines évaluée après validation AST.
domain            : conversation
capability_group  : data_exploration
client_visible    : true
"""
from __future__ import annotations

import ast
import math
import time
from pathlib import Path

import pandas as pd
import numpy as np


# ── AST whitelist ────────────────────────────────────────────────────────────

# Noms identifiés comme dangereux — refusés AVANT exécution.
_FORBIDDEN_NAMES: set[str] = {
    "open", "exec", "eval", "compile", "__import__",
    "getattr", "setattr", "delattr", "hasattr",
    "globals", "locals", "vars", "input", "breakpoint",
    "memoryview", "object", "type", "super",
    "exit", "quit", "help",
}

# Méthodes I/O bloquées même sur df / pd / np / plt / ll.
# (le AST walker détecte `.to_csv`, `.savefig`, `.read_csv`, etc.)
_FORBIDDEN_ATTRIBUTES: set[str] = {
    # pandas I/O
    "to_csv", "to_pickle", "to_excel", "to_parquet", "to_sql",
    "to_json", "to_html", "to_feather", "to_hdf", "to_stata",
    "to_msgpack", "to_xml", "to_orc", "to_latex", "to_clipboard",
    "read_csv", "read_pickle", "read_excel", "read_parquet", "read_sql",
    "read_json", "read_html", "read_feather", "read_hdf", "read_stata",
    "read_xml", "read_orc",
    # numpy I/O
    "save", "savez", "savez_compressed", "savetxt", "load", "loadtxt",
    "fromfile", "tofile", "memmap",
    # matplotlib I/O
    "savefig", "imsave", "imread",
}

# Builtins explicitement autorisés (le reste est filtré).
_ALLOWED_BUILTINS: dict = {
    "abs":       abs,
    "min":       min,
    "max":       max,
    "sum":       sum,
    "len":       len,
    "round":     round,
    "sorted":    sorted,
    "reversed":  reversed,
    "list":      list,
    "dict":      dict,
    "set":       set,
    "tuple":     tuple,
    "int":       int,
    "float":     float,
    "str":       str,
    "bool":      bool,
    "range":     range,
    "zip":       zip,
    "enumerate": enumerate,
    "map":       map,
    "filter":    filter,
    "any":       any,
    "all":       all,
}


class ExpressionRejected(ValueError):
    """L'expression contient une construction interdite par la whitelist AST."""


def _validate_ast(expression: str) -> None:
    """Parse l'expression et lève ExpressionRejected si interdite.
    Aucune exécution Python n'a lieu ici."""
    try:
        tree = ast.parse(expression, mode="exec")
    except SyntaxError as exc:
        raise ExpressionRejected(f"syntaxe invalide : {exc.msg}") from exc

    # Expression pure (pas de statement) ou max 1 Expression au top-level.
    for node in tree.body:
        if not isinstance(node, ast.Expr):
            raise ExpressionRejected(
                f"statement '{type(node).__name__}' interdit — "
                f"expression pure uniquement (pas d'assignation, de def, de import)"
            )

    for node in ast.walk(tree):
        # Imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ExpressionRejected("import interdit")
        # Statements de contrôle
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.For, ast.AsyncFor, ast.While, ast.With,
                             ast.AsyncWith, ast.Try, ast.Raise, ast.Global,
                             ast.Nonlocal, ast.Assign, ast.AugAssign, ast.AnnAssign,
                             ast.Delete)):
            raise ExpressionRejected(
                f"statement '{type(node).__name__}' interdit"
            )
        # Noms interdits
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ExpressionRejected(f"nom interdit : '{node.id}'")
        # Attributs interdits (méthodes I/O ou dunders)
        if isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_ATTRIBUTES:
                raise ExpressionRejected(
                    f"méthode interdite : '.{node.attr}' (I/O filesystem bloquée)"
                )
            if node.attr.startswith("_"):
                raise ExpressionRejected(
                    f"attribut dunder/privé interdit : '.{node.attr}'"
                )


# ── Exécution sandboxée ──────────────────────────────────────────────────────

def _build_namespace(df: pd.DataFrame) -> dict:
    """Construit le namespace d'exécution. Volontairement fixe et public."""
    import scipy.stats as _stats
    import matplotlib
    matplotlib.use("Agg")  # backend non-interactif
    import matplotlib.pyplot as _plt
    from datetime import datetime as _dt, date as _date, timedelta as _td

    ns = {
        "__builtins__": _ALLOWED_BUILTINS,
        "df":           df,
        "pd":           pd,
        "np":           np,
        "stats":        _stats,
        "plt":          _plt,
        "datetime":     _dt,
        "date":         _date,
        "timedelta":    _td,
    }
    # Imports optionnels (peuvent ne pas être installés en CI minimal)
    try:
        import seaborn as _sns
        ns["sns"] = _sns
    except ImportError:
        pass
    try:
        import lifelines as _ll
        ns["ll"] = _ll
    except ImportError:
        pass
    return ns


def _capture_plots() -> list[str]:
    """Sauvegarde toute figure matplotlib produite pendant l'eval, retourne
    les paths PNG. Aucune méthode savefig accessible au LLM."""
    import matplotlib.pyplot as plt
    saved: list[str] = []
    out_dir = Path("tmp/conversation_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    for i, num in enumerate(plt.get_fignums()):
        fig = plt.figure(num)
        # Heuristique : ignorer figures vides
        if not any(ax.has_data() for ax in fig.axes):
            continue
        path = out_dir / f"eval_{ts}_{i}.png"
        try:
            fig.savefig(path, dpi=100, bbox_inches="tight")
            saved.append(str(path))
        except Exception:
            pass
    plt.close("all")
    return saved


_MAX_RESULT_ROWS = 100_000


def _serialize_result(v):
    """Convertit récursivement en types JSON-safe. Refuse > 100k records."""
    if v is None:
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        fv = float(v)
        return fv if fv == fv and not math.isinf(fv) else None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, pd.DataFrame):
        if len(v) > _MAX_RESULT_ROWS:
            return {"erreur": f"résultat trop volumineux ({len(v)} lignes > {_MAX_RESULT_ROWS}). "
                              "Affine ta requête (filter, head, agg)."}
        return [_serialize_result(r) for r in v.head(_MAX_RESULT_ROWS).to_dict(orient="records")]
    if isinstance(v, pd.Series):
        if len(v) > _MAX_RESULT_ROWS:
            return {"erreur": f"résultat trop volumineux ({len(v)} lignes > {_MAX_RESULT_ROWS})."}
        return _serialize_result(v.to_dict())
    if isinstance(v, np.ndarray):
        if v.size > _MAX_RESULT_ROWS:
            return {"erreur": f"array trop volumineux ({v.size} éléments)."}
        return [_serialize_result(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _serialize_result(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_serialize_result(x) for x in v]
    if isinstance(v, (str, int, bool)):
        return v
    if isinstance(v, float):
        return v if v == v and v not in (float("inf"), float("-inf")) else None
    # Type inconnu — repr en string pour ne pas planter
    return repr(v)[:500]


def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    params = params or {}
    expression = params.get("expression", "")
    if not expression or not isinstance(expression, str):
        return {"erreur": "param 'expression' (string) requis"}

    # 1) Validation AST stricte AVANT toute exécution
    try:
        _validate_ast(expression)
    except ExpressionRejected as exc:
        return {"erreur": f"expression refusée : {exc}", "expression": expression}

    if df is None:
        return {"erreur": "DataFrame indisponible (pas de dataset chargé)."}

    # 2) Exécution dans namespace restreint
    ns = _build_namespace(df)
    import matplotlib.pyplot as plt
    plt.close("all")  # reset l'état avant l'eval

    try:
        compiled = compile(expression, "<eval_pandas>", "eval")
    except SyntaxError as exc:
        # L'expression n'est peut-être pas une expression "single" (eval mode).
        # On retombe sur exec mode et on capture la dernière valeur si possible.
        return {"erreur": f"syntaxe : {exc.msg}", "expression": expression}

    try:
        value = eval(compiled, ns, ns)  # noqa: S307 — namespace contrôlé
    except Exception as exc:
        return {"erreur": f"erreur d'exécution : {type(exc).__name__}: {exc}",
                "expression": expression}

    # 3) Capture des plots éventuels
    plots = _capture_plots()

    # 4) Sérialisation safe
    return {
        "expression": expression,
        "value":      _serialize_result(value),
        "plots":      plots,
    }
