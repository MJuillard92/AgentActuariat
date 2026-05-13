"""
TOOL CONTRACT — conversation.data_inspect
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : conversation.data_inspect
domain        : conversation
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-05-13

DESCRIPTION
-----------
Outil d'inspection rapide du DataFrame en mode conversationnel.
Permet à l'utilisateur de vérifier le contenu du fichier (colonnes,
types, échantillons, distribution) sans lancer le pipeline actuariel.

WHEN TO USE
-----------
Phase conversationnelle (kind=question). L'utilisateur veut "voir"
le fichier avant de demander des calculs : "quelles colonnes ?",
"montre-moi les 5 premières lignes", "quelles valeurs prend STATUT ?".

WHEN NOT TO USE
---------------
Pas en pipeline actuariel — utiliser builder.* ou
statistical_analysis.* qui produisent des résultats normés.

PREREQUISITES
-------------
required_data_store_keys: []
Note: le DataFrame est passé directement (chargé par tools_node).

INPUTS
------
params:
  function_name:
    type    : string
    values  : columns | shape | head | describe | value_counts | date_range
    default : columns
  column:
    type    : string
    default : null
    note    : Nom de colonne pour value_counts / date_range.
  n:
    type    : int
    default : 5
    note    : Nombre de lignes pour head, top-n pour value_counts.

OUTPUTS
-------
return_payload:
  function_name : str — fonction appelée
  result        : dict | list — payload sérialisable JSON (jamais de DataFrame)

AGENT GUIDANCE
--------------
reasoning_hint: >
  Préférer `data_inspect.columns` pour découvrir la structure avant
  toute autre exploration. `value_counts` sur une catégorielle pour
  voir les modalités (sexe, statut). `describe` sur les numériques.

CATALOGUE METADATA
------------------
display_name      : Inspection rapide du DataFrame
short_description : Headers, types, échantillon, distribution.
domain            : conversation
capability_group  : data_exploration
depends_on        : []
required_by       : []
client_visible    : true
"""
from __future__ import annotations

import pandas as pd


def _serialize(v):
    """Convertit récursivement en types Python natifs (JSON-safe)."""
    import math as _math
    import numpy as _np
    if v is None:
        return None
    if isinstance(v, (_np.bool_,)):
        return bool(v)
    if isinstance(v, (_np.integer,)):
        return int(v)
    if isinstance(v, (_np.floating,)):
        fv = float(v)
        return fv if fv == fv and not _math.isinf(fv) else None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return [_serialize(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _serialize(val) for k, val in v.items()}
    if isinstance(v, (str, int, bool)):
        return v
    if isinstance(v, float):
        return v if v == v and v not in (float("inf"), float("-inf")) else None
    return str(v)


def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    params = params or {}
    fn = params.get("function_name", "columns")
    column = params.get("column")
    n = int(params.get("n", 5))

    if df is None or len(df) == 0:
        return {"erreur": "DataFrame vide ou indisponible."}

    if fn == "columns":
        result = [
            {"name": str(c), "dtype": str(df[c].dtype)}
            for c in df.columns
        ]
        return {"function_name": fn, "result": result}

    if fn == "shape":
        return {"function_name": fn,
                "result": {"rows": int(len(df)), "cols": int(len(df.columns))}}

    if fn == "head":
        rows = df.head(n).to_dict(orient="records")
        return {"function_name": fn, "result": _serialize(rows)}

    if fn == "describe":
        try:
            desc = df.describe(include="number").to_dict()
        except Exception:
            return {"erreur": "Aucune colonne numérique à décrire."}
        return {"function_name": fn, "result": _serialize(desc)}

    if fn == "value_counts":
        if not column or column not in df.columns:
            return {"erreur": f"Colonne '{column}' absente. Disponibles : {list(df.columns)}"}
        vc = df[column].value_counts(dropna=False).head(n)
        return {"function_name": fn,
                "result": _serialize(vc.to_dict()),
                "column": column}

    if fn == "date_range":
        if not column or column not in df.columns:
            return {"erreur": f"Colonne '{column}' absente."}
        ser = pd.to_datetime(df[column], format="mixed", dayfirst=True, errors="coerce")
        valid = ser.dropna()
        if len(valid) == 0:
            return {"erreur": f"Colonne '{column}' non parsable en date."}
        return {
            "function_name": fn,
            "column":        column,
            "result": {
                "min":     valid.min().isoformat(),
                "max":     valid.max().isoformat(),
                "n_valid": int(len(valid)),
                "n_nat":   int(ser.isna().sum()),
            },
        }

    return {"erreur": f"function_name inconnu : '{fn}'. "
                       f"Valeurs : columns | shape | head | describe | value_counts | date_range"}
