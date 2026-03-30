"""
tool_registry.py
Registre central des tools actuariels.

Fait le lien entre :
  - builder_capabilities.json  (catalogue)
  - format OpenAI function-calling
  - exécution réelle des modules Python

Expose :
  get_capabilities() -> dict         — charge builder_capabilities.json
  get_openai_tools() -> list[dict]   — format OpenAI function-calling
  call_tool(tool_name, function_name, params, df, data) -> dict

Routing par tool_name :
  statistical_analysis → run(df, params)
  builder              → run(df, params)   — pour exposure ; sinon run(data, params)
  graphs               → run(data, params)
  reasoning            → run(context, params)
  build_pdf            → run(data, params)
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import pandas as pd

_CAPABILITIES_PATH = Path(__file__).parent.parent / "builder_capabilities.json"

# Tools qui reçoivent un DataFrame comme premier argument
_DF_TOOLS = {"statistical_analysis"}

# Fonctions builder qui nécessitent le df (les autres utilisent data_store)
_BUILDER_DF_FUNCTIONS = {"exposure"}


def get_capabilities() -> dict:
    """Charge et retourne builder_capabilities.json."""
    if _CAPABILITIES_PATH.exists():
        return json.loads(_CAPABILITIES_PATH.read_text(encoding="utf-8"))
    return {"version": "2.0", "tools": {}}


def get_openai_tools() -> list[dict]:
    """
    Construit la liste des tools au format OpenAI function-calling.
    Un tool par entrée dans capabilities["tools"] (fonctions disponibles uniquement).
    """
    caps = get_capabilities()
    tools = []
    for tool_name, tool_info in caps.get("tools", {}).items():
        fn_descriptions = {
            fn: info.get("description", "")
            for fn, info in tool_info.get("functions", {}).items()
            if info.get("disponible", True) is not False
        }
        if not fn_descriptions:
            continue

        fn_enum_desc = "\n".join(
            f"  - {fn}: {desc}" for fn, desc in fn_descriptions.items()
        )

        tools.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": (
                    tool_info.get("description", "")
                    + f"\n\nFonctions disponibles :\n{fn_enum_desc}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "enum": list(fn_descriptions),
                            "description": "Nom de la fonction à appeler.",
                        },
                        "params": {
                            "type": "object",
                            "description": "Paramètres optionnels de la fonction.",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["function_name"],
                },
            },
        })
    return tools


def call_tool(
    tool_name: str,
    function_name: str,
    params: dict | None,
    df: pd.DataFrame | None = None,
    data: dict | None = None,
    context: dict | None = None,
) -> dict:
    """
    Exécute la fonction demandée dans le tool.

    Arguments :
      tool_name     : "statistical_analysis" | "builder" | "graphs" | "reasoning" | "build_pdf"
      function_name : nom de la sous-fonction
      params        : paramètres libres passés à run()
      df            : DataFrame du portefeuille (statistical_analysis, builder.exposure)
      data          : dict de résultats accumulés (builder, graphs, build_pdf)
      context       : dict contextuel (reasoning)
    """
    params = params or {}
    data = data or {}

    # Vérification catalogue
    caps = get_capabilities()
    tool_info = caps.get("tools", {}).get(tool_name)
    if tool_info is None:
        hp = caps.get("hors_perimetre", {}).get(tool_name)
        if hp:
            return {"erreur": f"'{tool_name}' hors périmètre : {hp.get('raison', 'non disponible')}"}
        return {"erreur": f"Tool inconnu : '{tool_name}'"}

    fn_info = tool_info.get("functions", {}).get(function_name)
    if fn_info is None:
        return {"erreur": f"Fonction inconnue : '{tool_name}.{function_name}'"}
    if fn_info.get("disponible") is False:
        return {"erreur": f"'{tool_name}.{function_name}' : {fn_info.get('raison', 'non disponible')}"}

    # Import dynamique
    module_path = f"report_agent.tools.{tool_name}.{function_name}"
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        return {"erreur": f"Module introuvable : {module_path}"}

    if not hasattr(mod, "run"):
        return {"erreur": f"{module_path} n'expose pas de fonction run()"}

    try:
        if tool_name in _DF_TOOLS:
            # statistical_analysis : toujours run(df, params)
            if df is None:
                return {"erreur": f"{tool_name} nécessite un DataFrame (df=None)."}
            return mod.run(df, params)

        elif tool_name == "builder":
            # exposure nécessite le df ; les autres utilisent data_store
            if function_name in _BUILDER_DF_FUNCTIONS:
                if df is None:
                    return {"erreur": "builder.exposure nécessite un DataFrame (df=None)."}
                return mod.run(df, params)
            else:
                return mod.run(data, params)

        elif tool_name == "reasoning":
            # understand_request reçoit un context dict
            ctx = context or {
                "user_message": params.get("user_message", ""),
                "history": params.get("history", []),
                "csv_columns": list(df.columns) if df is not None else [],
            }
            return mod.run(ctx, params)

        else:
            # graphs, build_pdf : run(data, params)
            return mod.run(data, params)

    except Exception as exc:
        import traceback
        return {
            "erreur": f"Erreur lors de l'exécution de {tool_name}.{function_name} : {exc}",
            "traceback": traceback.format_exc(),
        }
