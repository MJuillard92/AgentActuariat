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

_CAPABILITIES_PATH = Path(__file__).parent / "builder_capabilities.json"

# Tools qui reçoivent un DataFrame comme premier argument
_DF_TOOLS = {"statistical_analysis"}

# Fonctions builder qui nécessitent le df seul comme premier argument (les autres utilisent data_store)
_BUILDER_DF_FUNCTIONS = {"exposure"}

# Fonctions builder qui nécessitent data ET df (signature run(data, params, df=None))
_BUILDER_DATA_DF_FUNCTIONS = {"cox_regression"}


def get_capabilities() -> dict:
    """Charge et retourne builder_capabilities.json."""
    if _CAPABILITIES_PATH.exists():
        return json.loads(_CAPABILITIES_PATH.read_text(encoding="utf-8"))
    return {"version": "2.0", "tools": {}}


def _build_params_schema(tool_name: str, catalogue: dict) -> dict:
    """
    Build the OpenAI JSON schema for the 'params' property of a tool,
    using the enriched catalogue (params section from tool contracts).
    """
    tools_cat = catalogue.get("tools", {})
    properties: dict = {}

    # Collect params from all functions of this tool in the catalogue
    for cat_name, cat_info in tools_cat.items():
        if not cat_name.startswith(tool_name + "."):
            continue
        fn_suffix = cat_name[len(tool_name) + 1:]
        fn_params = cat_info.get("params", {})
        for param_name, param_info in fn_params.items():
            if param_name in properties:
                continue  # already added from another function
            prop: dict = {"type": "string"}
            if isinstance(param_info, dict):
                raw_type = param_info.get("type", "string")
                # Map contract types to JSON Schema types
                type_map = {"int": "integer", "float": "number", "bool": "boolean",
                            "string": "string", "str": "string"}
                prop["type"] = type_map.get(str(raw_type).lower(), "string")

                parts = []
                if param_info.get("values"):
                    parts.append(f"Valeurs: {param_info['values']}")
                if param_info.get("default") is not None:
                    parts.append(f"Défaut: {param_info['default']}")
                if param_info.get("note"):
                    parts.append(param_info["note"])
                if parts:
                    prop["description"] = " | ".join(parts)
            properties[param_name] = prop

    # Always include function_name as a hint for sub-dispatched tools
    desc = "Paramètres de la fonction."
    if properties:
        param_hints = ", ".join(
            f"{k} ({v.get('description', '')})" for k, v in list(properties.items())[:6]
        )
        desc = f"Paramètres. Disponibles : {param_hints}"

    return {
        "type": "object",
        "description": desc,
        "properties": properties,
        "additionalProperties": True,
    }


def get_openai_tools() -> list[dict]:
    """
    Construit la liste des tools au format OpenAI function-calling.
    Enrichit le schéma params avec les paramètres documentés dans le catalogue.
    """
    caps = get_capabilities()

    # Load enriched catalogue for params schema
    try:
        import importlib.util as _ilu
        _cat_path = Path(__file__).parent / "catalogue.py"
        spec = _ilu.spec_from_file_location("catalogue", _cat_path)
        _cat_mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(_cat_mod)
        catalogue = _cat_mod.get_catalogue()
    except Exception:
        catalogue = {}

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

        params_schema = _build_params_schema(tool_name, catalogue)

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
                        "params": params_schema,
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
        # Suggest correct name if the call looks like an alias
        _ALIASES: dict[str, tuple[str, str]] = {
            # builder
            "credibility":           ("builder", "diagnostics"),
            "diagnostics_credibility": ("builder", "diagnostics"),
            "confidence_intervals":  ("builder", "validation"),
            "validation_ci":         ("builder", "validation"),
            "abatement_factors":     ("builder", "benchmarking"),
            "smr":                   ("builder", "benchmarking"),
            "comparison":            ("builder", "benchmarking"),
            "compute_exposure":      ("builder", "exposure"),
            "raw_rates":             ("builder", "crude_rates"),
        }
        suggestion = _ALIASES.get(function_name)
        if suggestion and suggestion[0] == tool_name:
            correct = f"{suggestion[0]}.{suggestion[1]}"
            available = list(tool_info.get("functions", {}).keys())
            return {
                "erreur": (
                    f"Fonction inconnue : '{tool_name}.{function_name}'. "
                    f"Nom correct : '{correct}'. "
                    f"Fonctions disponibles dans {tool_name} : {available}"
                )
            }
        available = list(tool_info.get("functions", {}).keys())
        return {
            "erreur": (
                f"Fonction inconnue : '{tool_name}.{function_name}'. "
                f"Fonctions disponibles dans {tool_name} : {available}"
            )
        }
    if fn_info.get("disponible") is False:
        return {"erreur": f"'{tool_name}.{function_name}' : {fn_info.get('raison', 'non disponible')}"}

    # Import dynamique
    module_path = f"tools.{tool_name}.{function_name}"
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
            # exposure nécessite le df seul ; cox_regression nécessite data + df
            if function_name in _BUILDER_DF_FUNCTIONS:
                if df is None:
                    return {"erreur": "builder.exposure nécessite un DataFrame (df=None)."}
                return mod.run(df, params)
            elif function_name in _BUILDER_DATA_DF_FUNCTIONS:
                return mod.run(data, params, df=df)
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
