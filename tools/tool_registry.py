"""
tool_registry.py
Registre central des tools actuariels.

Fait le lien entre :
  - tools/catalogue.py  (source de vérité dynamique — parsée depuis les docstrings)
  - format OpenAI function-calling
  - exécution réelle des modules Python

Expose :
  get_capabilities() -> dict         — construit le catalogue depuis catalogue.py
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
import importlib.util as _ilu
from pathlib import Path
import pandas as pd

_CAT_PATH = Path(__file__).parent / "catalogue.py"

# Tools qui reçoivent un DataFrame comme premier argument
_DF_TOOLS = {"statistical_analysis", "preprocessing", "conversation"}

# Fonctions du namespace `conversation` qui nécessitent ALSO data_store en
# argument (pour mutation des flags de session). Toutes les autres restent
# en signature simple run(df, params).
_CONVERSATION_DATA_FUNCTIONS = {"apply_normalization"}

# Fonctions builder qui nécessitent le df seul comme premier argument (les autres utilisent data_store)
_BUILDER_DF_FUNCTIONS = {"exposure"}

# Fonctions builder qui nécessitent data ET df (signature run(data, params, df=None))
_BUILDER_DATA_DF_FUNCTIONS = {"cox_regression"}

# Tools hors périmètre (non implémentés)
_HORS_PERIMETRE = {
    "chain_ladder":       {"disponible": False, "raison": "Module non-vie — hors périmètre actuariel vie."},
    "bornhuetter_ferguson": {"disponible": False, "raison": "Module non-vie — hors périmètre actuariel vie."},
    "tarification_auto":  {"disponible": False, "raison": "Tarification dommages — hors périmètre."},
    "ibnr":               {"disponible": False, "raison": "Module IBNR non implémenté."},
}


def _load_catalogue_module():
    """Charge catalogue.py dynamiquement et retourne le module."""
    spec = _ilu.spec_from_file_location("catalogue", _CAT_PATH)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_capabilities_from_catalogue() -> dict:
    """
    Transforme le catalogue plat (builder.exposure → {...}) en structure imbriquée
    compatible avec l'ancien format builder_capabilities.json :
      {tools: {tool_name: {description, functions: {fn_name: {description, params, outputs}}}}}
    """
    try:
        cat_mod = _load_catalogue_module()
        raw = cat_mod.get_catalogue()
    except Exception:
        return {"version": "2.0", "tools": {}}

    tools_flat = raw.get("tools", {})
    nested: dict = {}

    for qualified_name, info in tools_flat.items():
        # "builder.exposure" → tool="builder", fn="exposure"
        # "build_pdf.load_yaml_template" → tool="build_pdf", fn="load_yaml_template"
        if "." not in qualified_name:
            continue
        tool_name, fn_name = qualified_name.split(".", 1)

        if tool_name not in nested:
            nested[tool_name] = {
                "description": info.get("description", ""),
                "functions": {},
            }
        # Description au niveau tool = première description rencontrée
        if not nested[tool_name]["description"] and info.get("description"):
            nested[tool_name]["description"] = info["description"]

        fn_entry = {
            "description":  info.get("short_description") or info.get("description", ""),
            "params":       info.get("params", {}),
            "outputs":      info.get("outputs", {}),
            "disponible":   info.get("disponible", True),
        }
        if "raison" in info:
            fn_entry["raison"] = info["raison"]

        nested[tool_name]["functions"][fn_name] = fn_entry

    return {"version": "2.0", "tools": nested, "hors_perimetre": _HORS_PERIMETRE}


# Cache en mémoire pour éviter de reparser à chaque appel dans la même session
_capabilities_cache: dict | None = None


def get_capabilities() -> dict:
    """Retourne le catalogue des tools (construit depuis catalogue.py, mis en cache)."""
    global _capabilities_cache
    if _capabilities_cache is None:
        _capabilities_cache = _build_capabilities_from_catalogue()
    return _capabilities_cache


def invalidate_capabilities_cache() -> None:
    """Force la régénération du catalogue au prochain appel (après modification d'un tool)."""
    global _capabilities_cache
    _capabilities_cache = None


def _build_params_schema(tool_name: str, catalogue: dict) -> dict:
    """
    Build the OpenAI JSON schema for the 'params' property of a tool,
    using the enriched catalogue (params section from tool contracts).
    """
    tools_cat = catalogue.get("tools", {})
    tool_info = tools_cat.get(tool_name, {})
    properties: dict = {}

    # Collect params from all functions of this tool
    for fn_info in tool_info.get("functions", {}).values():
        fn_params = fn_info.get("params", {})
        for param_name, param_info in fn_params.items():
            if param_name in properties:
                continue  # already added from another function
            prop: dict = {"type": "string"}
            if isinstance(param_info, dict):
                raw_type = param_info.get("type", "string")
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
    # Le catalogue enrichi est déjà dans caps (construit depuis catalogue.py)
    catalogue = caps

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


_STUDY_PLAN_MAPPINGS: dict[tuple[str, str], dict[str, str]] = {
    # (tool_name, function_name) → {param_key: study_plan_key}
    ("builder", "smoothing"):    {"method": "smoothing_algorithm",
                                  "lambda_": "smoothing_parameters"},
    ("builder", "benchmarking"): {"reference_table": "baseline_regulatory_table"},
    ("builder", "exposure"):     {"start_date": "observation_start_date",
                                  "end_date":   "observation_end_date"},
    ("builder", "crude_rates"):  {"method": "crude_rate_method"},
}


def _accumulate_study_plan(tool_name: str, function_name: str, params: dict, data: dict) -> None:
    """
    Intercepte les params d'un appel Builder et persiste les valeurs métier
    dans data["study_plan"] pour que load_yaml_template puisse les résoudre.
    Non bloquant — erreurs ignorées silencieusement.
    """
    if data is None or not params:
        return
    mapping = _STUDY_PLAN_MAPPINGS.get((tool_name, function_name))
    if not mapping:
        return
    sp = data.setdefault("study_plan", {})
    for param_key, plan_key in mapping.items():
        val = params.get(param_key)
        if val is not None and plan_key not in sp:
            sp[plan_key] = val


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

    # Persister les paramètres métier dans data["study_plan"] au fil des appels
    _accumulate_study_plan(tool_name, function_name, params, data)

    try:
        if tool_name in _DF_TOOLS:
            if df is None:
                return {"erreur": f"{tool_name} nécessite un DataFrame (df=None)."}
            # conversation.apply_normalization a besoin de muter data_store
            # (flags column_mapping_confirmed / records_normalized / etc.).
            if tool_name == "conversation" and function_name in _CONVERSATION_DATA_FUNCTIONS:
                return mod.run(df, params, data=data)
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
