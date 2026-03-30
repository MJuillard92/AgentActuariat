"""
tool_registry.py
Registre central des tools actuariels.

tool_registry.py n’est pas une fonction mais un module registre. Il sert de point d’entrée central pour déclarer, exposer et exécuter les tools définis dans le projet, surtout statistical_analysis et build_pdf, via tool_registry.py.

Concrètement, il fait 3 choses :

get_capabilities() lit builder_capabilities.json et retourne le catalogue des tools disponibles, avec leurs fonctions, descriptions, paramètres, disponibilités, etc.

get_openai_tools() transforme ce catalogue en format “function calling” pour OpenAI.
Chaque tool devient une fonction appelable avec :

function_name : la sous-fonction à lancer, par exemple age_distribution
params : les paramètres libres à transmettre à cette sous-fonction
call_tool(...) exécute réellement une fonction demandée.
Il :

vérifie que le tool existe dans le catalogue
vérifie que la fonction existe et est disponible
importe dynamiquement le module correspondant, par exemple report_agent.tools.statistical_analysis.age_distribution
vérifie que ce module expose bien run(...)
appelle run(df, params) pour statistical_analysis
appelle run(data, params) pour build_pdf
En résumé, ce fichier fait le lien entre :

- le catalogue JSON,
- le format d’appel OpenAI,
- l’exécution réelle des modules Python.
Petit schéma mental :
builder_capabilities.json -> tool_registry.py -> import dynamique du bon fichier -> appel de run(...)


Expose :
  get_capabilities() -> dict         — charge builder_capabilities.json
  get_openai_tools() -> list[dict]   — format OpenAI function-calling
  call_tool(tool_name, function_name, params, df, data) -> dict
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

_CAPABILITIES_PATH = Path(__file__).parent.parent / "builder_capabilities.json"


def get_capabilities() -> dict:
    """Charge et retourne builder_capabilities.json."""
    if _CAPABILITIES_PATH.exists():
        return json.loads(_CAPABILITIES_PATH.read_text(encoding="utf-8"))
    return {"version": "1.0", "tools": {}}


def get_openai_tools() -> list[dict]:
    """
    Construit la liste des tools au format OpenAI function-calling.
    Un tool par entrée dans capabilities["tools"], avec un paramètre
    'function_name' (enum des fonctions disponibles) et 'params' (object libre).
    """
    caps = get_capabilities()
    tools = []
    for tool_name, tool_info in caps.get("tools", {}).items():
        fn_names = list(tool_info.get("functions", {}).keys())
        # Descriptions des fonctions pour enrichir l'enum
        fn_descriptions = {
            fn: info.get("description", "")
            for fn, info in tool_info.get("functions", {}).items()
            if info.get("disponible", True) is not False
        }
        available_fns = [fn for fn in fn_names if fn in fn_descriptions]
        if not available_fns:
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
                            "enum": available_fns,
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
) -> dict:
    """
    Exécute la fonction demandée dans le tool.

    Arguments :
      tool_name     : "statistical_analysis" | "build_pdf"
      function_name : nom de la fonction dans le sous-module
      params        : paramètres libres passés à run()
      df            : DataFrame du portefeuille (pour statistical_analysis)
      data          : dict de données consolidées (pour build_pdf)
    """
    params = params or {}

    # Vérification : tool et fonction existent dans le catalogue
    caps = get_capabilities()
    tool_info = caps.get("tools", {}).get(tool_name)
    if tool_info is None:
        # Vérifier hors_perimetre
        hp = caps.get("hors_perimetre", {}).get(tool_name)
        if hp:
            return {"erreur": f"'{tool_name}' hors périmètre : {hp.get('raison', 'non disponible')}"}
        return {"erreur": f"Tool inconnu : '{tool_name}'"}

    fn_info = tool_info.get("functions", {}).get(function_name)
    if fn_info is None:
        return {"erreur": f"Fonction inconnue : '{tool_name}.{function_name}'"}
    if fn_info.get("disponible") is False:
        return {"erreur": f"'{tool_name}.{function_name}' : {fn_info.get('raison', 'non disponible')}"}

    # Import dynamique du module
    module_path = f"report_agent.tools.{tool_name}.{function_name}"
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        return {"erreur": f"Module introuvable : {module_path}"}

    if not hasattr(mod, "run"):
        return {"erreur": f"{module_path} n'expose pas de fonction run()"}

    # Appel de la fonction
    try:
        if tool_name == "statistical_analysis":
            if df is None:
                return {"erreur": "statistical_analysis nécessite un DataFrame (df=None)."}
            return mod.run(df, params)
        elif tool_name == "build_pdf":
            return mod.run(data or {}, params)
        else:
            return mod.run(data or {}, params)
    except Exception as exc:
        import traceback
        return {
            "erreur": f"Erreur lors de l'exécution de {tool_name}.{function_name} : {exc}",
            "traceback": traceback.format_exc(),
        }
