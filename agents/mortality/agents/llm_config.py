"""
agents/mortality/agents/llm_config.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Lecture centralisée des choix de modèles LLM par rôle.

Lit `config/llm_models.yaml` à la racine du projet. Toute valeur peut être
écrasée par une variable d'environnement de la forme :

    LLM_MODEL_<ROLE_PATH>=<modèle>

où `ROLE_PATH` est le chemin du rôle dans le YAML, séparé par des `_` au
lieu de `.` :

    role "master.classify_intent" → env "LLM_MODEL_MASTER_CLASSIFY_INTENT"

API publique :
    get_llm_config(role: str) -> dict
        Retourne {model, temperature, max_tokens, ...} pour un rôle.

    get_optimization_flag(name: str, default: bool = False) -> bool
        Flag d'optimisation transversal (prompt_cache, history_size, etc.).

    get_optimization_value(name: str, default=None)
        Valeur arbitraire d'un flag d'optimisation (int, str, bool…).

Le YAML est lu une seule fois (LRU cache). Pour forcer un rechargement
en test, appeler `clear_cache()`.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "llm_models.yaml"


# ──────────────────────────────────────────────────────────────────────────
# Chargement YAML (cache LRU)
# ──────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_config() -> dict:
    """Charge le YAML de config. Retourne {} si fichier absent."""
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def clear_cache() -> None:
    """Force un rechargement du YAML au prochain appel (utile en tests)."""
    _load_config.cache_clear()


# ──────────────────────────────────────────────────────────────────────────
# API publique
# ──────────────────────────────────────────────────────────────────────────

def get_llm_config(role: str) -> dict:
    """Retourne la configuration LLM pour un rôle donné.

    Args:
        role: chemin séparé par des points dans le YAML.
              Exemples : "master.classify_intent", "builder.llm",
                         "writer.redaction", "master.conversation".

    Returns:
        Dict contenant au moins `model`, `temperature`, `max_tokens`. Les
        valeurs par défaut (`defaults`) sont fusionnées avec celles du rôle
        spécifique. La variable d'environnement
        `LLM_MODEL_<ROLE_PATH_UPPER>` (avec `_` au lieu de `.`) écrase
        la clé `model` si présente.

    Si le rôle n'est pas trouvé dans le YAML, retourne les defaults seuls.
    """
    cfg = _load_config()
    defaults = dict(cfg.get("defaults") or {})

    # Naviguer dans le YAML en suivant le chemin pointé
    node: Any = cfg
    for part in role.split("."):
        if not isinstance(node, dict):
            node = None
            break
        node = node.get(part)
        if node is None:
            break

    role_cfg = defaults
    if isinstance(node, dict):
        role_cfg = {**defaults, **node}

    # Override par variable d'environnement (uniquement pour `model`)
    env_key = "LLM_MODEL_" + role.replace(".", "_").upper()
    if env_key in os.environ and os.environ[env_key].strip():
        role_cfg["model"] = os.environ[env_key].strip()

    return role_cfg


def get_optimization_flag(name: str, default: bool = False) -> bool:
    """Retourne un flag booléen d'optimisation depuis le YAML.

    Args:
        name: nom du flag dans la section `optimization` (ex: "enable_prompt_cache").
        default: valeur retournée si la clé est absente.
    """
    val = get_optimization_value(name, default)
    return bool(val)


def get_optimization_value(name: str, default: Any = None) -> Any:
    """Retourne une valeur arbitraire depuis la section `optimization`.

    Args:
        name: nom de la clé (ex: "builder_max_history").
        default: valeur retournée si la clé est absente.
    """
    cfg = _load_config()
    opt = cfg.get("optimization") or {}
    return opt.get(name, default)
