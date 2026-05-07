"""Tests pour agents.mortality.agents.llm_config."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _clear_env_overrides():
    """Avant chaque test : retirer les variables d'environnement de test."""
    keys_to_clear = [
        k for k in os.environ
        if k.startswith("LLM_MODEL_")
    ]
    saved = {k: os.environ.pop(k) for k in keys_to_clear}
    yield
    for k, v in saved.items():
        os.environ[k] = v


@pytest.fixture(autouse=True)
def _clear_cache():
    """Avant chaque test : vider le cache du loader."""
    from agents.mortality.agents.llm_config import clear_cache
    clear_cache()
    yield
    clear_cache()


def test_default_role_returns_defaults():
    """Un rôle inexistant retourne les valeurs par défaut."""
    from agents.mortality.agents.llm_config import get_llm_config
    cfg = get_llm_config("nonexistent.role")
    # Les defaults YAML doivent au moins contenir un model
    assert "model" in cfg


def test_classify_intent_uses_mini():
    """Le rôle master.classify_intent utilise gpt-5.4-mini par défaut."""
    from agents.mortality.agents.llm_config import get_llm_config
    cfg = get_llm_config("master.classify_intent")
    assert cfg["model"] == "gpt-5.4-mini"
    assert cfg["max_tokens"] == 200
    assert cfg["temperature"] == 0.0


def test_builder_llm_uses_full_model():
    """builder.llm utilise gpt-5.4 (qualité tool-calling)."""
    from agents.mortality.agents.llm_config import get_llm_config
    cfg = get_llm_config("builder.llm")
    assert cfg["model"] == "gpt-5.4"


def test_writer_redaction_has_temperature():
    """writer.redaction utilise une température non nulle pour le style."""
    from agents.mortality.agents.llm_config import get_llm_config
    cfg = get_llm_config("writer.redaction")
    assert cfg["model"] == "gpt-5.4"
    assert cfg["temperature"] > 0


def test_env_var_overrides_yaml_for_model():
    """LLM_MODEL_<ROLE>=foo écrase la valeur YAML."""
    from agents.mortality.agents.llm_config import get_llm_config, clear_cache
    os.environ["LLM_MODEL_MASTER_CLASSIFY_INTENT"] = "gpt-5.4-nano"
    clear_cache()
    cfg = get_llm_config("master.classify_intent")
    assert cfg["model"] == "gpt-5.4-nano"
    # Les autres champs restent les valeurs YAML
    assert cfg["max_tokens"] == 200


def test_env_var_for_builder():
    """Override du modèle Builder via env var."""
    from agents.mortality.agents.llm_config import get_llm_config, clear_cache
    os.environ["LLM_MODEL_BUILDER_LLM"] = "o4-mini"
    clear_cache()
    cfg = get_llm_config("builder.llm")
    assert cfg["model"] == "o4-mini"


def test_env_var_does_not_override_other_fields():
    """LLM_MODEL_X n'écrase QUE le model, pas max_tokens/temperature."""
    from agents.mortality.agents.llm_config import get_llm_config, clear_cache
    os.environ["LLM_MODEL_BUILDER_LLM"] = "custom-model"
    clear_cache()
    cfg = get_llm_config("builder.llm")
    assert cfg["model"] == "custom-model"
    # max_tokens et temperature inchangés
    assert cfg["max_tokens"] == 4000
    assert cfg["temperature"] == 0.0


def test_optimization_flags():
    """get_optimization_flag lit la section optimization du YAML."""
    from agents.mortality.agents.llm_config import (
        get_optimization_flag, get_optimization_value,
    )
    assert get_optimization_flag("enable_prompt_cache") is True
    assert get_optimization_value("builder_max_history") == 20
    assert get_optimization_flag("nonexistent_flag", default=False) is False
    assert get_optimization_value("nonexistent_value", default=42) == 42


def test_lru_cache_avoids_reread():
    """Le YAML n'est lu qu'une fois grâce au cache LRU."""
    from agents.mortality.agents.llm_config import _load_config
    cache_info_before = _load_config.cache_info()
    _load_config()
    _load_config()
    _load_config()
    cache_info_after = _load_config.cache_info()
    # Le hit count doit augmenter (cache utilisé), miss reste à 1
    assert cache_info_after.hits >= cache_info_before.hits + 2
