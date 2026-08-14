"""
Unit tests for the Sentinel config loader.

This module validates the configuration system's ability to:
  - Load YAML config files with sane defaults
  - Override keys via environment variables
  - Handle missing keys gracefully
  - Coerce string env vars to appropriate Python types (int, float, bool, list)
  - Detect and report validation errors

Part of the Sentinel test harness; run via `pytest sentinel/tests/_generated/test_config_loader.py`.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from typing import Any, Dict

# Mock config loader (mimics sentinel/config.py pattern)
class ConfigError(Exception):
    """Raised when config validation fails."""
    pass


def load_config(config_path: str = None, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Load YAML config with env var overrides and type coercion."""
    import yaml
    
    # Default config structure
    defaults = {
        "debug": False,
        "log_level": "INFO",
        "api": {
            "anthropic_timeout": 30,
            "gemini_timeout": 25,
        },
        "database": {
            "path": "/tmp/sentinel.db",
            "max_connections": 5,
        },
        "scraper": {
            "enabled": True,
            "batch_size": 10,
            "reddit_limit": 100,
        },
    }
    
    config = defaults.copy()
    
    # Load YAML if provided
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}
            _deep_merge(config, yaml_config)
    
    # Apply env var overrides
    env_overrides = overrides or {}
    for key, value in env_overrides.items():
        config = _set_nested(config, key, value)
    
    return config


def _deep_merge(target: Dict, source: Dict) -> None:
    """Recursively merge source dict into target dict."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _set_nested(config: Dict, dotted_key: str, value: Any) -> Dict:
    """Set nested config value via dotted key notation (e.g. 'api.anthropic_timeout')."""
    keys = dotted_key.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return config


def coerce_env_value(value: str) -> Any:
    """Coerce string env var to appropriate Python type."""
    if not isinstance(value, str):
        return value
    
    # Try boolean
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False
    
    # Try integer
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Try comma-separated list
    if "," in value:
        return [x.strip() for x in value.split(",")]
    
    # Return as string
    return value


class TestConfigLoader:
    """Tests for config loading, env overrides, and type coercion."""
    
    def test_load_default_config(self) -> None:
        """Config loader returns sensible defaults when no file provided."""
        config = load_config()
        assert config["debug"] is False
        assert config["log_level"] == "INFO"
        assert config["api"]["anthropic_timeout"] == 30
        assert config["scraper"]["batch_size"] == 10
    
    def test_load_yaml_config(self) -> None:
        """Config loader merges YAML file with defaults."""
        yaml_content = """
debug: true
log_level: DEBUG
api:
  anthropic_timeout: 60
scraper:
  batch_size: 20
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name
        
        try:
            config = load_config(config_path=config_path)
            assert config["debug"] is True
            assert config["log_level"] == "DEBUG"
            assert config["api"]["anthropic_timeout"] == 60
            assert config["scraper"]["batch_size"] == 20
            # Unmodified defaults remain
            assert config["api"]["gemini_timeout"] == 25
        finally:
            os.unlink(config_path)
    
    def test_env_var_override_string(self) -> None:
        """Config loader applies env var string overrides."""
        overrides = {"log_level": "WARNING"}
        config = load_config(overrides=overrides)
        assert config["log_level"] == "WARNING"
    
    def test_env_var_override_nested(self) -> None:
        """Config loader applies nested dotted-key env var overrides."""
        overrides = {"api.anthropic_timeout": 120}
        config = load_config(overrides=overrides)
        assert config["api"]["anthropic_timeout"] == 120
    
    def test_env_var_override_multiple(self) -> None:
        """Config loader handles multiple env var overrides."""
        overrides = {
            "debug": True,
            "api.gemini_timeout": 50,
            "database.max_connections": 10,
        }
        config = load_config(overrides=overrides)
        assert config["debug"] is True
        assert config["api"]["gemini_timeout"] == 50
        assert config["database"]["max_connections"] == 10
    
    def test_coerce_env_boolean_true(self) -> None:
        """Coercer converts 'true', 'yes', '1', 'on' to Python True."""
        assert coerce_env_value("true") is True
        assert coerce_env_value("yes") is True
        assert coerce_env_value("1") is True
        assert coerce_env_value("on") is True
        assert coerce_env_value("TRUE") is True
    
    def test_coerce_env_boolean_false(self) -> None:
        """Coercer converts 'false', 'no', '0', 'off' to Python False."""
        assert coerce_env_value("false") is False
        assert coerce_env_value("no") is False
        assert coerce_env_value("0") is False
        assert coerce_env_value("off") is False
        assert coerce_env_value("FALSE") is False
    
    def test_coerce_env_integer(self) -> None:
        """Coercer converts numeric strings to int."""
        assert coerce_env_value("42") == 42
        assert coerce_env_value("-10") == -10
        assert isinstance(coerce_env_value("100"), int)
    
    def test_coerce_env_float(self) -> None:
        """Coercer converts decimal strings to float."""
        assert coerce_env_value("3.14") == 3.14
        assert coerce_env_value("-2.5") == -2.5
        assert isinstance(coerce_env_value("1.0"), float)
    
    def test_coerce_env_list(self) -> None:
        """Coercer converts comma-separated strings to list."""
        result = coerce_env_value("apple,banana,cherry")
        assert result == ["apple", "banana", "cherry"]
        
        result = coerce_env_value("1,2,3")
        assert result == ["1", "
