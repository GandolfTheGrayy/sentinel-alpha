"""
Unit tests for Sentinel's configuration loader.

This module validates the config system's ability to:
  - Load settings from environment variables
  - Override defaults with env var values
  - Handle missing keys gracefully with fallbacks
  - Coerce string env values to correct Python types (bool, int, float)
  - Detect invalid configurations and raise appropriate errors

Tests are organized by concern: env overrides, type coercion, missing keys,
and error handling. Each test is isolated and uses temporary env state.
"""

import os
import sys
import pytest
from typing import Any, Dict
from unittest.mock import patch

# Minimal config loader implementation for testing purposes
class ConfigLoader:
    """Simple configuration loader with env var override and type coercion."""

    DEFAULTS: Dict[str, Any] = {
        "ANTHROPIC_API_KEY": "",
        "GEMINI_API_KEY": "",
        "DEBUG_MODE": False,
        "MAX_RETRIES": 3,
        "TIMEOUT_SECONDS": 30.0,
        "BATCH_SIZE": 10,
        "DISCORD_WEBHOOK": "",
    }

    TYPES: Dict[str, type] = {
        "DEBUG_MODE": bool,
        "MAX_RETRIES": int,
        "TIMEOUT_SECONDS": float,
        "BATCH_SIZE": int,
    }

    @staticmethod
    def coerce_value(key: str, value: str) -> Any:
        """Coerce a string env value to its declared type."""
        if key not in ConfigLoader.TYPES:
            return value
        
        target_type = ConfigLoader.TYPES[key]
        
        if target_type is bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif target_type is int:
            try:
                return int(value)
            except ValueError:
                raise ValueError(f"Cannot coerce '{value}' to int for key {key}")
        elif target_type is float:
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"Cannot coerce '{value}' to float for key {key}")
        
        return value

    @staticmethod
    def load() -> Dict[str, Any]:
        """Load config from env vars, applying defaults and type coercion."""
        config = {}
        
        for key, default in ConfigLoader.DEFAULTS.items():
            env_value = os.environ.get(key)
            
            if env_value is not None:
                try:
                    config[key] = ConfigLoader.coerce_value(key, env_value)
                except ValueError as e:
                    raise ValueError(f"Config error for {key}: {e}")
            else:
                config[key] = default
        
        return config


class TestConfigLoaderEnvOverrides:
    """Test env var overrides of default values."""

    def test_string_env_override(self) -> None:
        """Env var string should override default string."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-123"}):
            config = ConfigLoader.load()
            assert config["ANTHROPIC_API_KEY"] == "test-key-123"

    def test_bool_env_override_true(self) -> None:
        """Env var 'true' should coerce to boolean True."""
        with patch.dict(os.environ, {"DEBUG_MODE": "true"}):
            config = ConfigLoader.load()
            assert config["DEBUG_MODE"] is True

    def test_bool_env_override_false(self) -> None:
        """Env var 'false' should coerce to boolean False."""
        with patch.dict(os.environ, {"DEBUG_MODE": "false"}):
            config = ConfigLoader.load()
            assert config["DEBUG_MODE"] is False

    def test_bool_env_override_variants(self) -> None:
        """Test various boolean string representations."""
        for truthy in ("1", "yes", "on"):
            with patch.dict(os.environ, {"DEBUG_MODE": truthy}):
                config = ConfigLoader.load()
                assert config["DEBUG_MODE"] is True, f"Failed for '{truthy}'"

        for falsy in ("0", "no", "off", "false"):
            with patch.dict(os.environ, {"DEBUG_MODE": falsy}):
                config = ConfigLoader.load()
                assert config["DEBUG_MODE"] is False, f"Failed for '{falsy}'"

    def test_int_env_override(self) -> None:
        """Env var should coerce to int."""
        with patch.dict(os.environ, {"MAX_RETRIES": "7"}):
            config = ConfigLoader.load()
            assert config["MAX_RETRIES"] == 7
            assert isinstance(config["MAX_RETRIES"], int)

    def test_float_env_override(self) -> None:
        """Env var should coerce to float."""
        with patch.dict(os.environ, {"TIMEOUT_SECONDS": "45.5"}):
            config = ConfigLoader.load()
            assert config["TIMEOUT_SECONDS"] == 45.5
            assert isinstance(config["TIMEOUT_SECONDS"], float)

    def test_multiple_env_overrides(self) -> None:
        """Multiple env var overrides should all apply."""
        env_vars = {
            "ANTHROPIC_API_KEY": "key-abc",
            "DEBUG_MODE": "true",
            "MAX_RETRIES": "5",
            "TIMEOUT_SECONDS": "60.0",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = ConfigLoader.load()
            assert config["ANTHROPIC_API_KEY"] == "key-abc"
            assert config["DEBUG_MODE"] is True
            assert config["MAX_RETRIES"] == 5
            assert config["TIMEOUT_SECONDS"] == 60.0


class TestConfigLoaderDefaults:
    """Test fallback to default values when env vars are absent."""

    def test_missing_string_key_uses_default(self) -> None:
        """Missing string key should use default."""
        with patch.dict(os.environ, {}, clear=True):
            config = ConfigLoader.load()
            assert config["ANTHROPIC_API_KEY"] == ""

    def test_missing_bool_key_uses_default(self) -> None:
        """Missing bool key should use default False."""
        with patch.dict(os.environ, {}, clear=True):
            config = ConfigLoader.load()
            assert config["DEBUG_MODE"] is False

    def test_missing_int_key_uses_default(self) -> None:
        """Missing int key should use default."""
        with patch.dict(os.environ, {}, clear=True):
            config = ConfigLoader.load()
            assert config["MAX_RETRIES"] == 3

    def test_missing_float_key_uses_default(self) -> None:
        """Missing float key should use default."""
        with patch.dict(os.environ, {}, clear=True):
            config = ConfigLoader.load()
            assert config["TIMEOUT_SECONDS"] == 30.0

    def test_all_defaults_present(self) -> None:
        """Config should contain all default keys even when env is empty."""
        with patch.dict(os.environ, {}, clear=True):
            config = ConfigLoader.load()
            for key in ConfigLoader.DEFAULTS:
                assert key in config


class TestConfigLoaderTypeCoercion:
    """Test type coercion behavior."""

    def test_int_coercion_valid(self) -> None:
        """Valid int string should coerce cleanly."""
        with patch.dict(os.environ, {"BATCH_SIZE": "42"}):
            config = ConfigLoader.load()
            assert config["BATCH_SIZE"] == 42
            assert type(config["BATCH_SIZE"]) is int

    def test_int_coercion_negative(self) -> None:
        """Negative int strings should coerce."""
        with patch.dict(os.environ, {"MAX_RETRIES": "-1"}):
            config = ConfigLoader.load()
            assert config["MAX_RETRIES"] == -1

    def test_float_coercion_valid(self) -> None
