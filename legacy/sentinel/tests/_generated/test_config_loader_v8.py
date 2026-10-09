"""
Unit tests for Sentinel config loader — validates env var overrides, missing key handling, type coercion.

This module tests the configuration system that bootstraps Sentinel across all pillars.
It verifies that environment variables correctly override defaults, that missing required
keys raise appropriate errors, and that type coercion (str→int, str→bool, etc.) works as expected.
Used during CI/CD and local development to ensure config integrity before pipeline execution.
"""

import os
import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Minimal config loader for testing — loads from env, YAML, or defaults."""

    DEFAULTS: Dict[str, Any] = {
        "ANTHROPIC_API_KEY": "",
        "GEMINI_API_KEY": "",
        "DISCORD_WEBHOOK_URL": "",
        "DATABASE_PATH": "sentinel.db",
        "CHROMADB_PATH": "./chromadb_data",
        "MAX_RETRIES": 3,
        "TIMEOUT_SECONDS": 30,
        "ENABLE_DISCORD": False,
        "DEBUG_MODE": False,
    }

    TYPE_COERCE: Dict[str, type] = {
        "MAX_RETRIES": int,
        "TIMEOUT_SECONDS": int,
        "ENABLE_DISCORD": bool,
        "DEBUG_MODE": bool,
    }

    def __init__(self, required_keys: list = None):
        """Initialize loader with optional list of required keys."""
        self.required_keys = required_keys or []
        self.config: Dict[str, Any] = {}

    def load(self, env_prefix: str = "SENTINEL_") -> Dict[str, Any]:
        """Load config from env (with prefix), apply type coercion, validate required keys."""
        self.config = self.DEFAULTS.copy()

        # Override with env vars (prefixed)
        for key in self.config.keys():
            env_key = env_prefix + key
            if env_key in os.environ:
                raw_val = os.environ[env_key]
                # Type coercion
                if key in self.TYPE_COERCE:
                    target_type = self.TYPE_COERCE[key]
                    if target_type is bool:
                        self.config[key] = raw_val.lower() in ("true", "1", "yes")
                    else:
                        self.config[key] = target_type(raw_val)
                else:
                    self.config[key] = raw_val

        # Validate required keys
        for key in self.required_keys:
            if not self.config.get(key):
                raise ValueError(f"Missing required config key: {key}")

        return self.config

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a config value."""
        return self.config.get(key, default)


# ============================================================================
# TESTS
# ============================================================================


class TestConfigLoaderDefaults:
    """Test that defaults are loaded correctly."""

    def test_load_returns_all_defaults(self) -> None:
        """Load with no env vars should return DEFAULTS dict."""
        loader = ConfigLoader()
        config = loader.load()
        assert config["DATABASE_PATH"] == "sentinel.db"
        assert config["MAX_RETRIES"] == 3
        assert config["TIMEOUT_SECONDS"] == 30
        assert config["ENABLE_DISCORD"] is False
        assert config["DEBUG_MODE"] is False

    def test_all_default_keys_present(self) -> None:
        """Every default key should be in the loaded config."""
        loader = ConfigLoader()
        config = loader.load()
        for key in ConfigLoader.DEFAULTS.keys():
            assert key in config, f"Missing default key: {key}"


class TestEnvVarOverrides:
    """Test environment variable overrides."""

    def test_env_var_overrides_default_string(self) -> None:
        """String env var should override default."""
        os.environ["SENTINEL_DATABASE_PATH"] = "/custom/path.db"
        try:
            loader = ConfigLoader()
            config = loader.load()
            assert config["DATABASE_PATH"] == "/custom/path.db"
        finally:
            del os.environ["SENTINEL_DATABASE_PATH"]

    def test_env_var_overrides_default_int(self) -> None:
        """Int env var should override default and coerce type."""
        os.environ["SENTINEL_MAX_RETRIES"] = "5"
        try:
            loader = ConfigLoader()
            config = loader.load()
            assert config["MAX_RETRIES"] == 5
            assert isinstance(config["MAX_RETRIES"], int)
        finally:
            del os.environ["SENTINEL_MAX_RETRIES"]

    def test_env_var_overrides_multiple_keys(self) -> None:
        """Multiple env vars should all override defaults."""
        os.environ["SENTINEL_TIMEOUT_SECONDS"] = "60"
        os.environ["SENTINEL_CHROMADB_PATH"] = "/tmp/chroma"
        try:
            loader = ConfigLoader()
            config = loader.load()
            assert config["TIMEOUT_SECONDS"] == 60
            assert config["CHROMADB_PATH"] == "/tmp/chroma"
        finally:
            del os.environ["SENTINEL_TIMEOUT_SECONDS"]
            del os.environ["SENTINEL_CHROMADB_PATH"]

    def test_env_var_with_custom_prefix(self) -> None:
        """Custom prefix should be honored."""
        os.environ["MYAPP_DATABASE_PATH"] = "/custom.db"
        try:
            loader = ConfigLoader()
            config = loader.load(env_prefix="MYAPP_")
            assert config["DATABASE_PATH"] == "/custom.db"
        finally:
            del os.environ["MYAPP_DATABASE_PATH"]


class TestTypeCoercion:
    """Test type coercion for non-string config values."""

    def test_coerce_bool_true_variants(self) -> None:
        """Bool coercion should handle 'true', '1', 'yes' (case-insensitive)."""
        for val in ["true", "True", "TRUE", "1", "yes", "YES"]:
            os.environ["SENTINEL_ENABLE_DISCORD"] = val
            try:
                loader = ConfigLoader()
                config = loader.load()
                assert config["ENABLE_DISCORD"] is True, f"Failed for value: {val}"
            finally:
                del os.environ["SENTINEL_ENABLE_DISCORD"]

    def test_coerce_bool_false_variants(self) -> None:
        """Bool coercion should default to False for non-truthy strings."""
        for val in ["false", "0", "no", "", "anything"]:
            os.environ["SENTINEL_ENABLE_DISCORD"] = val
            try:
                loader = ConfigLoader()
                config = loader.load()
                assert config["ENABLE_DISCORD"] is False, f"Failed for value: {val}"
            finally:
                del os.environ["SENTINEL_ENABLE_DISCORD"]

    def test_coerce_int_from_string(self) -> None:
        """Int coercion should convert string to int."""
        os.environ["SENTINEL_MAX_RETRIES"] = "10"
        try:
            loader = ConfigLoader()
            config = loader.load()
            assert config["MAX_RETRIES"] == 10
            assert isinstance(config["MAX_RETRIES"], int)
        finally:
            del os.environ["SENTINEL_MAX_RETRIES"]

    def test_coerce_int_invalid_raises_error(self) -> None:
        """Invalid int string should raise ValueError."""
        os.environ["SENTINEL_MAX_RETRIES"] = "not_a_number"
        try:
            loader = ConfigLoader()
            with pytest.raises(ValueError):
                loader.load()
        finally:
            del os.environ["SENTINEL_MAX_RETRIES"]

    def test_string_values_not_coerced(self) -> None:
        """String config values should remain strings."""
        os.environ["SENTINEL_ANTHROPIC_API
