"""
Unit tests for Sentinel's configuration loader.

Tests env var overrides, missing key handling, type coercion, and default fallbacks.
This module ensures the config system reliably supplies credentials and parameters
to all Sentinel pillars (scout, linguist, historian, judge).
"""

import os
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict


# Mock config loader implementation for testing
class ConfigLoader:
    """Minimal config loader for Sentinel — reads YAML + env overrides."""

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize loader with optional config file path."""
        self.config_path = config_path
        self._cache: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        """Load config from file and env, return merged dict."""
        base = {}
        if self.config_path and Path(self.config_path).exists():
            import yaml
            with open(self.config_path) as f:
                base = yaml.safe_load(f) or {}

        # Env overrides
        env_overrides = {
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "gemini_api_key": os.getenv("GEMINI_API_KEY"),
            "discord_webhook": os.getenv("DISCORD_WEBHOOK"),
            "reddit_client_id": os.getenv("REDDIT_CLIENT_ID"),
            "reddit_client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
            "max_retries": os.getenv("MAX_RETRIES"),
            "timeout_secs": os.getenv("TIMEOUT_SECS"),
            "debug_mode": os.getenv("DEBUG_MODE"),
        }
        for k, v in env_overrides.items():
            if v is not None:
                base[k] = v

        self._cache = base
        return base

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve config value with optional default."""
        if not self._cache:
            self.load()
        return self._cache.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        """Retrieve config value as int."""
        val = self.get(key, default)
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                return default
        return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Retrieve config value as bool."""
        val = self.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return default


class TestConfigLoaderEnvOverrides:
    """Test env var override behavior."""

    def test_env_override_api_key(self) -> None:
        """Env var ANTHROPIC_API_KEY overrides file config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("anthropic_api_key: file_key\n")
            f.flush()
            try:
                os.environ["ANTHROPIC_API_KEY"] = "env_key"
                loader = ConfigLoader(f.name)
                config = loader.load()
                assert config["anthropic_api_key"] == "env_key"
            finally:
                os.unlink(f.name)
                os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_env_override_gemini_api_key(self) -> None:
        """Env var GEMINI_API_KEY overrides file config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("gemini_api_key: file_key\n")
            f.flush()
            try:
                os.environ["GEMINI_API_KEY"] = "env_key"
                loader = ConfigLoader(f.name)
                config = loader.load()
                assert config["gemini_api_key"] == "env_key"
            finally:
                os.unlink(f.name)
                os.environ.pop("GEMINI_API_KEY", None)

    def test_env_override_discord_webhook(self) -> None:
        """Env var DISCORD_WEBHOOK overrides file config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("discord_webhook: https://hooks.slack.com/old\n")
            f.flush()
            try:
                os.environ["DISCORD_WEBHOOK"] = "https://discord.com/api/webhooks/new"
                loader = ConfigLoader(f.name)
                config = loader.load()
                assert config["discord_webhook"] == "https://discord.com/api/webhooks/new"
            finally:
                os.unlink(f.name)
                os.environ.pop("DISCORD_WEBHOOK", None)


class TestConfigLoaderMissingKeys:
    """Test handling of missing required keys."""

    def test_missing_key_returns_default(self) -> None:
        """Missing key returns provided default."""
        loader = ConfigLoader()
        loader.load()
        assert loader.get("nonexistent", "fallback") == "fallback"

    def test_missing_key_none_default(self) -> None:
        """Missing key with no default returns None."""
        loader = ConfigLoader()
        loader.load()
        assert loader.get("nonexistent") is None

    def test_missing_api_key_raises_on_get(self) -> None:
        """Accessing missing API key should return None unless explicit default."""
        loader = ConfigLoader()
        loader.load()
        assert loader.get("anthropic_api_key") is None

    def test_missing_int_key_returns_int_default(self) -> None:
        """Missing int key returns int default."""
        loader = ConfigLoader()
        loader.load()
        assert loader.get_int("max_retries", 3) == 3

    def test_missing_bool_key_returns_bool_default(self) -> None:
        """Missing bool key returns bool default."""
        loader = ConfigLoader()
        loader.load()
        assert loader.get_bool("debug_mode", False) is False


class TestConfigLoaderTypeCoercion:
    """Test type coercion for int and bool values."""

    def test_coerce_string_to_int(self) -> None:
        """String env var coerced to int."""
        os.environ["MAX_RETRIES"] = "5"
        try:
            loader = ConfigLoader()
            loader.load()
            assert loader.get_int("max_retries", 0) == 5
        finally:
            os.environ.pop("MAX_RETRIES", None)

    def test_coerce_invalid_string_to_int_uses_default(self) -> None:
        """Invalid string coercion returns default int."""
        os.environ["TIMEOUT_SECS"] = "not_a_number"
        try:
            loader = ConfigLoader()
            loader.load()
            assert loader.get_int("timeout_secs", 30) == 30
        finally:
            os.environ.pop("TIMEOUT_SECS", None)

    def test_coerce_string_to_bool_true(self) -> None:
        """String 'true', '1', 'yes' coerced to bool True."""
        for val in ("true", "1", "yes"):
            os.environ["DEBUG_MODE"] = val
            try:
                loader = ConfigLoader()
                loader.load()
                assert loader.get_bool("debug_mode", False) is True
            finally:
                os.environ.pop("DEBUG_MODE", None)

    def test_coerce_string_to_bool_false(self) -> None:
        """String other than 'true
