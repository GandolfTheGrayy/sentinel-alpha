"""
Unit tests for the Sentinel configuration loader.

This module validates the config system's ability to:
  - Load YAML configuration files with sensible defaults
  - Override settings via environment variables
  - Coerce types correctly (strings → ints, bools, lists)
  - Handle missing keys gracefully with fallback values
  - Raise informative errors on schema violations

Tests exercise both happy paths and edge cases to ensure the config
loader is robust before it's wired into the main pipeline orchestrator.
"""

import os
import tempfile
import pytest
from typing import Any, Dict
from pathlib import Path


# Mock config loader for testing (in production, import from sentinel/config.py)
class ConfigLoader:
    """Minimal config loader implementation for testing."""

    DEFAULTS: Dict[str, Any] = {
        "api_timeout": 30,
        "max_retries": 3,
        "debug_mode": False,
        "tickers": ["AAPL", "MSFT"],
        "embedding_model": "gemini-3.1-flash-lite-preview",
    }

    def __init__(self, yaml_path: str = None) -> None:
        """Initialize config from YAML file and environment overrides."""
        self.config: Dict[str, Any] = self.DEFAULTS.copy()
        if yaml_path and Path(yaml_path).exists():
            self._load_yaml(yaml_path)
        self._apply_env_overrides()

    def _load_yaml(self, path: str) -> None:
        """Load configuration from YAML file."""
        import yaml
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
                self.config.update(data)
        except Exception as e:
            raise ValueError(f"Failed to load config from {path}: {e}")

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides with type coercion."""
        env_map = {
            "SENTINEL_API_TIMEOUT": ("api_timeout", int),
            "SENTINEL_MAX_RETRIES": ("max_retries", int),
            "SENTINEL_DEBUG_MODE": ("debug_mode", self._parse_bool),
            "SENTINEL_TICKERS": ("tickers", self._parse_list),
            "SENTINEL_EMBEDDING_MODEL": ("embedding_model", str),
        }

        for env_var, (key, coerce_fn) in env_map.items():
            if env_var in os.environ:
                try:
                    self.config[key] = coerce_fn(os.environ[env_var])
                except ValueError as e:
                    raise ValueError(
                        f"Failed to coerce {env_var}={os.environ[env_var]} "
                        f"to type {coerce_fn.__name__}: {e}"
                    )

    @staticmethod
    def _parse_bool(value: str) -> bool:
        """Coerce string to boolean."""
        if isinstance(value, bool):
            return value
        return value.lower() in ("true", "1", "yes", "on")

    @staticmethod
    def _parse_list(value: str) -> list:
        """Coerce comma-separated string to list."""
        if isinstance(value, list):
            return value
        return [x.strip() for x in value.split(",")]

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve config value with optional fallback."""
        return self.config.get(key, default)

    def validate_required(self, *keys: str) -> None:
        """Ensure required keys are present; raise if missing."""
        missing = [k for k in keys if k not in self.config]
        if missing:
            raise KeyError(f"Missing required config keys: {missing}")


# ============================================================================
# TESTS
# ============================================================================


class TestConfigLoaderDefaults:
    """Test that defaults are loaded correctly."""

    def test_default_api_timeout(self) -> None:
        """Default api_timeout should be 30."""
        config = ConfigLoader()
        assert config.get("api_timeout") == 30

    def test_default_max_retries(self) -> None:
        """Default max_retries should be 3."""
        config = ConfigLoader()
        assert config.get("max_retries") == 3

    def test_default_debug_mode(self) -> None:
        """Default debug_mode should be False."""
        config = ConfigLoader()
        assert config.get("debug_mode") is False

    def test_default_tickers(self) -> None:
        """Default tickers should be a list of common symbols."""
        config = ConfigLoader()
        assert config.get("tickers") == ["AAPL", "MSFT"]

    def test_default_embedding_model(self) -> None:
        """Default embedding_model should be Gemini Flash Lite."""
        config = ConfigLoader()
        assert config.get("embedding_model") == "gemini-3.1-flash-lite-preview"


class TestConfigLoaderYamlOverride:
    """Test that YAML files override defaults."""

    def test_yaml_override_api_timeout(self) -> None:
        """YAML api_timeout should override default."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            f.write("api_timeout: 60\n")
            f.flush()
            try:
                config = ConfigLoader(f.name)
                assert config.get("api_timeout") == 60
            finally:
                os.unlink(f.name)

    def test_yaml_override_debug_mode(self) -> None:
        """YAML debug_mode should override default."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            f.write("debug_mode: true\n")
            f.flush()
            try:
                config = ConfigLoader(f.name)
                assert config.get("debug_mode") is True
            finally:
                os.unlink(f.name)

    def test_yaml_override_tickers(self) -> None:
        """YAML tickers should override default."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            f.write("tickers:\n  - GOOGL\n  - TSLA\n")
            f.flush()
            try:
                config = ConfigLoader(f.name)
                assert config.get("tickers") == ["GOOGL", "TSLA"]
            finally:
                os.unlink(f.name)

    def test_yaml_missing_file(self) -> None:
        """Missing YAML file should not raise; use defaults."""
        config = ConfigLoader("/nonexistent/config.yaml")
        assert config.get("api_timeout") == 30


class TestConfigLoaderEnvOverride:
    """Test that environment variables override YAML and defaults."""

    def test_env_override_api_timeout(self) -> None:
        """SENTINEL_API_TIMEOUT should override YAML and default."""
        os.environ["SENTINEL_API_TIMEOUT"] = "90"
        try:
            config = ConfigLoader()
            assert config.get("api_timeout") == 90
        finally:
            del os.environ["SENTINEL_API_TIMEOUT"]

    def test_env_override_debug_mode_true(self) -> None:
        """SENTINEL_DEBUG_MODE=true should set debug_mode to True."""
        os.environ["SENTINEL_DEBUG_MODE"] = "true"
        try:
            config = ConfigLoader()
            assert config.get("debug_mode") is True
        finally:
            del os.environ["SENTINEL_DEBUG_MODE"]

    def test_env_override_debug_mode_yes(self) -> None:
        """SENTINEL_DEBUG_MODE=yes should set debug_mode to True."""
        os.environ["SENTINEL_DEBUG_MODE"] = "yes"
        try:
            config = ConfigLoader()
            assert config.get("debug_mode") is True
        finally
