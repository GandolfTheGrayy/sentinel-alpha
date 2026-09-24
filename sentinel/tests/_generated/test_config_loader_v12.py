"""
Unit tests for the Sentinel configuration loader.

Tests environment variable overrides, missing key handling, type coercion,
and default fallback behavior. Part of the Spine validation suite.
"""

import os
import pytest
import tempfile
import json
from pathlib import Path
from typing import Any, Dict


# Mock config loader implementation for testing
class ConfigLoader:
    """Loads Sentinel configuration from env vars and YAML with type coercion."""

    def __init__(self, config_path: str | None = None):
        """Initialize config loader with optional YAML file path."""
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load config from file and environment."""
        if self.config_path and Path(self.config_path).exists():
            import yaml
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = {}

    def get(self, key: str, default: Any = None, coerce: type | None = None) -> Any:
        """
        Retrieve config value with env var override, type coercion, and fallback.
        
        Args:
            key: Config key (supports dot notation: "section.subsection.key")
            default: Fallback value if key not found
            coerce: Type to coerce value to (int, float, bool, str)
        
        Returns:
            Coerced config value or default
        """
        # Check environment variable first (uppercase, replace dots with underscores)
        env_key = "SENTINEL_" + key.upper().replace(".", "_")
        env_value = os.environ.get(env_key)

        if env_value is not None:
            return self._coerce(env_value, coerce)

        # Navigate nested dict via dot notation
        value = self._config
        for part in key.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
            if value is None:
                break

        if value is not None:
            return self._coerce(value, coerce)

        return self._coerce(default, coerce) if default is not None else None

    def _coerce(self, value: Any, coerce: type | None) -> Any:
        """Coerce value to target type."""
        if value is None or coerce is None:
            return value

        if coerce is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return bool(value)

        if coerce is int:
            return int(value)

        if coerce is float:
            return float(value)

        if coerce is str:
            return str(value)

        return value


class TestConfigLoader:
    """Test suite for ConfigLoader."""

    @pytest.fixture(autouse=True)
    def cleanup_env(self) -> None:
        """Clean up SENTINEL_ env vars before and after each test."""
        sentinel_vars = [k for k in os.environ if k.startswith("SENTINEL_")]
        for var in sentinel_vars:
            del os.environ[var]
        yield
        sentinel_vars = [k for k in os.environ if k.startswith("SENTINEL_")]
        for var in sentinel_vars:
            del os.environ[var]

    def test_get_default_value_when_key_missing(self) -> None:
        """Return default when key not in config or env."""
        loader = ConfigLoader()
        result = loader.get("missing_key", default="fallback")
        assert result == "fallback"

    def test_get_none_when_key_missing_and_no_default(self) -> None:
        """Return None when key not found and no default provided."""
        loader = ConfigLoader()
        result = loader.get("nonexistent")
        assert result is None

    def test_env_var_override_simple_key(self) -> None:
        """Environment variable overrides config file for simple key."""
        os.environ["SENTINEL_API_KEY"] = "env_secret"
        loader = ConfigLoader()
        result = loader.get("api_key")
        assert result == "env_secret"

    def test_env_var_override_nested_key(self) -> None:
        """Environment variable overrides nested config key via dot notation."""
        os.environ["SENTINEL_DATABASE_HOST"] = "env.example.com"
        loader = ConfigLoader()
        result = loader.get("database.host")
        assert result == "env.example.com"

    def test_coerce_string_to_int(self) -> None:
        """Coerce string env var to integer type."""
        os.environ["SENTINEL_PORT"] = "8080"
        loader = ConfigLoader()
        result = loader.get("port", coerce=int)
        assert result == 8080
        assert isinstance(result, int)

    def test_coerce_string_to_float(self) -> None:
        """Coerce string env var to float type."""
        os.environ["SENTINEL_THRESHOLD"] = "0.85"
        loader = ConfigLoader()
        result = loader.get("threshold", coerce=float)
        assert result == 0.85
        assert isinstance(result, float)

    def test_coerce_string_to_bool_true_variants(self) -> None:
        """Coerce string env var to boolean (true variants)."""
        loader = ConfigLoader()
        for value in ("true", "True", "TRUE", "1", "yes", "on"):
            os.environ["SENTINEL_ENABLED"] = value
            result = loader.get("enabled", coerce=bool)
            assert result is True, f"Failed for value: {value}"

    def test_coerce_string_to_bool_false_variants(self) -> None:
        """Coerce string env var to boolean (false variants)."""
        loader = ConfigLoader()
        for value in ("false", "False", "0", "no", "off", ""):
            os.environ["SENTINEL_ENABLED"] = value
            result = loader.get("enabled", coerce=bool)
            assert result is False, f"Failed for value: {value}"

    def test_coerce_int_to_bool(self) -> None:
        """Coerce integer to boolean."""
        loader = ConfigLoader()
        assert loader.get("test", default=1, coerce=bool) is True
        assert loader.get("test", default=0, coerce=bool) is False

    def test_load_from_yaml_file(self) -> None:
        """Load configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("api_key: file_secret\ndatabase:\n  host: localhost\n  port: 5432\n")
            f.flush()
            try:
                loader = ConfigLoader(config_path=f.name)
                assert loader.get("api_key") == "file_secret"
                assert loader.get("database.host") == "localhost"
                assert loader.get("database.port") == 5432
            finally:
                os.unlink(f.name)

    def test_env_var_overrides_yaml_file(self) -> None:
        """Environment variable takes precedence over YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("api_key: file_secret\n")
            f.flush()
            try:
                os.environ["SENTINEL_API_KEY"] = "env_override"
                loader = ConfigLoader(config_path=f.name)
                assert loader.get("api_key") == "env_override"
            finally:
                os.unlink(f.name)

    def test_coerce_none_value_returns_none(self) -> None:
        """Coercion of None value returns None."""
        loader = ConfigLoader()
        result = loader.get("missing", coerce=int)
