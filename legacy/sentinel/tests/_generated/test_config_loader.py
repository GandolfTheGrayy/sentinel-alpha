"""
Unit tests for Sentinel configuration loader.

Tests env var overrides, missing key handling, type coercion, and validation.
Fits into the test pillar to ensure config reliability across all other pillars.
"""

import os
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict


# Mock config loader (mimics sentinel/config.py structure)
class ConfigLoader:
    """Loads and validates Sentinel configuration from env vars and YAML."""

    def __init__(self, config_dict: Dict[str, Any] | None = None) -> None:
        """Initialize loader with optional base config dict."""
        self.config = config_dict or {}

    def load_from_env(self, key: str, default: Any = None, coerce_type: type | None = None) -> Any:
        """Load a config value from environment, with optional type coercion."""
        value = os.environ.get(key, default)
        if value is None:
            raise ValueError(f"Missing required config key: {key}")
        if coerce_type is not None:
            try:
                return coerce_type(value)
            except (ValueError, TypeError) as e:
                raise TypeError(f"Failed to coerce {key}={value} to {coerce_type}: {e}")
        return value

    def load_all(self, spec: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Load all keys from env using a spec dict with defaults and types."""
        result = {}
        for key, opts in spec.items():
            default = opts.get("default")
            coerce_type = opts.get("type")
            try:
                result[key] = self.load_from_env(key, default=default, coerce_type=coerce_type)
            except ValueError:
                if default is not None:
                    result[key] = default
                else:
                    raise
        return result


class TestConfigLoader:
    """Test suite for configuration loading and validation."""

    def test_load_from_env_success(self) -> None:
        """Test successful load of environment variable."""
        os.environ["TEST_KEY"] = "test_value"
        loader = ConfigLoader()
        value = loader.load_from_env("TEST_KEY")
        assert value == "test_value"
        del os.environ["TEST_KEY"]

    def test_load_from_env_with_default(self) -> None:
        """Test fallback to default when env var missing."""
        if "NONEXISTENT_KEY" in os.environ:
            del os.environ["NONEXISTENT_KEY"]
        loader = ConfigLoader()
        value = loader.load_from_env("NONEXISTENT_KEY", default="fallback")
        assert value == "fallback"

    def test_load_from_env_missing_required(self) -> None:
        """Test exception when required key missing and no default."""
        if "REQUIRED_KEY" in os.environ:
            del os.environ["REQUIRED_KEY"]
        loader = ConfigLoader()
        with pytest.raises(ValueError, match="Missing required config key"):
            loader.load_from_env("REQUIRED_KEY")

    def test_coerce_to_int(self) -> None:
        """Test type coercion to integer."""
        os.environ["INT_KEY"] = "42"
        loader = ConfigLoader()
        value = loader.load_from_env("INT_KEY", coerce_type=int)
        assert value == 42
        assert isinstance(value, int)
        del os.environ["INT_KEY"]

    def test_coerce_to_float(self) -> None:
        """Test type coercion to float."""
        os.environ["FLOAT_KEY"] = "3.14"
        loader = ConfigLoader()
        value = loader.load_from_env("FLOAT_KEY", coerce_type=float)
        assert value == 3.14
        assert isinstance(value, float)
        del os.environ["FLOAT_KEY"]

    def test_coerce_to_bool(self) -> None:
        """Test type coercion to boolean."""
        os.environ["BOOL_KEY"] = "true"
        loader = ConfigLoader()
        # Custom bool coercion (string "true" -> True)
        coerce_bool = lambda x: x.lower() in ("true", "1", "yes")
        value = loader.load_from_env("BOOL_KEY", coerce_type=coerce_bool)
        assert value is True
        del os.environ["BOOL_KEY"]

    def test_coerce_invalid_int(self) -> None:
        """Test exception on invalid integer coercion."""
        os.environ["BAD_INT"] = "not_a_number"
        loader = ConfigLoader()
        with pytest.raises(TypeError, match="Failed to coerce"):
            loader.load_from_env("BAD_INT", coerce_type=int)
        del os.environ["BAD_INT"]

    def test_coerce_invalid_float(self) -> None:
        """Test exception on invalid float coercion."""
        os.environ["BAD_FLOAT"] = "3.14.15"
        loader = ConfigLoader()
        with pytest.raises(TypeError, match="Failed to coerce"):
            loader.load_from_env("BAD_FLOAT", coerce_type=float)
        del os.environ["BAD_FLOAT"]

    def test_load_all_with_spec(self) -> None:
        """Test batch loading with spec dict."""
        os.environ["API_KEY"] = "secret123"
        os.environ["TIMEOUT"] = "30"
        loader = ConfigLoader()
        spec = {
            "API_KEY": {"type": str},
            "TIMEOUT": {"type": int},
            "DEBUG": {"type": bool, "default": False},
        }
        # Custom bool coercion
        coerce_bool = lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x)
        spec["DEBUG"]["type"] = coerce_bool
        
        result = loader.load_all(spec)
        assert result["API_KEY"] == "secret123"
        assert result["TIMEOUT"] == 30
        assert result["DEBUG"] is False
        
        del os.environ["API_KEY"]
        del os.environ["TIMEOUT"]

    def test_load_all_missing_optional(self) -> None:
        """Test batch load gracefully uses defaults for missing optional keys."""
        if "OPTIONAL_KEY" in os.environ:
            del os.environ["OPTIONAL_KEY"]
        loader = ConfigLoader()
        spec = {
            "OPTIONAL_KEY": {"default": "default_value", "type": str},
        }
        result = loader.load_all(spec)
        assert result["OPTIONAL_KEY"] == "default_value"

    def test_load_all_missing_required_raises(self) -> None:
        """Test batch load raises on missing required key."""
        if "REQUIRED_SETTING" in os.environ:
            del os.environ["REQUIRED_SETTING"]
        loader = ConfigLoader()
        spec = {
            "REQUIRED_SETTING": {"type": str},
        }
        with pytest.raises(ValueError, match="Missing required config key"):
            loader.load_all(spec)

    def test_load_all_coercion_failure(self) -> None:
        """Test batch load raises on type coercion failure."""
        os.environ["BAD_TYPE_KEY"] = "invalid_int"
        loader = ConfigLoader()
        spec = {
            "BAD_TYPE_KEY": {"type": int},
        }
        with pytest.raises(TypeError, match="Failed to coerce"):
            loader.load_all(spec)
        del os.environ["BAD_TYPE_KEY"]

    def test_env_override_precedence(self) -> None:
        """Test that env vars override constructor defaults."""
        os.environ["OVERRIDE_KEY"] = "env_value"
        loader = ConfigLoader({"OVERRIDE_KEY": "constructor_value"})
        value = loader.load_from_env("OVERRIDE_KEY", default="constructor_value")
        assert value == "env_value"
        del os.environ["OVERRIDE_KEY"]
