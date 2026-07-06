"""
Unit test suite for Sentinel's configuration loader.

Tests env var overrides, missing key handling, type coercion, and validation
of the config system that powers all pillars (scout, linguist, historian, judge).
Part of the Sentinel Sentiment Engine test harness.
"""

import os
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """Minimal config loader for testing — mimics production behavior."""

    def __init__(self, config_dict: Dict[str, Any] | None = None) -> None:
        """Initialize with optional base config dict."""
        self.config = config_dict or {}

    def load_env(self, key: str, default: Any = None, coerce_type: type | None = None) -> Any:
        """Load a config value from env var, with optional type coercion."""
        val = os.environ.get(key, default)
        if val is None:
            return None
        if coerce_type is int:
            return int(val)
        if coerce_type is float:
            return float(val)
        if coerce_type is bool:
            return val.lower() in ("true", "1", "yes")
        return val

    def require(self, key: str, coerce_type: type | None = None) -> Any:
        """Load a required env var; raise KeyError if missing."""
        val = os.environ.get(key)
        if val is None:
            raise KeyError(f"Required config key not found: {key}")
        if coerce_type is int:
            return int(val)
        if coerce_type is float:
            return float(val)
        if coerce_type is bool:
            return val.lower() in ("true", "1", "yes")
        return val

    def from_file(self, path: str) -> Dict[str, str]:
        """Load config from a YAML-like file (key=value format for simplicity)."""
        config = {}
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    config[k.strip()] = v.strip()
        return config


class TestConfigLoaderBasic:
    """Test basic config loader functionality."""

    def test_load_env_string_default(self) -> None:
        """Load string from env with fallback default."""
        os.environ["TEST_STRING"] = "hello"
        loader = ConfigLoader()
        assert loader.load_env("TEST_STRING", default="fallback") == "hello"
        del os.environ["TEST_STRING"]

    def test_load_env_string_missing_uses_default(self) -> None:
        """Load string from env; use default when key absent."""
        if "MISSING_KEY" in os.environ:
            del os.environ["MISSING_KEY"]
        loader = ConfigLoader()
        assert loader.load_env("MISSING_KEY", default="fallback") == "fallback"

    def test_load_env_string_missing_no_default(self) -> None:
        """Load string from env; return None when key absent and no default."""
        if "MISSING_KEY" in os.environ:
            del os.environ["MISSING_KEY"]
        loader = ConfigLoader()
        assert loader.load_env("MISSING_KEY") is None


class TestConfigLoaderIntCoercion:
    """Test integer type coercion."""

    def test_coerce_int_from_env(self) -> None:
        """Coerce string env var to int."""
        os.environ["TEST_INT"] = "42"
        loader = ConfigLoader()
        result = loader.load_env("TEST_INT", coerce_type=int)
        assert result == 42
        assert isinstance(result, int)
        del os.environ["TEST_INT"]

    def test_coerce_int_invalid_raises(self) -> None:
        """Raise ValueError on invalid int coercion."""
        os.environ["TEST_INT"] = "not_a_number"
        loader = ConfigLoader()
        with pytest.raises(ValueError):
            loader.load_env("TEST_INT", coerce_type=int)
        del os.environ["TEST_INT"]

    def test_coerce_int_default(self) -> None:
        """Return default (not coerced) when env var missing."""
        if "MISSING_INT" in os.environ:
            del os.environ["MISSING_INT"]
        loader = ConfigLoader()
        result = loader.load_env("MISSING_INT", default=10, coerce_type=int)
        assert result == 10


class TestConfigLoaderFloatCoercion:
    """Test float type coercion."""

    def test_coerce_float_from_env(self) -> None:
        """Coerce string env var to float."""
        os.environ["TEST_FLOAT"] = "3.14"
        loader = ConfigLoader()
        result = loader.load_env("TEST_FLOAT", coerce_type=float)
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)
        del os.environ["TEST_FLOAT"]

    def test_coerce_float_invalid_raises(self) -> None:
        """Raise ValueError on invalid float coercion."""
        os.environ["TEST_FLOAT"] = "not_a_float"
        loader = ConfigLoader()
        with pytest.raises(ValueError):
            loader.load_env("TEST_FLOAT", coerce_type=float)
        del os.environ["TEST_FLOAT"]


class TestConfigLoaderBoolCoercion:
    """Test boolean type coercion."""

    @pytest.mark.parametrize("val,expected", [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("YES", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
    ])
    def test_coerce_bool_variants(self, val: str, expected: bool) -> None:
        """Coerce various string representations to bool."""
        os.environ["TEST_BOOL"] = val
        loader = ConfigLoader()
        result = loader.load_env("TEST_BOOL", coerce_type=bool)
        assert result is expected
        assert isinstance(result, bool)
        del os.environ["TEST_BOOL"]


class TestConfigLoaderRequired:
    """Test required key enforcement."""

    def test_require_present_key(self) -> None:
        """Require an env var that is present."""
        os.environ["REQUIRED_KEY"] = "value123"
        loader = ConfigLoader()
        assert loader.require("REQUIRED_KEY") == "value123"
        del os.environ["REQUIRED_KEY"]

    def test_require_missing_key_raises(self) -> None:
        """Raise KeyError when required key is missing."""
        if "MISSING_REQUIRED" in os.environ:
            del os.environ["MISSING_REQUIRED"]
        loader = ConfigLoader()
        with pytest.raises(KeyError, match="Required config key not found"):
            loader.require("MISSING_REQUIRED")

    def test_require_with_int_coercion(self) -> None:
        """Require an env var and coerce to int."""
        os.environ["REQUIRED_INT"] = "99"
        loader = ConfigLoader()
        result = loader.require("REQUIRED_INT", coerce_type=int)
        assert result == 99
        assert isinstance(result, int)
        del os.environ["REQUIRED_INT"]

    def test_require_missing_with_int_coercion_raises(self) -> None:
        """Raise KeyError before attempting coercion on missing key."""
        if "MISSING_INT" in os.environ:
            del os.environ["MISSING_INT"]
        loader = ConfigLoader()
        with pytest.raises(KeyError):
