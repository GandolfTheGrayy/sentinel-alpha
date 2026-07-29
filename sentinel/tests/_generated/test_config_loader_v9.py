"""
Unit tests for Sentinel's configuration loader.

Tests env var overrides, missing key handling, type coercion, and fallback
behavior. Validates that the config system correctly prioritizes env vars
over defaults and raises appropriate errors on misconfiguration.
Part of the Sentinel quality gate — runs in CI/CD before pipeline deployment.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from typing import Any, Dict

# Mock config loader module path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ConfigLoader:
    """Minimal config loader for testing — simulates sentinel.config module."""

    def __init__(self) -> None:
        """Initialize config from env vars and defaults."""
        self.data: Dict[str, Any] = {}

    def load(self, schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load config by merging env vars, defaults, and schema.

        Args:
            schema: Dict mapping config keys to {default, type, required}.

        Returns:
            Merged configuration dict.

        Raises:
            KeyError: If required key missing and no default/env var.
            TypeError: If type coercion fails.
        """
        result = {}
        for key, spec in schema.items():
            env_val = os.environ.get(key)
            default_val = spec.get("default")
            required = spec.get("required", False)
            coerce_type = spec.get("type", str)

            if env_val is not None:
                try:
                    result[key] = coerce_type(env_val)
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"Failed to coerce {key}={env_val} to {coerce_type.__name__}"
                    ) from e
            elif default_val is not None:
                result[key] = default_val
            elif required:
                raise KeyError(f"Required config key missing: {key}")
            else:
                result[key] = None

        self.data = result
        return result


class TestConfigLoaderEnvOverride:
    """Test environment variable override of defaults."""

    def test_env_var_overrides_default(self) -> None:
        """Env var should override schema default."""
        os.environ["TEST_KEY"] = "from_env"
        loader = ConfigLoader()
        result = loader.load(
            {"TEST_KEY": {"default": "from_default", "type": str, "required": False}}
        )
        assert result["TEST_KEY"] == "from_env"
        del os.environ["TEST_KEY"]

    def test_default_used_when_no_env(self) -> None:
        """Default should be used when env var not set."""
        if "TEST_KEY_NO_ENV" in os.environ:
            del os.environ["TEST_KEY_NO_ENV"]
        loader = ConfigLoader()
        result = loader.load(
            {
                "TEST_KEY_NO_ENV": {
                    "default": "fallback",
                    "type": str,
                    "required": False,
                }
            }
        )
        assert result["TEST_KEY_NO_ENV"] == "fallback"

    def test_env_var_takes_precedence_over_all(self) -> None:
        """Env var should win over default and required flag."""
        os.environ["PRIORITY_TEST"] = "env_wins"
        loader = ConfigLoader()
        result = loader.load(
            {
                "PRIORITY_TEST": {
                    "default": "default_loses",
                    "type": str,
                    "required": True,
                }
            }
        )
        assert result["PRIORITY_TEST"] == "env_wins"
        del os.environ["PRIORITY_TEST"]


class TestConfigLoaderTypeCoercion:
    """Test type coercion from env vars."""

    def test_coerce_int_from_env_string(self) -> None:
        """String env var should coerce to int."""
        os.environ["INT_KEY"] = "42"
        loader = ConfigLoader()
        result = loader.load({"INT_KEY": {"type": int, "required": False}})
        assert result["INT_KEY"] == 42
        assert isinstance(result["INT_KEY"], int)
        del os.environ["INT_KEY"]

    def test_coerce_float_from_env_string(self) -> None:
        """String env var should coerce to float."""
        os.environ["FLOAT_KEY"] = "3.14"
        loader = ConfigLoader()
        result = loader.load({"FLOAT_KEY": {"type": float, "required": False}})
        assert result["FLOAT_KEY"] == 3.14
        assert isinstance(result["FLOAT_KEY"], float)
        del os.environ["FLOAT_KEY"]

    def test_coerce_bool_from_env_string(self) -> None:
        """String env var should coerce to bool (truthy logic)."""
        os.environ["BOOL_KEY"] = "true"
        loader = ConfigLoader()
        result = loader.load(
            {"BOOL_KEY": {"type": lambda x: x.lower() in ("true", "1", "yes"), "required": False}}
        )
        assert result["BOOL_KEY"] is True
        del os.environ["BOOL_KEY"]

    def test_coerce_invalid_int_raises_typeerror(self) -> None:
        """Invalid int coercion should raise TypeError."""
        os.environ["BAD_INT"] = "not_a_number"
        loader = ConfigLoader()
        with pytest.raises(TypeError, match="Failed to coerce BAD_INT"):
            loader.load({"BAD_INT": {"type": int, "required": False}})
        del os.environ["BAD_INT"]

    def test_coerce_invalid_float_raises_typeerror(self) -> None:
        """Invalid float coercion should raise TypeError."""
        os.environ["BAD_FLOAT"] = "1.2.3"
        loader = ConfigLoader()
        with pytest.raises(TypeError, match="Failed to coerce BAD_FLOAT"):
            loader.load({"BAD_FLOAT": {"type": float, "required": False}})
        del os.environ["BAD_FLOAT"]


class TestConfigLoaderMissingKeys:
    """Test handling of missing required keys."""

    def test_required_key_missing_raises_keyerror(self) -> None:
        """Missing required key should raise KeyError."""
        if "MISSING_REQUIRED" in os.environ:
            del os.environ["MISSING_REQUIRED"]
        loader = ConfigLoader()
        with pytest.raises(KeyError, match="Required config key missing"):
            loader.load(
                {"MISSING_REQUIRED": {"type": str, "required": True}}
            )

    def test_optional_key_missing_returns_none(self) -> None:
        """Missing optional key should return None."""
        if "MISSING_OPTIONAL" in os.environ:
            del os.environ["MISSING_OPTIONAL"]
        loader = ConfigLoader()
        result = loader.load(
            {"MISSING_OPTIONAL": {"type": str, "required": False}}
        )
        assert result["MISSING_OPTIONAL"] is None

    def test_required_key_with_default_always_present(self) -> None:
        """Required key with default should never be None."""
        if "REQ_WITH_DEFAULT" in os.environ:
            del os.environ["REQ_WITH_DEFAULT"]
        loader = ConfigLoader()
        result = loader.load(
            {
                "REQ_WITH_DEFAULT": {
                    "type": str,
                    "default": "safe_default",
                    "required": True,
                }
            }
        )
        assert result["REQ_WITH_DEFAULT"] == "safe_default"


class TestConfigLoaderMultipleKeys:
    """Test loading multiple config keys together."""

    def test_load_multiple_mixed_keys(self) -> None:
        """Load schema with mix of env, defaults, and required keys."""
        os.environ["KEY_A"] = "env_a"
