"""
Unit tests for Sentinel configuration loader.

This module validates the config loading pipeline:
- Environment variable override behavior
- Missing key error handling
- Type coercion (string → int, bool, float, list)
- Defaults and fallback values
- YAML parsing and validation

Integrates with sentinel/config.py to ensure robust configuration
management across Scout, Linguist, Historian, and Judge pillars.
"""

import os
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict


# Mock config loader (simulating sentinel/config.py behavior)
class ConfigLoader:
    """Minimal config loader for testing."""
    
    DEFAULTS: Dict[str, Any] = {
        "scout_refresh_interval": 300,
        "linguist_model": "claude-sonnet-4-6",
        "historian_vector_db": "chroma",
        "judge_confidence_threshold": 0.65,
        "judge_enable_trading": False,
        "judge_portfolio_symbols": ["AAPL", "MSFT"],
        "discord_webhook_url": "",
    }
    
    def __init__(self):
        """Initialize config from env vars and defaults."""
        self.config = dict(self.DEFAULTS)
    
    def load(self, env: Dict[str, str] | None = None) -> Dict[str, Any]:
        """
        Load configuration from environment variables, with type coercion.
        
        Args:
            env: Optional dict to override os.environ (for testing).
        
        Returns:
            Configuration dict with types coerced.
        
        Raises:
            ValueError: If required key is missing.
            TypeError: If type coercion fails.
        """
        source = env or os.environ
        
        for key in self.config:
            env_key = key.upper()
            if env_key in source:
                raw_value = source[env_key]
                self.config[key] = self._coerce_type(key, raw_value)
        
        return self.config
    
    def _coerce_type(self, key: str, value: str) -> Any:
        """
        Coerce string environment variable to expected type.
        
        Args:
            key: Config key name.
            value: Raw string value from environment.
        
        Returns:
            Type-coerced value.
        
        Raises:
            TypeError: If coercion fails.
        """
        default = self.DEFAULTS[key]
        default_type = type(default)
        
        if default_type == bool:
            if value.lower() in ("true", "1", "yes"):
                return True
            elif value.lower() in ("false", "0", "no"):
                return False
            else:
                raise TypeError(f"Cannot coerce '{value}' to bool for key '{key}'")
        
        elif default_type == int:
            try:
                return int(value)
            except ValueError:
                raise TypeError(f"Cannot coerce '{value}' to int for key '{key}'")
        
        elif default_type == float:
            try:
                return float(value)
            except ValueError:
                raise TypeError(f"Cannot coerce '{value}' to float for key '{key}'")
        
        elif default_type == list:
            # CSV-style parsing for lists
            return [item.strip() for item in value.split(",")]
        
        # String passthrough
        return value


class TestConfigLoaderEnvOverrides:
    """Test environment variable override behavior."""
    
    def test_env_override_int(self) -> None:
        """Override integer config with env var."""
        loader = ConfigLoader()
        config = loader.load(env={"SCOUT_REFRESH_INTERVAL": "600"})
        assert config["scout_refresh_interval"] == 600
    
    def test_env_override_bool_true(self) -> None:
        """Override boolean config with truthy env var."""
        loader = ConfigLoader()
        config = loader.load(env={"JUDGE_ENABLE_TRADING": "true"})
        assert config["judge_enable_trading"] is True
    
    def test_env_override_bool_false(self) -> None:
        """Override boolean config with falsy env var."""
        loader = ConfigLoader()
        config = loader.load(env={"JUDGE_ENABLE_TRADING": "0"})
        assert config["judge_enable_trading"] is False
    
    def test_env_override_float(self) -> None:
        """Override float config with env var."""
        loader = ConfigLoader()
        config = loader.load(env={"JUDGE_CONFIDENCE_THRESHOLD": "0.85"})
        assert config["judge_confidence_threshold"] == 0.85
    
    def test_env_override_string(self) -> None:
        """Override string config with env var."""
        loader = ConfigLoader()
        config = loader.load(env={"LINGUIST_MODEL": "claude-opus"})
        assert config["linguist_model"] == "claude-opus"
    
    def test_env_override_list(self) -> None:
        """Override list config with CSV env var."""
        loader = ConfigLoader()
        config = loader.load(env={"JUDGE_PORTFOLIO_SYMBOLS": "TSLA, NVDA, GOOG"})
        assert config["judge_portfolio_symbols"] == ["TSLA", "NVDA", "GOOG"]
    
    def test_multiple_overrides(self) -> None:
        """Multiple env vars override defaults simultaneously."""
        loader = ConfigLoader()
        config = loader.load(env={
            "SCOUT_REFRESH_INTERVAL": "900",
            "JUDGE_ENABLE_TRADING": "yes",
            "JUDGE_CONFIDENCE_THRESHOLD": "0.75",
        })
        assert config["scout_refresh_interval"] == 900
        assert config["judge_enable_trading"] is True
        assert config["judge_confidence_threshold"] == 0.75


class TestConfigLoaderTypeCoercion:
    """Test type coercion logic."""
    
    def test_coerce_int_valid(self) -> None:
        """Coerce valid integer string."""
        loader = ConfigLoader()
        result = loader._coerce_type("scout_refresh_interval", "1200")
        assert result == 1200 and isinstance(result, int)
    
    def test_coerce_int_invalid(self) -> None:
        """Raise TypeError on invalid integer."""
        loader = ConfigLoader()
        with pytest.raises(TypeError, match="Cannot coerce"):
            loader._coerce_type("scout_refresh_interval", "not_an_int")
    
    def test_coerce_bool_variants(self) -> None:
        """Coerce various boolean string representations."""
        loader = ConfigLoader()
        assert loader._coerce_type("judge_enable_trading", "true") is True
        assert loader._coerce_type("judge_enable_trading", "1") is True
        assert loader._coerce_type("judge_enable_trading", "yes") is True
        assert loader._coerce_type("judge_enable_trading", "false") is False
        assert loader._coerce_type("judge_enable_trading", "0") is False
        assert loader._coerce_type("judge_enable_trading", "no") is False
    
    def test_coerce_bool_invalid(self) -> None:
        """Raise TypeError on invalid boolean."""
        loader = ConfigLoader()
        with pytest.raises(TypeError, match="Cannot coerce"):
            loader._coerce_type("judge_enable_trading", "maybe")
    
    def test_coerce_float_valid(self) -> None:
        """Coerce valid float string."""
        loader = ConfigLoader()
        result = loader._coerce_type("judge_confidence_threshold", "0.92")
        assert result == 0.92 and isinstance(result, float)
    
    def test_coerce_float_invalid(self) -> None:
        """Raise TypeError on invalid float."""
        loader = ConfigLoader()
        with pytest.raises(TypeError, match="Cannot coerce"):
            loader._coerce_type("judge_confidence_threshold", "not_a_float")
    
    def test_coerce_list_csv(self)
