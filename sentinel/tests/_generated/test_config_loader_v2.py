"""
Unit tests for Sentinel's configuration loader.

This module tests the config system's ability to:
  - Load and parse YAML configuration files
  - Override values via environment variables
  - Handle missing keys with sensible defaults
  - Coerce types (strings to ints, bools, lists)
  - Raise on critical missing required keys

Tests are designed to run in isolation without external dependencies.
"""

import os
import tempfile
import pytest
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """Minimal config loader for testing purposes."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict

    def get(self, key: str, default: Any = None, coerce: type = None) -> Any:
        """Retrieve a config value with optional type coercion and environment override."""
        env_key = f"SENTINEL_{key.upper().replace('.', '_')}"
        
        if env_key in os.environ:
            value = os.environ[env_key]
            if coerce:
                if coerce is bool:
                    return value.lower() in ('true', '1', 'yes')
                elif coerce is int:
                    return int(value)
                elif coerce is float:
                    return float(value)
                elif coerce is list:
                    return value.split(',')
            return value
        
        keys = key.split('.')
        val = self.config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                val = None
            if val is None:
                return default
        
        if coerce and val is not None:
            if coerce is bool:
                return str(val).lower() in ('true', '1', 'yes')
            elif coerce is int:
                return int(val)
            elif coerce is float:
                return float(val)
            elif coerce is list:
                if isinstance(val, list):
                    return val
                return str(val).split(',')
        
        return val

    def require(self, key: str) -> Any:
        """Retrieve a required config value; raise KeyError if missing."""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Required config key '{key}' not found and no env override.")
        return value


class TestConfigLoader:
    """Test suite for ConfigLoader."""

    def setup_method(self) -> None:
        """Clear environment before each test."""
        for key in list(os.environ.keys()):
            if key.startswith('SENTINEL_'):
                del os.environ[key]

    def teardown_method(self) -> None:
        """Clean up environment after each test."""
        for key in list(os.environ.keys()):
            if key.startswith('SENTINEL_'):
                del os.environ[key]

    def test_simple_string_retrieval(self) -> None:
        """Test retrieval of a simple string value from config."""
        config = ConfigLoader({'api_key': 'secret123'})
        assert config.get('api_key') == 'secret123'

    def test_nested_key_retrieval(self) -> None:
        """Test retrieval of nested dictionary keys using dot notation."""
        config = ConfigLoader({
            'database': {
                'host': 'localhost',
                'port': 5432
            }
        })
        assert config.get('database.host') == 'localhost'
        assert config.get('database.port') == 5432

    def test_default_value_for_missing_key(self) -> None:
        """Test that missing keys return the provided default value."""
        config = ConfigLoader({'existing': 'value'})
        assert config.get('missing', default='fallback') == 'fallback'

    def test_none_default_for_truly_missing_key(self) -> None:
        """Test that missing keys with no default return None."""
        config = ConfigLoader({})
        assert config.get('missing') is None

    def test_env_override_simple_key(self) -> None:
        """Test that environment variables override config file values."""
        config = ConfigLoader({'api_key': 'file_secret'})
        os.environ['SENTINEL_API_KEY'] = 'env_secret'
        assert config.get('api_key') == 'env_secret'

    def test_env_override_nested_key(self) -> None:
        """Test that environment variables override nested config values."""
        config = ConfigLoader({'database': {'host': 'localhost'}})
        os.environ['SENTINEL_DATABASE_HOST'] = 'prod.example.com'
        assert config.get('database.host') == 'prod.example.com'

    def test_type_coercion_int(self) -> None:
        """Test coercion of string and int values to int type."""
        config = ConfigLoader({'timeout': 30})
        assert config.get('timeout', coerce=int) == 30
        assert isinstance(config.get('timeout', coerce=int), int)

    def test_type_coercion_int_from_env(self) -> None:
        """Test coercion of environment string to int."""
        config = ConfigLoader({})
        os.environ['SENTINEL_PORT'] = '8080'
        assert config.get('port', coerce=int) == 8080
        assert isinstance(config.get('port', coerce=int), int)

    def test_type_coercion_bool_true_variants(self) -> None:
        """Test coercion of various true-like strings to bool."""
        config = ConfigLoader({})
        os.environ['SENTINEL_DEBUG'] = 'true'
        assert config.get('debug', coerce=bool) is True
        
        os.environ['SENTINEL_ENABLED'] = '1'
        assert config.get('enabled', coerce=bool) is True
        
        os.environ['SENTINEL_ACTIVE'] = 'yes'
        assert config.get('active', coerce=bool) is True

    def test_type_coercion_bool_false_variants(self) -> None:
        """Test coercion of false-like strings to bool."""
        config = ConfigLoader({})
        os.environ['SENTINEL_DEBUG'] = 'false'
        assert config.get('debug', coerce=bool) is False
        
        os.environ['SENTINEL_ENABLED'] = '0'
        assert config.get('enabled', coerce=bool) is False
        
        os.environ['SENTINEL_ACTIVE'] = 'no'
        assert config.get('active', coerce=bool) is False

    def test_type_coercion_float(self) -> None:
        """Test coercion of string and numeric values to float."""
        config = ConfigLoader({'confidence_threshold': 0.85})
        assert config.get('confidence_threshold', coerce=float) == 0.85
        assert isinstance(config.get('confidence_threshold', coerce=float), float)

    def test_type_coercion_float_from_env(self) -> None:
        """Test coercion of environment string to float."""
        config = ConfigLoader({})
        os.environ['SENTINEL_THRESHOLD'] = '0.95'
        assert config.get('threshold', coerce=float) == 0.95

    def test_type_coercion_list_from_string(self) -> None:
        """Test coercion of comma-separated string to list."""
        config = ConfigLoader({})
        os.environ['SENTINEL_TICKERS'] = 'AAPL,MSFT,TSLA'
        result = config.get('tickers', coerce=list)
        assert result == ['AAPL', 'MSFT', 'TSLA']

    def test_type_coercion_list_from_list(self) -> None:
        """Test that list values remain lists when coerce=list."""
        config = ConfigLoader({'tickers': ['AAPL', 'MSFT']})
        result = config.get('tickers', coerce=list)
        assert result == ['AAPL', 'MSFT']

    def test_require_key_present(self) -> None:
