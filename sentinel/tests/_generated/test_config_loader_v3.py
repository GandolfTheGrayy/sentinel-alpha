"""
Unit tests for Sentinel's config loader — validates environment variable overrides,
missing key handling, type coercion, and YAML parsing in the config subsystem.
Part of the Sentinel Sentiment Engine test harness.
"""

import os
import tempfile
import pytest
import yaml
from pathlib import Path


# Mock config module (simulating sentinel/config.py behavior)
class ConfigLoader:
    """Simulates the config loader used by Sentinel."""
    
    def __init__(self, config_path: str | None = None) -> None:
        """Initialize loader with optional YAML config path."""
        self.config_path = config_path
        self.config = {}
        self._load()
    
    def _load(self) -> None:
        """Load config from YAML file and merge with environment overrides."""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        
        # Environment overrides (prefix: SENTINEL_)
        for key, value in os.environ.items():
            if key.startswith('SENTINEL_'):
                config_key = key[9:].lower()  # Strip 'SENTINEL_' and lowercase
                self.config[config_key] = self._coerce_type(value)
    
    def _coerce_type(self, value: str) -> bool | int | float | str:
        """Coerce string env var to appropriate Python type."""
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    
    def get(self, key: str, default=None):
        """Get config value with optional default."""
        return self.config.get(key, default)
    
    def require(self, key: str):
        """Get config value or raise KeyError if missing."""
        if key not in self.config:
            raise KeyError(f"Missing required config key: {key}")
        return self.config[key]


class TestConfigLoader:
    """Unit tests for ConfigLoader."""
    
    def test_load_from_yaml(self) -> None:
        """Load config from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'api_key': 'test-key', 'debug': True}, f)
            f.flush()
            temp_path = f.name
        
        try:
            loader = ConfigLoader(temp_path)
            assert loader.get('api_key') == 'test-key'
            assert loader.get('debug') is True
        finally:
            os.unlink(temp_path)
    
    def test_env_override_string(self) -> None:
        """Environment variable overrides YAML string config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'api_key': 'yaml-key'}, f)
            f.flush()
            temp_path = f.name
        
        try:
            os.environ['SENTINEL_API_KEY'] = 'env-key'
            loader = ConfigLoader(temp_path)
            assert loader.get('api_key') == 'env-key'
        finally:
            os.unlink(temp_path)
            del os.environ['SENTINEL_API_KEY']
    
    def test_env_override_bool_true(self) -> None:
        """Environment variable coerces to boolean True."""
        os.environ['SENTINEL_DEBUG'] = 'true'
        loader = ConfigLoader()
        assert loader.get('debug') is True
        del os.environ['SENTINEL_DEBUG']
    
    def test_env_override_bool_false(self) -> None:
        """Environment variable coerces to boolean False."""
        os.environ['SENTINEL_DEBUG'] = 'false'
        loader = ConfigLoader()
        assert loader.get('debug') is False
        del os.environ['SENTINEL_DEBUG']
    
    def test_env_override_int(self) -> None:
        """Environment variable coerces to integer."""
        os.environ['SENTINEL_MAX_WORKERS'] = '8'
        loader = ConfigLoader()
        assert loader.get('max_workers') == 8
        assert isinstance(loader.get('max_workers'), int)
        del os.environ['SENTINEL_MAX_WORKERS']
    
    def test_env_override_float(self) -> None:
        """Environment variable coerces to float."""
        os.environ['SENTINEL_CONFIDENCE_THRESHOLD'] = '0.85'
        loader = ConfigLoader()
        assert loader.get('confidence_threshold') == 0.85
        assert isinstance(loader.get('confidence_threshold'), float)
        del os.environ['SENTINEL_CONFIDENCE_THRESHOLD']
    
    def test_env_override_string_not_coercible(self) -> None:
        """Environment variable stays string if not coercible."""
        os.environ['SENTINEL_MODEL_NAME'] = 'claude-sonnet-4-6'
        loader = ConfigLoader()
        assert loader.get('model_name') == 'claude-sonnet-4-6'
        assert isinstance(loader.get('model_name'), str)
        del os.environ['SENTINEL_MODEL_NAME']
    
    def test_get_with_default(self) -> None:
        """get() returns default if key missing."""
        loader = ConfigLoader()
        assert loader.get('nonexistent', 'default-value') == 'default-value'
    
    def test_get_without_default(self) -> None:
        """get() returns None if key missing and no default."""
        loader = ConfigLoader()
        assert loader.get('nonexistent') is None
    
    def test_require_present(self) -> None:
        """require() returns value if key present."""
        os.environ['SENTINEL_API_KEY'] = 'secret'
        loader = ConfigLoader()
        assert loader.require('api_key') == 'secret'
        del os.environ['SENTINEL_API_KEY']
    
    def test_require_missing_raises(self) -> None:
        """require() raises KeyError if key missing."""
        loader = ConfigLoader()
        with pytest.raises(KeyError, match="Missing required config key"):
            loader.require('nonexistent')
    
    def test_empty_yaml(self) -> None:
        """Load from empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('')
            f.flush()
            temp_path = f.name
        
        try:
            loader = ConfigLoader(temp_path)
            assert loader.get('any_key') is None
        finally:
            os.unlink(temp_path)
    
    def test_nonexistent_yaml_path(self) -> None:
        """Load with nonexistent YAML path (graceful fallback)."""
        loader = ConfigLoader('/nonexistent/path.yaml')
        assert loader.config == {}
    
    def test_multiple_env_overrides(self) -> None:
        """Multiple environment variables all override correctly."""
        os.environ['SENTINEL_KEY1'] = 'value1'
        os.environ['SENTINEL_KEY2'] = '42'
        os.environ['SENTINEL_KEY3'] = 'true'
        
        loader = ConfigLoader()
        assert loader.get('key1') == 'value1'
        assert loader.get('key2') == 42
        assert loader.get('key3') is True
        
        del os.environ['SENTINEL_KEY1']
        del os.environ['SENTINEL_KEY2']
        del os.environ['SENTINEL_KEY3']
    
    def test_type_coercion_yes_no(self) -> None:
        """Type coercion treats 'yes'/'no' as
