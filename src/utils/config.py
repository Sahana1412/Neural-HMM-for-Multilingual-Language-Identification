"""YAML configuration loader with merging support."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass

import yaml


logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration object with attribute access."""

    _data: Dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        """Access config values as attributes."""
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            value = self._data[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        """Access config values as dict items."""
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._data.copy()


def load_config(config_path: str | Path) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with loaded parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading config from {config_path}")
    
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML in {config_path}: {e}")
    
    # Handle defaults
    if "defaults_config" in data:
        defaults_path = Path(data["defaults_config"])
        logger.info(f"Loading defaults from {defaults_path}")
        defaults_data = load_config(defaults_path).to_dict()
        # Merge: current config overrides defaults
        data = merge_configs(defaults_data, data)
    
    return Config(data)


def merge_configs(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge override config into defaults.
    
    Args:
        defaults: Default configuration
        overrides: Override configuration
        
    Returns:
        Merged configuration (overrides take precedence)
    """
    result = defaults.copy()
    
    for key, value in overrides.items():
        if key == "defaults_config":
            # Skip the defaults pointer
            continue
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Config | Dict[str, Any], output_path: str | Path) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        output_path: Path to save to
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = config.to_dict() if isinstance(config, Config) else config
    
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved config to {output_path}")
