"""
Configuration Management for Council v2 - Phase 5.

WHAT THIS FILE DOES:
-------------------
Loads and validates configuration from YAML files with sensible defaults.
Provides a clean interface for accessing model configurations, workspace
settings, and other options.

CONFIG FILE LOCATION:
--------------------
Default: ~/.council/config.yaml

CONFIG FORMAT:
-------------
```yaml
models:
  claude:
    provider: "anthropic"
    model: "claude-sonnet-4-20250514"
    api_key_env: "ANTHROPIC_API_KEY"
  gpt4:
    provider: "openai"
    model: "gpt-5.2"
    api_key_env: "OPENAI_API_KEY"

chairman:
  model: "claude"

workspaces:
  base: "./council-output"
```
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# =============================================================================
# CONFIGURATION DATA CLASSES
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    provider: str
    model: str
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None

    def get_api_key(self) -> Optional[str]:
        """Get API key from environment variable."""
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for provider compatibility."""
        result = {
            "provider": self.provider,
            "model": self.model,
        }
        if self.api_key_env:
            result["api_key_env"] = self.api_key_env
        if self.base_url:
            result["base_url"] = self.base_url
        return result


@dataclass
class ChairmanConfig:
    """Configuration for the chairman model."""
    model: str = "claude"
    memory_file: Optional[str] = None


@dataclass
class WorkspaceConfig:
    """Configuration for workspace/output settings."""
    base: str = "./council-output"
    per_agent: bool = False

    @property
    def base_path(self) -> Path:
        """Get base path, expanding ~ if present."""
        return Path(self.base).expanduser()


@dataclass
class AlertConfig:
    """Configuration for alerts and notifications."""
    terminal: bool = True
    macos_notification: bool = False


@dataclass
class TranscriptConfig:
    """Configuration for transcript handling."""
    show: str = "summary"  # "summary", "full", "none"
    save: str = "full"     # "full", "summary", "none"
    directory: str = "~/.council/transcripts"

    @property
    def directory_path(self) -> Path:
        """Get directory path, expanding ~ if present."""
        return Path(self.directory).expanduser()


@dataclass
class Config:
    """
    Complete configuration for Council v2.

    This is the main configuration object that holds all settings.
    It can be loaded from a YAML file or created with defaults.
    """
    models: dict[str, ModelConfig] = field(default_factory=dict)
    chairman: ChairmanConfig = field(default_factory=ChairmanConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    transcripts: TranscriptConfig = field(default_factory=TranscriptConfig)

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get a model configuration by name."""
        return self.models.get(name)

    def get_model_dict(self, name: str) -> Optional[dict]:
        """Get a model configuration as a dictionary."""
        model = self.models.get(name)
        return model.to_dict() if model else None

    def to_provider_config(self) -> dict:
        """
        Convert to the dict format expected by get_provider().

        This maintains compatibility with the existing provider system.
        """
        return {
            "models": {
                name: model.to_dict()
                for name, model in self.models.items()
            }
        }

    def list_models(self) -> list[str]:
        """List all configured model names."""
        return list(self.models.keys())


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

def get_default_config() -> Config:
    """
    Get the default configuration.

    Returns a Config with sensible defaults that work out of the box
    (assuming API keys are set in environment).
    """
    return Config(
        models={
            "claude": ModelConfig(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                api_key_env="ANTHROPIC_API_KEY"
            ),
            "gpt4": ModelConfig(
                provider="openai",
                model="gpt-5.2",
                api_key_env="OPENAI_API_KEY"
            ),
            "deepseek": ModelConfig(
                provider="ollama",
                model="deepseek-coder-v2:16b",
                base_url="http://localhost:11434"
            ),
        },
        chairman=ChairmanConfig(model="claude"),
        workspace=WorkspaceConfig(base="./council-output"),
        alerts=AlertConfig(terminal=True, macos_notification=False),
        transcripts=TranscriptConfig(
            show="summary",
            save="full",
            directory="~/.council/transcripts"
        ),
    )


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def _parse_model_config(data: dict) -> ModelConfig:
    """Parse a model configuration from dict."""
    return ModelConfig(
        provider=data.get("provider", "anthropic"),
        model=data.get("model", "claude-sonnet-4-20250514"),
        api_key_env=data.get("api_key_env"),
        base_url=data.get("base_url"),
    )


def _parse_config(data: dict) -> Config:
    """Parse a complete configuration from dict."""
    config = get_default_config()

    # Parse models
    if "models" in data:
        config.models = {}
        for name, model_data in data["models"].items():
            config.models[name] = _parse_model_config(model_data)

    # Parse chairman
    if "chairman" in data:
        chairman_data = data["chairman"]
        config.chairman = ChairmanConfig(
            model=chairman_data.get("model", "claude"),
            memory_file=chairman_data.get("memory_file"),
        )

    # Parse workspace (handle both "workspace" and "workspaces" keys)
    workspace_data = data.get("workspace") or data.get("workspaces", {})
    if workspace_data:
        config.workspace = WorkspaceConfig(
            base=workspace_data.get("base", "./council-output"),
            per_agent=workspace_data.get("per_agent", False),
        )

    # Parse alerts
    if "alerts" in data:
        alerts_data = data["alerts"]
        config.alerts = AlertConfig(
            terminal=alerts_data.get("terminal", True),
            macos_notification=alerts_data.get("macos_notification", False),
        )

    # Parse transcripts
    if "transcripts" in data:
        transcripts_data = data["transcripts"]
        config.transcripts = TranscriptConfig(
            show=transcripts_data.get("show", "summary"),
            save=transcripts_data.get("save", "full"),
            directory=transcripts_data.get("directory", "~/.council/transcripts"),
        )

    return config


def load_config(path: Optional[Path] = None) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to config file. If None, tries default locations:
              1. ~/.council/config.yaml
              2. ./council.yaml
              3. Falls back to defaults

    Returns:
        Loaded configuration (or defaults if file not found)
    """
    # Try provided path
    if path:
        path = Path(path).expanduser()
        if path.exists():
            return load_config_from_file(path)
        raise FileNotFoundError(f"Config file not found: {path}")

    # Try default locations
    default_paths = [
        Path.home() / ".council" / "config.yaml",
        Path("./council.yaml"),
        Path("./council.yml"),
    ]

    for default_path in default_paths:
        if default_path.exists():
            return load_config_from_file(default_path)

    # Return defaults
    return get_default_config()


def load_config_from_file(path: Path) -> Config:
    """
    Load configuration from a specific file.

    Args:
        path: Path to the YAML config file

    Returns:
        Parsed configuration

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    return _parse_config(data)


def save_config(config: Config, path: Path) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration to save
        path: Output path
    """
    data = {
        "models": {
            name: model.to_dict()
            for name, model in config.models.items()
        },
        "chairman": {
            "model": config.chairman.model,
        },
        "workspaces": {
            "base": config.workspace.base,
            "per_agent": config.workspace.per_agent,
        },
        "alerts": {
            "terminal": config.alerts.terminal,
            "macos_notification": config.alerts.macos_notification,
        },
        "transcripts": {
            "show": config.transcripts.show,
            "save": config.transcripts.save,
            "directory": config.transcripts.directory,
        },
    }

    if config.chairman.memory_file:
        data["chairman"]["memory_file"] = config.chairman.memory_file

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_config_path() -> Optional[Path]:
    """
    Get the path to the active config file, if any exists.

    Returns:
        Path to config file or None if using defaults
    """
    default_paths = [
        Path.home() / ".council" / "config.yaml",
        Path("./council.yaml"),
        Path("./council.yml"),
    ]

    for path in default_paths:
        if path.exists():
            return path

    return None


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing config module...")

    # Test default config
    config = get_default_config()
    print(f"\nDefault models: {config.list_models()}")
    print(f"Chairman model: {config.chairman.model}")
    print(f"Workspace base: {config.workspace.base}")

    # Test loading from default location
    try:
        loaded = load_config()
        print(f"\nLoaded config models: {loaded.list_models()}")
        print(f"Provider config: {loaded.to_provider_config()}")
    except FileNotFoundError as e:
        print(f"\nNo config file found: {e}")
        print("Using defaults.")

    # Test model access
    claude = config.get_model("claude")
    if claude:
        print(f"\nClaude config:")
        print(f"  Provider: {claude.provider}")
        print(f"  Model: {claude.model}")
        print(f"  API key env: {claude.api_key_env}")
