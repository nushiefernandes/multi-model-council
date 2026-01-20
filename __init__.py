"""
Council v2 - Multi-model deliberation with structured outputs.

This is a rewrite of the council system focusing on:
1. Real structured outputs (not text parsing)
2. Proper Pydantic validation
3. Incremental, tested development

PHASES:
------
Phase 1: Schemas and Providers (complete)
Phase 2: Real Deliberation Engine (complete)
Phase 3: Execution Layer (complete)
Phase 4: CLI Interface (complete)
Phase 5: Workspace & Config (complete)
"""

__version__ = "2.0.0-alpha"

# Re-export key classes for convenience
from .schemas import (
    Proposal,
    Critique,
    RoleAssignment,
    RoleDeliberationResult,
    Plan,
    PlanStep,
    CodeReview,
    ReviewIssue,
    DeliberationTranscript,
    ConsensusResult,
    ExecutionResult,
)

from .providers import (
    ModelProvider,
    AnthropicProvider,
    OpenAIProvider,
    OllamaProvider,
    get_provider,
)

from .deliberation import (
    Deliberation,
    DeliberationConfig,
    Agent,
    create_default_config,
    quick_deliberate_roles,
    quick_review,
    quick_ask,
)

from .execution import (
    ExecutionEngine,
    ToolRegistry,
    execute_plan_interactive,
    execute_plan_headless,
)

from .session import (
    Session,
    SessionManager,
    WorkflowPhase,
    create_session,
    load_session,
    save_session,
    list_sessions,
)

from .cli import main as cli_main

from .config import (
    Config,
    ModelConfig,
    ChairmanConfig,
    WorkspaceConfig,
    load_config,
    get_default_config,
    save_config,
)

from .workspace import (
    WorkspaceManager,
    create_workspace,
)

__all__ = [
    # Schemas
    "Proposal",
    "Critique",
    "RoleAssignment",
    "RoleDeliberationResult",
    "Plan",
    "PlanStep",
    "CodeReview",
    "ReviewIssue",
    "DeliberationTranscript",
    "ConsensusResult",
    "ExecutionResult",
    # Providers
    "ModelProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "get_provider",
    # Deliberation
    "Deliberation",
    "DeliberationConfig",
    "Agent",
    "create_default_config",
    "quick_deliberate_roles",
    "quick_review",
    "quick_ask",
    # Execution
    "ExecutionEngine",
    "ToolRegistry",
    "execute_plan_interactive",
    "execute_plan_headless",
    # Session
    "Session",
    "SessionManager",
    "WorkflowPhase",
    "create_session",
    "load_session",
    "save_session",
    "list_sessions",
    # CLI
    "cli_main",
    # Config
    "Config",
    "ModelConfig",
    "ChairmanConfig",
    "WorkspaceConfig",
    "load_config",
    "get_default_config",
    "save_config",
    # Workspace
    "WorkspaceManager",
    "create_workspace",
]
