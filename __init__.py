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
Phase 3: Execution Layer (planned)
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
]
