# Multi-Model Council

A multi-model AI deliberation system that enables structured collaboration between different LLMs (Claude, GPT-4, DeepSeek) to solve complex problems through proposals, critiques, and consensus building.

## Features

- **Multi-Model Deliberation**: Orchestrate discussions between Claude, GPT-4, and DeepSeek
- **Role-Based Architecture**: Assign models to roles (Product Lead, Architect, Backend Dev, etc.)
- **Structured Workflows**: Proposal → Critique → Synthesis pipeline
- **Code Review**: Automated security and quality analysis
- **Planning**: Generate structured implementation plans with dependencies
- **Token Tracking**: Monitor usage and costs across all providers

## Architecture

### Phase 1: Providers (`providers.py`)
Unified interface for multiple LLM providers with streaming support, token tracking, and cost calculation.

### Phase 2: Deliberation (`deliberation.py`)
Core deliberation engine with proposal collection, cross-critique, role assignment, and consensus synthesis.

### Phase 3: Execution (`execution.py`)
Execution layer for running generated plans with checkpointing, rollback, and multi-model task execution.

## Requirements

```bash
pip install anthropic openai pydantic
```

## Environment Variables

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"      # Optional, for GPT-4
export DEEPSEEK_API_KEY="your-key"    # Optional, for DeepSeek
```

## Quick Start

```python
from deliberation import DeliberationEngine
from providers import ProviderManager

# Initialize
manager = ProviderManager()
engine = DeliberationEngine(manager)

# Run a deliberation
result = engine.deliberate_roles(
    task="Design a REST API for user management",
    context={"requirements": ["authentication", "CRUD operations"]}
)

print(result.consensus)
print(result.role_assignments)
```

## Running Tests

```bash
# Phase 2 tests (deliberation)
python test_phase2.py

# Phase 3 tests (execution)
python test_phase3.py
```

## License

MIT
