"""
Cost tracking for Agent Council.
Tracks token usage and estimates costs across providers.
"""

from dataclasses import dataclass, field
from typing import Optional


# Cost per 1K tokens (as of Jan 2025 - update as needed)
COST_TABLE = {
    "claude": {
        "input": 0.003,   # Claude Sonnet
        "output": 0.015
    },
    "codex": {
        "input": 0.005,   # GPT-4o
        "output": 0.015
    },
    "deepseek": {
        "input": 0.0,     # Local = free
        "output": 0.0
    }
}


@dataclass
class PhaseUsage:
    """Token usage for a single phase."""
    input_tokens: int = 0
    output_tokens: int = 0
    model_breakdown: dict = field(default_factory=dict)
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CostTracker:
    """Tracks costs across an entire council session."""
    
    def __init__(self):
        self.phases: dict[str, PhaseUsage] = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens
        
    def add(
        self, 
        phase: str, 
        model: str, 
        input_tokens: int, 
        output_tokens: int
    ):
        """Add token usage for a phase."""
        if phase not in self.phases:
            self.phases[phase] = PhaseUsage()
            
        self.phases[phase].input_tokens += input_tokens
        self.phases[phase].output_tokens += output_tokens
        
        if model not in self.phases[phase].model_breakdown:
            self.phases[phase].model_breakdown[model] = {
                "input": 0, "output": 0
            }
        self.phases[phase].model_breakdown[model]["input"] += input_tokens
        self.phases[phase].model_breakdown[model]["output"] += output_tokens
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
    def get_phase_cost(self, phase: str) -> float:
        """Get cost for a specific phase."""
        if phase not in self.phases:
            return 0.0
            
        total = 0.0
        for model, usage in self.phases[phase].model_breakdown.items():
            costs = COST_TABLE.get(model, {"input": 0.01, "output": 0.03})
            total += (usage["input"] / 1000) * costs["input"]
            total += (usage["output"] / 1000) * costs["output"]
            
        return total
        
    def get_total_cost(self) -> float:
        """Get total cost across all phases."""
        return sum(self.get_phase_cost(phase) for phase in self.phases)
        
    def get_breakdown(self) -> dict:
        """Get detailed cost breakdown by phase."""
        breakdown = {}
        for phase, usage in self.phases.items():
            breakdown[phase] = {
                "tokens": usage.total_tokens,
                "cost": self.get_phase_cost(phase),
                "models": usage.model_breakdown
            }
        return breakdown
        
    def estimate_build_cost(self, plan: dict) -> float:
        """Estimate the cost to execute a build plan."""
        # Rough estimation based on task complexity
        num_agents = len(plan.get("agent_tasks", {}))
        
        # Assume ~2000 tokens input, ~4000 tokens output per agent task
        estimated_input = num_agents * 2000
        estimated_output = num_agents * 4000
        
        # Distribute across models based on typical assignments
        claude_ratio = 0.4  # Planning + quality
        deepseek_ratio = 0.3  # Backend + DB
        codex_ratio = 0.3  # Frontend + tests
        
        cost = 0.0
        
        # Claude cost
        cost += (estimated_input * claude_ratio / 1000) * COST_TABLE["claude"]["input"]
        cost += (estimated_output * claude_ratio / 1000) * COST_TABLE["claude"]["output"]
        
        # Codex cost
        cost += (estimated_input * codex_ratio / 1000) * COST_TABLE["codex"]["input"]
        cost += (estimated_output * codex_ratio / 1000) * COST_TABLE["codex"]["output"]
        
        # DeepSeek is free (local)
        
        return cost
        
    def format_summary(self) -> str:
        """Get a formatted cost summary."""
        lines = ["Cost Summary", "=" * 40]
        
        for phase, data in self.get_breakdown().items():
            lines.append(f"\n{phase}:")
            lines.append(f"  Tokens: {data['tokens']:,}")
            lines.append(f"  Cost: ${data['cost']:.4f}")
            
        lines.append(f"\n{'=' * 40}")
        lines.append(f"TOTAL: {self.total_tokens:,} tokens, ${self.get_total_cost():.4f}")
        
        return "\n".join(lines)
