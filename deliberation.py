"""
Real Deliberation Engine for Council v2.

WHAT'S DIFFERENT FROM V1:
------------------------
The original deliberation.py had hardcoded parsing that always returned defaults:

    def _parse_assignments(self, synthesis: str) -> dict:
        return {"Product Lead": "claude", ...}  # HARDCODED!

This version uses REAL structured outputs from the providers:

    result = await provider.complete_structured(messages, schema=RoleDeliberationResult)
    # Returns actual deliberation results, not hardcoded defaults!

HOW IT WORKS:
------------
1. GATHER PROPOSALS: Each model proposes via complete_structured(schema=Proposal)
2. CROSS-CRITIQUE: Each model critiques others via complete_structured(schema=Critique)
3. SYNTHESIZE: Chairman synthesizes via complete_structured(schema=RoleDeliberationResult)

The result is validated Pydantic models, not text parsing.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Type, TypeVar, Optional

from pydantic import BaseModel

from schemas import (
    Proposal,
    Critique,
    RoleAssignment,
    RoleDeliberationResult,
    Plan,
    PlanStep,
    CodeReview,
    ReviewIssue,
    AgentResponse,
    DeliberationPhase,
    DeliberationTranscript,
)
from providers import ModelProvider, get_provider

T = TypeVar('T', bound=BaseModel)


@dataclass
class Agent:
    """
    Represents an AI agent in the council.

    Each agent has a name, a provider (Claude, GPT, DeepSeek), and a role.
    """
    name: str
    provider: ModelProvider
    model_name: str
    role: str = ""

    # Cost tracking per 1K tokens
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


@dataclass
class DeliberationConfig:
    """Configuration for a deliberation session."""
    chairman_model: str = "claude"  # Model for synthesis
    proposal_models: list[str] = field(default_factory=lambda: ["claude", "gpt4"])
    critique_models: list[str] = field(default_factory=lambda: ["claude", "gpt4"])
    max_critique_rounds: int = 1
    temperature: float = 0.3  # Lower for more consistent structured output


class Deliberation:
    """
    Real deliberation engine with structured outputs.

    This replaces the stub implementations in v1 with actual
    model responses validated via Pydantic schemas.
    """

    def __init__(self, config: dict, delib_config: Optional[DeliberationConfig] = None):
        """
        Initialize the deliberation engine.

        Args:
            config: Model configuration dict with provider settings
            delib_config: Deliberation-specific configuration
        """
        self.config = config
        self.delib_config = delib_config or DeliberationConfig()
        self._agents: dict[str, Agent] = {}
        self._transcript: Optional[DeliberationTranscript] = None

    def _get_agent(self, model_name: str) -> Agent:
        """Get or create an agent for the given model name."""
        if model_name not in self._agents:
            provider = get_provider(self.config, model_name)
            self._agents[model_name] = Agent(
                name=model_name,
                provider=provider,
                model_name=provider.model if hasattr(provider, 'model') else model_name,
                cost_per_1k_input=getattr(provider, 'cost_per_1k_input', 0.0),
                cost_per_1k_output=getattr(provider, 'cost_per_1k_output', 0.0),
            )
        return self._agents[model_name]

    async def _agent_respond_structured(
        self,
        agent: Agent,
        prompt: str,
        schema: Type[T],
        system: Optional[str] = None,
        context: str = ""
    ) -> tuple[T, AgentResponse]:
        """
        Get a STRUCTURED response from an agent.

        This is the core method that replaces text parsing with real structured outputs.

        Args:
            agent: The agent to respond
            prompt: The prompt/question for the agent
            schema: Pydantic model class for the expected response
            system: Optional system prompt
            context: Additional context to include

        Returns:
            tuple of (validated_response, agent_response_metadata)
        """
        # Build messages
        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\n{prompt}"

        messages = [{"role": "user", "content": full_prompt}]

        # Get structured response from provider
        start_time = time.time()
        validated_response, usage = await agent.provider.complete_structured(
            messages=messages,
            schema=schema,
            system=system,
            temperature=self.delib_config.temperature
        )
        duration = time.time() - start_time

        # Calculate cost
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        cost = (
            (input_tokens / 1000) * agent.cost_per_1k_input +
            (output_tokens / 1000) * agent.cost_per_1k_output
        )

        # Create response metadata
        agent_response = AgentResponse(
            agent_name=agent.name,
            model=agent.model_name,
            response_type=schema.__name__,
            tokens_used=total_tokens,
            cost=cost
        )

        return validated_response, agent_response

    # =========================================================================
    # PHASE 1: GATHER PROPOSALS
    # =========================================================================

    async def gather_proposals(
        self,
        task: str,
        context: str = ""
    ) -> tuple[list[tuple[str, Proposal]], list[AgentResponse]]:
        """
        Collect proposals from all configured models in parallel.

        Each model proposes an approach using complete_structured(schema=Proposal).

        Args:
            task: The task description to propose solutions for
            context: Additional context about the task

        Returns:
            tuple of (list of (model_name, Proposal), list of AgentResponse metadata)
        """
        system_prompt = """You are an expert software architect participating in a council deliberation.
Your task is to propose an approach for the given task.
Be specific about your approach, rationale, potential risks, and effort estimate.
Rate your confidence from 1 (uncertain) to 10 (very confident)."""

        prompt = f"""Task: {task}

Please propose your approach to this task. Consider:
1. What approach would you recommend?
2. Why is this the right approach?
3. What are the potential risks?
4. How much effort will this require (low/medium/high)?
5. How confident are you in this approach (1-10)?"""

        # Gather proposals in parallel
        async def get_proposal(model_name: str) -> tuple[str, Proposal, AgentResponse]:
            agent = self._get_agent(model_name)
            proposal, response_meta = await self._agent_respond_structured(
                agent=agent,
                prompt=prompt,
                schema=Proposal,
                system=system_prompt,
                context=context
            )
            return model_name, proposal, response_meta

        # Run all proposals in parallel
        tasks = [
            get_proposal(model_name)
            for model_name in self.delib_config.proposal_models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and collect results
        proposals = []
        responses = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Warning: Proposal failed: {result}")
                continue
            model_name, proposal, response_meta = result
            proposals.append((model_name, proposal))
            responses.append(response_meta)

        return proposals, responses

    # =========================================================================
    # PHASE 2: CROSS-CRITIQUE
    # =========================================================================

    async def cross_critique(
        self,
        proposals: list[tuple[str, Proposal]]
    ) -> tuple[list[tuple[str, Critique]], list[AgentResponse]]:
        """
        Each model critiques the proposals from other models.

        Args:
            proposals: List of (model_name, Proposal) tuples

        Returns:
            tuple of (list of (critic_model, Critique), list of AgentResponse metadata)
        """
        system_prompt = """You are an expert software architect reviewing proposals from other team members.
Provide constructive critique: acknowledge strengths, identify weaknesses, suggest improvements.
Be specific and actionable in your feedback."""

        # Format proposals for critique
        proposals_text = ""
        for i, (model_name, prop) in enumerate(proposals, 1):
            proposals_text += f"""
Proposal {i} (from {model_name}):
- Approach: {prop.approach}
- Rationale: {prop.rationale}
- Risks: {', '.join(prop.risks) if prop.risks else 'None identified'}
- Effort: {prop.effort}
- Confidence: {prop.confidence}/10
"""

        prompt = f"""Review the following proposals:

{proposals_text}

For each proposal, provide your critique:
1. What do you agree with?
2. What do you disagree with?
3. What suggestions do you have for improvement?
4. Overall recommendation: accept, modify, or reject?

Focus on the most significant proposal or the one you have the strongest opinion about."""

        # Gather critiques in parallel
        async def get_critique(model_name: str) -> tuple[str, Critique, AgentResponse]:
            agent = self._get_agent(model_name)
            critique, response_meta = await self._agent_respond_structured(
                agent=agent,
                prompt=prompt,
                schema=Critique,
                system=system_prompt
            )
            return model_name, critique, response_meta

        tasks = [
            get_critique(model_name)
            for model_name in self.delib_config.critique_models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        critiques = []
        responses = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Warning: Critique failed: {result}")
                continue
            model_name, critique, response_meta = result
            critiques.append((model_name, critique))
            responses.append(response_meta)

        return critiques, responses

    # =========================================================================
    # PHASE 3: SYNTHESIZE ASSIGNMENTS
    # =========================================================================

    async def synthesize_assignments(
        self,
        task: str,
        proposals: list[tuple[str, Proposal]],
        critiques: list[tuple[str, Critique]]
    ) -> tuple[RoleDeliberationResult, AgentResponse]:
        """
        Chairman synthesizes proposals and critiques into final role assignments.

        This replaces the hardcoded _parse_assignments() from v1.

        Args:
            task: Original task description
            proposals: All proposals from gather_proposals()
            critiques: All critiques from cross_critique()

        Returns:
            tuple of (RoleDeliberationResult, AgentResponse metadata)
        """
        system_prompt = """You are the chairman of a council of AI models deliberating on role assignments.
Based on the proposals and critiques, you must assign roles to the most appropriate models.

Available roles to assign:
- Product Lead: Owns the overall vision and priorities
- Architect: Designs the technical approach
- Backend Dev: Implements server-side logic
- Frontend Dev: Implements user interfaces
- QA/Reviewer: Tests and reviews code quality

Available models to assign:
- claude: Strong at reasoning, code review, architecture
- gpt4: Strong at creative solutions, broad knowledge
- deepseek: Strong at code implementation, fast iteration

Synthesize the discussion and make final assignments."""

        # Format the deliberation for synthesis
        proposals_text = ""
        for model_name, prop in proposals:
            proposals_text += f"""
{model_name}'s Proposal:
- Approach: {prop.approach}
- Rationale: {prop.rationale}
- Effort: {prop.effort}
- Confidence: {prop.confidence}/10
"""

        critiques_text = ""
        for model_name, crit in critiques:
            critiques_text += f"""
{model_name}'s Critique:
- Target: {crit.target_proposal}
- Agrees: {', '.join(crit.agrees) if crit.agrees else 'None'}
- Disagrees: {', '.join(crit.disagrees) if crit.disagrees else 'None'}
- Suggestion: {', '.join(crit.suggestions) if crit.suggestions else 'None'}
- Recommendation: {crit.recommendation}
"""

        prompt = f"""Task: {task}

== PROPOSALS ==
{proposals_text}

== CRITIQUES ==
{critiques_text}

Based on this deliberation:
1. Assign each role to the most appropriate model
2. Explain your reasoning for each assignment
3. Summarize what the council agreed on
4. Note any significant disagreements or alternative views

Make the final role assignments."""

        # Get synthesis from chairman
        chairman = self._get_agent(self.delib_config.chairman_model)
        result, response_meta = await self._agent_respond_structured(
            agent=chairman,
            prompt=prompt,
            schema=RoleDeliberationResult,
            system=system_prompt
        )

        return result, response_meta

    # =========================================================================
    # FULL DELIBERATION FLOWS
    # =========================================================================

    async def deliberate_roles(self, task: str, context: str = "") -> RoleDeliberationResult:
        """
        Full role deliberation flow: propose -> critique -> synthesize.

        This is the main entry point for role deliberation.

        Args:
            task: The task to deliberate roles for
            context: Additional context

        Returns:
            RoleDeliberationResult with actual assignments (not hardcoded!)
        """
        session_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        phases = []

        # Phase 1: Gather proposals
        phase_start = time.time()
        proposals, proposal_responses = await self.gather_proposals(task, context)
        phases.append(DeliberationPhase(
            phase_name="proposals",
            responses=proposal_responses,
            duration_seconds=time.time() - phase_start
        ))

        # Phase 2: Cross-critique
        phase_start = time.time()
        critiques, critique_responses = await self.cross_critique(proposals)
        phases.append(DeliberationPhase(
            phase_name="critiques",
            responses=critique_responses,
            duration_seconds=time.time() - phase_start
        ))

        # Phase 3: Synthesize
        phase_start = time.time()
        result, synthesis_response = await self.synthesize_assignments(task, proposals, critiques)
        phases.append(DeliberationPhase(
            phase_name="synthesis",
            responses=[synthesis_response],
            duration_seconds=time.time() - phase_start
        ))

        # Build transcript
        total_tokens = sum(r.tokens_used for phase in phases for r in phase.responses)
        total_cost = sum(r.cost for phase in phases for r in phase.responses)

        self._transcript = DeliberationTranscript(
            session_id=session_id,
            task=task,
            phases=phases,
            total_tokens=total_tokens,
            total_cost=total_cost,
            duration_seconds=time.time() - start_time
        )

        return result

    async def deliberate_plan(
        self,
        task: str,
        context: str = "",
        role_assignments: Optional[RoleDeliberationResult] = None
    ) -> Plan:
        """
        Create an implementation plan through deliberation.

        This replaces the stub _parse_plan() from v1.

        Args:
            task: The task to plan
            context: Additional context
            role_assignments: Optional existing role assignments

        Returns:
            Plan with structured steps
        """
        system_prompt = """You are a senior software architect creating an implementation plan.
Break down the task into clear, actionable steps.
Each step should have:
- A clear title and description
- An assigned agent (Backend Dev, Frontend Dev, Architect, etc.)
- Dependencies on other steps
- Complexity estimate (low/medium/high)
- Acceptance criteria

Create a comprehensive but realistic plan."""

        # Build context about role assignments if available
        roles_context = ""
        if role_assignments:
            roles_context = "\nAvailable team members:\n"
            for assignment in role_assignments.assignments:
                roles_context += f"- {assignment.role}: {assignment.assigned_to}\n"

        prompt = f"""Task: {task}
{roles_context}
{f"Additional Context: {context}" if context else ""}

Create a detailed implementation plan:
1. Provide a high-level overview
2. Break down into numbered steps
3. Identify the critical path (blocking steps)
4. Estimate total effort
5. Identify risks and mitigations"""

        # Get plan from architect (or chairman)
        architect = self._get_agent(self.delib_config.chairman_model)
        plan, response_meta = await self._agent_respond_structured(
            agent=architect,
            prompt=prompt,
            schema=Plan,
            system=system_prompt
        )

        return plan

    async def quality_review(
        self,
        code: str,
        context: str = "",
        reviewer_model: Optional[str] = None
    ) -> CodeReview:
        """
        Perform a structured code review.

        This replaces the keyword-matching _extract_issues() from v1.

        Args:
            code: The code to review
            context: Context about the code (purpose, requirements)
            reviewer_model: Which model to use for review (default: chairman)

        Returns:
            CodeReview with structured issues
        """
        system_prompt = """You are an expert code reviewer performing a thorough review.
Look for issues in these categories:
- Security: SQL injection, XSS, authentication issues, etc.
- Performance: N+1 queries, memory leaks, inefficient algorithms
- Correctness: Logic errors, edge cases, error handling
- Maintainability: Code clarity, documentation, modularity
- Style: Naming conventions, formatting, best practices

Be specific about locations and provide actionable fix suggestions.
Only flag real issues - don't be overly pedantic."""

        prompt = f"""Please review this code:

```
{code}
```

{f"Context: {context}" if context else ""}

Provide a thorough review:
1. Identify any issues (security, performance, correctness, maintainability, style)
2. For each issue, specify severity and provide a suggested fix
3. Note what was done well
4. Give an overall approval recommendation"""

        model = reviewer_model or self.delib_config.chairman_model
        reviewer = self._get_agent(model)

        review, response_meta = await self._agent_respond_structured(
            agent=reviewer,
            prompt=prompt,
            schema=CodeReview,
            system=system_prompt
        )

        return review

    def get_transcript(self) -> Optional[DeliberationTranscript]:
        """Get the transcript from the last deliberation session."""
        return self._transcript


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_default_config() -> dict:
    """
    Create a default configuration for testing.

    This mirrors the structure expected by get_provider().
    """
    return {
        "models": {
            "claude": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "api_key_env": "ANTHROPIC_API_KEY"
            },
            "gpt4": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key_env": "OPENAI_API_KEY"
            },
            "deepseek": {
                "provider": "ollama",
                "model": "deepseek-coder-v2:16b",
                "base_url": "http://localhost:11434"
            }
        }
    }


async def quick_deliberate_roles(task: str) -> RoleDeliberationResult:
    """
    Quick way to run role deliberation with defaults.

    Example:
        result = await quick_deliberate_roles("Build a REST API for user management")
        for assignment in result.assignments:
            print(f"{assignment.role}: {assignment.assigned_to}")
    """
    config = create_default_config()
    delib_config = DeliberationConfig(
        chairman_model="claude",
        proposal_models=["claude", "gpt4"],
        critique_models=["claude", "gpt4"]
    )

    engine = Deliberation(config, delib_config)
    return await engine.deliberate_roles(task)


async def quick_review(code: str) -> CodeReview:
    """
    Quick way to run code review with defaults.

    Example:
        review = await quick_review(my_code)
        for issue in review.issues:
            print(f"[{issue.severity}] {issue.description}")
    """
    config = create_default_config()
    engine = Deliberation(config)
    return await engine.quality_review(code)
