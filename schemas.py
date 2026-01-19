"""
Pydantic schemas for structured outputs in Council v2.

WHY THIS FILE EXISTS:
--------------------
The original council used text parsing (regex, keyword matching) to extract
structured data from model responses. This was brittle and often failed.

Example of the old approach (from deliberation.py):
    def _parse_assignments(self, synthesis: str) -> dict:
        # Simplified parsing - in production would use structured output
        return default_assignments  # <- Just returned hardcoded values!

This file defines Pydantic models that:
1. Tell the model EXACTLY what structure we expect (via JSON schema)
2. Validate the response automatically
3. Give us typed Python objects to work with

HOW PYDANTIC WORKS:
------------------
Pydantic is a data validation library. When you define a model like:

    class Proposal(BaseModel):
        approach: str
        effort: Literal["low", "medium", "high"]

It automatically:
- Validates that 'approach' is a string
- Validates that 'effort' is one of the allowed values
- Raises ValidationError if data doesn't match
- Exports to JSON Schema (for API calls)
"""

from typing import Literal, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# DELIBERATION SCHEMAS
# =============================================================================
# These schemas are used when models discuss and debate approaches.

class Proposal(BaseModel):
    """
    A model's proposed approach to a task.

    Used in: Role deliberation, planning phase

    Example:
        User asks: "How should we build user authentication?"
        Model returns: Proposal(
            approach="Use JWT tokens with refresh rotation",
            rationale="JWTs are stateless, scale well, industry standard",
            risks=["Token theft if not using HTTPS", "Complexity of refresh logic"],
            effort="medium",
            confidence=8
        )
    """
    approach: str = Field(
        description="The proposed approach or solution"
    )
    rationale: str = Field(
        description="Why this approach is recommended"
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Potential risks or downsides"
    )
    effort: Literal["low", "medium", "high"] = Field(
        description="Estimated implementation effort"
    )
    confidence: int = Field(
        ge=1, le=10,  # ge=greater or equal, le=less or equal
        description="Confidence level from 1 (uncertain) to 10 (very confident)"
    )

    @field_validator('risks')
    @classmethod
    def risks_not_empty_strings(cls, v: list[str]) -> list[str]:
        """Ensure risk strings aren't empty."""
        return [risk for risk in v if risk.strip()]


class Critique(BaseModel):
    """
    A model's critique of another model's proposal.

    Used in: Cross-critique phase where models review each other's ideas

    This enables the "council" behavior - models don't just propose,
    they evaluate each other's proposals, surfacing tradeoffs.

    Example:
        After seeing a JWT proposal, another model returns:
        Critique(
            target_proposal="JWT with refresh rotation",
            agrees=["Stateless is good for scaling", "Industry standard"],
            disagrees=["Refresh rotation adds complexity we don't need yet"],
            suggestions=["Start with simple JWT, add refresh later if needed"],
            recommendation="modify"
        )
    """
    target_proposal: str = Field(
        description="Brief identifier of the proposal being critiqued"
    )
    agrees: list[str] = Field(
        default_factory=list,
        description="Points of agreement with the proposal"
    )
    disagrees: list[str] = Field(
        default_factory=list,
        description="Points of disagreement with the proposal"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested improvements or alternatives"
    )
    recommendation: Literal["accept", "modify", "reject"] = Field(
        description="Overall recommendation for this proposal"
    )


class RoleAssignment(BaseModel):
    """
    Assignment of a specific role to a model.

    Used in: Role deliberation output

    Example:
        RoleAssignment(
            role="Backend Dev",
            assigned_to="deepseek",
            reasoning="DeepSeek excels at algorithmic code and has lower latency for iterative work"
        )
    """
    role: str = Field(
        description="The role being assigned (e.g., 'Architect', 'Backend Dev')"
    )
    assigned_to: str = Field(
        description="The model assigned to this role (e.g., 'claude', 'deepseek', 'gpt4')"
    )
    reasoning: str = Field(
        description="Why this model was chosen for this role"
    )


class RoleDeliberationResult(BaseModel):
    """
    Complete result of role deliberation phase.

    This replaces the hardcoded _parse_assignments() in the original code.
    Instead of always returning the same defaults, we get actual deliberation results.
    """
    assignments: list[RoleAssignment] = Field(
        description="List of role assignments decided by the council"
    )
    consensus_notes: str = Field(
        description="Summary of what the council agreed on"
    )
    dissenting_views: list[str] = Field(
        default_factory=list,
        description="Any disagreements or alternative views that were raised"
    )


# =============================================================================
# PLANNING SCHEMAS
# =============================================================================
# These schemas are used when creating implementation plans.

class PlanStep(BaseModel):
    """
    A single step in an implementation plan.

    Example:
        PlanStep(
            step_number=1,
            title="Set up database schema",
            description="Create PostgreSQL tables for users, sessions, tokens",
            assigned_agent="DB Specialist",
            dependencies=[],
            estimated_complexity="medium",
            acceptance_criteria=["Tables created", "Migrations written", "Indexes added"]
        )
    """
    step_number: int = Field(
        ge=1,
        description="Step number (1-indexed)"
    )
    title: str = Field(
        description="Brief title for this step"
    )
    description: str = Field(
        description="Detailed description of what this step involves"
    )
    assigned_agent: str = Field(
        description="Which agent should handle this step"
    )
    dependencies: list[int] = Field(
        default_factory=list,
        description="Step numbers that must complete before this one"
    )
    estimated_complexity: Literal["low", "medium", "high"] = Field(
        description="How complex this step is"
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="How we know this step is done"
    )


class Plan(BaseModel):
    """
    Complete implementation plan from the planning phase.

    This replaces the stub _parse_plan() that just truncated text.
    """
    overview: str = Field(
        description="High-level summary of the plan"
    )
    steps: list[PlanStep] = Field(
        description="Ordered list of implementation steps"
    )
    critical_path: list[int] = Field(
        default_factory=list,
        description="Step numbers that are on the critical path (blocking)"
    )
    estimated_total_effort: Literal["low", "medium", "high"] = Field(
        description="Overall effort estimate"
    )
    risks_and_mitigations: dict[str, str] = Field(
        default_factory=dict,
        description="Identified risks and how to mitigate them"
    )

    @property
    def total_steps(self) -> int:
        """Convenience property for step count."""
        return len(self.steps)


# =============================================================================
# REVIEW SCHEMAS
# =============================================================================
# These schemas are used in the quality review phase.

class ReviewIssue(BaseModel):
    """
    A single issue found during code review.

    This replaces _extract_issues() which just searched for keywords like "critical".

    Example:
        ReviewIssue(
            severity="critical",
            category="security",
            location="auth.py:45",
            description="SQL query built with string concatenation, vulnerable to injection",
            suggested_fix="Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
            effort_to_fix="low"
        )
    """
    severity: Literal["critical", "major", "minor", "suggestion"] = Field(
        description="How serious is this issue"
    )
    category: Literal["security", "performance", "correctness", "maintainability", "style"] = Field(
        description="What type of issue this is"
    )
    location: str = Field(
        description="Where the issue is (file:line or file:function)"
    )
    description: str = Field(
        description="What the issue is"
    )
    suggested_fix: str = Field(
        description="How to fix it"
    )
    effort_to_fix: Literal["low", "medium", "high"] = Field(
        description="How much effort to fix"
    )


class CodeReview(BaseModel):
    """
    Complete code review result from a reviewer agent.
    """
    reviewer: str = Field(
        description="Who performed this review"
    )
    summary: str = Field(
        description="Overall summary of the review"
    )
    issues: list[ReviewIssue] = Field(
        default_factory=list,
        description="List of issues found"
    )
    approved: bool = Field(
        description="Whether the code is approved (no critical/major issues)"
    )
    positive_notes: list[str] = Field(
        default_factory=list,
        description="Things that were done well"
    )

    @property
    def blocking_issues(self) -> int:
        """Count of issues that block approval."""
        return sum(1 for i in self.issues if i.severity in ["critical", "major"])

    @property
    def by_severity(self) -> dict[str, list[ReviewIssue]]:
        """Group issues by severity."""
        result: dict[str, list[ReviewIssue]] = {
            "critical": [], "major": [], "minor": [], "suggestion": []
        }
        for issue in self.issues:
            result[issue.severity].append(issue)
        return result


# =============================================================================
# TOOL SCHEMAS (for Phase 3)
# =============================================================================
# These will be used when agents can execute tools. Defined now for completeness.

# =============================================================================
# DELIBERATION SESSION SCHEMAS
# =============================================================================
# These schemas track deliberation sessions and agent responses.

class AgentResponse(BaseModel):
    """
    Structured response from an agent during deliberation.

    Used to track what each agent contributed and the associated costs.
    """
    agent_name: str = Field(
        description="Name of the agent that responded"
    )
    model: str = Field(
        description="The model used by this agent"
    )
    response_type: str = Field(
        description="Type of response (proposal, critique, synthesis, etc.)"
    )
    tokens_used: int = Field(
        default=0,
        description="Total tokens used for this response"
    )
    cost: float = Field(
        default=0.0,
        description="Estimated cost for this response"
    )


class DeliberationPhase(BaseModel):
    """A single phase of deliberation."""
    phase_name: str = Field(
        description="Name of the phase (proposals, critiques, synthesis)"
    )
    responses: list[AgentResponse] = Field(
        default_factory=list,
        description="Responses from each agent in this phase"
    )
    duration_seconds: float = Field(
        default=0.0,
        description="How long this phase took"
    )


class DeliberationTranscript(BaseModel):
    """
    Record of a complete deliberation session.

    Tracks the entire deliberation flow including all phases,
    costs, and timing information for analysis and debugging.
    """
    session_id: str = Field(
        description="Unique identifier for this session"
    )
    task: str = Field(
        description="The task that was deliberated"
    )
    phases: list[DeliberationPhase] = Field(
        default_factory=list,
        description="Each phase of the deliberation"
    )
    total_tokens: int = Field(
        default=0,
        description="Total tokens used across all phases"
    )
    total_cost: float = Field(
        default=0.0,
        description="Total cost across all phases"
    )
    duration_seconds: float = Field(
        default=0.0,
        description="Total duration of the deliberation"
    )


# =============================================================================
# TOOL SCHEMAS (Phase 3 - Execution Layer)
# =============================================================================
# These schemas define tools and their execution.

class ToolParameter(BaseModel):
    """Schema for a single tool parameter."""
    name: str = Field(description="Parameter name")
    type: Literal["string", "integer", "boolean", "object", "array"] = Field(
        description="Parameter type"
    )
    description: str = Field(description="What this parameter does")
    required: bool = Field(default=True, description="Whether this parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value if not required")


class ToolDefinition(BaseModel):
    """
    Definition of an available tool.

    This describes a tool that agents can use during execution.
    The ToolRegistry uses these definitions to validate calls.
    """
    name: str = Field(description="Tool name, e.g., 'file_write'")
    description: str = Field(description="Description shown to agents")
    parameters: list[ToolParameter] = Field(
        default_factory=list,
        description="Parameters the tool accepts"
    )
    returns: str = Field(description="What the tool returns")
    dangerous: bool = Field(
        default=False,
        description="Whether this tool requires user approval"
    )


class ToolCall(BaseModel):
    """
    A structured request to execute a tool.

    Used in: Execution phase when agents want to call tools.
    """
    tool_name: str = Field(description="Name of the tool to call")
    parameters: dict = Field(
        default_factory=dict,
        description="Parameters to pass to the tool"
    )
    reasoning: str = Field(description="Why this tool call is needed")


class ToolCallRequest(BaseModel):
    """
    Tool call with full execution metadata.

    Extends ToolCall with tracking information for the execution transcript.
    """
    call_id: str = Field(description="Unique identifier for this call")
    tool_name: str = Field(description="Name of the tool to call")
    parameters: dict = Field(
        default_factory=dict,
        description="Parameters passed to the tool"
    )
    reasoning: str = Field(description="Why this tool call is needed")
    agent_name: str = Field(description="Agent that made this call")
    step_number: int = Field(description="Which plan step this belongs to")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this call was made"
    )


class ToolCallResult(BaseModel):
    """
    Result from executing a tool with metadata.
    """
    call_id: str = Field(description="ID of the original call")
    tool_name: str = Field(description="Name of the tool that was called")
    success: bool = Field(description="Whether the call succeeded")
    output: str = Field(description="Output from the tool")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    duration_ms: int = Field(default=0, description="Execution duration in milliseconds")


class ToolResult(BaseModel):
    """
    Simple result from executing a tool (legacy compatibility).
    """
    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None


# =============================================================================
# EXECUTION CONTEXT SCHEMAS
# =============================================================================
# These track state during plan execution.

class ExecutionContext(BaseModel):
    """
    State maintained during plan execution.

    This is passed to each step so it can access previous outputs,
    created files, and other shared state.
    """
    session_id: str = Field(description="Unique session identifier")
    workspace_path: str = Field(description="Path to the workspace directory")
    completed_steps: list[int] = Field(
        default_factory=list,
        description="Step numbers that have been completed"
    )
    step_outputs: dict[int, str] = Field(
        default_factory=dict,
        description="Outputs from each step (step_number -> output)"
    )
    created_files: list[str] = Field(
        default_factory=list,
        description="Files created during execution"
    )
    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Named results that can be referenced by later steps"
    )


class AgentToolCalls(BaseModel):
    """
    Structured response when an agent generates tool calls.

    This is what agents return when asked to execute a step.
    """
    thinking: str = Field(description="Agent's reasoning about the step")
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tools the agent wants to call"
    )
    explanation: str = Field(description="What these tool calls accomplish")


# =============================================================================
# STEP EXECUTION SCHEMAS
# =============================================================================

class StepStatus(str, Enum):
    """Status of a plan step during execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    REWORK_REQUESTED = "rework_requested"


class StepResult(BaseModel):
    """
    Result from executing a single plan step.

    Contains all the tool calls made, their results, and metadata.
    """
    step_number: int = Field(description="Which step this result is for")
    status: StepStatus = Field(description="Current status of the step")
    agent_name: str = Field(description="Agent that executed this step")
    tool_calls: list[ToolCallRequest] = Field(
        default_factory=list,
        description="Tool calls made during execution"
    )
    tool_results: list[ToolCallResult] = Field(
        default_factory=list,
        description="Results from each tool call"
    )
    output_summary: str = Field(
        default="",
        description="Summary of what was accomplished"
    )
    files_created: list[str] = Field(
        default_factory=list,
        description="Files created in this step"
    )
    files_modified: list[str] = Field(
        default_factory=list,
        description="Files modified in this step"
    )
    tokens_used: int = Field(default=0, description="Tokens consumed")
    cost: float = Field(default=0.0, description="Estimated cost")
    error: Optional[str] = Field(default=None, description="Error if step failed")


# =============================================================================
# USER FEEDBACK SCHEMAS
# =============================================================================

class FeedbackType(str, Enum):
    """Types of feedback a user can give at a checkpoint."""
    APPROVE = "approve"      # Proceed to next step
    MODIFY = "modify"        # Make small changes, then proceed
    REWORK = "rework"        # Redo this step with feedback
    SKIP = "skip"            # Skip this step
    ABORT = "abort"          # Stop execution entirely


class UserFeedback(BaseModel):
    """
    User's feedback at an execution checkpoint.
    """
    feedback_type: FeedbackType = Field(description="Type of feedback")
    message: Optional[str] = Field(
        default=None,
        description="User's message or instructions"
    )
    specific_changes: list[str] = Field(
        default_factory=list,
        description="Specific changes requested"
    )


class ChairmanSummary(BaseModel):
    """
    Chairman's summary of step execution for user review.

    After each step completes, the chairman summarizes what happened
    and makes a recommendation.
    """
    step_number: int = Field(description="Which step was executed")
    title: str = Field(description="Step title")
    what_was_done: str = Field(description="Summary of actions taken")
    files_changed: list[str] = Field(
        default_factory=list,
        description="Files that were created or modified"
    )
    key_decisions: list[str] = Field(
        default_factory=list,
        description="Important decisions made during execution"
    )
    potential_concerns: list[str] = Field(
        default_factory=list,
        description="Issues the user should be aware of"
    )
    recommendation: Literal["proceed", "review_recommended", "rework_suggested"] = Field(
        description="Chairman's recommendation for next action"
    )


# =============================================================================
# EXECUTION TRANSCRIPT SCHEMAS
# =============================================================================

class ExecutionEvent(BaseModel):
    """
    A single event in the execution timeline.

    Used to build a complete audit trail of execution.
    """
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this event occurred"
    )
    event_type: str = Field(
        description="Type of event (step_started, tool_called, checkpoint, etc.)"
    )
    step_number: Optional[int] = Field(
        default=None,
        description="Associated step number if applicable"
    )
    details: str = Field(description="Event details")


class ExecutionTranscript(BaseModel):
    """
    Complete record of plan execution.

    Includes all steps, events, costs, and timing for analysis.
    """
    session_id: str = Field(description="Session identifier")
    plan_overview: str = Field(description="Overview of the plan being executed")
    started_at: datetime = Field(
        default_factory=datetime.now,
        description="When execution started"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When execution completed"
    )
    status: Literal["in_progress", "completed", "aborted", "failed"] = Field(
        default="in_progress",
        description="Overall execution status"
    )
    step_results: list[StepResult] = Field(
        default_factory=list,
        description="Results from each step"
    )
    events: list[ExecutionEvent] = Field(
        default_factory=list,
        description="Timeline of all events"
    )
    total_tokens: int = Field(default=0, description="Total tokens used")
    total_cost: float = Field(default=0.0, description="Total estimated cost")
    rework_count: int = Field(default=0, description="Number of rework iterations")


class ExecutionResult(BaseModel):
    """
    Final result from executing a complete plan.
    """
    success: bool = Field(description="Whether execution succeeded")
    transcript: ExecutionTranscript = Field(description="Full execution record")
    final_context: ExecutionContext = Field(description="Final execution context")
    output_path: str = Field(description="Path to workspace with outputs")
    summary: str = Field(description="Human-readable summary")
    files_created: list[str] = Field(
        default_factory=list,
        description="All files created during execution"
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="Suggested next steps"
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_json_schema(model: type[BaseModel]) -> dict:
    """
    Get the JSON schema for a Pydantic model.

    This is what we send to the API to tell the model what structure we expect.

    Example:
        schema = get_json_schema(Proposal)
        # Returns a dict like:
        # {
        #   "type": "object",
        #   "properties": {
        #     "approach": {"type": "string", ...},
        #     "effort": {"enum": ["low", "medium", "high"], ...},
        #     ...
        #   },
        #   "required": ["approach", "rationale", "effort", "confidence"]
        # }
    """
    return model.model_json_schema()


# List of all schemas for easy iteration
ALL_SCHEMAS = [
    # Deliberation schemas
    Proposal,
    Critique,
    RoleAssignment,
    RoleDeliberationResult,
    PlanStep,
    Plan,
    ReviewIssue,
    CodeReview,
    AgentResponse,
    DeliberationPhase,
    DeliberationTranscript,
    # Tool schemas
    ToolParameter,
    ToolDefinition,
    ToolCall,
    ToolCallRequest,
    ToolCallResult,
    ToolResult,
    # Execution schemas
    ExecutionContext,
    AgentToolCalls,
    StepResult,
    UserFeedback,
    ChairmanSummary,
    ExecutionEvent,
    ExecutionTranscript,
    ExecutionResult,
]
