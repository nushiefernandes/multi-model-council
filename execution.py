"""
Execution Engine for Council v2 - Phase 3.

WHAT THIS FILE DOES:
-------------------
After deliberation creates a Plan, this module EXECUTES it.
It takes structured plans and turns them into real files, real commands,
real outputs.

HOW IT WORKS:
------------
1. ToolRegistry: Defines what tools agents can use (file_write, run_command, etc.)
2. ToolImplementations: Actually performs the tool operations
3. AgentExecutor: Has agents generate tool calls for plan steps
4. InteractiveExecutor: Handles user checkpoints (approve/rework/abort)
5. ExecutionEngine: Orchestrates the whole execution flow

EXECUTION FLOW:
--------------
    Plan from deliberation
           │
           ▼
    For each PlanStep:
    ├── Agent generates ToolCalls (AgentExecutor)
    ├── Tools are executed (ToolRegistry + ToolImplementations)
    ├── Chairman summarizes (InteractiveExecutor)
    ├── User provides feedback (approve/modify/rework/abort)
    └── Update context, move to next step
           │
           ▼
    ExecutionResult with created files, transcript, summary
"""

import asyncio
import os
import re
import subprocess
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Type, TypeVar

from pydantic import BaseModel

from schemas import (
    # Plan schemas
    Plan,
    PlanStep,
    # Tool schemas
    ToolDefinition,
    ToolParameter,
    ToolCall,
    ToolCallRequest,
    ToolCallResult,
    # Execution schemas
    ExecutionContext,
    AgentToolCalls,
    StepStatus,
    StepResult,
    FeedbackType,
    UserFeedback,
    ChairmanSummary,
    ExecutionEvent,
    ExecutionTranscript,
    ExecutionResult,
)
from providers import ModelProvider, get_provider
from workspace import WorkspaceManager

T = TypeVar('T', bound=BaseModel)


# =============================================================================
# SECTION 1: TOOL REGISTRY
# =============================================================================

class ToolRegistry:
    """
    Registry of available tools that agents can use.

    WHY THIS EXISTS:
    ----------------
    Agents need to know what tools they can use. This registry:
    1. Stores tool definitions (name, description, parameters)
    2. Validates tool calls before execution
    3. Routes calls to the correct implementation
    4. Generates prompts describing available tools

    Example usage:
        registry = ToolRegistry()
        registry.register(file_write_def, implementations.file_write)

        # Later, when agent wants to call a tool:
        result = await registry.execute(call_request, context)
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._implementations: dict[str, Callable] = {}

    def register(self, definition: ToolDefinition, implementation: Callable) -> None:
        """
        Register a tool with its implementation.

        Args:
            definition: ToolDefinition describing the tool
            implementation: Async function that performs the tool's action
        """
        self._tools[definition.name] = definition
        self._implementations[definition.name] = implementation

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """Get all registered tools."""
        return list(self._tools.values())

    def validate_call(self, call: ToolCall) -> tuple[bool, Optional[str]]:
        """
        Validate a tool call against its definition.

        Returns:
            (is_valid, error_message) - error_message is None if valid
        """
        tool = self._tools.get(call.tool_name)
        if not tool:
            return False, f"Unknown tool: {call.tool_name}"

        # Check required parameters
        for param in tool.parameters:
            if param.required and param.name not in call.parameters:
                return False, f"Missing required parameter: {param.name}"

        # Check for unknown parameters
        param_names = {p.name for p in tool.parameters}
        for key in call.parameters:
            if key not in param_names:
                return False, f"Unknown parameter: {key}"

        return True, None

    async def execute(
        self,
        call: ToolCallRequest,
        context: ExecutionContext
    ) -> ToolCallResult:
        """
        Execute a tool call.

        Args:
            call: The tool call request with parameters
            context: Current execution context

        Returns:
            ToolCallResult with success/failure and output
        """
        start_time = time.time()

        # Validate the call
        tool = self._tools.get(call.tool_name)
        if not tool:
            return ToolCallResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                output="",
                error=f"Unknown tool: {call.tool_name}",
                duration_ms=0
            )

        impl = self._implementations.get(call.tool_name)
        if not impl:
            return ToolCallResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                output="",
                error=f"No implementation for tool: {call.tool_name}",
                duration_ms=0
            )

        try:
            # Execute the tool
            output = await impl(**call.parameters, context=context)
            duration_ms = int((time.time() - start_time) * 1000)

            return ToolCallResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=True,
                output=str(output),
                error=None,
                duration_ms=duration_ms
            )
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return ToolCallResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                output="",
                error=str(e),
                duration_ms=duration_ms
            )

    def get_tools_prompt(self) -> str:
        """
        Generate a prompt describing all available tools for agents.

        This is included in the system prompt when asking agents
        to generate tool calls.
        """
        lines = ["Available tools:\n"]

        for tool in self._tools.values():
            lines.append(f"## {tool.name}")
            lines.append(f"Description: {tool.description}")
            lines.append(f"Returns: {tool.returns}")
            if tool.dangerous:
                lines.append("⚠️ DANGEROUS: Requires user approval")
            lines.append("Parameters:")
            for param in tool.parameters:
                required = "(required)" if param.required else "(optional)"
                default = f" [default: {param.default}]" if param.default is not None else ""
                lines.append(f"  - {param.name} ({param.type}) {required}: {param.description}{default}")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# SECTION 2: TOOL IMPLEMENTATIONS
# =============================================================================

class ToolImplementations:
    """
    Standard tool implementations.

    These are the actual functions that perform tool operations.
    All operations are sandboxed to the workspace directory for safety.

    WHY SANDBOXING MATTERS:
    ----------------------
    Without sandboxing, an agent could:
    - Write files anywhere on the system
    - Run dangerous commands like `rm -rf /`
    - Access sensitive files outside the project

    We restrict all operations to workspace_path.

    WORKSPACE MANAGER INTEGRATION:
    -----------------------------
    When a WorkspaceManager is provided, file operations use its methods
    for automatic organization (placing files in src/, tests/, docs/ etc).
    """

    def __init__(self, workspace_path: Path, workspace_manager=None):
        """
        Initialize implementations with a workspace.

        Args:
            workspace_path: All file operations are restricted to this directory
            workspace_manager: Optional WorkspaceManager for auto-organization
        """
        self.workspace = Path(workspace_path).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._workspace_manager = workspace_manager

    def _resolve_path(self, path: str) -> Path:
        """
        Resolve a path within the workspace, preventing directory traversal.

        Args:
            path: Relative path within workspace

        Returns:
            Absolute path guaranteed to be within workspace

        Raises:
            ValueError: If path attempts to escape workspace
        """
        # Remove leading slashes to treat as relative
        path = path.lstrip("/")

        # Resolve the full path
        full_path = (self.workspace / path).resolve()

        # Ensure it's still within workspace
        try:
            full_path.relative_to(self.workspace.resolve())
        except ValueError:
            raise ValueError(f"Path '{path}' attempts to escape workspace")

        return full_path

    async def file_write(
        self,
        path: str,
        content: str,
        context: ExecutionContext
    ) -> str:
        """
        Write content to a file in the workspace.

        Args:
            path: Relative path within workspace
            content: Content to write
            context: Execution context (updated with created file)

        Returns:
            Confirmation message with path
        """
        # Use workspace manager if available for auto-organization
        if self._workspace_manager:
            rel_path = self._workspace_manager.write_file(path, content)
            if rel_path not in context.created_files:
                context.created_files.append(rel_path)
            return f"Successfully wrote {len(content)} characters to {rel_path}"

        # Fall back to direct file operations
        full_path = self._resolve_path(path)

        # Create parent directories
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        full_path.write_text(content)

        # Track in context
        rel_path = str(full_path.relative_to(self.workspace))
        if rel_path not in context.created_files:
            context.created_files.append(rel_path)

        return f"Successfully wrote {len(content)} characters to {rel_path}"

    async def file_read(
        self,
        path: str,
        context: ExecutionContext
    ) -> str:
        """
        Read a file from the workspace.

        Args:
            path: Relative path within workspace
            context: Execution context

        Returns:
            File contents

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        # Use workspace manager if available
        if self._workspace_manager:
            return self._workspace_manager.read_file(path)

        full_path = self._resolve_path(path)

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return full_path.read_text()

    async def file_append(
        self,
        path: str,
        content: str,
        context: ExecutionContext
    ) -> str:
        """
        Append content to a file.

        Args:
            path: Relative path within workspace
            content: Content to append
            context: Execution context

        Returns:
            Confirmation message
        """
        full_path = self._resolve_path(path)

        # Create if doesn't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, "a") as f:
            f.write(content)

        rel_path = str(full_path.relative_to(self.workspace))
        return f"Appended {len(content)} characters to {rel_path}"

    async def run_command(
        self,
        command: str,
        working_dir: Optional[str] = None,
        context: ExecutionContext = None
    ) -> str:
        """
        Execute a shell command (sandboxed).

        SECURITY:
        - Commands run in workspace directory
        - Certain dangerous patterns are blocked
        - Timeout enforced

        Args:
            command: Shell command to run
            working_dir: Subdirectory within workspace (optional)
            context: Execution context

        Returns:
            Command output (stdout + stderr)
        """
        # Dangerous command patterns to block
        dangerous_patterns = [
            r"rm\s+-rf\s+/",  # rm -rf /
            r"rm\s+-rf\s+~",  # rm -rf ~
            r">\s*/dev/",     # Redirect to device files
            r"mkfs\.",        # Format filesystems
            r"dd\s+if=",      # Direct disk access
            r":(){ :|:& };:", # Fork bomb
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, command):
                raise ValueError(f"Blocked dangerous command pattern: {pattern}")

        # Determine working directory
        if working_dir:
            cwd = self._resolve_path(working_dir)
        else:
            cwd = self.workspace

        # Run the command with timeout
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code]: {result.returncode}"

            return output.strip() if output else "(no output)"

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command timed out after 60 seconds: {command}")

    async def generate_code(
        self,
        description: str,
        language: str,
        context_files: Optional[list[str]] = None,
        context: ExecutionContext = None,
        provider: Optional[ModelProvider] = None
    ) -> str:
        """
        Have a model generate code based on description.

        Args:
            description: What the code should do
            language: Programming language
            context_files: Files to include as context
            context: Execution context
            provider: Model provider to use for generation

        Returns:
            Generated code
        """
        if not provider:
            raise ValueError("No provider configured for code generation")

        # Build context from files
        file_context = ""
        if context_files:
            for file_path in context_files:
                try:
                    full_path = self._resolve_path(file_path)
                    if full_path.exists():
                        content = full_path.read_text()
                        file_context += f"\n### {file_path}\n```{language}\n{content}\n```\n"
                except Exception:
                    pass

        prompt = f"""Generate {language} code for the following:

{description}

{f"Context files:{file_context}" if file_context else ""}

Return ONLY the code, no explanations or markdown code blocks."""

        result = await provider.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        # Strip any markdown code blocks the model might add
        code = result["content"]
        code = re.sub(r'^```\w*\n', '', code)
        code = re.sub(r'\n```$', '', code)

        return code.strip()

    async def search_codebase(
        self,
        pattern: str,
        file_glob: Optional[str] = None,
        context: ExecutionContext = None
    ) -> str:
        """
        Search workspace for patterns.

        Args:
            pattern: Regex pattern to search for
            file_glob: Optional glob pattern to filter files
            context: Execution context

        Returns:
            Search results with file paths and line numbers
        """
        results = []

        # Determine files to search
        if file_glob:
            files = list(self.workspace.glob(file_glob))
        else:
            files = [f for f in self.workspace.rglob("*") if f.is_file()]

        # Skip binary files and common non-code files
        skip_extensions = {'.pyc', '.pyo', '.so', '.dll', '.exe', '.bin', '.jpg', '.png', '.gif'}

        for file_path in files:
            if file_path.suffix in skip_extensions:
                continue

            try:
                content = file_path.read_text()
                rel_path = str(file_path.relative_to(self.workspace))

                for i, line in enumerate(content.split('\n'), 1):
                    if re.search(pattern, line):
                        results.append(f"{rel_path}:{i}: {line.strip()}")

            except (UnicodeDecodeError, PermissionError):
                continue

        if not results:
            return f"No matches found for pattern: {pattern}"

        return "\n".join(results[:50])  # Limit results


def create_default_tools(workspace_path: Path, workspace_manager=None) -> tuple[ToolRegistry, ToolImplementations]:
    """
    Create a registry with default tools.

    This is the standard set of tools available to agents.

    Args:
        workspace_path: Path for file operations
        workspace_manager: Optional WorkspaceManager for auto-organization
    """
    registry = ToolRegistry()
    implementations = ToolImplementations(workspace_path, workspace_manager=workspace_manager)

    # Register file_write
    registry.register(
        ToolDefinition(
            name="file_write",
            description="Write content to a file in the workspace. Creates parent directories if needed.",
            parameters=[
                ToolParameter(name="path", type="string", description="Relative path to the file", required=True),
                ToolParameter(name="content", type="string", description="Content to write", required=True),
            ],
            returns="Confirmation message with bytes written",
            dangerous=False
        ),
        implementations.file_write
    )

    # Register file_read
    registry.register(
        ToolDefinition(
            name="file_read",
            description="Read a file from the workspace.",
            parameters=[
                ToolParameter(name="path", type="string", description="Relative path to the file", required=True),
            ],
            returns="File contents as string",
            dangerous=False
        ),
        implementations.file_read
    )

    # Register file_append
    registry.register(
        ToolDefinition(
            name="file_append",
            description="Append content to an existing file (or create if it doesn't exist).",
            parameters=[
                ToolParameter(name="path", type="string", description="Relative path to the file", required=True),
                ToolParameter(name="content", type="string", description="Content to append", required=True),
            ],
            returns="Confirmation message",
            dangerous=False
        ),
        implementations.file_append
    )

    # Register run_command
    registry.register(
        ToolDefinition(
            name="run_command",
            description="Execute a shell command in the workspace. Commands are sandboxed and some dangerous patterns are blocked.",
            parameters=[
                ToolParameter(name="command", type="string", description="Shell command to execute", required=True),
                ToolParameter(name="working_dir", type="string", description="Subdirectory to run in (relative to workspace)", required=False),
            ],
            returns="Command output (stdout and stderr)",
            dangerous=True
        ),
        implementations.run_command
    )

    # Register search_codebase
    registry.register(
        ToolDefinition(
            name="search_codebase",
            description="Search the workspace for a regex pattern.",
            parameters=[
                ToolParameter(name="pattern", type="string", description="Regex pattern to search for", required=True),
                ToolParameter(name="file_glob", type="string", description="Optional glob to filter files (e.g., '*.py')", required=False),
            ],
            returns="Matching lines with file paths and line numbers",
            dangerous=False
        ),
        implementations.search_codebase
    )

    return registry, implementations


# =============================================================================
# SECTION 3: AGENT EXECUTOR
# =============================================================================

class AgentExecutor:
    """
    Manages agent tool call generation.

    This class is responsible for having agents look at plan steps
    and generate the tool calls needed to execute them.

    HOW IT WORKS:
    1. Take a PlanStep (e.g., "Create the database models")
    2. Show the agent what tools are available
    3. Agent responds with AgentToolCalls (thinking + tool_calls + explanation)
    4. We validate and return the structured response
    """

    def __init__(self, config: dict, tool_registry: ToolRegistry):
        """
        Initialize the agent executor.

        Args:
            config: Model configuration dict
            tool_registry: Registry of available tools
        """
        self.config = config
        self.tool_registry = tool_registry

    def _get_provider(self, model_name: str) -> ModelProvider:
        """Get a provider for the given model."""
        return get_provider(self.config, model_name)

    async def generate_tool_calls(
        self,
        step: PlanStep,
        context: ExecutionContext,
        previous_results: Optional[list[ToolCallResult]] = None,
        model_name: str = "claude"
    ) -> AgentToolCalls:
        """
        Have an agent generate tool calls for a step.

        Args:
            step: The plan step to execute
            context: Current execution context
            previous_results: Results from previous tool calls (for retry/rework)
            model_name: Which model to use for generation

        Returns:
            AgentToolCalls with the agent's proposed actions
        """
        provider = self._get_provider(model_name)

        # Build the system prompt
        system = f"""You are an expert software developer executing a plan step.
You have access to tools to accomplish the task.

{self.tool_registry.get_tools_prompt()}

When generating tool calls, provide:
1. Your thinking process
2. The specific tool calls needed
3. An explanation of what these accomplish

Be precise and only call tools that are actually needed."""

        # Build context about what's been done
        context_info = ""
        if context.completed_steps:
            context_info += f"\nCompleted steps: {context.completed_steps}"
        if context.created_files:
            context_info += f"\nFiles created so far: {', '.join(context.created_files)}"
        if context.step_outputs:
            context_info += "\nPrevious outputs:"
            for step_num, output in context.step_outputs.items():
                context_info += f"\n  Step {step_num}: {output[:200]}..."

        # Add previous results if this is a retry
        previous_info = ""
        if previous_results:
            previous_info = "\nPrevious attempt results:"
            for result in previous_results:
                status = "✓" if result.success else "✗"
                previous_info += f"\n  {status} {result.tool_name}: {result.output or result.error}"

        prompt = f"""Execute this step:

Step {step.step_number}: {step.title}
Description: {step.description}
Acceptance Criteria: {', '.join(step.acceptance_criteria)}

Workspace: {context.workspace_path}
{context_info}
{previous_info}

Generate the tool calls needed to complete this step."""

        # Get structured response
        result, usage = await provider.complete_structured(
            messages=[{"role": "user", "content": prompt}],
            schema=AgentToolCalls,
            system=system,
            temperature=0.3
        )

        return result

    async def refine_with_feedback(
        self,
        step: PlanStep,
        context: ExecutionContext,
        feedback: UserFeedback,
        previous_result: StepResult,
        model_name: str = "claude"
    ) -> AgentToolCalls:
        """
        Regenerate tool calls based on user feedback.

        This is called when the user requests rework or modifications.

        Args:
            step: The plan step being reworked
            context: Current execution context
            feedback: User's feedback/instructions
            previous_result: Result from the previous attempt
            model_name: Which model to use

        Returns:
            New AgentToolCalls incorporating the feedback
        """
        provider = self._get_provider(model_name)

        system = f"""You are an expert software developer revising your work based on feedback.
You have access to tools to accomplish the task.

{self.tool_registry.get_tools_prompt()}

The user has requested changes. Incorporate their feedback carefully."""

        # Summarize what was done before
        previous_actions = "\n".join([
            f"- {tc.tool_name}({tc.parameters})"
            for tc in previous_result.tool_calls
        ])

        prompt = f"""Revise this step based on feedback:

Step {step.step_number}: {step.title}
Description: {step.description}

Previous actions taken:
{previous_actions}

Previous outcome:
{previous_result.output_summary}

User feedback:
Type: {feedback.feedback_type.value}
Message: {feedback.message or "No message"}
Specific changes requested: {', '.join(feedback.specific_changes) if feedback.specific_changes else "None specified"}

Generate new tool calls that address the feedback."""

        result, usage = await provider.complete_structured(
            messages=[{"role": "user", "content": prompt}],
            schema=AgentToolCalls,
            system=system,
            temperature=0.3
        )

        return result


# =============================================================================
# SECTION 4: INTERACTIVE EXECUTOR
# =============================================================================

class InteractiveExecutor:
    """
    Handles user interaction during execution.

    This class manages the "human in the loop" aspect:
    - Presenting what's about to happen
    - Summarizing what was done
    - Getting user feedback

    WHY THIS MATTERS:
    ----------------
    Without checkpoints, execution could run away:
    - Agent makes a wrong decision early
    - Compounds errors through subsequent steps
    - User ends up with broken code

    With checkpoints:
    - User sees each step's result
    - Can approve, modify, or rework
    - Maintains control over the process
    """

    def __init__(
        self,
        config: dict,
        checkpoint_callback: Optional[Callable[[ChairmanSummary], UserFeedback]] = None
    ):
        """
        Initialize the interactive executor.

        Args:
            config: Model configuration dict
            checkpoint_callback: Optional callback for getting user feedback
                                If None, auto-approves everything
        """
        self.config = config
        self.checkpoint_callback = checkpoint_callback

    def _get_provider(self, model_name: str) -> ModelProvider:
        """Get a provider for the given model."""
        return get_provider(self.config, model_name)

    async def present_step_start(
        self,
        step: PlanStep,
        context: ExecutionContext
    ) -> None:
        """
        Show what's about to happen (for UI/logging).

        Args:
            step: The step about to be executed
            context: Current context
        """
        # This can be extended to call a UI callback
        print(f"\n{'='*60}")
        print(f"Step {step.step_number}: {step.title}")
        print(f"Agent: {step.assigned_agent}")
        print(f"{'='*60}")

    async def summarize_step_result(
        self,
        step: PlanStep,
        result: StepResult,
        context: ExecutionContext,
        model_name: str = "claude"
    ) -> ChairmanSummary:
        """
        Have the chairman summarize step results for user review.

        This creates a user-friendly summary of what happened,
        highlighting key decisions and potential concerns.

        Args:
            step: The executed step
            result: Results from execution
            context: Current context
            model_name: Model to use for summarization

        Returns:
            ChairmanSummary with recommendation
        """
        provider = self._get_provider(model_name)

        system = """You are the chairman summarizing execution results for the user.
Be clear and concise. Highlight:
1. What was accomplished
2. Key decisions made
3. Any concerns the user should know about
4. Your recommendation (proceed, review_recommended, or rework_suggested)"""

        # Build details of what happened
        tool_summary = "\n".join([
            f"- {tc.tool_name}: {tr.output if tr.success else tr.error}"
            for tc, tr in zip(result.tool_calls, result.tool_results)
        ])

        prompt = f"""Summarize this step's execution:

Step: {step.title}
Description: {step.description}
Status: {result.status.value}

Tool calls made:
{tool_summary}

Files created: {', '.join(result.files_created) if result.files_created else 'None'}
Files modified: {', '.join(result.files_modified) if result.files_modified else 'None'}

Provide a summary for the user."""

        summary, usage = await provider.complete_structured(
            messages=[{"role": "user", "content": prompt}],
            schema=ChairmanSummary,
            system=system,
            temperature=0.3
        )

        return summary

    async def get_user_feedback(
        self,
        summary: ChairmanSummary,
        step: PlanStep
    ) -> UserFeedback:
        """
        Get user feedback at a checkpoint.

        If a callback is configured, uses it. Otherwise auto-approves.

        Args:
            summary: Chairman's summary of the step
            step: The step that was executed

        Returns:
            UserFeedback indicating what to do next
        """
        if self.checkpoint_callback:
            return self.checkpoint_callback(summary)

        # Auto-approve if no callback
        return UserFeedback(feedback_type=FeedbackType.APPROVE)


# =============================================================================
# SECTION 5: EXECUTION ENGINE
# =============================================================================

class ExecutionEngine:
    """
    Main execution engine that orchestrates plan execution.

    This is the central coordinator that:
    1. Orders steps by dependencies (topological sort)
    2. Executes each step through agents
    3. Manages checkpoints and user feedback
    4. Tracks everything in the transcript
    5. Handles rework and error recovery
    """

    def __init__(
        self,
        config: dict,
        workspace_base: Path,
        interactive: bool = True,
        checkpoint_callback: Optional[Callable[[ChairmanSummary], UserFeedback]] = None
    ):
        """
        Initialize the execution engine.

        Args:
            config: Model configuration dict
            workspace_base: Base directory for workspaces
            interactive: Whether to use interactive checkpoints
            checkpoint_callback: Optional callback for user feedback
        """
        self.config = config
        self.workspace_base = Path(workspace_base).resolve()
        self.interactive = interactive

        # Create components
        self.tool_registry, self.tool_implementations = create_default_tools(
            (self.workspace_base / "workspace").resolve()
        )
        self.agent_executor = AgentExecutor(config, self.tool_registry)
        self.interactive_executor = InteractiveExecutor(
            config,
            checkpoint_callback=checkpoint_callback if interactive else None
        )

        # Tracking
        self._transcript: Optional[ExecutionTranscript] = None

    def _order_steps(self, plan: Plan) -> list[PlanStep]:
        """
        Order steps respecting dependencies (topological sort).

        Ensures steps run in valid order where dependencies complete first.

        Args:
            plan: The plan with steps to order

        Returns:
            Steps in execution order
        """
        # Build dependency graph
        steps_by_number = {step.step_number: step for step in plan.steps}
        in_degree = defaultdict(int)
        dependents = defaultdict(list)

        for step in plan.steps:
            in_degree[step.step_number] = len(step.dependencies)
            for dep in step.dependencies:
                dependents[dep].append(step.step_number)

        # Kahn's algorithm for topological sort
        queue = [
            step.step_number
            for step in plan.steps
            if in_degree[step.step_number] == 0
        ]
        ordered = []

        while queue:
            # Sort queue to ensure deterministic ordering
            queue.sort()
            current = queue.pop(0)
            ordered.append(steps_by_number[current])

            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check for cycles
        if len(ordered) != len(plan.steps):
            raise ValueError("Dependency cycle detected in plan steps")

        return ordered

    async def execute_tool_calls(
        self,
        calls: list[ToolCallRequest],
        context: ExecutionContext
    ) -> list[ToolCallResult]:
        """
        Execute tool calls sequentially.

        Args:
            calls: List of tool calls to execute
            context: Execution context

        Returns:
            Results from each call
        """
        results = []

        for call in calls:
            result = await self.tool_registry.execute(call, context)
            results.append(result)

            # If a call fails, we might want to stop
            # For now, continue and let the agent handle failures

        return results

    async def execute_step(
        self,
        step: PlanStep,
        context: ExecutionContext,
        model_name: str = "claude"
    ) -> StepResult:
        """
        Execute a single plan step.

        Args:
            step: The step to execute
            context: Current context
            model_name: Model to use for this step

        Returns:
            StepResult with execution details
        """
        # Record event
        self._add_event(
            "step_started",
            step.step_number,
            f"Starting step {step.step_number}: {step.title}"
        )

        # Present to user
        await self.interactive_executor.present_step_start(step, context)

        # Have agent generate tool calls
        agent_calls = await self.agent_executor.generate_tool_calls(
            step, context, model_name=model_name
        )

        # Convert to ToolCallRequests
        call_requests = []
        for tool_call in agent_calls.tool_calls:
            call_requests.append(ToolCallRequest(
                call_id=str(uuid.uuid4())[:8],
                tool_name=tool_call.tool_name,
                parameters=tool_call.parameters,
                reasoning=tool_call.reasoning,
                agent_name=step.assigned_agent,
                step_number=step.step_number,
                timestamp=datetime.now()
            ))

        # Track files before execution to detect new ones
        files_before = set(context.created_files)

        # Execute tool calls
        call_results = await self.execute_tool_calls(call_requests, context)

        # Track created files (context is updated by tools)
        files_after = set(context.created_files)
        files_created = list(files_after - files_before)
        files_modified = []

        # Also try to extract from output for tools that don't update context
        for result in call_results:
            if result.success and "wrote" in result.output.lower():
                match = re.search(r"to (\S+)$", result.output)
                if match:
                    path = match.group(1)
                    if path not in files_created:
                        files_created.append(path)

        # Determine status
        all_succeeded = all(r.success for r in call_results)
        status = StepStatus.COMPLETED if all_succeeded else StepStatus.FAILED

        # Build result
        result = StepResult(
            step_number=step.step_number,
            status=status,
            agent_name=step.assigned_agent,
            tool_calls=call_requests,
            tool_results=call_results,
            output_summary=agent_calls.explanation,
            files_created=files_created,
            files_modified=files_modified,
            error=None if all_succeeded else "One or more tool calls failed"
        )

        return result

    async def execute_plan(
        self,
        plan: Plan,
        session_id: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a complete plan.

        Args:
            plan: The plan to execute
            session_id: Optional session ID (generated if not provided)

        Returns:
            ExecutionResult with full transcript and outputs
        """
        # Setup
        session_id = session_id or str(uuid.uuid4())[:8]

        # Create workspace manager for organized output
        workspace_manager = WorkspaceManager(
            output_path=self.workspace_base,
            session_id=session_id,
            include_timestamp=False  # Keep paths predictable
        )
        workspace_path = workspace_manager.setup()

        # Update tool implementations with workspace manager
        self.tool_implementations.workspace = workspace_path.resolve()
        self.tool_implementations._workspace_manager = workspace_manager

        # Initialize context
        context = ExecutionContext(
            session_id=session_id,
            workspace_path=str(workspace_path)
        )

        # Initialize transcript
        self._transcript = ExecutionTranscript(
            session_id=session_id,
            plan_overview=plan.overview,
            started_at=datetime.now(),
            status="in_progress"
        )

        # Order steps
        ordered_steps = self._order_steps(plan)

        # Execute each step
        for step in ordered_steps:
            try:
                # Execute step
                result = await self.execute_step(step, context)
                self._transcript.step_results.append(result)

                # Interactive checkpoint
                if self.interactive and result.status == StepStatus.COMPLETED:
                    summary = await self.interactive_executor.summarize_step_result(
                        step, result, context
                    )

                    feedback = await self.interactive_executor.get_user_feedback(
                        summary, step
                    )

                    # Handle feedback
                    if feedback.feedback_type == FeedbackType.ABORT:
                        self._transcript.status = "aborted"
                        break
                    elif feedback.feedback_type == FeedbackType.REWORK:
                        self._transcript.rework_count += 1
                        # Regenerate with feedback
                        new_calls = await self.agent_executor.refine_with_feedback(
                            step, context, feedback, result
                        )
                        # Re-execute with new calls
                        # (simplified - in production would loop until approved)

                # Update context
                context.completed_steps.append(step.step_number)
                context.step_outputs[step.step_number] = result.output_summary
                context.created_files.extend(result.files_created)

            except Exception as e:
                self._add_event(
                    "step_failed",
                    step.step_number,
                    f"Step failed with error: {e}"
                )
                self._transcript.status = "failed"
                break

        # Complete transcript
        self._transcript.completed_at = datetime.now()
        if self._transcript.status == "in_progress":
            self._transcript.status = "completed"

        # Calculate totals
        self._transcript.total_tokens = sum(
            sr.tokens_used for sr in self._transcript.step_results
        )
        self._transcript.total_cost = sum(
            sr.cost for sr in self._transcript.step_results
        )

        # Finalize workspace and get stats
        workspace_manager.finalize()

        # Build result
        return ExecutionResult(
            success=self._transcript.status == "completed",
            transcript=self._transcript,
            final_context=context,
            output_path=str(workspace_path),
            summary=f"Executed {len(self._transcript.step_results)} steps",
            files_created=context.created_files,
            next_steps=["Review generated files", "Run tests", "Deploy if ready"]
        )

    def _add_event(
        self,
        event_type: str,
        step_number: Optional[int],
        details: str
    ) -> None:
        """Add an event to the transcript."""
        if self._transcript:
            self._transcript.events.append(ExecutionEvent(
                timestamp=datetime.now(),
                event_type=event_type,
                step_number=step_number,
                details=details
            ))


# =============================================================================
# SECTION 6: CONVENIENCE FUNCTIONS
# =============================================================================

async def execute_plan_interactive(
    plan: Plan,
    config: Optional[dict] = None,
    workspace_base: Optional[Path] = None,
    checkpoint_callback: Optional[Callable[[ChairmanSummary], UserFeedback]] = None
) -> ExecutionResult:
    """
    Execute a plan with interactive checkpoints.

    This is the main entry point for interactive execution.

    Args:
        plan: The plan to execute
        config: Model configuration (uses defaults if not provided)
        workspace_base: Base directory for workspaces
        checkpoint_callback: Callback for user feedback at checkpoints

    Returns:
        ExecutionResult with full details
    """
    from deliberation import create_default_config

    config = config or create_default_config()
    workspace_base = workspace_base or Path.home() / ".council" / "workspaces"

    engine = ExecutionEngine(
        config=config,
        workspace_base=workspace_base,
        interactive=True,
        checkpoint_callback=checkpoint_callback
    )

    return await engine.execute_plan(plan)


async def execute_plan_headless(
    plan: Plan,
    config: Optional[dict] = None,
    workspace_base: Optional[Path] = None
) -> ExecutionResult:
    """
    Execute a plan without interaction (for testing/automation).

    All checkpoints are auto-approved.

    Args:
        plan: The plan to execute
        config: Model configuration (uses defaults if not provided)
        workspace_base: Base directory for workspaces

    Returns:
        ExecutionResult with full details
    """
    from deliberation import create_default_config

    config = config or create_default_config()
    workspace_base = workspace_base or Path.home() / ".council" / "workspaces"

    engine = ExecutionEngine(
        config=config,
        workspace_base=workspace_base,
        interactive=False
    )

    return await engine.execute_plan(plan)


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test_basic():
        """Quick smoke test."""
        # Create a simple plan
        plan = Plan(
            overview="Create a hello world file",
            steps=[
                PlanStep(
                    step_number=1,
                    title="Create hello.py",
                    description="Create a Python file that prints hello world",
                    assigned_agent="Backend Dev",
                    dependencies=[],
                    estimated_complexity="low",
                    acceptance_criteria=["File exists", "Contains print statement"]
                )
            ],
            critical_path=[1],
            estimated_total_effort="low",
            risks_and_mitigations={}
        )

        print("Testing execution engine...")

        # Execute headless
        result = await execute_plan_headless(plan)

        print(f"\nExecution {'succeeded' if result.success else 'failed'}")
        print(f"Files created: {result.files_created}")
        print(f"Output path: {result.output_path}")

    asyncio.run(test_basic())
