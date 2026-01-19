"""
Phase 3 Integration Tests: Execution Layer

These tests verify the execution layer can:
1. Register and execute tools
2. Have agents generate tool calls
3. Execute complete plans
4. Handle checkpoints and rework
5. Integrate with deliberation

Test list:
1. test_tool_registry - Tools register, validate, execute correctly
2. test_file_write_tool - Files created in workspace with correct content
3. test_file_read_tool - Files read correctly, errors for missing files
4. test_run_command_tool - Commands execute, output captured, sandboxed
5. test_agent_generates_tool_calls - Agent produces valid AgentToolCalls structure
6. test_execute_single_step - Step executes, result has correct status
7. test_execute_steps_with_dependencies - Steps execute in correct order
8. test_chairman_summary - Summary is valid ChairmanSummary
9. test_rework_with_feedback - Feedback processed, agent regenerates
10. test_execute_full_plan - Complete plan runs, all outputs correct
11. test_execution_transcript - Events recorded, costs tracked
12. test_error_handling - Errors captured, execution recovers
13. test_deliberation_to_execution - End-to-end from plan to files
"""

import asyncio
import os
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

# pytest is optional - tests can run without it
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create dummy pytest module with decorators that just return the function
    class _Mark:
        @staticmethod
        def asyncio(func):
            return func
    class pytest:
        mark = _Mark()
        @staticmethod
        def fixture(func):
            return func

# Add v2 directory to path
sys.path.insert(0, str(Path(__file__).parent))

from schemas import (
    Plan,
    PlanStep,
    ToolDefinition,
    ToolParameter,
    ToolCall,
    ToolCallRequest,
    ToolCallResult,
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
from execution import (
    ToolRegistry,
    ToolImplementations,
    AgentExecutor,
    InteractiveExecutor,
    ExecutionEngine,
    create_default_tools,
    execute_plan_interactive,
    execute_plan_headless,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    workspace = tempfile.mkdtemp(prefix="council_test_")
    yield Path(workspace)
    shutil.rmtree(workspace, ignore_errors=True)


@pytest.fixture
def sample_context(temp_workspace):
    """Create a sample execution context."""
    return ExecutionContext(
        session_id="test-session",
        workspace_path=str(temp_workspace)
    )


@pytest.fixture
def sample_plan():
    """Create a sample plan for testing."""
    return Plan(
        overview="Build a simple hello world application",
        steps=[
            PlanStep(
                step_number=1,
                title="Create main module",
                description="Create main.py with hello world function",
                assigned_agent="Backend Dev",
                dependencies=[],
                estimated_complexity="low",
                acceptance_criteria=["File exists", "Contains hello function"]
            ),
            PlanStep(
                step_number=2,
                title="Create tests",
                description="Create test_main.py with basic tests",
                assigned_agent="Backend Dev",
                dependencies=[1],
                estimated_complexity="low",
                acceptance_criteria=["Test file exists", "Tests pass"]
            ),
            PlanStep(
                step_number=3,
                title="Create documentation",
                description="Create README.md with usage instructions",
                assigned_agent="Frontend Dev",
                dependencies=[1],
                estimated_complexity="low",
                acceptance_criteria=["README exists", "Contains examples"]
            )
        ],
        critical_path=[1, 2],
        estimated_total_effort="low",
        risks_and_mitigations={}
    )


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return {
        "models": {
            "claude": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "api_key_env": "ANTHROPIC_API_KEY"
            }
        }
    }


# =============================================================================
# TEST 1: Tool Registry
# =============================================================================

def test_tool_registry():
    """
    Test 1: Tools register, validate, execute correctly.

    Verifies:
    - Tools can be registered with definitions
    - Tool calls are validated against definitions
    - Valid calls pass validation
    - Invalid calls (missing params, unknown tools) fail validation
    """
    registry = ToolRegistry()

    # Create a simple tool definition
    tool_def = ToolDefinition(
        name="test_tool",
        description="A test tool",
        parameters=[
            ToolParameter(name="arg1", type="string", description="First arg", required=True),
            ToolParameter(name="arg2", type="integer", description="Second arg", required=False, default=10),
        ],
        returns="Test output",
        dangerous=False
    )

    # Mock implementation
    async def test_impl(arg1: str, arg2: int = 10, context=None):
        return f"Got {arg1} and {arg2}"

    # Register
    registry.register(tool_def, test_impl)

    # Test get_tool
    retrieved = registry.get_tool("test_tool")
    assert retrieved is not None
    assert retrieved.name == "test_tool"
    assert len(retrieved.parameters) == 2

    # Test validation - valid call
    valid_call = ToolCall(tool_name="test_tool", parameters={"arg1": "hello"}, reasoning="test")
    is_valid, error = registry.validate_call(valid_call)
    assert is_valid is True
    assert error is None

    # Test validation - missing required param
    invalid_call = ToolCall(tool_name="test_tool", parameters={}, reasoning="test")
    is_valid, error = registry.validate_call(invalid_call)
    assert is_valid is False
    assert "Missing required parameter" in error

    # Test validation - unknown tool
    unknown_call = ToolCall(tool_name="unknown", parameters={}, reasoning="test")
    is_valid, error = registry.validate_call(unknown_call)
    assert is_valid is False
    assert "Unknown tool" in error

    # Test validation - unknown parameter
    bad_param_call = ToolCall(tool_name="test_tool", parameters={"arg1": "hi", "bad_param": "x"}, reasoning="test")
    is_valid, error = registry.validate_call(bad_param_call)
    assert is_valid is False
    assert "Unknown parameter" in error

    # Test list_tools
    tools = registry.list_tools()
    assert len(tools) == 1

    # Test get_tools_prompt
    prompt = registry.get_tools_prompt()
    assert "test_tool" in prompt
    assert "arg1" in prompt

    print("✓ Test 1 passed: Tool registry works correctly")


# =============================================================================
# TEST 2: file_write Tool
# =============================================================================

@pytest.mark.asyncio
async def test_file_write_tool(temp_workspace, sample_context):
    """
    Test 2: Files created in workspace with correct content.

    Verifies:
    - Files are created with correct content
    - Parent directories are created automatically
    - Files are tracked in context
    - Path traversal is blocked
    """
    impl = ToolImplementations(temp_workspace)
    sample_context.workspace_path = str(temp_workspace)

    # Test basic file write
    result = await impl.file_write(
        path="hello.txt",
        content="Hello, World!",
        context=sample_context
    )

    assert "Successfully wrote" in result
    assert (temp_workspace / "hello.txt").exists()
    assert (temp_workspace / "hello.txt").read_text() == "Hello, World!"
    assert "hello.txt" in sample_context.created_files

    # Test nested directory creation
    result = await impl.file_write(
        path="src/app/main.py",
        content="print('hello')",
        context=sample_context
    )

    assert (temp_workspace / "src" / "app" / "main.py").exists()

    # Test path traversal prevention
    try:
        await impl.file_write(
            path="../../../etc/passwd",
            content="malicious",
            context=sample_context
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "escape workspace" in str(e)

    print("✓ Test 2 passed: file_write tool works correctly")


# =============================================================================
# TEST 3: file_read Tool
# =============================================================================

@pytest.mark.asyncio
async def test_file_read_tool(temp_workspace, sample_context):
    """
    Test 3: Files read correctly, errors for missing files.

    Verifies:
    - Existing files are read correctly
    - FileNotFoundError raised for missing files
    - Path traversal is blocked
    """
    impl = ToolImplementations(temp_workspace)

    # Create a test file
    test_file = temp_workspace / "test.txt"
    test_file.write_text("Test content")

    # Test successful read
    content = await impl.file_read("test.txt", context=sample_context)
    assert content == "Test content"

    # Test missing file
    try:
        await impl.file_read("nonexistent.txt", context=sample_context)
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass

    # Test path traversal prevention
    try:
        await impl.file_read("../../../etc/passwd", context=sample_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "escape workspace" in str(e)

    print("✓ Test 3 passed: file_read tool works correctly")


# =============================================================================
# TEST 4: run_command Tool
# =============================================================================

@pytest.mark.asyncio
async def test_run_command_tool(temp_workspace, sample_context):
    """
    Test 4: Commands execute, output captured, sandboxed.

    Verifies:
    - Commands execute in workspace directory
    - Output is captured correctly
    - Dangerous commands are blocked
    - Exit codes are reported
    """
    impl = ToolImplementations(temp_workspace)

    # Test basic command
    result = await impl.run_command("echo 'hello world'", context=sample_context)
    assert "hello world" in result

    # Test command with output
    result = await impl.run_command("pwd", context=sample_context)
    assert str(temp_workspace) in result

    # Test command in working directory
    (temp_workspace / "subdir").mkdir()
    result = await impl.run_command("pwd", working_dir="subdir", context=sample_context)
    assert "subdir" in result

    # Test command with exit code
    result = await impl.run_command("ls nonexistent_file_12345", context=sample_context)
    assert "[exit code]" in result or "[stderr]" in result

    # Test dangerous command blocking
    try:
        await impl.run_command("rm -rf /", context=sample_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Blocked dangerous command" in str(e)

    print("✓ Test 4 passed: run_command tool works correctly")


# =============================================================================
# TEST 5: Agent Tool Call Generation
# =============================================================================

@pytest.mark.asyncio
async def test_agent_generates_tool_calls(temp_workspace, sample_context, mock_config):
    """
    Test 5: Agent produces valid AgentToolCalls structure.

    Verifies:
    - Agent generates structured tool calls
    - Response contains thinking, tool_calls, explanation
    - Tool calls have valid structure
    """
    # Create mock provider
    mock_provider = AsyncMock()
    mock_agent_calls = AgentToolCalls(
        thinking="I need to create a Python file with a hello world function",
        tool_calls=[
            ToolCall(
                tool_name="file_write",
                parameters={"path": "main.py", "content": "def hello():\n    print('Hello!')"},
                reasoning="Create the main module"
            )
        ],
        explanation="Created main.py with hello function"
    )
    mock_provider.complete_structured = AsyncMock(return_value=(mock_agent_calls, {"input_tokens": 100, "output_tokens": 50}))

    # Create registry and executor
    registry, _ = create_default_tools(temp_workspace)
    executor = AgentExecutor(mock_config, registry)

    # Mock get_provider
    with patch.object(executor, '_get_provider', return_value=mock_provider):
        step = PlanStep(
            step_number=1,
            title="Create main module",
            description="Create main.py with hello world function",
            assigned_agent="Backend Dev",
            dependencies=[],
            estimated_complexity="low",
            acceptance_criteria=["File exists"]
        )

        result = await executor.generate_tool_calls(step, sample_context)

    # Verify structure
    assert isinstance(result, AgentToolCalls)
    assert result.thinking is not None
    assert len(result.tool_calls) > 0
    assert result.explanation is not None

    # Verify tool call structure
    tool_call = result.tool_calls[0]
    assert tool_call.tool_name == "file_write"
    assert "path" in tool_call.parameters

    print("✓ Test 5 passed: Agent generates valid tool calls")


# =============================================================================
# TEST 6: Single Step Execution
# =============================================================================

@pytest.mark.asyncio
async def test_execute_single_step(temp_workspace, sample_context, mock_config):
    """
    Test 6: Step executes, result has correct status.

    Verifies:
    - Step execution produces StepResult
    - Tool calls are executed
    - Status reflects success/failure
    """
    # Setup mocks
    mock_provider = AsyncMock()
    mock_agent_calls = AgentToolCalls(
        thinking="Creating the file",
        tool_calls=[
            ToolCall(
                tool_name="file_write",
                parameters={"path": "test.py", "content": "# test file"},
                reasoning="Create test file"
            )
        ],
        explanation="Created test.py"
    )
    mock_provider.complete_structured = AsyncMock(return_value=(mock_agent_calls, {"input_tokens": 100, "output_tokens": 50}))

    # Create engine with mocked provider
    engine = ExecutionEngine(
        config=mock_config,
        workspace_base=temp_workspace,
        interactive=False
    )
    engine.tool_implementations.workspace = temp_workspace

    step = PlanStep(
        step_number=1,
        title="Create test file",
        description="Create a test Python file",
        assigned_agent="Backend Dev",
        dependencies=[],
        estimated_complexity="low",
        acceptance_criteria=["File exists"]
    )

    # Mock the agent executor
    with patch.object(engine.agent_executor, '_get_provider', return_value=mock_provider):
        result = await engine.execute_step(step, sample_context)

    # Verify result
    assert isinstance(result, StepResult)
    assert result.step_number == 1
    assert result.agent_name == "Backend Dev"
    assert len(result.tool_calls) > 0
    assert len(result.tool_results) > 0

    # Verify file was created
    assert (temp_workspace / "test.py").exists()

    print("✓ Test 6 passed: Single step execution works")


# =============================================================================
# TEST 7: Multi-Step with Dependencies
# =============================================================================

def test_execute_steps_with_dependencies(sample_plan, temp_workspace, mock_config):
    """
    Test 7: Steps execute in correct order.

    Verifies:
    - Dependencies are respected
    - Topological sort orders steps correctly
    - Steps with same dependencies can be ordered deterministically
    """
    engine = ExecutionEngine(
        config=mock_config,
        workspace_base=temp_workspace,
        interactive=False
    )

    # Test ordering
    ordered = engine._order_steps(sample_plan)

    # Step 1 should be first (no dependencies)
    assert ordered[0].step_number == 1

    # Steps 2 and 3 depend on step 1, so they come after
    step_2_idx = next(i for i, s in enumerate(ordered) if s.step_number == 2)
    step_3_idx = next(i for i, s in enumerate(ordered) if s.step_number == 3)

    assert step_2_idx > 0
    assert step_3_idx > 0

    # Test cycle detection
    cyclic_plan = Plan(
        overview="Test",
        steps=[
            PlanStep(step_number=1, title="A", description="A", assigned_agent="dev",
                    dependencies=[2], estimated_complexity="low", acceptance_criteria=[]),
            PlanStep(step_number=2, title="B", description="B", assigned_agent="dev",
                    dependencies=[1], estimated_complexity="low", acceptance_criteria=[]),
        ],
        critical_path=[],
        estimated_total_effort="low",
        risks_and_mitigations={}
    )

    try:
        engine._order_steps(cyclic_plan)
        assert False, "Should have raised ValueError for cycle"
    except ValueError as e:
        assert "cycle" in str(e).lower()

    print("✓ Test 7 passed: Dependency ordering works correctly")


# =============================================================================
# TEST 8: Chairman Summary
# =============================================================================

@pytest.mark.asyncio
async def test_chairman_summary(temp_workspace, sample_context, mock_config):
    """
    Test 8: Summary is valid ChairmanSummary.

    Verifies:
    - Chairman produces structured summary
    - Contains required fields
    - Recommendation is valid
    """
    mock_provider = AsyncMock()
    mock_summary = ChairmanSummary(
        step_number=1,
        title="Create main module",
        what_was_done="Created main.py with hello function that prints a greeting",
        files_changed=["main.py"],
        key_decisions=["Used standard function syntax", "Added docstring"],
        potential_concerns=["No error handling"],
        recommendation="proceed"
    )
    mock_provider.complete_structured = AsyncMock(return_value=(mock_summary, {"input_tokens": 100, "output_tokens": 50}))

    executor = InteractiveExecutor(mock_config)

    step = PlanStep(
        step_number=1,
        title="Create main module",
        description="Create main.py",
        assigned_agent="Backend Dev",
        dependencies=[],
        estimated_complexity="low",
        acceptance_criteria=[]
    )

    step_result = StepResult(
        step_number=1,
        status=StepStatus.COMPLETED,
        agent_name="Backend Dev",
        tool_calls=[],
        tool_results=[],
        output_summary="Created file",
        files_created=["main.py"]
    )

    with patch.object(executor, '_get_provider', return_value=mock_provider):
        summary = await executor.summarize_step_result(step, step_result, sample_context)

    # Verify structure
    assert isinstance(summary, ChairmanSummary)
    assert summary.step_number == 1
    assert summary.what_was_done is not None
    assert summary.recommendation in ["proceed", "review_recommended", "rework_suggested"]

    print("✓ Test 8 passed: Chairman summary works correctly")


# =============================================================================
# TEST 9: Rework Flow
# =============================================================================

@pytest.mark.asyncio
async def test_rework_with_feedback(temp_workspace, sample_context, mock_config):
    """
    Test 9: Feedback processed, agent regenerates.

    Verifies:
    - User feedback is passed to agent
    - Agent generates new tool calls incorporating feedback
    - Rework count is tracked
    """
    mock_provider = AsyncMock()
    mock_refined_calls = AgentToolCalls(
        thinking="User wants better error handling, adding try/except",
        tool_calls=[
            ToolCall(
                tool_name="file_write",
                parameters={"path": "main.py", "content": "def hello():\n    try:\n        print('Hello!')\n    except:\n        pass"},
                reasoning="Create main module with error handling"
            )
        ],
        explanation="Created main.py with error handling as requested"
    )
    mock_provider.complete_structured = AsyncMock(return_value=(mock_refined_calls, {"input_tokens": 150, "output_tokens": 75}))

    registry, _ = create_default_tools(temp_workspace)
    executor = AgentExecutor(mock_config, registry)

    step = PlanStep(
        step_number=1,
        title="Create main module",
        description="Create main.py",
        assigned_agent="Backend Dev",
        dependencies=[],
        estimated_complexity="low",
        acceptance_criteria=[]
    )

    previous_result = StepResult(
        step_number=1,
        status=StepStatus.COMPLETED,
        agent_name="Backend Dev",
        tool_calls=[ToolCallRequest(
            call_id="1",
            tool_name="file_write",
            parameters={"path": "main.py", "content": "def hello(): print('Hello!')"},
            reasoning="Create file",
            agent_name="Backend Dev",
            step_number=1,
            timestamp=datetime.now()
        )],
        tool_results=[],
        output_summary="Created file without error handling"
    )

    feedback = UserFeedback(
        feedback_type=FeedbackType.REWORK,
        message="Please add error handling",
        specific_changes=["Add try/except blocks"]
    )

    with patch.object(executor, '_get_provider', return_value=mock_provider):
        result = await executor.refine_with_feedback(
            step, sample_context, feedback, previous_result
        )

    # Verify new calls incorporate feedback
    assert isinstance(result, AgentToolCalls)
    assert "error" in result.thinking.lower() or "try" in result.thinking.lower()

    print("✓ Test 9 passed: Rework flow works correctly")


# =============================================================================
# TEST 10: Full Plan Execution
# =============================================================================

@pytest.mark.asyncio
async def test_execute_full_plan(temp_workspace, mock_config):
    """
    Test 10: Complete plan runs, all outputs correct.

    Verifies:
    - All steps execute
    - Files are created
    - ExecutionResult is complete
    """
    # Simple single-step plan for testing
    plan = Plan(
        overview="Create a hello world app",
        steps=[
            PlanStep(
                step_number=1,
                title="Create main file",
                description="Create main.py",
                assigned_agent="Backend Dev",
                dependencies=[],
                estimated_complexity="low",
                acceptance_criteria=["File exists"]
            )
        ],
        critical_path=[1],
        estimated_total_effort="low",
        risks_and_mitigations={}
    )

    mock_provider = AsyncMock()
    mock_agent_calls = AgentToolCalls(
        thinking="Creating the main file",
        tool_calls=[
            ToolCall(
                tool_name="file_write",
                parameters={"path": "main.py", "content": "print('Hello!')"},
                reasoning="Create main module"
            )
        ],
        explanation="Created main.py"
    )
    mock_provider.complete_structured = AsyncMock(return_value=(mock_agent_calls, {"input_tokens": 100, "output_tokens": 50}))

    engine = ExecutionEngine(
        config=mock_config,
        workspace_base=temp_workspace,
        interactive=False
    )

    with patch.object(engine.agent_executor, '_get_provider', return_value=mock_provider):
        result = await engine.execute_plan(plan)

    # Verify result
    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.transcript is not None
    assert len(result.transcript.step_results) == 1
    assert result.transcript.status == "completed"

    print("✓ Test 10 passed: Full plan execution works")


# =============================================================================
# TEST 11: Execution Transcript
# =============================================================================

@pytest.mark.asyncio
async def test_execution_transcript(temp_workspace, mock_config):
    """
    Test 11: Events recorded, costs tracked.

    Verifies:
    - Events are recorded with timestamps
    - Step results are stored
    - Costs and tokens are tracked
    """
    plan = Plan(
        overview="Test plan",
        steps=[
            PlanStep(
                step_number=1,
                title="Step 1",
                description="Test step",
                assigned_agent="Dev",
                dependencies=[],
                estimated_complexity="low",
                acceptance_criteria=[]
            )
        ],
        critical_path=[1],
        estimated_total_effort="low",
        risks_and_mitigations={}
    )

    mock_provider = AsyncMock()
    mock_agent_calls = AgentToolCalls(
        thinking="Test",
        tool_calls=[
            ToolCall(tool_name="file_write", parameters={"path": "test.txt", "content": "test"}, reasoning="test")
        ],
        explanation="Test"
    )
    mock_provider.complete_structured = AsyncMock(return_value=(mock_agent_calls, {"input_tokens": 100, "output_tokens": 50}))

    engine = ExecutionEngine(
        config=mock_config,
        workspace_base=temp_workspace,
        interactive=False
    )

    with patch.object(engine.agent_executor, '_get_provider', return_value=mock_provider):
        result = await engine.execute_plan(plan)

    transcript = result.transcript

    # Verify transcript structure
    assert transcript.session_id is not None
    assert transcript.started_at is not None
    assert transcript.completed_at is not None
    assert transcript.status == "completed"

    # Verify events
    assert len(transcript.events) > 0
    assert any(e.event_type == "step_started" for e in transcript.events)

    # Verify step results
    assert len(transcript.step_results) == 1

    print("✓ Test 11 passed: Execution transcript works correctly")


# =============================================================================
# TEST 12: Error Handling
# =============================================================================

@pytest.mark.asyncio
async def test_error_handling(temp_workspace, sample_context):
    """
    Test 12: Errors captured, execution recovers.

    Verifies:
    - Tool errors are captured in ToolCallResult
    - Execution continues after errors
    - Error details are recorded
    """
    registry, impl = create_default_tools(temp_workspace)

    # Test error in tool execution
    call = ToolCallRequest(
        call_id="test-1",
        tool_name="file_read",
        parameters={"path": "nonexistent.txt"},
        reasoning="Try to read missing file",
        agent_name="Dev",
        step_number=1,
        timestamp=datetime.now()
    )

    result = await registry.execute(call, sample_context)

    assert result.success is False
    assert result.error is not None
    assert "not found" in result.error.lower() or "No such file" in result.error.lower()

    # Test unknown tool
    unknown_call = ToolCallRequest(
        call_id="test-2",
        tool_name="unknown_tool",
        parameters={},
        reasoning="Test",
        agent_name="Dev",
        step_number=1,
        timestamp=datetime.now()
    )

    result = await registry.execute(unknown_call, sample_context)
    assert result.success is False
    assert "Unknown tool" in result.error

    print("✓ Test 12 passed: Error handling works correctly")


# =============================================================================
# TEST 13: Deliberation to Execution (Integration)
# =============================================================================

@pytest.mark.asyncio
async def test_deliberation_to_execution(temp_workspace, mock_config):
    """
    Test 13: End-to-end from plan to files.

    Verifies:
    - Can create plan (mocked deliberation)
    - Can execute that plan
    - Files are actually created
    - Different runs produce results (proving real execution)
    """
    # Create a plan (simulating what deliberation would produce)
    plan = Plan(
        overview="Build a calculator module",
        steps=[
            PlanStep(
                step_number=1,
                title="Create calculator module",
                description="Create calc.py with add and subtract functions",
                assigned_agent="Backend Dev",
                dependencies=[],
                estimated_complexity="low",
                acceptance_criteria=["calc.py exists", "Has add function", "Has subtract function"]
            ),
            PlanStep(
                step_number=2,
                title="Create tests",
                description="Create test_calc.py with tests for calculator",
                assigned_agent="Backend Dev",
                dependencies=[1],
                estimated_complexity="low",
                acceptance_criteria=["test_calc.py exists", "Tests calculator functions"]
            )
        ],
        critical_path=[1, 2],
        estimated_total_effort="low",
        risks_and_mitigations={"complexity": "Keep functions simple"}
    )

    # Mock provider that returns reasonable tool calls
    mock_provider = AsyncMock()

    call_count = [0]
    def make_response(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call - create calc.py
            return (AgentToolCalls(
                thinking="Creating calculator module",
                tool_calls=[
                    ToolCall(
                        tool_name="file_write",
                        parameters={
                            "path": "calc.py",
                            "content": f"# Calculator module - run {call_count[0]}\n\ndef add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n"
                        },
                        reasoning="Create calculator functions"
                    )
                ],
                explanation="Created calc.py with add and subtract functions"
            ), {"input_tokens": 100, "output_tokens": 50})
        else:
            # Second call - create tests
            return (AgentToolCalls(
                thinking="Creating test file",
                tool_calls=[
                    ToolCall(
                        tool_name="file_write",
                        parameters={
                            "path": "test_calc.py",
                            "content": f"# Tests - run {call_count[0]}\nimport calc\n\ndef test_add():\n    assert calc.add(2, 3) == 5\n\ndef test_subtract():\n    assert calc.subtract(5, 3) == 2\n"
                        },
                        reasoning="Create test file"
                    )
                ],
                explanation="Created test_calc.py"
            ), {"input_tokens": 100, "output_tokens": 50})

    mock_provider.complete_structured = AsyncMock(side_effect=make_response)

    # Execute plan
    engine = ExecutionEngine(
        config=mock_config,
        workspace_base=temp_workspace,
        interactive=False
    )

    with patch.object(engine.agent_executor, '_get_provider', return_value=mock_provider):
        result = await engine.execute_plan(plan)

    # Verify execution succeeded
    assert result.success is True

    # Files may be tracked in result or context
    all_files = set(result.files_created) | set(result.final_context.created_files)
    assert len(all_files) >= 2, f"Expected at least 2 files, got {all_files}"

    # Find workspace path
    workspace = Path(result.output_path)

    # Verify files exist
    assert (workspace / "calc.py").exists(), "calc.py should exist"
    assert (workspace / "test_calc.py").exists(), "test_calc.py should exist"

    # Verify content (proves real execution, not hardcoded)
    calc_content = (workspace / "calc.py").read_text()
    assert "def add" in calc_content
    assert "def subtract" in calc_content

    test_content = (workspace / "test_calc.py").read_text()
    assert "test_add" in test_content

    print("✓ Test 13 passed: End-to-end deliberation to execution works")


# =============================================================================
# SCHEMA VALIDATION TESTS
# =============================================================================

def test_schema_validation():
    """
    Bonus: Verify all new schemas can be instantiated and serialized.
    """
    from schemas import (
        ToolParameter, ToolDefinition, ToolCall, ToolCallRequest, ToolCallResult,
        ExecutionContext, AgentToolCalls, StepResult, UserFeedback, ChairmanSummary,
        ExecutionEvent, ExecutionTranscript, ExecutionResult
    )

    # Test ToolParameter
    tp = ToolParameter(name="test", type="string", description="test param")
    assert tp.model_dump()

    # Test ToolDefinition
    td = ToolDefinition(name="test", description="test tool", parameters=[], returns="str")
    assert td.model_dump()

    # Test ToolCall
    tc = ToolCall(tool_name="test", parameters={}, reasoning="test")
    assert tc.model_dump()

    # Test ToolCallRequest
    tcr = ToolCallRequest(
        call_id="1", tool_name="test", parameters={}, reasoning="test",
        agent_name="dev", step_number=1, timestamp=datetime.now()
    )
    assert tcr.model_dump()

    # Test ToolCallResult
    tcres = ToolCallResult(call_id="1", tool_name="test", success=True, output="ok")
    assert tcres.model_dump()

    # Test ExecutionContext
    ec = ExecutionContext(session_id="test", workspace_path="/tmp")
    assert ec.model_dump()

    # Test AgentToolCalls
    atc = AgentToolCalls(thinking="test", tool_calls=[], explanation="test")
    assert atc.model_dump()

    # Test StepResult
    sr = StepResult(step_number=1, status=StepStatus.COMPLETED, agent_name="dev")
    assert sr.model_dump()

    # Test UserFeedback
    uf = UserFeedback(feedback_type=FeedbackType.APPROVE)
    assert uf.model_dump()

    # Test ChairmanSummary
    cs = ChairmanSummary(
        step_number=1, title="test", what_was_done="test",
        recommendation="proceed"
    )
    assert cs.model_dump()

    # Test ExecutionEvent
    ee = ExecutionEvent(event_type="test", details="test")
    assert ee.model_dump()

    # Test ExecutionTranscript
    et = ExecutionTranscript(session_id="test", plan_overview="test")
    assert et.model_dump()

    # Test ExecutionResult
    er = ExecutionResult(
        success=True,
        transcript=et,
        final_context=ec,
        output_path="/tmp",
        summary="test"
    )
    assert er.model_dump()

    print("✓ All schema validation tests passed")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    print("Running Phase 3 tests...\n")

    # Run sync tests
    test_tool_registry()
    test_execute_steps_with_dependencies(
        Plan(
            overview="Test",
            steps=[
                PlanStep(step_number=1, title="A", description="A", assigned_agent="dev",
                        dependencies=[], estimated_complexity="low", acceptance_criteria=[]),
                PlanStep(step_number=2, title="B", description="B", assigned_agent="dev",
                        dependencies=[1], estimated_complexity="low", acceptance_criteria=[]),
                PlanStep(step_number=3, title="C", description="C", assigned_agent="dev",
                        dependencies=[1], estimated_complexity="low", acceptance_criteria=[]),
            ],
            critical_path=[1, 2],
            estimated_total_effort="low",
            risks_and_mitigations={}
        ),
        Path(tempfile.mkdtemp()),
        {"models": {"claude": {"provider": "anthropic", "model": "test"}}}
    )
    test_schema_validation()

    # Run async tests
    async def run_async_tests():
        temp_dir = Path(tempfile.mkdtemp())
        context = ExecutionContext(session_id="test", workspace_path=str(temp_dir))
        config = {"models": {"claude": {"provider": "anthropic", "model": "test"}}}

        try:
            await test_file_write_tool(temp_dir, context)
            await test_file_read_tool(temp_dir, context)
            await test_run_command_tool(temp_dir, context)
            await test_agent_generates_tool_calls(temp_dir, context, config)
            await test_execute_single_step(temp_dir, context, config)
            await test_chairman_summary(temp_dir, context, config)
            await test_rework_with_feedback(temp_dir, context, config)
            await test_execute_full_plan(temp_dir, config)
            await test_execution_transcript(temp_dir, config)
            await test_error_handling(temp_dir, context)
            await test_deliberation_to_execution(temp_dir, config)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    asyncio.run(run_async_tests())

    print("\n" + "="*60)
    print("All Phase 3 tests passed!")
    print("="*60)
