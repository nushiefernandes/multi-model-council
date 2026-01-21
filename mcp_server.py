#!/usr/bin/env python3
"""
MCP Server for Council v2 - Multi-model Deliberation.

IMPORTANT: Never print to stdout - it breaks JSON-RPC communication.
All logging must go to stderr.

This server exposes Council v2's deliberation capabilities via MCP tools:
- council_quick: Fast multi-model consensus
- council_session_start: Start a session (roles + plan)
- council_session_execute: Execute a session's plan
- council_session_resume: Resume an interrupted session
- council_session_list: List past sessions
- council_review: Structured code review

To run:
    python mcp_server.py

To register with Claude Code:
    claude mcp add council -- python /Users/anushfernandes/.council/v2/mcp_server.py
"""

import sys
import json
import logging
from pathlib import Path

# CRITICAL: Configure logging to stderr BEFORE any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("council-mcp")

# Add council v2 to path for imports
COUNCIL_PATH = Path(__file__).parent
sys.path.insert(0, str(COUNCIL_PATH))

# MCP imports
from mcp.server.fastmcp import FastMCP

# Council v2 imports (after path setup)
from deliberation import (
    Deliberation,
    DeliberationConfig,
    quick_ask,
)
from execution import execute_plan_headless
from session import (
    Session,
    SessionManager,
    WorkflowPhase,
    transition_session,
    get_next_phase,
)
from config import load_config

# Create MCP server
mcp = FastMCP("council")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parse_models(models_str: str) -> list[str]:
    """Parse comma-separated models string into a list."""
    return [m.strip() for m in models_str.split(",") if m.strip()]


def _get_config():
    """Load Council configuration."""
    try:
        config_obj = load_config()
        return config_obj.to_provider_config(), config_obj
    except Exception as e:
        logger.warning(f"Failed to load config, using defaults: {e}")
        from deliberation import create_default_config
        return create_default_config(), None


# =============================================================================
# MCP TOOLS
# =============================================================================

@mcp.tool()
async def council_quick(
    question: str,
    models: str = "claude"
) -> str:
    """
    Get quick multi-model consensus on a question.

    Fast way to get agreement from multiple AI models without going through
    the full deliberation workflow. Good for simple technical decisions.

    Args:
        question: The question to get consensus on (e.g., "Redis vs Memcached for sessions?")
        models: Comma-separated list of models to consult (default: "claude")

    Returns:
        JSON with the synthesized answer, confidence (1-10), agreement level, and dissenting views.
    """
    logger.info(f"council_quick called: {question[:50]}...")

    try:
        model_list = _parse_models(models)
        result = await quick_ask(question, models=model_list)
        return result.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"council_quick failed: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def council_session_start(
    task: str,
    models: str = "claude",
    skip_roles: bool = False
) -> str:
    """
    Start a new Council session for a task.

    Performs role deliberation (assigns models to roles like Architect, Backend Dev, etc.)
    and creates an implementation plan. Does NOT execute the plan - use council_session_execute
    for that.

    Args:
        task: The task description (e.g., "Build a REST API for user management")
        models: Comma-separated list of models to use for deliberation
        skip_roles: If true, skip role deliberation and use default assignments

    Returns:
        JSON with session_id, role_assignments, and plan. Use session_id to execute or resume.
    """
    logger.info(f"council_session_start: {task[:50]}...")

    try:
        model_list = _parse_models(models)
        config, config_obj = _get_config()

        # Create deliberation config
        chairman = model_list[0] if model_list else "claude"
        delib_config = DeliberationConfig(
            chairman_model=chairman,
            proposal_models=model_list,
            critique_models=model_list,
        )

        # Create session
        manager = SessionManager()
        session = manager.create(task, config=config)
        logger.info(f"Created session: {session.session_id}")

        # Initialize engine
        engine = Deliberation(config, delib_config)

        # Role deliberation
        role_result = None
        if not skip_roles:
            transition_session(session, WorkflowPhase.ROLE_DELIBERATION)
            manager.save(session)

            role_result = await engine.deliberate_roles(task)
            session.set_role_assignments(role_result)
            manager.save(session)
            logger.info("Role deliberation complete")

        # Planning
        transition_session(session, WorkflowPhase.PLANNING)
        manager.save(session)

        plan = await engine.deliberate_plan(
            task,
            role_assignments=role_result
        )
        session.set_plan(plan)
        manager.save(session)
        logger.info(f"Plan created with {plan.total_steps} steps")

        # Build response
        response = {
            "session_id": session.session_id,
            "task": task,
            "phase": session.phase,
            "role_assignments": role_result.model_dump() if role_result else None,
            "plan": {
                "overview": plan.overview,
                "total_steps": plan.total_steps,
                "steps": [
                    {
                        "step_number": s.step_number,
                        "title": s.title,
                        "assigned_agent": s.assigned_agent,
                        "complexity": s.estimated_complexity,
                    }
                    for s in plan.steps
                ],
                "estimated_effort": plan.estimated_total_effort,
            },
        }

        return json.dumps(response, indent=2, default=str)

    except Exception as e:
        logger.error(f"council_session_start failed: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def council_session_execute(
    session_id: str
) -> str:
    """
    Execute a Council session's plan.

    Takes a session that has been started (has a plan) and executes it.
    Files are created in the workspace directory. All checkpoints are auto-approved.

    Args:
        session_id: The session ID from council_session_start

    Returns:
        JSON with success status, files_created, output_path, and summary.
    """
    logger.info(f"council_session_execute: {session_id}")

    try:
        manager = SessionManager()
        session = manager.load(session_id)

        # Get plan
        plan = session.get_plan()
        if not plan:
            return json.dumps({"error": "Session has no plan. Call council_session_start first."})

        # Get config
        config, config_obj = _get_config()
        workspace_base = Path.home() / ".council" / "workspaces"

        # Transition to execution
        if session.phase != WorkflowPhase.EXECUTION:
            transition_session(session, WorkflowPhase.EXECUTION)
            manager.save(session)

        # Execute headless (no interactive checkpoints)
        exec_result = await execute_plan_headless(
            plan=plan,
            config=config,
            workspace_base=workspace_base
        )

        # Update session
        session.set_execution_result(exec_result)

        if exec_result.success:
            transition_session(session, WorkflowPhase.COMPLETE)
        else:
            transition_session(session, WorkflowPhase.FAILED)

        manager.save(session)

        # Build response
        response = {
            "session_id": session_id,
            "success": exec_result.success,
            "files_created": exec_result.files_created,
            "output_path": str(exec_result.output_path),
            "summary": exec_result.summary,
            "phase": session.phase,
            "total_tokens": exec_result.transcript.total_tokens if exec_result.transcript else 0,
            "total_cost": exec_result.transcript.total_cost if exec_result.transcript else 0,
        }

        return json.dumps(response, indent=2, default=str)

    except FileNotFoundError:
        return json.dumps({"error": f"Session not found: {session_id}"})
    except Exception as e:
        logger.error(f"council_session_execute failed: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def council_session_resume(
    session_id: str
) -> str:
    """
    Resume an interrupted Council session.

    Continues from where the session stopped. If planning was complete, starts execution.

    Args:
        session_id: The session ID to resume

    Returns:
        JSON with updated session state and results.
    """
    logger.info(f"council_session_resume: {session_id}")

    try:
        manager = SessionManager()
        session = manager.load(session_id)

        # Check if session can be resumed
        if session.phase in (WorkflowPhase.COMPLETE, WorkflowPhase.FAILED, WorkflowPhase.ABORTED):
            return json.dumps({
                "error": f"Session is {session.phase} and cannot be resumed",
                "session_id": session_id,
                "phase": session.phase,
            })

        # Determine what to do based on phase
        next_phase = get_next_phase(session)

        if next_phase == WorkflowPhase.EXECUTION or session.phase == WorkflowPhase.PLANNING:
            # Have a plan, execute it
            plan = session.get_plan()
            if plan:
                return await council_session_execute(session_id)

        # If we need planning, re-start the session
        return json.dumps({
            "message": "Session needs re-planning. Use council_session_start with the same task.",
            "session_id": session_id,
            "task": session.task,
            "phase": session.phase,
        })

    except FileNotFoundError:
        return json.dumps({"error": f"Session not found: {session_id}"})
    except Exception as e:
        logger.error(f"council_session_resume failed: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def council_session_list(
    include_completed: bool = True
) -> str:
    """
    List all Council sessions.

    Args:
        include_completed: If false, only shows resumable sessions (not complete/failed/aborted)

    Returns:
        JSON array of session summaries with session_id, task, phase, created_at, etc.
    """
    logger.info(f"council_session_list: include_completed={include_completed}")

    try:
        manager = SessionManager()

        if include_completed:
            sessions = manager.list_all()
        else:
            sessions = manager.list_resumable()

        return json.dumps(sessions, indent=2, default=str)

    except Exception as e:
        logger.error(f"council_session_list failed: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def council_review(
    code: str,
    context: str = ""
) -> str:
    """
    Perform a structured code review.

    Analyzes code for security, performance, correctness, maintainability, and style.
    Returns structured issues with severity, location, and suggested fixes.

    Args:
        code: The code to review
        context: Optional context about the code (purpose, requirements)

    Returns:
        JSON CodeReview with reviewer, summary, issues array, and approval status.
    """
    logger.info(f"council_review: {len(code)} characters")

    try:
        config, _ = _get_config()
        engine = Deliberation(config)

        review = await engine.quality_review(code, context=context)
        return review.model_dump_json(indent=2)

    except Exception as e:
        logger.error(f"council_review failed: {e}")
        return json.dumps({"error": str(e)})


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting Council MCP Server")
    mcp.run(transport="stdio")
