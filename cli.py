#!/usr/bin/env python3
"""
Council v2 CLI - Multi-model deliberation with structured outputs.

This is the main entry point for the Council v2 command-line interface.
It wraps the deliberation and execution engines with a rich terminal UI.

WHAT'S DIFFERENT FROM V1:
------------------------
V1's CLI used text parsing (brittle regex) to extract structured data.
V2's CLI uses validated Pydantic schemas from complete_structured(),
making it more reliable and type-safe.

USAGE:
------
  council "task description"      - Start a new session
  council --file prd.md           - Load task from file
  council --resume SESSION_ID     - Resume a session
  council --list                  - List past sessions
  council --quick "question"      - Quick consensus mode

WORKFLOW:
--------
  1. Role Deliberation - Assign roles to models
  2. Planning - Create implementation plan
  3. Execution - Execute the plan with checkpoints
  4. Review - Quality review of outputs
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Local imports
from deliberation import (
    Deliberation,
    DeliberationConfig,
    create_default_config,
    quick_ask,
)
from execution import (
    ExecutionEngine,
    execute_plan_interactive,
)
from schemas import (
    UserFeedback,
    FeedbackType,
    ChairmanSummary,
)
from session import (
    Session,
    SessionManager,
    WorkflowPhase,
    transition_session,
    get_next_phase,
)
import ui


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="council",
        description="Multi-model deliberation with structured outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  council "Build a REST API for user management"
  council --file prd.md
  council --quick "Redis vs Memcached for sessions?"
  council --list
  council --resume abc12345
        """
    )

    # Positional argument for direct task
    parser.add_argument(
        "task",
        nargs="?",
        help="Task description for the council to deliberate"
    )

    # File input
    parser.add_argument(
        "-f", "--file",
        type=Path,
        help="Load task from a file"
    )

    # Quick consensus mode
    parser.add_argument(
        "-q", "--quick",
        metavar="QUESTION",
        help="Quick consensus mode - get fast multi-model agreement"
    )

    # Session management
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List past sessions"
    )

    parser.add_argument(
        "-r", "--resume",
        metavar="SESSION_ID",
        help="Resume a previous session"
    )

    # Configuration
    parser.add_argument(
        "--models",
        nargs="+",
        default=["claude", "gpt4"],
        help="Models to use for deliberation (default: claude gpt4)"
    )

    parser.add_argument(
        "--chairman",
        default="claude",
        help="Model to use as chairman for synthesis (default: claude)"
    )

    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Run without interactive checkpoints (auto-approve all)"
    )

    parser.add_argument(
        "--skip-roles",
        action="store_true",
        help="Skip role deliberation phase (use defaults)"
    )

    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Stop after planning (don't execute)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Council v2.0.0-alpha"
    )

    return parser


# =============================================================================
# CHECKPOINT CALLBACK
# =============================================================================

def create_checkpoint_callback() -> callable:
    """
    Create a callback function for execution checkpoints.

    This callback is called after each step completes, allowing
    the user to approve, provide feedback, or abort.
    """
    def checkpoint_callback(summary: ChairmanSummary) -> UserFeedback:
        """Handle a checkpoint during execution."""
        choice = ui.prompt_checkpoint(summary)

        if choice == "approve":
            return UserFeedback(feedback_type=FeedbackType.APPROVE)

        elif choice == "feedback":
            feedback_text = ui.prompt_feedback()
            return UserFeedback(
                feedback_type=FeedbackType.MODIFY,
                message=feedback_text
            )

        elif choice == "rework":
            feedback_text = ui.prompt_feedback()
            return UserFeedback(
                feedback_type=FeedbackType.REWORK,
                message=feedback_text
            )

        elif choice == "skip":
            return UserFeedback(feedback_type=FeedbackType.SKIP)

        elif choice == "abort":
            return UserFeedback(feedback_type=FeedbackType.ABORT)

        # Default to approve
        return UserFeedback(feedback_type=FeedbackType.APPROVE)

    return checkpoint_callback


# =============================================================================
# WORKFLOW RUNNERS
# =============================================================================

async def run_quick_consensus(question: str, models: list[str]) -> None:
    """
    Run quick consensus mode.

    This is a fast way to get multi-model agreement on simple questions
    without going through the full deliberation workflow.
    """
    ui.show_header("Quick Consensus")
    ui.console.print(f"[bold]Question:[/bold] {question}\n")

    with ui.show_thinking("Gathering consensus..."):
        try:
            result = await quick_ask(question, models=models if models else None)
        except Exception as e:
            ui.show_error(f"Failed to get consensus: {e}")
            return

    ui.show_consensus_result(result, question)


async def run_session(
    task: str,
    config: dict,
    delib_config: DeliberationConfig,
    interactive: bool = True,
    skip_roles: bool = False,
    skip_execution: bool = False,
    verbose: bool = False
) -> Session:
    """
    Run a full deliberation session.

    This goes through all workflow phases:
    1. Role Deliberation
    2. Planning
    3. Execution
    4. Review

    Each phase has checkpoints where the user can approve or provide feedback.
    """
    # Create session
    manager = SessionManager()
    session = manager.create(task, config=config)

    ui.show_header("Council v2 Session", f"ID: {session.session_id}")
    ui.console.print(f"[bold]Task:[/bold] {task}\n")

    # Initialize deliberation engine
    engine = Deliberation(config, delib_config)

    try:
        # =====================================================================
        # PHASE 0: Role Deliberation
        # =====================================================================
        if not skip_roles:
            transition_session(session, WorkflowPhase.ROLE_DELIBERATION)
            manager.save(session)

            ui.show_section("Phase 0: Role Deliberation")
            ui.console.print("[dim]Deliberating role assignments...[/dim]\n")

            with ui.show_thinking("Models are proposing and critiquing..."):
                role_result = await engine.deliberate_roles(task)

            session.set_role_assignments(role_result)

            # Update costs from transcript
            transcript = engine.get_transcript()
            if transcript:
                session.update_costs(transcript.total_tokens, transcript.total_cost)

            ui.show_role_assignments(role_result)

            if interactive:
                if not ui.prompt_continue("Proceed with these role assignments?"):
                    transition_session(session, WorkflowPhase.ABORTED)
                    manager.save(session)
                    ui.show_warning("Session aborted by user")
                    return session

            manager.save(session)
        else:
            ui.show_info("Skipping role deliberation (using defaults)")

        # =====================================================================
        # PHASE 1: Planning
        # =====================================================================
        transition_session(session, WorkflowPhase.PLANNING)
        manager.save(session)

        ui.show_section("Phase 1: Planning")
        ui.console.print("[dim]Creating implementation plan...[/dim]\n")

        with ui.show_thinking("Architecting the plan..."):
            plan = await engine.deliberate_plan(
                task,
                role_assignments=session.get_role_assignments()
            )

        session.set_plan(plan)
        ui.show_plan(plan)

        if interactive:
            if not ui.prompt_continue("Proceed with this plan?"):
                transition_session(session, WorkflowPhase.ABORTED)
                manager.save(session)
                ui.show_warning("Session aborted by user")
                return session

        manager.save(session)

        if skip_execution:
            ui.show_info("Stopping after planning (--skip-execution)")
            transition_session(session, WorkflowPhase.COMPLETE)
            manager.save(session)
            return session

        # =====================================================================
        # PHASE 2: Execution
        # =====================================================================
        transition_session(session, WorkflowPhase.EXECUTION)
        manager.save(session)

        ui.show_section("Phase 2: Execution")
        ui.console.print(f"[dim]Executing {plan.total_steps} steps...[/dim]\n")

        # Create execution engine with checkpoint callback
        workspace_base = Path.home() / ".council" / "v2" / "workspaces"
        checkpoint_callback = create_checkpoint_callback() if interactive else None

        exec_result = await execute_plan_interactive(
            plan=plan,
            config=config,
            workspace_base=workspace_base,
            checkpoint_callback=checkpoint_callback
        )

        session.set_execution_result(exec_result)
        session.update_costs(
            exec_result.transcript.total_tokens,
            exec_result.transcript.total_cost
        )

        if not exec_result.success:
            ui.show_error(f"Execution failed: {exec_result.summary}")
            if exec_result.transcript.status == "aborted":
                transition_session(session, WorkflowPhase.ABORTED)
            else:
                transition_session(session, WorkflowPhase.FAILED)
            manager.save(session)
            return session

        ui.show_execution_result(exec_result)
        manager.save(session)

        # =====================================================================
        # PHASE 3: Review
        # =====================================================================
        transition_session(session, WorkflowPhase.REVIEW)
        manager.save(session)

        ui.show_section("Phase 3: Code Review")

        # Collect code from created files for review
        code_to_review = ""
        workspace = Path(exec_result.output_path)
        for file_path in exec_result.files_created[:5]:  # Review first 5 files
            full_path = workspace / file_path
            if full_path.exists() and full_path.suffix in (".py", ".js", ".ts", ".go", ".rs"):
                try:
                    code_to_review += f"\n### {file_path}\n```\n{full_path.read_text()}\n```\n"
                except Exception:
                    pass

        if code_to_review:
            ui.console.print("[dim]Reviewing generated code...[/dim]\n")

            with ui.show_thinking("Quality review in progress..."):
                review = await engine.quality_review(
                    code_to_review,
                    context=f"Code generated for task: {task}"
                )

            session.set_code_review(review)
            ui.show_code_review(review)
        else:
            ui.show_info("No code files to review")

        manager.save(session)

        # =====================================================================
        # COMPLETE
        # =====================================================================
        transition_session(session, WorkflowPhase.COMPLETE)
        manager.save(session)

        ui.show_header("Session Complete")
        ui.show_success(f"Session {session.session_id} completed successfully")
        ui.console.print(f"\n[bold]Total Tokens:[/bold] {session.total_tokens:,}")
        ui.console.print(f"[bold]Estimated Cost:[/bold] ${session.total_cost:.4f}")
        ui.console.print(f"\n[bold]Output:[/bold] {exec_result.output_path}")

        return session

    except KeyboardInterrupt:
        ui.show_warning("\nSession interrupted by user")
        transition_session(session, WorkflowPhase.ABORTED)
        manager.save(session)
        return session

    except Exception as e:
        ui.show_error(f"Session failed: {e}")
        try:
            transition_session(session, WorkflowPhase.FAILED)
        except ValueError:
            session.phase = WorkflowPhase.FAILED
        session.add_transcript_entry("error", {"error": str(e)})
        manager.save(session)
        raise


async def resume_session(session_id: str, interactive: bool = True) -> Session:
    """
    Resume a previous session from where it left off.

    Args:
        session_id: The session ID to resume
        interactive: Whether to use interactive checkpoints
    """
    manager = SessionManager()

    try:
        session = manager.load(session_id)
    except FileNotFoundError:
        ui.show_error(f"Session not found: {session_id}")
        sys.exit(1)

    ui.show_header("Resuming Session", f"ID: {session.session_id}")
    ui.console.print(f"[bold]Task:[/bold] {session.task}")
    ui.console.print(f"[bold]Current Phase:[/bold] {session.phase}\n")

    if session.phase in (WorkflowPhase.COMPLETE, WorkflowPhase.FAILED, WorkflowPhase.ABORTED):
        ui.show_warning(f"Session is already {session.phase} and cannot be resumed")
        return session

    # Get next phase
    next_phase = get_next_phase(session)
    if not next_phase:
        ui.show_info("Session has no remaining phases")
        return session

    ui.show_info(f"Resuming from phase: {session.phase}")
    ui.show_info(f"Next phase: {next_phase}")

    if not ui.prompt_continue("Continue session?"):
        return session

    # Re-run the session from the current phase
    config = session.config or create_default_config()
    delib_config = DeliberationConfig(
        chairman_model=config.get("chairman", "claude"),
        proposal_models=config.get("models", ["claude", "gpt4"]),
        critique_models=config.get("models", ["claude", "gpt4"]),
    )

    # Determine what to skip based on completed phases
    skip_roles = session.role_assignments is not None
    skip_planning = session.plan is not None

    # For now, restart the remaining phases
    # A more sophisticated implementation would pick up exactly where we left off
    return await run_session(
        task=session.task,
        config=config,
        delib_config=delib_config,
        interactive=interactive,
        skip_roles=skip_roles,
        skip_execution=False,
    )


def list_sessions() -> None:
    """List all past sessions."""
    manager = SessionManager()
    sessions = manager.list_all()

    ui.show_header("Past Sessions")

    if not sessions:
        ui.console.print("[dim]No sessions found.[/dim]")
        ui.console.print("\nStart a new session with:")
        ui.console.print('  council "Your task description"')
        return

    ui.show_sessions_list(sessions)

    resumable = [s for s in sessions if s["phase"] not in ("complete", "failed", "aborted")]
    if resumable:
        ui.console.print(f"\n[dim]{len(resumable)} session(s) can be resumed.[/dim]")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def async_main(args: argparse.Namespace) -> int:
    """
    Async main function that handles all commands.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Handle --list
    if args.list:
        list_sessions()
        return 0

    # Handle --resume
    if args.resume:
        try:
            await resume_session(
                args.resume,
                interactive=not args.no_interactive
            )
            return 0
        except Exception as e:
            ui.show_error(f"Failed to resume session: {e}")
            return 1

    # Handle --quick
    if args.quick:
        try:
            await run_quick_consensus(args.quick, args.models)
            return 0
        except Exception as e:
            ui.show_error(f"Quick consensus failed: {e}")
            return 1

    # Get task from file or argument
    task = None

    if args.file:
        if not args.file.exists():
            ui.show_error(f"File not found: {args.file}")
            return 1
        task = args.file.read_text().strip()
        ui.show_info(f"Loaded task from: {args.file}")

    elif args.task:
        task = args.task

    # No task provided
    if not task:
        ui.show_welcome()
        ui.show_quick_help()
        return 0

    # Create configuration
    config = create_default_config()
    delib_config = DeliberationConfig(
        chairman_model=args.chairman,
        proposal_models=args.models,
        critique_models=args.models,
    )

    # Run the session
    try:
        await run_session(
            task=task,
            config=config,
            delib_config=delib_config,
            interactive=not args.no_interactive,
            skip_roles=args.skip_roles,
            skip_execution=args.skip_execution,
            verbose=args.verbose,
        )
        return 0
    except KeyboardInterrupt:
        ui.console.print("\n")
        ui.show_warning("Interrupted")
        return 130
    except Exception as e:
        ui.show_error(f"Session failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Run async main
    try:
        exit_code = asyncio.run(async_main(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        ui.console.print("\n")
        sys.exit(130)


if __name__ == "__main__":
    main()
