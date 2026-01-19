"""
Rich terminal UI components for Council v2.

WHY THIS FILE EXISTS:
--------------------
The CLI needs to display structured information in a readable way.
Rich provides excellent terminal formatting: panels, tables, markdown, colors.

DESIGN PRINCIPLES:
-----------------
1. Consistent styling across all displays
2. Color-coded severity/status indicators
3. Collapsible detail sections
4. Clean formatting of structured Pydantic data

COMPONENTS:
----------
- show_proposal() - Display a model's proposal
- show_critique() - Display a critique
- show_role_assignments() - Display role deliberation results
- show_plan() - Display an implementation plan
- show_code_review() - Display code review results
- show_execution_result() - Display execution results
- prompt_checkpoint() - Interactive checkpoint prompt
"""

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich import box

from schemas import (
    Proposal,
    Critique,
    RoleAssignment,
    RoleDeliberationResult,
    Plan,
    PlanStep,
    CodeReview,
    ReviewIssue,
    ChairmanSummary,
    StepResult,
    ExecutionResult,
    ConsensusResult,
    DeliberationTranscript,
)

# Global console instance for consistent output
console = Console()


# =============================================================================
# COLOR SCHEMES
# =============================================================================

# Effort level colors
EFFORT_COLORS = {
    "low": "green",
    "medium": "yellow",
    "high": "red",
}

# Severity colors for code review
SEVERITY_COLORS = {
    "critical": "red bold",
    "major": "red",
    "minor": "yellow",
    "suggestion": "blue",
}

# Status colors
STATUS_COLORS = {
    "completed": "green",
    "in_progress": "yellow",
    "pending": "dim",
    "failed": "red",
    "aborted": "red",
}

# Agreement level colors
AGREEMENT_COLORS = {
    "unanimous": "green bold",
    "strong": "green",
    "moderate": "yellow",
    "divided": "red",
}


# =============================================================================
# HEADER/SECTION UTILITIES
# =============================================================================

def show_header(title: str, subtitle: str = "") -> None:
    """Display a styled header."""
    console.print()
    console.rule(f"[bold blue]{title}[/bold blue]")
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]", justify="center")
    console.print()


def show_section(title: str) -> None:
    """Display a section divider."""
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print("─" * 40)


def show_success(message: str) -> None:
    """Display a success message."""
    console.print(f"[green]✓[/green] {message}")


def show_error(message: str) -> None:
    """Display an error message."""
    console.print(f"[red]✗[/red] {message}")


def show_warning(message: str) -> None:
    """Display a warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def show_info(message: str) -> None:
    """Display an info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


# =============================================================================
# PROPOSAL DISPLAY
# =============================================================================

def show_proposal(proposal: Proposal, model_name: str = "Model") -> None:
    """
    Display a model's proposal in a rich panel.

    Args:
        proposal: The Proposal to display
        model_name: Name of the model that made the proposal
    """
    # Build content
    content = Text()

    # Approach
    content.append("Approach: ", style="bold")
    content.append(f"{proposal.approach}\n\n")

    # Rationale
    content.append("Rationale: ", style="bold")
    content.append(f"{proposal.rationale}\n\n")

    # Risks
    if proposal.risks:
        content.append("Risks:\n", style="bold")
        for risk in proposal.risks:
            content.append(f"  • {risk}\n", style="yellow")
        content.append("\n")

    # Effort and Confidence
    effort_color = EFFORT_COLORS.get(proposal.effort, "white")
    content.append("Effort: ", style="bold")
    content.append(f"{proposal.effort}", style=effort_color)
    content.append(" | ")
    content.append("Confidence: ", style="bold")
    conf_color = "green" if proposal.confidence >= 7 else "yellow" if proposal.confidence >= 4 else "red"
    content.append(f"{proposal.confidence}/10", style=conf_color)

    panel = Panel(
        content,
        title=f"[bold magenta]{model_name}'s Proposal[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED
    )
    console.print(panel)


# =============================================================================
# CRITIQUE DISPLAY
# =============================================================================

def show_critique(critique: Critique, model_name: str = "Model") -> None:
    """
    Display a model's critique in a rich panel.

    Args:
        critique: The Critique to display
        model_name: Name of the model that made the critique
    """
    content = Text()

    # Target
    content.append("Reviewing: ", style="bold")
    content.append(f"{critique.target_proposal}\n\n")

    # Agreements
    if critique.agrees:
        content.append("Agrees:\n", style="bold green")
        for point in critique.agrees:
            content.append(f"  ✓ {point}\n", style="green")
        content.append("\n")

    # Disagreements
    if critique.disagrees:
        content.append("Disagrees:\n", style="bold red")
        for point in critique.disagrees:
            content.append(f"  ✗ {point}\n", style="red")
        content.append("\n")

    # Suggestions
    if critique.suggestions:
        content.append("Suggestions:\n", style="bold blue")
        for suggestion in critique.suggestions:
            content.append(f"  → {suggestion}\n", style="blue")
        content.append("\n")

    # Recommendation
    rec_colors = {"accept": "green", "modify": "yellow", "reject": "red"}
    rec_color = rec_colors.get(critique.recommendation, "white")
    content.append("Recommendation: ", style="bold")
    content.append(f"{critique.recommendation.upper()}", style=f"{rec_color} bold")

    panel = Panel(
        content,
        title=f"[bold cyan]{model_name}'s Critique[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED
    )
    console.print(panel)


# =============================================================================
# ROLE ASSIGNMENTS DISPLAY
# =============================================================================

def show_role_assignments(result: RoleDeliberationResult) -> None:
    """
    Display role deliberation results with a table.

    Args:
        result: The RoleDeliberationResult to display
    """
    show_section("Role Assignments")

    # Create table
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Role", style="cyan")
    table.add_column("Assigned To", style="green")
    table.add_column("Reasoning", style="dim")

    for assignment in result.assignments:
        table.add_row(
            assignment.role,
            assignment.assigned_to,
            assignment.reasoning[:60] + "..." if len(assignment.reasoning) > 60 else assignment.reasoning
        )

    console.print(table)

    # Consensus notes
    console.print(f"\n[bold]Consensus:[/bold] {result.consensus_notes}")

    # Dissenting views
    if result.dissenting_views:
        console.print("\n[bold yellow]Dissenting Views:[/bold yellow]")
        for view in result.dissenting_views:
            console.print(f"  • {view}")


# =============================================================================
# PLAN DISPLAY
# =============================================================================

def show_plan(plan: Plan) -> None:
    """
    Display an implementation plan.

    Args:
        plan: The Plan to display
    """
    show_header("Implementation Plan", plan.overview[:80] + "..." if len(plan.overview) > 80 else plan.overview)

    # Overview panel
    console.print(Panel(
        plan.overview,
        title="[bold]Overview[/bold]",
        border_style="blue",
        box=box.ROUNDED
    ))

    # Steps table
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
        title="[bold]Steps[/bold]"
    )
    table.add_column("#", justify="right", style="cyan", width=3)
    table.add_column("Title", style="white")
    table.add_column("Agent", style="green")
    table.add_column("Complexity", justify="center")
    table.add_column("Dependencies", style="dim")

    critical_steps = set(plan.critical_path)

    for step in plan.steps:
        # Mark critical path steps
        step_num = f"[bold red]*{step.step_number}[/bold red]" if step.step_number in critical_steps else str(step.step_number)

        complexity_color = EFFORT_COLORS.get(step.estimated_complexity, "white")
        complexity = f"[{complexity_color}]{step.estimated_complexity}[/{complexity_color}]"

        deps = ", ".join(str(d) for d in step.dependencies) if step.dependencies else "-"

        table.add_row(step_num, step.title, step.assigned_agent, complexity, deps)

    console.print(table)

    if plan.critical_path:
        console.print(f"\n[red]*[/red] = Critical path step")

    # Effort estimate
    effort_color = EFFORT_COLORS.get(plan.estimated_total_effort, "white")
    console.print(f"\n[bold]Total Effort:[/bold] [{effort_color}]{plan.estimated_total_effort}[/{effort_color}]")

    # Risks
    if plan.risks_and_mitigations:
        console.print("\n[bold]Risks & Mitigations:[/bold]")
        for risk, mitigation in plan.risks_and_mitigations.items():
            console.print(f"  [yellow]Risk:[/yellow] {risk}")
            console.print(f"  [green]Mitigation:[/green] {mitigation}")
            console.print()


def show_plan_step(step: PlanStep, status: str = "pending") -> None:
    """
    Display a single plan step with its details.

    Args:
        step: The PlanStep to display
        status: Current status of the step
    """
    status_color = STATUS_COLORS.get(status, "white")

    content = Text()
    content.append(f"{step.description}\n\n")

    if step.acceptance_criteria:
        content.append("Acceptance Criteria:\n", style="bold")
        for criterion in step.acceptance_criteria:
            content.append(f"  □ {criterion}\n")

    panel = Panel(
        content,
        title=f"[{status_color}]Step {step.step_number}: {step.title}[/{status_color}]",
        subtitle=f"[dim]Agent: {step.assigned_agent}[/dim]",
        border_style=status_color,
        box=box.ROUNDED
    )
    console.print(panel)


# =============================================================================
# CODE REVIEW DISPLAY
# =============================================================================

def show_code_review(review: CodeReview) -> None:
    """
    Display code review results.

    Args:
        review: The CodeReview to display
    """
    show_section(f"Code Review by {review.reviewer}")

    # Summary
    approval_status = "[green]APPROVED[/green]" if review.approved else "[red]NOT APPROVED[/red]"
    console.print(f"{approval_status}: {review.summary}\n")

    # Issues by severity
    issues_by_severity = review.by_severity

    for severity in ["critical", "major", "minor", "suggestion"]:
        issues = issues_by_severity.get(severity, [])
        if not issues:
            continue

        color = SEVERITY_COLORS.get(severity, "white")
        console.print(f"\n[{color}]{severity.upper()} ({len(issues)})[/{color}]")

        for issue in issues:
            console.print(f"  [{color}]•[/{color}] [{issue.category}] {issue.location}")
            console.print(f"    {issue.description}")
            console.print(f"    [dim]Fix: {issue.suggested_fix}[/dim]")

    # Positive notes
    if review.positive_notes:
        console.print("\n[green bold]Positives:[/green bold]")
        for note in review.positive_notes:
            console.print(f"  [green]✓[/green] {note}")

    # Summary stats
    console.print(f"\n[bold]Summary:[/bold] {review.blocking_issues} blocking issues, {len(review.issues)} total")


def show_review_issue(issue: ReviewIssue) -> None:
    """Display a single review issue."""
    color = SEVERITY_COLORS.get(issue.severity, "white")

    console.print(f"[{color}][{issue.severity.upper()}][/{color}] [{issue.category}] {issue.location}")
    console.print(f"  {issue.description}")
    console.print(f"  [dim]Fix ({issue.effort_to_fix} effort): {issue.suggested_fix}[/dim]")


# =============================================================================
# EXECUTION DISPLAY
# =============================================================================

def show_step_result(result: StepResult) -> None:
    """
    Display the result of executing a step.

    Args:
        result: The StepResult to display
    """
    status_color = STATUS_COLORS.get(result.status.value, "white")

    content = Text()

    # Summary
    content.append(f"{result.output_summary}\n\n")

    # Tool calls
    if result.tool_calls:
        content.append("Actions taken:\n", style="bold")
        for tc, tr in zip(result.tool_calls, result.tool_results):
            status_icon = "✓" if tr.success else "✗"
            status_style = "green" if tr.success else "red"
            content.append(f"  {status_icon} ", style=status_style)
            content.append(f"{tc.tool_name}")
            content.append(f" ({tr.duration_ms}ms)\n", style="dim")

    # Files
    if result.files_created:
        content.append("\nFiles created:\n", style="bold")
        for f in result.files_created:
            content.append(f"  + {f}\n", style="green")

    if result.files_modified:
        content.append("\nFiles modified:\n", style="bold")
        for f in result.files_modified:
            content.append(f"  ~ {f}\n", style="yellow")

    # Error
    if result.error:
        content.append(f"\n[red]Error: {result.error}[/red]")

    panel = Panel(
        content,
        title=f"[{status_color}]Step {result.step_number} Result[/{status_color}]",
        subtitle=f"[dim]Agent: {result.agent_name}[/dim]",
        border_style=status_color,
        box=box.ROUNDED
    )
    console.print(panel)


def show_chairman_summary(summary: ChairmanSummary) -> None:
    """
    Display the chairman's summary at a checkpoint.

    Args:
        summary: The ChairmanSummary to display
    """
    content = Text()
    content.append(f"{summary.what_was_done}\n\n")

    if summary.files_changed:
        content.append("Files changed:\n", style="bold")
        for f in summary.files_changed:
            content.append(f"  • {f}\n")
        content.append("\n")

    if summary.key_decisions:
        content.append("Key decisions:\n", style="bold")
        for decision in summary.key_decisions:
            content.append(f"  → {decision}\n", style="blue")
        content.append("\n")

    if summary.potential_concerns:
        content.append("Concerns:\n", style="bold yellow")
        for concern in summary.potential_concerns:
            content.append(f"  ⚠ {concern}\n", style="yellow")

    # Recommendation
    rec_colors = {
        "proceed": "green",
        "review_recommended": "yellow",
        "rework_suggested": "red"
    }
    rec_color = rec_colors.get(summary.recommendation, "white")
    content.append(f"\n[bold]Recommendation:[/bold] [{rec_color}]{summary.recommendation.replace('_', ' ').title()}[/{rec_color}]")

    panel = Panel(
        content,
        title=f"[bold]Step {summary.step_number}: {summary.title}[/bold]",
        border_style="blue",
        box=box.DOUBLE
    )
    console.print(panel)


def show_execution_result(result: ExecutionResult) -> None:
    """
    Display the final execution result.

    Args:
        result: The ExecutionResult to display
    """
    status = "SUCCESS" if result.success else "FAILED"
    status_color = "green" if result.success else "red"

    show_header(f"Execution {status}", result.summary)

    # Stats
    transcript = result.transcript
    console.print(f"[bold]Session:[/bold] {transcript.session_id}")
    console.print(f"[bold]Status:[/bold] [{status_color}]{transcript.status}[/{status_color}]")
    console.print(f"[bold]Steps Completed:[/bold] {len(transcript.step_results)}")
    console.print(f"[bold]Rework Iterations:[/bold] {transcript.rework_count}")

    # Costs
    if transcript.total_tokens > 0:
        console.print(f"\n[bold]Token Usage:[/bold] {transcript.total_tokens:,} tokens")
    if transcript.total_cost > 0:
        console.print(f"[bold]Estimated Cost:[/bold] ${transcript.total_cost:.4f}")

    # Duration
    if transcript.started_at and transcript.completed_at:
        duration = (transcript.completed_at - transcript.started_at).total_seconds()
        console.print(f"[bold]Duration:[/bold] {duration:.1f}s")

    # Files created
    if result.files_created:
        console.print(f"\n[bold]Files Created ({len(result.files_created)}):[/bold]")
        for f in result.files_created[:10]:
            console.print(f"  [green]+[/green] {f}")
        if len(result.files_created) > 10:
            console.print(f"  [dim]... and {len(result.files_created) - 10} more[/dim]")

    # Output path
    console.print(f"\n[bold]Output:[/bold] {result.output_path}")

    # Next steps
    if result.next_steps:
        console.print("\n[bold]Suggested Next Steps:[/bold]")
        for i, step in enumerate(result.next_steps, 1):
            console.print(f"  {i}. {step}")


# =============================================================================
# CONSENSUS DISPLAY
# =============================================================================

def show_consensus_result(result: ConsensusResult, question: str = "") -> None:
    """
    Display quick consensus result.

    Args:
        result: The ConsensusResult to display
        question: The original question asked
    """
    if question:
        console.print(f"\n[bold]Question:[/bold] {question}\n")

    # Agreement indicator
    agreement_color = AGREEMENT_COLORS.get(result.agreement_level, "white")

    content = Text()
    content.append(f"{result.answer}\n\n")

    # Confidence and agreement
    conf_color = "green" if result.confidence >= 7 else "yellow" if result.confidence >= 4 else "red"
    content.append("Confidence: ", style="bold")
    content.append(f"{result.confidence}/10", style=conf_color)
    content.append(" | ")
    content.append("Agreement: ", style="bold")
    content.append(f"{result.agreement_level}", style=agreement_color)
    content.append("\n")

    # Sources
    if result.sources:
        content.append("\nSources: ", style="bold")
        content.append(", ".join(result.sources), style="dim")

    # Dissenting views
    if result.dissenting_views:
        content.append("\n\nDissenting views:\n", style="bold yellow")
        for view in result.dissenting_views:
            content.append(f"  • {view}\n", style="yellow")

    panel = Panel(
        content,
        title="[bold green]Consensus[/bold green]",
        border_style="green",
        box=box.DOUBLE
    )
    console.print(panel)


# =============================================================================
# TRANSCRIPT DISPLAY
# =============================================================================

def show_deliberation_transcript(transcript: DeliberationTranscript) -> None:
    """
    Display a deliberation transcript summary.

    Args:
        transcript: The DeliberationTranscript to display
    """
    show_section("Deliberation Transcript")

    console.print(f"[bold]Session:[/bold] {transcript.session_id}")
    console.print(f"[bold]Task:[/bold] {transcript.task[:60]}...")
    console.print(f"[bold]Duration:[/bold] {transcript.duration_seconds:.1f}s")

    # Phases table
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Phase")
    table.add_column("Responses", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Duration", justify="right")

    for phase in transcript.phases:
        tokens = sum(r.tokens_used for r in phase.responses)
        table.add_row(
            phase.phase_name,
            str(len(phase.responses)),
            f"{tokens:,}",
            f"{phase.duration_seconds:.1f}s"
        )

    console.print(table)

    # Totals
    console.print(f"\n[bold]Total Tokens:[/bold] {transcript.total_tokens:,}")
    if transcript.total_cost > 0:
        console.print(f"[bold]Estimated Cost:[/bold] ${transcript.total_cost:.4f}")


# =============================================================================
# INTERACTIVE PROMPTS
# =============================================================================

def prompt_checkpoint(summary: ChairmanSummary) -> str:
    """
    Prompt the user at an execution checkpoint.

    Args:
        summary: Chairman's summary of the step

    Returns:
        One of: "approve", "feedback", "rework", "skip", "abort"
    """
    show_chairman_summary(summary)

    console.print("\n[bold]Options:[/bold]")
    console.print("  [green]a[/green]pprove  - Continue to next step")
    console.print("  [yellow]f[/yellow]eedback - Approve with feedback for next step")
    console.print("  [yellow]r[/yellow]ework  - Redo this step with changes")
    console.print("  [dim]s[/dim]kip    - Skip this step")
    console.print("  [red]x[/red]      - Abort execution")

    while True:
        choice = Prompt.ask(
            "\n[bold]Choice[/bold]",
            choices=["a", "f", "r", "s", "x", "approve", "feedback", "rework", "skip", "abort"],
            default="a"
        )

        mapping = {
            "a": "approve", "approve": "approve",
            "f": "feedback", "feedback": "feedback",
            "r": "rework", "rework": "rework",
            "s": "skip", "skip": "skip",
            "x": "abort", "abort": "abort"
        }
        return mapping.get(choice, "approve")


def prompt_feedback() -> str:
    """
    Get detailed feedback from the user.

    Returns:
        User's feedback text
    """
    console.print("\n[bold]Enter your feedback[/bold] (press Enter twice to submit):")

    lines = []
    while True:
        line = Prompt.ask("", default="")
        if not line and lines:
            break
        lines.append(line)

    return "\n".join(lines)


def prompt_continue(message: str = "Continue?") -> bool:
    """
    Simple yes/no confirmation prompt.

    Args:
        message: The confirmation message

    Returns:
        True if user confirms, False otherwise
    """
    return Confirm.ask(f"[bold]{message}[/bold]", default=True)


# =============================================================================
# PROGRESS INDICATORS
# =============================================================================

def create_progress() -> Progress:
    """Create a progress indicator for long-running operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    )


def show_thinking(message: str = "Thinking..."):
    """
    Context manager that shows a spinner while processing.

    Usage:
        with show_thinking("Deliberating..."):
            result = await deliberate()
    """
    return console.status(f"[bold blue]{message}[/bold blue]", spinner="dots")


# =============================================================================
# SESSION LIST DISPLAY
# =============================================================================

def show_sessions_list(sessions: list[dict]) -> None:
    """
    Display a list of sessions.

    Args:
        sessions: List of session dicts with id, task, phase, etc.
    """
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("ID", style="cyan")
    table.add_column("Phase", style="yellow")
    table.add_column("Task")
    table.add_column("Created", style="dim")

    for session in sessions:
        phase_color = STATUS_COLORS.get(session.get("phase", ""), "white")
        table.add_row(
            session.get("session_id", "?")[:8],
            f"[{phase_color}]{session.get('phase', 'unknown')}[/{phase_color}]",
            session.get("task", "")[:50] + "..." if len(session.get("task", "")) > 50 else session.get("task", ""),
            session.get("created_at", "")
        )

    console.print(table)


# =============================================================================
# WELCOME / HELP
# =============================================================================

def show_welcome() -> None:
    """Display welcome message."""
    console.print(Panel.fit(
        "[bold blue]Council v2[/bold blue]\n"
        "[dim]Multi-model deliberation with structured outputs[/dim]",
        border_style="blue"
    ))


def show_quick_help() -> None:
    """Display quick help."""
    console.print("""
[bold]Usage:[/bold]
  council "task description"      Start a new session
  council --file task.txt         Load task from file
  council --quick "question"      Quick consensus mode
  council --list                  List past sessions
  council --resume SESSION_ID     Resume a session
  council --help                  Show full help
""")
