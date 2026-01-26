#!/usr/bin/env python3
"""
Agent Council - Multi-model, multi-agent deliberation system
Main CLI entry point
"""

import argparse
import asyncio
import json
import os
import select
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.table import Table

from providers import get_provider, ModelProvider
from agents import AgentRegistry, Agent
from chairman import Chairman
from deliberation import Deliberation
from workspace import WorkspaceManager
from alerts import AlertManager
from costs import CostTracker

console = Console()


def emit_checkpoint(checkpoint_type: str, data: dict, chairman_analysis: str,
                    options: list, timeout: int = 120) -> dict:
    """Output checkpoint as JSON for Claude Code, read response from stdin.

    This enables Claude Code to act as the Chairman, making decisions at each
    checkpoint instead of requiring interactive terminal prompts.

    Args:
        checkpoint_type: Type of checkpoint (role_approval, plan_review, etc.)
        data: Checkpoint data to send
        chairman_analysis: Chairman's analysis text
        options: Available options for this checkpoint
        timeout: Seconds to wait for response (default 120)

    Returns:
        Response dict with 'action' key and optional additional fields
    """
    checkpoint = {
        "type": checkpoint_type,
        "data": data,
        "chairman_analysis": chairman_analysis,
        "options": options,
        "timestamp": datetime.now().isoformat()
    }
    # Output with markers so Claude Code can parse it
    print(f"<<<CHECKPOINT:{json.dumps(checkpoint)}>>>", flush=True)

    # Wait for response from Claude Code via stdin with timeout
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            print(f"Warning: Checkpoint '{checkpoint_type}' timed out after {timeout}s",
                  file=sys.stderr)
            # Return timeout action - caller should handle escalation
            return {"action": "timeout", "escalate": True, "checkpoint_type": checkpoint_type}

        response_line = sys.stdin.readline().strip()
        if not response_line:
            # Empty response defaults to approve
            print("Info: Empty checkpoint response, defaulting to approve", file=sys.stderr)
            return {"action": "approve"}

        # Parse and validate JSON response
        try:
            response = json.loads(response_line)
            if not isinstance(response, dict):
                raise ValueError("Response must be a JSON object")
            if "action" not in response:
                raise ValueError("Response must contain 'action' field")
            return response
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Invalid checkpoint response: {e}", file=sys.stderr)
            return {"action": "error", "message": str(e), "raw_response": response_line}

    except Exception as e:
        print(f"Error reading checkpoint response: {e}", file=sys.stderr)
        return {"action": "error", "message": str(e)}


# Default config path
CONFIG_DIR = Path.home() / ".council"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
SESSIONS_DIR = CONFIG_DIR / "sessions"
TRANSCRIPTS_DIR = CONFIG_DIR / "transcripts"


def ensure_config():
    """Create default config if it doesn't exist."""
    CONFIG_DIR.mkdir(exist_ok=True)
    SESSIONS_DIR.mkdir(exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    
    if not CONFIG_FILE.exists():
        default_config = """# Agent Council Configuration

models:
  claude:
    provider: "anthropic"
    model: "claude-sonnet-4-20250514"
    api_key_env: "ANTHROPIC_API_KEY"
    
  deepseek:
    provider: "ollama"
    model: "deepseek-coder-v2:16b"
    base_url: "http://localhost:11434"
    
  codex:
    provider: "openai"
    model: "gpt-5.2"
    api_key_env: "OPENAI_API_KEY"

chairman:
  model: "claude"
  memory_file: "~/.council/chairman_memory.yaml"

alerts:
  terminal: true
  macos_notification: true

transcripts:
  show: "summary"
  save: "full"
  directory: "~/.council/transcripts"

workspaces:
  base: "./council-workspace"
  per_agent: true
"""
        CONFIG_FILE.write_text(default_config)
        console.print(f"[green]Created default config at {CONFIG_FILE}[/green]")


def load_config() -> dict:
    """Load configuration from YAML file."""
    import yaml
    ensure_config()
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


class CouncilSession:
    """Manages a single council session."""

    def __init__(self, task: str, config: dict, session_id: Optional[str] = None,
                 claude_code_mode: bool = False):
        self.task = task
        self.config = config
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.claude_code_mode = claude_code_mode
        self.transcript = []
        self.cost_tracker = CostTracker()
        
        # Initialize components
        self.workspace = WorkspaceManager(
            base_path=config.get("workspaces", {}).get("base", "./council-workspace"),
            session_id=self.session_id
        )
        self.alerts = AlertManager(
            terminal=config.get("alerts", {}).get("terminal", True),
            macos=config.get("alerts", {}).get("macos_notification", True)
        )
        self.chairman = Chairman(
            config=config,
            memory_file=Path(config.get("chairman", {}).get(
                "memory_file", "~/.council/chairman_memory.yaml"
            )).expanduser(),
            alerts=self.alerts
        )
        self.agents = AgentRegistry(config)
        self.deliberation = Deliberation(
            agents=self.agents,
            chairman=self.chairman,
            cost_tracker=self.cost_tracker
        )
        
    def log(self, phase: str, speaker: str, content: str):
        """Log to transcript."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "speaker": speaker,
            "content": content
        }
        self.transcript.append(entry)
        
    def save_transcript(self):
        """Save full transcript to file."""
        transcript_file = TRANSCRIPTS_DIR / f"{self.session_id}.json"
        with open(transcript_file, "w") as f:
            json.dump({
                "session_id": self.session_id,
                "task": self.task,
                "started": self.transcript[0]["timestamp"] if self.transcript else None,
                "entries": self.transcript
            }, f, indent=2)
        return transcript_file

    def _extract_code_blocks(self, content: str) -> dict[str, str]:
        """Extract filename: content pairs from markdown code blocks (P05).

        Looks for patterns like:
        ```python
        # filename.py
        code here
        ```
        """
        import re
        files = {}
        # Match code blocks with a filename comment on the first line
        # Support both # (Python/Shell) and // (JS/TS/Go/C++) style comments
        pattern = r'```[\w]*\n(?:#|//)\s*([^\n]+)\n(.*?)```'
        for match in re.finditer(pattern, content, re.DOTALL):
            filename = match.group(1).strip()
            code = match.group(2).strip()
            # Clean up filename (remove any trailing punctuation/comments)
            filename = filename.split()[0] if filename else None
            if filename and code:
                files[filename] = code
        return files

    async def run(self):
        """Run the full council workflow."""
        console.print(Panel(
            f"[bold]Agent Council Session: {self.session_id}[/bold]\n\n"
            f"Task: {self.task}",
            title="🏛️ Council Convened"
        ))

        # P07: Check for failed agents at start
        if self.agents.failed_agents:
            console.print(Panel(
                f"[yellow]Unavailable agents: {', '.join(self.agents.failed_agents)}[/yellow]\n"
                "These agents will be skipped during the session.",
                title="⚠️ Agent Status"
            ))

            if not self.claude_code_mode:
                if not Confirm.ask("Continue with available agents?"):
                    console.print("[red]Session cancelled.[/red]")
                    return
            else:
                decision = emit_checkpoint(
                    "agent_availability",
                    {"unavailable": list(self.agents.failed_agents)},
                    "Some agents are unavailable",
                    ["continue", "abort"]
                )
                if decision.get("action") == "abort":
                    console.print("[red]Session aborted due to unavailable agents.[/red]")
                    return
                elif decision.get("action") in ["timeout", "error"]:
                    # Checkpoint failed - fall back to terminal prompt
                    console.print(f"[yellow]Checkpoint issue: {decision.get('message', 'timeout')}[/yellow]")
                    console.print("[yellow]Falling back to terminal prompt...[/yellow]")
                    if not Confirm.ask("Continue with available agents?"):
                        console.print("[red]Session cancelled.[/red]")
                        return

        # Phase 0: Role Deliberation
        await self.phase_role_deliberation()
        
        # Phase 1: Planning
        plan = await self.phase_planning()
        if not plan:
            return
            
        # Phase 2: Build
        outputs = await self.phase_build(plan)
        if not outputs:
            return
            
        # Phase 3: Quality Review
        reviewed = await self.phase_quality(outputs)
        
        # Phase 4: Merge & Final
        await self.phase_final(reviewed)
        
        # Save transcript
        transcript_file = self.save_transcript()
        console.print(f"\n[dim]Transcript saved: {transcript_file}[/dim]")
        
    async def phase_role_deliberation(self):
        """Phase 0: Models deliberate on role assignments."""
        console.print("\n[bold cyan]═══ Phase 0: Role Deliberation ═══[/bold cyan]\n")
        
        # Get deliberation from models
        deliberation_result = await self.deliberation.deliberate_roles(self.task)
        
        # Log full transcript
        self.log("role_deliberation", "system", deliberation_result["full_transcript"])
        
        # Show summary
        console.print(Panel(
            Markdown(deliberation_result["summary"]),
            title="📋 Role Assignment Summary"
        ))
        
        # Chairman presents to user
        chairman_analysis = await self.chairman.analyze_role_assignments(
            deliberation_result["assignments"],
            self.task
        )
        
        console.print(Panel(
            Markdown(chairman_analysis),
            title="👔 Chairman's Analysis"
        ))
        
        # User approval (or Claude Code decision)
        if self.claude_code_mode:
            decision = emit_checkpoint(
                "role_approval",
                {"assignments": deliberation_result["assignments"]},
                chairman_analysis,
                ["approve", "feedback"]
            )
            # Log checkpoint decision for learning
            self.log("role_approval_checkpoint", "claude_code", json.dumps(decision))

            # Handle timeout/error by escalating to terminal prompt
            if decision["action"] in ["timeout", "error"]:
                console.print(f"[yellow]Checkpoint issue: {decision.get('message', 'timeout')}[/yellow]")
                console.print("[yellow]Falling back to terminal prompt...[/yellow]")
                if not Confirm.ask("\n[bold]Approve role assignments?[/bold]"):
                    feedback = Prompt.ask("Your feedback for the council")
                    self.log("role_approval_checkpoint", "user", f"feedback: {feedback}")
                    console.print("[yellow]Re-deliberating with your feedback...[/yellow]")
                    await self.phase_role_deliberation()
            elif decision["action"] != "approve":
                feedback = decision.get("content", "Please reconsider the role assignments")
                console.print(f"[yellow]Re-deliberating with feedback: {feedback}[/yellow]")
                await self.phase_role_deliberation()
        else:
            if not Confirm.ask("\n[bold]Approve role assignments?[/bold]"):
                feedback = Prompt.ask("Your feedback for the council")
                console.print("[yellow]Re-deliberating with your feedback...[/yellow]")
                await self.phase_role_deliberation()
            
    async def phase_planning(self) -> Optional[dict]:
        """Phase 1: Planning with checkpoint."""
        console.print("\n[bold cyan]═══ Phase 1: Planning ═══[/bold cyan]\n")
        
        # Agents deliberate on plan
        plan_result = await self.deliberation.deliberate_plan(self.task)
        
        self.log("planning", "system", plan_result["full_transcript"])
        
        # Show summary
        console.print(Panel(
            Markdown(plan_result["summary"]),
            title="📐 Architecture & Plan"
        ))
        
        # Cost estimate
        estimated_cost = self.cost_tracker.estimate_build_cost(plan_result["plan"])
        console.print(f"\n[bold]Estimated build cost:[/bold] ${estimated_cost:.2f}")
        
        # Chairman discussion
        console.print("\n[bold green]─── Checkpoint 1: Plan Review with Chairman ───[/bold green]\n")
        
        while True:
            chairman_thoughts = await self.chairman.review_plan(plan_result["plan"], self.task)
            console.print(Panel(Markdown(chairman_thoughts), title="👔 Chairman"))

            if self.claude_code_mode:
                decision = emit_checkpoint(
                    "plan_review",
                    {"plan": plan_result["plan"], "cost": estimated_cost},
                    chairman_thoughts,
                    ["approve", "feedback", "cancel"]
                )
                # Log checkpoint decision for learning
                self.log("plan_review_checkpoint", "claude_code", json.dumps(decision))

                # Handle timeout/error by falling back to terminal
                if decision["action"] in ["timeout", "error"]:
                    console.print(f"[yellow]Checkpoint issue: {decision.get('message', 'timeout')}[/yellow]")
                    user_input = Prompt.ask(
                        "\n[bold]Your response[/bold] (approve / feedback / question)",
                        default="approve"
                    ).lower()
                    feedback_content = user_input if user_input not in ["approve", "quit", "exit", "cancel"] else ""
                    self.log("plan_review_checkpoint", "user", f"{user_input}: {feedback_content}")
                else:
                    user_input = decision["action"]
                    feedback_content = decision.get("content", "")
            else:
                user_input = Prompt.ask(
                    "\n[bold]Your response[/bold] (approve / feedback / question)",
                    default="approve"
                ).lower()
                feedback_content = user_input if user_input not in ["approve", "quit", "exit", "cancel"] else ""

            if user_input == "approve":
                break
            elif user_input in ["quit", "exit", "cancel"]:
                console.print("[red]Session cancelled.[/red]")
                return None
            else:
                # Chairman helps craft correction
                correction = await self.chairman.craft_correction(
                    feedback_content or user_input,
                    plan_result["plan"]
                )
                console.print(Panel(Markdown(correction), title="👔 Proposed Correction"))

                # In Claude Code mode, auto-send corrections (Claude already decided to give feedback)
                if self.claude_code_mode:
                    send_correction = True
                else:
                    send_correction = Confirm.ask("Send this correction to the agents?")

                if send_correction:
                    plan_result = await self.deliberation.revise_plan(correction)
                    console.print(Panel(Markdown(plan_result["summary"]), title="📐 Revised Plan"))

        return plan_result["plan"]
        
    async def phase_build(self, plan: dict) -> Optional[dict]:
        """Phase 2: Parallel build with per-agent checkpoints."""
        console.print("\n[bold cyan]═══ Phase 2: Build ═══[/bold cyan]\n")
        
        # Create workspaces
        self.workspace.setup_agent_directories(plan["agents"])
        
        # Build in parallel
        console.print("[dim]Agents working in parallel...[/dim]\n")
        
        outputs = {}
        for agent_name, agent_task in plan["agent_tasks"].items():
            console.print(f"[yellow]► {agent_name} working on: {agent_task['summary']}[/yellow]")
            
            # Execute agent task
            result = await self.deliberation.execute_agent_task(
                agent_name,
                agent_task,
                self.workspace.get_agent_dir(agent_name)
            )

            # P05: Extract and write files from agent output
            files_written = []
            code_blocks = self._extract_code_blocks(result.get("output", ""))
            for filename, code_content in code_blocks.items():
                filepath = self.workspace.write_agent_file(agent_name, filename, code_content)
                files_written.append(str(filepath))
            result["files"] = files_written  # Update with actual written paths

            outputs[agent_name] = result
            self.log("build", agent_name, result["output"])
            
            # Checkpoint 2: Per-agent review
            console.print(f"\n[bold green]─── Checkpoint: {agent_name}'s Output ───[/bold green]\n")

            summary = await self.chairman.summarize_agent_output(agent_name, result)
            console.print(Panel(Markdown(summary), title=f"👔 Chairman on {agent_name}"))

            if self.claude_code_mode:
                decision = emit_checkpoint(
                    "agent_review",
                    {"agent": agent_name, "output": result.get("output", ""), "files": result.get("files", [])},
                    summary,
                    ["approve", "feedback"]
                )
                # Handle timeout/error by falling back to terminal
                if decision["action"] in ["timeout", "error"]:
                    console.print(f"[yellow]Checkpoint issue: {decision.get('message', 'timeout')}[/yellow]")
                    user_input = Prompt.ask(
                        "[bold]Your response[/bold] (approve / feedback)",
                        default="approve"
                    ).lower()
                    feedback_content = user_input if user_input != "approve" else ""
                else:
                    user_input = decision["action"]
                    feedback_content = decision.get("content", "")
            else:
                user_input = Prompt.ask(
                    "[bold]Your response[/bold] (approve / feedback)",
                    default="approve"
                ).lower()
                feedback_content = user_input if user_input != "approve" else ""

            if user_input != "approve":
                correction = await self.chairman.craft_agent_correction(
                    agent_name, feedback_content or user_input, result
                )
                # In Claude Code mode, auto-send corrections
                if self.claude_code_mode:
                    send_rework = True
                else:
                    send_rework = Confirm.ask("Send correction for rework?")

                if send_rework:
                    result = await self.deliberation.rework_agent_task(
                        agent_name, correction, agent_task
                    )
                    outputs[agent_name] = result
                    
        return outputs
        
    async def phase_quality(self, outputs: dict) -> dict:
        """Phase 3: Quality review with checkpoint."""
        console.print("\n[bold cyan]═══ Phase 3: Quality Review ═══[/bold cyan]\n")
        
        # Run quality agents
        review_result = await self.deliberation.quality_review(outputs)
        
        self.log("quality", "system", review_result["full_transcript"])
        
        # Checkpoint 3: Review with Chairman
        console.print("\n[bold green]─── Checkpoint 3: Quality Review with Chairman ───[/bold green]\n")
        
        chairman_review = await self.chairman.present_quality_review(review_result)
        console.print(Panel(Markdown(chairman_review), title="👔 Chairman's Quality Summary"))
        
        if review_result["issues"]:
            console.print("\n[bold]Issues found:[/bold]")
            for i, issue in enumerate(review_result["issues"], 1):
                severity_color = {
                    "critical": "red",
                    "major": "yellow", 
                    "minor": "dim"
                }.get(issue["severity"], "white")
                console.print(f"  [{severity_color}]{i}. [{issue['severity'].upper()}] {issue['description']}[/{severity_color}]")
                
            # Chairman's recommendation
            recommendation = await self.chairman.recommend_rework(review_result["issues"])
            console.print(Panel(Markdown(recommendation), title="👔 Chairman's Recommendation"))

            # User decides (or Claude Code decides)
            if self.claude_code_mode:
                decision = emit_checkpoint(
                    "quality_issues",
                    {"issues": review_result["issues"]},
                    recommendation,
                    ["all", "select_issues", "none"]
                )
                # Handle timeout/error by falling back to terminal
                if decision["action"] in ["timeout", "error"]:
                    console.print(f"[yellow]Checkpoint issue: {decision.get('message', 'timeout')}[/yellow]")
                    rework_input = Prompt.ask(
                        "[bold]Issues to send back[/bold] (all / numbers like '1,3' / none)",
                        default="all"
                    )
                elif decision["action"] == "none":
                    rework_input = "none"
                elif decision["action"] == "select_issues":
                    # Expect items as list of indices
                    rework_input = ",".join(str(i) for i in decision.get("items", []))
                else:
                    rework_input = "all"
            else:
                rework_input = Prompt.ask(
                    "[bold]Issues to send back[/bold] (all / numbers like '1,3' / none)",
                    default="all"
                )

            if rework_input.lower() != "none":
                if rework_input.lower() == "all":
                    issues_to_fix = review_result["issues"]
                else:
                    indices = [int(x.strip()) - 1 for x in rework_input.split(",")]
                    issues_to_fix = [review_result["issues"][i] for i in indices if 0 <= i < len(review_result["issues"])]

                # Send back for rework
                if issues_to_fix:
                    outputs = await self.deliberation.rework_issues(outputs, issues_to_fix)
                
        return outputs
        
    async def phase_final(self, outputs: dict):
        """Phase 4: Merge and final presentation."""
        console.print("\n[bold cyan]═══ Phase 4: Final Integration ═══[/bold cyan]\n")
        
        # Architect integrates
        final_result = await self.deliberation.integrate_outputs(outputs)
        
        self.log("final", "architect", final_result["summary"])
        
        # Chairman presents
        final_presentation = await self.chairman.present_final(
            final_result,
            self.cost_tracker.get_total_cost()
        )
        
        console.print(Panel(Markdown(final_presentation), title="👔 Final Deliverable"))
        
        # Cost summary
        cost_table = Table(title="💰 Cost Summary")
        cost_table.add_column("Phase")
        cost_table.add_column("Tokens")
        cost_table.add_column("Cost")
        
        for phase, data in self.cost_tracker.get_breakdown().items():
            cost_table.add_row(phase, str(data["tokens"]), f"${data['cost']:.4f}")
            
        cost_table.add_row(
            "[bold]TOTAL[/bold]", 
            str(self.cost_tracker.total_tokens),
            f"[bold]${self.cost_tracker.get_total_cost():.4f}[/bold]"
        )
        console.print(cost_table)
        
        # Update Chairman's memory with learnings
        await self.chairman.learn_from_session(self.transcript)
        
        # Final approval
        if self.claude_code_mode:
            decision = emit_checkpoint(
                "final_approval",
                {"summary": final_result.get("summary", ""), "total_cost": self.cost_tracker.get_total_cost()},
                final_presentation,
                ["approve", "reject"]
            )
            # Log checkpoint decision for learning
            self.log("final_approval_checkpoint", "claude_code", json.dumps(decision))

            # Handle timeout/error by falling back to terminal
            if decision["action"] in ["timeout", "error"]:
                console.print(f"[yellow]Checkpoint issue: {decision.get('message', 'timeout')}[/yellow]")
                accepted = Confirm.ask("\n[bold]Accept final deliverable?[/bold]")
                feedback = "" if accepted else Prompt.ask("Final feedback")
                self.log("final_approval_checkpoint", "user", f"accepted={accepted}: {feedback}")
            else:
                accepted = decision["action"] == "approve"
                feedback = decision.get("content", "")
        else:
            accepted = Confirm.ask("\n[bold]Accept final deliverable?[/bold]")
            feedback = ""

        if accepted:
            output_path = self.workspace.finalize()
            console.print(f"\n[bold green]✓ Project saved to: {output_path}[/bold green]")
        else:
            if not feedback and not self.claude_code_mode:
                feedback = Prompt.ask("Final feedback for the record")
            self.log("final", "user", f"Not accepted: {feedback}")


async def main():
    parser = argparse.ArgumentParser(
        description="Agent Council - Multi-model deliberation system"
    )
    parser.add_argument("task", nargs="?", help="Task description")
    parser.add_argument("--file", "-f", help="Load task from file")
    parser.add_argument("--resume", "-r", help="Resume session by ID")
    parser.add_argument("--transcript", "-t", help="View transcript for session ID")
    parser.add_argument("--memory", "-m", choices=["view", "edit", "clear"],
                       help="Manage Chairman's memory")
    parser.add_argument("--list", "-l", action="store_true", help="List past sessions")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Quick mode - skip role deliberation")
    parser.add_argument("--claude-code", "-cc", action="store_true",
                       help="Run in Claude Code mode (JSON checkpoint I/O)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Claude Code: ask user at every checkpoint")
    parser.add_argument("--smart", "-s", action="store_true",
                       help="Claude Code: smart automation with selective escalation")

    args = parser.parse_args()
    
    config = load_config()
    
    # Handle memory commands
    if args.memory:
        chairman = Chairman(config, 
            Path(config.get("chairman", {}).get("memory_file", "~/.council/chairman_memory.yaml")).expanduser(),
            None
        )
        if args.memory == "view":
            console.print(Panel(chairman.get_memory_display(), title="Chairman's Memory"))
        elif args.memory == "edit":
            os.system(f"$EDITOR {chairman.memory_file}")
        elif args.memory == "clear":
            if Confirm.ask("Clear all Chairman memory?"):
                chairman.clear_memory()
                console.print("[green]Memory cleared.[/green]")
        return
        
    # Handle transcript viewing
    if args.transcript:
        transcript_file = TRANSCRIPTS_DIR / f"{args.transcript}.json"
        if transcript_file.exists():
            with open(transcript_file) as f:
                data = json.load(f)
            console.print(Panel(json.dumps(data, indent=2), title=f"Transcript: {args.transcript}"))
        else:
            console.print(f"[red]Transcript not found: {args.transcript}[/red]")
        return
        
    # Handle list
    if args.list:
        sessions = list(SESSIONS_DIR.glob("*.json"))
        if sessions:
            table = Table(title="Past Sessions")
            table.add_column("ID")
            table.add_column("Task")
            table.add_column("Date")
            for s in sorted(sessions, reverse=True)[:10]:
                with open(s) as f:
                    data = json.load(f)
                table.add_row(s.stem, data.get("task", "")[:50], data.get("started", ""))
            console.print(table)
        else:
            console.print("[dim]No past sessions found.[/dim]")
        return
        
    # Get task
    task = args.task
    if args.file:
        task = Path(args.file).read_text()
    if not task:
        if args.claude_code or args.interactive or args.smart:
            # In Claude Code mode, task must be provided (no interactive prompt)
            console.print("[red]No task provided. In Claude Code mode, task must be specified.[/red]")
            return
        task = Prompt.ask("[bold]What would you like the council to work on?[/bold]")

    if not task:
        console.print("[red]No task provided.[/red]")
        return

    # Determine if running in Claude Code mode
    # --interactive or --smart implies --claude-code
    claude_code_mode = args.claude_code or args.interactive or args.smart

    # Run session
    session = CouncilSession(task, config, claude_code_mode=claude_code_mode)
    await session.run()


if __name__ == "__main__":
    asyncio.run(main())
