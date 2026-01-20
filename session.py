"""
Session management for Council v2.

WHY THIS FILE EXISTS:
--------------------
Sessions track the state of a deliberation workflow:
- What task is being worked on
- Which phase we're in (role_deliberation, planning, execution, review)
- Results from each phase
- Ability to pause/resume

This allows users to:
- Stop mid-workflow and resume later
- Review past sessions
- Iterate on specific phases

WORKFLOW PHASES:
---------------
1. role_deliberation - Assign roles to models
2. planning - Create implementation plan
3. execution - Execute the plan
4. review - Quality review of outputs
5. complete - Session finished

PERSISTENCE:
-----------
Sessions are stored as JSON files in ~/.council/v2/sessions/
Each session has a unique ID and can be loaded by that ID.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from schemas import (
    Plan,
    RoleDeliberationResult,
    ExecutionResult,
    CodeReview,
    DeliberationTranscript,
)


# =============================================================================
# SESSION DATA CLASSES
# =============================================================================

@dataclass
class SessionTranscriptEntry:
    """
    A single entry in the session transcript.

    The transcript is a log of everything that happened in the session,
    including user interactions, model responses, and tool calls.
    """
    timestamp: str
    event_type: str  # "phase_started", "model_response", "user_input", "checkpoint", etc.
    phase: str
    content: dict
    model: Optional[str] = None


@dataclass
class Session:
    """
    Complete session state.

    This is the main data structure that tracks everything about
    a deliberation session.
    """
    # Identity
    session_id: str
    created_at: str
    updated_at: str

    # Task
    task: str
    task_source: str = "direct"  # "direct", "file", "resumed"

    # Current state
    phase: str = "initialized"  # initialized, role_deliberation, planning, execution, review, complete

    # Results from each phase (stored as dicts for JSON serialization)
    role_assignments: Optional[dict] = None
    plan: Optional[dict] = None
    execution_result: Optional[dict] = None
    code_review: Optional[dict] = None

    # Transcript of all events
    transcript: list = field(default_factory=list)

    # Cost tracking
    total_tokens: int = 0
    total_cost: float = 0.0

    # Configuration used
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert session to a dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Create a Session from a dictionary."""
        return cls(**data)

    def add_transcript_entry(
        self,
        event_type: str,
        content: dict,
        model: Optional[str] = None
    ) -> None:
        """Add an entry to the session transcript."""
        entry = SessionTranscriptEntry(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            phase=self.phase,
            content=content,
            model=model
        )
        self.transcript.append(asdict(entry))
        self.updated_at = datetime.now().isoformat()

    def set_role_assignments(self, result: RoleDeliberationResult) -> None:
        """Store role deliberation results."""
        self.role_assignments = result.model_dump()
        self.add_transcript_entry("phase_completed", {
            "phase": "role_deliberation",
            "assignments_count": len(result.assignments)
        })

    def set_plan(self, plan: Plan) -> None:
        """Store the implementation plan."""
        self.plan = plan.model_dump()
        self.add_transcript_entry("phase_completed", {
            "phase": "planning",
            "steps_count": len(plan.steps)
        })

    def set_execution_result(self, result: ExecutionResult) -> None:
        """Store execution results."""
        self.execution_result = result.model_dump()
        self.add_transcript_entry("phase_completed", {
            "phase": "execution",
            "success": result.success,
            "files_created": len(result.files_created)
        })

    def set_code_review(self, review: CodeReview) -> None:
        """Store code review results."""
        self.code_review = review.model_dump()
        self.add_transcript_entry("phase_completed", {
            "phase": "review",
            "approved": review.approved,
            "issues_count": len(review.issues)
        })

    def get_role_assignments(self) -> Optional[RoleDeliberationResult]:
        """Get role assignments as Pydantic model."""
        if self.role_assignments:
            return RoleDeliberationResult.model_validate(self.role_assignments)
        return None

    def get_plan(self) -> Optional[Plan]:
        """Get plan as Pydantic model."""
        if self.plan:
            return Plan.model_validate(self.plan)
        return None

    def get_execution_result(self) -> Optional[ExecutionResult]:
        """Get execution result as Pydantic model."""
        if self.execution_result:
            return ExecutionResult.model_validate(self.execution_result)
        return None

    def get_code_review(self) -> Optional[CodeReview]:
        """Get code review as Pydantic model."""
        if self.code_review:
            return CodeReview.model_validate(self.code_review)
        return None

    def update_costs(self, tokens: int, cost: float) -> None:
        """Update token and cost tracking."""
        self.total_tokens += tokens
        self.total_cost += cost
        self.updated_at = datetime.now().isoformat()


# =============================================================================
# SESSION MANAGER
# =============================================================================

class SessionManager:
    """
    Manages session persistence and lifecycle.

    Sessions are stored as JSON files in the sessions directory.
    Each session has a unique ID that can be used to load it later.

    Usage:
        manager = SessionManager()
        session = manager.create("Build a REST API")
        manager.save(session)

        # Later...
        session = manager.load("abc12345")
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the session manager.

        Args:
            base_path: Base directory for session storage.
                      Defaults to ~/.council/v2/sessions
        """
        self.base_path = base_path or Path.home() / ".council" / "v2" / "sessions"
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return self.base_path / f"{session_id}.json"

    def create(self, task: str, task_source: str = "direct", config: dict = None) -> Session:
        """
        Create a new session.

        Args:
            task: The task description
            task_source: Where the task came from ("direct", "file", "resumed")
            config: Configuration to use for this session

        Returns:
            New Session instance
        """
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        session = Session(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            task=task,
            task_source=task_source,
            phase="initialized",
            config=config or {}
        )

        session.add_transcript_entry("session_created", {
            "task": task,
            "task_source": task_source
        })

        return session

    def save(self, session: Session) -> None:
        """
        Save a session to disk.

        Args:
            session: The session to save
        """
        session.updated_at = datetime.now().isoformat()
        path = self._session_path(session.session_id)

        with open(path, "w") as f:
            json.dump(session.to_dict(), f, indent=2, default=str)

    def load(self, session_id: str) -> Session:
        """
        Load a session from disk.

        Args:
            session_id: The session ID to load

        Returns:
            The loaded Session

        Raises:
            FileNotFoundError: If session doesn't exist
        """
        path = self._session_path(session_id)

        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        with open(path) as f:
            data = json.load(f)

        return Session.from_dict(data)

    def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return self._session_path(session_id).exists()

    def delete(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if didn't exist
        """
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_all(self) -> list[dict]:
        """
        List all sessions with summary info.

        Returns:
            List of dicts with session_id, task, phase, created_at
        """
        sessions = []

        for path in sorted(self.base_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(path) as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": data.get("session_id", path.stem),
                        "task": data.get("task", ""),
                        "phase": data.get("phase", "unknown"),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "total_tokens": data.get("total_tokens", 0),
                        "total_cost": data.get("total_cost", 0.0),
                    })
            except (json.JSONDecodeError, KeyError):
                # Skip invalid session files
                continue

        return sessions

    def list_resumable(self) -> list[dict]:
        """
        List sessions that can be resumed (not complete/failed).

        Returns:
            List of session summaries
        """
        return [
            s for s in self.list_all()
            if s.get("phase") not in ("complete", "failed", "aborted")
        ]

    def cleanup_old(self, days: int = 30) -> int:
        """
        Delete sessions older than specified days.

        Args:
            days: Delete sessions older than this many days

        Returns:
            Number of sessions deleted
        """
        import time
        cutoff = time.time() - (days * 24 * 60 * 60)
        deleted = 0

        for path in self.base_path.glob("*.json"):
            if path.stat().st_mtime < cutoff:
                path.unlink()
                deleted += 1

        return deleted


# =============================================================================
# WORKFLOW STATE MACHINE
# =============================================================================

class WorkflowPhase:
    """Constants for workflow phases."""
    INITIALIZED = "initialized"
    ROLE_DELIBERATION = "role_deliberation"
    PLANNING = "planning"
    EXECUTION = "execution"
    REVIEW = "review"
    COMPLETE = "complete"
    FAILED = "failed"
    ABORTED = "aborted"


# Valid phase transitions
PHASE_TRANSITIONS = {
    WorkflowPhase.INITIALIZED: [WorkflowPhase.ROLE_DELIBERATION, WorkflowPhase.PLANNING],
    WorkflowPhase.ROLE_DELIBERATION: [WorkflowPhase.PLANNING, WorkflowPhase.FAILED, WorkflowPhase.ABORTED],
    WorkflowPhase.PLANNING: [WorkflowPhase.EXECUTION, WorkflowPhase.COMPLETE, WorkflowPhase.ROLE_DELIBERATION, WorkflowPhase.FAILED, WorkflowPhase.ABORTED],
    WorkflowPhase.EXECUTION: [WorkflowPhase.REVIEW, WorkflowPhase.PLANNING, WorkflowPhase.FAILED, WorkflowPhase.ABORTED],
    WorkflowPhase.REVIEW: [WorkflowPhase.COMPLETE, WorkflowPhase.EXECUTION, WorkflowPhase.FAILED, WorkflowPhase.ABORTED],
    WorkflowPhase.COMPLETE: [],
    WorkflowPhase.FAILED: [],
    WorkflowPhase.ABORTED: [],
}


def can_transition(current_phase: str, target_phase: str) -> bool:
    """Check if a phase transition is valid."""
    return target_phase in PHASE_TRANSITIONS.get(current_phase, [])


def transition_session(session: Session, new_phase: str) -> bool:
    """
    Transition a session to a new phase.

    Args:
        session: The session to transition
        new_phase: The target phase

    Returns:
        True if transition succeeded, False if invalid

    Raises:
        ValueError: If transition is not allowed
    """
    if not can_transition(session.phase, new_phase):
        raise ValueError(
            f"Invalid phase transition: {session.phase} -> {new_phase}. "
            f"Allowed: {PHASE_TRANSITIONS.get(session.phase, [])}"
        )

    old_phase = session.phase
    session.phase = new_phase
    session.add_transcript_entry("phase_transition", {
        "from": old_phase,
        "to": new_phase
    })

    return True


def get_next_phase(session: Session) -> Optional[str]:
    """
    Get the recommended next phase for a session.

    Args:
        session: The session to check

    Returns:
        The next phase, or None if session is terminal
    """
    phase_order = [
        WorkflowPhase.INITIALIZED,
        WorkflowPhase.ROLE_DELIBERATION,
        WorkflowPhase.PLANNING,
        WorkflowPhase.EXECUTION,
        WorkflowPhase.REVIEW,
        WorkflowPhase.COMPLETE,
    ]

    try:
        current_idx = phase_order.index(session.phase)
        if current_idx < len(phase_order) - 1:
            return phase_order[current_idx + 1]
    except ValueError:
        pass

    return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_session(task: str, config: dict = None) -> Session:
    """
    Quick way to create and save a new session.

    Args:
        task: The task description
        config: Optional configuration

    Returns:
        New Session instance (already saved)
    """
    manager = SessionManager()
    session = manager.create(task, config=config)
    manager.save(session)
    return session


def load_session(session_id: str) -> Session:
    """
    Quick way to load a session.

    Args:
        session_id: The session ID

    Returns:
        The loaded Session
    """
    manager = SessionManager()
    return manager.load(session_id)


def save_session(session: Session) -> None:
    """
    Quick way to save a session.

    Args:
        session: The session to save
    """
    manager = SessionManager()
    manager.save(session)


def list_sessions() -> list[dict]:
    """
    Quick way to list all sessions.

    Returns:
        List of session summaries
    """
    manager = SessionManager()
    return manager.list_all()
