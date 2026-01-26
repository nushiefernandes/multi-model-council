"""
Chairman - The primary interface between user and council.
Handles memory, corrections, and alerts.
"""

import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from providers import get_provider, ModelProvider
from alerts import AlertManager


@dataclass
class ChairmanMemory:
    """Persistent memory for the Chairman."""
    user_preferences: dict = field(default_factory=dict)
    feedback_patterns: list = field(default_factory=list)
    project_context: dict = field(default_factory=dict)
    alert_history: list = field(default_factory=list)
    learned_corrections: list = field(default_factory=list)


class Chairman:
    """The Chairman facilitates council deliberations and interfaces with the user."""
    
    def __init__(
        self, 
        config: dict, 
        memory_file: Path,
        alerts: Optional[AlertManager]
    ):
        self.config = config
        self.memory_file = memory_file
        self.alerts = alerts
        self.memory = self._load_memory()
        
        # Initialize the Chairman's model (always Claude for best reasoning)
        self.provider = get_provider(config, "claude")
        
        self.system_prompt = """You are the Chairman of an AI agent council. Your role is to:

1. FACILITATE: Guide deliberations between agents, ensure productive discussions
2. SYNTHESIZE: Distill complex agent outputs into clear summaries for the user
3. ADVISE: Help the user understand tradeoffs and make decisions
4. CORRECT: Help craft precise feedback when the user wants changes
5. REMEMBER: Learn user preferences and apply them proactively
6. ALERT: Flag critical issues that need immediate attention

Communication style:
- Be concise but thorough
- Present information in order of importance
- Offer your recommendation, but respect user's final decision
- Reference past feedback patterns when relevant
- Use clear structure (bullets, sections) for complex information

You have memory of past interactions:
{memory_context}

Always act in the user's best interest while maintaining productive collaboration with the agent team."""

    def _load_memory(self) -> ChairmanMemory:
        """Load memory from file or create new."""
        if self.memory_file.exists():
            with open(self.memory_file) as f:
                data = yaml.safe_load(f) or {}
            return ChairmanMemory(
                user_preferences=data.get("user_preferences", {}),
                feedback_patterns=data.get("feedback_patterns", []),
                project_context=data.get("project_context", {}),
                alert_history=data.get("alert_history", []),
                learned_corrections=data.get("learned_corrections", [])
            )
        return ChairmanMemory()
        
    def _save_memory(self):
        """Save memory to file."""
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_file, "w") as f:
            yaml.dump({
                "user_preferences": self.memory.user_preferences,
                "feedback_patterns": self.memory.feedback_patterns,
                "project_context": self.memory.project_context,
                "alert_history": self.memory.alert_history,
                "learned_corrections": self.memory.learned_corrections
            }, f, default_flow_style=False)
            
    def _get_memory_context(self) -> str:
        """Format memory for inclusion in prompts."""
        lines = []
        
        if self.memory.user_preferences:
            lines.append("User preferences:")
            for k, v in self.memory.user_preferences.items():
                lines.append(f"  - {k}: {v}")
                
        if self.memory.feedback_patterns:
            lines.append("\nPast feedback patterns:")
            for pattern in self.memory.feedback_patterns[-5:]:  # Last 5
                lines.append(f"  - When '{pattern['trigger']}': {pattern['learned']}")
                
        if self.memory.project_context:
            lines.append(f"\nCurrent project: {self.memory.project_context.get('name', 'Unknown')}")
            
        return "\n".join(lines) if lines else "No memory yet - this is our first interaction."
        
    def get_memory_display(self) -> str:
        """Get formatted memory for user viewing."""
        return yaml.dump({
            "user_preferences": self.memory.user_preferences,
            "feedback_patterns": self.memory.feedback_patterns,
            "project_context": self.memory.project_context,
            "learned_corrections": self.memory.learned_corrections
        }, default_flow_style=False)
        
    def clear_memory(self):
        """Clear all memory."""
        self.memory = ChairmanMemory()
        self._save_memory()
        
    async def _complete(self, user_message: str, context: str = "") -> str:
        """Get a completion from the Chairman."""
        system = self.system_prompt.format(memory_context=self._get_memory_context())
        if context:
            system += f"\n\nCurrent context:\n{context}"
            
        result = await self.provider.complete(
            messages=[{"role": "user", "content": user_message}],
            system=system
        )
        return result["content"]
        
    async def analyze_role_assignments(
        self, 
        assignments: dict, 
        task: str
    ) -> str:
        """Analyze proposed role assignments and present to user."""
        prompt = f"""The agents have deliberated and proposed these role assignments for the task:

Task: {task}

Proposed assignments:
{yaml.dump(assignments, default_flow_style=False)}

Please:
1. Summarize why each model was assigned to each role
2. Highlight any particularly good matches
3. Note any concerns or alternatives worth considering
4. Give your recommendation (approve as-is or suggest changes)

Keep it concise - the user has already seen the full deliberation transcript."""

        return await self._complete(prompt)
        
    async def review_plan(self, plan: dict, task: str) -> str:
        """Review and present a plan to the user."""
        # Check for patterns from memory
        relevant_patterns = []
        for pattern in self.memory.feedback_patterns:
            if any(kw in task.lower() for kw in pattern.get("keywords", [])):
                relevant_patterns.append(pattern)
        
        prompt = f"""Review this plan and help me present it to the user:

Original task: {task}

Plan:
{yaml.dump(plan, default_flow_style=False)}

{"Based on past feedback, watch for: " + str([p['learned'] for p in relevant_patterns]) if relevant_patterns else ""}

Please:
1. Highlight the key architectural decisions
2. Note any risks or concerns
3. Estimate complexity/effort for each component
4. Suggest questions the user might want to ask
5. Give your overall assessment

Be conversational - you're helping the user understand and decide."""

        return await self._complete(prompt)
        
    async def craft_correction(self, user_feedback: str, current_plan: dict) -> str:
        """Help craft a correction based on user feedback."""
        prompt = f"""The user has feedback on the plan. Help me craft a clear correction for the agents.

User's feedback: "{user_feedback}"

Current plan:
{yaml.dump(current_plan, default_flow_style=False)}

Please draft a correction that:
1. Clearly states what needs to change
2. Explains why (based on user's feedback)
3. Is actionable for the agents
4. Preserves what's working well

Format the correction so it can be sent directly to the agents."""

        correction = await self._complete(prompt)
        
        # Learn from this feedback for future
        self.memory.learned_corrections.append({
            "date": datetime.now().isoformat(),
            "user_feedback": user_feedback,
            "correction": correction[:200]  # Store summary
        })
        self._save_memory()
        
        return correction
        
    async def craft_agent_correction(
        self, 
        agent_name: str, 
        user_feedback: str, 
        agent_output: dict
    ) -> str:
        """Craft a correction for a specific agent."""
        prompt = f"""The user has feedback for {agent_name}'s output.

User's feedback: "{user_feedback}"

{agent_name}'s output:
{agent_output.get('output', '')[:2000]}

Draft a specific, actionable correction for {agent_name} that addresses the user's concern."""

        return await self._complete(prompt)
        
    async def summarize_agent_output(self, agent_name: str, result: dict) -> str:
        """Summarize an agent's output for user review."""
        prompt = f"""Summarize {agent_name}'s work for the user:

Output:
{result.get('output', '')[:3000]}

Files created: {result.get('files', [])}

Provide:
1. What was accomplished
2. Key implementation decisions
3. Any concerns or items needing attention
4. Questions for the user (if any)

Keep it brief - user can ask for details."""

        return await self._complete(prompt)
        
    async def present_quality_review(self, review_result: dict) -> str:
        """Present quality review results."""
        prompt = f"""Present the quality review results to the user:

Issues found: {len(review_result.get('issues', []))}

Details:
{yaml.dump(review_result, default_flow_style=False)}

Summarize:
1. Overall code quality assessment
2. Critical issues that must be fixed
3. Recommendations for optional improvements
4. Your assessment of readiness for merge"""

        return await self._complete(prompt)
        
    async def recommend_rework(self, issues: list) -> str:
        """Recommend which issues to send back for rework."""
        prompt = f"""Based on these issues, recommend what to send back for rework:

Issues:
{yaml.dump(issues, default_flow_style=False)}

Consider:
- Which are truly critical vs nice-to-have
- Time/cost of fixing vs impact
- Dependencies between issues

Provide your recommendation on which to fix now vs defer."""

        return await self._complete(prompt)
        
    async def present_final(self, final_result: dict, total_cost: float) -> str:
        """Present the final deliverable."""
        prompt = f"""Present the final deliverable to the user:

Result:
{yaml.dump(final_result, default_flow_style=False)}

Total cost: ${total_cost:.4f}

Provide:
1. Summary of what was built
2. How to run/use it
3. Known limitations or future improvements
4. Any final notes

{"Remember: User previously mentioned: " + str(self.memory.user_preferences) if self.memory.user_preferences else ""}"""

        return await self._complete(prompt)
        
    async def learn_from_session(self, transcript: list):
        """Extract learnings from a session for memory."""
        # Find user feedback entries (terminal mode)
        user_entries = [e for e in transcript if e.get("speaker") == "user"]

        # Also find checkpoint decisions (Claude Code mode)
        # These entries have phase names ending with "_checkpoint" or contain action data
        checkpoint_entries = []
        for entry in transcript:
            phase = entry.get("phase", "")
            content = entry.get("content", "")

            # Check if it's a checkpoint-related entry
            if any([
                phase.endswith("_checkpoint"),
                "checkpoint" in phase.lower(),
                isinstance(content, str) and '"action"' in content,
                isinstance(content, dict) and "action" in content
            ]):
                checkpoint_entries.append(entry)

        # Also look for entries with feedback content
        feedback_entries = [
            e for e in transcript
            if e.get("speaker") in ["user", "feedback"]
            or "feedback" in str(e.get("content", "")).lower()[:100]
            or "correction" in str(e.get("content", "")).lower()[:100]
        ]

        # Combine all learning data
        learning_data = user_entries + checkpoint_entries + feedback_entries

        # Deduplicate based on content
        seen_content = set()
        unique_entries = []
        for entry in learning_data:
            content_key = str(entry.get("content", ""))[:200]
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_entries.append(entry)

        if not unique_entries:
            # Even if no explicit feedback, learn from the session structure
            if transcript:
                unique_entries = [{"type": "session_summary", "phases": [e.get("phase", "") for e in transcript]}]
            else:
                return

        prompt = f"""Analyze this session transcript to extract learnings about user preferences:

Session data:
{yaml.dump(unique_entries[:20], default_flow_style=False)}

Extract:
1. Any explicit preferences stated or implied
2. Patterns in what was approved vs corrected (look for action: approve/feedback)
3. Types of feedback given and when
4. Overall working style preferences

Format as structured data I can store. Focus on actionable patterns."""

        analysis = await self._complete(prompt)

        # Store learnings
        self.memory.feedback_patterns.append({
            "date": datetime.now().isoformat(),
            "session_learnings": analysis[:500],
            "entry_count": len(unique_entries),
            "had_corrections": any("feedback" in str(e).lower() or "correction" in str(e).lower()
                                   for e in unique_entries)
        })
        self._save_memory()
        
    def send_critical_alert(self, message: str, context: str = ""):
        """Send a critical alert to the user."""
        if self.alerts:
            self.alerts.critical(f"🚨 {message}", context)
            
        # Log to memory
        self.memory.alert_history.append({
            "date": datetime.now().isoformat(),
            "message": message,
            "context": context[:200]
        })
        self._save_memory()
