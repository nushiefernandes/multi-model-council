"""
Deliberation - Multi-agent debate orchestration.
Handles role assignment debates, planning discussions, and quality reviews.
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Optional
from pathlib import Path

from agents import AgentRegistry, Agent
from chairman import Chairman
from costs import CostTracker


# Valid model names for role assignments
VALID_MODELS = {"claude", "deepseek", "codex"}

# Default assignments as fallback
DEFAULT_ASSIGNMENTS = {
    "Product Lead": "claude",
    "Architect": "claude",
    "Backend Dev": "deepseek",
    "Frontend Dev": "codex",
    "DB Specialist": "deepseek",
    "Code Reviewer": "claude",
    "Test Engineer": "codex",
    "Security Auditor": "claude"
}


class Deliberation:
    """Orchestrates multi-agent deliberations and debates."""
    
    def __init__(
        self,
        agents: AgentRegistry,
        chairman: Chairman,
        cost_tracker: CostTracker
    ):
        self.agents = agents
        self.chairman = chairman
        self.cost_tracker = cost_tracker
        self.current_plan = None
        
    async def _agent_respond(
        self, 
        agent: Agent, 
        prompt: str, 
        context: str = ""
    ) -> dict:
        """Get a response from a single agent."""
        system = agent.get_system_prompt(context)
        
        result = await agent.provider.complete(
            messages=[{"role": "user", "content": prompt}],
            system=system
        )
        
        # Track costs
        self.cost_tracker.add(
            phase="deliberation",
            model=agent.default_model,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"]
        )
        
        return {
            "agent": agent.name,
            "model": agent.default_model,
            "content": result["content"],
            "tokens": result["input_tokens"] + result["output_tokens"]
        }
        
    async def deliberate_roles(self, task: str, prior_feedback: str = None) -> dict:
        """Have models deliberate on role assignments.

        Args:
            task: The task description
            prior_feedback: Optional feedback from previous deliberation round (P09)
        """
        transcript = []

        # P09: Incorporate prior feedback if provided
        task_context = task
        if prior_feedback:
            task_context = f"{task}\n\nPrevious feedback to incorporate:\n{prior_feedback}"

        # Round 1: Each model proposes role assignments
        initial_prompt = f"""We are forming an agent council to work on this task:

{task_context}

Available roles:
- Product Lead: PRD analysis, user stories
- Architect: System design, technical decisions
- Backend Dev: API, business logic, server-side
- Frontend Dev: UI components, user experience
- DB Specialist: Schema design, queries, migrations
- Code Reviewer: Quality, bugs, maintainability
- Test Engineer: Testing, coverage, edge cases
- Security Auditor: Vulnerabilities, auth, compliance

Available models:
- Claude (Anthropic): Best at reasoning, architecture, nuanced analysis
- DeepSeek (local): Good at structured code, efficient patterns, DB work
- Codex (OpenAI): Strong at UI/frontend, rapid generation, tests

Propose which model should take which roles for THIS specific task.
Explain your reasoning based on the task requirements.
Be willing to take roles that play to your strengths."""

        # Get proposals from each model type
        model_agents = {
            "claude": self.agents.get_agent("Architect"),  # Representative for Claude
            "deepseek": self.agents.get_agent("Backend Dev"),  # Representative for DeepSeek
            "codex": self.agents.get_agent("Frontend Dev"),  # Representative for Codex
        }
        
        proposals = {}
        for model_name, agent in model_agents.items():
            if agent and agent.provider:
                result = await self._agent_respond(
                    agent, 
                    initial_prompt,
                    f"You are representing the {model_name} model in this discussion."
                )
                proposals[model_name] = result["content"]
                transcript.append({
                    "round": 1,
                    "speaker": f"{model_name} (via {agent.name})",
                    "content": result["content"]
                })
                
        # Round 2: Cross-critique and refinement (P08: fix placeholder bug)
        CRITIQUE_TEMPLATE = """Other models have proposed:

{other_proposals}

Review their proposals. Do you agree? Disagree with anything?
Suggest any adjustments to the role assignments.
Focus on what's best for the task, not self-promotion."""

        critiques = {}
        for model_name, agent in model_agents.items():
            if agent and agent.provider:
                # Show other proposals (exclude self)
                other_proposals = {k: v for k, v in proposals.items() if k != model_name}
                critique_prompt = CRITIQUE_TEMPLATE.replace(
                    "{other_proposals}",
                    self._format_proposals(other_proposals)
                )
                result = await self._agent_respond(
                    agent,
                    critique_prompt,
                    f"You are representing {model_name}. Provide constructive feedback."
                )
                critiques[model_name] = result["content"]
                transcript.append({
                    "round": 2,
                    "speaker": f"{model_name} (critique)",
                    "content": result["content"]
                })
                
        # Round 3: Chairman synthesizes final assignments
        synthesis_prompt = f"""Based on this deliberation about role assignments:

Initial proposals:
{self._format_proposals(proposals)}

Critiques and adjustments:
{self._format_proposals(critiques)}

Task: {task}

Synthesize the final role assignments. You MUST output a JSON object with your assignments.

Output format (use exactly this structure):
```json
{{
    "assignments": {{
        "Product Lead": "claude",
        "Architect": "claude",
        "Backend Dev": "deepseek",
        "Frontend Dev": "codex",
        "DB Specialist": "deepseek",
        "Code Reviewer": "claude",
        "Test Engineer": "codex",
        "Security Auditor": "claude"
    }},
    "reasoning": {{
        "Product Lead": "why this model was chosen",
        "Architect": "why this model was chosen"
    }},
    "consensus_notes": "any dissenting opinions or strong consensus areas"
}}
```

Valid model values are: "claude", "deepseek", "codex"
Include reasoning for each assignment based on the deliberation."""

        chairman_result = await self.chairman._complete(synthesis_prompt)
        transcript.append({
            "round": 3,
            "speaker": "Chairman (synthesis)",
            "content": chairman_result
        })
        
        # Parse assignments (simplified - would use structured output in production)
        assignments = self._parse_assignments(chairman_result)
        
        # Apply assignments
        for role_name, model_name in assignments.items():
            agent = self.agents.get_agent(role_name)
            if agent:
                self.agents.reassign_model(role_name, model_name)
                
        return {
            "full_transcript": self._format_transcript(transcript),
            "summary": self._summarize_deliberation(transcript, "role assignment"),
            "assignments": assignments
        }
        
    async def deliberate_plan(self, task: str) -> dict:
        """Have agents deliberate on the project plan."""
        transcript = []
        
        # Planning agents propose
        planning_agents = self.agents.get_agents_by_role("planning")
        
        # P01: Initialize prd_content with default before conditional
        prd_content = f"Task requirements:\n{task}"

        # Product Lead first
        product_lead = self.agents.get_agent("Product Lead")
        if product_lead and product_lead.provider:
            result = await self._agent_respond(
                product_lead,
                f"""Analyze this task and create user stories:

{task}

Output:
1. User stories with acceptance criteria
2. Prioritized feature list
3. Out-of-scope items
4. Questions/ambiguities""",
                "Focus on completeness and clarity."
            )
            transcript.append({"speaker": "Product Lead", "content": result["content"]})
            prd_content = result["content"]
            
        # P02: Initialize architecture with default before conditional
        architecture = "No architecture provided. Use your best judgment based on the requirements."

        # Architect proposes technical design
        architect = self.agents.get_agent("Architect")
        if architect and architect.provider:
            result = await self._agent_respond(
                architect,
                f"""Based on these requirements, design the system:

Requirements:
{prd_content}

Output:
1. High-level architecture (ASCII diagram)
2. Technology choices with rationale
3. Component breakdown
4. Data models
5. API contracts
6. Task breakdown by agent""",
                "Consider the task requirements and team capabilities."
            )
            transcript.append({"speaker": "Architect", "content": result["content"]})
            architecture = result["content"]

        # Build agents review and refine (P04: use available agents only)
        build_agents = self.agents.get_available_agents_by_role("build")
        for agent in build_agents:
            result = await self._agent_respond(
                agent,
                f"""Review this architecture from your perspective:

{architecture}

Comment on:
1. Feasibility of your assigned components
2. Suggested improvements
3. Concerns or risks
4. Time/effort estimates""",
                f"You are {agent.name}. Focus on your domain."
            )
            transcript.append({"speaker": agent.name, "content": result["content"]})
            
        # Chairman synthesizes plan
        plan_synthesis = await self.chairman._complete(
            f"""Synthesize a final plan from this deliberation:

{self._format_transcript(transcript)}

Create a structured plan. You MUST output a JSON object with the plan.

Output format (use exactly this structure):
```json
{{
    "overview": "Brief description of what we're building",
    "architecture_decisions": [
        "Key decision 1",
        "Key decision 2"
    ],
    "agent_tasks": {{
        "Backend Dev": {{
            "summary": "What this agent will build",
            "description": "Detailed description of tasks",
            "requirements": "Specific requirements",
            "interfaces": "APIs/interfaces to implement",
            "dependencies": ["other agents this depends on"]
        }},
        "Frontend Dev": {{
            "summary": "...",
            "description": "...",
            "requirements": "...",
            "interfaces": "...",
            "dependencies": []
        }},
        "DB Specialist": {{
            "summary": "...",
            "description": "...",
            "requirements": "...",
            "interfaces": "...",
            "dependencies": []
        }}
    }},
    "execution_order": ["DB Specialist", "Backend Dev", "Frontend Dev"],
    "risks": ["Risk 1", "Risk 2"],
    "success_criteria": ["Criterion 1", "Criterion 2"]
}}
```

Only include build agents (Backend Dev, Frontend Dev, DB Specialist) in agent_tasks.
Be specific about what each agent should implement."""
        )
        
        transcript.append({"speaker": "Chairman", "content": plan_synthesis})
        
        # Parse into structured plan
        plan = self._parse_plan(plan_synthesis, transcript)
        self.current_plan = plan
        
        return {
            "full_transcript": self._format_transcript(transcript),
            "summary": self._summarize_deliberation(transcript, "planning"),
            "plan": plan
        }
        
    async def revise_plan(self, correction: str) -> dict:
        """Revise plan based on user correction."""
        # P03: Guard when Architect is unavailable
        architect = self.agents.get_agent("Architect")
        if not architect or not architect.provider:
            return {
                "error": "Architect unavailable for plan revision",
                "full_transcript": "",
                "summary": "Cannot revise - Architect agent unavailable",
                "plan": self.current_plan  # Return existing plan
            }

        transcript = []

        result = await self._agent_respond(
            architect,
            f"""Revise the plan based on this feedback:

Current plan:
{self.current_plan}

Correction:
{correction}

Provide updated plan addressing all feedback.""",
            "Incorporate the feedback while maintaining coherence."
        )
        transcript.append({"speaker": "Architect (revision)", "content": result["content"]})

        plan = self._parse_plan(result["content"], transcript)
        self.current_plan = plan
        
        return {
            "full_transcript": self._format_transcript(transcript),
            "summary": f"Plan revised based on feedback:\n{result['content'][:500]}...",
            "plan": plan
        }
        
    async def execute_agent_task(
        self, 
        agent_name: str, 
        task: dict,
        workspace_dir: Path
    ) -> dict:
        """Execute a single agent's task."""
        agent = self.agents.get_agent(agent_name)
        if not agent:
            return {"error": f"Agent {agent_name} not found"}
            
        prompt = f"""Execute this task:

{task.get('description', '')}

Requirements:
{task.get('requirements', '')}

Interfaces to implement:
{task.get('interfaces', '')}

Output complete, working code. Include:
1. All necessary files
2. Clear file structure
3. Documentation/comments
4. Example usage

IMPORTANT: Format each code block with the filename as a comment on the first line:
```python
# path/to/filename.py
your code here
```

For JavaScript/TypeScript files, use:
```javascript
// path/to/filename.js
your code here
```

Your workspace directory is: {workspace_dir}"""

        result = await self._agent_respond(agent, prompt, "Write production-ready code.")
        
        # Track cost for build phase
        self.cost_tracker.add(
            phase="build",
            model=agent.default_model,
            input_tokens=result.get("tokens", 0) // 2,
            output_tokens=result.get("tokens", 0) // 2
        )
        
        return {
            "agent": agent_name,
            "output": result["content"],
            "files": self._extract_files(result["content"]),
            "workspace": str(workspace_dir)
        }
        
    async def rework_agent_task(
        self, 
        agent_name: str, 
        correction: str,
        original_task: dict
    ) -> dict:
        """Have an agent rework their task based on feedback."""
        agent = self.agents.get_agent(agent_name)
        if not agent:
            return {"error": f"Agent {agent_name} not found"}
            
        prompt = f"""Rework your previous output based on this feedback:

Original task:
{original_task.get('description', '')}

Feedback:
{correction}

Provide corrected implementation addressing all feedback."""

        result = await self._agent_respond(agent, prompt, "Address the feedback thoroughly.")
        
        return {
            "agent": agent_name,
            "output": result["content"],
            "files": self._extract_files(result["content"]),
            "reworked": True
        }
        
    async def quality_review(self, outputs: dict) -> dict:
        """Run quality review on all outputs."""
        transcript = []
        all_issues = []
        
        # P04: Use available agents only
        quality_agents = self.agents.get_available_agents_by_role("quality")
        
        # Compile all code for review
        all_code = "\n\n".join([
            f"=== {name} ===\n{out.get('output', '')}"
            for name, out in outputs.items()
        ])
        
        for agent in quality_agents:
            result = await self._agent_respond(
                agent,
                f"""Review this code from your perspective:

{all_code[:8000]}

You MUST output your review as a JSON object.

Output format:
```json
{{
    "issues": [
        {{
            "severity": "critical",
            "description": "What the issue is",
            "location": "file or function name",
            "fix": "How to fix it",
            "assigned_to": "Backend Dev"
        }},
        {{
            "severity": "major",
            "description": "...",
            "location": "...",
            "fix": "...",
            "assigned_to": "Frontend Dev"
        }}
    ],
    "overall_assessment": "Your overall quality assessment",
    "positive_notes": ["Good patterns observed"]
}}
```

Severity must be one of: "critical", "major", "minor"
assigned_to must be one of: "Backend Dev", "Frontend Dev", "DB Specialist"

Be thorough but constructive. Include specific locations and fixes.""",
                f"You are {agent.name}. Output valid JSON."
            )
            
            transcript.append({"speaker": agent.name, "content": result["content"]})
            
            # Extract issues (simplified parsing)
            issues = self._extract_issues(result["content"], agent.name)
            all_issues.extend(issues)
            
            self.cost_tracker.add(
                phase="quality",
                model=agent.default_model,
                input_tokens=result.get("tokens", 0) // 2,
                output_tokens=result.get("tokens", 0) // 2
            )
            
        return {
            "full_transcript": self._format_transcript(transcript),
            "issues": all_issues,
            "summary": f"Found {len(all_issues)} issues across {len(quality_agents)} reviewers."
        }
        
    async def rework_issues(self, outputs: dict, issues: list) -> dict:
        """Send issues back to relevant agents for fixing."""
        # Group issues by agent
        issues_by_agent = {}
        for issue in issues:
            agent = issue.get("assigned_to", "Backend Dev")
            if agent not in issues_by_agent:
                issues_by_agent[agent] = []
            issues_by_agent[agent].append(issue)
            
        # Have each agent fix their issues
        for agent_name, agent_issues in issues_by_agent.items():
            agent = self.agents.get_agent(agent_name)
            if not agent:
                continue
                
            result = await self._agent_respond(
                agent,
                f"""Fix these issues in your code:

Issues:
{agent_issues}

Original output:
{outputs.get(agent_name, {}).get('output', '')[:4000]}

Provide corrected code.""",
                "Fix all listed issues."
            )
            
            outputs[agent_name]["output"] = result["content"]
            outputs[agent_name]["fixed_issues"] = [i["description"] for i in agent_issues]
            
        return outputs
        
    async def integrate_outputs(self, outputs: dict) -> dict:
        """Have Architect integrate all outputs."""
        architect = self.agents.get_agent("Architect")

        # P06: Guard when Architect is unavailable
        if not architect or not architect.provider:
            # Fallback: return combined outputs without integration
            combined = "\n\n".join([
                f"=== {name} ===\n{out.get('output', '')}"
                for name, out in outputs.items()
            ])
            return {
                "summary": f"Integration skipped (Architect unavailable).\n\n{combined[:4000]}",
                "components": list(outputs.keys()),
                "integrated": False
            }

        combined = "\n\n".join([
            f"=== {name} ===\n{out.get('output', '')[:2000]}"
            for name, out in outputs.items()
        ])

        result = await self._agent_respond(
            architect,
            f"""Integrate these components into a cohesive project:

{combined}

Provide:
1. Project structure
2. Integration points
3. Setup instructions
4. Run instructions
5. Any final adjustments needed""",
            "Create a complete, runnable project."
        )

        return {
            "summary": result["content"],
            "components": list(outputs.keys()),
            "integrated": True
        }
        
    # Helper methods
    def _format_proposals(self, proposals: dict) -> str:
        return "\n\n".join([f"**{k}**:\n{v}" for k, v in proposals.items()])
        
    def _format_transcript(self, transcript: list) -> str:
        lines = []
        for entry in transcript:
            speaker = entry.get("speaker", "Unknown")
            content = entry.get("content", "")
            lines.append(f"### {speaker}\n\n{content}\n")
        return "\n".join(lines)
        
    def _summarize_deliberation(self, transcript: list, topic: str) -> str:
        """Create a brief summary of deliberation."""
        speakers = [e.get("speaker", "") for e in transcript]
        return f"Deliberation on {topic} completed with input from: {', '.join(speakers)}"
        
    def _parse_assignments(self, synthesis: str) -> dict:
        """Parse role assignments from synthesis text."""
        assignments = None

        # Try to extract JSON from markdown code block
        json_match = re.search(r'```json\s*(.*?)\s*```', synthesis, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                assignments = data.get("assignments", data)
            except json.JSONDecodeError:
                pass

        # Try direct JSON parse if no code block found
        if assignments is None:
            try:
                data = json.loads(synthesis)
                assignments = data.get("assignments", data)
            except json.JSONDecodeError:
                pass

        # Try to find JSON-like structure in text
        if assignments is None:
            brace_match = re.search(r'\{[^{}]*"assignments"[^{}]*\{.*?\}.*?\}', synthesis, re.DOTALL)
            if brace_match:
                try:
                    data = json.loads(brace_match.group(0))
                    assignments = data.get("assignments", data)
                except json.JSONDecodeError:
                    pass

        # Validate and use parsed assignments, or fall back to defaults
        if assignments and isinstance(assignments, dict):
            # Validate each assignment
            validated = {}
            for role, model in assignments.items():
                if role in DEFAULT_ASSIGNMENTS:
                    if model in VALID_MODELS:
                        validated[role] = model
                    else:
                        print(f"Warning: Invalid model '{model}' for {role}, using default")
                        validated[role] = DEFAULT_ASSIGNMENTS[role]

            # Fill in any missing roles with defaults
            for role in DEFAULT_ASSIGNMENTS:
                if role not in validated:
                    validated[role] = DEFAULT_ASSIGNMENTS[role]

            print(f"Parsed role assignments from deliberation: {validated}")
            return validated

        print("Warning: Could not parse role assignments, using defaults")
        return DEFAULT_ASSIGNMENTS.copy()
        
    def _parse_plan(self, synthesis: str, transcript: list) -> dict:
        """Parse plan from synthesis."""
        plan = None

        # Try to extract JSON from markdown code block
        json_match = re.search(r'```json\s*(.*?)\s*```', synthesis, re.DOTALL)
        if json_match:
            try:
                plan = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try direct JSON parse
        if plan is None:
            try:
                plan = json.loads(synthesis)
            except json.JSONDecodeError:
                pass

        # Try to find JSON-like structure in text
        if plan is None:
            brace_match = re.search(r'\{[^{}]*"overview".*\}', synthesis, re.DOTALL)
            if brace_match:
                try:
                    plan = json.loads(brace_match.group(0))
                except json.JSONDecodeError:
                    pass

        # Validate and use parsed plan
        if plan and isinstance(plan, dict) and "agent_tasks" in plan:
            # Ensure required fields exist
            validated_plan = {
                "overview": plan.get("overview", synthesis[:500]),
                "agents": list(self.agents.agents.keys()),
                "architecture_decisions": plan.get("architecture_decisions", []),
                "agent_tasks": {},
                "execution_order": plan.get("execution_order", []),
                "risks": plan.get("risks", []),
                "success_criteria": plan.get("success_criteria", [])
            }

            # Validate agent_tasks
            build_agents = [name for name, agent in self.agents.agents.items()
                          if agent.role == "build"]

            for agent_name, task_data in plan.get("agent_tasks", {}).items():
                if agent_name in build_agents:
                    validated_plan["agent_tasks"][agent_name] = {
                        "summary": task_data.get("summary", f"Tasks for {agent_name}"),
                        "description": task_data.get("description", ""),
                        "requirements": task_data.get("requirements", ""),
                        "interfaces": task_data.get("interfaces", ""),
                        "dependencies": task_data.get("dependencies", [])
                    }

            # Add default tasks for any missing build agents
            for agent_name in build_agents:
                if agent_name not in validated_plan["agent_tasks"]:
                    validated_plan["agent_tasks"][agent_name] = {
                        "summary": f"Tasks for {agent_name}",
                        "description": f"Implement {agent_name} responsibilities",
                        "requirements": "",
                        "interfaces": "",
                        "dependencies": []
                    }

            print(f"Parsed plan with {len(validated_plan['agent_tasks'])} agent tasks")
            return validated_plan

        # Fall back to default structure
        print("Warning: Could not parse plan, using default structure")
        return {
            "overview": synthesis[:500],
            "agents": list(self.agents.agents.keys()),
            "architecture_decisions": [],
            "agent_tasks": {
                name: {
                    "summary": f"Tasks for {name}",
                    "description": f"Implement {name} responsibilities",
                    "requirements": "",
                    "interfaces": "",
                    "dependencies": []
                }
                for name in self.agents.agents.keys()
                if self.agents.agents[name].role == "build"
            },
            "execution_order": [],
            "risks": [],
            "success_criteria": []
        }
        
    def _extract_files(self, content: str) -> list:
        """Extract file names from code output."""
        files = []
        for line in content.split("\n"):
            if line.startswith("# ") and ("." in line or "/" in line):
                files.append(line[2:].strip())
        return files
        
    def _extract_issues(self, review: str, reviewer: str) -> list:
        """Extract issues from review text."""
        issues = []
        parsed_review = None

        # Try to extract JSON from markdown code block
        json_match = re.search(r'```json\s*(.*?)\s*```', review, re.DOTALL)
        if json_match:
            try:
                parsed_review = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try direct JSON parse
        if parsed_review is None:
            try:
                parsed_review = json.loads(review)
            except json.JSONDecodeError:
                pass

        # Extract issues from parsed JSON
        if parsed_review and isinstance(parsed_review, dict):
            for issue in parsed_review.get("issues", []):
                if isinstance(issue, dict):
                    severity = issue.get("severity", "minor").lower()
                    if severity not in ["critical", "major", "minor"]:
                        severity = "minor"

                    assigned_to = issue.get("assigned_to", "Backend Dev")
                    if assigned_to not in ["Backend Dev", "Frontend Dev", "DB Specialist"]:
                        assigned_to = "Backend Dev"

                    issues.append({
                        "severity": severity,
                        "description": issue.get("description", "Issue found"),
                        "location": issue.get("location", "unknown"),
                        "fix": issue.get("fix", ""),
                        "reviewer": reviewer,
                        "assigned_to": assigned_to
                    })

            if issues:
                print(f"Parsed {len(issues)} issues from {reviewer}")
                return issues

        # Fall back to keyword extraction if JSON parsing failed
        print(f"Warning: Could not parse issues from {reviewer}, using keyword fallback")
        for severity in ["critical", "major", "minor"]:
            # Look for severity keywords with surrounding context
            pattern = rf'({severity})[:\s]+([^\n.]+)'
            matches = re.findall(pattern, review, re.IGNORECASE)
            for _, description in matches:
                issues.append({
                    "severity": severity,
                    "description": description.strip()[:200],
                    "location": "unknown",
                    "fix": "",
                    "reviewer": reviewer,
                    "assigned_to": "Backend Dev"
                })

        return issues
