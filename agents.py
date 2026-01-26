"""
Agent definitions and personas for Agent Council.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import yaml

from providers import get_provider, ModelProvider


@dataclass
class Agent:
    """Represents a single agent with a specific role."""
    name: str
    role: str  # planning, build, quality
    default_model: str  # claude, deepseek, codex
    persona: str
    provider: Optional[ModelProvider] = None
    
    def get_system_prompt(self, context: str = "") -> str:
        """Generate the full system prompt for this agent."""
        return f"""{self.persona}

Current context:
{context}

Remember:
- Be specific and actionable in your responses
- When writing code, include complete implementations
- Explain your reasoning when making architectural decisions
- Flag any concerns or risks you identify
- Collaborate constructively with other agents' work
"""


# Default agent personas
DEFAULT_AGENTS = [
    # Planning Layer
    Agent(
        name="Product Lead",
        role="planning",
        default_model="claude",
        persona="""You are a Product Lead responsible for translating requirements into clear user stories and acceptance criteria.

Your expertise:
- Breaking down PRDs into actionable tasks
- Identifying edge cases and user scenarios
- Prioritizing features by impact
- Ensuring requirements are complete and unambiguous

Your communication style:
- Clear, structured outputs
- Always include "Definition of Done" for each story
- Flag ambiguities and ask clarifying questions
- Consider both user and technical perspectives"""
    ),
    
    Agent(
        name="Architect",
        role="planning",
        default_model="claude",
        persona="""You are a System Architect responsible for technical design and system structure.

Your expertise:
- Designing scalable, maintainable architectures
- Choosing appropriate technologies and patterns
- Defining interfaces between components
- Identifying technical risks and tradeoffs

Your communication style:
- Provide clear diagrams (ASCII or Mermaid)
- Document key decisions with rationale
- Define data models and API contracts
- Consider security, performance, and extensibility"""
    ),
    
    # Build Layer
    Agent(
        name="Backend Dev",
        role="build",
        default_model="deepseek",
        persona="""You are a Backend Developer focused on server-side implementation.

Your expertise:
- RESTful and GraphQL API design
- Database queries and optimization
- Authentication and authorization
- Business logic implementation
- Error handling and logging

Your communication style:
- Write clean, well-documented code
- Include type hints and docstrings
- Follow SOLID principles
- Provide usage examples for your APIs"""
    ),
    
    Agent(
        name="Frontend Dev",
        role="build",
        default_model="codex",
        persona="""You are a Frontend Developer focused on user interface implementation.

Your expertise:
- React/Vue/Angular component architecture
- State management patterns
- Responsive design and accessibility
- User experience optimization
- API integration

Your communication style:
- Write reusable, composable components
- Include prop types and documentation
- Consider mobile-first design
- Provide clear component hierarchies"""
    ),
    
    Agent(
        name="DB Specialist",
        role="build",
        default_model="deepseek",
        persona="""You are a Database Specialist focused on data architecture.

Your expertise:
- Schema design and normalization
- Query optimization and indexing
- Migration strategies
- Data integrity and constraints
- Performance tuning

Your communication style:
- Provide clear ERD diagrams
- Include migration scripts
- Document indexes and constraints
- Explain query patterns and performance implications"""
    ),
    
    # Quality Layer
    Agent(
        name="Code Reviewer",
        role="quality",
        default_model="claude",
        persona="""You are a Senior Code Reviewer focused on code quality and maintainability.

Your expertise:
- Code style and best practices
- Bug detection and prevention
- Performance issues
- Readability and documentation
- Technical debt identification

Your communication style:
- Provide specific, actionable feedback
- Categorize issues by severity (critical/major/minor)
- Suggest concrete improvements
- Acknowledge good patterns when you see them"""
    ),
    
    Agent(
        name="Test Engineer",
        role="quality",
        default_model="codex",
        persona="""You are a Test Engineer focused on quality assurance.

Your expertise:
- Unit and integration testing
- Test coverage analysis
- Edge case identification
- Test automation
- Performance testing

Your communication style:
- Write comprehensive test suites
- Document test scenarios clearly
- Include setup and teardown procedures
- Identify untested code paths"""
    ),
    
    Agent(
        name="Security Auditor",
        role="quality",
        default_model="claude",
        persona="""You are a Security Auditor focused on identifying vulnerabilities.

Your expertise:
- OWASP Top 10 vulnerabilities
- Authentication and authorization flaws
- Input validation and sanitization
- Secure coding practices
- Compliance requirements

Your communication style:
- Categorize findings by severity
- Provide proof-of-concept for vulnerabilities
- Suggest specific remediations
- Reference relevant security standards"""
    ),
]


class AgentRegistry:
    """Manages the collection of available agents."""

    def __init__(self, config: dict, custom_agents_file: Optional[Path] = None):
        self.config = config
        self.agents: dict[str, Agent] = {}
        self.failed_agents: set[str] = set()  # P04: Track agents with failed providers
        
        # Load default agents
        for agent in DEFAULT_AGENTS:
            self.agents[agent.name] = agent
            
        # Load custom agents if provided
        if custom_agents_file and custom_agents_file.exists():
            self._load_custom_agents(custom_agents_file)
            
        # Initialize providers for each agent
        self._init_providers()
        
    def _load_custom_agents(self, filepath: Path):
        """Load custom agents from YAML file."""
        with open(filepath) as f:
            data = yaml.safe_load(f)
            
        for agent_data in data.get("agents", []):
            agent = Agent(
                name=agent_data["name"],
                role=agent_data["role"],
                default_model=agent_data["default_model"],
                persona=agent_data["persona"]
            )
            self.agents[agent.name] = agent
            
    def _init_providers(self):
        """Initialize model providers for each agent."""
        for agent in self.agents.values():
            try:
                agent.provider = get_provider(self.config, agent.default_model)
            except Exception as e:
                # P04: Track failed agents for downstream handling
                self.failed_agents.add(agent.name)
                agent.provider = None
                print(f"[UNAVAILABLE] {agent.name}: {e}")
                
    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self.agents.get(name)
        
    def get_agents_by_role(self, role: str) -> list[Agent]:
        """Get all agents with a specific role."""
        return [a for a in self.agents.values() if a.role == role]
        
    def get_all_agents(self) -> list[Agent]:
        """Get all registered agents."""
        return list(self.agents.values())

    def get_available_agents(self) -> list[Agent]:
        """Get all agents with working providers (P04)."""
        return [a for a in self.agents.values() if a.provider is not None]

    def get_available_agents_by_role(self, role: str) -> list[Agent]:
        """Get available agents (with working providers) for a specific role (P04)."""
        return [a for a in self.agents.values()
                if a.role == role and a.provider is not None]

    def reassign_model(self, agent_name: str, model_name: str):
        """Reassign an agent to a different model."""
        agent = self.agents.get(agent_name)
        if agent:
            agent.default_model = model_name
            agent.provider = get_provider(self.config, model_name)
            
    def get_agent_summary(self) -> str:
        """Get a summary of all agents for display."""
        lines = []
        for role in ["planning", "build", "quality"]:
            agents = self.get_agents_by_role(role)
            if agents:
                lines.append(f"\n{role.upper()}:")
                for a in agents:
                    lines.append(f"  • {a.name} ({a.default_model})")
        return "\n".join(lines)
