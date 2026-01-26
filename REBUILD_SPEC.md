# Council v3 Rebuild Specification

> Comprehensive documentation for rebuilding the multi-model council system from scratch.

---

## 1. Vision & Goals

### What is Council?

Council is a **multi-model AI deliberation system** that orchestrates multiple LLMs (Claude, DeepSeek, Codex/GPT) to collaborate on software engineering tasks. Instead of using a single model, Council leverages the strengths of different models through structured debate and role assignment.

### Core Value Proposition

1. **Model Diversity**: Different models have different strengths
   - Claude: Reasoning, architecture, nuanced analysis
   - DeepSeek: Structured code, efficient patterns, database work
   - Codex/GPT: UI/frontend, rapid generation, tests

2. **Deliberation Over Dictation**: Models debate and critique each other's work, leading to better outcomes than single-model responses

3. **Human-in-the-Loop**: Checkpoints allow human oversight at critical decision points

4. **Chairman Pattern**: A coordinating entity (Claude) synthesizes debates and presents to humans

### Success Criteria

- Agents produce working, file-materialized code
- Deliberations lead to genuinely better decisions than single-model
- System degrades gracefully when models are unavailable
- Clear audit trail via transcripts
- Cost tracking and estimation

---

## 2. Architecture Overview

### Workflow Phases

```
┌─────────────────────────────────────────────────────────────────┐
│                         COUNCIL SESSION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 0: Role Deliberation                                     │
│  ├── Each model proposes role assignments                       │
│  ├── Cross-critique proposals                                   │
│  ├── Chairman synthesizes final assignments                     │
│  └── [CHECKPOINT] Human approves roles                          │
│                                                                  │
│  Phase 1: Planning                                              │
│  ├── Product Lead creates PRD/user stories                      │
│  ├── Architect designs system                                   │
│  ├── Build agents review feasibility                            │
│  ├── Chairman synthesizes plan                                  │
│  └── [CHECKPOINT] Human approves plan                           │
│                                                                  │
│  Phase 2: Build                                                 │
│  ├── Agents work in parallel on assigned tasks                  │
│  ├── [CHECKPOINT] Per-agent output review                       │
│  └── Rework loop if needed                                      │
│                                                                  │
│  Phase 3: Quality Review                                        │
│  ├── Code Reviewer checks quality                               │
│  ├── Security Auditor checks vulnerabilities                    │
│  ├── Test Engineer validates coverage                           │
│  └── [CHECKPOINT] Human selects issues to fix                   │
│                                                                  │
│  Phase 4: Integration                                           │
│  ├── Architect integrates all components                        │
│  ├── Final deliverable assembled                                │
│  └── [CHECKPOINT] Human accepts/rejects                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Role | Layer | Responsibility |
|------|-------|----------------|
| Product Lead | Planning | PRD analysis, user stories, requirements |
| Architect | Planning | System design, technical decisions, integration |
| Backend Dev | Build | API, business logic, server-side code |
| Frontend Dev | Build | UI components, user experience |
| DB Specialist | Build | Schema design, queries, migrations |
| Code Reviewer | Quality | Code quality, bugs, maintainability |
| Test Engineer | Quality | Testing, coverage, edge cases |
| Security Auditor | Quality | Vulnerabilities, auth, compliance |

### Component Responsibilities

| Component | Purpose |
|-----------|---------|
| `providers` | Abstract LLM API calls (Anthropic, OpenAI, Ollama) |
| `agents` | Define agent personas and role assignments |
| `deliberation` | Orchestrate multi-agent debates |
| `chairman` | Synthesize debates, present to humans, learn from sessions |
| `workspace` | Manage file output directories |
| `council` | Main CLI and session orchestration |
| `costs` | Track token usage and estimate costs |
| `alerts` | Notify user when attention needed |

---

## 3. Learnings from v1/v2

### What Went Wrong

#### Problem 1: Fragile File Extraction

**v1 Approach:**
```python
# Regex parsing of free-form LLM output
pattern = r'```[\w]*\n(?:#|//)\s*([^\n]+)\n(.*?)```'
```

**Why it failed:**
- Depends on LLM following exact format instructions
- If LLM uses different format, silently extracts nothing
- No validation or fallback

**Lesson:** Never depend on LLMs to format output a specific way. Use structured outputs or tool calls.

---

#### Problem 2: Scattered Error Handling

**v1 Approach:**
```python
# Error handling added piecemeal as bugs discovered
if decision.get("action") == "abort":
    return
# Forgot to handle timeout/error - added later as patch
```

**Why it failed:**
- Timeout/error cases fell through silently
- Each checkpoint had different handling logic
- No centralized error recovery strategy

**Lesson:** Design error handling upfront with a consistent pattern.

---

#### Problem 3: Unguarded Optional Dependencies

**v1 Approach:**
```python
# Code assumed agents/providers always exist
result = await agent.provider.complete(...)  # Crashes if provider is None
```

**Why it failed:**
- If a model API is unavailable, entire session crashes
- Guards added piecemeal via P01-P07 patches

**Lesson:** Always handle optional dependencies. Design for degraded operation.

---

#### Problem 4: JSON Parsing from Free-Form Text

**v1 Approach:**
```python
# Try multiple strategies to extract JSON from LLM prose
json_match = re.search(r'```json\s*(.*?)\s*```', synthesis, re.DOTALL)
if json_match:
    try:
        data = json.loads(json_match.group(1))
    except json.JSONDecodeError:
        pass
# ... more fallback attempts ...
# Eventually return hardcoded defaults
```

**Why it failed:**
- LLMs don't reliably output valid JSON in prose
- Complex fallback logic still often fails
- Defaults silently mask failures

**Lesson:** Use structured outputs (JSON mode, function calling, or schemas) to guarantee valid responses.

---

#### Problem 5: Prompt/Extraction Mismatch

**v1 Approach:**
- Prompt tells agent: "Output complete, working code"
- Extraction expects: `# filename.py` on first line of code block
- Agent doesn't know about extraction requirements

**Why it failed:**
- Prompt and extraction logic were developed separately
- No contract between what's asked and what's parsed

**Lesson:** Define explicit contracts. If you need structured output, request it explicitly and validate it.

---

#### Problem 6: Over-Engineering

**v1 had:**
- 8 Python files, ~3000+ lines
- Complex Chairman "memory" system
- Alert system with macOS notifications
- Cost estimation with per-model rates
- Multiple transcript formats (summary vs full)

**Reality:**
- Most features weren't essential for core value
- Complexity made debugging harder
- Each feature was another failure point

**Lesson:** Start minimal. Add features only when needed.

---

### What v2 Got Right

v2 attempted to fix the structured output problem:

```python
# v2 approach - structured outputs
result = await provider.complete_structured(messages, schema=RoleDeliberationResult)
```

**Good ideas from v2:**
- Pydantic schemas for all deliberation results
- Type safety for agent outputs
- Cleaner separation of concerns

**What v2 still had issues with:**
- Still complex (~15 files)
- Schema definitions were verbose
- Not fully tested/integrated

---

## 4. Rebuild Recommendations

### Principle 1: Structured Outputs Everywhere

**Every LLM call should return structured data:**

```python
# Define schema
class AgentOutput(BaseModel):
    files: list[FileOutput]
    summary: str

class FileOutput(BaseModel):
    path: str
    content: str
    language: str

# Request structured output
result = await provider.complete(
    messages=[...],
    response_format=AgentOutput  # Enforced by API
)

# Guaranteed valid
for file in result.files:
    write_file(file.path, file.content)
```

**Providers that support this:**
- Anthropic: Tool use with JSON schema
- OpenAI: `response_format={"type": "json_schema", ...}`
- Ollama: JSON mode (less strict)

---

### Principle 2: Explicit File Contracts

**Don't extract files from prose. Request them explicitly:**

```python
# System prompt
"""
You must output your response using the provided tools.
Use the `write_file` tool for each file you create.
"""

# Tool definition
tools = [{
    "name": "write_file",
    "parameters": {
        "path": {"type": "string"},
        "content": {"type": "string"}
    }
}]

# Parse tool calls - guaranteed structure
for tool_call in response.tool_calls:
    if tool_call.name == "write_file":
        write_file(tool_call.params.path, tool_call.params.content)
```

---

### Principle 3: Graceful Degradation

**Design for partial availability:**

```python
class ProviderPool:
    def __init__(self, config):
        self.providers = {}
        self.available = set()

        for name, cfg in config.items():
            try:
                self.providers[name] = create_provider(cfg)
                self.available.add(name)
            except ProviderError:
                log.warning(f"{name} unavailable")

    def get(self, preferred: str, fallback: str = "claude") -> Provider:
        if preferred in self.available:
            return self.providers[preferred]
        if fallback in self.available:
            return self.providers[fallback]
        raise NoProvidersAvailable()
```

---

### Principle 4: Centralized Checkpoint Handling

**One function for all checkpoints:**

```python
async def checkpoint(
    type: str,
    data: dict,
    options: list[str],
    auto_approve: bool = False
) -> CheckpointResult:
    """
    Unified checkpoint handling.

    - In CLI mode: prompts user
    - In Claude Code mode: emits JSON, reads response
    - Handles timeout/error consistently
    - Returns structured result
    """
    if auto_approve:
        return CheckpointResult(action="approve")

    if claude_code_mode:
        result = emit_and_wait(type, data, options, timeout=120)
    else:
        result = terminal_prompt(type, data, options)

    # Consistent timeout/error handling
    if result.action in ["timeout", "error"]:
        log.warning(f"Checkpoint {type} failed: {result.message}")
        return fallback_to_terminal(type, data, options)

    return result
```

---

### Principle 5: Minimal Viable Council

**Start with just the core loop:**

```
1. User provides task
2. Planner agent creates plan (structured output)
3. [Checkpoint] User approves plan
4. Builder agents create files (tool calls)
5. [Checkpoint] User reviews files
6. Reviewer agent checks quality (structured output)
7. [Checkpoint] User accepts or requests fixes
8. Done
```

**Skip for v3.0:**
- Role deliberation (use sensible defaults)
- Chairman memory/learning
- macOS notifications
- Cost estimation (just log tokens)
- Multiple transcript formats

**Add later if needed.**

---

### Principle 6: Simple Provider Abstraction

```python
class Provider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        system: str = "",
        tools: list[Tool] = None,
        response_schema: type[BaseModel] = None
    ) -> Response:
        """
        Unified completion interface.

        - tools: For file writing, code execution
        - response_schema: For structured outputs
        """
        ...

class AnthropicProvider(Provider):
    async def complete(self, messages, system="", tools=None, response_schema=None):
        kwargs = {"model": self.model, "messages": messages, "system": system}

        if response_schema:
            # Use tool_use to enforce schema
            kwargs["tools"] = [schema_to_tool(response_schema)]
            kwargs["tool_choice"] = {"type": "tool", "name": response_schema.__name__}
        elif tools:
            kwargs["tools"] = tools

        response = await self.client.messages.create(**kwargs)
        return self._parse_response(response)
```

---

## 5. Proposed v3 File Structure

```
council-v3/
├── council.py          # CLI entry point (~100 lines)
├── session.py          # Session orchestration (~150 lines)
├── providers.py        # LLM provider abstraction (~150 lines)
├── agents.py           # Agent definitions (~50 lines)
├── schemas.py          # Pydantic models for all outputs (~100 lines)
├── checkpoint.py       # Unified checkpoint handling (~50 lines)
└── workspace.py        # File writing (~50 lines)

Total: ~650 lines (vs ~3000 in v1)
```

---

## 6. Key Schemas

```python
from pydantic import BaseModel

# Planning
class Plan(BaseModel):
    overview: str
    tasks: list[AgentTask]
    execution_order: list[str]

class AgentTask(BaseModel):
    agent: str
    description: str
    requirements: list[str]
    outputs: list[str]  # Expected file paths

# Building
class BuildOutput(BaseModel):
    files: list[FileOutput]
    summary: str
    issues: list[str]

class FileOutput(BaseModel):
    path: str
    content: str
    language: str

# Review
class ReviewResult(BaseModel):
    issues: list[Issue]
    approved: bool
    summary: str

class Issue(BaseModel):
    severity: Literal["critical", "major", "minor"]
    file: str
    line: int | None
    description: str
    suggestion: str
```

---

## 7. Example Session Flow

```python
async def run_session(task: str):
    # 1. Plan
    plan = await planner.complete(
        messages=[{"role": "user", "content": task}],
        response_schema=Plan
    )

    if not await checkpoint("plan", plan.dict(), ["approve", "revise"]):
        return

    # 2. Build
    outputs = {}
    for agent_task in plan.tasks:
        agent = get_agent(agent_task.agent)
        result = await agent.complete(
            messages=[{"role": "user", "content": agent_task.description}],
            tools=[write_file_tool],
        )

        # Files written via tool calls - guaranteed
        outputs[agent_task.agent] = result

        if not await checkpoint("build", result.dict(), ["approve", "revise"]):
            # Rework loop
            ...

    # 3. Review
    review = await reviewer.complete(
        messages=[{"role": "user", "content": format_for_review(outputs)}],
        response_schema=ReviewResult
    )

    if review.issues and not review.approved:
        await checkpoint("review", review.dict(), ["fix_all", "fix_some", "skip"])
        # Fix loop...

    # 4. Done
    await checkpoint("complete", {"files": list_files()}, ["accept"])
```

---

## 8. Testing Strategy

### Unit Tests (No API Needed)

```python
def test_schema_validation():
    """Schemas reject invalid data"""
    with pytest.raises(ValidationError):
        Plan(overview="x", tasks="not a list", execution_order=[])

def test_checkpoint_timeout_handling():
    """Timeouts fall back to terminal"""
    result = checkpoint_with_mock_timeout("plan", {}, ["approve"])
    assert result.source == "terminal_fallback"

def test_file_writing():
    """Tool calls produce files"""
    workspace = Workspace(tmp_path)
    workspace.write_file("app.py", "print(1)")
    assert (tmp_path / "app.py").read_text() == "print(1)"
```

### Integration Tests (Mock Providers)

```python
def test_full_session_mock():
    """Full session with mocked LLM responses"""
    provider = MockProvider(responses=[
        Plan(overview="Test", tasks=[...], execution_order=["Backend"]),
        BuildOutput(files=[FileOutput(path="app.py", content="...", language="python")]),
        ReviewResult(issues=[], approved=True, summary="LGTM")
    ])

    session = Session(provider=provider, auto_approve=True)
    result = await session.run("Build a hello world app")

    assert "app.py" in result.files
```

### E2E Tests (Real APIs, Manual)

```bash
# Smoke test with real APIs
council "Build a simple Python CLI that prints hello world"
# Verify: files created, no crashes, checkpoints work
```

---

## 9. Migration Path

1. **v3.0**: Minimal viable council (structured outputs, checkpoints, file writing)
2. **v3.1**: Add role deliberation (optional, can skip)
3. **v3.2**: Add quality review phase
4. **v3.3**: Add cost tracking, transcripts
5. **v3.4**: Add Chairman synthesis (if valuable)

---

## 10. Open Questions

1. **Provider fallback strategy**: If preferred model unavailable, auto-fallback or ask user?

2. **Parallel vs sequential builds**: v1 claimed parallel but ran sequentially. Worth true parallelism?

3. **Deliberation value**: Does model debate actually improve outcomes? Should measure.

4. **Claude Code integration**: Keep checkpoint protocol or design new one?

5. **State persistence**: Resume interrupted sessions? Save/load session state?

---

## Appendix: v1 Patches Reference

For context, these were the identified issues in v1:

| ID | Issue | Status |
|----|-------|--------|
| P01 | `prd_content` used before assignment | In v1 |
| P02 | `architecture` used before assignment | In v1 |
| P03 | `revise_plan` crashes if Architect unavailable | In v1 |
| P04 | Build/quality loops crash on unavailable agents | In v1 |
| P05 | File extraction fragile (regex-based) | Partially fixed |
| P06 | `integrate_outputs` crashes if Architect unavailable | In v1 |
| P07 | Session start doesn't handle unavailable agents | Fixed in v3 |
| G1 | Extraction format not documented to agents | Fixed in v3 |
| G2 | Checkpoint timeout falls through | Fixed in v3 |

Phase 2 (P08-P16) and Phase 3 (P17-P22) patches were never applied.

---

*Document generated: 2026-01-26*
*For: Council v3 Rebuild*
