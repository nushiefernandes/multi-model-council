# /council

Invoke Agent Council for multi-model deliberation with anonymized proposals and consensus-based chairman selection.

## Usage

```
/council <task>                    - Start deliberation on a task
/council --file <path>             - Load task from PRD/requirements file
/council --quick <question>        - Quick consensus (no full deliberation)
/council --list                    - List past sessions
/council --resume <session_id>     - Resume a session
```

## What This Skill Does

When invoked, YOU (Claude Code) become the **orchestrator** of a multi-model deliberation:

1. **Gather Proposals** - Call other models (DeepSeek, GPT-4) for their proposals
2. **Anonymize** - Shuffle proposals to Model A, B, C (remove bias)
3. **Cross-Critique** - Each model critiques anonymously
4. **Chairman Vote** - Models vote on who should synthesize
5. **Synthesis** - Winning chairman produces final plan
6. **Checkpoint** - Show plan to user, get approval

## Implementation

### Setup (Required for all modes):

```python
import asyncio
import sys
sys.path.insert(0, "/Users/anushfernandes/.council/v2")

from deliberation import (
    Deliberation,
    DeliberationConfig,
    anonymize_proposals,
    deanonymize_result,
)
from config import load_config
from schemas import RoleDeliberationResult
```

### For Full Deliberation (`/council <task>`):

```python
async def run_council(task: str, context: str = ""):
    # Load config
    config_obj = load_config()
    config = config_obj.to_provider_config()

    # Configure deliberation
    delib_config = DeliberationConfig(
        chairman_model="claude",  # Will be overridden by vote
        proposal_models=["claude", "deepseek", "codex"],
        critique_models=["claude", "deepseek", "codex"],
        temperature=0.3,  # Important for structured output reliability
    )

    engine = Deliberation(config, delib_config)

    # Progress callback for live updates
    def on_progress(status: str, model: str, data: dict):
        if status == "waiting":
            print(f"  - {model} working...")
        elif status == "done":
            tokens = data.get("tokens", 0)
            cost = data.get("cost", 0)
            print(f"  - {model} done ({tokens} tokens, ${cost:.4f})")
        elif status == "error":
            print(f"  - {model} FAILED: {data.get('error')}")

    # PHASE 1: Gather proposals
    print("\n## Phase 1: Gathering Proposals")
    proposals, responses = await engine.gather_proposals(
        task, context, progress_callback=on_progress
    )
    print(f"\nGot {len(proposals)} proposals.")

    # Show proposal summaries (anonymized)
    anon_proposals, mapping = anonymize_proposals(proposals)
    for label, proposal in anon_proposals:
        print(f"  {label}: {proposal.approach[:80]}... (confidence: {proposal.confidence}/10)")

    # CHECKPOINT: Ask user to continue
    # (Claude Code will pause here for user confirmation)

    # PHASE 2: Cross-Critique
    print("\n## Phase 2: Cross-Critique")
    critiques, _ = await engine.cross_critique(
        anon_proposals, progress_callback=on_progress
    )
    print(f"\nGot {len(critiques)} critiques.")

    # PHASE 3: Chairman Vote
    print("\n## Phase 3: Chairman Vote")
    winner, votes = await engine.vote_for_chairman(
        anon_proposals, critiques, progress_callback=on_progress
    )
    print(f"\nVotes: {votes}")
    print(f"Winner: {winner}")

    # Update chairman to voted winner
    real_winner = mapping.get(winner, "claude")
    engine.delib_config.chairman_model = real_winner
    print(f"({winner} = {real_winner})")

    # PHASE 4: Synthesis
    print("\n## Phase 4: Synthesis by Chairman")
    result, _ = await engine.synthesize_assignments(task, anon_proposals, critiques)
    result = deanonymize_result(result, mapping)

    # Show final assignments
    print("\n" + "="*50)
    print("ROLE ASSIGNMENTS")
    print("="*50)
    for assignment in result.assignments:
        print(f"\n{assignment.role}: {assignment.assigned_to}")
        print(f"  Reasoning: {assignment.reasoning}")
    print(f"\nConsensus: {result.consensus_notes}")
    if result.dissenting_views:
        print(f"Dissenting: {', '.join(result.dissenting_views)}")
    print("="*50)

    # CHECKPOINT: Ask user to approve assignments
    return result

# Run the deliberation
result = asyncio.run(run_council(task, context))
```

### For Quick Consensus (`/council --quick <question>`):

```python
async def quick_consensus(question: str):
    config_obj = load_config()
    config = config_obj.to_provider_config()

    delib_config = DeliberationConfig(
        proposal_models=["claude", "deepseek", "codex"],
        temperature=0.3,
    )
    engine = Deliberation(config, delib_config)

    print(f"\n## Quick Consensus: {question}\n")
    print("Gathering answers...")

    result = await engine.quick_consensus(question)

    print("\n" + "="*50)
    print("CONSENSUS RESULT")
    print("="*50)
    print(f"\nAnswer: {result.final_answer}")
    print(f"Confidence: {result.confidence}/10")
    print(f"Agreement: {result.agreement_level}")

    if result.dissenting_views:
        print(f"\nDissenting views:")
        for view in result.dissenting_views:
            print(f"  - {view}")
    print("="*50)

    return result

result = asyncio.run(quick_consensus(question))
```

### For Listing Sessions (`/council --list`):

```python
from session import SessionManager

manager = SessionManager()
sessions = manager.list_sessions()

print("\n## Past Sessions\n")
if not sessions:
    print("No sessions found.")
else:
    for s in sessions:
        print(f"- {s['session_id']}: {s['task'][:50]}...")
        print(f"  Phase: {s['phase']} | Created: {s['created_at']}")
```

### For Resume (`/council --resume <session_id>`):

```python
from session import SessionManager

manager = SessionManager()
session = manager.load(session_id)

print(f"\n## Resuming Session: {session.session_id}")
print(f"Task: {session.task}")
print(f"Current Phase: {session.phase}")

# Continue from where we left off based on session.phase
```

## Interactive Checkpoints

At key phases, pause and show progress to the user:

### After Phase 1 (Proposals):
```
## Phase 1: Gathering Proposals
  - claude working...
  - claude done (245 tokens, $0.0073)
  - deepseek working...
  - deepseek done (198 tokens, $0.0000)
  - codex working...
  - codex done (312 tokens, $0.0094)

Got 3 proposals.
  Model A: Build microservices with FastAPI... (confidence: 8/10)
  Model B: Create monolith with Django... (confidence: 7/10)
  Model C: Use serverless with Lambda... (confidence: 6/10)

Continue to critique phase? [y/n]
```

### After Phase 3 (Chairman Vote):
```
## Phase 3: Chairman Vote
  - claude voting...
  - deepseek voting...
  - codex voting...

Votes: {'Model A': 'Model B', 'Model B': 'Model A', 'Model C': 'Model A'}
Winner: Model A
(Model A = claude)

Continue to synthesis? [y/n]
```

### After Phase 4 (Final Plan):
```
==================================================
ROLE ASSIGNMENTS
==================================================

Architect: deepseek
  Reasoning: Strong system design capabilities

Backend Dev: claude
  Reasoning: Excellent at API implementation

Frontend Dev: codex
  Reasoning: Good UX patterns

Consensus: The council agreed on role specialization
Dissenting: Model C preferred all-in-one approach
==================================================

Approve these assignments? [y/n/feedback]: _
```

## Handling User Feedback

If user provides feedback instead of approval:
```
User: n - I want to use GraphQL instead of REST

You: Got it. I'll re-run the deliberation with the constraint: "Must use GraphQL"
```

Then re-run Phase 1-4 with the additional context.

## Key Principles

1. **You are the ORCHESTRATOR, not the Chairman** - You coordinate, but don't bias the synthesis
2. **Anonymization is CRITICAL** - Always shuffle proposals before showing to critics/chairman
3. **Interactive by DEFAULT** - Pause at checkpoints, don't auto-proceed
4. **Respect the Vote** - The voted chairman synthesizes, even if it's not Claude
5. **Show Progress** - Keep user informed of what's happening with live updates

## Error Handling

If API calls fail:
```
  - deepseek working...
  - deepseek FAILED: Connection timeout

Continuing with 2 models. Proceed? [y/n]
```

If only one model responds, warn and offer to retry or continue with reduced council.

## Configuration

Models are configured in `~/.council/config.yaml`:
```yaml
models:
  claude:
    provider: anthropic
    model: claude-sonnet-4-20250514
  deepseek:
    provider: ollama
    model: deepseek-coder-v2:16b
  codex:
    provider: openai
    model: gpt-4
```

## Important: Do NOT Use V1 Scripts

The old `~/.council/scripts/council.py` uses a `<<<CHECKPOINT:>>>` protocol that clutters output.
Always use the V2 Python API directly as shown above for clean, readable progress.
