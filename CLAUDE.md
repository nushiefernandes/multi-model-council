# Council v2 - Architecture Notes for Claude

## How Council is Invoked

Council is triggered ONLY via the `/council` skill (explicit invocation).

**DO NOT:**
- Auto-register the MCP server (`mcp_server.py`)
- Use council tools without user explicitly saying `/council`

**The MCP server exists** but is intentionally not registered to prevent auto-triggering.

## Invocation Flow

1. User types `/council <task>`
2. Claude Code becomes the orchestrator
3. Calls other models (DeepSeek, GPT-4) for proposals
4. Anonymizes proposals (Model A, B, C)
5. Cross-critique phase
6. Chairman vote (democratic selection)
7. Synthesis by voted chairman
8. Show plan to user for approval

## Key Files

| File | Purpose |
|------|---------|
| `cli.py` | CLI entry point (used by skill) |
| `deliberation.py` | Core deliberation logic with anonymization |
| `mcp_server.py` | MCP server (NOT registered, for programmatic use only) |
| `~/.claude/plugins/skills/council/SKILL.md` | The /council skill definition |

## Design Decisions

1. **Explicit trigger only** - No auto-invocation, user must say `/council`
2. **Anonymized proposals** - Chairman doesn't know which model proposed what (reduces bias)
3. **Democratic chairman** - Models vote on who synthesizes
4. **CLI over MCP** - Subprocess approach for full control and interactivity

## Why MCP is NOT Registered

The MCP server provides programmatic access to council tools, but we intentionally don't register it because:
- MCP tools can be auto-triggered by Claude based on context
- User wants explicit `/council` invocation only
- CLI approach preserves interactive checkpoints

If you need programmatic access, use the Python modules directly:
```python
from deliberation import Deliberation, DeliberationConfig, anonymize_proposals
```
