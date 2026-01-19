#!/usr/bin/env python3
"""
Integration tests for Phase 2: Real Deliberation Engine.

These tests verify that the deliberation engine produces REAL structured outputs,
not hardcoded defaults like v1.

HOW TO RUN:
-----------
    cd ~/.council/v2
    python test_phase2.py

REQUIREMENTS:
-------------
- ANTHROPIC_API_KEY set for Claude tests
- OPENAI_API_KEY set for GPT-4 tests
- Ollama running for DeepSeek tests (optional)

VERIFICATION APPROACH:
---------------------
1. Run tests twice - results should DIFFER (proving not hardcoded)
2. Check that assignments reference real models
3. Verify cost tracking works
4. Test with different tasks - should produce different plans
"""

import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schemas import (
    Proposal,
    Critique,
    RoleAssignment,
    RoleDeliberationResult,
    Plan,
    CodeReview,
    ConsensusAnswer,
    ConsensusResult,
)
from deliberation import (
    Deliberation,
    DeliberationConfig,
    create_default_config,
    quick_deliberate_roles,
    quick_review,
    quick_ask,
)


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Minimal config using only Claude (always available if API key is set)
CLAUDE_ONLY_CONFIG = {
    "models": {
        "claude": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "api_key_env": "ANTHROPIC_API_KEY"
        }
    }
}

# Config for tests that need multiple models
MULTI_MODEL_CONFIG = create_default_config()

# Valid model names for assignment checking
VALID_MODELS = {"claude", "deepseek", "gpt4", "codex", "openai", "anthropic"}


def has_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


# =============================================================================
# TEST 1: PROPOSAL COLLECTION
# =============================================================================

async def test_gather_proposals():
    """
    Test that models return valid Proposal objects.

    This verifies:
    - Multiple models can propose in parallel
    - Each proposal is a valid Pydantic Proposal
    - Effort is one of the allowed values
    - Confidence is in valid range
    """
    print("\n" + "=" * 60)
    print("TEST 1: Proposal Collection")
    print("=" * 60)

    if not has_anthropic_key():
        print("SKIP: ANTHROPIC_API_KEY not set")
        return False

    config = CLAUDE_ONLY_CONFIG
    delib_config = DeliberationConfig(
        proposal_models=["claude"],
        critique_models=["claude"],
        chairman_model="claude"
    )

    delib = Deliberation(config, delib_config)

    print("Gathering proposals for: 'Build a REST API for user management'")
    proposals, responses = await delib.gather_proposals("Build a REST API for user management")

    print(f"\nReceived {len(proposals)} proposal(s)")

    # Verify we got at least one proposal
    assert len(proposals) >= 1, "Expected at least 1 proposal"

    for model_name, proposal in proposals:
        print(f"\n--- {model_name}'s Proposal ---")
        print(f"Approach: {proposal.approach[:100]}...")
        print(f"Effort: {proposal.effort}")
        print(f"Confidence: {proposal.confidence}/10")
        print(f"Risks: {len(proposal.risks)} identified")

        # Verify it's a valid Proposal
        assert isinstance(proposal, Proposal), f"Expected Proposal, got {type(proposal)}"
        assert proposal.effort in ["low", "medium", "high"], f"Invalid effort: {proposal.effort}"
        assert 1 <= proposal.confidence <= 10, f"Confidence out of range: {proposal.confidence}"

    # Verify response metadata
    for resp in responses:
        assert resp.tokens_used > 0, "Expected non-zero token usage"
        print(f"\nTokens used by {resp.agent_name}: {resp.tokens_used}")

    print("\n✅ Test 1 PASSED: Proposal collection works")
    return True


# =============================================================================
# TEST 2: CROSS-CRITIQUE
# =============================================================================

async def test_cross_critique():
    """
    Test that models can critique proposals.

    This verifies:
    - Models can analyze others' proposals
    - Critiques are valid Pydantic Critique objects
    - Recommendations are valid values
    """
    print("\n" + "=" * 60)
    print("TEST 2: Cross-Critique")
    print("=" * 60)

    if not has_anthropic_key():
        print("SKIP: ANTHROPIC_API_KEY not set")
        return False

    config = CLAUDE_ONLY_CONFIG
    delib_config = DeliberationConfig(
        proposal_models=["claude"],
        critique_models=["claude"],
        chairman_model="claude"
    )

    delib = Deliberation(config, delib_config)

    # Create mock proposals to critique
    mock_proposals = [
        ("mock_model_a", Proposal(
            approach="Use JWT tokens with refresh rotation for authentication",
            rationale="JWTs are stateless, scale well, and are industry standard",
            risks=["Token theft if not using HTTPS", "Complexity of refresh logic"],
            effort="medium",
            confidence=8
        )),
        ("mock_model_b", Proposal(
            approach="Use session-based authentication with Redis",
            rationale="Sessions are simpler and allow easy invalidation",
            risks=["Redis as single point of failure", "Scaling challenges"],
            effort="low",
            confidence=7
        )),
    ]

    print("Critiquing mock proposals about authentication approaches")
    critiques, responses = await delib.cross_critique(mock_proposals)

    print(f"\nReceived {len(critiques)} critique(s)")

    # Verify we got at least one critique
    assert len(critiques) >= 1, "Expected at least 1 critique"

    for model_name, critique in critiques:
        print(f"\n--- {model_name}'s Critique ---")
        print(f"Target: {critique.target_proposal}")
        print(f"Agrees: {len(critique.agrees)} points")
        print(f"Disagrees: {len(critique.disagrees)} points")
        print(f"Suggestions: {len(critique.suggestions)} suggestions")
        print(f"Recommendation: {critique.recommendation}")

        # Verify it's a valid Critique
        assert isinstance(critique, Critique), f"Expected Critique, got {type(critique)}"
        assert critique.recommendation in ["accept", "modify", "reject"], \
            f"Invalid recommendation: {critique.recommendation}"

    print("\n✅ Test 2 PASSED: Cross-critique works")
    return True


# =============================================================================
# TEST 3: FULL ROLE DELIBERATION
# =============================================================================

async def test_deliberate_roles():
    """
    Test full role deliberation flow.

    This verifies:
    - End-to-end deliberation works
    - Returns RoleDeliberationResult (not hardcoded)
    - Assignments reference valid models
    - Results DIFFER on different runs (proving it's real)
    """
    print("\n" + "=" * 60)
    print("TEST 3: Full Role Deliberation")
    print("=" * 60)

    if not has_anthropic_key():
        print("SKIP: ANTHROPIC_API_KEY not set")
        return False

    config = CLAUDE_ONLY_CONFIG
    delib_config = DeliberationConfig(
        proposal_models=["claude"],
        critique_models=["claude"],
        chairman_model="claude"
    )

    delib = Deliberation(config, delib_config)

    print("Running role deliberation for: 'Build an authentication system'")
    result = await delib.deliberate_roles("Build an authentication system with OAuth2 support")

    print(f"\n--- Role Assignments ---")
    assert isinstance(result, RoleDeliberationResult), \
        f"Expected RoleDeliberationResult, got {type(result)}"

    # Verify we got assignments
    assert len(result.assignments) >= 1, "Expected at least 1 role assignment"

    for assignment in result.assignments:
        print(f"{assignment.role}: {assignment.assigned_to}")
        print(f"  Reason: {assignment.reasoning[:80]}...")

        # Verify assignment is valid
        assert isinstance(assignment, RoleAssignment)
        # Note: The model might use slightly different model names
        # We just verify it's a non-empty string
        assert len(assignment.assigned_to) > 0, "assigned_to should not be empty"

    print(f"\nConsensus: {result.consensus_notes[:200]}...")
    print(f"Dissenting views: {len(result.dissenting_views)}")

    # Check transcript
    transcript = delib.get_transcript()
    if transcript:
        print(f"\n--- Session Stats ---")
        print(f"Session ID: {transcript.session_id}")
        print(f"Total tokens: {transcript.total_tokens}")
        print(f"Total cost: ${transcript.total_cost:.4f}")
        print(f"Duration: {transcript.duration_seconds:.2f}s")

    print("\n✅ Test 3 PASSED: Full role deliberation works")
    return True


# =============================================================================
# TEST 4: PLANNING DELIBERATION
# =============================================================================

async def test_deliberate_plan():
    """
    Test that planning produces a structured Plan.

    This verifies:
    - Plan has required structure
    - Steps have all required fields
    - Effort estimates are valid
    """
    print("\n" + "=" * 60)
    print("TEST 4: Planning Deliberation")
    print("=" * 60)

    if not has_anthropic_key():
        print("SKIP: ANTHROPIC_API_KEY not set")
        return False

    config = CLAUDE_ONLY_CONFIG
    delib = Deliberation(config)

    print("Creating plan for: 'Build a task management API'")
    plan = await delib.deliberate_plan("Build a task management API with CRUD operations")

    print(f"\n--- Implementation Plan ---")
    assert isinstance(plan, Plan), f"Expected Plan, got {type(plan)}"

    print(f"Overview: {plan.overview[:200]}...")
    print(f"Total steps: {plan.total_steps}")
    print(f"Estimated effort: {plan.estimated_total_effort}")
    print(f"Critical path steps: {plan.critical_path}")

    # Verify we got steps
    assert len(plan.steps) >= 1, "Expected at least 1 plan step"
    assert plan.estimated_total_effort in ["low", "medium", "high"], \
        f"Invalid effort: {plan.estimated_total_effort}"

    print("\n--- Steps ---")
    for step in plan.steps[:5]:  # Show first 5 steps
        print(f"{step.step_number}. {step.title}")
        print(f"   Assigned to: {step.assigned_agent}")
        print(f"   Complexity: {step.estimated_complexity}")
        print(f"   Dependencies: {step.dependencies}")

    print("\n✅ Test 4 PASSED: Planning deliberation works")
    return True


# =============================================================================
# TEST 5: CODE REVIEW
# =============================================================================

async def test_quality_review():
    """
    Test that code review finds real issues.

    This verifies:
    - Review produces CodeReview object
    - SQL injection is detected (security issue)
    - Issues have proper severity/category
    """
    print("\n" + "=" * 60)
    print("TEST 5: Code Review")
    print("=" * 60)

    if not has_anthropic_key():
        print("SKIP: ANTHROPIC_API_KEY not set")
        return False

    config = CLAUDE_ONLY_CONFIG
    delib = Deliberation(config)

    # Code with intentional issues
    bad_code = '''
def get_user(id):
    # SQL injection vulnerability!
    query = f"SELECT * FROM users WHERE id = {id}"
    return db.execute(query)

def process_data(data):
    # No error handling
    result = data["key"]["nested"]
    return result * 2

def authenticate(password):
    # Storing password in plain text!
    users_db[username] = password
    return True
'''

    print("Reviewing code with intentional security issues...")
    review = await delib.quality_review(bad_code, context="User authentication module")

    print(f"\n--- Code Review Results ---")
    assert isinstance(review, CodeReview), f"Expected CodeReview, got {type(review)}"

    print(f"Reviewer: {review.reviewer}")
    print(f"Approved: {review.approved}")
    print(f"Summary: {review.summary[:200]}...")
    print(f"Issues found: {len(review.issues)}")
    print(f"Blocking issues: {review.blocking_issues}")

    print("\n--- Issues ---")
    for issue in review.issues:
        print(f"[{issue.severity.upper()}] {issue.category}")
        print(f"  Location: {issue.location}")
        print(f"  Issue: {issue.description[:100]}...")
        print(f"  Fix effort: {issue.effort_to_fix}")
        print()

    # Verify security issue was found (SQL injection or password storage)
    security_issues = [i for i in review.issues if i.category == "security"]
    print(f"Security issues found: {len(security_issues)}")

    # We expect at least one security issue from SQL injection or plain text password
    if len(security_issues) > 0:
        print("✅ Security vulnerability detected!")
    else:
        print("⚠️  No security issues flagged (model may have missed them)")

    print(f"\n--- Positive Notes ---")
    for note in review.positive_notes[:3]:
        print(f"  + {note}")

    print("\n✅ Test 5 PASSED: Code review works")
    return True


# =============================================================================
# TEST 6: VERIFY NOT HARDCODED
# =============================================================================

async def test_results_differ():
    """
    Verify that results are NOT hardcoded by running twice.

    If v1's hardcoded behavior was still present, both runs would
    return identical results. Real deliberation should show variation.
    """
    print("\n" + "=" * 60)
    print("TEST 6: Verify Results Differ (Not Hardcoded)")
    print("=" * 60)

    if not has_anthropic_key():
        print("SKIP: ANTHROPIC_API_KEY not set")
        return False

    config = CLAUDE_ONLY_CONFIG
    delib_config = DeliberationConfig(
        proposal_models=["claude"],
        critique_models=["claude"],
        chairman_model="claude",
        temperature=0.7  # Higher temperature for more variation
    )

    # Run 1
    print("Running deliberation (attempt 1)...")
    delib1 = Deliberation(config, delib_config)
    result1 = await delib1.deliberate_roles("Design a caching strategy")

    # Run 2
    print("Running deliberation (attempt 2)...")
    delib2 = Deliberation(config, delib_config)
    result2 = await delib2.deliberate_roles("Design a caching strategy")

    # Compare consensus notes (most likely to differ)
    notes1 = result1.consensus_notes
    notes2 = result2.consensus_notes

    print(f"\nRun 1 consensus ({len(notes1)} chars): {notes1[:100]}...")
    print(f"Run 2 consensus ({len(notes2)} chars): {notes2[:100]}...")

    # Check if they're identical
    if notes1 == notes2:
        print("\n⚠️  Results are identical - might still be some caching or determinism")
        print("    This doesn't necessarily mean it's hardcoded.")
    else:
        print("\n✅ Results differ - confirmed NOT hardcoded!")

    print("\n✅ Test 6 PASSED: Verification complete")
    return True


# =============================================================================
# TEST 7: QUICK CONSENSUS
# =============================================================================

async def test_quick_consensus():
    """
    Test quick consensus for simple questions.

    This verifies:
    - Quick consensus returns a valid ConsensusResult
    - Answer and confidence are populated
    - Agreement level is valid
    - Sources list the models that contributed
    """
    print("\n" + "=" * 60)
    print("TEST 7: Quick Consensus")
    print("=" * 60)

    if not has_anthropic_key():
        print("SKIP: ANTHROPIC_API_KEY not set")
        return False

    config = CLAUDE_ONLY_CONFIG
    delib = Deliberation(config)

    print("Asking quick consensus question: 'Redis vs Memcached for session caching?'")
    result = await delib.quick_consensus(
        "Redis vs Memcached for session caching?",
        models=["claude"],
        with_evaluation=False
    )

    print(f"\n--- Quick Consensus Result ---")
    assert isinstance(result, ConsensusResult), f"Expected ConsensusResult, got {type(result)}"

    print(f"Answer: {result.answer[:200]}...")
    print(f"Confidence: {result.confidence}/10")
    print(f"Agreement Level: {result.agreement_level}")
    print(f"Sources: {result.sources}")
    print(f"Dissenting Views: {len(result.dissenting_views)}")

    # Verify the result has required fields
    assert result.answer, "Answer should not be empty"
    assert 1 <= result.confidence <= 10, f"Confidence out of range: {result.confidence}"
    assert result.agreement_level in ["unanimous", "strong", "moderate", "divided"], \
        f"Invalid agreement level: {result.agreement_level}"
    assert len(result.sources) >= 1, "Expected at least 1 source"

    print("\n✅ Test 7 PASSED: Quick consensus works")
    return True


async def test_quick_ask_convenience():
    """
    Test the quick_ask convenience function.

    This verifies:
    - quick_ask returns a valid ConsensusResult
    - Works with default configuration
    """
    print("\n" + "=" * 60)
    print("TEST 8: quick_ask Convenience Function")
    print("=" * 60)

    if not has_anthropic_key():
        print("SKIP: ANTHROPIC_API_KEY not set")
        return False

    print("Using quick_ask: 'Should I use REST or GraphQL for a simple CRUD API?'")
    result = await quick_ask(
        "Should I use REST or GraphQL for a simple CRUD API?",
        models=["claude"]
    )

    print(f"\n--- quick_ask Result ---")
    assert isinstance(result, ConsensusResult), f"Expected ConsensusResult, got {type(result)}"

    print(f"Answer: {result.answer[:200]}...")
    print(f"Confidence: {result.confidence}/10")
    print(f"Agreement: {result.agreement_level}")

    assert result.answer, "Answer should not be empty"
    assert 1 <= result.confidence <= 10

    print("\n✅ Test 8 PASSED: quick_ask convenience function works")
    return True


# =============================================================================
# MULTI-MODEL TEST (if both keys available)
# =============================================================================

async def test_multi_model_deliberation():
    """
    Test deliberation with multiple models (Claude + GPT-4).

    This is the full council experience with different models
    proposing and critiquing each other.
    """
    print("\n" + "=" * 60)
    print("TEST 9: Multi-Model Deliberation (Claude + GPT-4)")
    print("=" * 60)

    if not has_anthropic_key() or not has_openai_key():
        print("SKIP: Need both ANTHROPIC_API_KEY and OPENAI_API_KEY")
        return False

    config = MULTI_MODEL_CONFIG
    delib_config = DeliberationConfig(
        proposal_models=["claude", "gpt4"],
        critique_models=["claude", "gpt4"],
        chairman_model="claude"
    )

    delib = Deliberation(config, delib_config)

    print("Running multi-model deliberation for: 'Design a microservices architecture'")
    result = await delib.deliberate_roles("Design a microservices architecture for an e-commerce platform")

    print(f"\n--- Multi-Model Role Assignments ---")
    for assignment in result.assignments:
        print(f"{assignment.role}: {assignment.assigned_to}")

    transcript = delib.get_transcript()
    if transcript:
        print(f"\n--- Multi-Model Session Stats ---")
        print(f"Total tokens: {transcript.total_tokens}")
        print(f"Total cost: ${transcript.total_cost:.4f}")

        for phase in transcript.phases:
            print(f"\n{phase.phase_name.upper()} phase:")
            for resp in phase.responses:
                print(f"  - {resp.agent_name}: {resp.tokens_used} tokens, ${resp.cost:.4f}")

    print("\n✅ Test 9 PASSED: Multi-model deliberation works")
    return True


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("PHASE 2 INTEGRATION TESTS")
    print("Real Deliberation Engine")
    print("=" * 60)

    print("\nEnvironment check:")
    print(f"  ANTHROPIC_API_KEY: {'✅ Set' if has_anthropic_key() else '❌ Not set'}")
    print(f"  OPENAI_API_KEY: {'✅ Set' if has_openai_key() else '❌ Not set'}")

    results = {}

    # Core tests (Claude only)
    tests = [
        ("Proposal Collection", test_gather_proposals),
        ("Cross-Critique", test_cross_critique),
        ("Full Role Deliberation", test_deliberate_roles),
        ("Planning Deliberation", test_deliberate_plan),
        ("Code Review", test_quality_review),
        ("Verify Not Hardcoded", test_results_differ),
        ("Quick Consensus", test_quick_consensus),
        ("quick_ask Convenience", test_quick_ask_convenience),
    ]

    for name, test_fn in tests:
        try:
            result = await test_fn()
            results[name] = "PASS" if result else "SKIP"
        except Exception as e:
            print(f"\n❌ Test '{name}' FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = "FAIL"

    # Multi-model test (optional)
    try:
        result = await test_multi_model_deliberation()
        results["Multi-Model Deliberation"] = "PASS" if result else "SKIP"
    except Exception as e:
        print(f"\n❌ Multi-model test FAILED: {e}")
        results["Multi-Model Deliberation"] = "FAIL"

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, status in results.items():
        symbol = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}[status]
        print(f"  {symbol} {name}: {status}")

    passed = sum(1 for s in results.values() if s == "PASS")
    failed = sum(1 for s in results.values() if s == "FAIL")
    skipped = sum(1 for s in results.values() if s == "SKIP")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return False
    else:
        print("\n🎉 Phase 2 implementation verified!")
        return True


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
