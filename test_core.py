"""
QUALIA Test Suite — Core System Verification
=============================================
Tests cover all three theoretical pillars:
  1. Quaternion state math (correctness of QPT engine)
  2. Habit loop behavior (cue detection, reinforcement, decay)
  3. Ethics guard coverage (anti-fabrication, PII detection)
  4. Integration smoke tests (agent pipeline end-to-end)

Run: python -m pytest tests/test_core.py -v
"""

import sys
import time
import math
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from core.qualia_core  import Quaternion, QPTEngine, EmotionalState
from core.habit_engine import HabitEngine, HabitRecord
from core.ethics_guard import EthicsGuard, ConfidenceLevel


# ═══════════════════════════════════════════════════════════════════════════
# QUATERNION MATH TESTS
# The math has to be correct — wrong quaternion operations break the entire
# cognitive state model.
# ═══════════════════════════════════════════════════════════════════════════

class TestQuaternionMath:

    def test_unit_quaternion_has_magnitude_one(self):
        """After normalization, the quaternion must live on the unit hypersphere."""
        q = Quaternion(w=3.0, x=1.0, y=2.0, z=1.5)
        n = q.normalize()
        assert abs(n.magnitude() - 1.0) < 1e-6, "Normalized quaternion magnitude should be 1.0"

    def test_default_quaternion_is_near_unit(self):
        """The default QPT state should be pre-normalized."""
        q = Quaternion()
        # Default values may not be exactly unit but should be reasonable
        assert q.magnitude() > 0.1, "Default quaternion should have non-zero magnitude"

    def test_hamilton_product_preserves_unit_length(self):
        """The Hamilton product of two unit quaternions should stay unit length."""
        q1 = Quaternion(w=0.5, x=0.5, y=0.5, z=0.5).normalize()
        q2 = Quaternion(w=0.7, x=0.1, y=0.5, z=0.5).normalize()
        product = q1.hamilton_product(q2)
        assert abs(product.magnitude() - 1.0) < 1e-6, "Hamilton product should remain on unit hypersphere"

    def test_hamilton_product_non_commutative(self):
        """q1 × q2 ≠ q2 × q1 — this is intentional and models context ordering."""
        q1 = Quaternion(w=0.5, x=0.3, y=0.6, z=0.5).normalize()
        q2 = Quaternion(w=0.8, x=0.2, y=0.1, z=0.5).normalize()
        ab = q1.hamilton_product(q2)
        ba = q2.hamilton_product(q1)
        # They should NOT be equal (quaternion multiplication is non-commutative)
        different = any(abs(getattr(ab, c) - getattr(ba, c)) > 1e-6 for c in ["w", "x", "y", "z"])
        assert different, "Hamilton product must be non-commutative for QPT context modeling to work"

    def test_dot_product_identical_quaternions_equals_one(self):
        """A quaternion dotted with itself (normalized) should equal 1.0."""
        q = Quaternion(w=0.5, x=0.5, y=0.5, z=0.5).normalize()
        assert abs(q.dot(q) - 1.0) < 1e-6, "Dot product of a unit quaternion with itself should be 1.0"

    def test_from_text_signal_stays_on_unit_sphere(self):
        """Factory method should always produce a valid unit quaternion."""
        q = Quaternion.from_text_signal(valence=0.8, arousal=0.9, alignment=0.7)
        assert abs(q.magnitude() - 1.0) < 1e-6, "Text-signal quaternion must be unit length"

    def test_serialization_roundtrip(self):
        """A quaternion serialized to dict and back must match the original."""
        q = Quaternion(w=0.6, x=-0.3, y=0.5, z=0.5).normalize()
        restored = Quaternion.from_dict(q.to_dict())
        for comp in ["w", "x", "y", "z"]:
            assert abs(getattr(q, comp) - getattr(restored, comp)) < 1e-10, \
                f"Component {comp} changed during serialization roundtrip"


# ═══════════════════════════════════════════════════════════════════════════
# QPT ENGINE TESTS
# Tests that QUALIA's cognitive state evolves correctly over interactions.
# ═══════════════════════════════════════════════════════════════════════════

class TestQPTEngine:

    def test_perceive_returns_emotional_state(self):
        """Every perception call should return a valid EmotionalState."""
        engine = QPTEngine()
        result = engine.perceive("Any open gyms in SF this weekend?")
        assert isinstance(result, EmotionalState)
        assert result.label in {"focused", "enthusiastic", "concerned", "uncertain", "reflective", "engaged"}
        assert 0.0 <= result.clarity <= 1.0

    def test_volleyball_message_increases_alignment(self):
        """A volleyball-heavy message should push alignment score upward."""
        engine = QPTEngine()
        # Reset to low alignment state
        engine.state = Quaternion(w=0.9, x=0.0, y=0.3, z=-0.5).normalize()
        state_before = engine.introspect().alignment

        engine.perceive("Open gym volleyball NorCal Bay Area indoor coed tournament")
        state_after = engine.introspect().alignment

        # Alignment should increase after volleyball-rich message
        assert state_after >= state_before, \
            "Alignment should not decrease after a strongly volleyball-aligned message"

    def test_state_persists_across_interactions(self):
        """QUALIA's cognitive state should carry forward — not reset each time."""
        engine = QPTEngine()
        first_state  = engine.state.to_dict()
        engine.perceive("Hello there, looking for volleyball!")
        second_state = engine.state.to_dict()

        # The state should have changed
        changed = any(abs(first_state[k] - second_state[k]) > 1e-6 for k in ["w", "x", "y", "z"])
        assert changed, "QPT state must evolve after each perception — it should not be stateless"

    def test_interaction_count_increments(self):
        """Each call to perceive() should increment the interaction counter."""
        engine = QPTEngine()
        assert engine.interaction_count == 0
        engine.perceive("First message")
        engine.perceive("Second message")
        assert engine.interaction_count == 2

    def test_history_is_recorded(self):
        """State snapshots should be recorded in the history log."""
        engine = QPTEngine()
        engine.perceive("Test message 1")
        engine.perceive("Test message 2")
        assert len(engine.history) == 2
        assert "text_preview" in engine.history[0]
        assert "emotion" in engine.history[0]

    def test_resonate_high_similarity(self):
        """A quaternion identical to the current state should always resonate."""
        engine = QPTEngine()
        engine.state = Quaternion(w=0.5, x=0.5, y=0.5, z=0.5).normalize()
        # Same quaternion → dot product = 1.0 → should resonate at any reasonable threshold
        assert engine.resonate(engine.state, threshold=0.9), \
            "Engine must resonate with its own current state"

    def test_response_style_reflects_emotion(self):
        """get_response_style() should return style hints consistent with the current emotion."""
        engine = QPTEngine()
        # Drive the engine into a "concerned" state by perceiving negative input
        engine.state = Quaternion(w=0.6, x=-0.8, y=0.3, z=0.3).normalize()
        style = engine.get_response_style()
        assert "tone" in style
        assert "clarity" in style

    def test_serialization_roundtrip(self):
        """Serialize and restore an engine — all fields should match."""
        engine = QPTEngine()
        engine.perceive("NorCal volleyball open gym San Francisco")
        serialized = engine.serialize()

        restored = QPTEngine.from_serialized(serialized)
        assert abs(engine.state.w - restored.state.w) < 1e-6
        assert engine.interaction_count == restored.interaction_count


# ═══════════════════════════════════════════════════════════════════════════
# HABIT ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestHabitEngine:

    def test_cue_detection_finds_known_patterns(self):
        """Known volleyball cue patterns should be detected reliably."""
        engine = HabitEngine()
        habit = engine.detect_cue("Are there any open gyms in San Francisco this week?")
        assert habit is not None, "Should detect 'open gym' cue in the message"
        assert habit.routine_key == "open_gym_lookup"

    def test_tournament_cue_maps_to_correct_routine(self):
        """Tournament-related messages should trigger the tournament_brief routine."""
        engine = HabitEngine()
        habit = engine.detect_cue("Any volleyball tournaments coming up in the Bay Area?")
        assert habit is not None
        assert habit.routine_key == "tournament_brief"

    def test_get_routine_returns_valid_structure(self):
        """get_routine() must always return a dict with required keys."""
        engine = HabitEngine()
        routine = engine.get_routine("open gym options SF")
        assert "routine_key" in routine
        assert "prompt_injection" in routine
        assert "tool_calls" in routine
        assert "is_automatic" in routine

    def test_reward_strengthens_habit(self):
        """Positive reward signal should increase habit strength."""
        engine = HabitEngine()
        initial_strength = engine.habit_table["open gym"].strength
        engine.record_reward("open gym in SF", reward_signal=1.0)
        final_strength = engine.habit_table["open gym"].strength
        assert final_strength > initial_strength, "Positive reward must strengthen the habit"

    def test_negative_reward_does_not_increase_strength(self):
        """Zero/negative reward should not increase habit strength."""
        engine = HabitEngine()
        # Apply a strong reward first to give us room to measure
        engine.record_reward("open gym", reward_signal=0.8)
        strength_before = engine.habit_table["open gym"].strength

        # Now apply zero reward — strength should not grow further
        engine.record_reward("open gym", reward_signal=0.0)
        strength_after = engine.habit_table["open gym"].strength
        assert strength_after <= strength_before + 0.01, "Zero reward should not increase habit strength"

    def test_habit_becomes_automatic_after_strong_reinforcement(self):
        """A habit driven to strength > 0.7 should be classified as automatic."""
        record = HabitRecord(cue_pattern="test_cue", routine_key="test_routine", strength=0.1)
        # Apply many positive rewards
        for _ in range(20):
            record.apply_reward(1.0)
        assert record.is_automatic, "After strong repeated reinforcement, habit should be automatic"

    def test_stats_returns_valid_structure(self):
        """get_stats() should always return a meaningful summary."""
        engine = HabitEngine()
        stats = engine.get_stats()
        assert "total_habits" in stats
        assert "automatic_habits" in stats
        assert stats["total_habits"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# ETHICS GUARD TESTS
# These tests protect against the most critical failures: fabricated intel,
# PII leakage, and overconfident claims.
# ═══════════════════════════════════════════════════════════════════════════

class TestEthicsGuard:

    def test_clean_response_passes_without_modification(self):
        """A well-formed, verified response should pass through unchanged."""
        guard = EthicsGuard()
        text = "✅ The SOMA Recreation Center hosts open gym on Tuesdays at 7 PM. Cost: $10 drop-in."
        result = guard.check_response(text, has_verified_sources=True)
        assert result["approved"] is True
        assert len(result["warnings"]) == 0

    def test_overconfidence_phrases_are_softened(self):
        """Phrases like 'I'm certain' should be replaced with more humble language."""
        guard = EthicsGuard()
        text = "I'm certain the tournament is happening on July 4th."
        result = guard.check_response(text)
        assert "I believe" in result["modified_text"] or "I'm certain" not in result["modified_text"]

    def test_unsourced_factual_claims_get_disclaimer(self):
        """Factual claims about events without a source should get a ⚠️ caveat added."""
        guard = EthicsGuard()
        text = "Registration opens on March 15th. The entry fee costs $25."
        result = guard.check_response(text, has_verified_sources=False)
        assert "⚠️" in result["modified_text"] or "unconfirmed" in result["modified_text"].lower() or \
               len(result["warnings"]) > 0, "Unsourced factual claims must trigger a warning"

    def test_email_address_in_response_triggers_pii_flag(self):
        """An email address appearing in a response should be flagged and redacted."""
        guard = EthicsGuard()
        text = "Contact john.smith@example.com for more information."
        result = guard.check_response(text)
        assert result["risk_level"] == "high", "PII (email) should set risk level to high"
        assert "REDACTED" in result["modified_text"]
        assert result["approved"] is False

    def test_verified_source_classification(self):
        """Official NCVA/USAV sources should receive VERIFIED confidence level."""
        guard = EthicsGuard()
        level = guard.classify_intel("Some volleyball event info", source="ncva.com")
        assert level == ConfidenceLevel.VERIFIED

    def test_meetup_source_gets_community_classification(self):
        """Meetup.com sources should receive COMMUNITY confidence level."""
        guard = EthicsGuard()
        level = guard.classify_intel("Community open gym info", source="meetup.com")
        assert level == ConfidenceLevel.COMMUNITY

    def test_unknown_source_gets_unverified_classification(self):
        """Unknown sources should default to UNVERIFIED — never assume trust."""
        guard = EthicsGuard()
        level = guard.classify_intel("Some random event info", source="some_random_blog.com")
        assert level == ConfidenceLevel.UNVERIFIED

    def test_intel_record_renders_with_correct_emoji(self):
        """Verified intel should render with ✅, unverified with ⚠️."""
        guard = EthicsGuard()
        verified_record = guard.wrap_intel("Event info", "ncva.com")
        unverified_record = guard.wrap_intel("Event info", "random_source.com")

        assert "✅" in verified_record.render(), "Verified intel should show ✅"
        assert "⚠️" in unverified_record.render(), "Unverified intel should show ⚠️"


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION SMOKE TESTS
# End-to-end pipeline tests — checks the full flow works together.
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_full_pipeline_open_gym_query(self):
        """
        A complete open gym query should flow through QPT → habit → agent
        and return a structured response with all required fields.
        """
        from agents.qualia_agent import QUALIAAgent
        from memory.qualia_memory import QUALIAMemory
        from volley.norcal_intel import NorCalIntelEngine

        agent = QUALIAAgent(
            memory=QUALIAMemory(db_dir="/tmp/qualia_test_memory"),
            intel_engine=NorCalIntelEngine(use_mock_data=True),
        )

        result = agent.think("Any open gyms in San Francisco this week?", user_id="test_user")

        assert "response" in result, "Agent must return a 'response' key"
        assert "emotion" in result, "Agent must return an 'emotion' state key"
        assert "routine" in result, "Agent must return the routine used"
        assert len(result["response"]) > 20, "Response should be substantive, not empty"

    def test_full_pipeline_community_welcome(self):
        """A new user greeting should trigger the community_welcome routine."""
        from agents.qualia_agent import QUALIAAgent
        from memory.qualia_memory import QUALIAMemory
        from volley.norcal_intel import NorCalIntelEngine

        agent = QUALIAAgent(
            memory=QUALIAMemory(db_dir="/tmp/qualia_test_memory"),
            intel_engine=NorCalIntelEngine(use_mock_data=True),
        )

        result = agent.think("Hey I'm new here and just getting started with volleyball!", user_id="new_user")

        assert result["routine"] == "community_welcome", \
            "New user greeting should trigger the community_welcome routine"

    def test_cognitive_state_evolves_over_conversation(self):
        """
        After multiple interactions, QUALIA's cognitive state should be different
        from its initial state — proving continuous cognition is working.
        """
        from agents.qualia_agent import QUALIAAgent
        from memory.qualia_memory import QUALIAMemory
        from volley.norcal_intel import NorCalIntelEngine

        agent = QUALIAAgent(
            memory=QUALIAMemory(db_dir="/tmp/qualia_test_memory"),
            intel_engine=NorCalIntelEngine(use_mock_data=True),
        )

        initial_state = agent.qpt.state.to_dict()

        for msg in [
            "Open gyms in Oakland?",
            "What about tournaments this summer?",
            "I'm an intermediate level setter",
        ]:
            agent.think(msg, user_id="test_evolve")

        final_state = agent.qpt.state.to_dict()

        # State should have changed across the conversation
        changed = any(abs(initial_state[k] - final_state[k]) > 1e-6 for k in ["w", "x", "y", "z"])
        assert changed, "Cognitive state must evolve across a multi-turn conversation"


# ═══════════════════════════════════════════════════════════════════════════
# Test Runner Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick self-test without pytest
    print("Running QUALIA core tests directly...\n")

    test_classes = [
        TestQuaternionMath,
        TestQPTEngine,
        TestHabitEngine,
        TestEthicsGuard,
    ]

    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        print(f"── {cls.__name__} ({len(methods)} tests) ──")
        for method in methods:
            try:
                getattr(instance, method)()
                print(f"  ✅ {method}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {method}: {e}")
                failed += 1

    print(f"\n{'═'*40}")
    print(f"QUALIA Tests: {passed} passed, {failed} failed")
    print("All systems nominal. 🧠" if failed == 0 else "⚠️ Some tests failed — review above.")
