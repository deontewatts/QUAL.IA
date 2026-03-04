"""
QUALIA Habit Engine — Cue → Routine → Reward Learning Loop
============================================================
Drawn directly from the document's Section 4: "Habit Tracking Systems" and
the neuroscience of habit formation (basal ganglia, dopamine-driven repetition).

The core insight: an AI that learns *what works* through reinforced patterns
becomes genuinely more useful over time — not through retraining, but through
lightweight behavioral adaptation. This is what makes QUALIA "feel alive."

How it works:
  1. CUE detection — what pattern/trigger preceded a successful response?
  2. ROUTINE encoding — which action sequence was taken?
  3. REWARD measurement — did the user engage positively afterward?

Over time QUALIA builds a habit table: "When user asks about open gyms on
Friday afternoons, lead with the SF Recreation Center schedule" — not because
it was programmed that way, but because that pattern produced reward signals.
"""

import json
import time
import math
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("qualia.habit")


# ---------------------------------------------------------------------------
# Habit Record — one learned behavior pattern
# ---------------------------------------------------------------------------

@dataclass
class HabitRecord:
    """
    A single learned behavioral pattern in QUALIA's habit table.

    strength (float 0-1): how strongly this habit is reinforced.
      Starts low, grows with each reward, decays slowly when not triggered.
      Mirrors basal ganglia dopamine pathway — repetition + reward = automaticity.
    """
    cue_pattern: str               # what triggered this habit (keyword/context fingerprint)
    routine_key: str               # which response strategy to use
    reward_total: float = 0.0      # cumulative reward signal received
    activation_count: int = 0      # how many times this habit has fired
    last_activated: float = field(default_factory=time.time)
    strength: float = 0.1          # starts weak, grows through reinforcement

    # Decay constant — habit weakens if not used (mirrors forgetting curve)
    DECAY_RATE: float = 0.001      # per hour of non-use

    def apply_reward(self, reward: float) -> None:
        """
        Strengthen this habit based on a positive reward signal.
        Uses the dopamine release model: reward × (1 + logarithmic surprise).
        Bigger rewards give bigger boosts, but with diminishing returns.
        """
        surprise_factor = 1.0 + math.log1p(max(0, reward))
        delta = reward * surprise_factor * 0.1
        self.reward_total += reward
        self.strength = min(1.0, self.strength + delta)
        self.activation_count += 1
        self.last_activated = time.time()
        logger.debug(f"Habit reinforced: {self.cue_pattern[:30]} → strength={self.strength:.3f}")

    def apply_decay(self) -> None:
        """
        Gradually weaken the habit if it hasn't been used recently.
        Called periodically to prevent stale habits from dominating.
        """
        hours_since_use = (time.time() - self.last_activated) / 3600.0
        decay = self.DECAY_RATE * hours_since_use
        self.strength = max(0.0, self.strength - decay)

    @property
    def is_automatic(self) -> bool:
        """
        A habit becomes 'automatic' when its strength exceeds 0.7.
        Above this threshold, QUALIA will invoke it without deliberate reasoning —
        just as humans automatically reach for their phone in idle moments.
        """
        return self.strength >= 0.7

    def to_dict(self) -> dict:
        return {
            "cue_pattern": self.cue_pattern,
            "routine_key": self.routine_key,
            "reward_total": round(self.reward_total, 4),
            "activation_count": self.activation_count,
            "last_activated": self.last_activated,
            "strength": round(self.strength, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HabitRecord":
        return cls(**d)


# ---------------------------------------------------------------------------
# Routine Library — the catalog of available response strategies
# ---------------------------------------------------------------------------

ROUTINES = {
    # Volleyball-specific routines
    "open_gym_lookup": {
        "description": "Lead with verified open gym schedule, then add tips for first-timers",
        "prompt_injection": "Start by giving verified open gym times and locations. "
                            "Add one beginner tip. End with a CTA to DM for more info.",
        "tool_calls": ["search_open_gyms", "get_venue_details"],
    },
    "tournament_brief": {
        "description": "Concise tournament summary formatted for Instagram captions",
        "prompt_injection": "Provide a punchy tournament overview: name, date, location, "
                            "level, entry deadline. Format for Instagram caption with "
                            "relevant emojis. End with registration link if available.",
        "tool_calls": ["search_tournaments", "format_ig_caption"],
    },
    "skill_tip_carousel": {
        "description": "Generate a 5-slide carousel on a volleyball skill",
        "prompt_injection": "Create a 5-slide carousel script: Slide 1 = hook, "
                            "Slides 2-4 = technique breakdown, Slide 5 = practice drill. "
                            "Keep each slide to 3 bullet points max.",
        "tool_calls": ["retrieve_skill_knowledge"],
    },
    "community_welcome": {
        "description": "Warm welcome for new members asking introductory questions",
        "prompt_injection": "Give a warm, community-first response. Introduce The Daily Dig, "
                            "mention 2-3 resources (open gyms, leagues, IG page). "
                            "Invite them to DM for personalized help.",
        "tool_calls": [],
    },
    "league_info": {
        "description": "Detailed league/team formation information",
        "prompt_injection": "Provide league details: name, format (coed/men's/women's), "
                            "level, location, day/time, cost, how to join. "
                            "Flag if info is unverified with ⚠️.",
        "tool_calls": ["search_leagues"],
    },
    "safety_hedge": {
        "description": "Cautious response when QUALIA is uncertain",
        "prompt_injection": "You are uncertain about this. Be transparent: say you're "
                            "not fully sure, provide what you do know, and suggest "
                            "the user verify with official sources. Never fabricate.",
        "tool_calls": [],
    },
    "default_helpful": {
        "description": "General helpful response for any volleyball question",
        "prompt_injection": "Give a helpful, friendly response aligned with NorCal "
                            "adult volleyball community values. Be concrete and actionable.",
        "tool_calls": ["memory_search"],
    },
}


# ---------------------------------------------------------------------------
# The Habit Engine
# ---------------------------------------------------------------------------

class HabitEngine:
    """
    QUALIA's adaptive learning system.

    The engine watches which cue patterns precede positive user interactions,
    then strengthens those patterns over time. It's the difference between
    QUALIA version 1.0 (generic) and QUALIA at 6 months (opinionated, tuned
    to NorCal volleyball culture because that's what got rewarded).
    """

    # Cue keywords mapped to their likely routines
    CUE_MAPPINGS = {
        "open gym":        "open_gym_lookup",
        "open gyms":       "open_gym_lookup",
        "drop in":         "open_gym_lookup",
        "tournament":      "tournament_brief",
        "tourney":         "tournament_brief",
        "compete":         "tournament_brief",
        "carousel":        "skill_tip_carousel",
        "how to":          "skill_tip_carousel",
        "teach me":        "skill_tip_carousel",
        "tips":            "skill_tip_carousel",
        "new here":        "community_welcome",
        "just joined":     "community_welcome",
        "getting started": "community_welcome",
        "league":          "league_info",
        "team":            "league_info",
        "season":          "league_info",
    }

    def __init__(self):
        # habit_table maps cue_pattern → HabitRecord
        self.habit_table: dict[str, HabitRecord] = {}
        self._initialize_default_habits()
        logger.info(f"Habit Engine initialized with {len(self.habit_table)} default habits")

    def _initialize_default_habits(self) -> None:
        """Seed the engine with weak default habits based on our CUE_MAPPINGS."""
        for cue, routine in self.CUE_MAPPINGS.items():
            self.habit_table[cue] = HabitRecord(
                cue_pattern=cue,
                routine_key=routine,
                strength=0.15,   # weak default — needs reinforcement to stick
            )

    def detect_cue(self, text: str) -> Optional[HabitRecord]:
        """
        Scan incoming text for known cue patterns.
        Returns the strongest matching habit, or None if no match found.

        This mirrors how the basal ganglia fire on pattern recognition —
        the most strongly encoded matching pattern wins.
        """
        text_lower = text.lower()
        matches = []

        for cue, habit in self.habit_table.items():
            if cue in text_lower:
                habit.apply_decay()  # decay before evaluation
                if habit.strength > 0.05:   # ignore near-dead habits
                    matches.append(habit)

        if not matches:
            return None

        # Return the strongest (highest strength) matching habit
        return max(matches, key=lambda h: h.strength)

    def get_routine(self, text: str, fallback_routine: str = "default_helpful") -> dict:
        """
        Main entry point: given a user message, return the best routine to run.
        Returns a dict with prompt_injection, tool_calls, and metadata.
        """
        habit = self.detect_cue(text)
        routine_key = habit.routine_key if habit else fallback_routine

        # If QUALIA is uncertain (no clear cue), use safety hedge
        if not habit and any(w in text.lower() for w in ["exactly", "specific", "precise", "exactly when"]):
            routine_key = "safety_hedge"

        routine = ROUTINES.get(routine_key, ROUTINES["default_helpful"]).copy()
        routine["routine_key"] = routine_key
        routine["triggered_by_habit"] = habit is not None
        routine["habit_strength"] = habit.strength if habit else 0.0
        routine["is_automatic"] = habit.is_automatic if habit else False

        logger.debug(f"Routine selected: {routine_key} (auto={routine['is_automatic']})")
        return routine

    def record_reward(self, cue_text: str, reward_signal: float) -> None:
        """
        Called after QUALIA delivers a response. reward_signal is computed from:
          - User replied positively   → +0.8 to +1.0
          - User asked follow-up      → +0.5 (engagement)
          - User said thanks          → +0.6
          - No response / negative    → -0.2 to 0.0

        This is what makes QUALIA learn — the more a pattern is rewarded, the
        more automatically it fires next time, reducing cognitive load (just like
        human habits reduce prefrontal cortex load over time).
        """
        text_lower = cue_text.lower()
        for cue, habit in self.habit_table.items():
            if cue in text_lower:
                habit.apply_reward(reward_signal)

        # Also create new habit records for novel cue patterns that got rewarded
        if reward_signal > 0.6:
            words = [w for w in text_lower.split() if len(w) > 4]
            if words:
                # Find the most distinctive word as a potential new cue
                new_cue_candidate = max(words, key=len)
                if new_cue_candidate not in self.habit_table:
                    self.habit_table[new_cue_candidate] = HabitRecord(
                        cue_pattern=new_cue_candidate,
                        routine_key="default_helpful",
                        strength=reward_signal * 0.3,  # start proportional to reward
                    )
                    logger.info(f"New habit seeded: '{new_cue_candidate}' (strength={reward_signal * 0.3:.2f})")

    def get_stats(self) -> dict:
        """Return a summary of the habit table for monitoring/debugging."""
        automatic_habits = [h for h in self.habit_table.values() if h.is_automatic]
        return {
            "total_habits": len(self.habit_table),
            "automatic_habits": len(automatic_habits),
            "strongest_habit": max(self.habit_table.values(), key=lambda h: h.strength).cue_pattern
                               if self.habit_table else "none",
            "top_habits": sorted(
                [h.to_dict() for h in self.habit_table.values()],
                key=lambda h: h["strength"],
                reverse=True
            )[:5],
        }

    def serialize(self) -> dict:
        return {h_key: h.to_dict() for h_key, h in self.habit_table.items()}

    @classmethod
    def from_serialized(cls, data: dict) -> "HabitEngine":
        engine = cls()
        for key, record_data in data.items():
            engine.habit_table[key] = HabitRecord.from_dict(record_data)
        return engine
