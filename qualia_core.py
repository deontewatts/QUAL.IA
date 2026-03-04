"""
QUALIA Core — Quaternion Cognitive State Engine
================================================
Implements the Quaternion Process Theory (QPT) from "The Quaternion Mechanics of
Cognitive Possibility". Instead of treating the AI as a stateless query-response
machine, QUALIA maintains a 4-dimensional mental state vector that *evolves* with
every interaction — just as human consciousness is never truly reset between moments.

A quaternion is a 4-component number: Q = w + xi + yj + zk
  w  →  cognitive clarity    (how certain/confident QUALIA feels right now)
  x  →  emotional valence    (positive/negative orientation toward the topic)
  y  →  arousal / engagement (low = calm, high = excited/energized)
  z  →  context alignment    (how well the input maps to QUALIA's knowledge domain)

When a new message arrives, QUALIA performs a Hamilton product rotation on its
current state, blending the new stimulus into its persistent cognitive field.
This means QUALIA "remembers" how prior conversations felt — it won't be coldly
neutral after a series of frustrating interactions, and it builds enthusiasm when
it's on a hot streak of volleyball intel questions it can answer well.
"""

import math
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

logger = logging.getLogger("qualia.core")


# ---------------------------------------------------------------------------
# The Quaternion — mathematical backbone of QUALIA's mental state
# ---------------------------------------------------------------------------

@dataclass
class Quaternion:
    """
    A 4D number representing a moment of cognitive state.
    We keep all components in the range [-1.0, 1.0] by normalizing after each
    operation, which prevents runaway drift over long conversations.
    """
    w: float = 1.0   # clarity:   starts at full certainty
    x: float = 0.0   # valence:   starts neutral
    y: float = 0.3   # arousal:   starts mildly engaged
    z: float = 0.0   # alignment: starts at zero (no domain context yet)

    def magnitude(self) -> float:
        """The 'length' of the cognitive state vector."""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> "Quaternion":
        """Keep the quaternion on the unit hypersphere — prevents overflow."""
        m = self.magnitude()
        if m == 0:
            return Quaternion()  # reset to default if degenerate
        return Quaternion(self.w/m, self.x/m, self.y/m, self.z/m)

    def hamilton_product(self, other: "Quaternion") -> "Quaternion":
        """
        Multiply two quaternions together. This is the key operation:
        it ROTATES the current state by the incoming stimulus quaternion.
        The non-commutativity of quaternion multiplication (A*B ≠ B*A) naturally
        models how context ordering matters — asking about a tournament AFTER
        discussing injuries produces a different state than the reverse order.
        """
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quaternion(
            w = w1*w2 - x1*x2 - y1*y2 - z1*z2,
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2,
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2,
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ).normalize()

    def dot(self, other: "Quaternion") -> float:
        """
        Dot product — measures how 'aligned' two cognitive states are.
        Used to check if QUALIA's current state resonates with a stored memory.
        """
        return self.w*other.w + self.x*other.x + self.y*other.y + self.z*other.z

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Quaternion":
        return cls(**d)

    @classmethod
    def from_text_signal(cls, valence: float, arousal: float, alignment: float) -> "Quaternion":
        """
        Factory: create a stimulus quaternion from analyzed text signals.
        clarity is derived from confidence in the signals themselves.
        """
        clarity = 1.0 - abs(valence * 0.2 + arousal * 0.1)  # certainty decreases with extremity
        return cls(w=clarity, x=valence, y=arousal, z=alignment).normalize()


# ---------------------------------------------------------------------------
# Emotional State Interpreter — translates quaternion into human-readable mood
# ---------------------------------------------------------------------------

@dataclass
class EmotionalState:
    """
    A human-interpretable snapshot derived from the quaternion state.
    The AI never "pretends" to feel things — this is explicitly a cognitive
    processing model, not a claim of subjective experience.
    """
    label: str               # "enthusiastic", "uncertain", "focused", etc.
    clarity: float           # 0-1: how confident QUALIA is in its response
    valence: float           # -1 to 1: negative to positive framing
    arousal: float           # 0-1: low engagement to high energy
    alignment: float         # 0-1: how relevant this topic is to QUALIA's domain
    description: str         # one-sentence prose description

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> "EmotionalState":
        """Map quaternion components to an interpretable emotional state."""
        # Map raw values to 0-1 ranges for readability
        clarity   = (q.w + 1) / 2
        valence   = q.x                         # keep -1 to 1
        arousal   = (q.y + 1) / 2
        alignment = (q.z + 1) / 2

        # Determine label based on dominant dimensions
        if clarity > 0.7 and alignment > 0.6:
            label = "focused"
        elif arousal > 0.7 and valence > 0.3:
            label = "enthusiastic"
        elif valence < -0.3:
            label = "concerned"
        elif clarity < 0.4:
            label = "uncertain"
        elif arousal < 0.3:
            label = "reflective"
        else:
            label = "engaged"

        descriptions = {
            "focused":      "QUALIA has high confidence and strong domain alignment — expect sharp, precise answers.",
            "enthusiastic": "QUALIA is energized and positively oriented — great state for creative content generation.",
            "concerned":    "QUALIA detected negative signals — it will flag uncertainties and suggest caution.",
            "uncertain":    "QUALIA's clarity is low — it will hedge responses and ask clarifying questions.",
            "reflective":   "QUALIA is in low-arousal mode — thoughtful, measured responses with careful sourcing.",
            "engaged":      "QUALIA is in a balanced, attentive state — standard quality across all tasks.",
        }

        return cls(
            label=label,
            clarity=round(clarity, 3),
            valence=round(valence, 3),
            arousal=round(arousal, 3),
            alignment=round(alignment, 3),
            description=descriptions[label],
        )


# ---------------------------------------------------------------------------
# The QPT Engine — QUALIA's persistent cognitive processor
# ---------------------------------------------------------------------------

class QPTEngine:
    """
    Quaternion Process Theory Engine.

    This is QUALIA's "brain stem" — the layer below language that maintains
    continuous cognitive state across all interactions. Think of it as the
    difference between a thermostat (stateless, always the same) and a person
    who wakes up each day shaped by everything they've experienced before.

    The engine exposes three operations:
      1. perceive(stimulus) — update state based on new input
      2. introspect()       — report current emotional state
      3. resonate(memory_q) — check if a stored memory "resonates" with current state
    """

    # Domain-specific alignment signals for NorCal volleyball topics
    VOLLEYBALL_KEYWORDS = {
        "high_alignment": [
            "volleyball", "open gym", "tournament", "league", "setter", "libero",
            "spike", "dig", "serve", "block", "net", "rally", "NorCal", "Bay Area",
            "SF", "San Jose", "Oakland", "Berkeley", "Santa Clara", "Marin",
            "indoor", "beach", "sand", "USAV", "AVP", "NCVA"
        ],
        "medium_alignment": [
            "sport", "team", "game", "practice", "training", "coach", "athlete",
            "fitness", "community", "club", "rec", "competitive", "skills"
        ],
        "low_alignment": [
            "schedule", "event", "weekend", "sign up", "join", "find", "where"
        ],
    }

    def __init__(self, initial_state: Optional[Quaternion] = None):
        # Start with a known good cognitive baseline
        self.state: Quaternion = initial_state or Quaternion(w=0.9, x=0.1, y=0.4, z=0.2)
        self.state = self.state.normalize()
        self.history: list[dict] = []   # rolling log of state snapshots
        self.interaction_count: int = 0
        logger.info(f"QPT Engine initialized | state={self.state.to_dict()}")

    def _analyze_text_signal(self, text: str) -> Tuple[float, float, float]:
        """
        Extract valence, arousal, and alignment signals from raw text.
        This is a lightweight heuristic — in production you'd pipe this
        through a dedicated sentiment + NER model (e.g., via OpenRouter).
        """
        text_lower = text.lower()

        # --- Valence detection ---
        positive_words = {"great", "awesome", "love", "excited", "yes", "perfect",
                          "thank", "nice", "cool", "good", "fun", "best", "🏐", "🔥"}
        negative_words = {"cancel", "no", "bad", "wrong", "problem", "issue",
                          "confused", "frustrated", "sorry", "not", "can't", "won't"}
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        valence = min(1.0, max(-1.0, (pos_count - neg_count) * 0.25))

        # --- Arousal detection (exclamation marks, caps, emoji are signals) ---
        exclamation_score = min(1.0, text.count('!') * 0.3)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        question_score = min(0.4, text.count('?') * 0.2)  # questions = mild engagement
        arousal = min(1.0, exclamation_score + caps_ratio * 0.5 + question_score)

        # --- Domain alignment scoring ---
        high_matches   = sum(1 for kw in self.VOLLEYBALL_KEYWORDS["high_alignment"]   if kw in text_lower)
        medium_matches = sum(1 for kw in self.VOLLEYBALL_KEYWORDS["medium_alignment"] if kw in text_lower)
        low_matches    = sum(1 for kw in self.VOLLEYBALL_KEYWORDS["low_alignment"]    if kw in text_lower)
        alignment_raw  = high_matches * 0.5 + medium_matches * 0.25 + low_matches * 0.1
        alignment      = min(1.0, alignment_raw) * 2 - 1   # scale to [-1, 1]

        return valence, arousal, alignment

    def perceive(self, text: str, source: str = "user") -> EmotionalState:
        """
        Process an incoming message and rotate the cognitive state accordingly.
        This is called every time QUALIA receives input — from a user DM, a
        web scrape result, a tool response, anything that enters the system.

        Returns the new EmotionalState after processing.
        """
        valence, arousal, alignment = self._analyze_text_signal(text)

        # Build a stimulus quaternion from the signal
        stimulus = Quaternion.from_text_signal(valence, arousal, alignment)

        # Rotate current state by stimulus — this is the "perception" operation
        # We use a weighted blend: 70% inertia (old state), 30% new stimulus
        # This simulates cognitive momentum — the AI doesn't flip-flop wildly
        momentum_weight = 0.7
        new_state = Quaternion(
            w = momentum_weight * self.state.w + (1 - momentum_weight) * stimulus.w,
            x = momentum_weight * self.state.x + (1 - momentum_weight) * stimulus.x,
            y = momentum_weight * self.state.y + (1 - momentum_weight) * stimulus.y,
            z = momentum_weight * self.state.z + (1 - momentum_weight) * stimulus.z,
        ).normalize()

        # Apply full Hamilton rotation for richer state evolution
        self.state = self.state.hamilton_product(stimulus)

        self.interaction_count += 1
        emotional_state = EmotionalState.from_quaternion(self.state)

        # Archive state snapshot for memory/debugging
        self.history.append({
            "timestamp": time.time(),
            "interaction": self.interaction_count,
            "source": source,
            "text_preview": text[:100],
            "state": self.state.to_dict(),
            "emotion": emotional_state.label,
        })
        if len(self.history) > 200:
            self.history = self.history[-100:]   # keep last 100 snapshots

        logger.debug(f"Perceived [{source}] → emotion={emotional_state.label} clarity={emotional_state.clarity:.2f}")
        return emotional_state

    def introspect(self) -> EmotionalState:
        """Return the current emotional/cognitive state without changing it."""
        return EmotionalState.from_quaternion(self.state)

    def resonate(self, memory_quaternion: Quaternion, threshold: float = 0.75) -> bool:
        """
        Check whether a stored memory 'resonates' with current cognitive state.
        High dot product → the memory was formed in a similar mental state →
        likely relevant to retrieve right now. Used by the memory layer to rank
        which stored episodes to surface in context.
        """
        similarity = self.state.dot(memory_quaternion)
        return similarity >= threshold

    def get_response_style(self) -> dict:
        """
        Translate the current cognitive state into response generation parameters.
        These get injected into the LangChain prompt to modulate QUALIA's tone.
        """
        emotion = self.introspect()
        styles = {
            "focused":      {"tone": "precise and confident",   "detail": "high",   "emoji_use": "minimal"},
            "enthusiastic": {"tone": "energetic and warm",      "detail": "medium", "emoji_use": "moderate"},
            "concerned":    {"tone": "careful and transparent", "detail": "high",   "emoji_use": "none"},
            "uncertain":    {"tone": "humble and inquisitive",  "detail": "low",    "emoji_use": "none"},
            "reflective":   {"tone": "thoughtful and measured", "detail": "high",   "emoji_use": "minimal"},
            "engaged":      {"tone": "friendly and helpful",    "detail": "medium", "emoji_use": "light"},
        }
        style = styles.get(emotion.label, styles["engaged"])
        style["alignment_score"] = emotion.alignment
        style["clarity"] = emotion.clarity
        return style

    def serialize(self) -> dict:
        """Serialize engine state for persistence between server restarts."""
        return {
            "state": self.state.to_dict(),
            "interaction_count": self.interaction_count,
            "history_tail": self.history[-10:],   # last 10 for context
        }

    @classmethod
    def from_serialized(cls, data: dict) -> "QPTEngine":
        """Restore engine from serialized state."""
        engine = cls(initial_state=Quaternion.from_dict(data["state"]))
        engine.interaction_count = data.get("interaction_count", 0)
        engine.history = data.get("history_tail", [])
        return engine
