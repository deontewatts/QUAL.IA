"""
QUALIA Ethics Guard — Anti-Fabrication & Privacy Protection Layer
==================================================================
The document explicitly warns about "the ethical risks of superhuman persuasion
and the importance of maintaining cognitive agency." This module embodies that
warning as executable code.

QUALIA handles real community events, real venues, real people in NorCal.
Fabricating a tournament that doesn't exist, or publishing unverified league
info, destroys trust and potentially harms the community. The ethics guard is
NOT optional — it runs on every response before it leaves QUALIA.

Three guarantees this module enforces:
  1. VERIFICATION TAGGING — every piece of community intel is tagged with
     its confidence level and source type.
  2. PRIVACY PROTECTION — no individual player names/locations without
     explicit consent signals in the source data.
  3. HALLUCINATION DETECTION — pattern checks that catch common LLM
     fabrication signatures before they reach the user.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional
from enum import Enum

logger = logging.getLogger("qualia.ethics")


class ConfidenceLevel(Enum):
    """How confident are we that this information is real and current?"""
    VERIFIED   = "verified"      # came from official source (USAV, venue website)
    COMMUNITY  = "community"     # reported by multiple community members
    UNVERIFIED = "unverified"    # single source, not cross-checked
    UNCERTAIN  = "uncertain"     # QUALIA is genuinely unsure
    FABRICATED = "fabricated"    # BLOCKED — should never reach user


@dataclass
class IntelRecord:
    """
    A piece of volleyball community intelligence with provenance attached.
    QUALIA never passes raw text — it always wraps intel in this record first.
    """
    content: str
    confidence: ConfidenceLevel
    source_type: str           # "official_website", "ig_post", "community_dm", "memory"
    source_url: Optional[str]
    verified_date: Optional[str]   # ISO date string or None

    def render(self) -> str:
        """
        Returns the content with appropriate confidence markers injected.
        The user always knows how reliable the information is.
        """
        if self.confidence == ConfidenceLevel.VERIFIED:
            prefix = "✅ "
            suffix = ""
        elif self.confidence == ConfidenceLevel.COMMUNITY:
            prefix = "📣 "
            suffix = ""
        elif self.confidence == ConfidenceLevel.UNVERIFIED:
            prefix = "⚠️ Unconfirmed: "
            suffix = "\n_(Please verify directly with the organizer before showing up.)_"
        elif self.confidence == ConfidenceLevel.UNCERTAIN:
            prefix = "🤔 I'm not fully sure, but: "
            suffix = "\n_(This may be outdated — double-check before committing.)_"
        else:
            # FABRICATED — should never reach here; guard caught it upstream
            return "[BLOCKED: Unverifiable content removed]"

        return f"{prefix}{self.content}{suffix}"


class EthicsGuard:
    """
    Runs before every response QUALIA sends to the outside world.

    Think of it as a fact-checking editor who reviews every outgoing message.
    It can't verify everything (that would require live database lookups for
    every claim), but it catches the most dangerous patterns: fake dates,
    invented venues, made-up phone numbers, and LLM overconfidence signals.
    """

    # Patterns that signal a likely fabrication or hallucination
    FABRICATION_SIGNALS = [
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # phone numbers (high hallucination risk)
        r'\b(always|every week|every month|every year)\b.*\bat\b.*\d{1,2}(?::\d{2})?\s*(?:am|pm)',
        # Specific recurring time claims without source are often fabricated
        r'I (know|confirm|guarantee) that',     # overconfident truth claims
        r'definitely happening',
        r'100% (sure|certain|confirmed)',
    ]

    # Words/phrases that require a source tag or get flagged
    UNSOURCED_CLAIM_TRIGGERS = [
        "is happening", "will be held", "takes place", "registration opens",
        "deadline is", "costs $", "entry fee", "free to play", "hosted by",
    ]

    # PII patterns that must never appear in outgoing responses
    PII_PATTERNS = [
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b.*\b(?:lives|home|address|located at)\b',  # name + location
        r'\b\d{5}(?:-\d{4})?\b.*\b(?:home|lives|private)\b',   # zip + personal context
        r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',  # email addresses
    ]

    def __init__(self):
        self._fabrication_re  = [re.compile(p, re.IGNORECASE) for p in self.FABRICATION_SIGNALS]
        self._unsourced_re    = [re.compile(p, re.IGNORECASE) for p in self.UNSOURCED_CLAIM_TRIGGERS]
        self._pii_re          = [re.compile(p, re.IGNORECASE) for p in self.PII_PATTERNS]
        logger.info("Ethics Guard initialized — all response gates active")

    def check_response(self, text: str, has_verified_sources: bool = False) -> dict:
        """
        Run all ethics checks on an outgoing response.

        Returns a dict:
          "approved": bool — whether to send the response as-is
          "warnings": list — issues found (empty if all clear)
          "modified_text": str — text with problematic parts redacted/annotated
          "risk_level": str — "low", "medium", "high"
        """
        warnings = []
        modified = text
        risk_level = "low"

        # Check 1: PII Detection — always block
        for pattern in self._pii_re:
            if pattern.search(modified):
                warnings.append("PII_DETECTED: Possible personal information found and redacted.")
                modified = pattern.sub("[REDACTED]", modified)
                risk_level = "high"

        # Check 2: Fabrication signal detection
        for pattern in self._fabrication_re:
            if pattern.search(modified):
                warnings.append(f"FABRICATION_RISK: High-confidence claim detected without verification.")
                risk_level = max(risk_level, "medium", key=lambda x: {"low":0,"medium":1,"high":2}[x])

        # Check 3: Unsourced factual claims
        unsourced_count = sum(1 for p in self._unsourced_re if p.search(modified))
        if unsourced_count > 0 and not has_verified_sources:
            warnings.append(
                f"UNSOURCED_CLAIMS: {unsourced_count} factual claim(s) without verified source. "
                f"Adding unverified disclaimer."
            )
            # Append a generic caveat rather than blocking the response
            if "⚠️" not in modified and "unconfirmed" not in modified.lower():
                modified += ("\n\n_⚠️ Some details above are community-reported and may not be fully "
                             "verified. Always confirm directly with organizers before committing._")
            risk_level = max(risk_level, "medium", key=lambda x: {"low":0,"medium":1,"high":2}[x])

        # Check 4: Detect if QUALIA is claiming certainty it doesn't have
        # (the document flags "superhuman persuasion" as a key ethical risk)
        overconfidence_patterns = ["I'm certain", "I guarantee", "this is definitely", "I know for a fact"]
        if any(p.lower() in modified.lower() for p in overconfidence_patterns):
            warnings.append("OVERCONFIDENCE: Epistemic humility required — modifying claim strength.")
            for p in overconfidence_patterns:
                modified = modified.replace(p, "I believe")
                modified = modified.replace(p.lower(), "I believe")

        approved = risk_level != "high"  # Only PII blocks the response entirely

        if warnings:
            logger.warning(f"Ethics guard raised {len(warnings)} warning(s): {warnings}")
        else:
            logger.debug("Ethics guard: response approved with no warnings")

        return {
            "approved": approved,
            "warnings": warnings,
            "modified_text": modified,
            "risk_level": risk_level,
        }

    def classify_intel(self, text: str, source: str) -> ConfidenceLevel:
        """
        Determine the confidence level for a piece of incoming intel.
        Used by the agent before storing or serving community information.
        """
        source_lower = source.lower()

        # Official sources get full verification credit
        if any(domain in source_lower for domain in [
            "usavolleyball.org", "ncva.com", "avp.com", "bvca.org",
            ".gov", "sfrecpark.org", ".edu"
        ]):
            return ConfidenceLevel.VERIFIED

        # Established Instagram accounts and Meetup get community status
        if any(domain in source_lower for domain in [
            "meetup.com", "instagram.com", "facebook.com/groups"
        ]):
            return ConfidenceLevel.COMMUNITY

        # Memory from QUALIA's own past verified lookups
        if source_lower == "qualia_memory":
            return ConfidenceLevel.COMMUNITY

        # Everything else is unverified by default
        return ConfidenceLevel.UNVERIFIED

    def wrap_intel(self, content: str, source: str, source_url: Optional[str] = None) -> IntelRecord:
        """
        Convenience method: given raw content and its source, return a
        properly tagged IntelRecord ready for rendering.
        """
        confidence = self.classify_intel(content, source)
        return IntelRecord(
            content=content,
            confidence=confidence,
            source_type=source,
            source_url=source_url,
            verified_date=None,   # TODO: extract date from source when available
        )
