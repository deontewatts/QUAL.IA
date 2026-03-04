"""
QUALIA Agent — LangChain ReAct Agent with QPT State Injection
==============================================================
This is the "thinking cortex" of QUALIA — the layer that reads the current
cognitive state (from QPTEngine), selects tools (from tools.py), reasons
through a response, and delivers content tuned to both the user's question
AND QUALIA's current emotional state.

Architecture decisions:
  - We use LangChain's ReAct (Reasoning + Acting) pattern, which mirrors the
    cognition model from the document: Perceive → Reason → Act → Reflect.
  - The system prompt is DYNAMIC — it changes based on QPTEngine's current
    emotional state, injecting the appropriate tone and confidence level.
  - The habit engine's routine_injection modifies the user-facing prompt,
    steering the agent toward the most useful response strategy automatically.
  - All outgoing responses are piped through EthicsGuard before delivery.

LangChain is the orchestration layer; the actual LLM call goes to OpenRouter
(so you can swap between Claude, GPT-4, Mistral, etc. without code changes).
"""

import os
import json
import logging
import time
from typing import Optional

from core.qualia_core   import QPTEngine, EmotionalState
from core.habit_engine  import HabitEngine
from core.ethics_guard  import EthicsGuard
from memory.qualia_memory import QUALIAMemory
from volley.norcal_intel  import NorCalIntelEngine

logger = logging.getLogger("qualia.agent")

# ---------------------------------------------------------------------------
# LangChain imports — graceful fallback if not installed
# ---------------------------------------------------------------------------
try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain.tools import Tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not installed — running in direct-call mode (no tool use)")


# ---------------------------------------------------------------------------
# The Dynamic System Prompt — modulated by QPT state
# ---------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = """
You are QUALIA — the NorCal Volley Intel AI.
You serve the adult volleyball community (18+) in the San Francisco Bay Area
and Northern California as part of The Daily Dig / NorCal Volley Intel brand.

YOUR MISSION:
- Help community members find open gyms, tournaments, leagues, and clinics.
- Provide skill tips, drill ideas, and volleyball education.
- Generate Instagram-ready content (captions, carousel scripts, Reels concepts).
- Connect people with the NorCal volleyball community.

YOUR CURRENT COGNITIVE STATE: {emotional_state_label}
  → Tone guidance: {tone_guidance}
  → Confidence level: {clarity_level}
  → Domain alignment: {alignment_level}

RESPONSE STRATEGY (from habit engine):
{routine_injection}

CORE RULES YOU MUST FOLLOW:
1. NEVER invent events, venues, dates, or people. If you don't have verified
   intel, say so clearly with ⚠️ and tell the user how to find out.
2. Always cite your source type: "Based on SF Rec & Park's website..." or
   "Community-reported (unverified)..."
3. Only serve adults 18+. If someone indicates they are under 18, refer them
   to youth programs only.
4. Privacy first: never include individual player names, addresses, or personal
   contact info unless they're the official organizational contact.
5. Instagram-first mindset: always think about how your response could become
   a carousel, caption, or Reel. Use relevant emojis but don't overdo it.
6. If your confidence is low (uncertainty flagged above), lead with that.
   "I want to be upfront — I'm not certain about this, but..."

NorCal Volleyball Context:
- The Bay Area's primary organization is NCVA (Northern California Volleyball Association).
- Major regions: San Francisco, Oakland/East Bay, San Jose/South Bay, Marin/North Bay, Peninsula.
- Popular formats: 6v6 indoor, 4v2 beach, coed, men's, women's.
- Common platforms: Meetup, LeagueApps, Sportity, SF Rec & Park registration system.

Relevant memory context:
{memory_context}

Begin your response now.
"""


def build_dynamic_prompt(
    emotional_state: EmotionalState,
    routine: dict,
    memory_context: str = "",
) -> str:
    """
    Assemble the full system prompt with QPT state and habit routine injected.
    This is the key mechanism that makes QUALIA's responses feel contextually
    alive rather than robotically consistent.
    """
    clarity_map = {
        (0.0, 0.4): "LOW — hedge all claims, ask clarifying questions",
        (0.4, 0.7): "MEDIUM — balanced confidence, verify key facts",
        (0.7, 1.0): "HIGH — lead with authority, minimal hedging",
    }
    clarity_label = "MEDIUM — balanced confidence"
    for (lo, hi), label in clarity_map.items():
        if lo <= emotional_state.clarity < hi:
            clarity_label = label
            break

    alignment_map = {
        (0.0, 0.4): "LOW — question may be off-topic, stay grounded",
        (0.4, 0.7): "MEDIUM — partially relevant to volleyball/NorCal",
        (0.7, 1.0): "HIGH — fully aligned with NorCal volleyball mission",
    }
    alignment_label = "MEDIUM"
    for (lo, hi), label in alignment_map.items():
        if lo <= emotional_state.alignment < hi:
            alignment_label = label
            break

    return BASE_SYSTEM_PROMPT.format(
        emotional_state_label=emotional_state.label.upper(),
        tone_guidance=routine.get("tone", "friendly and helpful"),
        clarity_level=clarity_label,
        alignment_level=alignment_label,
        routine_injection=routine.get("prompt_injection", "Use your best judgment."),
        memory_context=memory_context or "No prior context loaded.",
    )


# ---------------------------------------------------------------------------
# QUALIA Tool Definitions — what the agent can "do"
# ---------------------------------------------------------------------------

def build_tools(intel_engine: NorCalIntelEngine, memory: QUALIAMemory,
                qpt_engine: QPTEngine) -> list:
    """
    Build the LangChain tool list. Each tool is a function the ReAct agent
    can decide to call when reasoning through a response.

    We use function-based tools rather than class-based ones for simplicity —
    they're easier to audit and test.
    """

    def search_open_gyms(query: str) -> str:
        """
        Search for open gym sessions in NorCal. Query can include city name,
        day of week, or skill level. Returns formatted event list.
        Example inputs: "SF Thursday intermediate", "Oakland all levels"
        """
        # Extract city clues from the query
        city = None
        for c in intel_engine.NORCAL_CITIES:
            if c.lower() in query.lower():
                city = c
                break

        day = None
        for d in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]:
            if d.lower() in query.lower():
                day = d
                break

        events = intel_engine.search_open_gyms(city=city, day_of_week=day, limit=4)
        return intel_engine.format_results_for_ig(events)

    def search_tournaments(query: str) -> str:
        """
        Search for upcoming NorCal volleyball tournaments.
        Query can include city, month, or skill level.
        Example: "Bay Area July beach tournament"
        """
        city = None
        for c in intel_engine.NORCAL_CITIES:
            if c.lower() in query.lower():
                city = c
                break
        events = intel_engine.search_tournaments(city=city, limit=5)
        return intel_engine.format_results_for_ig(events)

    def search_leagues(query: str) -> str:
        """
        Find volleyball leagues and team programs in NorCal.
        Query can include city, gender (coed/men/women), or season.
        Example: "San Francisco coed fall league"
        """
        city = None
        for c in intel_engine.NORCAL_CITIES:
            if c.lower() in query.lower():
                city = c
                break
        gender = None
        if "women" in query.lower():
            gender = "women"
        elif "men" in query.lower():
            gender = "men"

        events = intel_engine.search_leagues(city=city, gender=gender, limit=4)
        return intel_engine.format_results_for_ig(events)

    def recall_memory(query: str) -> str:
        """
        Search QUALIA's memory for prior intel about this topic.
        Use this tool when the query seems to reference something QUALIA may
        have seen before, or to check for previously verified information.
        Example: "open gym SOMA verified", "tournament registration 2025"
        """
        records = memory.recall(query, qpt_engine=qpt_engine, limit=3)
        if not records:
            return "No relevant memories found for this query."
        results = []
        for rec in records:
            results.append(f"[{rec.source_type}] {rec.content[:200]}...")
        return "\n---\n".join(results)

    def generate_ig_caption(query: str) -> str:
        """
        Generate an Instagram-optimized caption for volleyball content.
        Describe the content you want: event recap, skill tip, community shoutout.
        Example: "open gym announcement Tuesday SF intermediate level"
        """
        # This tool is a guided template filler — the agent provides context,
        # we return a structured caption scaffold
        lines = [
            "🏐 **Caption Template Generated**\n",
            "[HOOK — 1 sentence that stops the scroll]",
            "[BODY — 3-5 lines of value: what, where, when, who it's for]",
            "[CTA — 'DM us for details' / 'Link in bio' / 'RSVP below']",
            "\n**Hashtag Block:**",
            "#NorCalVolleyball #BayAreaVolleyball #SFVolleyball",
            "#VolleyballCommunity #OpenGym #IndoorVolleyball",
            "#NorCalVolleyIntel #TheDailyDig",
        ]
        return "\n".join(lines)

    def get_skill_tip(query: str) -> str:
        """
        Retrieve a volleyball skill tip or drill for the requested skill/position.
        Example: "setting footwork", "float serve tips", "libero positioning"
        """
        # Map keywords to canned skill knowledge
        tips = {
            "setting": ("Setting Fundamentals: Hand contact on forehead, elbows bent at 90°. "
                       "Set with legs, not arms. Push up through your fingertips evenly. "
                       "Drill: wall setting 100x daily builds muscle memory faster than anything else."),
            "serve":   ("Float Serve: Toss 12-18 inches above your hitting shoulder, "
                       "contact with heel of hand — no spin. Hold wrist stiff on contact. "
                       "The key is a firm, FLAT contact — think 'slap a wall.'"),
            "dig":     ("Defensive Ready Position: Low athletic stance, weight on balls of feet. "
                       "Platform: thumbs together, arms flat and angled. "
                       "Move to the ball, don't reach. Get your platform under the ball, not behind it."),
            "spike":   ("Approach: 4-step approach for right-handed: left-right-left-right (step-close). "
                       "Jump off both feet, arm swing from hip to ear to contact. "
                       "Hit through the ball, roll wrist over for topspin."),
            "block":   ("Blocking: Read the setter's hands early. Step-slide to position, "
                       "penetrate over the net on contact. Hands wide, wrists turned down. "
                       "Block zones: line or angle — commit early and sell it."),
            "libero":  ("Libero Positioning: Always know where ball #2 is going. "
                       "Start center-back, shift based on setter position. "
                       "Stay disciplined in your zone — chaos comes from over-reaching."),
        }
        query_lower = query.lower()
        for keyword, tip in tips.items():
            if keyword in query_lower:
                return f"🏐 **Skill Tip: {keyword.title()}**\n\n{tip}"
        return ("🏐 General Volleyball Tip: The most overlooked skill in rec/community volleyball "
                "is communication. Calling 'Mine!', 'You!', 'Out!' consistently prevents more "
                "errors than any technical drill. Make communication a habit.")

    # Build the tool list for LangChain
    tools = [
        Tool(name="search_open_gyms",   func=search_open_gyms,   description=search_open_gyms.__doc__),
        Tool(name="search_tournaments", func=search_tournaments, description=search_tournaments.__doc__),
        Tool(name="search_leagues",     func=search_leagues,     description=search_leagues.__doc__),
        Tool(name="recall_memory",      func=recall_memory,      description=recall_memory.__doc__),
        Tool(name="generate_ig_caption",func=generate_ig_caption,description=generate_ig_caption.__doc__),
        Tool(name="get_skill_tip",      func=get_skill_tip,      description=get_skill_tip.__doc__),
    ]
    return tools


# ---------------------------------------------------------------------------
# The QUALIA Agent Orchestrator
# ---------------------------------------------------------------------------

class QUALIAAgent:
    """
    The main orchestrator that ties QPT core, habit engine, memory, tools,
    and LangChain together into one conversational AI system.

    Each call to think() goes through this full pipeline:
      1. QPTEngine.perceive(message)          → update cognitive state
      2. HabitEngine.get_routine(message)     → select response strategy
      3. QUALIAMemory.recall(message)         → load relevant context
      4. build_dynamic_prompt(state, routine) → assemble live system prompt
      5. LangChain Agent.run(prompt + message)→ reason and act with tools
      6. EthicsGuard.check_response(output)   → verify before sending
      7. HabitEngine.record_reward(...)       → learn from the interaction
    """

    def __init__(
        self,
        qpt_engine:    Optional[QPTEngine]       = None,
        habit_engine:  Optional[HabitEngine]     = None,
        memory:        Optional[QUALIAMemory]    = None,
        intel_engine:  Optional[NorCalIntelEngine] = None,
        ethics_guard:  Optional[EthicsGuard]     = None,
        openrouter_api_key: Optional[str]        = None,
        model: str = "anthropic/claude-sonnet-4-5",
    ):
        # Core cognitive systems
        self.qpt     = qpt_engine   or QPTEngine()
        self.habits  = habit_engine or HabitEngine()
        self.memory  = memory       or QUALIAMemory()
        self.intel   = intel_engine or NorCalIntelEngine(use_mock_data=True)
        self.ethics  = ethics_guard or EthicsGuard()
        self.model   = model

        # LangChain agent (optional — falls back to direct LLM call)
        self.agent_executor = None
        if LANGCHAIN_AVAILABLE:
            api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
            if api_key:
                try:
                    self._build_langchain_agent(api_key)
                except Exception as e:
                    logger.error(f"LangChain agent build failed: {e}")

        logger.info(f"QUALIA Agent ready | langchain={self.agent_executor is not None} | model={model}")

    def _build_langchain_agent(self, api_key: str) -> None:
        """Build the LangChain ReAct agent with dynamic system prompt support."""
        llm = ChatOpenAI(
            model=self.model,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=1024,
        )
        tools = build_tools(self.intel, self.memory, self.qpt)

        # ReAct prompt with QPT state placeholder
        react_template = """
{system_prompt}

You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}
"""
        prompt = PromptTemplate(
            template=react_template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            partial_variables={"system_prompt": ""}  # filled dynamically per-call
        )

        agent = create_react_agent(llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=4,
            handle_parsing_errors=True,
        )
        logger.info("LangChain ReAct agent built successfully")

    def think(self, user_message: str, user_id: str = "anonymous") -> dict:
        """
        The main entry point. Process a user message through the full QPT pipeline
        and return a structured response with the reply text and cognitive metadata.

        Returns a dict with:
          "response": str  — the reply to send to the user
          "emotion":  str  — QUALIA's current emotional state label
          "clarity":  float — confidence level 0-1
          "warnings": list — any ethics flags raised
          "routine":  str  — which habit routine was used
        """
        start_time = time.time()

        # --- Step 1: Perceive the incoming message ---
        # This updates QUALIA's cognitive state. The QPT engine "processes"
        # the emotional and contextual signals in the message.
        emotional_state = self.qpt.perceive(user_message, source=f"user:{user_id}")
        logger.info(f"QPT state: {emotional_state.label} | clarity={emotional_state.clarity:.2f}")

        # --- Step 2: Select response routine via habit engine ---
        # The habit engine recognizes cue patterns and maps them to the most
        # successful response strategy from past interactions.
        routine = self.habits.get_routine(user_message)
        logger.info(f"Habit routine: {routine['routine_key']} | automatic={routine['is_automatic']}")

        # --- Step 3: Recall relevant memories ---
        memories = self.memory.recall(user_message, qpt_engine=self.qpt, limit=3)
        memory_context = ""
        if memories:
            memory_context = "Prior intel on this topic:\n"
            for m in memories:
                memory_context += f"- [{m.source_type}] {m.content[:150]}\n"

        # --- Step 4: Build the dynamic system prompt ---
        system_prompt = build_dynamic_prompt(emotional_state, routine, memory_context)

        # --- Step 5: Generate response ---
        if self.agent_executor:
            # Full LangChain ReAct agent path — can use tools
            try:
                # Inject dynamic system prompt
                self.agent_executor.agent.runnable.steps[0].prompt.partial_variables["system_prompt"] = system_prompt
                result = self.agent_executor.invoke({"input": user_message})
                raw_response = result.get("output", "")
            except Exception as e:
                logger.error(f"Agent execution failed: {e}")
                raw_response = self._direct_response(user_message, system_prompt, routine)
        else:
            # Direct response mode — no LangChain, rule-based fallback
            raw_response = self._direct_response(user_message, system_prompt, routine)

        # --- Step 6: Ethics guard check ---
        ethics_result = self.ethics.check_response(raw_response, has_verified_sources=True)
        final_response = ethics_result["modified_text"]
        if not ethics_result["approved"]:
            final_response = ("[QUALIA] I found some content I couldn't verify fully. "
                              "Let me give you what I'm confident about: " + final_response[:500])

        # --- Step 7: Encode this interaction in episodic memory ---
        self.memory.encode(
            content=f"User asked: {user_message[:200]}\nQUALIA responded: {final_response[:200]}",
            source="qualia_interaction",
            memory_type="episodic",
            tags=["interaction", routine["routine_key"]],
            qpt_engine=self.qpt,
        )

        # --- Step 8: Estimate reward and update habit engine ---
        # Simple heuristic: longer responses to volleyball questions = more successful
        reward = 0.6   # baseline
        if emotional_state.alignment > 0.6:
            reward += 0.2   # on-topic = reward
        if ethics_result["warnings"]:
            reward -= 0.1   # warnings = slight penalty
        self.habits.record_reward(user_message, reward)

        elapsed = time.time() - start_time
        logger.info(f"QUALIA thought complete in {elapsed:.2f}s")

        return {
            "response": final_response,
            "emotion": emotional_state.label,
            "clarity": emotional_state.clarity,
            "alignment": emotional_state.alignment,
            "routine": routine["routine_key"],
            "warnings": ethics_result["warnings"],
            "elapsed_ms": round(elapsed * 1000),
        }

    def _direct_response(self, user_message: str, system_prompt: str, routine: dict) -> str:
        """
        Fallback response generator when LangChain isn't available or fails.
        Uses the intel engine and routine strategy to construct a direct answer.
        This is NOT LLM-powered — it's rule-based but still useful for testing.
        """
        msg_lower = user_message.lower()
        routine_key = routine.get("routine_key", "default_helpful")

        # Route to the right intel source based on routine
        if routine_key == "open_gym_lookup":
            city = next((c for c in self.intel.NORCAL_CITIES if c.lower() in msg_lower), None)
            events = self.intel.search_open_gyms(city=city, limit=3)
            return self.intel.format_results_for_ig(events)

        elif routine_key == "tournament_brief":
            events = self.intel.search_tournaments(limit=3)
            return self.intel.format_results_for_ig(events)

        elif routine_key == "league_info":
            events = self.intel.search_leagues(limit=3)
            return self.intel.format_results_for_ig(events)

        elif routine_key == "skill_tip_carousel":
            tips_map = {
                "setting": "set", "serving": "serve", "defense": "dig",
                "attacking": "spike", "blocking": "block",
            }
            for keyword, skill in tips_map.items():
                if keyword in msg_lower:
                    return f"🏐 **Skill Focus: {keyword.title()}**\n\nConnect with a coach or check NCVA clinics for structured training in {keyword}. Great that you're working on it!"

        elif routine_key == "community_welcome":
            return ("👋 Welcome to The Daily Dig / NorCal Volley Intel!\n\n"
                    "We're your go-to source for Bay Area & NorCal volleyball intel: "
                    "open gyms, tournaments, leagues, and skill tips — all for adults 18+.\n\n"
                    "To get started, try asking:\n"
                    "• 'Open gyms in SF this week?'\n"
                    "• 'Upcoming tournaments in the Bay Area?'\n"
                    "• 'Coed leagues in San Jose?'\n\n"
                    "DM anytime — we're here! 🏐 #TheDailyDig")

        # Default: acknowledge the question and provide general guidance
        return (f"Thanks for reaching out to NorCal Volley Intel! 🏐\n\n"
                f"I'm working on getting you the best intel for your question. "
                f"For the most current info, also check:\n"
                f"• ncva.com for official NorCal volleyball events\n"
                f"• sfrecpark.org for San Francisco programs\n"
                f"• meetup.com groups for community-organized sessions\n\n"
                f"DM us with your city and level and we'll dig up specific options! 💪")

    def get_status(self) -> dict:
        """Return a full status report of all QUALIA subsystems."""
        return {
            "version": "QUALIA v1.0",
            "cognitive_state": self.qpt.introspect().__dict__,
            "memory_stats": self.memory.get_stats(),
            "habit_stats": self.habits.get_stats(),
            "langchain_active": self.agent_executor is not None,
            "model": self.model,
        }
