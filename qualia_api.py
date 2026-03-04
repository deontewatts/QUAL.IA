"""
QUALIA FastAPI Server
=====================
The HTTP interface that Instagram webhooks, n8n/Make automations, and
Bubble/Wix frontends all talk to. Every request passes through the full
QPT pipeline (perceive → habit → memory → agent → ethics → respond).

Key endpoints:
  POST /ask              — main conversational endpoint (Instagram DMs, web chat)
  POST /webhook/instagram — receives Instagram webhook events from Meta
  GET  /status           — QUALIA system health + cognitive state report
  GET  /intel/open-gyms  — direct open gym search (for n8n/Make automations)
  GET  /intel/tournaments — tournament listings
  POST /reward           — submit reward signal after user interaction
  POST /encode-memory    — manually store a piece of verified intel

CORS, rate limiting, and structured error handling are all included.
The server is stateful (one global QPTEngine per process) — in production
you'd use Redis to share state across multiple containers.
"""

import os
import time
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.qualia_core   import QPTEngine
from core.habit_engine  import HabitEngine
from core.ethics_guard  import EthicsGuard
from memory.qualia_memory import QUALIAMemory
from volley.norcal_intel  import NorCalIntelEngine
from agents.qualia_agent  import QUALIAAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("qualia.api")


# ---------------------------------------------------------------------------
# Global QUALIA agent instance — persists across requests
# This is the "continuous consciousness" layer: the QPT engine accumulates
# state across all interactions, just like a person carries their mood
# through an entire day of conversations.
# ---------------------------------------------------------------------------
_agent: Optional[QUALIAAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize QUALIA subsystems on startup, persist them across all requests."""
    global _agent
    logger.info("🧠 QUALIA is waking up...")

    _agent = QUALIAAgent(
        qpt_engine    = QPTEngine(),
        habit_engine  = HabitEngine(),
        memory        = QUALIAMemory(db_dir=os.environ.get("QUALIA_DB_DIR", "/tmp/qualia_memory")),
        intel_engine  = NorCalIntelEngine(use_mock_data=os.environ.get("MOCK_DATA", "true").lower() == "true"),
        ethics_guard  = EthicsGuard(),
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY"),
        model         = os.environ.get("QUALIA_MODEL", "anthropic/claude-sonnet-4-5"),
    )

    logger.info("✅ QUALIA is conscious and ready.")
    yield
    # Shutdown: serialize state for restart continuity
    logger.info("😴 QUALIA is sleeping — serializing cognitive state...")
    # TODO: persist _agent.qpt.serialize() and _agent.habits.serialize() to Redis


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="QUALIA — NorCal Volley Intel AI",
    description="Quaternion-Unified Adaptive Learning Intelligence Agent for The Daily Dig",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this to your Bubble/Wix domain in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models — typed contracts for every endpoint
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    user_id: str = Field(default="anonymous", description="Stable user identifier (IG user ID, Bubble user ID, etc.)")
    message: str = Field(..., min_length=1, max_length=2000, description="The user's message")
    context: Optional[str] = Field(None, description="Optional extra context (e.g., 'user is a setter', 'Bay Area')")

class AskResponse(BaseModel):
    response: str
    emotion: str              # QUALIA's current cognitive state label
    clarity: float            # confidence level 0.0 – 1.0
    alignment: float          # how volleyball-relevant the topic was
    routine: str              # which habit routine was used
    warnings: list[str]       # any ethics flags raised
    elapsed_ms: int

class RewardRequest(BaseModel):
    user_id: str
    original_message: str
    reward_signal: float = Field(..., ge=-1.0, le=1.0,
        description="Reward: +1.0=very positive, 0=neutral, -1.0=negative")
    feedback_note: Optional[str] = None

class IntelStoreRequest(BaseModel):
    content: str
    source: str
    source_url: Optional[str] = None
    intel_type: str = "open_gym"   # "open_gym", "tournament", "league", "skill_tip"
    location: Optional[str] = None

class InstagramWebhookPayload(BaseModel):
    """
    Simplified Instagram webhook payload shape.
    Real Instagram webhooks are more complex — this handles the essentials.
    See Meta's webhook docs for full schema.
    """
    object: str                    # "instagram"
    entry: list[dict]


# ---------------------------------------------------------------------------
# Request Rate Limiting — simple in-memory token bucket
# Replace with Redis-backed rate limiter in production
# ---------------------------------------------------------------------------

_request_log: dict[str, list[float]] = {}

def is_rate_limited(user_id: str, max_requests: int = 20, window_seconds: float = 60.0) -> bool:
    now = time.time()
    cutoff = now - window_seconds
    _request_log.setdefault(user_id, [])
    _request_log[user_id] = [t for t in _request_log[user_id] if t > cutoff]
    if len(_request_log[user_id]) >= max_requests:
        return True
    _request_log[user_id].append(now)
    return False


# ---------------------------------------------------------------------------
# Core Endpoints
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse, summary="Ask QUALIA a volleyball question")
async def ask(request: AskRequest) -> AskResponse:
    """
    The primary endpoint — powers Instagram DM auto-replies, web chat widgets,
    and any automation (n8n, Make, Zapier) that wants a natural language response.

    The request flows through the complete QPT pipeline:
    perception → habit selection → memory recall → agent reasoning → ethics check.
    """
    if not _agent:
        raise HTTPException(status_code=503, detail="QUALIA is not yet initialized")

    if is_rate_limited(request.user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded — please wait 60 seconds")

    # Append optional context to the message
    full_message = request.message
    if request.context:
        full_message = f"{request.message}\n[Context: {request.context}]"

    result = _agent.think(full_message, user_id=request.user_id)
    return AskResponse(**result)


@app.post("/webhook/instagram", summary="Receive Instagram DM webhook events")
async def instagram_webhook(
    payload: InstagramWebhookPayload,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """
    Meta/Instagram sends DM events here via webhook.
    We process them asynchronously so the webhook returns 200 immediately
    (Meta will retry if it doesn't get a fast 200), then QUALIA thinks
    in the background and posts the reply via the Instagram Messaging API.

    Setup: in Meta for Developers → Webhooks → Instagram → messages field.
    Your webhook URL: https://your-domain.com/webhook/instagram
    """
    if payload.object != "instagram":
        return JSONResponse(content={"status": "ignored"})

    # Process each webhook entry asynchronously
    for entry in payload.entry:
        for messaging_event in entry.get("messaging", []):
            sender_id  = messaging_event.get("sender", {}).get("id")
            message_text = messaging_event.get("message", {}).get("text")

            if sender_id and message_text:
                background_tasks.add_task(
                    _process_ig_dm,
                    sender_id=sender_id,
                    message_text=message_text,
                )

    return JSONResponse(content={"status": "received"})


async def _process_ig_dm(sender_id: str, message_text: str) -> None:
    """
    Background task: process an Instagram DM and send a reply.
    In production, after _agent.think() you'd call the Instagram
    Send Message API with the response. That API call requires your
    Page Access Token from Meta for Developers.
    """
    if not _agent:
        return

    result = _agent.think(message_text, user_id=f"ig:{sender_id}")

    # TODO: Send reply via Instagram Messaging API
    # POST https://graph.facebook.com/v18.0/me/messages
    # Headers: {"Authorization": f"Bearer {os.environ['IG_PAGE_ACCESS_TOKEN']}"}
    # Body: {"recipient": {"id": sender_id}, "message": {"text": result["response"]}}

    logger.info(f"IG DM processed for {sender_id[:6]}... | "
                f"emotion={result['emotion']} | routine={result['routine']}")


@app.post("/reward", summary="Submit a reward signal to improve QUALIA's habits")
async def submit_reward(request: RewardRequest) -> dict:
    """
    Call this endpoint after a user interaction to tell QUALIA how well it did.
    This is what drives the Habit Engine's reinforcement learning loop.

    Integration pattern: after your Bubble/Wix frontend shows QUALIA's response,
    add a 👍/👎 button. Thumbs up → reward_signal=0.9, thumbs down → reward_signal=-0.3.
    You can also infer reward from behavior: if user asked a follow-up question, +0.5.
    """
    if not _agent:
        raise HTTPException(status_code=503, detail="QUALIA not initialized")

    _agent.habits.record_reward(request.original_message, request.reward_signal)
    _agent.qpt.perceive(
        f"User feedback: {request.feedback_note or ('positive' if request.reward_signal > 0 else 'negative')}",
        source="reward_signal"
    )

    return {
        "status": "reward_recorded",
        "reward": request.reward_signal,
        "habit_stats": _agent.habits.get_stats(),
    }


@app.post("/encode-memory", summary="Manually store verified volleyball intel")
async def encode_memory(request: IntelStoreRequest) -> dict:
    """
    Manually push verified intel into QUALIA's semantic memory.
    Use this for bulk onboarding of known-good event data, or when a community
    admin verifies information and wants it in the system immediately.
    The QPT engine's current state is captured as the memory's 'mood fingerprint.'
    """
    if not _agent:
        raise HTTPException(status_code=503, detail="QUALIA not initialized")

    record = _agent.memory.store_volleyball_intel(
        intel_text=request.content,
        source=request.source,
        intel_type=request.intel_type,
        location=request.location,
        qpt_engine=_agent.qpt,
    )

    return {
        "status": "memory_encoded",
        "memory_id": record.memory_id,
        "tags": record.tags,
    }


# ---------------------------------------------------------------------------
# Intel Direct-Access Endpoints (for n8n / Make automations)
# ---------------------------------------------------------------------------

@app.get("/intel/open-gyms", summary="Direct open gym search — for automations")
async def get_open_gyms(
    city: Optional[str] = None,
    day: Optional[str] = None,
    level: Optional[str] = None,
    limit: int = 5,
) -> dict:
    """
    Structured JSON endpoint for direct event data access.
    Bypasses the LLM layer — returns raw structured data for automations.
    Perfect for n8n nodes that build Airtable records or Sheets rows.
    """
    if not _agent:
        raise HTTPException(status_code=503, detail="QUALIA not initialized")

    events = _agent.intel.search_open_gyms(city=city, day_of_week=day, skill_level=level, limit=limit)
    return {
        "count": len(events),
        "events": [e.to_dict() for e in events],
        "ig_caption": _agent.intel.format_results_for_ig(events),
    }


@app.get("/intel/tournaments", summary="Direct tournament search — for automations")
async def get_tournaments(
    city: Optional[str] = None,
    month: Optional[str] = None,
    limit: int = 5,
) -> dict:
    if not _agent:
        raise HTTPException(status_code=503, detail="QUALIA not initialized")

    events = _agent.intel.search_tournaments(city=city, month=month, limit=limit)
    return {
        "count": len(events),
        "events": [e.to_dict() for e in events],
        "ig_caption": _agent.intel.format_results_for_ig(events),
    }


# ---------------------------------------------------------------------------
# System Status & Health
# ---------------------------------------------------------------------------

@app.get("/status", summary="QUALIA system health and cognitive state report")
async def status() -> dict:
    """
    Returns full system status including QUALIA's current cognitive state,
    memory stats, habit table summary, and service health.
    Pin this to a dashboard (PostHog, Sentry, Grafana) for ongoing monitoring.
    """
    if not _agent:
        return {"status": "initializing", "message": "QUALIA is waking up..."}

    return {
        "status": "conscious",
        **_agent.get_status(),
        "uptime_check": "ok",
        "timestamp": time.time(),
    }


@app.get("/health", summary="Simple health check for load balancers")
async def health() -> dict:
    return {"status": "ok", "service": "QUALIA"}


@app.get("/", summary="QUALIA API root")
async def root() -> dict:
    return {
        "name": "QUALIA — NorCal Volley Intel AI",
        "tagline": "Quaternion-Unified Adaptive Learning Intelligence Agent",
        "brand": "The Daily Dig / NorCal Volley Intel",
        "docs": "/docs",
        "status": "/status",
    }


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "QUALIA encountered an unexpected error.",
            "detail": str(exc),
            "path": str(request.url),
        },
    )


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "qualia_api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("ENV", "production") == "development",
        log_level="info",
    )
