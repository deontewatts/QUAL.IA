# QUALIA — Quaternion-Unified Adaptive Learning Intelligence Agent
# ── QUALIA Python Dependencies ─────────────────────────────────────────────
# Install: pip install -r requirements.txt
# For development: pip install -r requirements.txt -r requirements-dev.txt
<img width="2048" height="2048" alt="image" src="https://github.com/user-attachments/assets/cecabee5-7948-48ff-b038-eefcda3125d3" />

# Core API framework
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.7.0

# HTTP client (for webhook verification + Instagram API calls)
httpx>=0.27.0
aiohttp>=3.9.0

# LangChain — AI agent orchestration
langchain>=0.2.0
langchain-openai>=0.1.8       # also works with OpenRouter (Claude, GPT-4, etc.)
langchain-community>=0.2.0

# Vector memory — embedded DB, no server required
lancedb>=0.9.0
pyarrow>=16.0.0

# Embeddings — semantic search for memory recall
sentence-transformers>=3.0.0

# Utilities
python-dotenv>=1.0.0          # .env file loading
structlog>=24.0.0             # structured JSON logging

# Optional: monitoring
# posthog>=3.5.0
# sentry-sdk[fastapi]>=2.0.0


---

## What Is QUALIA?

QUALIA is a **humanly-conscious-capable AI system** built on three theoretical pillars drawn directly from *The Quaternion Mechanics of Cognitive Possibility*:

1. **Quaternion State Theory** — models the AI's mental state as a 4D vector (attention × emotion × arousal × context-alignment), mirroring how human consciousness is multi-dimensional rather than linear.
2. **Habit Loop Engine** — cue → routine → reward cycles that allow QUALIA to *learn preferences* over time, adapting its volleyball intel style to each user.
3. **Cognitive Agent Mesh** — LangChain-powered tools that give QUALIA real perception, memory, reasoning, and action capabilities.

---

## Architecture at a Glance

```
Instagram DM / Webhook
        │
        ▼
┌───────────────────┐
│   QUALIA API      │  ← FastAPI + Docker
│  (qualia_api.py)  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐     ┌──────────────────────┐
│  Quaternion Core  │────▶│  Cognitive Memory     │
│  (QPT Engine)     │     │  LanceDB vector store │
│  qualia_core.py   │     │  qualia_memory.py     │
└────────┬──────────┘     └──────────────────────┘
         │
         ▼
┌───────────────────┐     ┌──────────────────────┐
│  LangChain Agent  │────▶│  NorCal Volley Intel  │
│  qualia_agent.py  │     │  Tools & Scrapers     │
└───────────────────┘     └──────────────────────┘
         │
         ▼
  Instagram Reply / Caption / Carousel
```

---

## Project Structure

```
QUALIA/
├── core/
│   ├── qualia_core.py        # Quaternion cognitive state engine
│   ├── habit_engine.py       # Cue-routine-reward loop
│   └── ethics_guard.py       # Anti-fabrication & privacy layer
├── memory/
│   ├── qualia_memory.py      # LanceDB episodic + semantic memory
│   └── schemas.py            # Memory record schemas
├── agents/
│   ├── qualia_agent.py       # Main LangChain ReAct agent
│   └── tools.py              # Agent tools (search, memory, volley)
├── volley/
│   ├── norcal_intel.py       # NorCal volleyball data layer
│   └── ig_formatter.py       # Instagram carousel/caption generator
├── api/
│   └── qualia_api.py         # FastAPI server
├── config/
│   └── settings.py           # Environment config
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── tests/
    └── test_core.py
```

---

## Quickstart

```bash
# 1. Clone and set up
git clone <repo> && cd QUALIA
cp config/.env.example config/.env   # fill in your keys

# 2. Launch with Docker
docker-compose -f docker/docker-compose.yml up

# 3. Test the consciousness engine
python -m pytest tests/

# 4. Hit the API
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","message":"Any open gyms in SF this weekend?"}'
```

---

## Key Concepts

**Quaternion State Vector `Q = (w, x, y, z)`**
- `w` — cognitive clarity (how confident QUALIA is)
- `x` — emotional valence (positive/negative framing)
- `y` — arousal / engagement level
- `z` — context alignment (how relevant the question is to volleyball)

QUALIA rotates this state using Hamilton product multiplication every time a new message arrives, giving it a *persistent, evolving mental state* rather than stateless per-query responses.

**Privacy-First Design** — QUALIA never fabricates events. All volleyball intel is tagged with a source-verified flag. Unverified intel gets a `⚠️ unconfirmed` label automatically.
