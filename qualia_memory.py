"""
QUALIA Memory — Episodic + Semantic Memory with Quaternion Resonance
======================================================================
QUALIA's memory system uses two types of storage, mirroring the dual-memory
model in cognitive neuroscience:

  1. EPISODIC MEMORY — "What happened?" Records of actual interactions,
     conversations, and intel lookup results. Tagged with the quaternion state
     at the time of encoding (mood-congruent memory: we recall things better
     when in a similar emotional state to when we learned them).

  2. SEMANTIC MEMORY — "What do I know?" Structured facts about volleyball
     venues, leagues, tournaments, and community knowledge that don't expire
     quickly and don't need emotional context.

The "resonance retrieval" system means QUALIA prioritizes memories that were
formed in a cognitive state similar to its current state — just like how
a happy person more easily recalls happy memories. This is the QPT
(Quaternion Process Theory) applied to memory architecture.

Storage backend: LanceDB (embedded vector DB, no separate server needed).
For production deployment on Docker, the LanceDB data directory is mounted
as a persistent volume.
"""

import os
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

# LanceDB is an embedded vector store — no server required.
# Install: pip install lancedb sentence-transformers
try:
    import lancedb
    import pyarrow as pa
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False
    lancedb = None

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

from core.qualia_core import Quaternion, QPTEngine

logger = logging.getLogger("qualia.memory")


# ---------------------------------------------------------------------------
# Memory Record Schema
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    """
    A single entry in QUALIA's memory.
    Every memory carries its quaternion state fingerprint — this is how
    resonance retrieval works: similarity in state → more likely to be recalled.
    """
    memory_id: str
    memory_type: str       # "episodic" or "semantic"
    content: str           # The actual text content of the memory
    source: str            # Where this came from
    source_url: Optional[str]
    tags: List[str]        # Volleyball-specific tags: ["open_gym", "SF", "coed"]
    created_at: float      # Unix timestamp

    # Quaternion state at time of encoding — the "emotional fingerprint" of this memory
    q_w: float = 0.9
    q_x: float = 0.0
    q_y: float = 0.3
    q_z: float = 0.1

    # Vector embedding of content (for semantic similarity search)
    # Stored as a list of floats in LanceDB
    embedding: Optional[List[float]] = None

    # Retrieval stats
    retrieval_count: int = 0
    last_retrieved: Optional[float] = None
    relevance_score: float = 0.0   # filled in at retrieval time

    def get_quaternion(self) -> Quaternion:
        """Return the quaternion state at the time this memory was encoded."""
        return Quaternion(w=self.q_w, x=self.q_x, y=self.q_y, z=self.q_z)

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "source": self.source,
            "source_url": self.source_url or "",
            "tags": json.dumps(self.tags),
            "created_at": self.created_at,
            "q_w": self.q_w,
            "q_x": self.q_x,
            "q_y": self.q_y,
            "q_z": self.q_z,
            "retrieval_count": self.retrieval_count,
            "last_retrieved": self.last_retrieved or 0.0,
        }


# ---------------------------------------------------------------------------
# Fallback In-Memory Store (when LanceDB is unavailable)
# ---------------------------------------------------------------------------

class InMemoryStore:
    """
    A simple list-based fallback for when LanceDB isn't installed.
    Not suitable for production (no vector search, no persistence) but
    lets the system run in development without installing heavy dependencies.
    """
    def __init__(self):
        self.records: List[MemoryRecord] = []
        logger.warning("Using InMemoryStore fallback — install lancedb for production!")

    def add(self, record: MemoryRecord) -> None:
        self.records.append(record)

    def search_by_text(self, query: str, limit: int = 5) -> List[MemoryRecord]:
        """Basic substring search as fallback for vector similarity."""
        query_lower = query.lower()
        results = [r for r in self.records if query_lower in r.content.lower()]
        return sorted(results, key=lambda r: r.created_at, reverse=True)[:limit]

    def get_all(self, memory_type: Optional[str] = None) -> List[MemoryRecord]:
        if memory_type:
            return [r for r in self.records if r.memory_type == memory_type]
        return list(self.records)


# ---------------------------------------------------------------------------
# QUALIA Memory Manager
# ---------------------------------------------------------------------------

class QUALIAMemory:
    """
    The main memory interface for QUALIA.

    Key behaviors:
    - encode(content, ...) → stores a new memory with the current QPT state embedded
    - recall(query, engine) → retrieves memories, ranking by both semantic similarity
      AND quaternion resonance (memories from similar cognitive states score higher)
    - forget(threshold) → removes very old, low-utility memories (cognitive cleanup)
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions
    DB_DIR = "/data/qualia_memory"         # Override with env var QUALIA_DB_DIR

    def __init__(self, db_dir: Optional[str] = None):
        self.db_dir = db_dir or os.environ.get("QUALIA_DB_DIR", self.DB_DIR)
        os.makedirs(self.db_dir, exist_ok=True)

        # Initialize embedding model
        self.encoder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer(self.EMBEDDING_MODEL)
                logger.info(f"Embedding model loaded: {self.EMBEDDING_MODEL}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")

        # Initialize LanceDB tables
        self.db = None
        self.episodic_table = None
        self.semantic_table = None

        if LANCEDB_AVAILABLE and lancedb:
            try:
                self.db = lancedb.connect(self.db_dir)
                self._init_tables()
                logger.info(f"LanceDB connected at {self.db_dir}")
            except Exception as e:
                logger.error(f"LanceDB init failed: {e}")

        # Fall back to in-memory if LanceDB unavailable
        if not self.db:
            self._fallback = InMemoryStore()
            logger.warning("Falling back to InMemoryStore")
        else:
            self._fallback = None

    def _init_tables(self) -> None:
        """Create LanceDB tables with the correct schema if they don't exist."""
        # Define schema using PyArrow
        schema = pa.schema([
            pa.field("memory_id",      pa.string()),
            pa.field("memory_type",    pa.string()),
            pa.field("content",        pa.string()),
            pa.field("source",         pa.string()),
            pa.field("source_url",     pa.string()),
            pa.field("tags",           pa.string()),   # JSON encoded list
            pa.field("created_at",     pa.float64()),
            pa.field("q_w",            pa.float32()),
            pa.field("q_x",            pa.float32()),
            pa.field("q_y",            pa.float32()),
            pa.field("q_z",            pa.float32()),
            pa.field("retrieval_count",pa.int32()),
            pa.field("last_retrieved", pa.float64()),
            # Vector embedding — 384 dims for all-MiniLM-L6-v2
            pa.field("vector",         pa.list_(pa.float32(), 384)),
        ])

        try:
            self.episodic_table = self.db.open_table("episodic")
        except Exception:
            # Table doesn't exist yet — create it
            self.episodic_table = self.db.create_table("episodic", schema=schema)
            logger.info("Created episodic memory table")

        try:
            self.semantic_table = self.db.open_table("semantic")
        except Exception:
            self.semantic_table = self.db.create_table("semantic", schema=schema)
            logger.info("Created semantic memory table")

    def _embed(self, text: str) -> Optional[List[float]]:
        """Generate a vector embedding for text."""
        if not self.encoder:
            return None
        try:
            embedding = self.encoder.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def encode(
        self,
        content: str,
        source: str,
        memory_type: str = "episodic",
        tags: Optional[List[str]] = None,
        source_url: Optional[str] = None,
        qpt_engine: Optional[QPTEngine] = None,
    ) -> MemoryRecord:
        """
        Store a new memory. The QPT engine's current state is captured and
        embedded into the memory as its 'emotional fingerprint.'

        This is what enables mood-congruent retrieval later — you'll find this
        memory more easily when QUALIA is in a similar cognitive state.
        """
        # Capture the current QPT state
        q = qpt_engine.state if qpt_engine else Quaternion()

        record = MemoryRecord(
            memory_id=f"mem_{int(time.time() * 1000)}",
            memory_type=memory_type,
            content=content,
            source=source,
            source_url=source_url,
            tags=tags or [],
            created_at=time.time(),
            q_w=q.w, q_x=q.x, q_y=q.y, q_z=q.z,
        )

        # Generate embedding for semantic search
        embedding = self._embed(content)

        # Store in LanceDB or fallback
        if self.db and embedding:
            table = self.episodic_table if memory_type == "episodic" else self.semantic_table
            row = record.to_dict()
            row["vector"] = embedding
            try:
                table.add([row])
                logger.debug(f"Memory encoded: {record.memory_id} [{memory_type}]")
            except Exception as e:
                logger.error(f"LanceDB write failed: {e}")
                if self._fallback:
                    self._fallback.add(record)
        elif self._fallback:
            self._fallback.add(record)

        return record

    def recall(
        self,
        query: str,
        qpt_engine: Optional[QPTEngine] = None,
        memory_type: Optional[str] = None,
        limit: int = 5,
        resonance_boost: float = 0.3,
    ) -> List[MemoryRecord]:
        """
        Retrieve the most relevant memories for a query.

        The ranking formula is:
          final_score = (1 - resonance_boost) × semantic_similarity
                      +  resonance_boost      × quaternion_resonance

        This means that when QUALIA is in an 'enthusiastic' state, it will
        slightly prefer memories that were encoded while also enthusiastic —
        making it more likely to recall successful past volleyball intel
        interactions rather than cautious/uncertain ones.

        Adjust resonance_boost (0.0 = pure semantic, 1.0 = pure QPT resonance)
        to tune how much cognitive state influences retrieval.
        """
        results = []

        if self._fallback:
            # Simple text search fallback
            results = self._fallback.search_by_text(query, limit=limit * 2)
        elif self.db:
            embedding = self._embed(query)
            if embedding:
                table = (
                    self.episodic_table if memory_type == "episodic"
                    else self.semantic_table if memory_type == "semantic"
                    else self.episodic_table   # default to episodic
                )
                try:
                    rows = table.search(embedding).limit(limit * 2).to_list()
                    for row in rows:
                        record = MemoryRecord(
                            memory_id=row["memory_id"],
                            memory_type=row["memory_type"],
                            content=row["content"],
                            source=row["source"],
                            source_url=row.get("source_url"),
                            tags=json.loads(row.get("tags", "[]")),
                            created_at=row["created_at"],
                            q_w=row["q_w"], q_x=row["q_x"],
                            q_y=row["q_y"], q_z=row["q_z"],
                            retrieval_count=row.get("retrieval_count", 0),
                        )
                        results.append(record)
                except Exception as e:
                    logger.error(f"LanceDB recall failed: {e}")

        # Apply QPT resonance re-ranking
        if qpt_engine and results:
            current_q = qpt_engine.state
            for record in results:
                memory_q = record.get_quaternion()
                resonance = (current_q.dot(memory_q) + 1) / 2    # normalize 0→1

                # Base semantic rank (earlier results = higher base score)
                rank_pos = results.index(record)
                semantic_score = 1.0 - (rank_pos / len(results))

                record.relevance_score = (
                    (1 - resonance_boost) * semantic_score +
                    resonance_boost * resonance
                )

            # Re-sort by combined score
            results.sort(key=lambda r: r.relevance_score, reverse=True)

        return results[:limit]

    def store_volleyball_intel(
        self,
        intel_text: str,
        source: str,
        intel_type: str,   # "open_gym", "tournament", "league", "venue", "skill_tip"
        location: Optional[str] = None,
        qpt_engine: Optional[QPTEngine] = None,
    ) -> MemoryRecord:
        """
        Specialized encode for volleyball community intelligence.
        Automatically tags by type and location for filtered recall later.
        """
        tags = [intel_type]
        if location:
            tags.append(location.lower().replace(" ", "_"))
        tags.append("norcal_volley_intel")

        return self.encode(
            content=intel_text,
            source=source,
            memory_type="semantic",
            tags=tags,
            qpt_engine=qpt_engine,
        )

    def get_stats(self) -> dict:
        """Return memory system statistics for monitoring."""
        if self._fallback:
            records = self._fallback.get_all()
            return {
                "backend": "InMemoryStore",
                "total_records": len(records),
                "episodic_count": len([r for r in records if r.memory_type == "episodic"]),
                "semantic_count": len([r for r in records if r.memory_type == "semantic"]),
            }
        try:
            return {
                "backend": "LanceDB",
                "episodic_count": self.episodic_table.count_rows() if self.episodic_table else 0,
                "semantic_count": self.semantic_table.count_rows() if self.semantic_table else 0,
                "db_dir": self.db_dir,
            }
        except Exception:
            return {"backend": "LanceDB", "error": "stats unavailable"}
