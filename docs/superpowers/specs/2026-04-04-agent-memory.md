# Agent Memory Spec

**Date:** 2026-04-04  
**Status:** Draft  
**Depends on:** extended-game-state-schema, reward-function, spatial-perimeter-system, policy-training-pipeline

---

## Overview

Each training episode starts fresh. GRPO encodes learning implicitly into model weights over
many episodes, but the agent has no way to explicitly recall "last time oxygen dropped at
cycle 8, I hadn't built a backup deoxidizer." Agent Memory adds an explicit retrieval layer
via ChromaDB, allowing the agent to recover relevant past experiences at decision time.

ChromaDB is already running on the Linux desktop (10.0.0.10). No new infrastructure required.

---

## What Memory Solves

| Problem | Without Memory | With Memory |
|---------|---------------|-------------|
| Repeated mistakes | Agent relearns same failure every episode | Causal warning retrieved: "O2 dropped cycle 7 — build backup earlier" |
| Spatial knowledge | Agent has no recall of what worked where | "SPOM at (132,195) worked on this seed — water depth sufficient" |
| Strategy carry-over | Phase wins only encode in weights over many episodes | Explicit strategy record injected into next episode prompt |
| Credit assignment | Hard to connect early decisions to late outcomes | Episode summary links early actions to final reward |

---

## Memory Types

Four types of memory events are written to the store. Nothing is written on every tick —
only at meaningful game events to keep the store clean and high signal-to-noise.

### 1. Perimeter Outcome
**Written:** when a perimeter completes (win, partial, or fail)

Records what was built, where, what reward it produced, and any notable friction during
construction (missing prerequisites, dupe routing failures, overpressure stalls, etc.).

### 2. Causal Warning
**Written:** on dupe death, critical alert (oxygen depleted, food exhausted, mass stress)

Records what happened, what cycle, what the state was leading up to it, and what should
have been done differently. These are the highest-value memories for early-phase training.

### 3. Strategy Success
**Written:** when a phase win condition is met

Records the sequence of high-level decisions that led to the win: what was built first,
what was researched, when the first dupe was printed. Generalizes across episodes.

### 4. Episode Summary
**Written:** at episode end (win or fail)

High-level outcome record: phase, final reward, dupe count, key milestones, cause of
failure (if fail). Used for cross-episode trend retrieval.

---

## Memory Schema

All records stored in ChromaDB collection `oni_agent_memory`.

```python
{
  "id": "{type}_{episode_id}_{cycle}_{slug}",
  # e.g. "perimeter_ep042_cycle009_spom-basic"

  "document": str,
  # Natural language description — used for semantic similarity retrieval.
  # Written by MemoryStore.format_document() for each memory type.
  # Target: 80–150 tokens. Concise, factual, agent-readable.

  "metadata": {
    "type": str,          # "perimeter_outcome" | "causal_warning" | "strategy" | "episode_summary"
    "seed": str,          # "V-SNDST-C-1644640403-0-0-0" — enables seed-specific filtering
    "episode_id": str,    # "ep_042"
    "cycle": int,         # game cycle when event occurred
    "phase": int,         # training phase (0–4)
    "outcome": str,       # "win" | "partial" | "fail" | "death" | "alert" (type-dependent)
    "reward": float,      # reward associated with this event (perimeter reward, episode total, etc.)
    "blueprint_id": str,  # perimeter_outcome only — e.g. "spom-basic"
    "anchor_x": int,      # perimeter_outcome only — world coordinates of perimeter anchor
    "anchor_y": int,
  }
}
```

### Example Records

**Perimeter outcome (win):**
```json
{
  "id": "perimeter_ep042_cycle009_spom-basic",
  "document": "Placed basic SPOM at anchor (132, 195) starting cycle 6. Electrolyzer operational by cycle 9. Water pump needed repositioning at cycle 7 — water depth only 2 tiles at initial position, moved 3 tiles east. Oxygen stable at 1.8kg/s thereafter. Perimeter reward: +4.2.",
  "metadata": {
    "type": "perimeter_outcome", "seed": "V-SNDST-C-1644640403-0-0-0",
    "episode_id": "ep_042", "cycle": 9, "phase": 1,
    "outcome": "win", "reward": 4.2,
    "blueprint_id": "spom-basic", "anchor_x": 132, "anchor_y": 195
  }
}
```

**Causal warning (dupe death):**
```json
{
  "id": "causal_ep031_cycle007_o2-death",
  "document": "Dupe Lindsay died cycle 7. Cause: oxygen exhausted. Only one algae deoxidizer built; it stalled at cycle 5 due to algae depletion. No backup oxygen source. Stress had been climbing since cycle 4. Should have built second deoxidizer by cycle 3 or started electrolyzer research immediately.",
  "metadata": {
    "type": "causal_warning", "seed": "V-SNDST-C-1644640403-0-0-0",
    "episode_id": "ep_031", "cycle": 7, "phase": 0,
    "outcome": "death", "reward": -10.0
  }
}
```

**Strategy success:**
```json
{
  "id": "strategy_ep055_cycle050_phase1-win",
  "document": "Phase 1 win (survive 50 cycles). Key decisions: researched Basic Farming cycle 4, built mealwood farm cycle 6 (food stable by cycle 10), SPOM operational cycle 12, printed 2nd dupe cycle 18. Stress never exceeded 0.4. Avoid delaying food research past cycle 5.",
  "metadata": {
    "type": "strategy", "seed": "V-SNDST-C-1644640403-0-0-0",
    "episode_id": "ep_055", "cycle": 50, "phase": 1,
    "outcome": "win", "reward": 84.3
  }
}
```

---

## Retrieval: Seed-Priority Hybrid

Retrieval uses a two-pass strategy: seed-specific memories are retrieved first (spatial and
strategy records are map-dependent), then cross-seed memories fill remaining slots for
causal warnings and general strategies.

```python
def retrieve(
    self,
    state_summary: str,
    n: int = 3,
    current_seed: str = CANONICAL_SEED,
) -> list[MemoryRecord]:
    """
    Retrieve top-n relevant memories for injection into state prompt.

    Pass 1: seed-specific (n results, filter seed == current_seed)
    Pass 2: cross-seed causal warnings (fill remaining slots if pass 1 < n)
    Dedup by id, return up to n total.
    """
    seed_results = self._collection.query(
        query_texts=[state_summary],
        n_results=n,
        where={"seed": current_seed},
    )
    if len(seed_results) >= n:
        return seed_results[:n]

    remaining = n - len(seed_results)
    cross_seed_results = self._collection.query(
        query_texts=[state_summary],
        n_results=remaining,
        where={"type": {"$in": ["causal_warning", "strategy"]}},
    )
    return deduplicate(seed_results + cross_seed_results)[:n]
```

---

## Prompt Injection

Memories are injected as a `## Relevant Past Experience` section in the state prompt,
immediately before the action decision section. Format is concise bullet points.

```
## Relevant Past Experience
- [ep_042, cycle 9] SPOM placed at (132,195). Water pump needed repositioning
  3 tiles east — shallow water at initial position. Operational cycle 9. ✓
- [ep_031, cycle 7] Dupe died from O2 depletion. Single deoxidizer stalled
  cycle 5. Build backup O2 source before cycle 3.
- [ep_055, cycle 50] Phase 1 win: Basic Farming research by cycle 4 kept food
  stable. Don't delay food research past cycle 5.
```

**Token budget:** 3 memories × ~100 tokens = ~300 tokens per tick.  
Combined with state payload (~800–1200 tokens) → ~1100–1500 total. Within budget.

Retrieval fires once per agent tick using the current state summary as the query text.
The state summary (cycle, resources, active alerts, current task) provides sufficient
semantic signal for relevant retrieval without sending the full state payload to ChromaDB.

---

## MemoryStore Implementation

```python
# src/agent/memory.py

class MemoryStore:
    """Wraps ChromaDB for agent memory read/write."""

    COLLECTION_NAME = "oni_agent_memory"
    CHROMA_HOST = "10.0.0.10"
    CHROMA_PORT = 8300  # confirmed: mapped 0.0.0.0:8300->8000/tcp

    def __init__(self) -> None:
        self._client = chromadb.HttpClient(
            host=self.CHROMA_HOST, port=self.CHROMA_PORT
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # --- Write ---

    def write_perimeter_outcome(
        self,
        episode_id: str,
        cycle: int,
        phase: int,
        blueprint_id: str,
        anchor_x: int,
        anchor_y: int,
        outcome: str,
        reward: float,
        notes: str,
        seed: str = CANONICAL_SEED,
    ) -> None: ...

    def write_causal_warning(
        self,
        episode_id: str,
        cycle: int,
        phase: int,
        cause: str,
        description: str,
        reward: float,
        seed: str = CANONICAL_SEED,
    ) -> None: ...

    def write_strategy(
        self,
        episode_id: str,
        cycle: int,
        phase: int,
        description: str,
        reward: float,
        seed: str = CANONICAL_SEED,
    ) -> None: ...

    def write_episode_summary(
        self,
        episode_id: str,
        phase: int,
        outcome: str,
        total_reward: float,
        description: str,
        seed: str = CANONICAL_SEED,
    ) -> None: ...

    # --- Retrieve ---

    def retrieve(
        self,
        state_summary: str,
        n: int = 3,
        current_seed: str = CANONICAL_SEED,
    ) -> list[MemoryRecord]: ...

    def format_for_prompt(self, records: list[MemoryRecord]) -> str:
        """Format retrieved memories as prompt section."""
        if not records:
            return ""
        lines = ["## Relevant Past Experience"]
        for r in records:
            lines.append(f"- [{r.episode_id}, cycle {r.cycle}] {r.document}")
        return "\n".join(lines)
```

---

## Integration Points

### runner.py

```python
# Initialise once
memory = MemoryStore()

# Each tick — retrieve and inject
state_summary = build_state_summary(state)  # short text, ~50 tokens
memories = await memory.retrieve(state_summary)
memory_section = memory.format_for_prompt(memories)
prompt = build_prompt(state, memory_section)

# On perimeter complete (from state diff)
if perimeter_just_completed:
    memory.write_perimeter_outcome(...)

# On dupe death (from EventDetector)
if death_event:
    memory.write_causal_warning(...)

# On episode end
memory.write_episode_summary(...)
```

### EventDetector (existing, from reward-function spec)

The existing `EventDetector` class already detects dupe deaths by diffing duplicant ID sets.
Extend it with a `on_death` callback that triggers `MemoryStore.write_causal_warning()`.

---

## Retention Policy

ChromaDB can hold millions of records. No immediate retention limit needed. Future:
- Archive records older than 500 episodes to `/data/oni-memory-archive/`
- Keep all `strategy` and `causal_warning` types indefinitely (high signal)
- Prune `perimeter_outcome` and `episode_summary` after 200 episodes

---

## Embedding Model

ChromaDB default embedding: `all-MiniLM-L6-v2` (sentence-transformers).  
Sufficient for v1 — memory documents are short factual text, semantic similarity works well.  
Upgrade path: swap to a larger embedding model served on DGX A if retrieval quality degrades.

---

## Open Items

| Item | Owner | Priority |
|------|-------|----------|
| Implement `MemoryStore` class in `src/agent/memory.py` | Dev Claude | P2 |
| Extend `EventDetector` with `on_death` callback | Dev Claude | P2 |
| Add `write_perimeter_outcome()` call to perimeter completion handler | Dev Claude | P2 |
| Add `retrieve()` + prompt injection to runner.py tick loop | Dev Claude | P2 |
| Verify ChromaDB at 10.0.0.10:8300 is accessible from Mac runner | User | P1 |
