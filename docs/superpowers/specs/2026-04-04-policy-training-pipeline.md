# Policy & Training Pipeline Spec

**Date:** 2026-04-04  
**Status:** Draft  
**Depends on:** training-configuration, reward-function, game-reload-automation, extended-game-state-schema

---

## Overview

This spec defines how the ONI-AI agent moves from cloud-hosted inference (Gemini API) to a
fully on-prem training and inference loop running across two DGX Spark nodes. It covers the
model backend abstraction, vision pipeline, episode data format, GRPO training loop
orchestrated by NemoClaw, checkpoint promotion, and phase-based progression.

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Loop                          │
│                                                             │
│  Linux Desktop (10.0.0.10)                                  │
│    ONI game + ONIBridge → TCP state stream                  │
│    Screenshot capture every 10–15s                          │
│    Vision model → text description                          │
│         │                                                   │
│         ▼                                                   │
│  Mac Mini (10.0.0.210)                                      │
│    runner.py — assembles state prompt + vision text         │
│    ModelBackend.call() → inference node                     │
│         │                                                   │
│         ▼                                                   │
│  DGX A (10.0.0.69) — inference                             │
│    vLLM :8000 serving current best checkpoint               │
│    Returns action JSON                                      │
│         │                                                   │
│         ▼ (episode trajectories written to disk)            │
│  DGX B (192.168.3.20) — training / NemoClaw                │
│    NemoClaw :8080 orchestrates GRPO loop                    │
│    Llama-3.3-Nemotron-Super-49B-FP8 as policy model        │
│    Promotes checkpoint → DGX A after each update           │
└─────────────────────────────────────────────────────────────┘
```

---

## Node Roles

| Node | IP | Role | Key Service |
|------|----|------|-------------|
| Linux Desktop | 10.0.0.10 | Game host + vision | ONI + ONIBridge, vision model inference |
| Mac Mini | 10.0.0.210 | Agent runner | runner.py, dashboard |
| DGX Spark A | 10.0.0.69 | Inference | vLLM :8000 — current best checkpoint |
| DGX Spark B | 192.168.3.20 | Training + orchestration | NemoClaw :8080, GRPO training loop |

---

## Model Backend Abstraction

The agent runner uses a swappable `ModelBackend` interface. Switching from Gemini to on-prem
requires one config change — no code changes.

```python
class ModelBackend(Protocol):
    supports_vision: bool

    async def call(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
    ) -> str: ...


class GeminiBackend(ModelBackend):
    """Current default. Used until on-prem is validated."""
    supports_vision = True


class OpenAICompatibleBackend(ModelBackend):
    """Covers vLLM on DGX A, NemoClaw gateway, external OpenAI API."""
    supports_vision = True  # when serving a multimodal model
    endpoint: str           # e.g. http://10.0.0.69:8000/v1
    api_key: str
    model_name: str
```

### Backend Config (dashboard Config tab)

```
Model Backend:  [ Gemini API | vLLM on-prem | OpenAI-compatible ]
Endpoint URL:   http://10.0.0.69:8000/v1
API Key:        (masked)
Model Name:     Qwen2.5-72B-Instruct-AWQ
Vision:         [ On | Off ]
Vision interval: 10 seconds
```

### Phase 1 On-Prem Inference Model

`Qwen/Qwen2.5-72B-Instruct-AWQ` is already serving on DGX A at `http://10.0.0.69:8000/v1`.
Switching from Gemini requires only updating the dashboard config. No redeployment.

---

## Vision Pipeline

Vision gives the agent a spatial anchor that the tile grid (64×64 text coordinates) alone
cannot provide. It directly addresses the known issue: *"agent loses spatial reasoning after
a few actions."*

### Short-Term: Linux Desktop Vision Models

Two GGUF vision models are deployed on the Linux desktop (RTX 4070):

| Model | Format | Size | Role |
|-------|--------|------|------|
| `ggml-org/SmolVLM2-2.2B-Instruct-GGUF` | GGUF | ~2.2B | **Default** — fast, sufficient for game state descriptions |
| `unsloth/Qwen2.5-VL-7B-Instruct-GGUF` | GGUF | ~7B | Fallback — higher quality if SmolVLM2 descriptions prove insufficient |

Both models are served via **llama.cpp server** (not vLLM — GGUF format requires llama.cpp).
Both fit in RTX 4070 VRAM alongside running TTS containers.

Pipeline:

```
scrot on Linux desktop (every 10–15s)
    → JPEG compressed screenshot
    → llama.cpp server (SmolVLM2-2.2B, :8080 suggested)
    → text description: layout, notable structures, dupe positions
    → injected into state prompt before ModelBackend.call()
```

The text description is treated as an additional field in the state prompt, not a separate
API call. Token budget: ~150–250 tokens per vision tick.

### Medium-Term: Qwen2-VL-7B on DGX B

`Qwen/Qwen2-VL-7B-Instruct` is cached on DGX B. Once vision models on the Linux desktop
are evaluated, if native multimodal is preferred:

- Serve Qwen2-VL-7B on DGX B :8001 (separate port from training vLLM)
- Send raw image bytes directly — no text translation step
- ModelBackend routes image to DGX B for vision, text to DGX A for action

### Vision Capture Implementation

```python
# In runner.py — fires every VISION_INTERVAL_SEC seconds
async def capture_vision_description(ssh_client) -> str | None:
    """Capture screenshot on Linux desktop, run vision model, return text."""
    if not vision_enabled:
        return None
    screenshot_b64 = await ssh_client.capture_screenshot()  # scrot → base64
    description = await vision_model.describe(screenshot_b64)
    return description  # injected into next state prompt
```

`VISION_INTERVAL_SEC = 10` — default. Configurable via dashboard.

---

## Episode Data Format

Each game run from `training-start.sav` to win/fail is one **episode**. Episodes are stored
as trajectory files for GRPO training.

### Trajectory Record (per tick)

```json
{
  "episode_id": "ep_20260404_143022_001",
  "tick": 847,
  "cycle": 2,
  "phase": 0,
  "state_prompt": "...",
  "vision_description": "Base is 12 tiles wide. Two dupes digging east...",
  "action_taken": {"action": "dig", "cell_x": 135, "cell_y": 202},
  "action_raw": "...",
  "reward": 0.03,
  "cumulative_reward": 14.2,
  "terminal": false
}
```

### Episode Metadata

```json
{
  "episode_id": "ep_20260404_143022_001",
  "seed": "V-SNDST-C-1644640403-0-0-0",
  "colony": "training-start",
  "phase": 0,
  "phase_win": true,
  "final_cycle": 3,
  "total_reward": 18.4,
  "dupe_deaths": 0,
  "tick_count": 1847,
  "wall_time_sec": 94.3,
  "model_checkpoint": "checkpoint-007"
}
```

### Storage

Episodes written to DGX B: `/data/oni-episodes/{episode_id}/`  
Format: newline-delimited JSON (`.jsonl`) for trajectory, separate `.json` for metadata.  
Retention: keep last 500 episodes; archive older episodes to `/data/oni-episodes/archive/`.

---

## GRPO Training Loop

NemoClaw on DGX B (:8080) orchestrates the full loop. Python runner.py is responsible only
for game-side collection; it does not manage training.

### Policy Model

`nvidia/Llama-3.3-Nemotron-Super-49B-v1.5-FP8`  
- FP8 fine-tunable with NeMo framework
- ~49G — fits in DGX B 128GB with headroom for activations
- NVIDIA-native: deepest NemoClaw integration
- Strong instruction-following baseline for game action parsing

### GRPO Hyperparameters (Phase 0 / Phase 1 defaults)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Group size (G) | 4 | Completions per prompt per GRPO step |
| Episodes per update | 8 | Collect 8 full episodes, then update |
| Learning rate | 5e-6 | Conservative — model is already instruction-tuned |
| KL penalty (β) | 0.01 | Low initially; prevents collapse away from base |
| Max sequence length | 4096 | Covers full state prompt + action response |
| Gradient accumulation | 4 | Effective batch = 32 |
| Discount factor (γ) | 0.999 | Per reward-function spec |
| Reward clipping | [-2.0, +2.0] | Per reward-function spec |

Episode batch size scales down as phases get longer:

| Phase | Win condition | Episodes per update |
|-------|--------------|---------------------|
| 0 | Survive 3 cycles | 8 |
| 1 | Survive 50 cycles | 4 |
| 2+ | Longer horizons | 2–4 |

### NemoClaw Orchestration Responsibilities

NemoClaw owns everything on the training side:

1. **Episode intake** — watches `/data/oni-episodes/` for new completed episodes
2. **Batch assembly** — waits for N episodes, assembles GRPO training batch
3. **GRPO update** — fine-tunes Nemotron-49B checkpoint on batch
4. **Checkpoint save** — writes to `/data/oni-checkpoints/checkpoint-{N:03d}/`
5. **Checkpoint promotion** — copies new checkpoint to DGX A vLLM serving path, signals reload
6. **Logging** — reward curves, KL divergence, loss to `/data/oni-training-logs/`

Python runner.py is **not** involved in steps 1–6. Clean boundary.

### Checkpoint Promotion Policy (Phase 0/1)

**Immediate promotion** — after each GRPO update, the new checkpoint is promoted to DGX A
unconditionally. Rationale: early phases are low-stakes, fast iteration is more valuable than
stability guarantees. A bad update is recoverable by resetting training from a prior checkpoint.

Gated promotion (promote only if eval score improves) is deferred to Phase 3+.

### Checkpoint Promotion Mechanism

```
DGX B (training)
  NemoClaw writes checkpoint → /data/oni-checkpoints/checkpoint-{N}/
  HTTP POST http://10.0.0.69:8080/reload-model  ← DGX A management endpoint
      body: { "checkpoint_path": "/data/oni-checkpoints/checkpoint-{N}/" }

DGX A (inference)
  vLLM receives reload signal
  Hot-swaps model weights (no service interruption if vLLM supports it)
  Falls back to: stop vLLM → swap symlink → restart vLLM
```

NVLink 200GB/s cable between DGX A and DGX B is available for checkpoint transfer if
network path is insufficient, though 49B FP8 checkpoint (~49G) transfers in ~2s at 200GB/s.

---

## Phase Progression Reference

Defined fully in `2026-04-04-training-configuration.md`. Summary for pipeline context:

| Phase | Win Condition | Expected Episode Length | Notes |
|-------|--------------|------------------------|-------|
| 0 | Survive 3 cycles, 0 dupe deaths | ~1–5 min | Smoke test |
| 1 | Survive to cycle 50 | ~15–45 min | Core survival loop |
| 2 | SPOM operational by cycle 30 | ~30–60 min | Infrastructure milestone |
| 3 | Print 2nd dupe by cycle 20 | ~30–60 min | Colony growth |
| 4 | Survive to cycle 200 | Hours | Long-horizon mastery |

Model is never retrained from scratch between phases. Each phase continues fine-tuning from
the best checkpoint of the prior phase.

---

## Colony Type Policy

`COLONY_TYPE_POLICY = "organic_only"` (Phase 0–2)

The agent must not print bionic duplicants from the printing pod during early training phases.
Bionic dupes do not eat or breathe, removing the core survival constraints the agent is
learning. This is enforced in `llm.py` prompt instructions and in `ActionExecutor.cs`
(`accept_print` action validates dupe type before accepting).

---

## ChromaDB — Future Agent Memory

ChromaDB is running on the Linux desktop (10.0.0.10). It is not used in this spec but is
flagged for a future **Agent Memory spec**:

- Store successful perimeter blueprints with reward outcomes
- Store colony layouts by phase that led to wins
- Retrieve relevant past strategies as context for the current state prompt
- Replaces or augments the wiki tool as the agent's long-term knowledge store

This is out of scope for Phase 0/1 but the infrastructure is already in place.

---

## Open Items

| Item | Owner | Priority |
|------|-------|----------|
| Deploy llama.cpp server for SmolVLM2-2.2B on Linux desktop :8080 | User | P1 — blocks vision pipeline |
| Implement `OpenAICompatibleBackend` in `llm.py` | Dev Claude | P1 |
| Implement `capture_vision_description()` in `runner.py` | Dev Claude | P1 |
| Add Config tab to dashboard | Dev Claude | P1 (in progress) |
| Set up vLLM reload endpoint on DGX A | User | P2 |
| Configure NemoClaw episode intake pipeline on DGX B | User | P2 |
| Write Agent Memory spec (ChromaDB) | Brainstorm Claude | P3 |
