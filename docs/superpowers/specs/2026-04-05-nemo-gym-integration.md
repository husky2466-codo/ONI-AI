# NeMo Gym Integration Spec

**Date:** 2026-04-05  
**Status:** Draft  
**Depends on:** policy-training-pipeline, game-reload-automation  
**Reference:** https://github.com/NVIDIA-NeMo/Gym | https://docs.nvidia.com/nemo/gym/latest/

---

## Overview

NVIDIA NeMo Gym provides the **OpenEnv protocol** — a standardized async HTTP interface
that lets NemoClaw call any custom environment (our ONI game) as a microservice. It eliminates
the need to write custom training loop glue code between the game bridge and NemoClaw.

This is the answer to the open question in the training pipeline spec: *"how does NemoClaw
actually call our ONI game?"*

```
Before (custom glue — DON'T BUILD):
  NemoClaw → custom Python → runner.py → game

After (NeMo Gym OpenEnv — BUILD THIS):
  NemoClaw → OpenEnv HTTP → ONIEnvironmentService → runner.py → game
```

---

## What NeMo Gym OpenEnv Is

NeMo Gym decomposes RL training into composable async HTTP microservices:

| Service | Role | Who runs it |
|---------|------|-------------|
| Environment service | Steps the game, returns observations | Our code (wraps runner.py) |
| Reward service | Computes rewards | Our code (wraps reward.py) |  
| Policy service | Generates actions (inference) | NemoClaw (DGX A, vLLM) |
| Trainer service | Runs GRPO updates | NemoClaw (DGX B) |

We implement the **environment service** and **reward service**. NemoClaw handles the rest.
Our existing `runner.py`, `reward.py`, `client.py` and `reload.py` are the engine underneath —
the OpenEnv wrapper is just the HTTP interface on top.

---

## OpenEnv Protocol (what we implement)

The environment service exposes three HTTP endpoints:

### `POST /reset`

Start a new episode. Triggers `EpisodeReloader` to load `training-start.sav`, waits for
bridge ready, returns the first observation.

**Request:**
```json
{ "config": { "phase": 0, "seed": "V-SNDST-C-1644640403-0-0-0" } }
```

**Response:**
```json
{
  "observation": {
    "state_prompt": "Cycle: 1\n\nResources:\n  oxygen_kg: 2.1...",
    "vision_description": "Base visible. 3 dupes near spawn...",
    "memory_context": "## Relevant Past Experience\n...",
    "raw_state": { ... }  // full state dict from bridge
  },
  "episode_id": "ep_043"
}
```

### `POST /step`

Apply one action, advance the game one tick, return the new observation + reward + done flag.

**Request:**
```json
{
  "episode_id": "ep_043",
  "action": { "action": "dig", "cell_x": 115, "cell_y": 202 }
}
```

**Response:**
```json
{
  "observation": { "state_prompt": "...", "vision_description": "...", "memory_context": "..." },
  "reward": 0.03,
  "reward_breakdown": {
    "survival": 0.03,
    "progress": 0.0,
    "events": 0.0
  },
  "done": false,
  "info": {
    "cycle": 1,
    "tick": 12,
    "end_condition": null,
    "dupe_deaths": 0
  }
}
```

### `GET /health`

Liveness check. Returns `{"status": "ok", "game_connected": true/false}`.

---

## Architecture

```
DGX B (NemoClaw trainer — port 8080)
  ↓  POST /reset, POST /step
Mac Mini (ONIEnvironmentService — port 8090)
  └── ONIEnvironmentService (FastAPI)
       ├── EpisodeReloader  (game reset)
       ├── BridgeClient     (TCP → ONI)
       ├── RewardCalculator (reward.py)
       ├── VisionCapture    (screenshot → SmolVLM2)
       ├── MemoryStore      (ChromaDB)
       └── PipelineCapture  (for W&B logging)
  ↓  TCP port 9999
Linux Desktop (ONI + ONIBridge)
```

The OpenEnv service runs on the Mac Mini alongside the existing runner, but as a
**separate process** on port 8090. During GRPO training, NemoClaw calls the service
directly. During Gemini testing, `runner.py` runs as normal — the two modes are independent.

---

## Implementation: `src/agent/env_service.py`

New module. FastAPI service implementing OpenEnv.

```python
# src/agent/env_service.py

class ONIEnvironmentService:
    """
    NeMo Gym OpenEnv protocol implementation.
    Wraps runner.py components as async HTTP microservice.
    """

    def __init__(
        self,
        host: str,          # Linux desktop IP
        port: int = 9999,   # ONIBridge port
        service_port: int = 8090,
    ) -> None:
        self.bridge = BridgeClient(host, port)
        self.reloader = EpisodeReloader(host)
        self.reward_calc = RewardCalculator()
        self.memory = MemoryStore()
        self.tracker = None  # set at reset()

    async def reset(self, config: dict) -> dict:
        """Load training-start.sav, wait for bridge, return first observation."""
        await self.reloader.reset()
        state = await self.bridge.next_state()
        self.reward_calc.reset()
        self.tracker = RunTracker(phase=config.get("phase", 0), config=config)
        return self._build_observation(state)

    async def step(self, episode_id: str, action: dict) -> dict:
        """Send action, wait for next state, compute reward."""
        await self.bridge.send_action(action)
        state = await self.bridge.next_state()
        reward, breakdown = self.reward_calc.tick(state)
        done, end_condition = self._check_done(state)
        obs = self._build_observation(state)
        self.tracker.log_tick(self.reward_calc.tick_count, state.cycle, reward, state)
        if done:
            self.tracker.log_episode(self.reward_calc.episode_record, {})
            self.tracker.finish()
            self.memory.write_episode_summary(...)
        return {"observation": obs, "reward": reward, "reward_breakdown": breakdown,
                "done": done, "info": {"cycle": state.cycle, "end_condition": end_condition}}

    def _build_observation(self, state) -> dict:
        """Assemble state_prompt + vision + memory into observation dict."""
        vision = self.vision.describe(self._latest_screenshot())
        memories = self.memory.retrieve(self._state_summary(state))
        prompt = build_prompt(state, vision, memories)
        return {"state_prompt": prompt, "raw_state": state.data}
```

### FastAPI app

```python
app = FastAPI()
service = ONIEnvironmentService(host=GAME_HOST)

@app.post("/reset")
async def reset(req: ResetRequest) -> ObservationResponse:
    return await service.reset(req.config)

@app.post("/step")
async def step(req: StepRequest) -> StepResponse:
    return await service.step(req.episode_id, req.action)

@app.get("/health")
async def health():
    return {"status": "ok", "game_connected": service.bridge.is_connected}
```

Start with: `uvicorn src.agent.env_service:app --port 8090 --host 0.0.0.0`

---

## NemoClaw Configuration (DGX B side)

NemoClaw/NeMo Gym config pointing at our service:

```yaml
# nemo_gym_config.yaml (on DGX B)
environment:
  type: openenv
  endpoint: http://10.0.0.210:8090   # Mac Mini
  reset_timeout_s: 120               # matches EpisodeReloader timeout
  step_timeout_s: 30

policy:
  type: vllm
  endpoint: http://10.0.0.69:8000/v1  # DGX A inference
  model: Qwen2.5-72B-Instruct-AWQ

trainer:
  type: grpo
  episodes_per_update: 8             # Phase 0/1
  group_size: 4
  learning_rate: 5e-6
  kl_penalty: 0.01
  checkpoint_dir: /data/oni-checkpoints/
  checkpoint_promotion:
    target_endpoint: http://10.0.0.69:8080/reload-model

tracking:
  backend: wandb
  project: oni-ai
  entity: ${WANDB_ENTITY}
```

---

## Reward Service (Optional Separation)

NeMo Gym supports an optional separate reward microservice. For Phase 0/1, rewards are
computed inside the environment service (simpler). If reward computation becomes expensive
(vision model + LLM judge), separate it:

```
POST http://10.0.0.210:8091/reward
  { "state": {...}, "action": {...}, "prev_state": {...} }
→ { "reward": 0.03, "breakdown": {...} }
```

Deferred until Phase 2. Environment service computes rewards inline for now.

---

## W&B Training Step Callback

NeMo Gym supports training step callbacks. Register our W&B tracker to receive GRPO metrics:

```python
# In NemoClaw config or trainer setup:
def on_training_step(step: int, metrics: dict) -> None:
    tracker.log_training_step(step, metrics)
    # metrics contains: policy_loss, kl_divergence, group_reward_mean, etc.
```

This is the bridge between NemoClaw's training loop and our W&B run — same W&B run logs
both episode data (from the environment service) and training step data (from NemoClaw).

---

## Dual-Mode Operation

The system runs in two modes. They are independent — no code conflict.

| Mode | How to start | Who calls the game |
|------|-------------|-------------------|
| **Gemini test mode** | `python -m src.agent.runner` | runner.py directly |
| **GRPO training mode** | `uvicorn src.agent.env_service:app` + NemoClaw | NemoClaw → OpenEnv → env_service |

During Gemini test mode, `env_service.py` is not running. During training mode, `runner.py`
is not running as a standalone process (its components are imported by `env_service.py`).

---

## Open Items

| Item | Owner | Priority |
|------|-------|----------|
| Implement `ONIEnvironmentService` in `src/agent/env_service.py` | Dev Claude | P1 |
| Add `/reset`, `/step`, `/health` FastAPI endpoints | Dev Claude | P1 |
| Wire `RunTracker` into `env_service.py` | Dev Claude | P1 |
| Write NemoClaw `nemo_gym_config.yaml` for DGX B | User (with Ross) | P1 |
| Set up vLLM reload endpoint on DGX A (`/reload-model`) | User | P2 |
| Test OpenEnv service with mock NemoClaw client | Dev Claude | P2 |
| Implement reward microservice separation | Dev Claude | P3 (Phase 2+) |
