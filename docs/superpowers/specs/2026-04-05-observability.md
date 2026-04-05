# Observability Spec

**Date:** 2026-04-05  
**Status:** Draft  
**Depends on:** reward-function, policy-training-pipeline, pipeline-inspector

---

## Overview

The Pipeline Inspector (built 2026-04-04) provides per-tick agent visibility: what prompt went
in, what the LLM said, what was validated, what was sent. That is the "right now" layer.

This spec defines the "over time" layer: how the agent is performing across episodes and
training steps. Without this, Phase 0 runs produce data but no insight.

```
Per-tick (Pipeline Inspector — done)
  prompt → LLM → validation → action → ack
  Live in browser. Session-only.

Per-episode (this spec)
  reward curve, win/loss rate, action distribution, perimeter success
  Written to W&B. Persistent. Queryable.

Per-training-step (this spec, GRPO phase)
  KL divergence, group-normalized score, policy loss
  Written to W&B by NemoClaw. Same run, same dashboard.
```

---

## Weights & Biases (W&B)

W&B is the observability backend. All episode and training metrics flow there.

**Why W&B over TensorBoard:**
- Hosted — no infrastructure to run
- Persistent across restarts (TensorBoard event files are lost if machine reboots)
- W&B MCP server (`https://mcp.withwandb.com`) lets Claude Code query runs directly
- Standard in GRPO/RL literature — veRL, OpenRLHF, CleanRL all use W&B

**Why not MLflow:**
MLflow is a good self-hosted alternative if W&B cloud is not acceptable. Architecture is the
same — swap `wandb` for `mlflow` in the tracker. Deferred; W&B is default.

### Setup

```bash
pip install wandb
wandb login  # one-time — stores key in ~/.netrc
```

W&B project: `oni-ai`  
Entity: configurable via `WANDB_ENTITY` env var  
Run naming: `{phase}-ep{episode_count}-{date}` (e.g. `phase0-ep012-20260405`)

---

## What Gets Logged

### Per-Episode Metrics (written at episode end)

| Metric | W&B key | Description |
|--------|---------|-------------|
| Episode number | `episode` | Global episode counter |
| Phase | `phase` | Current training phase (0–4) |
| End condition | `end_condition` | "win" / "loss" / "cycle_limit" |
| Total reward | `reward/total` | Sum of all tick rewards |
| Survival reward | `reward/survival` | Survival layer total |
| Progress reward | `reward/progress` | Progress layer total |
| Event reward | `reward/events` | Event-response layer total |
| Outcome reward | `reward/outcome` | Episode outcome layer total |
| Cycles survived | `cycles_survived` | How many cycles completed |
| Dupe deaths | `dupe_deaths` | Total deaths this episode |
| Perimeter wins | `perimeter/wins` | Perimeters completed 100% |
| Perimeter fails | `perimeter/fails` | Perimeters abandoned or failed |
| Action count | `actions/total` | Non-no_op actions this episode |
| no_op rate | `actions/noop_rate` | Fraction of ticks that were no_op |
| LLM latency p50 | `llm/latency_p50_ms` | Median LLM call time |
| LLM latency p95 | `llm/latency_p95_ms` | 95th percentile LLM call time |
| Validation blocked | `validation/blocked_rate` | Fraction of actions blocked by validator |
| Prompt length avg | `prompt/tokens_avg` | Average prompt token estimate |

### Per-Tick Metrics (sampled — every 10 ticks to avoid W&B rate limits)

| Metric | W&B key | Description |
|--------|---------|-------------|
| Tick reward | `tick/reward` | Raw reward this tick |
| O2 level | `tick/o2_kg` | Current oxygen |
| Dupe count | `tick/dupe_count` | Living dupes |
| Avg stress | `tick/avg_stress` | Mean dupe stress |
| Perimeter progress | `tick/perimeter_pct` | Active perimeter % complete |

### Per-Training-Step Metrics (written by NemoClaw/NeMo Gym — GRPO phase)

| Metric | W&B key | Description |
|--------|---------|-------------|
| Policy loss | `train/policy_loss` | GRPO policy gradient loss |
| KL divergence | `train/kl_divergence` | KL from reference model |
| Group reward mean | `train/group_reward_mean` | Mean reward across group |
| Group reward std | `train/group_reward_std` | Reward variance in group |
| Checkpoint | `train/checkpoint` | Checkpoint number promoted |

veRL metric naming convention adopted — compatible with existing GRPO tooling.

---

## Implementation: `src/agent/tracker.py`

New module. Wraps W&B with fallback to no-op when W&B is not configured.

```python
# src/agent/tracker.py

class RunTracker:
    """
    Wraps W&B for episode and training metrics.
    Falls back to no-op if WANDB_API_KEY not set.
    """

    def __init__(self, phase: int, config: dict) -> None:
        """
        Initialise W&B run. Call once at episode start.
        config: dict of hyperparameters logged as W&B config
                (phase, canonical_seed, model_backend, episode_batch_size, etc.)
        """

    def log_tick(self, tick: int, cycle: int, reward: float, state: dict) -> None:
        """Log sampled per-tick metrics. Only writes every LOG_TICK_INTERVAL ticks."""

    def log_episode(self, episode_record: "EpisodeRecord", pipeline_stats: dict) -> None:
        """Log full episode metrics at episode end. Always writes."""

    def log_training_step(self, step: int, metrics: dict) -> None:
        """Log GRPO training step metrics. Called by NeMo Gym callback."""

    def finish(self) -> None:
        """Close the W&B run. Call at episode end before reset."""
```

### Integration in `runner.py`

```python
tracker = RunTracker(phase=CURRENT_PHASE, config={
    "phase": CURRENT_PHASE,
    "seed": CANONICAL_SEED,
    "model_backend": "gemini",  # or "vllm"
    "episode_max_cycle": EPISODE_MAX_CYCLE,
    "colony_type_policy": COLONY_TYPE_POLICY,
})

# Each tick
tracker.log_tick(tick, cycle, reward, state)

# Episode end
tracker.log_episode(reward_calc.episode_record, pipeline_stats)
tracker.finish()
```

`pipeline_stats` is assembled from the `PipelineCapture` data already collected by the
Pipeline Inspector — no new data collection needed, just aggregation.

---

## Dashboard: Training Tab

New dashboard tab (alongside Pipeline tab) showing cross-episode training progress.
Reads from two sources:
1. W&B API (if configured) — live run data
2. Local episode JSONL files in `data/episodes/` (fallback, always available)

### Layout

```
┌─ Training ──────────────────────────────────────────────────────┐
│                                                                  │
│  Phase: 0  │  Episodes: 12  │  Win Rate: 58%  │  Best: 94.3    │
│                                                                  │
│  ┌─ Reward Curve ──────────────────────┐  ┌─ Action Dist. ───┐ │
│  │  (sparkline — total reward/ep)      │  │  dig: 34%        │ │
│  │                                     │  │  no_op: 41%      │ │
│  └─────────────────────────────────────┘  │  place_bldg: 18% │ │
│                                           │  perimeter: 7%   │ │
│  ┌─ Last 5 Episodes ─────────────────┐   └──────────────────┘ │
│  │  ep12  win    94.3  cycle 3  ✓    │                         │
│  │  ep11  loss    2.1  cycle 1  ✗    │   ┌─ Alerts ─────────┐ │
│  │  ep10  win    88.7  cycle 3  ✓    │   │  ✓ No anomalies  │ │
│  │  ep09  loss   18.4  cycle 2  ✗    │   └──────────────────┘ │
│  │  ep08  win    91.2  cycle 3  ✓    │                         │
│  └───────────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

### JSONL Aggregator (`src/agent/episode_aggregator.py`)

Reads `data/episodes/*.jsonl`, computes:
- Rolling win rate (last 20 episodes)
- Reward mean/std (last 20 episodes)
- Action distribution (last episode)
- Phase progression history

Exposed via FastAPI endpoint `GET /training-stats` — dashboard polls every 30s.
Does not require W&B to be configured.

---

## Anomaly Alerts

Alerts surface in the Dashboard Training tab and are logged to W&B as events.

| Condition | Alert | Severity |
|-----------|-------|----------|
| 5+ consecutive episodes fail in ≤ cycle 1 | "Agent dying immediately — check O2/food baseline" | Critical |
| Reward < -5.0 for 3+ episodes | "Reward collapsed — check reward function" | High |
| no_op rate > 70% for an episode | "Agent stuck in no_op loop" | Medium |
| Validation blocked rate > 50% | "High hallucination rate — check prompt" | Medium |
| LLM latency p95 > 30s | "LLM too slow for real-time play" | Low |
| Episode JSONL write failure | "Data loss — check disk space" | High |

Alerts are not blocking — they appear in the dashboard and are logged, but do not stop the run.

---

## Anomaly Thresholds (tunable)

All thresholds configurable via dashboard Config tab:

```python
ALERT_CONSECUTIVE_FAIL_CYCLES = 1    # fail within this many cycles = alert
ALERT_CONSECUTIVE_FAIL_COUNT  = 5    # this many consecutive fails = alert
ALERT_REWARD_COLLAPSE_THRESHOLD = -5.0
ALERT_REWARD_COLLAPSE_EPISODES  = 3
ALERT_NOOP_RATE_THRESHOLD       = 0.70
ALERT_BLOCK_RATE_THRESHOLD      = 0.50
ALERT_LLM_LATENCY_P95_MS        = 30_000
```

---

## W&B MCP Server Integration

The W&B MCP server (`https://mcp.withwandb.com`) allows Claude Code to query W&B runs
directly from the editor. Add to `.claude/settings.json`:

```json
{
  "mcpServers": {
    "wandb": {
      "type": "http",
      "url": "https://mcp.withwandb.com",
      "headers": {
        "Authorization": "Bearer ${WANDB_API_KEY}"
      }
    }
  }
}
```

This enables: "show me the reward curve for the last 20 episodes" or "which phase 0 run had
the highest win rate" as direct queries from Claude Code without opening the W&B browser UI.

---

## Open Items

| Item | Owner | Priority |
|------|-------|----------|
| Implement `RunTracker` in `src/agent/tracker.py` | Dev Claude | P1 |
| Integrate `RunTracker` into `runner.py` | Dev Claude | P1 |
| Implement `EpisodeAggregator` in `src/agent/episode_aggregator.py` | Dev Claude | P1 |
| Add Training tab to dashboard + `/training-stats` endpoint | Dev Claude | P1 |
| Add W&B MCP server to `.claude/settings.json` | User | P1 |
| Sign up for W&B account + set `WANDB_API_KEY` | User | P1 |
| Define W&B training step callback in NeMo Gym wrapper (see nemo-gym-integration spec) | Dev Claude | P2 |
