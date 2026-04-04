# Reward Function & Event Detection — Design Spec

**Date:** 2026-04-04
**Status:** Approved for implementation

---

## Overview

The reward function is the learning signal — it tells the agent what "good play" means in
numerical terms. Without it, the training loop has no direction. With a poorly designed one,
the agent learns the wrong thing.

### Design Philosophy

Three principles guide every decision in this spec:

1. **Align with human judgment** — a human player evaluates a colony run by asking: did my
   dupes survive, did the colony grow, did I respond well to crises, did I avoid stupid
   mistakes? The reward function encodes exactly these four questions as scalar signals.

2. **Dense enough to learn from, sparse enough to trust** — per-tick survival signals give
   the agent feedback every step. Per-cycle progress signals reward deliberate strategy.
   Episode-level outcomes reward whole-run quality. All three are needed.

3. **Close every reward hacking door** — for each reward component, ask: "what's the dumbest
   way to maximize this without playing well?" Then close it. All hacks identified and
   mitigated in this spec.

### What the agent experiences

Each tick, the agent sees a **Colony Health dashboard** in its prompt — a human-readable
summary of its reward components and open obligations. This is not a raw scalar; it's
interpretable feedback the agent can reason about. Over training, the model learns to
associate good dashboard states with high cumulative reward.

---

## Reward Structure: Four Layers

```
R_total(tick) = R_survival(tick) + R_progress(tick) + R_event_response(tick)
R_final       = R_total_accumulated + R_episode_outcome
```

---

### Layer 1: Survival Signal (per-tick, dense)

Computed every tick. The foundational signal — hard to hack because it measures binary facts.

```python
def survival_reward(state: dict) -> float:
    r = 0.0
    dupes = state.get("duplicants", [])
    colony = state.get("colony", {})

    # Living dupes: base positive signal
    r += 0.1 * len(dupes)

    # Breathability: colony-wide oxygen check
    # Use BreathabilityDiagnostic alert as proxy until atmosphere_zones are serialized
    alerts = [a.lower() for a in state.get("alerts", [])]
    breathable = not any("breathability" in a for a in alerts)
    r += 0.05 if breathable else -0.5

    # Food adequacy: no FoodDiagnostic alert
    fed = not any("food" in a for a in alerts)
    r += 0.05 if fed else -0.3

    # Net power: not in the red
    power_kw = state.get("resources", {}).get("power_kw", 0.0)
    r += 0.02 if power_kw >= 0 else -0.1

    # Stress penalty: only kicks in above critical threshold (0.8)
    # Below 0.8, stress is normal and should not be penalized
    for dupe in dupes:
        stress = dupe.get("stress", 0.0)
        if stress > 0.8:
            r -= 0.1 * (stress - 0.8) / 0.2  # linear from 0 at 0.8 to -0.1 at 1.0

    return round(r, 4)
```

**Hack mitigation:**
- Oxygen is breathable/not, not quantity — spamming algae terrariums doesn't inflate the score
- Food is fed/not, not kcal quantity — hoarding food doesn't help
- Stress penalty only above 0.8 — minor stress from normal work is not penalized

---

### Layer 2: Progress Signal (per-cycle, medium density)

Computed once per cycle (when `state.cycle > self._last_cycle`). Rewards colony development.

```python
def progress_reward(state: dict, prev_state: dict, ledger: SpatialLedger) -> float:
    r = 0.0
    curr_cycle = state.get("cycle", 0)
    prev_cycle = prev_state.get("cycle", 0)
    if curr_cycle <= prev_cycle:
        return 0.0  # only fire on cycle boundary

    # Survived the cycle
    if state.get("duplicants"):
        r += 1.0

    # New dupes who have survived 5+ cycles
    # Tracked via DupeTracker (see below)
    r += 1.0 * self._dupe_tracker.new_graduates(curr_cycle)

    # Research milestone: new tech completed
    prev_unlocked = set(prev_state.get("research", {}).get("unlocked", []))
    curr_unlocked = set(state.get("research", {}).get("unlocked", []))
    new_techs = curr_unlocked - prev_unlocked
    r += 1.0 * len(new_techs)

    # Perimeter completion: reward scales with blueprint complexity
    new_completions = ledger.completions_since_last_cycle()
    for entry in new_completions:
        blueprint = BlueprintLibrary.get(entry.blueprint_id)
        complexity = len(blueprint["buildings"]) if blueprint else 5
        r += 0.5 * complexity  # SPOM (15 buildings) = +7.5, trivial zone (1 building) = +0.5

    return round(r, 4)
```

**Hack mitigation:**
- New dupe bonus requires 5-cycle survival — accept-and-starve doesn't pay off
- Perimeter reward scales with blueprint size — trivial 1×1 perimeters return minimal reward
- Research reward is per-tech, not per-point — can't grind the same tech

---

### Layer 3: Episode Outcome (sparse, end-of-episode only)

Added to the episode's total reward when the episode ends. Rewards whole-run quality.

```python
def episode_outcome_reward(episode: EpisodeRecord) -> float:
    r = 0.0

    # Survival duration: reward per cycle beyond the baseline
    cycles = episode.final_cycle
    r += max(0.0, cycles - 10) * 1.0  # +1 per cycle survived past cycle 10

    # Perfect run: no dupe ever died
    if episode.total_deaths == 0:
        r += 50.0

    # Infrastructure milestones (time-weighted: earlier = better)
    for milestone, cycle_achieved in episode.milestones.items():
        base = MILESTONE_REWARDS[milestone]
        # Earlier achievement gets higher reward (decays linearly to 50% by cycle 50)
        time_bonus = max(0.5, 1.0 - (cycle_achieved / 100.0))
        r += base * time_bonus

    return round(r, 4)

MILESTONE_REWARDS = {
    "spom_online":        10.0,  # electrolyzer + H2 generator producing O2
    "food_sustainable":    8.0,  # farm actively producing, no ration box dependency
    "5_dupes_alive":       5.0,
    "first_research":      3.0,
    "power_positive":      3.0,  # battery net charging
}
```

---

### Layer 4: Event Response (triggered, sparse)

The most important layer for encoding ONI-specific gameplay knowledge. Fires when specific
game events occur and the agent responds correctly (or fails to).

Events are detected by `EventDetector` (see below) and tracked as open obligations with
response windows.

```python
EVENT_REWARDS = {
    # format: (reward_if_responded, penalty_per_cycle_ignored, response_window_cycles)

    "dupe_death":      (-10.0, 0.0,  0),   # penalty fires immediately on death
    "memorial_placed": (+2.0,  0.0,  3),   # bonus if memorial placed within 3 cycles

    "print_pod_ready": (0.0,  -0.2,  0),   # no bonus for deciding, but penalty for stalling
    "print_decision":  (+0.5,  0.0,  2),   # small bonus for deciding within 2 cycles

    "dupe_entombed":   (0.0,  -1.0,  0),   # penalty per cycle trapped
    "rescue_issued":   (+1.0,  0.0,  3),   # bonus for issuing rescue dig promptly

    "o2_critical":     (0.0,  -0.5,  0),   # penalty per cycle breathability alert active
    "o2_resolved":     (+1.0,  0.0,  5),   # bonus if resolved within 5 cycles
}
```

**Death math confirmation — never gameable:**
```
Dupe death fires:    -10.0
Memorial bonus:       +2.0  (if placed)
Net best case:        -8.0  (still very bad)
Net without memorial: -10.0 (plus ongoing morale cascade)
```

---

## EventDetector

Runs every tick in `runner.py`, comparing consecutive state snapshots to detect transitions.

```python
@dataclass
class GameEvent:
    type: str
    cycle: int
    tick: int
    data: dict
    resolved: bool = False
    reward_issued: bool = False

class EventDetector:
    def __init__(self):
        self._last_dupe_ids: Set[int] = set()
        self._last_positions: Dict[int, dict] = {}
        self._open_events: List[GameEvent] = []

    def detect(self, prev_state: dict, curr_state: dict) -> List[GameEvent]:
        new_events = []
        curr_cycle = curr_state.get("cycle", 0)
        curr_tick  = curr_state.get("tick", 0)

        curr_dupes = {d["id"]: d for d in curr_state.get("duplicants", [])}
        prev_dupes = {d["id"]: d for d in prev_state.get("duplicants", [])}

        # --- Dupe deaths ---
        for lost_id, lost_dupe in prev_dupes.items():
            if lost_id not in curr_dupes:
                new_events.append(GameEvent(
                    type="dupe_death", cycle=curr_cycle, tick=curr_tick,
                    data={"dupe_id": lost_id, "name": lost_dupe["name"],
                          "x": lost_dupe["x"], "y": lost_dupe["y"]},
                ))

        # --- Memorial placed (closes a dupe_death event) ---
        curr_buildings = {(b["type"], b["x"], b["y"]) for b in curr_state.get("buildings", [])}
        prev_buildings = {(b["type"], b["x"], b["y"]) for b in prev_state.get("buildings", [])}
        new_buildings = curr_buildings - prev_buildings
        memorial_ids = {"TastefulMemorial", "GraveMarker"}  # verify exact IDs via decompile
        for btype, bx, by in new_buildings:
            if btype in memorial_ids:
                for event in self._open_events:
                    if event.type == "dupe_death" and not event.resolved:
                        # Check proximity (within 10 tiles of death location)
                        dx = abs(bx - event.data["x"])
                        dy = abs(by - event.data["y"])
                        if dx <= 10 and dy <= 10:
                            event.resolved = True
                            new_events.append(GameEvent(
                                type="memorial_placed", cycle=curr_cycle, tick=curr_tick,
                                data={"for_dupe": event.data["name"]},
                            ))

        # --- Print pod became available ---
        prev_pod = prev_state.get("printing_pod", {})
        curr_pod = curr_state.get("printing_pod", {})
        if (curr_pod.get("status") == "waiting_for_decision"
                and prev_pod.get("status") != "waiting_for_decision"):
            new_events.append(GameEvent(
                type="print_pod_ready", cycle=curr_cycle, tick=curr_tick, data={},
            ))

        # --- Print decision made (closes print_pod_ready) ---
        if (prev_pod.get("status") == "waiting_for_decision"
                and curr_pod.get("status") != "waiting_for_decision"):
            for event in self._open_events:
                if event.type == "print_pod_ready" and not event.resolved:
                    event.resolved = True
                    new_events.append(GameEvent(
                        type="print_decision", cycle=curr_cycle, tick=curr_tick, data={},
                    ))

        # --- Entombment ---
        for dupe in curr_state.get("duplicants", []):
            if dupe.get("current_task") == "Entombed":
                already_open = any(
                    e.type == "dupe_entombed" and e.data["dupe_id"] == dupe["id"]
                    and not e.resolved
                    for e in self._open_events
                )
                if not already_open:
                    new_events.append(GameEvent(
                        type="dupe_entombed", cycle=curr_cycle, tick=curr_tick,
                        data={"dupe_id": dupe["id"], "name": dupe["name"],
                              "x": dupe["x"], "y": dupe["y"]},
                    ))

        # Add new events to open list, remove resolved+rewarded ones
        self._open_events.extend(new_events)
        self._open_events = [e for e in self._open_events if not (e.resolved and e.reward_issued)]

        return new_events

    def open_obligations(self, curr_cycle: int) -> List[dict]:
        """Returns open events formatted for the prompt context block."""
        obs = []
        for event in self._open_events:
            if event.resolved:
                continue
            _, penalty_per_cycle, window = EVENT_REWARDS.get(event.type, (0, 0, 0))
            cycles_open = curr_cycle - event.cycle
            if event.type == "dupe_death":
                obs.append({
                    "urgency": "HIGH",
                    "message": (f"[DEATH - cycle {event.cycle}] {event.data['name']} died at "
                                f"({event.data['x']},{event.data['y']}) — "
                                f"place TastefulMemorial nearby ({max(0, window - cycles_open)} cycles remaining for bonus)"),
                })
            elif event.type == "print_pod_ready":
                obs.append({
                    "urgency": "MEDIUM",
                    "message": (f"[PRINT POD - cycle {event.cycle}] Decision pending "
                                f"({cycles_open} cycles unattended, -{abs(penalty_per_cycle):.1f}/cycle)"),
                })
            elif event.type == "dupe_entombed":
                obs.append({
                    "urgency": "HIGH",
                    "message": (f"[ENTOMBED - cycle {event.cycle}] {event.data['name']} trapped at "
                                f"({event.data['x']},{event.data['y']}) — issue rescue dig immediately"),
                })
        return obs
```

---

## RewardCalculator

Central class that aggregates all layers and records per-tick reward to the episode log.

```python
class RewardCalculator:
    def __init__(self, ledger: SpatialLedger, dupe_tracker: "DupeTracker"):
        self._ledger      = ledger
        self._dupe_tracker = dupe_tracker
        self._event_detector = EventDetector()
        self._episode_record = EpisodeRecord()
        self._prev_state: Optional[dict] = None

    def tick(self, state: dict) -> float:
        if self._prev_state is None:
            self._prev_state = state
            return 0.0

        r = 0.0

        # Layer 1
        r += survival_reward(state)

        # Layer 2
        r += progress_reward(state, self._prev_state, self._ledger)

        # Layer 4: detect events, compute event rewards
        new_events = self._event_detector.detect(self._prev_state, state)
        for event in new_events:
            base_reward, _, _ = EVENT_REWARDS.get(event.type, (0.0, 0.0, 0))
            r += base_reward
            if event.type == "dupe_death":
                self._episode_record.total_deaths += 1

        # Per-cycle penalty for ignored open obligations
        cycle = state.get("cycle", 0)
        for obligation in self._event_detector._open_events:
            if not obligation.resolved:
                _, penalty, _ = EVENT_REWARDS.get(obligation.type, (0.0, 0.0, 0))
                r += penalty  # penalty_per_cycle is negative

        self._episode_record.accumulate(r, state)
        self._prev_state = state
        return round(r, 4)

    def episode_end(self) -> float:
        outcome = episode_outcome_reward(self._episode_record)
        self._episode_record.outcome_reward = outcome
        return outcome

    def open_obligations(self) -> List[dict]:
        return self._event_detector.open_obligations(
            self._prev_state.get("cycle", 0) if self._prev_state else 0
        )
```

---

## DupeTracker

Tracks when each dupe was printed so the 5-cycle graduation rule can be enforced.

```python
class DupeTracker:
    def __init__(self):
        self._arrival_cycle: Dict[int, int] = {}  # dupe_id → cycle they appeared

    def update(self, state: dict) -> None:
        curr_cycle = state.get("cycle", 0)
        for dupe in state.get("duplicants", []):
            if dupe["id"] not in self._arrival_cycle:
                self._arrival_cycle[dupe["id"]] = curr_cycle

    def new_graduates(self, curr_cycle: int) -> int:
        """Returns count of dupes who just crossed 5-cycle survival threshold."""
        count = 0
        for dupe_id, arrival in self._arrival_cycle.items():
            if curr_cycle - arrival == 5:  # exactly 5 cycles → graduate this cycle
                count += 1
        return count
```

---

## Colony Health Dashboard (Prompt Context)

Injected into every state prompt by `llm.py`. This is the agent's self-awareness layer —
it sees its own performance numerically and can reason about it.

```python
def format_colony_health(state: dict, tick_reward: float, episode_reward: float,
                         obligations: List[dict]) -> str:
    lines = ["--- Colony Health ---"]

    # Reward summary
    trend = "+" if tick_reward >= 0 else ""
    lines.append(f"Tick reward: {trend}{tick_reward:.3f}  |  Episode total: {episode_reward:.1f}")

    # Status indicators
    dupes = state.get("duplicants", [])
    alerts = state.get("alerts", [])
    breathable = not any("breathability" in a.lower() for a in alerts)
    fed        = not any("food" in a.lower() for a in alerts)

    lines.append(f"Dupes: {len(dupes)} alive  |  "
                 f"O2: {'OK' if breathable else 'CRITICAL'}  |  "
                 f"Food: {'OK' if fed else 'LOW'}")

    avg_stress = sum(d.get("stress", 0) for d in dupes) / max(len(dupes), 1)
    lines.append(f"Avg stress: {avg_stress*100:.0f}%"
                 + ("  ← CRITICAL" if avg_stress > 0.8 else ""))

    # Open obligations
    if obligations:
        lines.append("Open Obligations:")
        for ob in sorted(obligations, key=lambda x: x["urgency"]):
            lines.append(f"  [{ob['urgency']}] {ob['message']}")

    lines.append("--- End Health ---")
    return "\n".join(lines)
```

**Example prompt injection:**

```
--- Colony Health ---
Tick reward: +0.342  |  Episode total: +18.7
Dupes: 3 alive  |  O2: OK  |  Food: OK
Avg stress: 14%
Open Obligations:
  [HIGH] [DEATH - cycle 8] Otto died at (132,203) — place TastefulMemorial nearby (1 cycle remaining for bonus)
  [MEDIUM] [PRINT POD - cycle 10] Decision pending (2 cycles unattended, -0.2/cycle)
--- End Health ---
```

---

## Episode Record

Stored per-episode and written to the JSONL training log.

```python
@dataclass
class EpisodeRecord:
    episode_id: str = ""
    start_cycle: int = 0
    final_cycle: int = 0
    total_deaths: int = 0
    total_reward: float = 0.0
    outcome_reward: float = 0.0
    milestones: Dict[str, int] = field(default_factory=dict)
    tick_rewards: List[float] = field(default_factory=list)

    def accumulate(self, r: float, state: dict) -> None:
        self.total_reward += r
        self.tick_rewards.append(r)
        self.final_cycle = state.get("cycle", 0)
        self._check_milestones(state)

    def _check_milestones(self, state: dict) -> None:
        cycle = state.get("cycle", 0)
        buildings = {b["type"] for b in state.get("buildings", [])}

        if "spom_online" not in self.milestones:
            if "Electrolyzer" in buildings and "HydrogenGenerator" in buildings:
                self.milestones["spom_online"] = cycle

        if "first_research" not in self.milestones:
            unlocked = state.get("research", {}).get("unlocked", [])
            if len(unlocked) > 0:
                self.milestones["first_research"] = cycle

        if "5_dupes_alive" not in self.milestones:
            if len(state.get("duplicants", [])) >= 5:
                self.milestones["5_dupes_alive"] = cycle
```

---

## How This Builds Over Time

The system is designed to self-improve through three mechanisms:

**1. The learning loop gets richer as more specs are implemented**
- Right now: survival + alert-based signals
- After extended state schema: full dupe needs, machine state, power networks — richer signals
- After perimeter system: blueprint completion rewards, prerequisite resolution rewards
- Each new telemetry field unlocks more precise reward components

**2. Win condition phases drive progressive difficulty**
- Phase 1: survive to cycle 10 (baseline)
- Phase 2: SPOM online by cycle 20 (infrastructure milestone)
- Phase 3: 5 dupes alive at cycle 50 (colony growth)
- Phase 4: 100 cycles, no deaths (mastery)
The agent trains on Phase 1 until it's consistent, then the bar raises. The reward structure
doesn't change — only the episode termination threshold and outcome weights.

**3. The event library grows with the agent's capability**
New event types can be added to `EventDetector` and `EVENT_REWARDS` without touching any
other component. As the agent handles early-game events reliably, add mid-game events
(geyser eruption management, disease outbreak, overheat warning) to the event library.

---

## Files Changed

| File | Change |
|------|--------|
| `src/agent/reward.py` | New file — RewardCalculator, EventDetector, DupeTracker, EpisodeRecord |
| `src/agent/runner.py` | Instantiate RewardCalculator; call `.tick()` each state; log reward to episode JSONL; pass open_obligations to llm |
| `src/agent/llm.py` | Add `format_colony_health()` call in `_format_state()` |
| `data/episodes/` | Episode JSONL output (gitignored, already in .gitignore per training config spec) |

---

## Out of Scope

- GRPO / fine-tuning integration (separate pipeline spec, DGX Spark side)
- Reward normalization across episodes (implement after first 10+ episodes of data)
- Automated milestone detection beyond the 5 listed (add incrementally)
- Negative reward for repeated identical actions (anti-loop) — defer until loop behavior
  is observed in live sessions; don't pre-optimize for it
