# src/agent/reward.py
"""
Reward function, event detection, and colony health dashboard for the ONI AI agent.

Components:
  survival_reward       — per-tick dense signal
  progress_reward       — per-cycle medium signal
  episode_outcome_reward — end-of-episode sparse signal
  EVENT_REWARDS         — event-driven sparse signals
  EventDetector         — detects game events from consecutive state diffs
  DupeTracker           — tracks dupe arrival cycles for graduation rule
  RewardCalculator      — aggregates all layers per tick
  EpisodeRecord         — per-episode accounting
  format_colony_health  — prompt context block
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reward clipping
# ---------------------------------------------------------------------------

REWARD_CLIP_MIN = -2.0
REWARD_CLIP_MAX = +2.0


def clip(r: float) -> float:
    return max(REWARD_CLIP_MIN, min(REWARD_CLIP_MAX, r))


# ---------------------------------------------------------------------------
# Event definitions
# ---------------------------------------------------------------------------

# format: (reward_if_event_fires, penalty_per_cycle_ignored, response_window_cycles)
EVENT_REWARDS: dict[str, tuple[float, float, int]] = {
    "dupe_death":      (-10.0, 0.0,  0),
    "memorial_placed": (+2.0,  0.0,  3),
    "print_pod_ready": (0.0,  -0.2,  0),
    "print_decision":  (+0.5,  0.0,  2),
    "dupe_entombed":   (0.0,  -1.0,  0),
    "rescue_issued":   (+1.0,  0.0,  3),
    "o2_critical":     (0.0,  -0.5,  0),
    "o2_resolved":     (+1.0,  0.0,  5),
}

MILESTONE_REWARDS: dict[str, float] = {
    "spom_online":      10.0,
    "food_sustainable":  8.0,
    "5_dupes_alive":     5.0,
    "first_research":    3.0,
    "power_positive":    3.0,
}


# ---------------------------------------------------------------------------
# Layer 1: Survival reward (per-tick)
# ---------------------------------------------------------------------------

def survival_reward(state: dict[str, Any]) -> float:
    r = 0.0
    dupes = state.get("duplicants", [])
    alerts = [a.lower() for a in state.get("alerts", [])]

    # Living dupes: base positive signal
    r += 0.1 * len(dupes)

    # Breathability: colony-wide oxygen check
    breathable = not any("breathability" in a for a in alerts)
    r += 0.05 if breathable else -0.5

    # Food adequacy
    fed = not any("food" in a for a in alerts)
    r += 0.05 if fed else -0.3

    # Net power
    power_kw = state.get("resources", {}).get("power_kw", 0.0)
    r += 0.02 if power_kw >= 0 else -0.1

    # Stress penalty (only above 0.8 threshold)
    for dupe in dupes:
        stress = dupe.get("stress", 0.0)
        if stress > 0.8:
            r -= 0.1 * (stress - 0.8) / 0.2  # linear: 0 at 0.8, -0.1 at 1.0

    return round(r, 4)


# ---------------------------------------------------------------------------
# Layer 2: Progress reward (per-cycle)
# ---------------------------------------------------------------------------

def progress_reward(
    state: dict[str, Any],
    prev_state: dict[str, Any],
    dupe_tracker: "DupeTracker",
    ledger: "Any | None" = None,  # SpatialLedger — import avoided to prevent circular
) -> float:
    curr_cycle = state.get("cycle", 0)
    prev_cycle = prev_state.get("cycle", 0)
    if curr_cycle <= prev_cycle:
        return 0.0  # only fire on cycle boundary

    r = 0.0

    # Survived the cycle
    if state.get("duplicants"):
        r += 1.0

    # New dupe graduates (5-cycle survival)
    r += 1.0 * dupe_tracker.new_graduates(curr_cycle)

    # Research milestones
    prev_unlocked = set(prev_state.get("research", {}).get("unlocked", []))
    curr_unlocked = set(state.get("research", {}).get("unlocked", []))
    new_techs = curr_unlocked - prev_unlocked
    r += 1.0 * len(new_techs)

    # Perimeter completion (blueprint complexity reward)
    if ledger is not None:
        try:
            new_completions = ledger.completions_since_last_cycle()
            for entry in new_completions:
                complexity = 5  # default if blueprint unknown
                r += 0.5 * complexity
        except AttributeError:
            pass  # ledger may not implement completions_since_last_cycle yet

    return round(r, 4)


# ---------------------------------------------------------------------------
# Layer 3: Episode outcome (end-of-episode)
# ---------------------------------------------------------------------------

def episode_outcome_reward(episode: "EpisodeRecord") -> float:
    r = 0.0

    # Survival duration
    cycles = episode.final_cycle
    r += max(0.0, cycles - 10) * 1.0  # +1 per cycle past cycle 10

    # Perfect run: no deaths
    if episode.total_deaths == 0:
        r += 50.0

    # Infrastructure milestones (time-weighted)
    for milestone, cycle_achieved in episode.milestones.items():
        base = MILESTONE_REWARDS.get(milestone, 0.0)
        time_bonus = max(0.5, 1.0 - (cycle_achieved / 100.0))
        r += base * time_bonus

    return round(r, 4)


# ---------------------------------------------------------------------------
# EpisodeRecord
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    episode_id: str = ""
    start_cycle: int = 0
    final_cycle: int = 0
    total_deaths: int = 0
    total_reward: float = 0.0
    outcome_reward: float = 0.0
    milestones: dict[str, int] = field(default_factory=dict)
    tick_rewards: list[float] = field(default_factory=list)

    def accumulate(self, r: float, state: dict[str, Any]) -> None:
        self.total_reward += r
        self.tick_rewards.append(r)
        self.final_cycle = state.get("cycle", 0)
        self._check_milestones(state)

    def _check_milestones(self, state: dict[str, Any]) -> None:
        cycle = state.get("cycle", 0)
        buildings = {b["type"] for b in state.get("buildings", [])}

        if "spom_online" not in self.milestones:
            if "Electrolyzer" in buildings and "HydrogenGenerator" in buildings:
                self.milestones["spom_online"] = cycle

        if "first_research" not in self.milestones:
            if len(state.get("research", {}).get("unlocked", [])) > 0:
                self.milestones["first_research"] = cycle

        if "5_dupes_alive" not in self.milestones:
            if len(state.get("duplicants", [])) >= 5:
                self.milestones["5_dupes_alive"] = cycle

        if "power_positive" not in self.milestones:
            if state.get("resources", {}).get("power_kw", 0) > 0:
                self.milestones["power_positive"] = cycle


# ---------------------------------------------------------------------------
# GameEvent
# ---------------------------------------------------------------------------

@dataclass
class GameEvent:
    type: str
    cycle: int
    tick: int
    data: dict[str, Any]
    resolved: bool = False
    reward_issued: bool = False


# ---------------------------------------------------------------------------
# EventDetector
# ---------------------------------------------------------------------------

MEMORIAL_BUILDING_IDS = {"TastefulMemorial", "GraveMarker"}


class EventDetector:
    def __init__(self) -> None:
        self._open_events: list[GameEvent] = []
        self._o2_crisis_active: bool = False

    def detect(self, prev_state: dict[str, Any], curr_state: dict[str, Any]) -> list[GameEvent]:
        new_events: list[GameEvent] = []
        curr_cycle = curr_state.get("cycle", 0)
        curr_tick = curr_state.get("tick", 0)

        curr_dupes = {d["id"]: d for d in curr_state.get("duplicants", [])}
        prev_dupes = {d["id"]: d for d in prev_state.get("duplicants", [])}

        # --- Dupe deaths ---
        for lost_id, lost_dupe in prev_dupes.items():
            if lost_id not in curr_dupes:
                new_events.append(GameEvent(
                    type="dupe_death", cycle=curr_cycle, tick=curr_tick,
                    data={
                        "dupe_id": lost_id,
                        "name": lost_dupe["name"],
                        "x": lost_dupe["x"],
                        "y": lost_dupe["y"],
                    },
                ))

        # --- Memorial placed (resolves a dupe_death event) ---
        curr_buildings = {(b["type"], b["x"], b["y"]) for b in curr_state.get("buildings", [])}
        prev_buildings = {(b["type"], b["x"], b["y"]) for b in prev_state.get("buildings", [])}
        new_buildings = curr_buildings - prev_buildings
        for btype, bx, by in new_buildings:
            if btype in MEMORIAL_BUILDING_IDS:
                for event in self._open_events:
                    if event.type == "dupe_death" and not event.resolved:
                        dx = abs(bx - event.data["x"])
                        dy = abs(by - event.data["y"])
                        if dx <= 10 and dy <= 10:
                            event.resolved = True
                            new_events.append(GameEvent(
                                type="memorial_placed", cycle=curr_cycle, tick=curr_tick,
                                data={"for_dupe": event.data["name"]},
                            ))
                            break

        # --- Printing pod state transitions ---
        prev_pod = prev_state.get("printing_pod", {})
        curr_pod = curr_state.get("printing_pod", {})
        if (curr_pod.get("status") == "waiting_for_decision"
                and prev_pod.get("status") != "waiting_for_decision"):
            new_events.append(GameEvent(
                type="print_pod_ready", cycle=curr_cycle, tick=curr_tick, data={},
            ))
        if (prev_pod.get("status") == "waiting_for_decision"
                and curr_pod.get("status") != "waiting_for_decision"):
            for event in self._open_events:
                if event.type == "print_pod_ready" and not event.resolved:
                    event.resolved = True
                    new_events.append(GameEvent(
                        type="print_decision", cycle=curr_cycle, tick=curr_tick, data={},
                    ))
                    break

        # --- Entombment ---
        curr_entombed_ids = {
            d["id"] for d in curr_state.get("duplicants", [])
            if d.get("current_task") == "Entombed"
        }
        open_entombed_ids = {
            e.data["dupe_id"] for e in self._open_events
            if e.type == "dupe_entombed" and not e.resolved
        }
        for dupe in curr_state.get("duplicants", []):
            if dupe.get("current_task") == "Entombed" and dupe["id"] not in open_entombed_ids:
                new_events.append(GameEvent(
                    type="dupe_entombed", cycle=curr_cycle, tick=curr_tick,
                    data={"dupe_id": dupe["id"], "name": dupe["name"],
                          "x": dupe["x"], "y": dupe["y"]},
                ))

        # Resolve entombed events when dupe is no longer entombed
        for event in self._open_events:
            if event.type == "dupe_entombed" and not event.resolved:
                if event.data["dupe_id"] not in curr_entombed_ids:
                    event.resolved = True

        # --- O2 crisis / resolution ---
        alerts = [a.lower() for a in curr_state.get("alerts", [])]
        o2_crisis_now = any("breathability" in a for a in alerts)
        if o2_crisis_now and not self._o2_crisis_active:
            self._o2_crisis_active = True
            new_events.append(GameEvent(
                type="o2_critical", cycle=curr_cycle, tick=curr_tick, data={},
            ))
        elif not o2_crisis_now and self._o2_crisis_active:
            self._o2_crisis_active = False
            new_events.append(GameEvent(
                type="o2_resolved", cycle=curr_cycle, tick=curr_tick, data={},
            ))

        # Add new events; prune resolved+rewarded
        self._open_events.extend(new_events)
        self._open_events = [e for e in self._open_events if not (e.resolved and e.reward_issued)]

        return new_events

    def open_obligations(self, curr_cycle: int) -> list[dict[str, Any]]:
        """Returns formatted open obligations for prompt injection."""
        obs: list[dict[str, Any]] = []
        for event in self._open_events:
            if event.resolved:
                continue
            _, penalty_per_cycle, window = EVENT_REWARDS.get(event.type, (0.0, 0.0, 0))
            cycles_open = curr_cycle - event.cycle

            if event.type == "dupe_death":
                obs.append({
                    "urgency": "HIGH",
                    "message": (
                        f"[DEATH - cycle {event.cycle}] {event.data['name']} died at "
                        f"({event.data['x']},{event.data['y']}) — "
                        f"place TastefulMemorial nearby "
                        f"({max(0, window - cycles_open)} cycles remaining for bonus)"
                    ),
                })
            elif event.type == "print_pod_ready":
                obs.append({
                    "urgency": "MEDIUM",
                    "message": (
                        f"[PRINT POD - cycle {event.cycle}] Decision pending "
                        f"({cycles_open} cycles unattended, "
                        f"-{abs(penalty_per_cycle):.1f}/cycle)"
                    ),
                })
            elif event.type == "dupe_entombed":
                obs.append({
                    "urgency": "HIGH",
                    "message": (
                        f"[ENTOMBED - cycle {event.cycle}] {event.data['name']} trapped at "
                        f"({event.data['x']},{event.data['y']}) — issue rescue dig immediately"
                    ),
                })
            elif event.type == "o2_critical":
                obs.append({
                    "urgency": "HIGH",
                    "message": (
                        f"[O2 CRITICAL - cycle {event.cycle}] Breathability alert active "
                        f"for {cycles_open} cycles — fix oxygen production immediately"
                    ),
                })
        return obs

    def total_open_penalty(self) -> float:
        """Sum of per-cycle penalties for all open (unresolved) obligations."""
        total = 0.0
        for event in self._open_events:
            if not event.resolved:
                _, penalty, _ = EVENT_REWARDS.get(event.type, (0.0, 0.0, 0))
                total += penalty
        return total


# ---------------------------------------------------------------------------
# DupeTracker
# ---------------------------------------------------------------------------

class DupeTracker:
    def __init__(self) -> None:
        self._arrival_cycle: dict[int, int] = {}

    def update(self, state: dict[str, Any]) -> None:
        curr_cycle = state.get("cycle", 0)
        for dupe in state.get("duplicants", []):
            if dupe["id"] not in self._arrival_cycle:
                self._arrival_cycle[dupe["id"]] = curr_cycle

    def new_graduates(self, curr_cycle: int) -> int:
        """Count dupes who just crossed the 5-cycle survival threshold."""
        count = 0
        for arrival in self._arrival_cycle.values():
            if curr_cycle - arrival == 5:
                count += 1
        return count


# ---------------------------------------------------------------------------
# RewardCalculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    def __init__(self, ledger: "Any | None" = None) -> None:
        self._ledger = ledger
        self._dupe_tracker = DupeTracker()
        self._event_detector = EventDetector()
        self._episode_record = EpisodeRecord()
        self._prev_state: dict[str, Any] | None = None
        self._episode_reward_total: float = 0.0

    def tick(self, state: dict[str, Any]) -> float:
        """Compute reward for this tick. Returns raw (unclipped) reward."""
        self._dupe_tracker.update(state)

        if self._prev_state is None:
            self._prev_state = state
            return 0.0

        r = 0.0

        # Layer 1: survival
        r += survival_reward(state)

        # Layer 2: progress (fires on cycle boundaries)
        r += progress_reward(state, self._prev_state, self._dupe_tracker, self._ledger)

        # Layer 4: event rewards
        new_events = self._event_detector.detect(self._prev_state, state)
        for event in new_events:
            base_reward, _, _ = EVENT_REWARDS.get(event.type, (0.0, 0.0, 0))
            r += base_reward
            event.reward_issued = True
            if event.type == "dupe_death":
                self._episode_record.total_deaths += 1
                logger.warning(
                    "DUPE DEATH: %s at cycle %d — reward penalty %.1f",
                    event.data.get("name"), event.cycle, base_reward,
                )

        # Per-cycle penalty for ignored open obligations
        r += self._event_detector.total_open_penalty()

        self._episode_record.accumulate(r, state)
        self._episode_reward_total += r
        self._prev_state = state

        return round(r, 4)

    def tick_clipped(self, state: dict[str, Any]) -> float:
        """tick() with reward clipped for optimizer input."""
        return clip(self.tick(state))

    def episode_end(self) -> float:
        """Call at episode end. Returns outcome reward."""
        outcome = episode_outcome_reward(self._episode_record)
        self._episode_record.outcome_reward = outcome
        self._episode_reward_total += outcome
        logger.info(
            "Episode ended: cycles=%d deaths=%d total_reward=%.1f outcome=%.1f",
            self._episode_record.final_cycle,
            self._episode_record.total_deaths,
            self._episode_record.total_reward,
            outcome,
        )
        return outcome

    def open_obligations(self) -> list[dict[str, Any]]:
        curr_cycle = self._prev_state.get("cycle", 0) if self._prev_state else 0
        return self._event_detector.open_obligations(curr_cycle)

    @property
    def episode_record(self) -> EpisodeRecord:
        return self._episode_record

    @property
    def episode_total(self) -> float:
        return self._episode_reward_total


# ---------------------------------------------------------------------------
# Colony Health Dashboard (prompt context)
# ---------------------------------------------------------------------------

def format_colony_health(
    state: dict[str, Any],
    tick_reward: float,
    episode_reward: float,
    obligations: list[dict[str, Any]],
) -> str:
    lines = ["--- Colony Health ---"]

    trend = "+" if tick_reward >= 0 else ""
    lines.append(f"Tick reward: {trend}{tick_reward:.3f}  |  Episode total: {episode_reward:.1f}")

    dupes = state.get("duplicants", [])
    alerts = state.get("alerts", [])
    breathable = not any("breathability" in a.lower() for a in alerts)
    fed = not any("food" in a.lower() for a in alerts)

    lines.append(
        f"Dupes: {len(dupes)} alive  |  "
        f"O2: {'OK' if breathable else 'CRITICAL'}  |  "
        f"Food: {'OK' if fed else 'LOW'}"
    )

    if dupes:
        avg_stress = sum(d.get("stress", 0.0) for d in dupes) / len(dupes)
        stress_note = "  <- CRITICAL" if avg_stress > 0.8 else ""
        lines.append(f"Avg stress: {avg_stress * 100:.0f}%{stress_note}")

    if obligations:
        lines.append("Open Obligations:")
        for ob in sorted(obligations, key=lambda x: x["urgency"], reverse=True):
            lines.append(f"  [{ob['urgency']}] {ob['message']}")

    lines.append("--- End Health ---")
    return "\n".join(lines)
