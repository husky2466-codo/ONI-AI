# src/agent/llm.py
"""
LLM inference layer for the ONI AI agent.

Converts a state snapshot dict into a Gemini prompt, calls the API,
and decodes the response into a valid ActionCommand dict.
Falls back to no_op on any parse or API failure.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from google import genai
from google.genai import types

from src.agent.protocol import VALID_ACTIONS, build_action, build_no_op

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are an autonomous colony manager for the game Oxygen Not Included (ONI).
Your job is to keep duplicants alive and the colony growing.

## Map & coordinates
- 2D grid map. Duplicants start near center (~x=120, y=200 on a standard Sandstone map).
- Y increases upward. Tiles must be DUG before buildings can be placed in them.
- The buildings list in state shows only COMPLETED buildings — queued/under-construction are invisible to you.
- power_kw = currently active generation. 0 is normal on cycle 1.
- oxygen_kg = free gaseous O2. Near-zero on cycle 1-2 is normal — algae hasn't been processed yet.
- food_kcal values above 1,000,000 are save artifacts — ignore them, treat food as unknown.

## Building reference (use these exact building_id values)

### Survival priority (place in this order, cycles 1-5)
| building_id       | Name              | Power   | Key inputs          | Key outputs            | Size  | Notes |
|-------------------|-------------------|---------|---------------------|------------------------|-------|-------|
| Outhouse          | Outhouse          | none    | 13 kg Dirt/use      | Polluted Dirt          | 2×3   | Required for toilets. Without it duplicants stress immediately. Empty when full (15 uses). |
| Bed               | Cot               | none    | —                   | +1 morale (Barracks)   | 2×2   | One per duplicant. Without a bed, duplicants get sore backs. |
| MicrobeMusher     | Microbe Musher    | -240 W  | Dirt + Water        | 800 kcal/kg Mush Bar   | 2×3   | Assign Mush Bar recipe. Needs power. Place near start. |
| OxygenDiffuser    | Oxygen Diffuser   | -120 W  | 550 g/s Algae       | 500 g/s O2 @ 30°C      | 2×2   | Primary early O2. Stops at 1.8 kg overpressure. Needs power. |
| ManualGenerator   | Manual Generator  | +400 W  | Duplicant labor     | 400 W                  | 2×2   | Needs Wire to distribute power. Pair with Battery. |
| Battery           | Battery           | storage | —                   | 10 kJ stored           | 1×2   | Buffers power. Place adjacent to generators on same wire. |
| Wire              | Wire              | —       | —                   | connects power         | 1×1   | Must connect generator → battery → buildings. |
| ResearchCenter    | Research Station  | -60 W   | 50 kg Dirt/point    | research points        | 2×2   | Mandatory to unlock tech. Place early. |

### Mid-game oxygen (after research)
| building_id  | Name         | Power   | Key inputs       | Key outputs                    | Notes |
|--------------|--------------|---------|------------------|--------------------------------|-------|
| Electrolyzer | Electrolyzer | -120 W  | 1,000 g/s Water  | 888 g/s O2 + 112 g/s H2 @ 70°C | Needs water pipes, gas pumps to vent H2. Superior to OxygenDiffuser long-term. |

## Survival rules
1. ALWAYS have at least 1 Outhouse per 3 duplicants.
2. ALWAYS have 1 Cot (Bed) per duplicant.
3. Power chain: ManualGenerator → Wire → Battery → Wire → powered buildings. No wire = no power delivery.
4. OxygenDiffuser needs power — place ManualGenerator + Wire + Battery BEFORE or simultaneously.
5. MicrobeMusher needs power and a recipe assigned — Mush Bar (Dirt + Water) works with starting resources.
6. Dig corridors before placing buildings — duplicants cannot path through solid tiles.
7. Do NOT place a building type that already appears in the buildings list nearby.
8. Issue one dig or place action per tick — do not spam. Use no_op when waiting for duplicants to finish.

## Early game sequence (cycles 1-5)
1. Dig out 3 rooms: toilet room, bedroom, food/power room (~4 tiles wide, 3 tiles tall each)
2. Place Outhouse in toilet room
3. Place Bed for each duplicant in bedroom
4. Place ManualGenerator + Battery + Wire in power room
5. Place OxygenDiffuser (needs power — wire it)
6. Place MicrobeMusher (needs power — wire it), assign Mush Bar recipe
7. Place ResearchCenter, start researching

## ONI survival knowledge

### Oxygen math
- Each duplicant consumes 100g O2/second = 8.64 kg/cycle.
- 3 dupes need ~26 kg/cycle. One OxygenDiffuser produces 500g/s = 43.2 kg/cycle — enough for 3-5 dupes.
- Target air pressure: 1500-2000g/tile minimum. OxygenDiffuser stops at 1800g overpressure.
- Algae Terrarium (id: AlgaeHabitat): 40g/s O2, needs 300g/s water + 30g/s algae, zero power. Use 8 for 3 dupes.

### Food math
- Each duplicant needs 1000 kcal/day.
- Mush Bar from MicrobeMusher: 800 kcal/kg, needs Dirt + Water. Assign recipe explicitly.
- Mealwood plant (id: Mealwood): 600 kcal per harvest every 3 cycles = 200 kcal/day. Need 15 plants for 3 dupes.
- Keep 5+ cycle food reserve (15,000+ kcal for 3 dupes).
- Mealwood stops growing above 30°C — keep farm areas cool.

### Power chain (CRITICAL)
ManualGenerator → Wire → Battery → Wire → (OxygenDiffuser, MicrobeMusher, ResearchCenter, etc.)
- Without Wire connecting them, power does NOT flow. Wire every building to the network.
- ManualGenerator outputs 400W. OxygenDiffuser needs 120W, MicrobeMusher needs 240W, ResearchCenter needs 60W.
- Total early draw: 420W. One ManualGenerator is enough if a dupe keeps running it.
- Battery (10kJ) buffers power so buildings run while dupe isn't on the generator.

### Room bonuses (morale)
- Barracks (1+ Cot, 12-64 tiles, no industrial): +1 morale per dupe
- Latrine (Outhouse + WashBasin, 12-64 tiles): +1 morale
- Recreation Room (12-64 tiles, decor + rec building): +4 morale during breaks
- Mess Hall (12-64 tiles, MessTable, no industrial): +3 morale
- Morale needed: base=1, Tier1 job=2, Tier2 job=4. Keep morale above job tier requirement.

### Gas physics
- Gases settle by weight: CO2 sinks to bottom, H2 rises to top, O2 stays in middle.
- Two gas types cannot share a tile — they trade places.
- Seal rooms to prevent O2 from leaking into unneeded areas.
- CO2 accumulation at floor level is normal — don't panic unless it reaches dupe height.

### Stress management
- Stress rises when: no toilet, no sleep, low morale, high temperature.
- Keep stress below 10% early game.
- Outhouse + Cot + food = stress stays near 0 in early cycles.
- Stress break actions (vomiting, crying, destroying buildings) start around 20-30%.

### Research priority (early game)
1. Sanitation → unlocks Lavatory, WashSink
2. Basic Farming → unlocks Farm Tile, Planter Box
3. Meal Preparation → unlocks Electric Grill, better food
4. Jobs → unlocks job boards, skill progression
5. Interior Decor → unlocks sculptures, morale buildings

### Morale and stress (CRITICAL on higher difficulty)
- Every skill a dupe takes adds morale requirement (1-4 pts per skill tier).
- "Sufficient Morale" buff (morale met): -5% stress/cycle. High morale: -10 to -20%/cycle.
- Morale deficit causes stress gain. At 100% stress, destructive breakdowns occur.
- Key morale sources: room bonuses (see above), food quality, decor, recreation breaks.
- Traits to avoid when printing dupes: Gourmet (-1 food quality), Flatulent (random gas), Anemic (-5 athletics), Narcoleptic (random sleep).

### Dupe expansion milestones
- Start: 3 dupes
- Add 4th: cycle 5, once O2 + food production consistently exceeds demand
- Add 5th-8th: once sealed O2 system and sustainable food established
- Never expand faster than your life support can handle

### Biome safety
- Do NOT dig slime biome tiles — releases Slimelung germs into base air.
- Dig around slime using adjacent non-slime tiles.
- Build Deodorizers (id: AirFilter) before entering slime areas.

### Build order summary (cycles 1-5)
1. Dig 3 rooms: toilet room, bedroom, food+power room (each ~4 wide, 3-4 tall)
2. Outhouse (priority 8) + WashBasin in toilet room
3. Bed for each dupe in bedroom
4. ManualGenerator + Battery + Wire in power room
5. OxygenDiffuser wired to power network
6. MicrobeMusher wired to power, assign Mush Bar recipe
7. ResearchCenter, start Sanitation research

## Response format
Output ONLY a single JSON object — no explanation, no markdown, no code fences:
  {"type": "action", "action": "no_op"}
  {"type": "action", "action": "dig", "cell_x": <int>, "cell_y": <int>}
  {"type": "action", "action": "cancel_dig", "cell_x": <int>, "cell_y": <int>}
  {"type": "action", "action": "place_building", "building_id": "<id>", "cell_x": <int>, "cell_y": <int>}
  {"type": "action", "action": "set_priority", "cell_x": <int>, "cell_y": <int>, "priority": <1-9>}
"""


def _format_state(data: dict[str, Any]) -> str:
    """Format a state snapshot dict into a concise prompt string."""
    res = data.get("resources", {})
    dups = data.get("duplicants", [])
    alerts = data.get("alerts", [])
    buildings = data.get("buildings", [])

    lines = [
        f"Cycle: {data.get('cycle', '?')}",
        "",
        "Resources:",
        f"  oxygen_kg:  {res.get('oxygen_kg', 0):.2f}",
        f"  water_kg:   {res.get('water_kg', 0):.2f}",
        f"  food_kcal:  {res.get('food_kcal', res.get('food_kcal_today', 0)):.0f}",
        f"  power_kw:   {res.get('power_kw', 0):.2f}",
        f"  co2_kg:     {res.get('co2_kg', 0):.3f}",
        "",
        f"Duplicants ({len(dups)}):",
    ]
    for d in dups:
        lines.append(
            f"  {d.get('name','?')} @ ({d.get('x','?')},{d.get('y','?')}) "
            f"stress={d.get('stress',0)*100:.0f}% hp={d.get('health',0):.0f} "
            f"task={d.get('current_task','?')}"
        )

    if alerts:
        lines.append("")
        lines.append("Alerts:")
        for a in alerts:
            lines.append(f"  ! {a}")

    lines.append("")
    lines.append(f"Buildings on map: {len(buildings)}")
    for b in buildings[:20]:
        op = "OK" if b.get("operational") else "OFFLINE"
        lines.append(f"  {b.get('type','?')} @ ({b.get('x','?')},{b.get('y','?')}) [{op}]")
    if len(buildings) > 20:
        lines.append(f"  ...and {len(buildings) - 20} more")

    # Summarize what survival buildings are still missing
    built_types = {b.get("type") for b in buildings}
    dup_count = len(dups) or 3
    survival = []
    if "Outhouse" not in built_types:
        survival.append("Outhouse (no toilet!)")
    if "Bed" not in built_types:
        survival.append(f"Bed x{dup_count} (no sleep!)")
    if "OxygenDiffuser" not in built_types and "Electrolyzer" not in built_types:
        survival.append("OxygenDiffuser (no oxygen production!)")
    if "MicrobeMusher" not in built_types:
        survival.append("MicrobeMusher (no food production!)")
    if "ManualGenerator" not in built_types:
        survival.append("ManualGenerator (no power!)")
    if "Battery" not in built_types:
        survival.append("Battery (no power storage!)")
    if "ResearchCenter" not in built_types:
        survival.append("ResearchCenter (no research!)")

    if survival:
        lines.append("")
        lines.append("MISSING survival buildings (build these NOW):")
        for s in survival:
            lines.append(f"  ! {s}")

    tile_summary = _summarize_tiles(data.get("tiles", {}))
    if tile_summary:
        lines.append("")
        lines.append(tile_summary)

    return "\n".join(lines)


def _summarize_tiles(tiles: dict) -> "str | None":
    """Return a compact text summary of a tile window for inclusion in the prompt."""
    data = tiles.get("data")
    if not data:
        return None
    x, y, w, h = tiles.get("x", 0), tiles.get("y", 0), tiles.get("w", 0), tiles.get("h", 0)
    total = len(data)
    counts: dict[str, int] = {}
    for entry in data:
        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            name = str(entry[0])
            counts[name] = counts.get(name, 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:6]
    top_str = ", ".join(f"{n}({c})" for n, c in top)
    return f"Tile window: x={x} y={y} w={w} h={h} ({total} tiles) | top elements: {top_str}"


# Gemini 2.5 Flash pricing (per million tokens, as of 2026-04)
_COST_INPUT_PER_M  = 0.15   # $0.15 / 1M input tokens
_COST_OUTPUT_PER_M = 0.60   # $0.60 / 1M output tokens


class GeminiAgent:
    """Calls Gemini Flash to decide the next ONI action given a state snapshot."""

    def __init__(self, api_key: str, model: str = MODEL):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self.total_calls         = 0
        self.total_cost_usd      = 0.0

    @property
    def stats(self) -> dict:
        return {
            "calls":         self.total_calls,
            "input_tokens":  self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cost_usd":      round(self.total_cost_usd, 6),
        }

    def decide(self, state_data: dict[str, Any]) -> dict[str, Any]:
        """
        Given a state snapshot dict, return an ActionCommand dict.
        Falls back to no_op on any failure.
        """
        prompt = _format_state(state_data)
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=1024,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            # Track token usage and cost
            usage = response.usage_metadata
            if usage:
                inp  = usage.prompt_token_count or 0
                out  = usage.candidates_token_count or 0
                self.total_input_tokens  += inp
                self.total_output_tokens += out
                self.total_calls         += 1
                self.total_cost_usd += (inp / 1_000_000) * _COST_INPUT_PER_M
                self.total_cost_usd += (out / 1_000_000) * _COST_OUTPUT_PER_M

            raw = response.text.strip()
            logger.debug("Gemini raw response: %s", raw)
            return self._parse_action(raw)
        except Exception as e:
            logger.warning("Gemini call failed: %s — sending no_op", e)
            return build_no_op()

    def _parse_action(self, raw: str) -> dict[str, Any]:
        """Parse the model's JSON response into a valid action dict."""
        # Strip markdown code fences if the model ignored instructions
        if raw.startswith("```"):
            raw = "\n".join(
                line for line in raw.splitlines()
                if not line.startswith("```")
            ).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("JSON parse failed: %s — raw: %r", e, raw)
            return build_no_op()

        action = obj.get("action")
        if action not in VALID_ACTIONS:
            logger.warning("Unknown action %r — sending no_op", action)
            return build_no_op()

        try:
            if action == "no_op":
                return build_no_op()
            elif action in ("dig", "cancel_dig"):
                return build_action(action,
                    cell_x=int(obj["cell_x"]),
                    cell_y=int(obj["cell_y"]))
            elif action == "place_building":
                return build_action(action,
                    building_id=str(obj["building_id"]),
                    cell_x=int(obj["cell_x"]),
                    cell_y=int(obj["cell_y"]))
            elif action == "set_priority":
                return build_action(action,
                    cell_x=int(obj["cell_x"]),
                    cell_y=int(obj["cell_y"]),
                    priority=max(1, min(9, int(obj.get("priority", 5)))))
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Action param error: %s — raw: %r", e, raw)
            return build_no_op()

        return build_no_op()
