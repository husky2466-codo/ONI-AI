#!/bin/bash
# Copies required ONI DLLs from the game install into mod/ONIBridge/lib/
# Run this once on the Ubuntu machine (where ONI is installed) or
# after a game update that changes the managed DLLs.
#
# Usage: ./copy-game-dlls.sh [game_managed_path]

GAME_MANAGED="${1:-$HOME/.steam/steam/steamapps/common/OxygenNotIncluded/OxygenNotIncluded_Data/Managed}"
LIB_DIR="$(dirname "$0")/../lib"

if [ ! -d "$GAME_MANAGED" ]; then
  echo "ERROR: Game Managed folder not found at: $GAME_MANAGED"
  echo "Usage: $0 /path/to/OxygenNotIncluded_Data/Managed"
  exit 1
fi

mkdir -p "$LIB_DIR"

DLLS=(
  "Assembly-CSharp.dll"
  "Assembly-CSharp-firstpass.dll"
  "0Harmony.dll"
  "UnityEngine.dll"
  "UnityEngine.CoreModule.dll"
  "Newtonsoft.Json.dll"
)

echo "Copying DLLs from: $GAME_MANAGED"
for dll in "${DLLS[@]}"; do
  src="$GAME_MANAGED/$dll"
  if [ -f "$src" ]; then
    cp "$src" "$LIB_DIR/"
    echo "  OK: $dll"
  else
    echo "  MISSING: $dll"
  fi
done

echo "Done. DLLs are in: $LIB_DIR"
