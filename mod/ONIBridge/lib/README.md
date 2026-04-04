# ONIBridge — Game DLL References

This directory holds game DLLs required to build the ONIBridge mod. They are
**not included in the repository** (non-redistributable Klei / Unity binaries).

## Setup

Run the helper script from a machine that has Oxygen Not Included installed:

```bash
cd mod/ONIBridge
./scripts/copy-game-dlls.sh
```

This copies the following DLLs from the ONI Managed folder:

| File | Source |
|------|--------|
| `Assembly-CSharp.dll` | ONI game logic |
| `Assembly-CSharp-firstpass.dll` | ONI pre-compiled scripts |
| `0Harmony.dll` | Harmony patching library (ships with ONI) |
| `UnityEngine.dll` | Unity core |
| `UnityEngine.CoreModule.dll` | Unity core module |
| `Newtonsoft.Json.dll` | JSON serialization |

Default Steam path: `~/.steam/steam/steamapps/common/OxygenNotIncluded/OxygenNotIncluded_Data/Managed/`

If your install is elsewhere, pass the path as the first argument:

```bash
./scripts/copy-game-dlls.sh /path/to/OxygenNotIncluded_Data/Managed
```
