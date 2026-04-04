# Model Catalog

<!--
last_updated: 2026-04-04
maintainer: auto-maintained — update manually when models are added/removed/reloaded
format: human-readable + machine-parseable (consistent tables, no merged cells)
scope: all inference nodes in the ONI-AI infrastructure
-->

```yaml
last_updated: "2026-04-04T22:00"
maintainer: "manual — update when serving state changes"
nodes:
  - id: dgx-a
    host: 10.0.0.69
    role: inference
    inference_port: 8000
  - id: dgx-b
    host: 192.168.3.20
    role: training/inference
    inference_port: 8000
    orchestration_port: 8080
  - id: linux-desktop
    host: 10.0.0.10
    role: game-host/support-services
    inference: false
```

---

## Node: DGX Spark A — 10.0.0.69 (inference node)

Inference stack: vLLM on :8000

### Serving (active)

| hf_model_id | type | quantization | disk_size | status | endpoint |
|---|---|---|---|---|---|
| Qwen/Qwen2.5-72B-Instruct-AWQ | LLM instruct | AWQ | 39G | serving | http://10.0.0.69:8000/v1 |

### Cached (not serving)

| hf_model_id | type | quantization | disk_size | status | endpoint |
|---|---|---|---|---|---|
| Qwen/Qwen2.5-72B-Instruct | LLM instruct | BF16 | 136G | cached | none |
| Qwen/QwQ-32B-AWQ | LLM reasoning | AWQ | 19G | cached | none |
| Qwen/Qwen2.5-Coder-14B-Instruct | LLM code | BF16 | 28G | cached | none |
| Qwen/Qwen2.5-7B-Instruct | LLM instruct | BF16 | ~7G | cached | none |
| meta-llama/Llama-3.1-8B-Instruct | LLM instruct | BF16 | ~8G | cached | none |
| nvidia/Nemotron-3-Super-120B-A12B-NVFP4 | LLM instruct | NVFP4 | 15G | cached | none |
| hexgrad/Kokoro-82M | TTS | none | 314M | cached | none |
| hubertsiuzdak/snac_24khz | Audio codec | none | 76M | cached | none |

---

## Node: DGX Spark B — 192.168.3.20 (training / NemoClaw node, Ross's machine)

Inference stack: vLLM on :8000, NemoClaw orchestration cluster on :8080

### Serving (active)

| hf_model_id | type | quantization | disk_size | status | endpoint |
|---|---|---|---|---|---|
| Qwen/Qwen2.5-72B-Instruct-AWQ | LLM instruct | AWQ | 39G | serving | http://192.168.3.20:8000/v1 |

### Cached (not serving)

| hf_model_id | type | quantization | disk_size | status | endpoint |
|---|---|---|---|---|---|
| Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 | LLM instruct | GPTQ Int8 | 33G | cached | none |
| Qwen/Qwen2.5-Coder-14B-Instruct | LLM code | BF16 | 28G | cached | none |
| nvidia/Nemotron-3-Super-120B-A12B-NVFP4 | LLM instruct | NVFP4 | 6.9G | cached | none |
| nvidia/Llama-3.3-Nemotron-Super-49B-v1.5-FP8 | LLM instruct | FP8 | ~49G | cached | none |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 | LLM instruct | FP8 | ~30G | cached | none |
| Qwen/Qwen2-VL-7B-Instruct | VLM | BF16 | ~7G | cached | none |

---

## Node: Linux Desktop — 10.0.0.10 (game host / support services)

GPU: NVIDIA GeForce RTX 4070

### Running Containers (support services)

| service | image | type | status | port |
|---|---|---|---|---|
| Kokoro FastAPI | ghcr.io/remsky/kokoro-fastapi-gpu | TTS (GPU) | running | — |
| Piper TTS | kamilkrawiec/piper-openai-tts | TTS (CPU) | running | — |
| ChromaDB | chromadb/chroma | Vector DB | running | 8300 (→internal 8000) |

### Models in HF Cache

| hf_model_id | type | format | disk_size | status |
|---|---|---|---|---|
| ggml-org/SmolVLM2-2.2B-Instruct-GGUF | VLM | GGUF | ~2.2B | cached — **vision pipeline default** |
| unsloth/Qwen2.5-VL-7B-Instruct-GGUF | VLM | GGUF | ~7B | cached — vision pipeline fallback |
| mobiuslabsgmbh/faster-whisper-large-v3-turbo | STT | none | 1.6G | cached |
| openai/whisper-medium | STT | none | 2.9G | cached |

### Vision Pipeline (planned)

GGUF models require **llama.cpp server** (not vLLM). Suggested deployment:

```bash
# SmolVLM2-2.2B as default vision model on :8080
llama-server --model ~/.cache/huggingface/hub/models--ggml-org--SmolVLM2-2.2B-Instruct-GGUF/snapshots/.../SmolVLM2-2.2B-Instruct-Q8_0.gguf \
  --port 8080 --n-gpu-layers 99 --host 0.0.0.0
```

Vision endpoint: `http://10.0.0.10:8080/v1/chat/completions` (llama.cpp OpenAI-compatible)

---

## Service Port Reference

| service | host | port | notes |
|---|---|---|---|
| vLLM inference | 10.0.0.69 | 8000 | DGX A — Qwen2.5-72B-AWQ |
| vLLM inference | 192.168.3.20 | 8000 | DGX B — Qwen2.5-72B-AWQ |
| NemoClaw orchestration | 192.168.3.20 | 8080 | DGX B training loop |
| ChromaDB | 10.0.0.10 | 8300 | Linux desktop — agent memory |
| llama.cpp vision | 10.0.0.10 | 8080 | Linux desktop — SmolVLM2 (planned) |
| ONIBridge TCP | 10.0.0.10 | 9999 | Game bridge |
| Agent WebSocket relay | 10.0.0.210 | 8182 | runner.py → dashboard |
| Dashboard | 10.0.0.210 | 8181 | FastAPI |

---

## Gaps and Planned Work

| gap | node | priority | notes |
|---|---|---|---|
| llama.cpp server for SmolVLM2-2.2B | linux-desktop | P1 | Blocks vision pipeline |
| Kokoro + SNAC serving on DGX-A | dgx-a | not started | Models cached, no FastAPI wrapper deployed |
| Qwen2-VL-7B serving on DGX-B | dgx-b | not started | Cached, config not set up |
| BF16 Qwen2.5-72B on DGX-A | dgx-a | on-demand | Too large for concurrent serving; load manually |
| vLLM reload endpoint on DGX-A | dgx-a | P2 | Needed for checkpoint promotion from DGX B |
| NemoClaw episode intake pipeline | dgx-b | P2 | Training loop orchestration |
