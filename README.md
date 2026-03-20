# KV3D Engine

> Run more concurrent LLM sessions on the same hardware.

KV3D is an inference server for open-weight models that converts shared prompt prefixes into reusable KV-cache snapshots, then stores per-session state as compact int8 deltas — so you get more sessions per GPU without touching the model weights.

```
curl -fsSL https://install.kv3d.dev | bash
kv3d serve --model ./qwen2.5-7b-instruct.Q4_K_M.gguf
```

---

## The problem

Every LLM session carries a full KV cache in GPU memory. When hundreds of sessions share the same system prompt or RAG scaffold, you're paying for the same prefix over and over. Memory fills up. Sessions get queued. Cost per request climbs.

## How KV3D fixes it

```
Session A ──┐
Session B ──┼──► [shared prefix KV snapshot] + [per-session Δ (int8, 4× smaller)]
Session C ──┘
```

1. **Detect** — canonicalize and hash the prompt prefix on every request
2. **Reuse** — serve the shared prefix KV snapshot from the hot/warm cache
3. **Compress** — encode the per-session residual as a quantized int8 delta
4. **Tier** — spill cold state to host RAM or SSD; restore on resume

---

## Features

| ID  | Feature                      | Status  |
|-----|------------------------------|---------|
| F1  | OpenAI-compatible HTTP API   | ✅ MVP  |
| F2  | llama.cpp execution backend  | ✅ MVP  |
| F3  | Exact-prefix family detection | ✅ MVP |
| F4  | Shared prefix KV snapshot    | ✅ MVP  |
| F5  | Compressed session deltas    | ✅ MVP  |
| F7  | GPU hot / RAM warm cache tiers | ✅ MVP |
| F8  | Auto fallback / safe mode    | ✅ MVP  |
| F6  | Collaborative block codec    | 🔜 P1  |
| F9  | Workload analytics dashboard | 🔜 P1  |

---

## Quick start

### Local (curl install)

```bash
curl -fsSL https://install.kv3d.dev | bash

kv3d doctor                                    # verify setup
kv3d serve --model ./qwen2.5-7b.Q4_K_M.gguf   # start server
```

### Docker

```bash
docker run --rm -p 8080:8080 \
  -v $PWD/models:/models \
  ghcr.io/0xbadcaffe/kv3d:latest \
  kv3d serve --model /models/qwen2.5-7b.Q4_K_M.gguf
```

### Build from source

```bash
git clone --recurse-submodules https://github.com/0xbadcaffe/kv3d.git
cd kv3d

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

./build/kv3d serve --model ./models/qwen.gguf
```

---

## API

Drop-in replacement for the OpenAI chat completions endpoint.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": "What is the capital of France?"}
    ]
  }'
```

**Streaming**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-7b-instruct", "stream": true, "messages": [...]}'
```

**Metrics** (Prometheus)

```bash
curl http://localhost:8080/metrics
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Client / API Layer                │
│          /v1/chat/completions · /metrics            │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│               Session & Scheduler Layer             │
│     prefix detection · batching · state machine     │
└───────────┬─────────────────────────┬───────────────┘
            │                         │
┌───────────▼──────────┐  ┌──────────▼───────────────┐
│   Prefix Family Layer │  │    KV Compression Layer  │
│  FNV-1a hash · index  │  │  snapshot · int8 delta   │
└───────────┬──────────┘  └──────────┬───────────────┘
            │                         │
┌───────────▼─────────────────────────▼───────────────┐
│                Storage Tier Manager                 │
│     GPU hot cache · RAM warm cache · SSD cold       │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│              Execution Backend (llama.cpp)          │
│              GGUF · GQA · CPU + CUDA               │
└─────────────────────────────────────────────────────┘
```

---

## Benchmarks

> Numbers will be published as the engine matures. Target claims:

| Claim                        | Measurement                        | Target stage |
|------------------------------|------------------------------------|--------------|
| More sessions per GPU        | Max stable concurrent sessions     | MVP          |
| Lower memory per session     | Avg GPU + RAM per active session   | MVP          |
| Fast resume                  | p95 warm-cache restore latency     | MVP          |
| Negligible quality loss      | Perplexity / logit drift vs baseline | MVP        |
| Lower serving cost           | Cost per 10k requests              | Post-MVP     |

Run the included benchmark driver:

```bash
# 1000 requests, 80% sharing a system prompt
REQUESTS=1000 SHARED_RATIO=0.8 ./scripts/bench.sh
```

---

## Configuration

`~/.config/kv3d/config.json` (created by `kv3d doctor`):

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8080,
    "threads": 4,
    "api_key": ""
  },
  "cache": {
    "gpu_hot_mb": 2048,
    "ram_warm_mb": 8192,
    "ssd_cold_path": ""
  },
  "model": {
    "path": "./models/qwen2.5-7b-instruct.Q4_K_M.gguf",
    "id": "qwen2.5-7b-instruct"
  }
}
```

---

## Development

```bash
# Debug build with sanitizers
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DKV3D_ENABLE_SANITIZERS=ON
cmake --build build -j

# Run tests
ctest --test-dir build --output-on-failure

# Run unit tests only
./build/tests/unit/kv3d_unit_tests

# Run integration tests
./build/tests/integration/kv3d_integration_tests
```

### Repository layout

```
kv3d/
├── include/kv3d/       # public headers
│   ├── core/           # hashing, canonicalization, guardrails
│   ├── kv/             # KV block types, prefix store, delta codec
│   ├── api/            # OpenAI types, server interface
│   ├── sched/          # session manager
│   ├── storage/        # cache tier interfaces
│   └── metrics/        # metrics collector
├── src/                # implementations (mirrors include/)
├── tests/
│   ├── unit/           # fast, in-process tests
│   ├── integration/    # end-to-end session tests
│   └── load/           # benchmark driver
├── scripts/            # install.sh · bench.sh · package.sh
├── cmake/              # build helpers
└── third_party/        # llama.cpp (submodule)
```

---

## Roadmap

| Phase | Focus                    | Gate                              |
|-------|--------------------------|-----------------------------------|
| P0    | Baseline runner          | Stable end-to-end inference       |
| P1    | Exact-prefix reuse       | Hit-rate and correctness validated |
| P2    | Delta codec              | Memory savings proven             |
| P3    | Benchmark & dashboard    | ROI story ready                   |
| P4    | Collaborative block codec | Measured gain beyond simple deltas |

---

## License

Apache 2.0
