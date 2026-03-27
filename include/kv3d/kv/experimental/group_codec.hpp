#pragma once

// Phase 2: Collaborative block codec (F6)
//
// Instead of compressing each session's KV state relative to a fixed shared-prefix
// snapshot, GroupCodec maintains a pool of running prototype centroids.  Incoming
// KV blocks are matched to the nearest prototype and stored as a per-tensor int8
// residual.  Because blocks from sessions sharing similar prompts cluster tightly,
// residual magnitudes are far smaller than raw deltas — yielding compression well
// beyond the 4× of Phase 1 scalar quantization.
//
// This is the KV-cache analogue of BM3D: group similar "patches" (blocks), derive
// a shared basis (the running prototype), then transform-code the residual.

#include "kv3d/kv/kv_block.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace kv3d::experimental {

// ── Config ────────────────────────────────────────────────────────────────────

struct BlockGroupConfig {
    /// Max number of prototypes held in the pool simultaneously.
    uint32_t max_prototypes = 256;

    /// Number of key-tensor elements sampled for the fast similarity probe.
    /// Lower = faster matching; higher = more accurate grouping.
    uint32_t probe_dims = 64;

    /// EMA learning rate for prototype updates.
    /// 0 = prototype never moves; 1 = always replaced by the new block.
    float ema_alpha = 0.05f;

    /// Minimum cosine similarity required to reuse an existing prototype.
    /// Below this threshold a new prototype is created.
    float similarity_threshold = 0.85f;

    /// Minimum assignment count before a prototype is eligible for LRU eviction.
    uint32_t min_assignments = 3;
};

// ── Encode / decode data types ────────────────────────────────────────────────

struct GroupEncodeResult {
    uint64_t prototype_id;

    // Per-tensor int8 residuals and their dequantization scales.
    // Using separate scales for keys and values (Phase 1 used a single unified
    // scale which under-compressed the better-quantizable tensor).
    std::vector<int8_t> residual_keys;
    float               scale_keys   = 1.0f;
    std::vector<int8_t> residual_values;
    float               scale_values = 1.0f;

    /// Cosine similarity to the matched prototype [0, 1].
    /// Higher means more compression; useful for observability.
    float similarity = 0.0f;

    /// Effective compression ratio vs storing the raw float32 block.
    /// = raw_bytes / (int8_bytes + 2 * sizeof(float) for scales)
    [[nodiscard]] float compression_ratio() const noexcept;
};

// ── GroupCodec ────────────────────────────────────────────────────────────────

class GroupCodec {
public:
    explicit GroupCodec(BlockGroupConfig cfg = {});
    ~GroupCodec();

    // Non-copyable; the mutex and prototype pool are owned state.
    GroupCodec(const GroupCodec&)            = delete;
    GroupCodec& operator=(const GroupCodec&) = delete;
    GroupCodec(GroupCodec&&)                 = default;
    GroupCodec& operator=(GroupCodec&&)      = default;

    /// Encode a KV block: find (or create) the nearest prototype, then
    /// quantize the residual.  Thread-safe.
    [[nodiscard]] GroupEncodeResult encode(const KVBlock& block);

    /// Reconstruct a KV block from prototype + residual.
    /// `shape` supplies layer_idx / token_offset / token_count metadata.
    /// Thread-safe.
    [[nodiscard]] KVBlock decode(const KVBlock& shape,
                                 const GroupEncodeResult& result) const;

    // ── Diagnostics ──────────────────────────────────────────────────────────

    [[nodiscard]] size_t  prototype_count()       const;
    [[nodiscard]] size_t  total_prototype_bytes()  const;  ///< RAM used by the pool
    [[nodiscard]] double  mean_similarity()        const;  ///< rolling average
    [[nodiscard]] double  mean_compression_ratio() const;  ///< rolling average
    [[nodiscard]] uint64_t total_encodes()         const;

    /// Drop all prototypes and reset statistics (e.g. between benchmark runs).
    void clear();

    /// Evict the least-recently-used prototype that meets min_assignments.
    /// Returns true if something was evicted.
    bool evict_lru();

    [[nodiscard]] BlockGroupConfig config() const;
    void set_config(BlockGroupConfig cfg);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kv3d::experimental
