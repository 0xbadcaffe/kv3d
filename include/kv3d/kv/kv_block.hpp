#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace kv3d {

/// Model geometry needed to interpret KV layout.
struct ModelConfig {
    std::string model_id;
    uint32_t n_layers{0};
    uint32_t n_heads{0};
    uint32_t n_kv_heads{0};  // GQA: may be < n_heads
    uint32_t head_dim{0};
    uint32_t context_length{0};
};

/// One layer's KV tensors covering a contiguous token range.
/// Layout: keys/values are [n_kv_heads, token_count, head_dim] row-major.
struct KVBlock {
    uint32_t layer_idx{0};
    uint32_t token_offset{0};  // first token index in context
    uint32_t token_count{0};   // number of tokens covered

    std::vector<float> keys;    // n_kv_heads * token_count * head_dim
    std::vector<float> values;  // n_kv_heads * token_count * head_dim

    [[nodiscard]] size_t byte_size() const noexcept {
        return (keys.size() + values.size()) * sizeof(float);
    }

    [[nodiscard]] bool empty() const noexcept { return keys.empty(); }
};

/// Full KV cache snapshot for a shared prefix — stored once, reused by N sessions.
/// Not copyable; use shared_ptr<KVSnapshot>.
struct KVSnapshot {
    uint64_t family_id{0};
    uint32_t token_count{0};
    std::string model_id;
    std::vector<KVBlock> blocks;  // one entry per layer
    std::chrono::steady_clock::time_point created_at{std::chrono::steady_clock::now()};
    mutable std::atomic<int32_t> ref_count{0};

    KVSnapshot() = default;
    KVSnapshot(const KVSnapshot&) = delete;
    KVSnapshot& operator=(const KVSnapshot&) = delete;
    KVSnapshot(KVSnapshot&&) noexcept = default;
    KVSnapshot& operator=(KVSnapshot&&) noexcept = default;

    [[nodiscard]] size_t byte_size() const noexcept {
        size_t total = 0;
        for (const auto& b : blocks) total += b.byte_size();
        return total;
    }
};

/// Per-session compressed delta — the KV residual beyond the shared prefix.
struct SessionDelta {
    uint64_t session_id{0};
    uint64_t family_id{0};          // prefix this delta is relative to
    uint32_t delta_token_count{0};  // tokens beyond the shared prefix
    float scale_factor{1.0f};       // quantization scale (for int8 codec)
    std::vector<int8_t> compressed_keys;
    std::vector<int8_t> compressed_values;

    [[nodiscard]] size_t byte_size() const noexcept {
        return (compressed_keys.size() + compressed_values.size()) * sizeof(int8_t);
    }

    [[nodiscard]] bool empty() const noexcept { return compressed_keys.empty(); }
};

}  // namespace kv3d
