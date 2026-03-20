#pragma once

#include "kv3d/kv/kv_block.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

namespace kv3d {

/// Hot-tier KV cache backed by GPU VRAM (or a CPU buffer when CUDA is absent).
/// Capacity is measured in bytes; eviction is LRU.
class GPUCache {
public:
    explicit GPUCache(size_t capacity_bytes);
    ~GPUCache();

    GPUCache(const GPUCache&) = delete;
    GPUCache& operator=(const GPUCache&) = delete;

    /// Insert a snapshot. May evict existing entries to make room.
    /// Returns false if the snapshot alone exceeds capacity.
    bool insert(std::shared_ptr<KVSnapshot> snapshot);

    /// Retrieve a snapshot from the hot tier. Updates LRU order.
    [[nodiscard]] std::shared_ptr<KVSnapshot> get(uint64_t family_id);

    /// Remove a snapshot.
    bool evict(uint64_t family_id);

    [[nodiscard]] size_t capacity_bytes() const noexcept;
    [[nodiscard]] size_t used_bytes() const noexcept;
    [[nodiscard]] size_t size() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace kv3d
