#pragma once

#include "kv3d/kv/kv_block.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace kv3d {

/// Warm-tier KV cache in host RAM.
/// Stores compressed SessionDeltas and KVSnapshots that were evicted from GPU.
class RAMCache {
public:
    explicit RAMCache(size_t capacity_bytes);
    ~RAMCache() = default;

    RAMCache(const RAMCache&) = delete;
    RAMCache& operator=(const RAMCache&) = delete;

    // ── KV Snapshots (shared prefix state) ───────────────────────────────────
    bool insert_snapshot(std::shared_ptr<KVSnapshot> snapshot);
    [[nodiscard]] std::shared_ptr<KVSnapshot> get_snapshot(uint64_t family_id);
    bool evict_snapshot(uint64_t family_id);

    // ── Session deltas (per-session compressed residuals) ────────────────────
    bool insert_delta(uint64_t session_id, SessionDelta delta);
    [[nodiscard]] std::optional<SessionDelta> get_delta(uint64_t session_id);
    bool evict_delta(uint64_t session_id);

    [[nodiscard]] size_t capacity_bytes() const noexcept;
    [[nodiscard]] size_t used_bytes() const noexcept;

private:
    mutable std::mutex mutex_;
    size_t capacity_bytes_;
    size_t used_bytes_{0};

    struct SnapshotEntry {
        std::shared_ptr<KVSnapshot> snapshot;
        std::chrono::steady_clock::time_point last_access;
    };
    std::unordered_map<uint64_t, SnapshotEntry> snapshots_;

    struct DeltaEntry {
        SessionDelta delta;
        std::chrono::steady_clock::time_point last_access;
    };
    std::unordered_map<uint64_t, DeltaEntry> deltas_;

    void evict_lru_unlocked();
};

}  // namespace kv3d
