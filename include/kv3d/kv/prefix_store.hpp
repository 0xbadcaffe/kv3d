#pragma once

#include "kv3d/kv/kv_block.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace kv3d {

/// Thread-safe store mapping family_id -> shared KV prefix snapshot.
/// The store owns all snapshots and manages their lifecycle.
class PrefixStore {
public:
    PrefixStore() = default;
    ~PrefixStore() = default;

    PrefixStore(const PrefixStore&) = delete;
    PrefixStore& operator=(const PrefixStore&) = delete;

    /// Insert or replace a snapshot.  Ownership is transferred to the store.
    void insert(std::shared_ptr<KVSnapshot> snapshot);

    /// Retrieve a snapshot by family ID. Returns nullptr on miss.
    [[nodiscard]] std::shared_ptr<KVSnapshot> get(uint64_t family_id) const;

    /// Return true if a snapshot exists for family_id.
    [[nodiscard]] bool contains(uint64_t family_id) const;

    /// Remove a snapshot from the store.
    bool evict(uint64_t family_id);

    /// Number of stored snapshots.
    [[nodiscard]] size_t size() const;

    /// Total bytes used by all stored snapshots.
    [[nodiscard]] size_t total_bytes() const;

    /// Remove the least recently used snapshot. Returns true if something was evicted.
    bool evict_lru();

private:
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, std::shared_ptr<KVSnapshot>> store_;
    // LRU tracking: family_id -> last access epoch
    std::unordered_map<uint64_t, uint64_t> access_epoch_;
    uint64_t epoch_{0};
};

}  // namespace kv3d
