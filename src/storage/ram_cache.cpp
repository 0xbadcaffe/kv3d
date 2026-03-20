#include "kv3d/storage/ram_cache.hpp"

#include <algorithm>
#include <stdexcept>

namespace kv3d {

RAMCache::RAMCache(size_t capacity_bytes) : capacity_bytes_(capacity_bytes) {}

// ── Snapshots ─────────────────────────────────────────────────────────────────

bool RAMCache::insert_snapshot(std::shared_ptr<KVSnapshot> snapshot) {
    if (!snapshot) return false;
    const size_t sz = snapshot->byte_size();

    std::lock_guard lock(mutex_);
    while (used_bytes_ + sz > capacity_bytes_ && !snapshots_.empty()) {
        evict_lru_unlocked();
    }
    if (sz > capacity_bytes_) return false;

    const uint64_t id = snapshot->family_id;
    auto existing = snapshots_.find(id);
    if (existing != snapshots_.end()) {
        used_bytes_ -= existing->second.snapshot->byte_size();
    }
    used_bytes_ += sz;
    snapshots_[id] = {std::move(snapshot), std::chrono::steady_clock::now()};
    return true;
}

std::shared_ptr<KVSnapshot> RAMCache::get_snapshot(uint64_t family_id) {
    std::lock_guard lock(mutex_);
    auto it = snapshots_.find(family_id);
    if (it == snapshots_.end()) return nullptr;
    it->second.last_access = std::chrono::steady_clock::now();
    return it->second.snapshot;
}

bool RAMCache::evict_snapshot(uint64_t family_id) {
    std::lock_guard lock(mutex_);
    auto it = snapshots_.find(family_id);
    if (it == snapshots_.end()) return false;
    used_bytes_ -= it->second.snapshot->byte_size();
    snapshots_.erase(it);
    return true;
}

// ── Deltas ────────────────────────────────────────────────────────────────────

bool RAMCache::insert_delta(uint64_t session_id, SessionDelta delta) {
    const size_t sz = delta.byte_size();

    std::lock_guard lock(mutex_);
    while (used_bytes_ + sz > capacity_bytes_ && (!snapshots_.empty() || !deltas_.empty())) {
        evict_lru_unlocked();
    }
    if (sz > capacity_bytes_) return false;

    auto existing = deltas_.find(session_id);
    if (existing != deltas_.end()) {
        used_bytes_ -= existing->second.delta.byte_size();
    }
    used_bytes_ += sz;
    deltas_[session_id] = {std::move(delta), std::chrono::steady_clock::now()};
    return true;
}

std::optional<SessionDelta> RAMCache::get_delta(uint64_t session_id) {
    std::lock_guard lock(mutex_);
    auto it = deltas_.find(session_id);
    if (it == deltas_.end()) return std::nullopt;
    it->second.last_access = std::chrono::steady_clock::now();
    return it->second.delta;
}

bool RAMCache::evict_delta(uint64_t session_id) {
    std::lock_guard lock(mutex_);
    auto it = deltas_.find(session_id);
    if (it == deltas_.end()) return false;
    used_bytes_ -= it->second.delta.byte_size();
    deltas_.erase(it);
    return true;
}

// ── LRU eviction (must hold mutex_) ──────────────────────────────────────────

void RAMCache::evict_lru_unlocked() {
    using TP = std::chrono::steady_clock::time_point;
    TP oldest = std::chrono::steady_clock::now();
    bool found_snap = false, found_delta = false;
    uint64_t snap_id = 0, delta_id = 0;

    for (const auto& [id, e] : snapshots_) {
        if (e.last_access < oldest) {
            oldest = e.last_access;
            snap_id = id;
            found_snap = true;
            found_delta = false;
        }
    }
    for (const auto& [id, e] : deltas_) {
        if (e.last_access < oldest) {
            oldest = e.last_access;
            delta_id = id;
            found_delta = true;
            found_snap = false;
        }
    }

    if (found_snap) {
        used_bytes_ -= snapshots_[snap_id].snapshot->byte_size();
        snapshots_.erase(snap_id);
    } else if (found_delta) {
        used_bytes_ -= deltas_[delta_id].delta.byte_size();
        deltas_.erase(delta_id);
    }
}

size_t RAMCache::capacity_bytes() const noexcept { return capacity_bytes_; }
size_t RAMCache::used_bytes() const noexcept {
    std::lock_guard lock(mutex_);
    return used_bytes_;
}

}  // namespace kv3d
